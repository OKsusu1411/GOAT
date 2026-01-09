# goat_control/core/comm/can.py
from __future__ import annotations

import time
import threading
import logging
from typing import Optional

import can


class CanInterface:
    """Dedicated CAN transport class.

    This class owns the python-can Bus and provides:
      - open()/close()
      - send()/receive()
      - txrx(): request/response helper with echo filtering

    Important:
      - txrx() is serialized with a lock so that concurrent callers do not
        consume each other's response frames via recv().
      - If you later need higher throughput, replace the lock with a background
        RX thread and per-(arbitration_id, cmd_byte) dispatch queues.
    """

    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        bitrate: int | None = None,  
        receive_own_messages: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.channel = channel
        self.interface = interface
        self.bitrate = bitrate
        self.receive_own_messages = receive_own_messages

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.bus: can.BusABC | None = None

        # Serialize txrx() to prevent response-frame “stealing” between callers.
        self.txrx_lock = threading.Lock()

    def open(self) -> None:
        """Open the CAN bus if it is not already opened."""
        if self.bus is not None:
            return

        self.bus = can.Bus(
            interface=self.interface,
            channel=self.channel,
            bitrate=self.bitrate,
            receive_own_messages=self.receive_own_messages,
        )
        self.logger.info(f"[CAN] opened: {self.interface}:{self.channel}")

    def close(self) -> None:
        """Close the CAN bus if it is opened."""
        if self.bus is None:
            return

        self.bus.shutdown()
        self.bus = None
        self.logger.info("[CAN] closed")

    def send(self, arbitration_id: int, data: bytes) -> can.Message | None:
        """Send a raw CAN frame."""
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")

        try:
            sent_message = can.Message(
                arbitration_id=arbitration_id,
                data=data,
                is_extended_id=False,
            )
            self.bus.send(sent_message)
            return sent_message
        except can.CanError as can_error:
            self.logger.error(f"[CAN] send failed (id=0x{arbitration_id:X}): {can_error}")
            return None

    def receive(self, timeout: float = 0.05) -> can.Message | None:
        """Receive a raw CAN frame."""
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")
        return self.bus.recv(timeout=timeout)

    def _drain_rx(self, max_frames: int = 200) -> int:
        """Drain pending RX frames.

        Motivation:
          - If old response frames remain in RX queue,
            a new txrx() call could accidentally pick up a stale frame.
        """
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")

        drained = 0
        for _ in range(int(max_frames)):
            msg = self.bus.recv(timeout=0.0)
            if msg is None:
                break
            drained += 1
        return drained

    def txrx(
        self,
        tx_id: int,
        rx_id: int,
        cmd_byte: int,
        payload7: bytes,
        timeout: float = 0.5,
        accept_rx_id: bool = True,
        accept_tx_echo_diff: bool = True,
    ) -> can.Message | None:
        """Transmit one command frame and wait for a matching response.

        Filtering rules:
          - Accept a response from rx_id (0x180 + node_id) when accept_rx_id=True.
          - Some setups return a response using tx_id (0x140 + node_id) but with different data.
            If accept_tx_echo_diff=True, accept frames on tx_id whose data differs from the sent frame.
          - Pure loopback echo (same tx_id and identical data) is always ignored.
        """
        if len(payload7) != 7:
            raise ValueError("payload7 must be 7 bytes")

        transmitted_data = bytes([cmd_byte]) + payload7

        # with self.txrx_lock:
        #     # Drain any pending frames to avoid matching a stale response.
        #     drained = self._drain_rx(max_frames=200)
        #     if drained > 0:
        #         self.logger.debug(f"[CAN] drained {drained} stale RX frames before txrx")

        #     sent_message = self.send(tx_id, transmitted_data)

        with self.txrx_lock:
            sent_message = self.send(tx_id, transmitted_data)
            if sent_message is None:
                return None

            deadline_time = time.time() + timeout
            while time.time() < deadline_time:
                remaining_time = max(0.0, deadline_time - time.time())
                receive_timeout = min(0.05, remaining_time)

                received_message = self.receive(timeout=receive_timeout)
                if received_message is None:
                    continue

                if len(received_message.data) != 8:
                    continue
                if received_message.data[0] != cmd_byte:
                    continue

                # 1) Normal protocol response (0x180 + node_id)
                if accept_rx_id and (received_message.arbitration_id == rx_id):
                    return received_message

                # 2) Response on tx_id with different data (ignore pure echo)
                if (
                    accept_tx_echo_diff
                    and (received_message.arbitration_id == tx_id)
                    and (received_message.data != sent_message.data)
                ):
                    return received_message

            return None
