# goat_control/core/comm/can.py
from __future__ import annotations

import time
import threading
import logging
import can

class CanInterface:
    """Owns one physical CAN bus.

    Startup mode:
      - txrx() sends a request and waits synchronously.
      - The background reader must not be running.

    Runtime mode:
      - send_only() sends commands.
      - The background reader receives and caches replies.
    """

    def __init__(self,
                 channel: str = "can0",
                 interface: str = "socketcan",
                 bitrate: int | None = None,
                 receive_own_messages: bool = False,
                 logger: logging.Logger | None = None):
        self.channel = channel
        self.interface = interface
        self.bitrate = bitrate
        self.receive_own_messages = receive_own_messages

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.bus: can.BusABC | None = None

        # Serialize txrx() to prevent response-frame "stealing" between callers.
        self.txrx_lock = threading.Lock()

        # Background reader state.
        self.latest_rx_frames_by_key: dict[tuple[int, int], can.Message] = {}
        self._rx_lock = threading.Lock()
        self._rx_stop_event = threading.Event()
        self._rx_thread: threading.Thread | None = None

        self.rx_frame_count: int = 0
        self._rx_first_keys_seen: set[tuple[int, int]] = set()

        # Per-key arrival events for synchronous hot-path dispatch. 
        self.frame_events: dict[tuple[int, int], threading.Event] = {}
        self._events_lock = threading.Lock()

    # ------------------------------------------------------------------
    # CAN Bus Open & Close 
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Open the CAN bus if it is not already opened."""
        if self.bus is not None:
            return

        self.bus = can.Bus(interface=self.interface,
                           channel=self.channel,
                           bitrate=self.bitrate,
                           receive_own_messages=self.receive_own_messages)
        self.logger.info(f"[CAN] opened: {self.interface}:{self.channel}")

    def close(self) -> None:
        """Close the CAN bus if it is opened."""
        # Stop reader first so it doesn't try to recv() on a closed bus.
        self.stop_reader_thread()

        if self.bus is None:
            return

        self.bus.shutdown()
        self.bus = None
        self.logger.info("[CAN] closed")

    # -------------
    # State
    # -------------
    def is_reader_running(self) -> bool:
        """True if the background reader is draining the bus."""
        return self._rx_thread is not None and self._rx_thread.is_alive()

    def get_latest_frame(self, arbitration_id: int, cmd_byte: int) -> can.Message | None:
        """Return the most recent cached frame for (arbitration_id, cmd_byte)."""
        with self._rx_lock:
            return self.latest_rx_frames_by_key.get((arbitration_id, cmd_byte))

    # ---------------
    # Event Managing
    # ---------------
    def event_for_key(self, arbitration_id: int, cmd_byte: int) -> threading.Event:
        """Return (creating if needed) the arrival Event for one reply key.
        Lazily allocated so we only carry events for keys the hot path uses."""
        key = (arbitration_id, cmd_byte)
        with self._events_lock:
            ev = self.frame_events.get(key)
            if ev is None:
                ev = threading.Event()
                self.frame_events[key] = ev
            return ev

    def alias_event_keys(self, key_a: tuple[int, int], key_b: tuple[int, int]) -> threading.Event:
        """Make two reply keys share ONE arrival Event.
        A motor may answer the same command on either rx_id (0x180+id) or
        tx_id (0x140+id) depending on its setup."""
        with self._events_lock:
            ev = (self.frame_events.get(key_a)
                  or self.frame_events.get(key_b)
                  or threading.Event())
            self.frame_events[key_a] = ev
            self.frame_events[key_b] = ev
            return ev

    # ------------------------------------------------------------------
    # Background reader thread 
    # ------------------------------------------------------------------
    def start_reader_thread(self) -> None:
        """Spawn a daemon thread that drains the bus into a per-key cache.

        Must be called AFTER all init-time txrx() calls have completed.
        Once running, callers must use send_only()/get_latest_frame().
        """
        if self._rx_thread is not None and self._rx_thread.is_alive():
            return
        if self.bus is None:
            raise RuntimeError("CAN bus not opened; call open() before start_reader_thread().")
        self._rx_stop_event.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop,
                                           daemon=True,
                                           name=f"can-rx-{self.channel}")
        self._rx_thread.start()
        self.logger.info(f"[CAN] reader thread started on {self.channel}")

    def stop_reader_thread(self, timeout: float = 1.0) -> None:
        """Signal the reader thread to exit and join it."""
        self._rx_stop_event.set()
        if self.is_reader_running():
            self._rx_thread.join(timeout=timeout)
        if self._rx_thread.is_alive():
            raise RuntimeError(f"CAN reader thread on {self.channel} did not stop.")
        self._rx_thread = None


    def _rx_loop(self) -> None:
        """Continuously drain the bus into latest_rx_frames_by_key."""
        while not self._rx_stop_event.is_set():
            try:
                msg = self.bus.recv(timeout=0.05) if self.bus is not None else None
            except Exception as exc:
                self.logger.warning(f"[CAN] reader recv error on {self.channel}: {exc}")
                time.sleep(0.01)
                continue
            if msg is None or not msg.data:
                continue
            # Arbiration id : which motor ? msg.data[0] : which cause this reply ?
            key = (msg.arbitration_id, msg.data[0])
            with self._rx_lock:
                self.latest_rx_frames_by_key[key] = msg
                self.rx_frame_count += 1
            ev = self.frame_events.get(key)
            if ev is not None:
                ev.set()
            if key not in self._rx_first_keys_seen and len(self._rx_first_keys_seen) < 16:
                self._rx_first_keys_seen.add(key)
                self.logger.info(f"[CAN] {self.channel} first frame for arb_id=0x{key[0]:03X} "
                                 f"cmd=0x{key[1]:02X} (total seen={self.rx_frame_count})")


    # -----------------------
    # Transmit (Tx) helpers
    # -----------------------
    def send_only(self, arbitration_id: int, data: bytes) -> None:
        """Fire-and-forget send. No lock, no response wait."""
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")
        try:
            self.bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False))
        except can.CanError as can_error:
            self.logger.error(f"[CAN] send_only failed (id=0x{arbitration_id:X}): {can_error}")

    def send(self, arbitration_id: int, data: bytes) -> can.Message | None:
        """Send a raw CAN frame."""
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")
        try:
            sent_message = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)
            self.bus.send(sent_message)
            return sent_message
        except can.CanError as can_error:
            self.logger.error(f"[CAN] send failed (id=0x{arbitration_id:X}): {can_error}")
            return None

    # -----------------------
    # Read (Rx) helpers
    # -----------------------
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

    # -----------------------
    # Initialization
    # -----------------------
    def txrx(self,
            tx_id: int,
            rx_id: int,
            cmd_byte: int,
            payload7: bytes,
            timeout: float = 0.5,
            accept_rx_id: bool = True,
            accept_tx_echo_diff: bool = True) -> can.Message | None:
        """Transmit one command frame and wait for a matching response for Initialization."""
        if len(payload7) != 7:
            raise ValueError("payload7 must be 7 bytes")

        transmitted_data = bytes([cmd_byte]) + payload7

        # Acquire bus lock, then drain stale frames left by the previous motor's txrx on this bus. 
        # Without the drain, the next motor pulls the previous motor's response out of recv().
        with self.txrx_lock:
            # Stale-frame purge: pull anything already sitting in the kernel RX queue.
            drained = self._drain_rx(max_frames=200)
            if drained > 0:
                self.logger.debug(f"[CAN] drained {drained} stale RX frames before txrx")

            # Issue the command on the bus; abort early if send itself failed.
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

                if accept_rx_id and (received_message.arbitration_id == rx_id):
                    return received_message

                if (accept_tx_echo_diff and (received_message.arbitration_id == tx_id) and (received_message.data != sent_message.data)):
                    return received_message

            return None
