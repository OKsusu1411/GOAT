# core/comm/can.py
from __future__ import annotations

import time
import threading
import logging
from typing import Optional

import can


class CanInterface:
    """
    CAN transport 전담 클래스.
    - open/close
    - send/recv
    - txrx (에코 필터 포함)
    주의: txrx는 락으로 직렬화(응답 프레임 뺏김 방지)
    """

    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        bitrate: int | None = None,  # socketcan은 보통 OS에서 이미 설정됨
        receive_own_messages: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.channel = channel
        self.interface = interface
        self.bitrate = bitrate
        self.receive_own_messages = receive_own_messages

        self._log = logger or logging.getLogger(self.__class__.__name__)
        self.bus: can.BusABC | None = None

        # 여러 곳에서 동시에 txrx하면 recv 프레임을 서로 뺏어먹음 → 우선 직렬화
        self._txrx_lock = threading.Lock()

    def open(self) -> None:
        if self.bus is not None:
            return
        self.bus = can.Bus(
            interface=self.interface,
            channel=self.channel,
            bitrate=self.bitrate,
            receive_own_messages=self.receive_own_messages,
        )
        self._log.info(f"[CAN] opened: {self.interface}:{self.channel}")

    def close(self) -> None:
        if self.bus is None:
            return
        self.bus.shutdown()
        self.bus = None
        self._log.info("[CAN] closed")

    def send(self, arbitration_id: int, data: bytes) -> Optional[can.Message]:
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")
        try:
            msg = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)
            self.bus.send(msg)
            return msg
        except can.CanError as caneError:
            self._log.error(f"[CAN] send failed (id=0x{arbitration_id:X}): {caneError}")
            return None

    def receive(self, timeout: float = 0.05) -> Optional[can.Message]:
        if self.bus is None:
            raise RuntimeError("CAN bus is not opened. Call open() first.")
        return self.bus.recv(timeout=timeout)

    def txrx(
        self,
        tx_id: int,
        rx_id: int,
        cmd_byte: int,
        payload7: bytes,
        timeout: float = 0.5,
        accept_rx_id: bool = True,
        accept_tx_echo_diff: bool = True,
    ) -> Optional[can.Message]:
        """
        - 정상 응답(0x180+ID)을 우선 허용(accept_rx_id=True 권장)
        - 일부 환경에서 TX ID로 '내용이 다르면' 실제 응답으로 간주(accept_tx_echo_diff=True)
        - 순수 루프백(보낸 프레임과 data 완전 동일)은 무시
        """
        if len(payload7) != 7:
            raise ValueError("payload7 must be 7 bytes")

        data = bytes([cmd_byte]) + payload7

        with self._txrx_lock:
            sent = self.send(tx_id, data)
            if sent is None:
                return None

            deadline = time.time() + timeout
            while time.time() < deadline:
                message = self.receive(timeout=min(0.05, max(0.0, deadline - time.time())))
                if not message:
                    continue
                if len(message.data) != 8 or message.data[0] != cmd_byte:
                    continue

                if accept_rx_id and (message.arbitration_id == rx_id):
                    return message
                if accept_tx_echo_diff and (message.arbitration_id == tx_id) and (message.data != sent.data):
                    return message

            return None
