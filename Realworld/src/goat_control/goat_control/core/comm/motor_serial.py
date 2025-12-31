# core/comm/motor_serial.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .can import CanInterface

@dataclass
class MotorParams:
    node_id: int
    gear_ratio: float = 1.0
    torque_constant_nm_per_a: float = 0.0  # 실험으로 구한 값 넣기
    direction: int = 1  # +1 or -1
    torque_limit_nm: float | None = None

@dataclass
class MotorState:
    # 필요한 것만 시작하고 점점 확장
    voltage_v: float | None = None
    iq_a: float | None = None
    speed_dps: float | None = None
    pos_deg: float | None = None
    raw_msg: object | None = None

class MotorDriver:
    """단일 모터 드라이버(명령/상태). transport는 CanInterface에 위임."""
    def __init__(self, can_if: CanInterface, params: MotorParams):
        self.can = can_if
        self.p = params

    # ---- 예시: 읽기(프로토콜 파싱은 나중에 protocol.py로 빼는 걸 추천)
    def read_state2(self, timeout: float = 0.05) -> MotorState | None:
        # cmd 0x9C: temp, iq, speed, encoder ... (네가 이미 쓰던 것 기반)
        m = self.can.txrx_mg(self.p.node_id, 0x9C, timeout=timeout)
        if m is None:
            return None
        st = MotorState(raw_msg=m)
        # TODO: protocol.py로 decode 함수 만들고 여기선 호출만 하도록
        return st

    # ---- 예시: 토크(전류) 명령
    def set_torque_amps(self, amps: float, timeout: float = 0.05):
        # cmd 0xA1: torque closed-loop
        # payload[3:5] = iq (2 bytes little-endian signed) 라는 기존 로직을 재사용
        # (pack 함수는 protocol.py로 빼는 걸 권장)
        iq_lsb_per_a = 2048.0 / 33.0
        a = max(min(float(amps), 33.0), -33.0)
        iq = int(round(a * iq_lsb_per_a))
        iq = max(min(iq, 2048), -2048)
        iq_bytes = int(iq).to_bytes(2, byteorder="little", signed=True)
        payload = b"\x00\x00\x00" + iq_bytes + b"\x00\x00"  # 7 bytes
        return self.can.txrx_mg(self.p.node_id, 0xA1, payload, timeout=timeout)
