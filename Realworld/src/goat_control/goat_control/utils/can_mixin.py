# lk_can_mixin.py
import time
import logging
import can

class CanMixin:
    """LingKong(MG 시리즈) CAN 공용 루틴: TX/RX, 에코 필터, 유틸리티."""
    E7 = b'\x00' * 7
    MG_IQ_LSB_PER_A = 2048.0 / 33.0  # ≈ 62.0606 LSB/A (MG: ±33A ↔ ±2048)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    def _log(self):
        try:
            return self.get_logger()
        except Exception:
            # ROS2 Node가 아니더라도 동작하도록
            return logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _ids(node_id: int):
        nid = int(node_id)
        return (0x140 + nid, 0x180 + nid)  # (tx_id, rx_id)

    # ------------------------------------------------------------------
    # 송신만
    def can_send(self, node_id: int, cmd_byte: int, payload7: bytes = E7):
        tx_id, _ = self._ids(node_id)
        data = bytes([cmd_byte]) + payload7
        try:
            msg = can.Message(arbitration_id=tx_id, data=data, is_extended_id=False)
            self.bus.send(msg)
            return msg
        except can.CanError as e:
            self._log().error(f"[CAN] send failed (node {node_id}, cmd 0x{cmd_byte:02X}): {e}")
            return None

    # ------------------------------------------------------------------
    # 송수신(에코 필터 포함)
    def can_txrx(self,
                 node_id: int,
                 cmd_byte: int,
                 payload7: bytes = E7,
                 timeout: float = 0.5,
                 accept_rx_id: bool = False,
                 accept_tx_echo_diff: bool = True):
        """
        - 기본은 정상 응답(0x180+ID)을 우선 허용(accept_rx_id=True)
        - 일부 하드웨어에서 TX ID로 돌아오는 프레임이 '내용이 다르면' 실제 응답으로 간주(accept_tx_echo_diff=True)
        - 순수 루프백(보낸 프레임과 data 완전 동일)은 항상 무시
        """
        sent = self.can_send(node_id, cmd_byte, payload7)
        if sent is None:
            return None

        tx_id, rx_id = self._ids(node_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            m = self.bus.recv(timeout=min(0.05, max(0.0, deadline - time.time())))
            if not m:
                continue
            if len(m.data) != 8 or m.data[0] != cmd_byte:
                continue

            # 1) 정상 프로토콜 응답(0x180+ID)
            if accept_rx_id and (m.arbitration_id == rx_id):
                return m

            # 2) 동일 ID(0x140+ID)지만 내용이 다른 경우 → 실제 응답으로 간주
            if accept_tx_echo_diff and (m.arbitration_id == tx_id) and (m.data != sent.data):
                return m

            # 3) 그 외: 루프백 에코 혹은 다른 프레임 → 무시
        return None

    # ------------------------------------------------------------------
    # 유틸(토크전류 iq packing) — MG 전용
    @classmethod
    def pack_iq_from_amp(cls, amps: float) -> bytes:
        """MG: ±33A ↔ ±2048 LSB. 포화 및 little-endian 2바이트."""
        a = max(min(float(amps), 33.0), -33.0)
        iq = int(round(a * cls.MG_IQ_LSB_PER_A))  # signed
        if iq < -2048: iq = -2048
        if iq > 2048: iq = 2048
        return int(iq).to_bytes(2, byteorder='little', signed=True)

    # ------------------------------------------------------------------
    # 자주 쓰는 명령 래퍼 (필요하면 확장)
    def cmd_read_state1(self, node_id: int, timeout=0.05):
        # 0x9B: 상태1 (전압, 전류, 위치)
        return self.can_txrx(node_id, 0x9A, self.E7, timeout)

    def cmd_read_state2(self, node_id: int, timeout=0.05):
        # 0x9C: temp, iq, speed, encoder
        return self.can_txrx(node_id, 0x9C, self.E7, timeout)

    def cmd_read_multi_turn(self, node_id: int, timeout=0.05):
        # 0x92: multi-turn angle (int64, 0.01°/LSB)
        return self.can_txrx(node_id, 0x92, self.E7, timeout)

    def cmd_read_single_turn(self, node_id: int, timeout=0.05):
        # 0x94: single-turn angle (uint32, 0.01°/LSB)
        return self.can_txrx(node_id, 0x94, self.E7, timeout)

    def cmd_torque_mode(self, node_id: int, amps: float, timeout=0.05):
        # 0xA1: torque closed-loop, payload[4:6] = iq
        iq = self.pack_iq_from_amp(amps)
        payload = b'\x00\x00\x00' + iq + b'\x00\x00'  # 총 7바이트
        return self.can_txrx(node_id, 0xA1, payload, timeout)
