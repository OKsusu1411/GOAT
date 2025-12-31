# goat_control/core/comm/protocol.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanIds:
    """Standard MG-series CAN IDs for a given node id."""
    tx_id: int
    rx_id: int


def mg_ids(node_id: int) -> CanIds:
    """Return (tx_id, rx_id) for MG-series motors."""
    node_id_int = int(node_id)
    return CanIds(tx_id=0x140 + node_id_int, rx_id=0x180 + node_id_int)


# Common constants
E7 = b"\x00" * 7

# MG-series current encoding:
# ±33 A  <->  ±2048 LSB  (signed)
MG_IQ_LSB_PER_A = 2048.0 / 33.0


def pack_iq_from_amp(amps: float) -> bytes:
    """Pack motor current (amps) into MG-series iq LSB (2 bytes, little-endian, signed)."""
    clamped_amps = max(min(float(amps), 33.0), -33.0)

    iq_lsb = int(round(clamped_amps * MG_IQ_LSB_PER_A))
    iq_lsb = max(min(iq_lsb, 2048), -2048)

    return int(iq_lsb).to_bytes(2, byteorder="little", signed=True)


def payload_torque_mode_from_amp(amps: float) -> bytes:
    """Build 0xA1 torque-mode payload (7 bytes): 00 00 00 + iq(2B) + 00 00."""
    packed_iq = pack_iq_from_amp(amps)
    return b"\x00\x00\x00" + packed_iq + b"\x00\x00"
