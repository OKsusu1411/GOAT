# goat_control/core/comm/protocol.py
from __future__ import annotations

import yaml

from dataclasses import dataclass

# -----------------------
# Parameters by Yaml File
# -----------------------
YAML_PATH = "src/goat_control/config/goat_config.yaml"
with open(YAML_PATH, "r", encoding="utf-8") as file_handle:
    CFG = yaml.safe_load(file_handle)
if not isinstance(CFG, dict):
    raise ValueError("YAML root must be a mapping/dict.")

# -----------------------
# Common constants
# -----------------------
E7 = b"\x00" * 7  # payload7 = DATA[1..7]

# -----------------------
# Unit scales
# -----------------------
@dataclass
class MGUnitScales:
    """
    Unit scales that convert between LSB and physical units.
    These should match your YAML:
      estimation:
        motor_current_amp_per_lsb
        angle_deg_per_lsb
        speed_deg_per_sec_per_lsb
    """
    motor_current_amp_per_lsb: float
    angle_deg_per_lsb: float
    speed_deg_per_sec_per_lsb: float
    max_current_per_lsb: float

_DEFAULT_MG_UNIT_SCALES = MGUnitScales(
    motor_current_amp_per_lsb=CFG["motor_current_amp_per_lsb"],
    angle_deg_per_lsb=CFG["angle_deg_per_lsb"],
    speed_deg_per_sec_per_lsb=CFG["speed_deg_per_sec_per_lsb"],
    max_current_per_lsb=CFG["max_current_per_lsb"])


# -----------------------
# CAN IDs
# -----------------------
@dataclass
class CanIds:
    """Standard MG-series CAN IDs for a given node id."""
    tx_id: int
    rx_id: int


def mg_ids(node_id: int) -> CanIds:
    """Return (tx_id, rx_id) for MG-series motors."""
    node_id_int = int(node_id)
    return CanIds(tx_id=0x140 + node_id_int, rx_id=0x180 + node_id_int)

# -----------------------
# Generic packers (little-endian)
# -----------------------
def pack_int16_little_endian_signed(value_int: int) -> bytes:
    """Pack signed int16 into 2 bytes (little-endian)."""
    return int(value_int).to_bytes(2, byteorder="little", signed=True)


def pack_uint16_little_endian(value_int: int) -> bytes:
    """Pack unsigned uint16 into 2 bytes (little-endian) with saturation."""
    value_int = int(value_int)
    if value_int < 0:
        value_int = 0
    if value_int > 0xFFFF:
        value_int = 0xFFFF
    return value_int.to_bytes(2, byteorder="little", signed=False)


def pack_int32_little_endian_signed(value_int: int) -> bytes:
    """Pack signed int32 into 4 bytes (little-endian)."""
    return int(value_int).to_bytes(4, byteorder="little", signed=True)


def pack_uint32_little_endian(value_int: int) -> bytes:
    """Pack unsigned uint32 into 4 bytes (little-endian) with saturation."""
    value_int = int(value_int)
    if value_int < 0:
        value_int = 0
    if value_int > 0xFFFFFFFF:
        value_int = 0xFFFFFFFF
    return value_int.to_bytes(4, byteorder="little", signed=False)


# -----------------------
# Unit conversion helpers (physical <-> LSB)
# -----------------------
def current_amp_to_lsb(current_amp: float) -> int:
    unit_scales = _DEFAULT_MG_UNIT_SCALES
    motor_current_amp_per_lsb = unit_scales.motor_current_amp_per_lsb
    return int(round(float(current_amp) / motor_current_amp_per_lsb))


def angle_deg_to_lsb(angle_deg: float) -> int:
    unit_scales = _DEFAULT_MG_UNIT_SCALES
    angle_deg_per_lsb = unit_scales.angle_deg_per_lsb
    return int(round(float(angle_deg) / angle_deg_per_lsb))


def speed_deg_per_sec_to_lsb(speed_deg_per_sec: float) -> int:
    unit_scales = _DEFAULT_MG_UNIT_SCALES
    speed_deg_per_sec_per_lsb = unit_scales.speed_deg_per_sec_per_lsb
    return int(round(float(speed_deg_per_sec) / speed_deg_per_sec_per_lsb))


def lsb_to_current_amp(current_lsb: int) -> float:
    unit_scales = _DEFAULT_MG_UNIT_SCALES
    return float(current_lsb) * unit_scales.motor_current_amp_per_lsb


def lsb_to_angle_deg(angle_lsb: int) -> float:
    unit_scales = _DEFAULT_MG_UNIT_SCALES
    return float(angle_lsb) * unit_scales.angle_deg_per_lsb


def lsb_to_speed_deg_per_sec(speed_lsb: int) -> float:
    unit_scales = _DEFAULT_MG_UNIT_SCALES
    return float(speed_lsb) * unit_scales.speed_deg_per_sec_per_lsb


# -----------------------
# Existing core API (A1 uses YAML scale)
# -----------------------
def pack_iq_from_amp(current_amp: float) -> bytes:
    """
    Pack motor current (amps) into iq LSB (2 bytes, little-endian, signed).
    """
    max_current_per_lsb = _DEFAULT_MG_UNIT_SCALES.max_current_per_lsb
    max_current_amp = float(max_current_per_lsb) * _DEFAULT_MG_UNIT_SCALES.motor_current_amp_per_lsb

    clamped_current_amp = max(min(float(current_amp), max_current_amp), -max_current_amp)

    current_lsb = current_amp_to_lsb(clamped_current_amp)
    current_lsb = max(min(current_lsb, max_current_per_lsb), -max_current_per_lsb)

    return pack_int16_little_endian_signed(current_lsb)


def payload_torque_mode_from_amp(current_amp: float) -> bytes:
    """Build 0xA1 torque-mode payload (7 bytes): 00 00 00 + iq(2B) + 00 00."""
    packed_current_lsb_bytes = pack_iq_from_amp(current_amp)
    return b"\x00\x00\x00" + packed_current_lsb_bytes + b"\x00\x00"