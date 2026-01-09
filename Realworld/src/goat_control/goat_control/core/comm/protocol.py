# goat_control/core/comm/protocol.py
from __future__ import annotations

from dataclasses import dataclass


# -----------------------
# CAN IDs
# -----------------------
@dataclass(frozen=True)
class CanIds:
    """Standard MG-series CAN IDs for a given node id."""
    tx_id: int
    rx_id: int


def mg_ids(node_id: int) -> CanIds:
    """Return (tx_id, rx_id) for MG-series motors."""
    node_id_int = int(node_id)
    return CanIds(tx_id=0x140 + node_id_int, rx_id=0x180 + node_id_int)


# -----------------------
# Common constants
# -----------------------
E7 = b"\x00" * 7  # payload7 = DATA[1..7]


# -----------------------
# Unit scales (YAML-driven)
# -----------------------
@dataclass(frozen=True)
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


# Default values = your YAML values
_DEFAULT_MG_UNIT_SCALES = MGUnitScales(
    motor_current_amp_per_lsb=0.01611328125,  # 66.0 / 4096.0
    angle_deg_per_lsb=0.001,
    speed_deg_per_sec_per_lsb=0.01,
)

_current_mg_unit_scales: MGUnitScales = _DEFAULT_MG_UNIT_SCALES


def set_mg_unit_scales(
    motor_current_amp_per_lsb: float,
    angle_deg_per_lsb: float,
    speed_deg_per_sec_per_lsb: float,
) -> None:
    """
    Call once at startup after loading YAML.
    """
    global _current_mg_unit_scales
    _current_mg_unit_scales = MGUnitScales(
        motor_current_amp_per_lsb=float(motor_current_amp_per_lsb),
        angle_deg_per_lsb=float(angle_deg_per_lsb),
        speed_deg_per_sec_per_lsb=float(speed_deg_per_sec_per_lsb),
    )


def get_mg_unit_scales() -> MGUnitScales:
    return _current_mg_unit_scales


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
    unit_scales = _current_mg_unit_scales
    motor_current_amp_per_lsb = unit_scales.motor_current_amp_per_lsb
    return int(round(float(current_amp) / motor_current_amp_per_lsb))


def angle_deg_to_lsb(angle_deg: float) -> int:
    unit_scales = _current_mg_unit_scales
    angle_deg_per_lsb = unit_scales.angle_deg_per_lsb
    return int(round(float(angle_deg) / angle_deg_per_lsb))


def speed_deg_per_sec_to_lsb(speed_deg_per_sec: float) -> int:
    unit_scales = _current_mg_unit_scales
    speed_deg_per_sec_per_lsb = unit_scales.speed_deg_per_sec_per_lsb
    return int(round(float(speed_deg_per_sec) / speed_deg_per_sec_per_lsb))


def lsb_to_current_amp(current_lsb: int) -> float:
    unit_scales = _current_mg_unit_scales
    return float(current_lsb) * unit_scales.motor_current_amp_per_lsb


def lsb_to_angle_deg(angle_lsb: int) -> float:
    unit_scales = _current_mg_unit_scales
    return float(angle_lsb) * unit_scales.angle_deg_per_lsb


def lsb_to_speed_deg_per_sec(speed_lsb: int) -> float:
    unit_scales = _current_mg_unit_scales
    return float(speed_lsb) * unit_scales.speed_deg_per_sec_per_lsb


# -----------------------
# Existing core API (A1 uses YAML scale)
# -----------------------
def pack_iq_from_amp(current_amp: float) -> bytes:
    """
    Pack motor current (amps) into iq LSB (2 bytes, little-endian, signed).

    Uses YAML scale:
      motor_current_amp_per_lsb

    For MG-series typical range: +/-2048 LSB.
    """
    max_current_lsb = 2048
    max_current_amp = float(max_current_lsb) * _current_mg_unit_scales.motor_current_amp_per_lsb

    clamped_current_amp = max(min(float(current_amp), max_current_amp), -max_current_amp)

    current_lsb = current_amp_to_lsb(clamped_current_amp)
    current_lsb = max(min(current_lsb, max_current_lsb), -max_current_lsb)

    return pack_int16_little_endian_signed(current_lsb)


def payload_torque_mode_from_amp(current_amp: float) -> bytes:
    """Build 0xA1 torque-mode payload (7 bytes): 00 00 00 + iq(2B) + 00 00."""
    packed_current_lsb_bytes = pack_iq_from_amp(current_amp)
    return b"\x00\x00\x00" + packed_current_lsb_bytes + b"\x00\x00"


# -----------------------
# Additional command payload builders (not used yet)
# NOTE: payload7 corresponds to DATA[1]..DATA[7]
# Full CAN data = [command_byte] + payload7  (8 bytes)
# -----------------------
def payload_all_zeros_7bytes() -> bytes:
    """Return 7 bytes of zeros (DATA[1..7])."""
    return E7


def payload_open_loop_power_control(power_control_int16: int) -> bytes:
    """
    Command 0xA0 (open-loop):
      DATA[4..5] = int16 powerControl
      others = 0
    payload7 = DATA[1..7] = 00 00 00 + power(2B) + 00 00
    """
    if power_control_int16 < -850:
        power_control_int16 = -850
    if power_control_int16 > 850:
        power_control_int16 = 850

    power_control_bytes = pack_int16_little_endian_signed(power_control_int16)
    payload7_bytes = b"\x00\x00\x00" + power_control_bytes + b"\x00\x00"
    return payload7_bytes


def payload_speed_closed_loop(iq_limit_amps: float, target_speed_deg_per_sec: float) -> bytes:
    """
    Command 0xA2 (speed closed-loop):
      DATA[2..3] = int16 iqControl (same LSB encoding as A1 iq)
      DATA[4..7] = int32 speedControl (scaled by YAML: speed_deg_per_sec_per_lsb)

    payload7 = DATA[1..7] = [DATA1] + [DATA2..3] + [DATA4..7]
             = 00 + iq(2B) + speed(4B)
    """
    motor_current_lsb_bytes = pack_iq_from_amp(iq_limit_amps)

    target_speed_lsb_int = speed_deg_per_sec_to_lsb(target_speed_deg_per_sec)
    target_speed_lsb_bytes = pack_int32_little_endian_signed(target_speed_lsb_int)

    payload7_bytes = b"\x00" + motor_current_lsb_bytes + target_speed_lsb_bytes
    return payload7_bytes


def payload_position_multi_turn_mode1(target_angle_deg: float) -> bytes:
    """
    Command 0xA3 (multi-turn position closed-loop #1):
      DATA[4..7] = int32 angleControl (scaled by YAML: angle_deg_per_lsb)

    payload7 = 00 00 00 + angle(4B)
    """
    target_angle_lsb_int = angle_deg_to_lsb(target_angle_deg)
    target_angle_lsb_bytes = pack_int32_little_endian_signed(target_angle_lsb_int)

    payload7_bytes = b"\x00\x00\x00" + target_angle_lsb_bytes
    return payload7_bytes


def payload_position_multi_turn_mode2(max_speed_dps_uint16: int, target_angle_deg: float) -> bytes:
    """
    Command 0xA4 (multi-turn position closed-loop #2):
      DATA[2..3] = uint16 maxSpeed (1 dps/LSB per your table examples)
      DATA[4..7] = int32 angleControl (scaled by YAML: angle_deg_per_lsb)

    payload7 = [DATA1]=00 + [DATA2..3]=maxSpeed(2B) + [DATA4..7]=angle(4B)
    """
    max_speed_bytes = pack_uint16_little_endian(max_speed_dps_uint16)

    target_angle_lsb_int = angle_deg_to_lsb(target_angle_deg)
    target_angle_bytes = pack_int32_little_endian_signed(target_angle_lsb_int)

    payload7_bytes = b"\x00" + max_speed_bytes + target_angle_bytes
    return payload7_bytes


def payload_position_single_turn_mode1(spin_direction_uint8: int, target_angle_deg: float) -> bytes:
    """
    Command 0xA5 (single-turn position #1):
      DATA[1]   = uint8 spinDirection (0:CW, 1:CCW)
      DATA[4..7]= uint32 angleControl (scaled by YAML: angle_deg_per_lsb)

    payload7 = [DATA1]=spinDir + [DATA2..3]=00 00 + [DATA4..7]=angle(4B)
    """
    spin_direction_byte = bytes([int(spin_direction_uint8) & 0xFF])

    # single-turn is usually non-negative; keep it non-negative for uint32
    if float(target_angle_deg) < 0.0:
        target_angle_deg = 0.0

    target_angle_lsb_int = angle_deg_to_lsb(target_angle_deg)
    target_angle_lsb_uint32_bytes = pack_uint32_little_endian(target_angle_lsb_int)

    payload7_bytes = spin_direction_byte + b"\x00\x00" + target_angle_lsb_uint32_bytes
    return payload7_bytes


def payload_position_single_turn_mode2(
    spin_direction_uint8: int,
    max_speed_dps_uint16: int,
    target_angle_deg: float
) -> bytes:
    """
    Command 0xA6 (single-turn position #2):
      DATA[1]   = uint8  spinDirection (0:CW, 1:CCW)
      DATA[2..3]= uint16 maxSpeed (1 dps/LSB)
      DATA[4..7]= uint32 angleControl (scaled by YAML: angle_deg_per_lsb)
    """
    spin_direction_byte = bytes([int(spin_direction_uint8) & 0xFF])
    max_speed_bytes = pack_uint16_little_endian(max_speed_dps_uint16)

    if float(target_angle_deg) < 0.0:
        target_angle_deg = 0.0

    target_angle_lsb_int = angle_deg_to_lsb(target_angle_deg)
    target_angle_lsb_uint32_bytes = pack_uint32_little_endian(target_angle_lsb_int)

    payload7_bytes = spin_direction_byte + max_speed_bytes + target_angle_lsb_uint32_bytes
    return payload7_bytes


def payload_increment_position_mode1(delta_angle_deg: float) -> bytes:
    """
    Command 0xA7 (incremental position #1):
      DATA[4..7] = int32 angleIncrement (scaled by YAML: angle_deg_per_lsb)
    """
    delta_angle_lsb_int = angle_deg_to_lsb(delta_angle_deg)
    delta_angle_bytes = pack_int32_little_endian_signed(delta_angle_lsb_int)

    payload7_bytes = b"\x00\x00\x00" + delta_angle_bytes
    return payload7_bytes


def payload_increment_position_mode2(max_speed_dps_uint16: int, delta_angle_deg: float) -> bytes:
    """
    Command 0xA8 (incremental position #2):
      DATA[2..3] = uint16 maxSpeed (1 dps/LSB)
      DATA[4..7] = int32 angleIncrement (scaled by YAML: angle_deg_per_lsb)
    """
    max_speed_bytes = pack_uint16_little_endian(max_speed_dps_uint16)

    delta_angle_lsb_int = angle_deg_to_lsb(delta_angle_deg)
    delta_angle_bytes = pack_int32_little_endian_signed(delta_angle_lsb_int)

    payload7_bytes = b"\x00" + max_speed_bytes + delta_angle_bytes
    return payload7_bytes


def payload_write_int32_into_data4_to_data7(value_int32: int) -> bytes:
    """
    For commands like 0x34(accel) / 0x38(max torque) examples:
      DATA[4..7] = int32 value
    payload7 = 00 00 00 + int32(4B)
    """
    value_bytes = pack_int32_little_endian_signed(value_int32)
    payload7_bytes = b"\x00\x00\x00" + value_bytes
    return payload7_bytes
