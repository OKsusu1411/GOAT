# goat_control/core/comm/motor_driver.py
from __future__ import annotations

from dataclasses import dataclass

from .can import CanInterface
from . import protocol


@dataclass
class MotorParams:
    """Static configuration for a single motor."""
    node_id: int
    direction: int = 1  # +1 or -1
    gear_ratio: float = 1.0
    torque_constant_nm_per_a: float = 0.0  # Identified experimentally
    torque_limit_nm: float | None = None


class MotorDriver:
    """Single motor driver abstraction.

    This class does NOT own the CAN bus.
    It uses:
      - CanInterface for transport (raw TX/RX)
      - protocol module for MG-series CAN IDs and payload encoding
    """

    def __init__(self, can_interface: CanInterface, motor_params: MotorParams):
        self.can_interface = can_interface
        self.motor_params = motor_params
        self.can_ids = protocol.mg_ids(self.motor_params.node_id)

    def _txrx(self, command_byte: int, payload7: bytes = protocol.E7, timeout: float = 0.05):
        """Low-level helper: send a command and wait for a response."""
        return self.can_interface.txrx(
            tx_id=self.can_ids.tx_id,
            rx_id=self.can_ids.rx_id,
            cmd_byte=command_byte,
            payload7=payload7,
            timeout=timeout,
            accept_rx_id=True,
            accept_tx_echo_diff=True,
        )

    # ---- Command wrappers (extend as needed)
    def read_state1(self, timeout: float = 0.05):
        """Read state1 (voltage, current, position) error flags."""
        return self._txrx(0x9A, protocol.E7, timeout)

    def read_state2(self, timeout: float = 0.05):
        """Read state2 (torque/current, output voltage, speed, encoder position).

        Note: angle unit is 0.001°/LSB and speed unit is typically 0.01°/s per LSB (per YAML/manual).
        """
        return self._txrx(0x9C, protocol.E7, timeout)

    def read_multi_turn(self, timeout: float = 0.05):
        '''Read multi-turn angle (int64, 0.001°/LSB, degree per second : 0.01/LSB).'''
        return self._txrx(0x92, protocol.E7, timeout)

    def read_single_turn(self, timeout: float = 0.05):
        '''Read single-turn angle (uint32, position : 0.001°/LSB, degree per second : 0.01/LSB).'''
        return self._txrx(0x94, protocol.E7, timeout)

    def torque_mode_amp(self, amps: float, timeout: float = 0.05):
        """Set torque mode with specified current (amps)."""
        torque_payload = protocol.payload_torque_mode_from_amp(amps)
        return self._txrx(0xA1, torque_payload, timeout)

    # additional command wrappers (not used yet)

    # ------------------------
    # Motor power / stop / run
    # ------------------------
    def motor_power_switch(self, timeout: float = 0.05):
        """0x80: Motor power ON/OFF."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x80, payload7_bytes, timeout)

    def motor_stop(self, timeout: float = 0.05):
        """0x81: Motor stop."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x81, payload7_bytes, timeout)

    def motor_run(self, timeout: float = 0.05):
        """0x88: Motor run/start."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x88, payload7_bytes, timeout)

    # ------------------------
    # Control modes (A0 ~ A8)
    # ------------------------
    def open_loop_control(self, power_control_int16: int, timeout: float = 0.05):
        """0xA0: Open-loop control."""
        payload7_bytes = protocol.payload_open_loop_power_control(power_control_int16)
        return self._txrx(0xA0, payload7_bytes, timeout)

    def speed_closed_loop(
        self,
        target_speed_deg_per_sec: float,
        iq_limit_amps: float = 0.0,
        timeout: float = 0.05,
    ):
        """0xA2: Speed closed-loop."""
        payload7_bytes = protocol.payload_speed_closed_loop(iq_limit_amps=iq_limit_amps, target_speed_deg_per_sec=target_speed_deg_per_sec)
        return self._txrx(0xA2, payload7_bytes, timeout)

    def position_multi_turn_mode1(self, target_angle_deg: float, timeout: float = 0.05):
        """0xA3: Multi-turn position mode1."""
        payload7_bytes = protocol.payload_position_multi_turn_mode1(target_angle_deg=target_angle_deg)
        return self._txrx(0xA3, payload7_bytes, timeout)

    def position_multi_turn_mode2(self, target_angle_deg: float, max_speed_dps_uint16: int, timeout: float = 0.05):
        """0xA4: Multi-turn position mode2."""
        payload7_bytes = protocol.payload_position_multi_turn_mode2(
            max_speed_dps_uint16=max_speed_dps_uint16,
            target_angle_deg=target_angle_deg,
        )
        return self._txrx(0xA4, payload7_bytes, timeout)

    def position_single_turn_mode1(self, target_angle_deg: float, spin_direction_uint8: int = 0, timeout: float = 0.05):
        """0xA5: Single-turn position mode1. spin_direction: 0=CW, 1=CCW"""
        payload7_bytes = protocol.payload_position_single_turn_mode1(
            spin_direction_uint8=spin_direction_uint8,
            target_angle_deg=target_angle_deg,
        )
        return self._txrx(0xA5, payload7_bytes, timeout)

    def position_single_turn_mode2(
        self,
        target_angle_deg: float,
        max_speed_dps_uint16: int,
        spin_direction_uint8: int = 0,
        timeout: float = 0.05,
    ):
        """0xA6: Single-turn position mode2."""
        payload7_bytes = protocol.payload_position_single_turn_mode2(
            spin_direction_uint8=spin_direction_uint8,
            max_speed_dps_uint16=max_speed_dps_uint16,
            target_angle_deg=target_angle_deg,
        )
        return self._txrx(0xA6, payload7_bytes, timeout)

    def increment_position_mode1(self, delta_angle_deg: float, timeout: float = 0.05):
        """0xA7: Incremental position mode1."""
        payload7_bytes = protocol.payload_increment_position_mode1(delta_angle_deg=delta_angle_deg)
        return self._txrx(0xA7, payload7_bytes, timeout)

    def increment_position_mode2(self, delta_angle_deg: float, max_speed_dps_uint16: int, timeout: float = 0.05):
        """0xA8: Incremental position mode2."""
        payload7_bytes = protocol.payload_increment_position_mode2(
            max_speed_dps_uint16=max_speed_dps_uint16,
            delta_angle_deg=delta_angle_deg,
        )
        return self._txrx(0xA8, payload7_bytes, timeout)

    # ------------------------
    # PID (0x30~0x32)
    # ------------------------
    def pid_read(self, timeout: float = 0.05):
        """0x30: Read PID."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x30, payload7_bytes, timeout)

    def pid_write_to_ram(self, pid_payload7_bytes: bytes, timeout: float = 0.05):
        """0x31: Write PID to RAM. payload7 must be exactly 7 bytes."""
        if len(pid_payload7_bytes) != 7:
            raise ValueError("pid_payload7_bytes must be exactly 7 bytes (DATA[1..7]).")
        return self._txrx(0x31, pid_payload7_bytes, timeout)

    def pid_write_to_rom(self, pid_payload7_bytes: bytes, timeout: float = 0.05):
        """0x32: Write PID to ROM. payload7 must be exactly 7 bytes."""
        if len(pid_payload7_bytes) != 7:
            raise ValueError("pid_payload7_bytes must be exactly 7 bytes (DATA[1..7]).")
        return self._txrx(0x32, pid_payload7_bytes, timeout)

    # ------------------------
    # Acceleration (0x33~0x34)
    # ------------------------
    def acceleration_read(self, timeout: float = 0.05):
        """0x33: Read acceleration."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x33, payload7_bytes, timeout)

    def acceleration_write_to_ram(self, acceleration_value_int32: int, timeout: float = 0.05):
        """0x34: Write acceleration to RAM. int32 is placed into DATA[4..7]."""
        payload7_bytes = protocol.payload_write_int32_into_data4_to_data7(acceleration_value_int32)
        return self._txrx(0x34, payload7_bytes, timeout)

    # ------------------------
    # Max torque (0x37~0x38)
    # ------------------------
    def max_torque_read(self, timeout: float = 0.05):
        """0x37: Read max torque."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x37, payload7_bytes, timeout)

    def max_torque_write_to_ram(self, max_torque_value_int32: int, timeout: float = 0.05):
        """0x38: Write max torque to RAM. int32 is placed into DATA[4..7]."""
        payload7_bytes = protocol.payload_write_int32_into_data4_to_data7(max_torque_value_int32)
        return self._txrx(0x38, payload7_bytes, timeout)

    # ------------------------
    # Encoder / Zero (0x90~0x91, 0x19)
    # ------------------------
    def encoder_read(self, timeout: float = 0.05):
        """0x90: Read encoder."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x90, payload7_bytes, timeout)

    def zero_point_save_to_rom(self, timeout: float = 0.05):
        """0x91: Save zero point to ROM."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x91, payload7_bytes, timeout)

    def set_current_position_as_zero(self, timeout: float = 0.05):
        """0x19: Set current position as zero point."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x19, payload7_bytes, timeout)

    # ------------------------
    # Error / state reads (0x9A~0x9D)
    # ------------------------
    def error_state1_read(self, timeout: float = 0.05):
        """0x9A: Read error state1. (alias of read_state1)"""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x9A, payload7_bytes, timeout)

    def error_flag_clear(self, timeout: float = 0.05):
        """0x9B: Clear error flags."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x9B, payload7_bytes, timeout)

    def state2_read(self, timeout: float = 0.05):
        """0x9C: Read state2. (alias of read_state2)"""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x9C, payload7_bytes, timeout)

    def state3_read(self, timeout: float = 0.05):
        """0x9D: Read state3."""
        payload7_bytes = protocol.payload_all_zeros_7bytes()
        return self._txrx(0x9D, payload7_bytes, timeout)
