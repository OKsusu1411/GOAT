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

    # [DEPRECATED] Blocking torque command. Replaced by send_torque_only() +
    # background reader (latest_state2()) on the hot path. Not called anywhere.
    # def torque_mode_amp(self, amps: float, timeout: float = 0.05):
    #     """Set torque mode with specified current (amps)."""
    #     torque_payload = protocol.payload_torque_mode_from_amp(amps)
    #     return self._txrx(0xA1, torque_payload, timeout)

    # ------------------------
    # Fire-and-forget API (Step 3 — TX/RX decouple)
    # Use these on the control hot path. Responses are consumed by the
    # CanInterface background reader thread and pulled from its cache via
    # latest_state2() / latest_state1().
    # ------------------------
    def send_torque_only(self, amps: float) -> None:
        """Send 0xA1 torque command; response will be cached by reader thread."""
        torque_payload = protocol.payload_torque_mode_from_amp(amps)
        self.can_interface.send_only(
            self.can_ids.tx_id,
            bytes([0xA1]) + torque_payload,
        )

    def send_state1_request(self) -> None:
        """Send 0x9A state1 (error flags) request; reply cached by reader thread."""
        self.can_interface.send_only(
            self.can_ids.tx_id,
            bytes([0x9A]) + protocol.E7,
        )

    def latest_state2(self):
        """Return the most recent cached 0xA1 reply for this motor (or None).

        Tries rx_id (0x180+node_id) first, then falls back to tx_id
        (0x140+node_id). Mirrors the OLD txrx()'s `accept_tx_echo_diff=True`
        path — some motor setups echo replies on the tx_id with modified data.
        """
        msg = self.can_interface.get_latest_frame(self.can_ids.rx_id, 0xA1)
        if msg is not None:
            return msg
        return self.can_interface.get_latest_frame(self.can_ids.tx_id, 0xA1)

    def latest_state1(self):
        """Return the most recent cached 0x9A reply for this motor (or None).

        Tries rx_id first, then tx_id — see latest_state2() for rationale.
        """
        msg = self.can_interface.get_latest_frame(self.can_ids.rx_id, 0x9A)
        if msg is not None:
            return msg
        return self.can_interface.get_latest_frame(self.can_ids.tx_id, 0x9A)

    # ============================================================================
    # [DEPRECATED] Additional command wrappers (not used yet). Defined for the
    # full MG-series command set but never called by the live control path.
    # ============================================================================

# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # Motor power / stop / run
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def motor_power_switch(self, timeout: float = 0.05):
# [DEPRECATED]         """0x80: Motor power ON/OFF."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x80, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def motor_stop(self, timeout: float = 0.05):
# [DEPRECATED]         """0x81: Motor stop."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x81, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def motor_run(self, timeout: float = 0.05):
# [DEPRECATED]         """0x88: Motor run/start."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x88, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # Control modes (A0 ~ A8)
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def open_loop_control(self, power_control_int16: int, timeout: float = 0.05):
# [DEPRECATED]         """0xA0: Open-loop control."""
# [DEPRECATED]         payload7_bytes = protocol.payload_open_loop_power_control(power_control_int16)
# [DEPRECATED]         return self._txrx(0xA0, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def speed_closed_loop(
# [DEPRECATED]         self,
# [DEPRECATED]         target_speed_deg_per_sec: float,
# [DEPRECATED]         iq_limit_amps: float = 0.0,
# [DEPRECATED]         timeout: float = 0.05,
# [DEPRECATED]     ):
# [DEPRECATED]         """0xA2: Speed closed-loop."""
# [DEPRECATED]         payload7_bytes = protocol.payload_speed_closed_loop(iq_limit_amps=iq_limit_amps, target_speed_deg_per_sec=target_speed_deg_per_sec)
# [DEPRECATED]         return self._txrx(0xA2, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def position_multi_turn_mode1(self, target_angle_deg: float, timeout: float = 0.05):
# [DEPRECATED]         """0xA3: Multi-turn position mode1."""
# [DEPRECATED]         payload7_bytes = protocol.payload_position_multi_turn_mode1(target_angle_deg=target_angle_deg)
# [DEPRECATED]         return self._txrx(0xA3, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def position_multi_turn_mode2(self, target_angle_deg: float, max_speed_dps_uint16: int, timeout: float = 0.05):
# [DEPRECATED]         """0xA4: Multi-turn position mode2."""
# [DEPRECATED]         payload7_bytes = protocol.payload_position_multi_turn_mode2(
# [DEPRECATED]             max_speed_dps_uint16=max_speed_dps_uint16,
# [DEPRECATED]             target_angle_deg=target_angle_deg,
# [DEPRECATED]         )
# [DEPRECATED]         return self._txrx(0xA4, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def position_single_turn_mode1(self, target_angle_deg: float, spin_direction_uint8: int = 0, timeout: float = 0.05):
# [DEPRECATED]         """0xA5: Single-turn position mode1. spin_direction: 0=CW, 1=CCW"""
# [DEPRECATED]         payload7_bytes = protocol.payload_position_single_turn_mode1(
# [DEPRECATED]             spin_direction_uint8=spin_direction_uint8,
# [DEPRECATED]             target_angle_deg=target_angle_deg,
# [DEPRECATED]         )
# [DEPRECATED]         return self._txrx(0xA5, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def position_single_turn_mode2(
# [DEPRECATED]         self,
# [DEPRECATED]         target_angle_deg: float,
# [DEPRECATED]         max_speed_dps_uint16: int,
# [DEPRECATED]         spin_direction_uint8: int = 0,
# [DEPRECATED]         timeout: float = 0.05,
# [DEPRECATED]     ):
# [DEPRECATED]         """0xA6: Single-turn position mode2."""
# [DEPRECATED]         payload7_bytes = protocol.payload_position_single_turn_mode2(
# [DEPRECATED]             spin_direction_uint8=spin_direction_uint8,
# [DEPRECATED]             max_speed_dps_uint16=max_speed_dps_uint16,
# [DEPRECATED]             target_angle_deg=target_angle_deg,
# [DEPRECATED]         )
# [DEPRECATED]         return self._txrx(0xA6, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def increment_position_mode1(self, delta_angle_deg: float, timeout: float = 0.05):
# [DEPRECATED]         """0xA7: Incremental position mode1."""
# [DEPRECATED]         payload7_bytes = protocol.payload_increment_position_mode1(delta_angle_deg=delta_angle_deg)
# [DEPRECATED]         return self._txrx(0xA7, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def increment_position_mode2(self, delta_angle_deg: float, max_speed_dps_uint16: int, timeout: float = 0.05):
# [DEPRECATED]         """0xA8: Incremental position mode2."""
# [DEPRECATED]         payload7_bytes = protocol.payload_increment_position_mode2(
# [DEPRECATED]             max_speed_dps_uint16=max_speed_dps_uint16,
# [DEPRECATED]             delta_angle_deg=delta_angle_deg,
# [DEPRECATED]         )
# [DEPRECATED]         return self._txrx(0xA8, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # PID (0x30~0x32)
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def pid_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x30: Read PID."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x30, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def pid_write_to_ram(self, pid_payload7_bytes: bytes, timeout: float = 0.05):
# [DEPRECATED]         """0x31: Write PID to RAM. payload7 must be exactly 7 bytes."""
# [DEPRECATED]         if len(pid_payload7_bytes) != 7:
# [DEPRECATED]             raise ValueError("pid_payload7_bytes must be exactly 7 bytes (DATA[1..7]).")
# [DEPRECATED]         return self._txrx(0x31, pid_payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def pid_write_to_rom(self, pid_payload7_bytes: bytes, timeout: float = 0.05):
# [DEPRECATED]         """0x32: Write PID to ROM. payload7 must be exactly 7 bytes."""
# [DEPRECATED]         if len(pid_payload7_bytes) != 7:
# [DEPRECATED]             raise ValueError("pid_payload7_bytes must be exactly 7 bytes (DATA[1..7]).")
# [DEPRECATED]         return self._txrx(0x32, pid_payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # Acceleration (0x33~0x34)
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def acceleration_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x33: Read acceleration."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x33, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def acceleration_write_to_ram(self, acceleration_value_int32: int, timeout: float = 0.05):
# [DEPRECATED]         """0x34: Write acceleration to RAM. int32 is placed into DATA[4..7]."""
# [DEPRECATED]         payload7_bytes = protocol.payload_write_int32_into_data4_to_data7(acceleration_value_int32)
# [DEPRECATED]         return self._txrx(0x34, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # Max torque (0x37~0x38)
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def max_torque_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x37: Read max torque."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x37, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def max_torque_write_to_ram(self, max_torque_value_int32: int, timeout: float = 0.05):
# [DEPRECATED]         """0x38: Write max torque to RAM. int32 is placed into DATA[4..7]."""
# [DEPRECATED]         payload7_bytes = protocol.payload_write_int32_into_data4_to_data7(max_torque_value_int32)
# [DEPRECATED]         return self._txrx(0x38, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # Encoder / Zero (0x90~0x91, 0x19)
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def encoder_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x90: Read encoder."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x90, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def zero_point_save_to_rom(self, timeout: float = 0.05):
# [DEPRECATED]         """0x91: Save zero point to ROM."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x91, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def set_current_position_as_zero(self, timeout: float = 0.05):
# [DEPRECATED]         """0x19: Set current position as zero point."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x19, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     # Error / state reads (0x9A~0x9D)
# [DEPRECATED]     # ------------------------
# [DEPRECATED]     def error_state1_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x9A: Read error state1. (alias of read_state1)"""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x9A, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def error_flag_clear(self, timeout: float = 0.05):
# [DEPRECATED]         """0x9B: Clear error flags."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x9B, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def state2_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x9C: Read state2. (alias of read_state2)"""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x9C, payload7_bytes, timeout)
# [DEPRECATED] 
# [DEPRECATED]     def state3_read(self, timeout: float = 0.05):
# [DEPRECATED]         """0x9D: Read state3."""
# [DEPRECATED]         payload7_bytes = protocol.payload_all_zeros_7bytes()
# [DEPRECATED]         return self._txrx(0x9D, payload7_bytes, timeout)
