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
        """Read state2 (torque amps, output voltage, degree per second, encoder position)."""
        return self._txrx(0x9C, protocol.E7, timeout)

    def read_multi_turn(self, timeout: float = 0.05):
        '''Read multi-turn angle (int64, 0.001°/LSB).'''
        return self._txrx(0x92, protocol.E7, timeout)

    def read_single_turn(self, timeout: float = 0.05):
        '''Read single-turn angle (uint32, 0.001°/LSB).'''
        return self._txrx(0x94, protocol.E7, timeout)

    def torque_mode_amp(self, amps: float, timeout: float = 0.05):
        """Set torque mode with specified current (amps)."""
        torque_payload = protocol.payload_torque_mode_from_amp(amps)
        return self._txrx(0xA1, torque_payload, timeout)
