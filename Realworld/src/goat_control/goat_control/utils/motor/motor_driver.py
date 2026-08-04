# goat_control/core/comm/motor_driver.py
from __future__ import annotations

import time
from dataclasses import dataclass

from .can import CanInterface
from . import protocol


@dataclass
class MotorParams:
    """Static configuration for a single motor."""
    node_id: int


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

        # 0xA1 (torque) reply may land on rx_id OR tx_id depending on the motor setup.
        self.can_interface.alias_event_keys((self.can_ids.rx_id, 0xA1), (self.can_ids.tx_id, 0xA1))

    # ======================
    # CAN send and wait
    # ======================
    def _txrx(self, command_byte: int, payload7: bytes = protocol.E7, timeout: float = 0.05):
        """Low-level helper: send a command and wait for a response."""
        return self.can_interface.txrx(tx_id=self.can_ids.tx_id,
                                       rx_id=self.can_ids.rx_id,
                                       cmd_byte=command_byte,
                                       payload7=payload7,
                                       timeout=timeout,
                                       accept_rx_id=True,
                                       accept_tx_echo_diff=True)

    # ======================
    # Manager helpers [READ]
    # ======================
    def read_state1(self, timeout: float = 0.05):
        """Read state1 (voltage, current, position) error flags."""
        return self._txrx(0x9A, protocol.E7, timeout)

    def read_state2(self, timeout: float = 0.05):
        """Read state2 (torque/current, output voltage, speed, encoder position).

        Note: angle unit is 0.001°/LSB and speed unit is typically 0.01°/s per LSB (per YAML/manual).
        """
        return self._txrx(0x9C, protocol.E7, timeout)

    def read_multi_turn(self, timeout: float = 0.05):
        '''Read multi-turn angle'''
        return self._txrx(0x92, protocol.E7, timeout)

    def read_single_turn(self, timeout: float = 0.05):
        '''Read single-turn angle'''
        return self._txrx(0x94, protocol.E7, timeout)

    # =======================
    # Manager helpers [WRITE]
    # =======================
    def send_torque_only(self, amps: float) -> None:
        """Send 0xA1 torque command; response will be cached by reader thread."""
        torque_payload = protocol.payload_torque_mode_from_amp(amps)
        self.can_interface.send_only(self.can_ids.tx_id, bytes([0xA1]) + torque_payload)

    def send_state1_request(self) -> None:
        """Send 0x9A state1 (error flags) request; reply cached by reader thread."""
        self.can_interface.send_only(self.can_ids.tx_id, bytes([0x9A]) + protocol.E7)

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

    def clear_state2_event(self) -> None:
        """Arm this motor for a fresh 0xA1 reply.

        Must be called BEFORE send_torque_only() each tick so the subsequent
        wait blocks until THIS tick's reply lands (not the previous one's)."""
        self.can_interface.event_for_key(self.can_ids.rx_id, 0xA1).clear()
        self.can_interface.event_for_key(self.can_ids.tx_id, 0xA1).clear()

    def await_state2(self, deadline_monotonic: float):
        """Block until a fresh 0xA1 reply arrives for this motor, or the
        shared deadline elapses. Returns the can.Message or None on timeout."""
        # rx_id and tx_id 0xA1 events are aliased to one shared Event (see
        # __init__), so this wakes whether the motor answers on rx_id or tx_id.
        ev = self.can_interface.event_for_key(self.can_ids.rx_id, 0xA1)
        remaining = max(0.0, deadline_monotonic - time.monotonic())
        arrive = ev.wait(remaining)  # woken on arrival, or fell through on timeout
        if not arrive:
            return None # Timeout Signal
        # Prefer rx_id frame; fall back to tx_id (this hardware replies on tx_id).
        return (self.can_interface.get_latest_frame(self.can_ids.rx_id, 0xA1)
                or self.can_interface.get_latest_frame(self.can_ids.tx_id, 0xA1))