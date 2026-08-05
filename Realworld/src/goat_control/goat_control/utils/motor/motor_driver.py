# goat_control/core/comm/motor_driver.py
from __future__ import annotations

import time
from dataclasses import dataclass

from .can import CanInterface

# Payload7 = DATA[1..7]
E7 = b"\x00" * 7 

@dataclass
class CanIds:
    """Standard MG-series CAN IDs for a given node id."""
    tx_id: int
    rx_id: int


class MotorDriver:
    """Single motor driver abstraction.

    This class does NOT own the CAN bus.
    It uses:
      - CanInterface for transport (raw TX/RX)
    """

    def __init__(self, can_interface: CanInterface, node_id: int):
        self.can_interface = can_interface
        self.can_ids = CanIds(tx_id=0x140 + node_id, rx_id=0x180 + node_id)

        # 0xA1 (torque) reply may land on rx_id OR tx_id depending on the motor setup.
        self.reply_event = self.can_interface.alias_event_keys((self.can_ids.rx_id, 0xA1), (self.can_ids.tx_id, 0xA1))

    # ========================
    # Initialization helpers
    # ========================
    def _txrx(self, command_byte: int, payload7: bytes = E7, timeout: float = 0.05):
        """Low-level helper: send a command and wait for a response."""
        return self.can_interface.txrx(tx_id=self.can_ids.tx_id,
                                       rx_id=self.can_ids.rx_id,
                                       cmd_byte=command_byte,
                                       payload7=payload7,
                                       timeout=timeout,
                                       accept_rx_id=True,
                                       accept_tx_echo_diff=True)

    def read_state1(self, timeout: float = 0.05):
        """Read state1 (voltage, current, position) error flags."""
        return self._txrx(0x9A, E7, timeout)

    def read_state2(self, timeout: float = 0.05):
        """Read state2 (torque/current, output voltage, speed, encoder position).

        Note: angle unit is 0.001°/LSB and speed unit is typically 0.01°/s per LSB (per YAML/manual).
        """
        return self._txrx(0x9C, E7, timeout)

    def read_multi_turn(self, timeout: float = 0.05):
        '''Read multi-turn angle'''
        return self._txrx(0x92, E7, timeout)

    def read_single_turn(self, timeout: float = 0.05):
        '''Read single-turn angle'''
        return self._txrx(0x94, E7, timeout)

    # =======================
    # Manager helpers [WRITE]
    # =======================
    def send_torque_only(self, amps: float, max_current_lsb: int, motor_current_amp_per_lsb: float) -> None:
        """Send 0xA1 torque command; response will be cached by reader thread."""
        current_lsb = int(round(amps / motor_current_amp_per_lsb))
        current_lsb = max(min(current_lsb, max_current_lsb), -max_current_lsb)
        # Pack signed int16 into 7 bytes (00 00 00 + iq(2B) + 00 00)
        out = b"\x00\x00\x00" + int(current_lsb).to_bytes(2, byteorder="little", signed=True) + b"\x00\x00"
        self.can_interface.send_only(self.can_ids.tx_id, bytes([0xA1]) + out)

    def send_state1_request(self) -> None:
        """Send 0x9A state1 (error flags) request; reply cached by reader thread."""
        self.can_interface.send_only(self.can_ids.tx_id, bytes([0x9A]) + E7)

    # =======================
    # State variables
    # =======================
    def latest_state1(self):
        """Return the most recent cached 0x9A reply for this motor (or None).

        Tries rx_id first, then tx_id — see latest_state2() for rationale.
        """
        msg = self.can_interface.get_latest_frame(self.can_ids.rx_id, 0x9A)
        if msg is not None:
            return msg
        return self.can_interface.get_latest_frame(self.can_ids.tx_id, 0x9A)
    
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

    # =======================
    # External helpers
    # =======================
    def clear_state2_event(self) -> None:
        """Arm this motor for a fresh 0xA1 reply."""
        self.reply_event.clear()

    def await_state2(self, deadline_monotonic: float):
        """Block until a fresh 0xA1 reply arrives for this motor."""
        # rx_id and tx_id 0xA1 events are aliased to one shared Event.
        remaining = max(0.0, deadline_monotonic - time.monotonic())
        arrive = self.reply_event.wait(remaining)  # woken on arrival, or fell through on timeout
        if not arrive:
            return None # Timeout Signal
        # Prefer rx_id frame; fall back to tx_id (this hardware replies on tx_id).
        return (self.can_interface.get_latest_frame(self.can_ids.rx_id, 0xA1)
                or self.can_interface.get_latest_frame(self.can_ids.tx_id, 0xA1))