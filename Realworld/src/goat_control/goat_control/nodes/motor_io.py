# src/goat_control/goat_control/nodes/motor_io.py
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import JointState

from goat_control.utils.motor import CanInterface, MotorDriver
from goat_control.utils.motor.motor_manager import MotorManager


class MotorIO:
    """In-process CAN owner (no longer a ROS2 Node).

    Owns both CAN buses. Instantiated and driven directly by ControllerNode:
    every control tick calls `step(torque_cmd_nm)`, which writes torque to the
    motors and reads back joint state in the same CAN transaction.

    Replaces the old motor_io_node + `/commands` + `/joint_states` topics with a
    direct function call, eliminating the cross-process DDS latency.
    """

    def __init__(self, 
                 cfg: dict, 
                 logger, 
                 can_interface: str = "socketcan", 
                 can_tx_timeout_sec: float = 0.05):
        # Config + logger are owned by ControllerNode and passed in.
        self.cfg = cfg
        self.logger = logger
        # Per-motor response wait used by every CAN txrx.
        self.can_tx_timeout_sec = float(can_tx_timeout_sec)

        self.num_joints = cfg["num_joints"]
        self.joint_names = list(cfg["joint_names"])

        # CAN topology from YAML.
        can_channels = list(cfg["can_channels"])      # e.g. ["can0", "can1"]
        motor_bus_idx = list(cfg["motor_bus_index"])  # bus index per joint
        motor_node_ids = list(cfg["motor_node_ids"])  # CAN node id per joint

        # Length / range sanity (same checks as the old node).
        assert len(motor_bus_idx) == self.num_joints, "motor_bus_index length mismatch"
        assert len(motor_node_ids) == self.num_joints, "motor_node_ids length mismatch"
        assert all(0 <= b < len(can_channels) for b in motor_bus_idx), "motor_bus_index error"

        # Each (bus, node_id) pair must be unique — a duplicate means the YAML
        # was not updated for the dual-bus split.
        seen_bus_node_pairs = set()
        for joint_i, (bus_i, node_id_i) in enumerate(zip(motor_bus_idx, motor_node_ids)):
            pair_key = (int(bus_i), int(node_id_i))
            if pair_key in seen_bus_node_pairs:
                raise RuntimeError(
                    f"MotorIO: duplicate (bus={bus_i}, node_id={node_id_i}) at joint index {joint_i}."
                )
            seen_bus_node_pairs.add(pair_key)

        # Open both CAN buses.
        self.cans: list[CanInterface] = [CanInterface(channel=ch, interface=can_interface) for ch in can_channels]
        for c in self.cans:
            c.open()

        # One driver per joint, bound to its bus.
        self.motor_drivers: list[MotorDriver] = [MotorDriver(self.cans[int(bus_i)], nid) for nid, bus_i in zip(motor_node_ids, motor_bus_idx)]

        # Shared CAN scan / torque conversion helper.
        self.motor_manager = MotorManager(motor_drivers=self.motor_drivers, cfg=cfg)

        # Initial blocking read so the controller has valid state on tick 0.
        states = self.motor_manager.read_initial_motor_state()
        self.latest_joint_state: JointState = self._to_joint_state_msg(states)

        # Switch every bus into background-reader mode.
        for can in self.cans:
            can.start_reader_thread()

        # Initialization Logging
        if self.logger is not None:
            self.logger.info("[MotorIO] initialized — owns both CAN buses (in-process).")
            pi_gain_lines = ["[MotorIO] Motor Specification:"]
            for i, (name, bus_i, node_id) in enumerate(zip(self.joint_names, motor_bus_idx, motor_node_ids)):
                iq_kp, iq_ki = self.motor_manager.motor_pi_gain[i]

                pi_gain_lines.append(f"  {name:<12} can{bus_i} id={node_id:<2} " f"Iq_Kp={iq_kp:<3} Iq_Ki={iq_ki:<3}")

            self.logger.info("\n".join(pi_gain_lines))

    def _to_joint_state_msg(self, states) -> JointState:
        """Pack MotorStatesData into a JointState container (no ROS node needed)."""
        js = JointState()
        js.name = list(self.joint_names)
        js.position = list(states.joint_position_rad)
        js.velocity = list(states.joint_velocity_rad_per_sec)
        js.effort = list(states.joint_effort_like)
        return js

    def read_write_motor(self, torque_cmd_nm: np.ndarray):
        """Write torque to all motors and read back joint state in one CAN pass.

        Slow-poll (0x9A error-flag read) moved off the hot path — driven by a
        ~1 Hz ROS timer via `poll_error_flags_once()` instead. Removing it
        kills the periodic 4-5 ms spike that previously hit every 10th tick.
        """
        # Torque clip -> current conversion
        clipped_torque_cmd = self.motor_manager.torque_clipping(torque_cmd_nm.flatten()) # Joint space torque
        current_cmd_amp = self.motor_manager.torque_to_current(clipped_torque_cmd) # Joint torque -> Motor torque -> Motor current
        motor_states_data = self.motor_manager.write_torques_and_read_states(current_cmd_amp, 
                                                                             timeout=self.can_tx_timeout_sec)
        # Cache for next tick + return to caller.
        self.latest_joint_state = self._to_joint_state_msg(motor_states_data)

        return self.latest_joint_state

    def close(self) -> None:
        """Close both CAN buses on shutdown (stops reader threads first)."""
        for can_interface in getattr(self, "cans", []):
            try:
                can_interface.close()
                print("[Motor] CAN interface closed successfully.")
            except Exception:
                print("[Motor] Error closing CAN interface")
