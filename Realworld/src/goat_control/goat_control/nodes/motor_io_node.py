# src/goat_control/goat_control/nodes/motor_io_node.py
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

from goat_control.utils.motor import CanInterface, MotorDriver, MotorParams
from goat_control.utils.motor.motor_manager import MotorManager


class MotorIONode(Node):
    """Single CAN owner node for real hardware.

    - Reads motor states from CAN and publishes `joint_states`.
    - Subscribes `goat/torque_commands` (Nm per joint) and sends motor current commands.

    This replaces the previous pattern:
      StateEstimationNode (CAN read) + MotorCommandNode (CAN write)
    which opened the CAN device twice.
    """

    def __init__(self):
        super().__init__("motor_io_node")

        # Parameters
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("motor_node_ids", [1, 2, 3, 4, 5, 6, 7, 8])
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("io_rate_hz", 200.0)

        # Safety / timing
        self.declare_parameter("command_timeout_sec", 0.1)
        self.declare_parameter("zero_on_timeout", True)
        self.declare_parameter("can_tx_timeout_sec", 0.05)

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        # motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        yaml_path = str(self.get_parameter("yaml_path").value)

        io_rate_hz = float(self.get_parameter("io_rate_hz").value)

        self.command_timeout_sec = float(self.get_parameter("command_timeout_sec").value)
        self.zero_on_timeout = bool(self.get_parameter("zero_on_timeout").value)
        self.can_tx_timeout_sec = float(self.get_parameter("can_tx_timeout_sec").value)

        # YAML file
        with open(yaml_path, 'r', encoding='utf-8') as file_handle:
            self.cfg = yaml.safe_load(file_handle)
        self.num_joints = self.cfg["num_joints"]
        self.joint_names = self.cfg["joint_names"]


        # CAN

        can_channels   = list(self.cfg["can_channels"])           # ["can0", "can1"]
        motor_bus_idx  = list(self.cfg["motor_bus_index"])
        motor_node_ids = list(self.cfg["motor_node_ids"])

        # Error detection
        assert len(motor_bus_idx)  == self.num_joints, "motor_bus_index length mismatch"
        assert len(motor_node_ids) == self.num_joints, "motor_node_ids length mismatch"
        assert all(0 <= b < len(can_channels) for b in motor_bus_idx), "motor_bus_index error"

        # Can interface
        self.cans: list[CanInterface] = [
            CanInterface(channel=ch, interface=can_interface) for ch in can_channels
        ]
        for c in self.cans:
            c.open()

        # Motor driver
        self.motor_drivers: list[MotorDriver] = []
        for nid, bus_i in zip(motor_node_ids, motor_bus_idx):
            self.motor_drivers.append(
                MotorDriver(self.cans[int(bus_i)], MotorParams(node_id=int(nid)))
            )

        # Debugging log
        mapping_str = ", ".join(
            f"{name}=can{b}#id{nid}"
            for name, b, nid in zip(self.joint_names, motor_bus_idx, motor_node_ids)
        )
        self.get_logger().info(f"motor bus map: {mapping_str}")



        # self.can = CanInterface(channel=can_channel, interface=can_interface)
        # self.can.open()

        # self.motor_drivers: list[MotorDriver] = []
        # for node_id in motor_node_ids:
        #     params = MotorParams(node_id=int(node_id))
        #     self.motor_drivers.append(MotorDriver(self.can, params))

        # if len(self.motor_drivers) != self.num_joints:
        #     self.get_logger().warn(
        #         f"motor_node_ids length ({len(self.motor_drivers)}) != YAML num_joints ({self.num_joints}). "
        #         "State vector sizes may mismatch. Ensure YAML joint_names matches motor count."
        #     )
        #     # still proceed; we use driver count as the truth for CAN IO
        #     self.num_joints = len(self.motor_drivers)


        self.motor_manager = MotorManager(
            motor_drivers=self.motor_drivers,
            cfg=self.cfg
        )

        # Latest command buffer
        self._latest_torque_cmd: Optional[np.ndarray] = None
        self._latest_torque_cmd_time_sec: float = 0.0
        
        # ROS pubs/subs
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.torque_sub = self.create_subscription(JointState, "/commands", self._on_command, 10)

        # IO loop
        period_sec = 1.0 / max(io_rate_hz, 1.0)
        self._timer = self.create_timer(period_sec, self._tick)
        self.poll_counter = 0

        self.get_logger().info(
            "MotorIONode started. Owns CAN (read+write). "
            f"Publishing '/joint_states', subscribing '/commands'."
        )

    def _on_command(self, msg: JointState) -> None:
        # Receive torque command
        torque = np.asarray(msg.effort, dtype=float).flatten()
        if torque.size != self.num_joints:
            self.get_logger().warn(
                f"Torque command size mismatch: got {torque.size}, expected {self.num_joints}. Ignored."
            )
            return

        self._latest_torque_cmd = torque
        self._latest_torque_cmd_time_sec = time.time()

    def _tick(self) -> None:
        # Read motors
        torque_cmd = self._latest_torque_cmd

        if torque_cmd is None:
            return

        age = time.time() - self._latest_torque_cmd_time_sec
        if age > self.command_timeout_sec:
            if not self.zero_on_timeout:
                return
            torque_cmd = np.zeros(self.num_joints, dtype=float)
        
        # Torque clipping
        clipped_torque_cmd = self.motor_manager.torque_clipping(np.asarray(torque_cmd, dtype=float))
        
        # Torque LPF
        lpf_torque_cmd = self.motor_manager.torque_lpf(np.asarray(clipped_torque_cmd, dtype=float))

        # Convert torque into current
        current_cmd_amp = self.motor_manager.torque_to_current(np.asarray(lpf_torque_cmd, dtype=float))

        # Send torque command to motor
        # for motor_index, motor_driver in enumerate(self.motor_drivers):
        #     command_amp = float(current_cmd_amp[motor_index])
        #     motor_driver.torque_mode_amp(command_amp, timeout=self.can_tx_timeout_sec)
        
        t1_command = time.monotonic()
        do_slow_poll = (self.poll_counter % 10 == 0)
        motor_states_data = self.motor_manager.write_torques_and_read_states(
            current_cmd_amp, 
            timeout=self.can_tx_timeout_sec, 
            perform_slow_poll=do_slow_poll
        )
        t2_command = time.monotonic()
        
        # Publish JointState
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.joint_names)
        js.position = list(motor_states_data.joint_position_rad)
        js.velocity = list(motor_states_data.joint_velocity_rad_per_sec)
        js.effort = list(motor_states_data.joint_effort_like)

        self.joint_state_pub.publish(js)
        
        dt_command = t2_command - t1_command

        print(f"dt_command : {dt_command:.4f}")

        self.poll_counter += 1

    def destroy_node(self):
        try:
            self.can.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorIONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
