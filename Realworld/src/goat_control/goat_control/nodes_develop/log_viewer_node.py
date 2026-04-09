# goat_control/nodes/log_viewer_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import JointState

@dataclass
class LatestLog:
    vector: Optional[np.ndarray] = None


class LogViewerNode(Node):
    """
    Subscribe:  goat/torque_log (Float32MultiArray)
      - data (supported layouts):
          (B) [q(rad) xN, dq(rad/s) xN, u(cmd) xN, ref xN]               => length = 4 * N

        where:
          - q: measured joint angle
          - dq: measured joint velocity
          - u: command value (usually torque [Nm] unless you log current)
          - ref: reference (by convention: position ref for joints, speed ref for wheels)

    Optional Subscribe: joint_states (JointState) to get joint names automatically.
    """

    def __init__(self):
        super().__init__("log_viewer_node")

        # Parameters
        self.declare_parameter("yaml_path", "src/goat_control/config/goat_config.yaml")
        self.declare_parameter("sample_count", 20)

        # YAML file
        yaml_path = str(self.get_parameter("yaml_path").value)

        with open(yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)

        self.declare_parameter("print_rate_hz", 50.0)
        self.declare_parameter("print_degrees", True)
        self.declare_parameter("precision", 3)

        self.print_rate_hz = float(self.get_parameter("print_rate_hz").value)
        self.print_degrees = bool(self.get_parameter("print_degrees").value)
        self.precision = int(self.get_parameter("precision").value)
        
        # YAML parameters
        self.num_joints = self.cfg["num_joints"]
        self.joint_names = self.cfg["joint_names"]
        self.wheel_indices = self.cfg["wheel_indices"]
        self.gear_ratio = self.cfg["motor_gear_ratio"]

        self.joint_current: Optional[JointState] = None
        self.joint_ref: Optional[JointState] = None
        self._print_count = 0
        self.command_unit = "Nm"

        # Subscribers
        self.create_subscription(JointState, "/commands", self._on_joint_ref, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        # self.command_subscriber = Subscriber(self, JointState, "/commands", 10)
        # self.state_subscriber = Subscriber(self, JointState, "/joint_states", 10)
        # self.time_sync = ApproximateTimeSynchronizer([self.command_subscriber, self.state_subscriber], 10, 0.01)
        # self.time_sync.registerCallback(self._sync_callback)

        # Timer (rate-limit printing)
        period_sec = 1.0 / max(self.print_rate_hz, 0.5)
        self.create_timer(period_sec, self._tick)

        self.get_logger().info(
            "LogViewerNOde started."
            f"(names of Joints: {self.joint_names})."
        )

    def _on_joint_ref(self, msg: JointState) -> None:
        self.joint_ref = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_current = msg

    def _sync_callback(self, msg_ref, msg_current):
        self._on_joint_ref(msg_ref)
        self._on_joint_state(msg_current)

    def _tick(self) -> None:
        # No subscription
        if self.joint_current is None:
            return
        
        if self.joint_ref is None:
            self.joint_ref = self.joint_current

        # Decode JointState msg
        if self.print_degrees:
            joint_pos_current = np.rad2deg(self.joint_current.position)
            joint_vel_current = np.rad2deg(self.joint_current.velocity)
            joint_pos_ref = np.rad2deg(self.joint_ref.position)
            joint_vel_ref = np.rad2deg(self.joint_ref.velocity)
            position_unit = "deg"
            velocity_unit = "deg/s"
        else:
            joint_pos_current = self.joint_current.position
            joint_vel_current = self.joint_current.velocity
            joint_pos_ref = self.joint_ref.position
            joint_vel_ref = self.joint_ref.velocity
            position_unit = "rad"
            velocity_unit = "rad/s"

        joint_effort_ref = self.joint_ref.effort
        joint_effort_current = self.joint_current.effort

        # Gear ratio
        motor_pos_current = joint_pos_current * self.gear_ratio
        motor_vel_current = joint_vel_current * self.gear_ratio
        motor_effort_current = joint_effort_current * self.gear_ratio
        motor_pos_ref = joint_pos_ref * self.gear_ratio
        motor_vel_ref = joint_vel_ref * self.gear_ratio
        motor_effort_ref = joint_effort_ref * self.gear_ratio

        # Print rows (batch: all joints in ONE log block)
        header_str = (
            f"{'ID':>3}  {'NAME':<12}  "
            f"{'POS':>15}  {'VEL':>12}  {'CMD_TAU':>12}  {'CMD_POS':>12}  {'CMD_VEL':>12}"
        )
        div_str = "-" * len(header_str)
        lines = [header_str, div_str]

        # Print log data
        for joint_index in range(self.num_joints):
            name = self.joint_names[joint_index] if joint_index < len(self.joint_names) else f"joint_{joint_index}"

            # Integrate data (all data is !!motor!! based)
            lines.append(
                f"{joint_index:>3}  {name[:12]:<12}  "
                f"{float(motor_pos_current[joint_index]):>9.{self.precision}f} {position_unit:<5}  "
                f"{float(motor_vel_current[joint_index]):>9.{self.precision}f} {velocity_unit:<5}  "
                f"{float(motor_effort_ref[joint_index]):>9.{self.precision}f} {self.command_unit:<5}  "
                f"{float(motor_pos_ref[joint_index]):>9.{self.precision}f} {position_unit:<5}  "
                f"{float(motor_vel_ref[joint_index]):>9.{self.precision}f} {velocity_unit:<5}"
            )

        # Leading newline: print one line below the logger prefix ([INFO] ...)
        self.get_logger().info("\n" + "\n".join(lines))
        self._print_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = LogViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
