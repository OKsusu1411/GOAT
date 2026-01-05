# goat_control/nodes/log_viewer_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState


@dataclass
class LatestLog:
    vector: Optional[np.ndarray] = None


class MotorTorqueLogViewer(Node):
    """
    Subscribe:  motor_torque_log (Float32MultiArray)
      - data = [q(rad) ... , dq(rad/s) ... , u(cmd) ...]
      - length = 3 * num_joints

    Optional Subscribe: joint_states (JointState) to get joint names automatically.
    """

    def __init__(self):
        super().__init__("motor_torque_log_viewer")

        # Params
        self.declare_parameter("log_topic", "motor_torque_log")
        self.declare_parameter("joint_state_topic", "joint_states")
        self.declare_parameter("use_joint_state_names", True)

        self.declare_parameter("num_joints", 8)
        self.declare_parameter(
            "joint_names",
            ["hip_L", "hip_R", "thigh_L", "thigh_R", "knee_L", "knee_R", "wheel_L", "wheel_R"]
        )

        self.declare_parameter("print_rate_hz", 200.0)
        self.declare_parameter("print_degrees", False)
        self.declare_parameter("command_unit", "torque_nm")  # torque_nm or amp
        self.declare_parameter("precision", 3)
        self.declare_parameter("header_every", 20)

        self.log_topic = str(self.get_parameter("log_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.use_joint_state_names = bool(self.get_parameter("use_joint_state_names").value)

        self.num_joints = int(self.get_parameter("num_joints").value)
        self.joint_names: List[str] = [str(x) for x in self.get_parameter("joint_names").value]

        self.print_rate_hz = float(self.get_parameter("print_rate_hz").value)
        self.print_degrees = bool(self.get_parameter("print_degrees").value)
        self.command_unit = str(self.get_parameter("command_unit").value)
        self.precision = int(self.get_parameter("precision").value)
        self.header_every = int(self.get_parameter("header_every").value)

        if len(self.joint_names) != self.num_joints:
            self.get_logger().warn(
                f"joint_names length ({len(self.joint_names)}) != num_joints ({self.num_joints}). "
                "Falling back to generic names."
            )
            self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]

        self.latest = LatestLog()
        self._print_count = 0

        # Subs
        self.create_subscription(Float32MultiArray, self.log_topic, self._on_log, 10)

        if self.use_joint_state_names:
            self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)

        # Timer (rate-limit printing)
        period_sec = 1.0 / max(self.print_rate_hz, 0.5)
        self.create_timer(period_sec, self._tick)

        self.get_logger().info(
            f"MotorTorqueLogViewer started. Subscribing '{self.log_topic}' "
            f"(names from JointState: {self.use_joint_state_names})."
        )

    def _on_log(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=float).flatten()
        self.latest.vector = data

    def _on_joint_state(self, msg: JointState) -> None:
        # Use incoming names if they look valid and match num_joints
        if msg.name and len(msg.name) == self.num_joints:
            self.joint_names = list(msg.name)

    def _tick(self) -> None:
        if self.latest.vector is None:
            return

        vector = self.latest.vector
        expected = 3 * self.num_joints
        if vector.size != expected:
            self.get_logger().warn(f"log length mismatch: got {vector.size}, expected {expected}")
            return

        joint_position_rad = vector[0 : self.num_joints]
        joint_velocity_rad_per_sec = vector[self.num_joints : 2 * self.num_joints]
        command_value = vector[2 * self.num_joints : 3 * self.num_joints]

        if self.print_degrees:
            joint_position = np.rad2deg(joint_position_rad)
            joint_velocity = np.rad2deg(joint_velocity_rad_per_sec)
            position_unit = "deg"
            velocity_unit = "deg/s"
        else:
            joint_position = joint_position_rad
            joint_velocity = joint_velocity_rad_per_sec
            position_unit = "rad"
            velocity_unit = "rad/s"

        command_unit = "Nm" if self.command_unit == "torque_nm" else "A"

        # Print header periodically
        if (self._print_count % max(self.header_every, 1)) == 0:
            header = (
                f"{'idx':>3}  {'name':<12}  "
                f"{('q['+position_unit+']'):>12}  {('dq['+velocity_unit+']'):>12}  {('u['+command_unit+']'):>12}"
            )
            self.get_logger().info(header)
            self.get_logger().info("-" * len(header))
            
        # Print rows (batch: print all joints in one log message)
        fmt = f"{{:>3}}  {{:<12}}  {{:>12.{self.precision}f}}  {{:>12.{self.precision}f}}  {{:>12.{self.precision}f}}"

        lines = []
        for joint_index in range(self.num_joints):
            name = self.joint_names[joint_index] if joint_index < len(self.joint_names) else f"joint_{joint_index}"
            lines.append(
                fmt.format(
                    joint_index,
                    name[:12],
                    float(joint_position[joint_index]),
                    float(joint_velocity[joint_index]),
                    float(command_value[joint_index]),
                )
            )

        self.get_logger().info("\n" + "\n".join(lines))


        self._print_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = MotorTorqueLogViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
