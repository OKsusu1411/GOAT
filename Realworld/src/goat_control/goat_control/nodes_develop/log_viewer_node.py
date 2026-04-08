# goat_control/nodes/log_viewer_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
# TODO: 이거 마무리해야됨
# TODO: gear 비 다시 되돌려서 logging
# TODO: Ref position, vel 모두 logging
# TODO: ApproximateTimeSynchronizer 이거 적용

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

        # Params
        # self.declare_parameter("log_topic", "/torque")
        # self.declare_parameter("joint_state_topic", "/joint_states")
        # self.declare_parameter("use_joint_state_names", True)

        self.declare_parameter("num_joints", 8)
        self.declare_parameter(
            "joint_names",
            ["hip_L",
             "hip_R",
             "thigh_L",
             "thigh_R",
             "knee_L", 
             "knee_R",
             "wheel_L",
             "wheel_R"]
        )

        # For nicer ref printing (ref is position for joints, speed for wheels)
        self.declare_parameter("wheel_indices", [6, 7])

        self.declare_parameter("print_rate_hz", 50.0)
        self.declare_parameter("print_degrees", True)
        self.declare_parameter("command_unit", "torque_nm")  # torque_nm or amp
        self.declare_parameter("precision", 3)

        # self.log_topic = str(self.get_parameter("log_topic").value)
        # self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        # self.use_joint_state_names = bool(self.get_parameter("use_joint_state_names").value)

        self.num_joints = int(self.get_parameter("num_joints").value)
        self.joint_names: List[str] = [str(x) for x in self.get_parameter("joint_names").value]
        self.wheel_indices = [int(x) for x in self.get_parameter("wheel_indices").value]

        self.print_rate_hz = float(self.get_parameter("print_rate_hz").value)
        self.print_degrees = bool(self.get_parameter("print_degrees").value)
        self.command_unit = str(self.get_parameter("command_unit").value)
        self.precision = int(self.get_parameter("precision").value)

        if len(self.joint_names) != self.num_joints:
            self.get_logger().warn(
                f"joint_names length ({len(self.joint_names)}) != num_joints ({self.num_joints}). "
                "Falling back to generic names."
            )
            self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]

        self.latest = LatestLog()
        self._print_count = 0

        # Subscribers
        self.create_subscription(Float32MultiArray, "/torque", self._on_log, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription()

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

        # Layout parsing (ONLY accept 4N, 3N support removed)
        expected_len = 5 * self.num_joints
        if vector.size != expected_len:
            self.get_logger().warn(
                f"log length mismatch: got {vector.size}, expected {expected_len}"
            )
            return

        # Data slicing
        joint_position_rad = vector[0 : self.num_joints]
        joint_velocity_rad_per_sec = vector[self.num_joints : 2 * self.num_joints]
        command_value = vector[2 * self.num_joints : 3 * self.num_joints]
        safe_joint_targets = vector[3 * self.num_joints : 4 * self.num_joints]
        ref_vector = vector[4 * self.num_joints : 5 * self.num_joints]

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

        # Print rows (batch: all joints in ONE log block)
        header_str = (
            f"{'ID':>3}  {'NAME':<12}  "
            f"{'POS':>15}  {'VEL':>15}  {'CMD':>15}  {'SAFE':>15}  {'REF':>15}"
        )
        div_str = "-" * len(header_str)

        lines = [header_str, div_str]

        # Print log data
        for joint_index in range(self.num_joints):
            name = self.joint_names[joint_index] if joint_index < len(self.joint_names) else f"joint_{joint_index}"

            raw_ref = float(ref_vector[joint_index])
            raw_safe = float(safe_joint_targets[joint_index])
            if joint_index in self.wheel_indices:
                # Wheel: Reference is Speed
                ref_val = float(np.rad2deg(raw_ref)) if self.print_degrees else raw_ref
                ref_safe = float(np.rad2deg(raw_safe)) if self.print_degrees else raw_safe
                ref_unit = velocity_unit
            else:
                # Joint: Reference is Position
                ref_val = float(np.rad2deg(raw_ref)) if self.print_degrees else raw_ref
                ref_safe = float(np.rad2deg(raw_safe)) if self.print_degrees else raw_safe
                ref_unit = position_unit

            # Integrate data
            lines.append(
                f"{joint_index:>3}  {name[:12]:<12}  "
                f"{float(joint_position[joint_index]):>9.{self.precision}f} {position_unit:<5}  "
                f"{float(joint_velocity[joint_index]):>9.{self.precision}f} {velocity_unit:<5}  "
                f"{float(command_value[joint_index]):>9.{self.precision}f} {command_unit:<5}  "
                f"{float(ref_safe):>9.{self.precision}f} {ref_unit:<5}  "
                f"{float(ref_val):>9.{self.precision}f} {ref_unit:<5}"
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
