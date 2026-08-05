# goat_control/nodes/log_viewer_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import time
import rclpy
import yaml
import csv
from pathlib import Path
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
        self.declare_parameter("csv_path", "experiment_logs.csv")
        self.declare_parameter("log_degrees", False)
        self.declare_parameter("csv", True)

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
        self.gear_ratio = np.array(self.cfg["motor_gear_ratio"], dtype=float)

        self.joint_current: Optional[JointState] = None
        self.joint_ref: Optional[JointState] = None
        self._print_count = 0
        self.command_unit = "Nm"

        # CSV logging
        self.csv_path = str(Path(self.get_parameter("csv_path").value).expanduser().resolve().with_name(f"{time.strftime('%Y%m%d_%H%M%S')}_real_experiment_logs.csv"))
        self.is_csv_logging = bool(self.get_parameter("csv").value)
        self.log_degrees = bool(self.get_parameter("log_degrees").value)
        self.csv_logging_interval_sec = 0.02

        if self.is_csv_logging:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)

            header = ["time_sec"] + [f"{name}_pos_{'deg' if self.log_degrees else 'rad'}" for name in self.joint_names]
            header += [f"{name}_vel_{'deg/s' if self.log_degrees else 'rad/s'}" for name in self.joint_names]
            header += [f"{name}_torque" for name in self.joint_names]

            self.csv_writer.writerow(header)
            self.csv_file.flush()

            self.get_logger().info(f"CSV logging enabled: {self.csv_path}")
        else:
            self.get_logger().info("CSV logging disabled.")
            
        # Subscribers
        self.create_subscription(JointState, "/commands", self._on_joint_ref, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

        # Timer (rate-limit printing)
        period_sec = 1.0 / max(self.print_rate_hz, 0.5)
        self.create_timer(period_sec, self._tick)

        # Logging interval
        self.csv_logging_interval = max(1, int(round(self.csv_logging_interval_sec / period_sec)))

        self.get_logger().info("LogViewerNode started." f" (names of Joints: {self.joint_names}).")

    def _on_joint_ref(self, msg: JointState) -> None:
        self.joint_ref = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_current = msg

    def _tick(self) -> None:
        # No subscription
        if self.joint_current is None:
            return
        
        if self.joint_ref is None:
            self.joint_ref = self.joint_current

        # Decode torque msg
        joint_effort_ref = np.array(self.joint_ref.effort, dtype=float)
        joint_effort_real = np.array(self.joint_current.effort, dtype=float)

        if self.print_degrees:
            joint_pos_current = np.rad2deg(self.joint_current.position)
            joint_vel_current = np.rad2deg(self.joint_current.velocity)
            joint_pos_ref = np.rad2deg(self.joint_ref.position)
            joint_vel_ref = np.rad2deg(self.joint_ref.velocity)
            position_unit = "deg"
            velocity_unit = "deg/s"
        else:
            joint_pos_current = np.array(self.joint_current.position, dtype=float)
            joint_vel_current = np.array(self.joint_current.velocity, dtype=float)
            joint_pos_ref = np.array(self.joint_ref.position, dtype=float)
            joint_vel_ref = np.array(self.joint_ref.velocity, dtype=float)
            position_unit = "rad"
            velocity_unit = "rad/s"

        # Print rows (batch: all joints in ONE log block)
        header_str = (f"{'ID':>3}  {'NAME':<12}  "f"{'POS':>15}  {'VEL':>12}  {'CMD_TAU':>12} {'REAL_TAU':>11}  {'CMD_POS':>12}  {'CMD_VEL':>12}")
        div_str = "-" * len(header_str)
        lines = [header_str, div_str]

        # Print log data
        for joint_index in range(self.num_joints):
            name = self.joint_names[joint_index] if joint_index < len(self.joint_names) else f"joint_{joint_index}"

            # Integrate data (all data is !!motor!! based)
            lines.append(
                f"{joint_index:>3}  {name[:12]:<12}  "
                f"{float(joint_pos_current[joint_index]):>9.{self.precision}f} {position_unit:<5}  "
                f"{float(joint_vel_current[joint_index]):>9.{self.precision}f} {velocity_unit:<5}  "
                f"{float(joint_effort_ref[joint_index]):>9.{self.precision}f} {self.command_unit:<5}  "
                f"{float(joint_effort_real[joint_index]):>9.{self.precision}f} {self.command_unit:<5}  "
                f"{float(joint_pos_ref[joint_index]):>9.{self.precision}f} {position_unit:<5}  "
                f"{float(joint_vel_ref[joint_index]):>9.{self.precision}f} {velocity_unit:<5}"
            )

        # Leading newline: print one line below the logger prefix ([INFO] ...)
        self.get_logger().info("\n" + "\n".join(lines))

        # Save joint position only to CSV
        if self.is_csv_logging and (self._print_count % self.csv_logging_interval == 0):
            if self.log_degrees:
                joint_pos_log = np.rad2deg(np.array(self.joint_current.position, dtype=float))
                joint_vel_log = np.rad2deg(np.array(self.joint_current.velocity, dtype=float))
            else:
                joint_pos_log = np.array(self.joint_current.position, dtype=float)
                joint_vel_log = np.array(self.joint_current.velocity, dtype=float)
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            row  = [now_sec] + [float(joint_pos_log[i]) for i in range(self.num_joints)]
            row += [float(joint_vel_log[i]) for i in range(self.num_joints)]
            row += [float(joint_effort_ref[i]) for i in range(self.num_joints)]

            self.csv_writer.writerow(row)
            self.csv_file.flush()

        # Update count
        self._print_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = LogViewerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if getattr(node, "csv_file", None) is not None and not node.csv_file.closed:
            node.csv_file.flush()
            node.csv_file.close()
            print(f"CSV file '{node.csv_path}' closed.")

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
