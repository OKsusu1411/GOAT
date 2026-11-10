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
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import JointState
from motor_interfaces.msg import States

from ament_index_python.packages import get_package_share_directory

@dataclass
class LatestLog:
    vector: Optional[np.ndarray] = None


class SimLogViewerNode(Node):
    """
    Subscribe:  goat/torque_log (States)
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
        super().__init__("sim_log_viewer_node")

        # Parameters
        default_yaml_path = str(Path(get_package_share_directory("goat_control")) / "config" / "goat_config.yaml")
        self.declare_parameter("yaml_path", default_yaml_path)
        self.declare_parameter("csv_path", "experiment_logs.csv")
        self.declare_parameter("log_degrees", False)
        self.declare_parameter("csv", True)

        # YAML file
        yaml_path = str(self.get_parameter("yaml_path").value)

        with open(yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)

        self.declare_parameter("print_rate_hz", 100.0)
        self.declare_parameter("print_degrees", True)
        self.declare_parameter("precision", 3)

        self.print_rate_hz = float(self.get_parameter("print_rate_hz").value)
        self.print_degrees = bool(self.get_parameter("print_degrees").value)
        self.precision = int(self.get_parameter("precision").value)
        self.start_time = None
        self.log_start = False
        self.log_interval = 2 # 100hz
        self._print_count = 0
        
        # YAML parameters
        self.num_joints = self.cfg["num_joints"]
        self.joint_names = self.cfg["joint_names"]
        self.wheel_indices = self.cfg["wheel_indices"]
        self.gear_ratio = np.array(self.cfg["motor_gear_ratio"], dtype=float)

        self.joint_current: Optional[JointState] = None
        self.joint_ref: Optional[JointState] = None
        self.obs: Optional[States] = None
        self.command_unit = "Nm"

        # CSV logging
        self.csv_path = str(Path(self.get_parameter("csv_path").value).expanduser().resolve().with_name(f"{time.strftime('%Y%m%d_%H%M%S')}_sim_experiment_logs.csv"))
        self.is_csv_logging = bool(self.get_parameter("csv").value)
        self.log_degrees = bool(self.get_parameter("log_degrees").value)

        if self.is_csv_logging:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)

            header = ["time_sec"] + [f"{name}_pos_{'deg' if self.log_degrees else 'rad'}" for name in self.joint_names]
            header += [f"{name}_vel_{'deg/s' if self.log_degrees else 'rad/s'}" for name in self.joint_names]
            header += [f"{name}_torque" for name in self.joint_names]
            header += [f"observation_{i}" for i in range(26)]

            self.csv_writer.writerow(header)
            self.csv_file.flush()


        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        # Subscribers
        self._joint_states_sub = Subscriber(self, JointState, "/joint_states", qos_profile=sim_qos_profile)
        self._joint_command_sub = Subscriber(self, JointState, "/commands", qos_profile=sim_qos_profile)
        self._obs_sub = Subscriber(self, States, "/obs", qos_profile=sim_qos_profile)
        self.sync = TimeSynchronizer([self._joint_states_sub, self._joint_command_sub, self._obs_sub], 10)
        self.sync.registerCallback(self._tick)


    def _write_csv(self, joint_state_msg: JointState, joint_command_msg: JointState, obs_msg: States) -> None:
        if self.log_degrees:
            joint_pos_log = np.rad2deg(np.asarray(joint_state_msg.position, dtype=float))
            joint_vel_log = np.rad2deg(np.asarray(joint_state_msg.velocity, dtype=float))
        else:
            joint_pos_log = np.asarray(joint_state_msg.position, dtype=float)
            joint_vel_log = np.asarray(joint_state_msg.velocity, dtype=float)

        joint_effort_ref = np.asarray(joint_command_msg.effort, dtype=float)

        obs = obs_msg.data

        stamp_ns = (joint_state_msg.header.stamp.sec * 1_000_000_000 + joint_state_msg.header.stamp.nanosec)

        if not self.log_start:
            if np.any(np.abs(joint_effort_ref) > 0.1):
                self.log_start = True
                self.start_time = stamp_ns
            else:
                return

        now_sec = (stamp_ns - self.start_time) * 1e-9

        row = [now_sec]
        row += [float(joint_pos_log[i]) for i in range(self.num_joints)]
        row += [float(joint_vel_log[i]) for i in range(self.num_joints)]
        row += [float(joint_effort_ref[i]) for i in range(self.num_joints)]
        row += [float(obs[i]) for i in range(26)]

        self.csv_writer.writerow(row)


    def _tick(self, joint_state_msg: JointState, joint_command_msg: JointState, obs_msg: States) -> None:
        self.joint_current = joint_state_msg
        self.joint_ref = joint_command_msg
        self.obs = obs_msg

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

        if self.is_csv_logging and (self._print_count % self.log_interval == 0):
            self._write_csv(joint_state_msg, joint_command_msg, obs_msg)

        self._print_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = SimLogViewerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if getattr(node, "csv_file", None) is not None and not node.csv_file.closed:
            node.csv_file.flush()
            node.csv_file.close()
            print(f"CSV file '{node.csv_path}' closed.")

            # Check csv file validity
            with open(node.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                # Call header and first low data
                _ = next(reader, None)
                first_data = next(reader, None) 
            # Delete empty csv file
            if first_data is None:
                Path(node.csv_path).unlink()
                print(f"Delete CSV file because it is empty.")

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
