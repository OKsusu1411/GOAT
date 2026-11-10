from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import copy
import time
import csv
import termios
import tty
import threading
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState
from message_filters import Subscriber, TimeSynchronizer

class SimFrictionIdtNode(Node):
    def __init__(self):
        super().__init__("sim_friction_id_node")


        # Parameters by Launch File
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")
        self.declare_parameter("joint_id", 0)
        self.declare_parameter("duration", 60.0)
        self.declare_parameter("repeat", 30)

        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, False)])

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.yaml_path = str(self.get_parameter("yaml_path").value)
        self.joint_id = int(self.get_parameter("joint_id").value)
        self.duration = float(self.get_parameter("duration").value)
        self.repeat = int(self.get_parameter("repeat").value)

        # Parameters by Yaml File
        with open(self.yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)
        if not isinstance(self.cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")
        
        self.joint_names = self.cfg["joint_names"]
        self.target_joint_name = self.joint_names[self.joint_id]
        self.num_joints = len(self.joint_names)
        self.is_leg = True if self.joint_id < 6 else False
        self.num_points = int(self.duration * self.control_rate_hz)


        # CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"response_{self.target_joint_name}_{timestamp}.csv"      
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)  

        header =  ["time_sec"] 
        header += [f"{name}_pos_rad" for name in self.joint_names]
        header += [f"{name}_vel_rad/s" for name in self.joint_names]
        header += [f"{name}_actual_torque" for name in self.joint_names] 
        header += [f"{self.target_joint_name}_target_pos_rad"]
        header += [f"{self.target_joint_name}_target_torque"]

        self.csv_writer.writerow(header)
        self.csv_file.flush()

        # Logger
        self.logger = self.get_logger()

        # Subscribers
        sim_qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
                                               durability=rclpy.qos.DurabilityPolicy.VOLATILE, 
                                               history=rclpy.qos.HistoryPolicy.KEEP_ALL)
        self._joint_states_sub = Subscriber(self, JointState, "/joint_states", qos_profile=sim_qos_profile)
        self._imu_sub = Subscriber(self, ImuState, "/imu", qos_profile=sim_qos_profile)

        self.sync = TimeSynchronizer([self._joint_states_sub, self._imu_sub], 10)
        self.sync.registerCallback(self._tick)

        # Publisher
        self.torque_command_pub = self.create_publisher(JointState, 
                                                        "/commands", 
                                                        qos_profile=sim_qos_profile)

        # Messages
        self.now_stamp = self.get_clock().now().to_msg()

        # Manual command
        self.max_torque_per_joint = 2.0
        self.max_velocity_per_joint = 15.0
        self.position_limit = np.asarray(self.cfg["joint_pos_limit"], dtype=np.float32).reshape(-1, 2) * 0.4
        if self.is_leg:
            self.command = self.set_sine_position_reference(self.position_limit[self.joint_id, :])
        else:
            self.command = self.set_sine_velocity_reference()

        # Controller gain
        self.kp_leg = self.cfg["policy_leg_proportional_gain"][0]
        self.kd_leg = self.cfg["policy_leg_derivative_gain"][0]
        self.kp_wheel = self.cfg["policy_wheel_proportional_gain"][0]

        # Timing
        self.prev_torque = np.zeros(self.num_joints, dtype=np.float32)
        self.start_time_sec = None
        self.last_tick_time = time.perf_counter()
        self.count = 0


    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def set_sine_position_reference(self, position_range: np.ndarray):
        lower, upper = position_range

        center = 0.5 * (lower + upper)
        amplitude = 0.5 * (upper - lower)

        initial_phase = np.arcsin(-center / amplitude)
        phase = (2.0 * np.pi * self.repeat * np.arange(self.num_points) / self.num_points + initial_phase)

        return center + amplitude * np.sin(phase)

    def set_sine_velocity_reference(self):
        phase = 2.0 * np.pi * self.repeat * np.arange(self.num_points) / self.num_points
        return float(self.max_velocity_per_joint) * np.sin(phase)

    def stamp_to_sec(self, stamp):
        return (stamp.sec + stamp.nanosec * 1e-9)

    # ---------------------------------------------------------------------
    # Controller
    # ---------------------------------------------------------------------
    def leg_control(self, q: np.ndarray , q_dot: np.ndarray, q_ref: np.ndarray) -> None:
        """PD control for torque command."""
        return np.clip(self.kp_leg * (q_ref - q) + self.kd_leg * (-q_dot), -self.max_torque_per_joint, self.max_torque_per_joint)

    def wheel_control(self, q_dot: np.ndarray, q_dot_ref: np.ndarray) -> None:
        """P control for torque command."""
        return np.clip(self.kp_wheel * (q_dot_ref - q_dot), -self.max_torque_per_joint, self.max_torque_per_joint)

    # ---------------------------------------------------------------------
    # Publish function
    # ---------------------------------------------------------------------
    def _publish_joint_command(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        """Publish command to /joint_command.

        Message semantic:
          position: q_ref
          velocity: v_ref
          effort:   torque
        """
        msg = JointState()
        msg.header.stamp = self.now_stamp

        msg.name = [
            "hip_L_Joint",
            "hip_R_Joint",
            "thigh_L_Joint",
            "thigh_R_Joint",
            "knee_L_Joint",
            "knee_R_Joint",
            "wheel_L_Joint",
            "wheel_R_Joint",
        ]

        msg.position = position.tolist()
        msg.velocity = velocity.tolist()
        msg.effort = torque.tolist()

        self.torque_command_pub.publish(msg)

    # ---------------------------------------------------------------------
    # Control Loop
    # ---------------------------------------------------------------------    
    def _tick(self, joint_state_msg: JointState, imu_msg: ImuState):
        """Main control loop called by create_timer at control_rate_hz."""
        if self.start_time_sec is None:
            self.start_time_sec = self.stamp_to_sec(joint_state_msg.header.stamp)
        self.now_stamp = joint_state_msg.header.stamp
        self.now_sec = self.stamp_to_sec(joint_state_msg.header.stamp)

        elapsed_time = self.now_sec - self.start_time_sec
        if (elapsed_time >= self.duration) or (self.count >= self.num_points):
            self.logger.info("Time Expire.")
            rclpy.shutdown()
            return
        # Commands
        tau = np.zeros(self.num_joints, dtype=np.float32)
        ref_i = self.command[self.count]

        # Current state
        q = np.asarray(joint_state_msg.position, dtype=np.float32)
        q_dot = np.asarray(joint_state_msg.velocity, dtype=np.float32)
        q_tau = np.asarray(joint_state_msg.effort, dtype=np.float32)

        # Control logic
        if self.is_leg:
            tau[self.joint_id] = self.leg_control(q[self.joint_id], q_dot[self.joint_id], ref_i)
        else:
            tau[self.joint_id] = self.wheel_control(q_dot[self.joint_id], ref_i)

        # CSV logging
        row = [self.now_sec]
        row += [q[i] for i in range(self.num_joints)]
        row += [q_dot[i] for i in range(self.num_joints)]
        row += [q_tau[i] for i in range(self.num_joints)]
        row += [ref_i]
        row += [self.prev_torque[self.joint_id]]
        self.csv_writer.writerow(row)

        # Publish
        self._publish_joint_command(np.zeros(self.num_joints), np.zeros(self.num_joints), tau)

        # Update
        self.prev_torque[self.joint_id] = tau[self.joint_id]
        self.count += 1



def main(args=None):
    rclpy.init(args=args)
    node = SimFrictionIdtNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.logger.info(f"CSV saved : {node.csv_path}")
        node.csv_file.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()