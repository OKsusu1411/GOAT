from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import copy
import time
import termios
import tty
import threading
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState
from goat_control.nodes.motor_io import MotorIO
from goat_control.nodes.imu_io import ImuIO

class ActuatorTargetTestNode(Node):
    def __init__(self):
        super().__init__("actuator_target_test_node")


        # Parameters by Launch File
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")
        self.declare_parameter("action_timeout_sec", 0.05)
        self.declare_parameter("imu_port", "/dev/ttyUSB0")
        self.declare_parameter("imu_baudrate", 115200)
        self.declare_parameter("imu_timeout", 1.0)

        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time",
                rclpy.Parameter.Type.BOOL,
                False,
            )
        ])

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.yaml_path = str(self.get_parameter("yaml_path").value)
        imu_port = str(self.get_parameter("imu_port").value)
        imu_baudrate = int(self.get_parameter("imu_baudrate").value)
        imu_timeout = float(self.get_parameter("imu_timeout").value)

        # Parameters by Yaml File
        with open(self.yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)
        if not isinstance(self.cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")
        self.cfg["nsc_urdf_path"] = copy.deepcopy(self.urdf_path) # URDF path should be assigned in runtime

        # Checkpoint path
        self.checkpoint_path = copy.deepcopy(self.cfg["policy_checkpoint_path"])

        # Logger
        self.logger = self.get_logger()

        # In-process CAN owner — replaces motor_io + /commands + /joint_states.
        self.motor_io = MotorIO(
            self.cfg,
            self.logger,
            can_tx_timeout_sec=float(self.cfg.get("can_tx_timeout_sec", 0.05)),
        )

        # In-process IMU owner — replaces imu_io_node + /imu subscription.
        self.imu_io = ImuIO(
            self.cfg,
            self.logger,
            imu_port=imu_port,
            imu_baudrate=imu_baudrate,
            imu_timeout=imu_timeout,
        )

        # QoS: latest-sample control. Avoid stale backlog from KEEP_ALL.
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publisher
        self.joint_state_pub = self.create_publisher(JointState,
                                                     "/joint_states",
                                                     qos_profile=qos_profile)

        # Publish IMU for logging (controller now reads in-process; this is the
        # external view, not a feedback path).
        self.imu_state_pub = self.create_publisher(ImuState,
                                                   "/imu",
                                                   qos_profile=qos_profile)

        self.torque_command_pub = self.create_publisher(JointState, 
                                                        "/commands", 
                                                        qos_profile=qos_profile)

        # Messages
        self.now_stamp = self.get_clock().now().to_msg()

        # Manual command
        self.joint_ids = self.cfg["joint_indices"]
        self.wheel_ids = self.cfg["wheel_indices"]
        self.max_torque_per_joint = 2.0
        self.max_torque_per_wheel = 1.0
        self.velocity_increment = 0.5
        self.current_wheel_index = 6
        self.joint_names = self.cfg["joint_names"]
        self.num_joints = len(self.joint_names)
        self.position_command = np.asarray([0.0, 0.0, 0.1745, -0.1745, 0.1745, -0.1745, 0.0, 0.0])
        self.velocity_command = np.zeros(self.num_joints, dtype=np.float32)
        self.leg_test = False
        self.wheel_test = False

        # Controller gain
        self.kp_leg = self.cfg["policy_leg_proportional_gain"]
        self.kd_leg = self.cfg["policy_leg_derivative_gain"]
        self.kp_wheel = self.cfg["policy_wheel_proportional_gain"]

        # Timing — use ROS clock so it works under sim time too.
        self.last_tick_time = time.perf_counter()

        self.logger.info("Main Controller Node started")
        self._print_menu()

        self.tty = open("/dev/tty", "rb+", buffering=0)
        self.settings = termios.tcgetattr(self.tty.fileno())
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

        # Control loop timer
        control_period_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self.control_timer = self.create_timer(control_period_sec, self._control_loop)


    def reset(self) -> None:
        """Reset internal states (controller + safety limiter memory)."""
        self.logger.info(f"Reset internal states.")
        self.velocity_command[:] = 0.0
        self.current_wheel_index = 6
        self.leg_test = False
        self.wheel_test = False

    def leg_control(self, q: np.ndarray , q_dot: np.ndarray, q_ref: np.ndarray) -> None:
        """PD control for torque command."""
        return np.clip(self.kp_leg * (q_ref - q) + self.kd_leg * (-q_dot), -self.max_torque_per_joint, self.max_torque_per_joint)

    def wheel_control(self, q_dot: np.ndarray, q_dot_ref: np.ndarray) -> None:
        """P control for torque command."""
        return np.clip(self.kp_wheel * (q_dot_ref - q_dot), -self.max_torque_per_wheel, self.max_torque_per_wheel)

    # ---------------------------------------------------------------------
    # Callback Functions
    # ---------------------------------------------------------------------
    def _print_menu(self) -> None:
        self.logger.info("===========================================")
        self.logger.info("[Keydown Menu]")
        self.logger.info("'l': Leg position tracking test")
        self.logger.info("'w': Wheel velocity tracking test")
        self.logger.info("'r': Controller reset")
        self.logger.info("'q': Quit")
        self.logger.info("[Command Mode]")
        self.logger.info("===========================================\r")

    def _get_key(self):
        try:
            # NOTE: stdin -> tty
            tty.setraw(self.tty.fileno())
            ch = self.tty.read(1).decode(errors="ignore")
            if ch == "\x1b":
                seq = self.tty.read(2).decode(errors="ignore")
                return {
                    "[A": "UP",
                    "[B": "DOWN",
                    "[C": "RIGHT",
                    "[D": "LEFT",
                }.get(seq, "\x1b")
            return ch
        finally:
            # NOTE: stdin -> tty
            termios.tcsetattr(self.tty.fileno(), termios.TCSADRAIN, self.settings)

    def _keyboard_listener_loop(self):
        """Main loop to monitor keyboard input."""
        while rclpy.ok():
            key = self._get_key()

            if key == "l":
                self.leg_test = True
                self.wheel_test = False
                self.logger.info("Leg position tracking test mode activated.\r")

            elif key == "w":
                self.wheel_test = True
                self.leg_test = False
                self.logger.info("Wheel velocity tracking test mode activated.\r")

            elif key == 'r':
                self.reset()

            elif key == "UP":
                if not self.wheel_test:
                    self.logger.info("Wheel test mode is not activated. Please press 'w' to activate wheel test mode.\r")
                    continue
                velocity_command = self.velocity_command[self.current_wheel_index] + self.velocity_increment   
                self.velocity_command[self.current_wheel_index] = velocity_command
                self.logger.info(f"Joint [{self.current_wheel_index}] {self.joint_names[self.current_wheel_index]} velocity command: {velocity_command:.3f} rad/s\r")

            elif key == "DOWN":
                if not self.wheel_test:
                    self.logger.info("Wheel test mode is not activated. Please press 'w' to activate wheel test mode.\r")
                    continue
                velocity_command = self.velocity_command[self.current_wheel_index] - self.velocity_increment 
                self.velocity_command[self.current_wheel_index] = velocity_command
                self.logger.info(f"Joint [{self.current_wheel_index}] {self.joint_names[self.current_wheel_index]} velocity command: {velocity_command:.3f} rad/s\r")

            elif key == "LEFT":
                if not self.wheel_test:
                    self.logger.info("Wheel test mode is not activated. Please press 'w' to activate wheel test mode.\r")
                    continue
                self.velocity_command[self.current_wheel_index] = 0
                self.current_wheel_index = max(self.current_wheel_index - 1, 6)
                self.logger.info(f"Selected joint: [{self.current_wheel_index}] {self.joint_names[self.current_wheel_index]}\r")

            elif key == "RIGHT":
                if not self.wheel_test:
                    self.logger.info("Wheel test mode is not activated. Please press 'w' to activate wheel test mode.\r")
                    continue                
                self.velocity_command[self.current_wheel_index] = 0
                self.current_wheel_index = min(self.current_wheel_index + 1, 7)
                self.logger.info(f"Selected joint: [{self.current_wheel_index}] {self.joint_names[self.current_wheel_index]}\r")

            elif key == 'q':
                self.logger.info("Shutting down Agent Node...\r")
                self.motor_io.read_write_motor(np.zeros(self.num_joints, dtype=np.float32))
                rclpy.shutdown()
                break
            else:
                self.logger.info("Wrong key! Please enter the right key")
                self._print_menu()

    # ---------------------------------------------------------------------
    # Publish function
    # ---------------------------------------------------------------------
    def _publish(self, position: np.ndarray, velocity: np.ndarray, effort: np.ndarray, joint_state_msg, imu_msg) -> None:
        """Publish joint state, IMU, and torque commands for logging."""
        # Interrupting handling
        if not rclpy.ok():
            return
        # Update joint state message for logging
        msg_joint = JointState()
        msg_joint.header.stamp = self.now_stamp
        msg_joint.header.frame_id = "base_link"
        msg_joint.name = joint_state_msg.name
        msg_joint.position = joint_state_msg.position
        msg_joint.velocity = joint_state_msg.velocity
        msg_joint.effort = joint_state_msg.effort
        # Update IMU message for logging 
        msg_imu = ImuState()
        msg_imu.header.stamp = self.now_stamp
        msg_imu.quat = imu_msg.quat
        msg_imu.gyro = imu_msg.gyro
        msg_imu.vel = imu_msg.vel
        msg_imu.mag = imu_msg.mag
        msg_imu.time_ms = imu_msg.time_ms

        # Update joint command message
        msg_command = JointState()
        msg_command.header.stamp = self.now_stamp
        # Use configured joint names so /commands stays consistent with the
        # actual num_joints (8 in normal setup, 6 in wheel-less bring-up).
        msg_command.name = list(self.cfg["joint_names"])
        msg_command.position = position.tolist()
        msg_command.velocity = velocity.tolist()
        msg_command.effort = effort.tolist()

        # Publish
        self.joint_state_pub.publish(msg_joint)
        self.imu_state_pub.publish(msg_imu)
        self.torque_command_pub.publish(msg_command)


    # ---------------------------------------------------------------------
    # Control Loop
    # ---------------------------------------------------------------------    
    def _control_loop(self):
        """Main control loop called by create_timer at control_rate_hz."""
        self.now_stamp = self.get_clock().now().to_msg()
        now_time = time.perf_counter()

        # Time - Time → Duration; convert to seconds via nanoseconds.
        dt_sec = (now_time - self.last_tick_time)
        if dt_sec <= 0.0:
            dt_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self.last_tick_time = now_time

        # Joint state
        joint_state_msg = self.motor_io.latest_joint_state
        # IMU state
        imu_msg = self.imu_io.read_imu()       

        # Commands
        q_ref = np.zeros(self.num_joints, dtype=np.float32)
        v_ref = np.zeros(self.num_joints, dtype=np.float32)
        tau   = np.zeros(self.num_joints, dtype=np.float32)

        # Current state
        q = np.asarray(joint_state_msg.position, dtype=np.float32)
        q_dot = np.asarray(joint_state_msg.velocity, dtype=np.float32)

        # Control logic
        if self.leg_test:
            q_ref[self.joint_ids] = self.position_command[self.joint_ids]
            tau[self.joint_ids] = self.leg_control(q[self.joint_ids], q_dot[self.joint_ids], q_ref[self.joint_ids])
        if self.wheel_test:
            v_ref[self.wheel_ids] = self.velocity_command[self.wheel_ids]
            tau[self.wheel_ids] = self.wheel_control(q_dot[self.wheel_ids], v_ref[self.wheel_ids])

        # pulbish torque command
        self.motor_io.read_write_motor(tau)   
        self._publish(q_ref, v_ref, tau, joint_state_msg, imu_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorTargetTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hasattr(node, "tty"):
                termios.tcsetattr(node.tty.fileno(), termios.TCSADRAIN, node.settings)
                node.tty.close()
        finally:
            if hasattr(node, "motor_io"):
                node.motor_io.close()
            if hasattr(node, "imu_io"):
                node.imu_io.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == "__main__":
    main()