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
from message_filters import Subscriber, TimeSynchronizer

class SimActuatorTargetTestNode(Node):
    def __init__(self):
        super().__init__("sim_actuator_target_test_node")


        # Parameters by Launch File
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")

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

        # QoS: latest-sample control. Avoid stale backlog from KEEP_ALL.
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )


        # Subscribers
        self._joint_states_sub = Subscriber(self, JointState, "/joint_states", qos_profile=sim_qos_profile)
        self._imu_sub = Subscriber(self, ImuState, "/imu", qos_profile=sim_qos_profile)

        self.sync = TimeSynchronizer([self._joint_states_sub, self._imu_sub], 10)
        self.sync.registerCallback(self._tick)


        self.torque_command_pub = self.create_publisher(JointState, 
                                                        "/commands", 
                                                        qos_profile=sim_qos_profile)

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
        self.position_command = np.asarray([0.0, 0.0, 0.5235, -0.5235, 0.5235, -0.5235, 0.0, 0.0])
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
        self.logger.info("'w': Wheel position tracking test")
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
                self._publish_joint_command(np.zeros(self.num_joints), np.zeros(self.num_joints), np.zeros(self.num_joints))
                rclpy.shutdown()
                break

            elif key == "\x03":  # Ctrl+C
                rclpy.shutdown()
                break
            else:
                self.logger.info("Wrong key! Please enter the right key")
                self._print_menu()


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
        self.now_stamp = joint_state_msg.header.stamp
        now_time = time.perf_counter()

        # Time - Time → Duration; convert to seconds via nanoseconds.
        dt_sec = (now_time - self.last_tick_time)
        if dt_sec <= 0.0:
            dt_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self.last_tick_time = now_time   

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
        self._publish_joint_command(q_ref, v_ref, tau)


def main(args=None):
    rclpy.init(args=args)
    node = SimActuatorTargetTestNode()

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
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()