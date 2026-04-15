# controller_node.py — ROS2 ControllerNode (main control loop)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
import copy
import sys
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

from goat_control.utils.controller.nominal_controller import NominalController
from goat_control.utils.controller.policy_controller import PolicyController
from goat_control.utils.controller.safety_limiter import SafetyLimiter

@dataclass
class LatestBuffers:
    """Thread-safe buffers for incoming messages."""
    joint_state_msg: Optional[JointState] = None
    imu_msg: Optional[ImuState] = None


class ControllerNode(Node):
    """ROS2 control node: sensor reception -> controller selection -> torque publishing.

    Flow:
      1) Receive JointState + ImuState via time-synced subscribers
      2) Keyboard selects active controller (policy / nominal)
      3) Active controller computes raw torque
      4) SafetyLimiter applies LPF + clipping + kill switch
      5) Publish safe torque command
    """
    def __init__(self):
        super().__init__("controller_node")

        # Parameters by Launch File
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("action_timeout_sec", 0.05)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.yaml_path = str(self.get_parameter("yaml_path").value)
        self.checkpoint_path = self.get_parameter("checkpoint_path").value or None
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)

        # Parameters by Yaml File
        with open(self.yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)
        if not isinstance(self.cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")
        self.cfg["nsc_urdf_path"] = copy.deepcopy(self.urdf_path) # URDF path should be assigned in runtime

        if self.checkpoint_path is not None:
            self.cfg["policy_checkpoint_path"] = copy.deepcopy(self.checkpoint_path) # Default is None

        # Logger
        self.logger = self.get_logger()

        # Controller
        self.nominal_controller = NominalController(self.cfg, self.logger)
        self.policy_controller = PolicyController(self.cfg, self.logger)
        self.safety_limiter = SafetyLimiter(self.cfg, self.logger)

        # Subscriber
        self.joint_state_subscriber = Subscriber(self, JointState, '/joint_states', 10)
        self.imu_subscriber = Subscriber(self, ImuState, '/imu', 10)
        self.time_sync = ApproximateTimeSynchronizer([self.joint_state_subscriber, self.imu_subscriber], 10, 0.01)
        self.time_sync.registerCallback(self.sync_callback)

        # Publisher
        self.torque_command_publisher = self.create_publisher(JointState, "/commands", 10)

        # Buffer for observation, action
        self.buffers = LatestBuffers()
        
        # Mode switch (None = idle, no torque until keyboard selects a mode)
        self.publish_mode = None
        self._prev_mode = None

        # Base command state [v_x, v_y, w_z]
        self._base_command = np.zeros(3, dtype=np.float64)
        self._vx_step  = float(self.cfg.get("policy_command_vx_step",  0.1))
        self._wz_step  = float(self.cfg.get("policy_command_wz_step",  0.01))
        self._vx_limit = float(self.cfg.get("policy_command_vx_limit", 1.0))
        self._wz_limit = float(self.cfg.get("policy_command_wz_limit", 0.5))

        self.logger.info("Main Controller Node started")
        self.logger.info("===========================================")
        self.logger.info("[Keydown Menu]")
        self.logger.info("'p': Policy Control Mode")
        self.logger.info("'n': Nominal Control Mode")
        self.logger.info("'q': Quit")
        self.logger.info("--- Policy Command (active in Policy mode) ---")
        self.logger.info("'w'/'s': v_x +/-  |  'a'/'d': w_z +/-  |  'space': reset")
        self.logger.info("===========================================\r")

        # NOTE: 이전 버전 코드와 달라진 점 : Launch file로 한번에 운용하기 때문에, 키보드 입력을 받기 위해선 터미널 추가 설정이 필요함
        self.tty = open("/dev/tty", "rb+", buffering=0)
        self.settings = termios.tcgetattr(self.tty.fileno())
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

        # Timing
        self.num_joints = len(self.cfg["joint_names"])
        self.last_control_time = self.get_clock().now()

        # Control loop timer
        control_period_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self.control_timer = self.create_timer(control_period_sec, self._control_loop)

    # ---------------------------------------------------------------------
    # Callback Functions
    # ---------------------------------------------------------------------

    def _get_key(self):
        """Read a single character from the terminal immediately (Blocking)."""
        try:
            # NOTE: stdin -> tty
            tty.setraw(self.tty.fileno())
            key = self.tty.read(1).decode(errors="ignore")
        finally:
            # NOTE: stdin -> tty
            termios.tcsetattr(self.tty.fileno(), termios.TCSADRAIN, self.settings)
        return key

    def _keyboard_listener_loop(self):
        """Main loop to monitor keyboard input."""
        while rclpy.ok():
            key = self._get_key()
            
            if key == 'p':
                self.publish_mode = 'policy'
                self.logger.info("Mode changed: [Policy]\r")

            elif key == 'n':
                self.publish_mode = 'nominal'
                self.logger.info("Mode changed: [Nominal]\r")

            elif key == 'q':
                self.logger.info("Shutting down Agent Node...\r")
                rclpy.shutdown()
                break

            elif key == '\x03': # Ctrl+C
                rclpy.shutdown()
                break

            elif key == 'w':
                self._base_command[0] = float(np.clip(
                    self._base_command[0] + self._vx_step, -self._vx_limit, self._vx_limit))
                self.logger.info(f"Command: vx={self._base_command[0]:.2f} wz={self._base_command[2]:.2f}\r")

            elif key == 's':
                self._base_command[0] = float(np.clip(
                    self._base_command[0] - self._vx_step, 0, self._vx_limit))
                self.logger.info(f"Command: vx={self._base_command[0]:.2f} wz={self._base_command[2]:.2f}\r")

            elif key == 'a':
                self._base_command[2] = float(np.clip(
                    self._base_command[2] + self._wz_step, -self._wz_limit, self._wz_limit))
                self.logger.info(f"Command: vx={self._base_command[0]:.2f} wz={self._base_command[2]:.2f}\r")

            elif key == 'd':
                self._base_command[2] = float(np.clip(
                    self._base_command[2] - self._wz_step, -self._wz_limit, self._wz_limit))
                self.logger.info(f"Command: vx={self._base_command[0]:.2f} wz={self._base_command[2]:.2f}\r")

            elif key == ' ':
                self._base_command[:] = 0.0
                self.logger.info("Command reset to zero\r")

            else:
                self.logger.info("Wrong key! Please enter the right key")
                self.logger.info("===========================================")
                self.logger.info("[Keydown Menu]")
                self.logger.info("'p': Policy Control Mode")
                self.logger.info("'n': Nominal Control Mode")
                self.logger.info("'q': Quit")
                self.logger.info("--- Policy Command ---")
                self.logger.info("'w'/'s': v_x +/-  |  'a'/'d': w_z +/-  |  'space': reset")
                self.logger.info("===========================================\r")
                continue

    def sync_callback(self, joint_msg, imu_msg):
        self.joint_callback(joint_msg)
        self.imu_callback(imu_msg)

    def joint_callback(self, msg: JointState):
        self.buffers.joint_state_msg = msg

    def imu_callback(self, msg: ImuState):
        self.buffers.imu_msg = msg

    def reset(self) -> None:
        """Reset internal states (controller + safety limiter memory)."""
        self.safety_limiter.reset()
        self.policy_controller.reset()
        self.nominal_controller.reset()

    # ---------------------------------------------------------------------
    # Mode Switch
    # ---------------------------------------------------------------------

    def _switch_mode(self, new_mode: str) -> None:
        """Handle mode transition: reset previous controller + safety limiter LPF."""
        if new_mode == self._prev_mode:
            return

        # Reset previous controller state
        if self._prev_mode == 'policy':
            self.policy_controller.reset()
        elif self._prev_mode == 'nominal':
            self.nominal_controller.reset()

        # Reset LPF to prevent torque jump on mode switch
        self.safety_limiter.reset()

        # Reset command to zero on policy entry for safety
        if new_mode == 'policy':
            self._base_command[:] = 0.0

        self.logger.info(f"Controller switched: {self._prev_mode} -> {new_mode}\r")
        self._prev_mode = new_mode

    # ---------------------------------------------------------------------
    # Main Functions
    # ---------------------------------------------------------------------

    def _control_loop(self):
        """Main control loop called by create_timer at control_rate_hz."""
        # dt calculation
        now_time = self.get_clock().now()
        dt_sec = (now_time - self.last_control_time).nanoseconds * 1e-9
        if dt_sec <= 0.0:
            dt_sec = 1e-3
        self.last_control_time = now_time

        # Sensor data validity check
        joint_msg = self.buffers.joint_state_msg
        imu_msg = self.buffers.imu_msg
        if joint_msg is None or imu_msg is None:
            return  # No sensor data yet -> skip

        # Mode switch detection
        if self.publish_mode is not None:
            self._switch_mode(self.publish_mode)

        # Commands
        q_ref = np.zeros(self.num_joints, dtype=np.float32)
        v_ref = np.zeros(self.num_joints, dtype=np.float32)
        tau   = np.zeros(self.num_joints, dtype=np.float32)

        # Active controller execution
        if self.publish_mode == 'policy':
            self.policy_controller.set_command(self._base_command)
            raw_torque, q_ref, v_ref = self.policy_controller.compute(joint_msg, imu_msg, dt_sec)
            q_ref[:] = q_ref
            v_ref[-2:] = v_ref # Only for wheel

        elif self.publish_mode == 'nominal':
            raw_torque, q_ref, _ = self.nominal_controller.compute(joint_msg, imu_msg, dt_sec)
            q_ref[:] = q_ref
            v_ref[-2:] = 0 # Only for wheel
        else:
            raw_torque = tau        # Zero command

        self.logger.info(f"Mode : {self.publish_mode} | Raw torque : {raw_torque:.2f}")

        # SafetyLimiter
        joint_pos = np.asarray(joint_msg.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_msg.velocity, dtype=float).flatten()
        safe_torque, is_blocked = self.safety_limiter.apply(raw_torque, joint_pos, joint_vel)

        # Block handling (latching kill switch)
        if is_blocked:
            self.logger.error("SafetyLimiter BLOCKED! Publishing zero torque.\r")
            self.control_timer.cancel()
            safe_torque = np.zeros(self.num_joints)
        tau[:] = safe_torque

        # Publish torque command
        self._publish_torque_command(q_ref, v_ref, tau)

    def _publish_torque_command(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        """Publish torque command to /commands topic."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 
            'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]
        msg.position = position.tolist()
        msg.velocity = velocity.tolist()
        msg.effort = torque.tolist()
        self.torque_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
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