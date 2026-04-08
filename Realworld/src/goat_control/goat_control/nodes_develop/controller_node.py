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
from motor_interfaces.msg import BaseStates
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ..util_develop.nominal_controller import NominalController
from ..util_develop.policy_controller import PolicyController
from ..util_develop.safety_limiter import SafetyLimiter

@dataclass
class LatestBuffers:
    """Thread-safe buffers for incoming messages."""
    joint_state_msg: Optional[JointState] = None
    imu_msg: Optional[BaseStates] = None


class ControllerNode(Node):
    """ROS2 control node: sensor reception -> controller selection -> torque publishing.

    Flow:
      1) Receive JointState + BaseStates via time-synced subscribers
      2) Keyboard selects active controller (policy / nominal)
      3) Active controller computes raw torque
      4) SafetyLimiter applies LPF + clipping + kill switch
      5) Publish safe torque command
    """
    def __init__(self):
        super().__init__("goat_control_node")

        # Parameters by Launch File
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")
        self.declare_parameter("checkpoint_path", None)
        self.declare_parameter("action_timeout_sec", 0.05)
        self.declare_parameter("debug_print_period_sec", 0.2)

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.yaml_path = str(self.get_parameter("yaml_path").value)
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value) or None # TODO: Launch file에서 checkpoint path 명시해야 함
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self.debug_print_period_sec = float(self.get_parameter("debug_print_period_sec").value)

        # Parameters by Yaml File
        with open(self.yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)
        if not isinstance(self.cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")
        self.cfg["nsc_urdf_path"] = copy.deepcopy(self.urdf_path) # URDF path should be assigned in runtime

        if self.checkpoint_path is not None:
            self.cfg["policy_checkpoint_path"] = copy.deepcopy(self.checkpoint_path) # Default is None

        # Controller
        self.nominal_controller = NominalController(self.cfg)
        self.policy_controller = PolicyController(self.cfg)
        self.safety_limiter = SafetyLimiter(self.cfg)

        # Subscriber
        self.joint_state_subscriber = Subscriber(self, JointState, '/joint_states', 10)
        self.imu_subscriber = Subscriber(self, BaseStates, '/imu', 10) # TODO: BaseStates ----> BaseState로 변경 (Topic명 일치) + 메시지 구성 논의
        self.time_sync = ApproximateTimeSynchronizer([self.joint_state_subscriber, self.imu_subscriber], 10, 0.01)
        self.time_sync.registerCallback(self.sync_callback)

        # Publisher
        self.torque_command_publisher = self.create_publisher(Float32MultiArray, "/torque", 10)

        # Buffer for observation, action
        self.buffers = LatestBuffers()
        
        # Mode switch (None = idle, no torque until keyboard selects a mode)
        self.publish_mode = None
        self._prev_mode = None
        self.settings = termios.tcgetattr(sys.stdin)
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

        # Timing
        self.num_joints = len(self.cfg["joint_names"])
        self.last_control_time = self.get_clock().now()

        # Control loop timer
        control_period_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self.control_timer = self.create_timer(control_period_sec, self._control_loop)

        self.get_logger().info("!! Main Controller Node started !!")
        print("="*30)
        print("[Keydown Menu]")
        print("'p': Policy Control Mode")
        print("'n': Nominal Control Mode")
        print("'q': Quit")
        print("="*30)


    # ---------------------------------------------------------------------
    # Callback Functions
    # ---------------------------------------------------------------------

    def _get_key(self):
        """Read a single character from the terminal immediately (Blocking)."""
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def _keyboard_listener_loop(self):
        """Main loop to monitor keyboard input."""
        while rclpy.ok():
            key = self._get_key()
            
            if key == 'p':
                self.publish_mode = 'policy'
                self.get_logger().info("Mode changed: [Policy]")
                
            elif key == 'n':
                self.publish_mode = 'nominal'
                self.get_logger().info("Mode changed: [Nominal]")
                
            elif key == 'q':
                self.get_logger().info("Shutting down Agent Node...")
                rclpy.shutdown()
                break
            
            elif key == '\x03': # Ctrl+C
                rclpy.shutdown()
                break
            
            else:
                self.get_logger().info("Wrong key! Please enter the right key")
                print("="*30)
                print("[Keydown Menu]")
                print("'p': Policy Control Mode")
                print("'n': Nominal Control Mode")
                print("'q': Quit")
                print("="*30)
                continue

    def sync_callback(self, joint_msg, imu_msg):
        self.joint_callback(joint_msg)
        self.imu_callback(imu_msg)

    def joint_callback(self, msg: JointState):
        self.buffers.joint_state_msg = msg

    def imu_callback(self, msg: BaseStates):
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

        self.get_logger().info(f"Controller switched: {self._prev_mode} -> {new_mode}")
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
        self._switch_mode(self.publish_mode)

        # Active controller execution
        if self.publish_mode == 'policy':
            raw_torque, q_ref, v_ref = self.policy_controller.compute(joint_msg, imu_msg, dt_sec)

        elif self.publish_mode == 'nominal':
            raw_torque, q_ref, _ = self.nominal_controller.compute(joint_msg, imu_msg, dt_sec)

        else:
            # None (wait) or unknown mode -> zero torque, skip safety/publish
            return

        # SafetyLimiter
        joint_pos = np.asarray(joint_msg.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_msg.velocity, dtype=float).flatten()
        safe_torque, is_blocked = self.safety_limiter.apply(raw_torque, joint_pos, joint_vel)

        # Block handling (latching kill switch)
        if is_blocked:
            self.get_logger().error("SafetyLimiter BLOCKED! Publishing zero torque.")
            self.control_timer.cancel()
            safe_torque = np.zeros(self.num_joints)

        # Publish torque command
        self._publish_torque_command(safe_torque)

    def _publish_torque_command(self, torque: np.ndarray) -> None:
        """Publish torque command to /torque topic."""
        msg = Float32MultiArray()
        msg.data = torque.astype(np.float32).tolist()
        self.torque_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()