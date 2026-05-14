# controller_node.py — ROS2 ControllerNode (main control loop)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
import time
import copy
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState
from message_filters import Subscriber, ApproximateTimeSynchronizer

from goat_control.utils.controller.nominal_controller import NominalController
from goat_control.utils.controller.policy_controller import PolicyController
from goat_control.utils.controller.safety_limiter import SafetyLimiter


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

        self.num_joints = len(self.cfg["joint_names"])

        # QoS: latest-sample control. Avoid stale backlog from KEEP_ALL.
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriber
        self.joint_state_subscriber = self.create_subscription(JointState, 
                                                               "/joint_states", 
                                                               self.joint_callback, 
                                                               qos_profile=qos_profile,)

        self.imu_subscriber = self.create_subscription(ImuState, 
                                                       "/imu", 
                                                       self.imu_callback, 
                                                       qos_profile=qos_profile,)

        # Publisher
        self.torque_command_publisher = self.create_publisher(JointState, 
                                                              "/commands", 
                                                              qos_profile=qos_profile)
        
        # Messages
        self.joint_state_msg = None
        self.imu_msg = None
        self.prev_joint_rx_time = None
        self.prev_imu_rx_time = None
        self.last_joint_rx_time = None
        self.last_imu_rx_time = None
        
        # Mode switch (None = idle, no torque until keyboard selects a mode)
        self.publish_mode = None
        self._prev_mode = None

        # HIL safety latch
        self.kill_switch_on = False
        self.kill_reason = ""

        # Base command state [v_x, v_y, w_z]
        self._base_command = np.zeros(3, dtype=np.float64)
        self._vx_step  = float(self.cfg.get("policy_command_vx_step",  0.1))
        self._wz_step  = float(self.cfg.get("policy_command_wz_step",  0.01))
        self._vx_limit = float(self.cfg.get("policy_command_vx_limit", 1.0))
        self._wz_limit = float(self.cfg.get("policy_command_wz_limit", 0.5))

        # Timing
        self.last_tick_time = time.monotonic()

        # ROS-clock timestamp marking the start of the current control cycle (T0).
        # Carried in the /commands header so motor_io_node can measure end-to-end latency.
        self._cycle_start_stamp = self.get_clock().now()

        self.logger.info("Main Controller Node started")
        self.logger.info("===========================================")
        self.logger.info("[Keydown Menu]")
        self.logger.info("'p': Policy Control Mode")
        self.logger.info("'n': Nominal Control Mode")
        self.logger.info("'r': Controller reset")
        self.logger.info("'q': Quit")
        self.logger.info("--- Policy Command ---")
        self.logger.info("'w'/'s': v_x +/-  |  'a'/'d': w_z +/-")
        self.logger.info("===========================================\r")

        self.tty = open("/dev/tty", "rb+", buffering=0)
        self.settings = termios.tcgetattr(self.tty.fileno())
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

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
                if self.kill_switch_on:
                    self.logger.error(f"Cannot enter Policy mode. Kill switch is ON: {self.kill_reason}\r")
                    continue
                self.publish_mode = 'policy'
                self.logger.info("Mode changed: [Policy]\r")

            elif key == 'n':
                if self.kill_switch_on:
                    self.logger.error(f"Cannot enter Nominal mode. Kill switch is ON: {self.kill_reason}\r")
                    continue
                self.publish_mode = 'nominal'
                self.logger.info("Mode changed: [Nominal]\r")

            elif key == 'q':
                self.logger.info("Shutting down Agent Node...\r")
                rclpy.shutdown()
                break

            elif key == 'r':
                self._manual_reset()

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

            else:
                self.logger.info("Wrong key! Please enter the right key")
                self.logger.info("===========================================")
                self.logger.info("[Keydown Menu]")
                self.logger.info("'p': Policy Control Mode")
                self.logger.info("'n': Nominal Control Mode")
                self.logger.info("'r': Controller reset")
                self.logger.info("'q': Quit")
                self.logger.info("--- Policy Command ---")
                self.logger.info("'w'/'s': v_x +/-  |  'a'/'d': w_z +/-")
                self.logger.info("===========================================\r")
                continue

    def joint_callback(self, msg: JointState):
        self.joint_state_msg = msg
        self.prev_joint_rx_time = copy.deepcopy(self.last_joint_rx_time) if self.last_joint_rx_time is not None else time.monotonic()
        self.last_joint_rx_time = time.monotonic()

    def imu_callback(self, msg: ImuState):
        self.imu_msg = msg
        self.prev_imu_rx_time = copy.deepcopy(self.last_imu_rx_time) if self.last_imu_rx_time is not None else time.monotonic()
        self.last_imu_rx_time = time.monotonic()

    def reset(self) -> None:
        """Reset internal states (controller + safety limiter memory)."""
        # Prevent automatic controller re-entry.
        self.publish_mode = None
        self._prev_mode = None
        self._base_command[:] = 0.0

        # Reset stateful memories to avoid unsafe recovery transients.
        self.safety_limiter.reset()
        self.policy_controller.reset()
        self.nominal_controller.reset()

    # ---------------------------------------------------------------------
    # Mode Switch
    # ---------------------------------------------------------------------

    def _trigger_kill_switch(self, reason: str) -> None:
        """Latch kill switch. Manual reset is required before control resumes."""
        if not self.kill_switch_on:
            self.logger.error(f"KILL SWITCH ON: {reason}\r")

        self.kill_switch_on = True
        self.kill_reason = reason
        self.reset()

    def _manual_reset(self) -> None:
        """Clear kill latch and return controller to idle."""
        if self.kill_switch_on:
            self.logger.info(f"KILL SWITCH RESET: previous reason = {self.kill_reason}\r")

        self.kill_switch_on = False
        self.kill_reason = ""
        self.reset()

        self.logger.info("Controller is idle. Press 'p' or 'n' to re-enter control mode.\r")

    def _switch_mode(self, new_mode: str) -> None:
        """Handle mode transition: reset previous controller + safety limiter"""
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
        now_time = time.monotonic()

        dt_sec = now_time - self.last_tick_time
        if dt_sec <= 0.0:
            dt_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self.last_tick_time = now_time

        # T0: start of this control cycle (moment the agent begins computing).
        # ROS clock is used because it is comparable across processes.
        self._cycle_start_stamp = self.get_clock().now()

        # Commands
        q_ref = np.zeros(self.num_joints, dtype=np.float32)
        v_ref = np.zeros(self.num_joints, dtype=np.float32)
        tau   = np.zeros(self.num_joints, dtype=np.float32)

        #r =================== Proactive Condition Check ====================
        # Kill latch: do not auto-recover.
        if self.kill_switch_on:
            self.logger.error(f"Kill switch is ON: {self.kill_reason}. Publishing zero torque.\r", throttle_duration_sec=1.0)
            self._publish_torque_command(q_ref, v_ref, tau)
            return
        # Idle: zero command, no controller compute.
        if self.publish_mode is None:
            self._publish_torque_command(q_ref, v_ref, tau)
            return
        # Stale check
        is_stale, stale_reason = self._sensor_data_is_stale()
        if is_stale:
            self._trigger_kill_switch(f"Sensor stale: {stale_reason}")
            # self._publish_torque_command(q_ref, v_ref, tau)
            # return
        # ==================================================================

        # Mode switch detection
        self._switch_mode(self.publish_mode)

        # Active controller execution
        if self.publish_mode == 'policy':
            self.policy_controller.set_command(self._base_command)
            raw_torque, q_ref, wheel_v_ref = self.policy_controller.compute(self.joint_state_msg, 
                                                                            self.imu_msg, 
                                                                            dt_sec)
            v_ref[-2:] = wheel_v_ref # Only for wheel

        elif self.publish_mode == 'nominal':
            raw_torque, q_ref, _ = self.nominal_controller.compute(self.joint_state_msg, 
                                                                   self.imu_msg, 
                                                                   dt_sec)
            v_ref[-2:] = 0 # Only for wheel
        else:
            self._trigger_kill_switch(f"Invalid publish mode: {self.publish_mode}")
            self._publish_torque_command(q_ref, v_ref, tau)
            return

        # Safety Limiter
        joint_pos = np.asarray(self.joint_state_msg.position, dtype=float).flatten()
        joint_vel = np.asarray(self.joint_state_msg.velocity, dtype=float).flatten()
        safe_torque, is_blocked = self.safety_limiter.apply(raw_torque, joint_pos, joint_vel)

        # Time spent inside controller_node this cycle: agent inference + safety limiter.
        controller_internal_sec = (self.get_clock().now() - self._cycle_start_stamp).nanoseconds * 1e-9
        self.logger.info(
            f"[timing] controller_internal: {controller_internal_sec * 1e3:.3f} ms\r",
            throttle_duration_sec=1.0,
        )

        # Block handling (latching kill switch)
        if is_blocked:
            self._trigger_kill_switch("SafetyLimiter blocked command")
            self._publish_torque_command(q_ref * 0.0, v_ref * 0.0, tau)
            return

        # Publish torque command
        tau[:] = safe_torque

        self._publish_torque_command(q_ref, v_ref, tau)

        # inference_dt = time.monotonic() - now_time
        # self.logger.info(f"Inference time: {inference_dt:.6f} s\r")

    def _publish_torque_command(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        """Publish torque command to /commands topic."""
        msg = JointState()
        # Stamp with cycle-start time T0 (not publish time) so motor_io_node measures
        # latency from the moment the agent began computing this action.
        msg.header.stamp = self._cycle_start_stamp.to_msg()
        msg.name = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 
            'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]
        msg.position = position.tolist()
        msg.velocity = velocity.tolist()
        msg.effort = torque.tolist()

        self.torque_command_publisher.publish(msg)


    def _sensor_data_is_stale(self) -> tuple[bool, str]:
        """Return (is_stale, reason) using monotonic wall-clock receive age."""
        if self.joint_state_msg is None:
            return True, "joint_state_msg is None"
        if self.imu_msg is None:
            return True, "imu_msg is None"
        if self.last_joint_rx_time is None:
            return True, "last_joint_rx_time is None"
        if self.last_imu_rx_time is None:
            return True, "last_imu_rx_time is None"

        # now = self.get_clock().now()
        joint_age = self.last_joint_rx_time - self.prev_joint_rx_time
        imu_age = self.last_imu_rx_time - self.prev_imu_rx_time

        if joint_age > self.action_timeout_sec:
            return True, f"joint state stale: age={joint_age:.4f}s"
        if imu_age > self.action_timeout_sec:
            return True, f"imu stale: age={imu_age:.4f}s"
        return False, ""


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

if __name__ == "__main__":
    main()