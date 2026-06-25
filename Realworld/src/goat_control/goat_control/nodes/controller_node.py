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
from goat_control.utils.controller.safety_limiter import SafetyLimiter
from goat_control.utils.controller.fixed_policy_controller import FixedBasePolicyController
from goat_control.utils.controller.movable_policy_controller import MovableBasePolicyController
from goat_control.nodes.motor_io import MotorIO
from goat_control.nodes.imu_io import ImuIO


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
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
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

        # Controller
        self.safety_limiter = SafetyLimiter(self.cfg, self.logger)
        self.nominal_controller = NominalController(self.cfg, self.logger)
        if self.cfg["policy_mode"] == "fixed":
            self.policy_controller = FixedBasePolicyController(self.cfg, self.logger)
        elif self.cfg["policy_mode"] == "movable":
            self.policy_controller = MovableBasePolicyController(self.cfg, self.logger)
        else:
            raise RuntimeError(f"Invalid Mode : {self.cfg['policy_mode']}")

        self.num_joints = len(self.cfg["joint_names"])

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
        self.joint_state_msg = self.motor_io.latest_joint_state  # seeded by MotorIO's initial read
        self.imu_msg = self.imu_io.latest_imu_state              # seeded by ImuIO's initial read
        
        # Mode switch (None = idle, no torque until keyboard selects a mode)
        self.publish_mode = None
        self._prev_mode = None

        # HIL safety latch
        self.kill_switch_on = False
        self.kill_reason = ""

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

    # ---------------------------------------------------------------------
    # Callback Functions
    # ---------------------------------------------------------------------

    def _print_menu(self) -> None:
        """Print the keyboard menu. Command keys are described by the active
        policy controller (per its tracking mode)."""
        self.logger.info("===========================================")
        self.logger.info("[Keydown Menu]")
        self.logger.info("'p': Policy Control Mode")
        self.logger.info("'n': Nominal Control Mode")
        self.logger.info("'r': Controller reset")
        self.logger.info("'q': Quit")
        self.logger.info("[Command Mode]")
        for line in self.policy_controller.command_help():
            self.logger.info(line)
        self.logger.info("===========================================\r")

    def _get_key(self):
        """Read a key from the terminal immediately (Blocking).

        Arrow keys arrive as 3-byte escape sequences (ESC '[' 'A'..'D') and are
        normalized to 'UP'/'DOWN'/'RIGHT'/'LEFT' tokens. All other keys are
        returned as their single-character string.
        """
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
                self.motor_io.read_write_motor(np.zeros(self.num_joints, dtype=np.float32))
                rclpy.shutdown()
                break

            elif key == 'r':
                self._manual_reset()

            elif key == '\x03': # Ctrl+C
                rclpy.shutdown()
                break

            else:
                # Delegate command keys to the active policy controller
                log = self.policy_controller.handle_key(key)
                if log is not None:
                    self.logger.info(log)
                else:
                    self.logger.info("Wrong key! Please enter the right key")
                    self._print_menu()
                continue

    def _sensor_data_has_nan(self, joint_state_msg, imu_msg) -> bool:
        """Return True if any element of the joint or IMU state is NaN.

        A motor that does not answer the state read leaves NaN in joint
        velocity/effort; a bad IMU frame leaves NaN in the IMU fields. Either
        would propagate into the torque command and crash the CAN current
        conversion, so the caller treats True as a kill condition. Does not
        modify the data — detection only.
        """
        # Joint state: position / velocity / effort arrays.
        js = joint_state_msg
        if js is not None:
            for field in (js.position, js.velocity, js.effort):
                if np.any(np.isnan(np.asarray(field, dtype=float))):
                    return True

        # IMU state: quaternion + gyro / vel / mag vectors.
        imu = imu_msg
        if imu is not None:
            imu_values = [
                imu.quat.x, imu.quat.y, imu.quat.z, imu.quat.w,
                imu.gyro.x, imu.gyro.y, imu.gyro.z,
                imu.vel.x, imu.vel.y, imu.vel.z,
                imu.mag.x, imu.mag.y, imu.mag.z,
            ]
            if np.any(np.isnan(np.asarray(imu_values, dtype=float))):
                return True

        return False

    def reset(self) -> None:
        """Reset internal states (controller + safety limiter memory)."""
        # Prevent automatic controller re-entry.
        self.publish_mode = None
        self._prev_mode = None

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
            self.policy_controller.reset()

        self.logger.info(f"Controller switched: {self._prev_mode} -> {new_mode}\r")
        self._prev_mode = new_mode

    # ---------------------------------------------------------------------
    # Main Functions
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
        t_imu_start = time.perf_counter()    
        imu_msg = self.imu_io.read_imu()
        imu_read_ms = (time.perf_counter() - t_imu_start) * 1e3                 

        # Commands
        q_ref = np.zeros(self.num_joints, dtype=np.float32)
        v_ref = np.zeros(self.num_joints, dtype=np.float32)
        tau   = np.zeros(self.num_joints, dtype=np.float32)

        #r =================== Proactive Condition Check ====================
        # Kill latch: do not auto-recover.
        if self.kill_switch_on:
            self.logger.error(f"Kill switch is ON: {self.kill_reason}. Publishing zero torque.\r", throttle_duration_sec=1.0)
            self.motor_io.read_write_motor(tau)
            self._publish(q_ref, v_ref, tau, joint_state_msg, imu_msg)
            return
        
        # Idle: zero command, no controller compute.
        if self.publish_mode is None:
            self.motor_io.read_write_motor(tau)
            self._publish(q_ref, v_ref, tau, joint_state_msg, imu_msg)
            return
        
        # Sensor validity check: a NaN in joint/IMU state would propagate into the torque
        if self._sensor_data_has_nan(joint_state_msg, imu_msg):
            self._trigger_kill_switch("NaN detected in joint/IMU state")
            self.motor_io.read_write_motor(tau)
            self._publish(q_ref, v_ref, tau, joint_state_msg, imu_msg)
            return
        # ==================================================================

        # Mode switch detection
        self._switch_mode(self.publish_mode)

        # Active controller execution
        t_ctrl_start = time.perf_counter()                                    
        if self.publish_mode == 'policy':
            # Command is owned and updated by the controller itself (handle_key).
            joint_torque, q_ref, wheel_v_ref = self.policy_controller.compute(joint_state_msg,
                                                                              imu_msg,
                                                                              dt_sec)
            v_ref[-2:] = wheel_v_ref # Only for wheel

        elif self.publish_mode == 'nominal':
            joint_torque, q_ref, _ = self.nominal_controller.compute(joint_state_msg,
                                                                     imu_msg,
                                                                     dt_sec)
            v_ref[-2:] = 0 # Only for wheel

        else:
            self._trigger_kill_switch(f"Invalid publish mode: {self.publish_mode}")
            self._publish(q_ref, v_ref, tau, joint_state_msg, imu_msg)
            self.motor_io.read_write_motor(tau * 0.0)
            return

        # Safety Limiter
        joint_pos = np.asarray(joint_state_msg.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_state_msg.velocity, dtype=float).flatten()
        safe_torque, is_blocked = self.safety_limiter.apply(joint_torque, joint_pos, joint_vel)

        # Block handling (latching kill switch)
        if is_blocked:
            self._trigger_kill_switch("SafetyLimiter blocked command")
            self.motor_io.read_write_motor(tau)
            return

        # Publish torque command
        tau[:] = safe_torque
        ctrl_compute_ms = (time.perf_counter() - t_ctrl_start) * 1e3                    # [timing] controller compute duration in ms

        # Apply action
        t_can_start = time.perf_counter()                                               # [timing] start CAN write+read window
        self.motor_io.read_write_motor(tau)                                     
        can_io_ms = (time.perf_counter() - t_can_start) * 1e3                           # [timing] CAN write+read duration in ms

        # Publish for logging
        self._publish(q_ref, v_ref, tau, joint_state_msg, imu_msg)

        # Per-segment timing breakdown. Comment out once bottleneck confirmed.
        total_ms = (time.perf_counter() - now_time) * 1e3                               # [timing] full _control_loop duration in ms
        tx_submit_ms = getattr(self.motor_io.motor_manager, "_last_tx_submit_ms", 0.0)  # [timing] send phase cost
        tx_wait_ms = getattr(self.motor_io.motor_manager, "_last_tx_wait_ms", 0.0)      # [timing] cache-read+parse cost

        # # Time logging
        self.logger.info(
            f"[timing] total: {total_ms:6.2f} ms | {1.0 / max(total_ms * 1e-3, 1e-6):6.1f} Hz "
            f"| can: {can_io_ms:6.2f} ms (tx {tx_submit_ms:5.2f} / rx {tx_wait_ms:6.2f}) "
            f"| imu: {imu_read_ms:5.2f} ms | ctrl: {ctrl_compute_ms:5.2f} ms \r",
            throttle_duration_sec=5.0,
        )

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
        msg_command.name = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 
            'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]
        msg_command.position = position.tolist()
        msg_command.velocity = velocity.tolist()
        msg_command.effort = effort.tolist()

        # Publish
        self.joint_state_pub.publish(msg_joint)
        self.imu_state_pub.publish(msg_imu)
        self.torque_command_pub.publish(msg_command)


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
            if hasattr(node, "motor_io"):
                node.motor_io.close()
            if hasattr(node, "imu_io"):
                node.imu_io.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == "__main__":
    main()