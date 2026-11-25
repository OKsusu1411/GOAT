# controller_node.py — ROS2 ControllerNode (main control loop)
from __future__ import annotations

import numpy as np
import sys
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
from motor_interfaces.msg import BaseCommands

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
      4) SafetyLimiter applies clipping + kill switch
      5) Publish safe torque command
    """
    def __init__(self):
        super().__init__("controller_node")

        # Parameters by Launch File
        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")
        self.declare_parameter("action_timeout_sec", 0.05)
        self.declare_parameter("imu_port", "/dev/ttyUSB0")
        self.declare_parameter("imu_baudrate", 115200)
        self.declare_parameter("imu_timeout", 1.0)

        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, False)])

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
        self.cfg["nsc_urdf_path"] = copy.deepcopy(self.urdf_path)

        # Logger
        self.logger = self.get_logger()
        self.logger.info(f"Python thread switch interval: {sys.getswitchinterval() * 1e3:.3f} ms")

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
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", qos_profile=qos_profile)
        self.imu_state_pub = self.create_publisher(ImuState, "/imu", qos_profile=qos_profile)
        self.torque_command_pub = self.create_publisher(JointState,  "/commands",  qos_profile=qos_profile)
        self.base_command_pub = self.create_publisher(BaseCommands, "/base_commands", qos_profile=qos_profile)
        
        # Mode switch (None = idle, no torque until keyboard selects a mode)
        self.publish_mode = None
        self._prev_mode = None

        # HIL safety latch
        self.kill_switch_on = False
        self.kill_reason = ""

        self.logger.info("Main Controller Node started")
        self._print_menu()

        self.tty = open("/dev/tty", "rb+", buffering=0)
        self.settings = termios.tcgetattr(self.tty.fileno())
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

        # Timing
        self.last_tick_time = time.perf_counter()
        self.prev_write_time = None
        self.write_delta_sum = 0.0
        self.write_delta_min = 0.0
        self.write_delta_max = 0.0

        # Hardware-control thread to skip ROS overhead and sleep
        self.period_ns = int(1.0 / max(self.control_rate_hz, 1.0) * 1e9)
        self.deadline_miss_count = 0

        self._rate_cycle_count = 0
        self._rate_check_interval = 500
        self._rate_window_start = time.perf_counter()
        self._control_stop_event = threading.Event()
        self.control_thread = threading.Thread(target=self._control_thread_loop, name="goat_control_thread", daemon=False,)
        self.control_thread.start()

    # ---------------------------------------------------------------------
    # Callback Functions
    # ---------------------------------------------------------------------

    def _print_menu(self) -> None:
        """Print the keyboard menu. Command keys are described by the active
        policy controller (per its tracking mode)."""
        self.logger.info("===========================================")
        self.logger.info("[Keydown Menu]")
        self.logger.info("'p': Policy Control Mode")
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
            tty.setcbreak(self.tty.fileno())
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

            elif key == 'q':
                self.logger.info("Shutting down Agent Node...\r")
                rclpy.shutdown()
                break

            elif key == 'r':
                self.reset()

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
        """Return True if any element of the joint or IMU state is NaN."""
        # Joint state: position / velocity / effort arrays.
        js = joint_state_msg
        if js is not None:
            for field in (js.position, js.velocity, js.effort):
                if np.any(np.isnan(np.asarray(field, dtype=float))):
                    return True

        # IMU state: quaternion + gyro / vel / mag vectors.
        imu = imu_msg
        if imu is not None:
            imu_values = [imu.quat.x, imu.quat.y, imu.quat.z, imu.quat.w,
                          imu.gyro.x, imu.gyro.y, imu.gyro.z,
                          imu.acc.x, imu.acc.y, imu.acc.z,
                          imu.mag.x, imu.mag.y, imu.mag.z,
                          ]
            if np.any(np.isnan(np.asarray(imu_values, dtype=float))):
                return True

        return False

    def reset(self) -> None:
        """Reset internal states (controller + safety limiter memory)."""
        if self.kill_switch_on:
            self.logger.info(f"KILL SWITCH RESET: previous reason = {self.kill_reason}\r")

        self.kill_switch_on = False
        self.kill_reason = ""

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

    def _switch_mode(self, new_mode: str) -> None:
        """Handle mode transition: reset previous controller + safety limiter"""
        if new_mode == self._prev_mode:
            return

        # Reset previous controller state
        if self._prev_mode == 'policy':
            self.policy_controller.reset()
        elif self._prev_mode == 'nominal':
            self.nominal_controller.reset()

        # Reset Limiter
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
        # Time information
        now_time = time.perf_counter()
        dt_sec = (now_time - self.last_tick_time)
        if dt_sec <= 0.0: dt_sec = 1.0 / max(self.control_rate_hz, 1.0)

        # Messages
        joint_state_msg = self.motor_io.read_joint_state()
        imu_msg = self.imu_io.read_imu() 

        # Commands
        q_ref = np.zeros(self.num_joints, dtype=np.float32)
        v_ref = np.zeros(self.num_joints, dtype=np.float32)
        tau   = np.zeros(self.num_joints, dtype=np.float32)
        base_commands = self.policy_controller._base_command

        # =================== Proactive Condition Check ====================
        # Kill latch: do not auto-recover.
        if self.kill_switch_on:
            self.logger.error(f"Kill switch is ON: {self.kill_reason}. Publishing zero torque.\r", throttle_duration_sec=1.0)
            self.motor_io.write_motor(tau)
            self._publish(q_ref, v_ref, tau, base_commands, joint_state_msg, imu_msg)
            return
        
        # Idle: zero command, no controller compute.
        if self.publish_mode is None:
            self.motor_io.write_motor(tau)
            self._publish(q_ref, v_ref, tau, base_commands, joint_state_msg, imu_msg)
            return
        
        # Sensor validity check: a NaN in joint/IMU state would propagate into the torque
        if self._sensor_data_has_nan(joint_state_msg, imu_msg):
            self._trigger_kill_switch("NaN detected in joint/IMU state")
            self.motor_io.write_motor(tau)
            self._publish(q_ref, v_ref, tau, base_commands, joint_state_msg, imu_msg)
            return

        # =================== Controller ====================

        # Mode switch detection
        self._switch_mode(self.publish_mode)

        # Active controller execution                                 
        if self.publish_mode == 'policy':
            # Command is owned and updated by the controller itself (handle_key).
            joint_torque, q_ref, wheel_v_ref = self.policy_controller.compute(joint_state_msg, imu_msg, dt_sec)
            # Only write into wheel slots that actually exist in this config.
            wheel_indices = self.cfg["wheel_indices"]
            if len(wheel_indices) > 0:
                v_ref[wheel_indices] = wheel_v_ref

        else:
            self._trigger_kill_switch(f"Invalid publish mode: {self.publish_mode}")
            self._publish(q_ref, v_ref, tau, base_commands, joint_state_msg, imu_msg)
            self.motor_io.write_motor(tau)
            return

        # Safety Limiter
        joint_pos = np.asarray(joint_state_msg.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_state_msg.velocity, dtype=float).flatten()
        safe_torque, is_blocked = self.safety_limiter.apply(joint_torque, joint_pos, joint_vel)

        # Block handling (latching kill switch)
        if is_blocked:
            self._trigger_kill_switch("SafetyLimiter blocked command")
            self.motor_io.write_motor(tau)
            return

        # Publish torque command
        tau[:] = safe_torque
        self.motor_io.write_motor(tau)

        # =================== Logging ====================

        # write_time = time.perf_counter()
        # if self.prev_write_time is None:
        #     self.prev_write_time = time.perf_counter()

        # delta_write_time = write_time - self.prev_write_time
        # self.prev_write_time = write_time   

        # self.write_delta_min = min(self.write_delta_min, delta_write_time)
        # self.write_delta_max = max(self.write_delta_max, delta_write_time)
        # self.write_delta_sum += delta_write_time                          

        # Publish for logging
        # self._publish(q_ref, v_ref, safe_torque, joint_state_msg, imu_msg)  

        # self._rate_cycle_count += 1
        # if self._rate_cycle_count >= self._rate_check_interval:
        #     avg_dt_ms = (self._dt_sum / self._rate_cycle_count) * 1e3
        #     avg_write_dt_ms = (self.write_delta_sum / self._rate_cycle_count) * 1e3

        #     self.logger.info(
        #         f"[Apply] mean={avg_write_dt_ms:.3f} ms | "
        #         f"[Apply] min={self.write_delta_min * 1e3:.3f} ms | "
        #         f"[Apply] max={self.write_delta_max * 1e3:.3f} ms | "
        #         f"[CAN] read_request={self.motor_io.motor_manager._last_read_request_ms:.3f} | "
        #         f"[CAN] read_wait={self.motor_io.motor_manager._last_read_wait_ms:.3f} | "
        #         f"[CAN] write_request={self.motor_io.motor_manager._last_write_request_ms:.3f}\r"
        #     )

        #     self._rate_cycle_count = 0
        #     self._dt_min = float("inf")
        #     self._dt_max = 0.0
        #     self._dt_sum = 0.0

        #     self.write_delta_max = 0.0
        #     self.write_delta_min = float("inf")
        #     self.write_delta_sum = 0.0

    def _publish(self, position: np.ndarray, velocity: np.ndarray, effort: np.ndarray, base_command: np.ndarray, joint_state_msg, imu_msg) -> None:
        """Publish joint state, IMU, and torque commands for logging."""
        # Interrupting handling
        if not rclpy.ok():
            return
        
        # Update joint state message for logging
        msg_joint = JointState()
        msg_joint.header.stamp = joint_state_msg.header.stamp
        msg_joint.header.frame_id = "base_link"
        msg_joint.name = joint_state_msg.name
        msg_joint.position = joint_state_msg.position
        msg_joint.velocity = joint_state_msg.velocity
        msg_joint.effort = joint_state_msg.effort

        # Update IMU message for logging 
        msg_imu = ImuState()
        msg_imu.header.stamp = joint_state_msg.header.stamp
        msg_imu.quat = imu_msg.quat
        msg_imu.gyro = imu_msg.gyro
        msg_imu.acc = imu_msg.acc
        msg_imu.mag = imu_msg.mag
        msg_imu.time_ms = imu_msg.time_ms

        # Update joint command message
        msg_command = JointState()
        msg_command.header.stamp = joint_state_msg.header.stamp
        msg_command.name = joint_state_msg.name
        msg_command.position = position.tolist()
        msg_command.velocity = velocity.tolist()
        msg_command.effort = effort.tolist()

        msg_base_command = BaseCommands()
        msg_base_command.header.stamp = joint_state_msg.header.stamp
        msg_base_command.vx = base_command[0]
        msg_base_command.wz = base_command[2]
        msg_base_command.height = base_command[3]

        # Publish
        self.joint_state_pub.publish(msg_joint)
        self.imu_state_pub.publish(msg_imu)
        self.torque_command_pub.publish(msg_command)
        self.base_command_pub.publish(msg_base_command)


    def _control_thread_loop(self):
        """Run the hardware control step at a fixed absolute rate."""
        next_tick_ns = time.perf_counter_ns()
        prev_start_ns = None

        # Timing statistics
        log_interval = 2000
        count = 0
        period_sum = 0
        period_min = float("inf")
        period_max = 0
        period_count = 0
        wake_sum = 0
        wake_min = float("inf")
        wake_max = 0
        wake_count = 0
        exec_sum = 0
        exec_min = float("inf")
        exec_max = 0

        def fmt_timing(total, minimum, maximum, n):
            if n == 0:
                return "N/A"
            return (
                f"mean={total / n / 1e6:.3f} ms | "
                f"min={minimum / 1e6:.3f} ms | "
                f"max={maximum / 1e6:.3f} ms"
            )

        while rclpy.ok() and not self._control_stop_event.is_set():

            # 1. Wait until absolute scheduled time
            remaining_ns = next_tick_ns - time.perf_counter_ns()
            waited = remaining_ns > 0

            if self._control_stop_event.wait(remaining_ns * 1e-9):
                break

            # 2. Record actual start time
            start_ns = time.perf_counter_ns()

            # [STATS] Actual period
            if prev_start_ns is not None:
                period_ns = start_ns - prev_start_ns
                period_sum += period_ns
                period_min = min(period_min, period_ns)
                period_max = max(period_max, period_ns)
                period_count += 1
            # [STATS] Wake-up delay
            if waited:
                wake_delay_ns = max(0, start_ns - next_tick_ns)
                wake_sum += wake_delay_ns
                wake_min = min(wake_min, wake_delay_ns)
                wake_max = max(wake_max, wake_delay_ns)
                wake_count += 1

            prev_start_ns = start_ns

            # 3. Execute control loop
            self._control_loop()

            # 4. Schedule next absolute cycle
            next_tick_ns += self.period_ns

            # [STATS] Execution time
            end_ns = time.perf_counter_ns()
            execution_ns = end_ns - start_ns
            exec_sum += execution_ns
            exec_min = min(exec_min, execution_ns)
            exec_max = max(exec_max, execution_ns)
            # [STATS] Skip missed cycles
            if end_ns >= next_tick_ns:
                skipped = (end_ns - next_tick_ns) // self.period_ns + 1
                self.deadline_miss_count += skipped
                next_tick_ns += skipped * self.period_ns

            # [LOGGING] Periodic timing report
            count += 1
            if count >= log_interval:
                self.logger.info(
                    "\n[Control Thread Timing]\n"
                    f"Period    : {fmt_timing(period_sum, period_min, period_max, period_count)}\n"
                    f"Wake Delay: {fmt_timing(wake_sum, wake_min, wake_max, wake_count)}\n"
                    f"Execution : {fmt_timing(exec_sum, exec_min, exec_max, count)}\n"
                    f"Total Deadline Miss : {self.deadline_miss_count}"
                )
                count = 0
                period_sum = 0
                period_min = float("inf")
                period_max = 0
                period_count = 0
                wake_sum = 0
                wake_min = float("inf")
                wake_max = 0
                wake_count = 0
                exec_sum = 0
                exec_min = float("inf")
                exec_max = 0

    def stop_control_thread(self) -> None:
        """Stop and join the dedicated hardware-control thread."""
        self._control_stop_event.set()
        control_thread = getattr(self, "control_thread", None)
        if control_thread is not None and control_thread.is_alive():
            control_thread.join(timeout=1.0)

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
            node.stop_control_thread()
            for _ in range(10):
                node.motor_io.write_motor(np.zeros(node.num_joints, dtype=np.float32))
                time.sleep(0.1)
            if hasattr(node, "motor_io"):
                node.motor_io.close()
            if hasattr(node, "imu_io"):
                node.imu_io.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == "__main__":
    main()
