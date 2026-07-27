# controller_node.py — ROS2 Sim ControllerNode with H1-style synchronized I/O pipeline
from __future__ import annotations

import copy
import threading
import termios
import tty

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState
from message_filters import Subscriber, TimeSynchronizer

from goat_control.utils.controller.nominal_controller import NominalController
from goat_control.utils.controller.safety_limiter import SafetyLimiter
from goat_control.utils.controller.fixed_policy_controller import FixedBasePolicyController
from goat_control.utils.controller.movable_policy_controller import MovableBasePolicyController


class SimControllerNode(Node):
    """ROS2 simulation controller node with H1-style topic I/O pipeline.

    Pipeline:
      1) /joint_states + /imu are received through message_filters.Subscriber
      2) TimeSynchronizer pairs both messages by header.stamp
      3) _tick(joint_msg, imu_msg) becomes the control tick
      4) Active controller computes command
      5) SafetyLimiter filters/clips torque
      6) JointState command is published to /joint_command
    """

    def __init__(self):
        super().__init__("sim_controller_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("urdf_path", "WF_GOAT.urdf")
        self.declare_parameter("sync_queue_size", 10)

        # use simulation time.
        self.set_parameters(
            [
                rclpy.parameter.Parameter(
                    "use_sim_time",
                    rclpy.Parameter.Type.BOOL,
                    True,
                )
            ]
        )

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.yaml_path = str(self.get_parameter("yaml_path").value)

        # ------------------------------------------------------------------
        # YAML config
        # ------------------------------------------------------------------
        with open(self.yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)

        if not isinstance(self.cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")

        self.cfg["nsc_urdf_path"] = copy.deepcopy(self.urdf_path)

        # Checkpoint path
        self.checkpoint_path = copy.deepcopy(self.cfg["policy_checkpoint_path"])

        # ------------------------------------------------------------------
        # Logger
        # ------------------------------------------------------------------
        self.logger = self.get_logger()

        # ------------------------------------------------------------------
        # Controllers
        # ------------------------------------------------------------------
        self.safety_limiter = SafetyLimiter(self.cfg, self.logger)
        self.nominal_controller = NominalController(self.cfg, self.logger)
        if self.cfg["policy_mode"] == "fixed":
            self.policy_controller = FixedBasePolicyController(self.cfg, self.logger)
        elif self.cfg["policy_mode"] == "movable":
            self.policy_controller = MovableBasePolicyController(self.cfg, self.logger)
        else:
            raise RuntimeError(f"Invalid Mode : {self.cfg['policy_mode']}")

        self.num_joints = len(self.cfg["joint_names"])

        # ------------------------------------------------------------------
        # QoS: match NVIDIA H1 ROS2 RL controller style
        # ------------------------------------------------------------------
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        # ------------------------------------------------------------------
        # synchronized subscribers
        # ------------------------------------------------------------------
        self._joint_states_sub = Subscriber(self, JointState, "/joint_states", qos_profile=sim_qos_profile)
        self._imu_sub = Subscriber(self, ImuState, "/imu", qos_profile=sim_qos_profile)

        self.sync = TimeSynchronizer([self._joint_states_sub, self._imu_sub], 10)
        self.sync.registerCallback(self._tick)

        # ------------------------------------------------------------------
        # command publisher
        # ------------------------------------------------------------------
        self.joint_command_publisher = self.create_publisher(JointState, "/commands", qos_profile=sim_qos_profile)

        # ------------------------------------------------------------------
        # Mode switch
        # ------------------------------------------------------------------
        self.publish_mode = None
        self._prev_mode = None

        # Timing
        self.last_tick_time = self.get_clock().now()

        self.logger.info("SimControllerNode started with synchronized I/O pipeline")
        self.logger.info("===========================================")
        self.logger.info("[Keydown Menu]")
        self.logger.info("'p': Policy Control Mode")
        self.logger.info("'n': Nominal Control Mode")
        self.logger.info("'q': Quit")
        # Command keys are interpreted by the active policy controller.
        self.logger.info("[Command Mode]")
        for line in self.policy_controller.command_help():
            self.logger.info(line)
        self.logger.info("===========================================\r")

        # Keyboard input
        self.tty = open("/dev/tty", "rb+", buffering=0)
        self.settings = termios.tcgetattr(self.tty.fileno())
        self.input_thread = threading.Thread(
            target=self._keyboard_listener_loop,
            daemon=True,
        )
        self.input_thread.start()

    # ----------------------------------------------------------------------
    # Keyboard
    # ----------------------------------------------------------------------
    def _get_key(self):
        """Read a key from the terminal immediately.

        Arrow keys arrive as 3-byte escape sequences (ESC '[' 'A'..'D') and are
        normalized to 'UP'/'DOWN'/'RIGHT'/'LEFT' tokens. All other keys are
        returned as their single-character string.
        """
        try:
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
            termios.tcsetattr(self.tty.fileno(), termios.TCSADRAIN, self.settings)

    def _keyboard_listener_loop(self):
        """Main loop to monitor keyboard input."""
        while rclpy.ok():
            key = self._get_key()

            if key == "p":
                self.publish_mode = "policy"
                self.logger.info("Mode changed: [Policy]\r")

            elif key == "n":
                self.publish_mode = "nominal"
                self.logger.info("Mode changed: [Nominal]\r")

            elif key == "q":
                self.logger.info("Shutting down SimControllerNode...\r")
                rclpy.shutdown()
                break

            elif key == "\x03":  # Ctrl+C
                rclpy.shutdown()
                break

            else:
                # Delegate command keys to the active policy controller, which
                # interprets them per its tracking mode (joint position / base
                # velocity) and returns a log string (or None if unhandled).
                log = self.policy_controller.handle_key(key)
                if log is not None:
                    self.logger.info(log)
                else:
                    self.logger.info("Wrong key! Please enter the right key\r")

    # ----------------------------------------------------------------------
    # Reset / mode switch
    # ----------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal states."""
        # Prevent automatic controller re-entry.
        self.publish_mode = None
        self._prev_mode = None

        # Reset stateful memories to avoid unsafe recovery transients.
        self.safety_limiter.reset()
        self.policy_controller.reset()
        self.nominal_controller.reset()

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

    # ----------------------------------------------------------------------
    # Synchronized control tick
    # ----------------------------------------------------------------------
    def _tick(self, joint_msg: JointState, imu_msg: ImuState) -> None:
        """Control tick called by TimeSynchronizer.

        This replaces the previous create_timer() based control loop.
        The arrival of synchronized /joint_states and /imu messages becomes
        the controller update trigger.
        """
        now_time = self.get_clock().now()

        # Reset timing if simulation time jumps backwards.
        if now_time < self.last_tick_time:
            self.logger.error("ROS time jumped backwards. Resetting controller states.\r")
            self.reset()
            self.last_tick_time = now_time
            return

        dt_sec = (now_time - self.last_tick_time).nanoseconds * 1e-9
        if dt_sec <= 0.0:
            dt_sec = 1.0  / max(self.control_rate_hz, 1.0)
        self.last_tick_time = now_time

        # Default command buffers
        q_ref = np.zeros(self.num_joints, dtype=np.float32)
        v_ref = np.zeros(self.num_joints, dtype=np.float32)
        tau = np.zeros(self.num_joints, dtype=np.float32)

        # --------------------------------------------------------------
        # Proactive condition check
        # --------------------------------------------------------------
        if self.publish_mode is None:
            q_ref[:] = [0.0, 0.0, 1.09, -1.09, 1.89, -1.89, 0.0, 0.0]
            self._publish_joint_command(q_ref, v_ref, tau)
            return
        
        # --------------------------------------------------------------
        # Mode switch detection
        # --------------------------------------------------------------
        self._switch_mode(self.publish_mode)

        # --------------------------------------------------------------
        # Controller execution
        # --------------------------------------------------------------
        if self.publish_mode == "policy":
            # Command is owned and updated by the controller itself (handle_key).
            joint_torque, q_ref, wheel_v_ref = self.policy_controller.compute(joint_msg,
                                                                            imu_msg,
                                                                            dt_sec)

            v_ref[-2:] = wheel_v_ref

        elif self.publish_mode == "nominal":
            joint_torque, q_ref, _ = self.nominal_controller.compute(joint_msg, 
                                                                   imu_msg, 
                                                                   dt_sec)
            v_ref[-2:] = 0.0

        else:
            joint_torque = tau

        # # --------------------------------------------------------------
        # # Safety limiter
        # # --------------------------------------------------------------
        joint_pos = np.asarray(joint_msg.position, dtype=float).flatten()
        joint_vel = np.asarray(joint_msg.velocity, dtype=float).flatten()
        safe_torque, is_blocked = self.safety_limiter.apply(joint_torque, joint_pos, joint_vel)

        if is_blocked:
            q_ref = np.zeros(self.num_joints, dtype=np.float32)
            v_ref = np.zeros(self.num_joints, dtype=np.float32)
            safe_torque = np.zeros(self.num_joints, dtype=np.float32)

        tau[:] = safe_torque

        # --------------------------------------------------------------
        # Publish command immediately in the same synchronized callback
        # --------------------------------------------------------------
        self._publish_joint_command(q_ref, v_ref, tau)

        self.logger.info(f"Publish Torque : {safe_torque.tolist()}\r", throttle_duration_sec=1.0)

    def _publish_joint_command(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        """Publish command to /joint_command.

        Message semantic:
          position: q_ref
          velocity: v_ref
          effort:   torque
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

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

        self.joint_command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimControllerNode()

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