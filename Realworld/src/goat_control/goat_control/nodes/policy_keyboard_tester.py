#!/usr/bin/env python3
from __future__ import annotations

import sys
import math
import termios
import tty
import select
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


@dataclass
class KeyboardConfig:
    # GOAT has 8 actuators total:
    #   0~5: joints (position control)
    #   6~7: wheels (velocity control)
    num_actuators: int = 8
    joint_count: int = 6
    wheel_count: int = 2

    # Publish fast enough to avoid ControlNode action watchdog (default 0.05s)
    publish_hz: float = 50.0

    # Must match ControlNode "policy_action" topic (default: goat/action)
    action_topic: str = "goat/action"

    # Joint step size in degrees (converted to rad before publishing)
    joint_step_deg: float = 0.0
    joint_step_deg_min: float = -50.0
    joint_step_deg_delta: float = 5.0  # W/S to +/- this

    # Wheel speed step in deg/s (converted to rad/s before publishing)
    wheel_step_deg_per_sec: float = 0.0
    wheel_step_deg_per_sec_min: float = -50.0
    wheel_step_deg_per_sec_delta: float = 10.0  # E/D to +/- this

    # Simple clamps for safer testing
    max_abs_joint_deg: float = 180.0
    max_abs_wheel_deg_per_sec: float = 720.0  # 2 rev/s = 720 deg/s


class RawKeyboard:
    """Non-blocking raw keyboard reader for terminal."""

    def __init__(self) -> None:
        self._file_descriptor = sys.stdin.fileno()
        self._original_terminal_settings = termios.tcgetattr(self._file_descriptor)

    def __enter__(self) -> "RawKeyboard":
        # cbreak: immediate key read without waiting for Enter
        tty.setcbreak(self._file_descriptor)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        termios.tcsetattr(
            self._file_descriptor, termios.TCSADRAIN, self._original_terminal_settings
        )

    def read_key(self) -> str | None:
        """Return a key string or None if no input.

        Arrow keys come as escape sequences:
          - Up:    '\\x1b[A'
          - Down:  '\\x1b[B'
          - Right: '\\x1b[C'
          - Left:  '\\x1b[D'
        """
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None

        first_char = sys.stdin.read(1)
        if first_char != "\x1b":
            return first_char

        # Escape sequence handling
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return first_char
        second_char = sys.stdin.read(1)

        if second_char != "[":
            return first_char + second_char

        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return first_char + second_char
        third_char = sys.stdin.read(1)

        return first_char + second_char + third_char


class PolicyKeyboardTesterNode(Node):
    """Keyboard-based action publisher for GoatControlNode.

    Action format (Float32MultiArray, len=8):
      - action[0:6] : desired joint position [rad]   (j0~j5)
      - action[6:8] : desired wheel speed [rad/s]    (wheel_l, wheel_r)

    Note:
      GoatControlNode has an action timeout (default 0.05s).
      Therefore this node publishes continuously at publish_hz (default 50Hz).
    """

    def __init__(self) -> None:
        super().__init__("policy_keyboard_tester")

        # Parameters (use safe parameter names without '/')
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("joint_step_deg", 20.0)
        self.declare_parameter("wheel_step_deg_per_sec", 60.0)

        self.config = KeyboardConfig(
            action_topic=str(self.get_parameter("action_topic").value),
            publish_hz=float(self.get_parameter("publish_hz").value),
            joint_step_deg=float(self.get_parameter("joint_step_deg").value),
            wheel_step_deg_per_sec=float(self.get_parameter("wheel_step_deg_per_sec").value),
        )

        # Publisher
        self.action_publisher = self.create_publisher(
            Float32MultiArray, self.config.action_topic, 10
        )

        # Selected actuator index:
        #   0~5 => joint selection
        #   6   => wheel_l
        #   7   => wheel_r
        self.selected_index = 0

        # Desired targets in "human-friendly units" for editing
        self.desired_joint_deg = [0.0] * self.config.joint_count           # 0~5
        self.desired_wheel_deg_per_sec = [0.0] * self.config.wheel_count   # [wheel_l, wheel_r]

        # Periodic publish timer (non-blocking keyboard handled inside tick)
        period_sec = 1.0 / max(1e-6, self.config.publish_hz)
        self.timer = self.create_timer(period_sec, self._tick)

        self._print_help()

    def attach_keyboard(self, keyboard: RawKeyboard) -> None:
        self._keyboard = keyboard

    def _print_help(self) -> None:
        self.get_logger().info(
            "\n[Policy Keyboard Tester]\n"
            f"  Publishing: /{self.config.action_topic} @ {self.config.publish_hz:.1f} Hz\n"
            "  Action format len=8: [j0..j5 position(rad), wheel_l/wheel_r speed(rad/s)]\n"
            "\n  Selection:\n"
            "    0~5 : select joint\n"
            "    6   : select wheel_l\n"
            "    7   : select wheel_r\n"
            "    [ / ] : prev/next selection (0~7)\n"
            "\n  Joint control (selected 0~5):\n"
            "    ↑/→ : +joint_step_deg\n"
            "    ↓/← : -joint_step_deg\n"
            "    W/S : joint_step_deg +5 / -5 (min 1)\n"
            "\n  Wheel control (selected 6~7):\n"
            "    ↑/→ : +wheel_step_deg_per_sec\n"
            "    ↓/← : -wheel_step_deg_per_sec\n"
            "    E/D : wheel_step_deg_per_sec +10 / -10 (min 1)\n"
            "\n  Common:\n"
            "    r : reset all targets\n"
            "    q : quit\n"
        )

    # ---------------------------
    # Clamp helpers
    # ---------------------------
    def _clamp_joint_deg(self, joint_deg: float) -> float:
        limit = abs(self.config.max_abs_joint_deg)
        return max(-limit, min(limit, joint_deg))

    def _clamp_wheel_deg_per_sec(self, wheel_deg_per_sec: float) -> float:
        limit = abs(self.config.max_abs_wheel_deg_per_sec)
        return max(-limit, min(limit, wheel_deg_per_sec))

    # ---------------------------
    # Key handling
    # ---------------------------
    def _handle_key(self, key: str | None) -> bool:
        """Return False to quit."""
        if key is None:
            return True

        # Quit
        if key in ("q", "Q"):
            self.get_logger().info("Quit requested.")
            return False

        # Reset
        if key in ("r", "R"):
            self.desired_joint_deg = [0.0] * self.config.joint_count
            self.desired_wheel_deg_per_sec = [0.0] * self.config.wheel_count
            self.get_logger().info("Reset: all joint positions=0 deg, wheel speeds=0 deg/s.")
            return True

        # Select index directly (0~7)
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < self.config.num_actuators:
                self.selected_index = idx
                self.get_logger().info(f"Selected index: {self.selected_index}")
            return True

        # Select index by brackets
        if key == "[":
            self.selected_index = (self.selected_index - 1) % self.config.num_actuators
            self.get_logger().info(f"Selected index: {self.selected_index}")
            return True
        if key == "]":
            self.selected_index = (self.selected_index + 1) % self.config.num_actuators
            self.get_logger().info(f"Selected index: {self.selected_index}")
            return True

        # Adjust joint step size
        if key in ("w", "W"):
            self.config.joint_step_deg = max(
                self.config.joint_step_deg_min,
                self.config.joint_step_deg + self.config.joint_step_deg_delta,
            )
            self.get_logger().info(f"Joint step: {self.config.joint_step_deg:.1f} deg")
            return True
        if key in ("s", "S"):
            self.config.joint_step_deg = max(
                self.config.joint_step_deg_min,
                self.config.joint_step_deg - self.config.joint_step_deg_delta,
            )
            self.get_logger().info(f"Joint step: {self.config.joint_step_deg:.1f} deg")
            return True

        # Adjust wheel step size
        if key in ("e", "E"):
            self.config.wheel_step_deg_per_sec = max(
                self.config.wheel_step_deg_per_sec_min,
                self.config.wheel_step_deg_per_sec + self.config.wheel_step_deg_per_sec_delta,
            )
            self.get_logger().info(f"Wheel step: {self.config.wheel_step_deg_per_sec:.1f} deg/s")
            return True
        if key in ("d", "D"):
            self.config.wheel_step_deg_per_sec = max(
                self.config.wheel_step_deg_per_sec_min,
                self.config.wheel_step_deg_per_sec - self.config.wheel_step_deg_per_sec_delta,
            )
            self.get_logger().info(f"Wheel step: {self.config.wheel_step_deg_per_sec:.1f} deg/s")
            return True

        # Apply arrows to selected target
        if key in ("\x1b[A", "\x1b[C"):  # Up / Right
            self._apply_delta(+1.0)
            return True
        if key in ("\x1b[B", "\x1b[D"):  # Down / Left
            self._apply_delta(-1.0)
            return True

        return True

    def _apply_delta(self, direction_sign: float) -> None:
        """Apply +/- step to either joint position (deg) or wheel speed (deg/s)."""
        idx = int(self.selected_index)

        # Joint indices 0~5
        if 0 <= idx < self.config.joint_count:
            new_value_deg = self.desired_joint_deg[idx] + direction_sign * self.config.joint_step_deg
            new_value_deg = self._clamp_joint_deg(new_value_deg)
            self.desired_joint_deg[idx] = new_value_deg
            self.get_logger().info(f"Joint j{idx}: {new_value_deg:.1f} deg")
            return

        # Wheel indices 6~7 -> map to wheel array [0,1]
        if idx in (6, 7):
            wheel_array_index = idx - 6
            new_speed_deg_per_sec = (
                self.desired_wheel_deg_per_sec[wheel_array_index]
                + direction_sign * self.config.wheel_step_deg_per_sec
            )
            new_speed_deg_per_sec = self._clamp_wheel_deg_per_sec(new_speed_deg_per_sec)
            self.desired_wheel_deg_per_sec[wheel_array_index] = new_speed_deg_per_sec
            wheel_name = "wheel_l" if idx == 6 else "wheel_r"
            self.get_logger().info(f"{wheel_name}: {new_speed_deg_per_sec:.1f} deg/s")
            return

    # ---------------------------
    # Publish
    # ---------------------------
    def _publish_action(self) -> None:
        # Build action len=8:
        # - [0:6] joint positions in rad
        # - [6:8] wheel speeds in rad/s
        joint_position_rad = [
            float(math.radians(self._clamp_joint_deg(deg_value)))
            for deg_value in self.desired_joint_deg
        ]
        wheel_speed_rad_per_sec = [
            float(math.radians(self._clamp_wheel_deg_per_sec(deg_per_sec_value)))
            for deg_per_sec_value in self.desired_wheel_deg_per_sec
        ]

        action_vector = joint_position_rad + wheel_speed_rad_per_sec
        if len(action_vector) != 8:
            # Should never happen, but keep safe.
            self.get_logger().error(f"Internal error: action length is {len(action_vector)} (expected 8).")
            return

        msg = Float32MultiArray()
        msg.data = action_vector
        self.action_publisher.publish(msg)

    def _tick(self) -> None:
        # Process multiple keys per tick, but keep publishing no matter what.
        for _ in range(8):
            key = self._keyboard.read_key()
            if key is None:
                break
            keep_running = self._handle_key(key)
            if not keep_running:
                rclpy.shutdown()
                return

        # Continuous publish for watchdog safety
        self._publish_action()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyKeyboardTesterNode()

    try:
        with RawKeyboard() as keyboard:
            node.attach_keyboard(keyboard)
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
