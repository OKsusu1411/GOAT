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
    num_joints: int = 8
    publish_hz: float = 50.0
    action_topic: str = "goat/action"   # must match ControlNode default
    step_deg: float = 20.0              # default step size in degrees
    step_deg_min: float = 1.0
    step_deg_delta: float = 5.0         # w/s to +/- this
    max_abs_deg: float = 180.0          # clamp for safety (you can change)


class RawKeyboard:
    """Non-blocking raw keyboard reader for terminal."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._orig = termios.tcgetattr(self._fd)

    def __enter__(self) -> "RawKeyboard":
        tty.setcbreak(self._fd)  # raw-ish, but still lets ctrl+c work reasonably
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._orig)

    def read_key(self) -> str | None:
        """Return a key string or None if no input.

        Arrow keys come as escape sequences: '\x1b[A' etc.
        """
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None

        ch1 = sys.stdin.read(1)
        if ch1 != "\x1b":
            return ch1

        # Escape sequence (arrow keys)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch2 = sys.stdin.read(1)
        else:
            return ch1

        if ch2 != "[":
            return ch1 + ch2

        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch3 = sys.stdin.read(1)
        else:
            return ch1 + ch2

        return ch1 + ch2 + ch3  # e.g. '\x1b[A'


class PolicyKeyboardTesterNode(Node):
    """Keyboard-based policy action publisher for GoatControlNode.

    Publishes Float32MultiArray (len=16):
      [q_des(rad) x8, dq_des(rad/s) x8]
    """

    def __init__(self) -> None:
        super().__init__("policy_keyboard_tester")

        # Parameters (safe names: no slashes)
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("step_deg", 20.0)
        self.declare_parameter("max_abs_deg", 180.0)

        self.config = KeyboardConfig(
            action_topic=str(self.get_parameter("action_topic").value),
            publish_hz=float(self.get_parameter("publish_hz").value),
            step_deg=float(self.get_parameter("step_deg").value),
            max_abs_deg=float(self.get_parameter("max_abs_deg").value),
        )

        self.publisher = self.create_publisher(Float32MultiArray, self.config.action_topic, 10)

        self.selected_joint_index = 0
        self.desired_position_deg = [0.0] * self.config.num_joints
        self.desired_velocity_rad_per_sec = [0.0] * self.config.num_joints  # keep 0 by default

        period = 1.0 / max(1e-6, self.config.publish_hz)
        self.timer = self.create_timer(period, self._tick)

        self._print_help()

    def _print_help(self) -> None:
        self.get_logger().info(
            "\n[Keyboard Tester]\n"
            f"  Publishing to: /{self.config.action_topic} @ {self.config.publish_hz:.1f} Hz (len=16)\n"
            "  Keys:\n"
            "    0~7 : select joint\n"
            "    [ / ] : prev/next joint\n"
            "    ↑/→ : +step deg (selected joint)\n"
            "    ↓/← : -step deg (selected joint)\n"
            "    w/s : step +5 / -5 deg (min 1 deg)\n"
            "    r   : reset all targets to 0\n"
            "    q   : quit\n"
        )

    def _clamp_deg(self, deg_value: float) -> float:
        limit = abs(self.config.max_abs_deg)
        if limit <= 0.0:
            return deg_value
        return max(-limit, min(limit, deg_value))

    def _handle_key(self, key: str) -> bool:
        """Handle key. Return False to quit."""
        if key is None:
            return True

        # Quit
        if key in ("q", "Q"):
            self.get_logger().info("Quit requested.")
            return False

        # Reset
        if key in ("r", "R"):
            self.desired_position_deg = [0.0] * self.config.num_joints
            self.get_logger().info("Reset all desired positions to 0 deg.")
            return True

        # Step size adjust
        if key in ("w", "W"):
            self.config.step_deg = max(self.config.step_deg_min, self.config.step_deg + self.config.step_deg_delta)
            self.get_logger().info(f"Step size: {self.config.step_deg:.1f} deg")
            return True
        if key in ("s", "S"):
            self.config.step_deg = max(self.config.step_deg_min, self.config.step_deg - self.config.step_deg_delta)
            self.get_logger().info(f"Step size: {self.config.step_deg:.1f} deg")
            return True

        # Select joint by number
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < self.config.num_joints:
                self.selected_joint_index = idx
                self.get_logger().info(f"Selected joint: {self.selected_joint_index}")
            return True

        # Select joint by bracket
        if key == "[":
            self.selected_joint_index = (self.selected_joint_index - 1) % self.config.num_joints
            self.get_logger().info(f"Selected joint: {self.selected_joint_index}")
            return True
        if key == "]":
            self.selected_joint_index = (self.selected_joint_index + 1) % self.config.num_joints
            self.get_logger().info(f"Selected joint: {self.selected_joint_index}")
            return True

        # Arrows
        if key in ("\x1b[A", "\x1b[C"):  # Up / Right
            j = self.selected_joint_index
            self.desired_position_deg[j] = self._clamp_deg(self.desired_position_deg[j] + self.config.step_deg)
            self.get_logger().info(f"j{j} -> {self.desired_position_deg[j]:.1f} deg")
            return True
        if key in ("\x1b[B", "\x1b[D"):  # Down / Left
            j = self.selected_joint_index
            self.desired_position_deg[j] = self._clamp_deg(self.desired_position_deg[j] - self.config.step_deg)
            self.get_logger().info(f"j{j} -> {self.desired_position_deg[j]:.1f} deg")
            return True

        return True

    def _publish_action(self) -> None:
        # Convert deg -> rad
        desired_position_rad = [math.radians(self._clamp_deg(d)) for d in self.desired_position_deg]
        desired_velocity_rad_per_sec = list(self.desired_velocity_rad_per_sec)

        # Build action length 16
        action_vector = desired_position_rad + desired_velocity_rad_per_sec

        msg = Float32MultiArray()
        msg.data = [float(x) for x in action_vector]
        self.publisher.publish(msg)

    def _tick(self) -> None:
        # Read and handle at most a few keys per tick (avoid starving publish)
        for _ in range(5):
            key = self._keyboard.read_key()
            if key is None:
                break
            keep_running = self._handle_key(key)
            if not keep_running:
                rclpy.shutdown()
                return

        self._publish_action()

    def attach_keyboard(self, keyboard: RawKeyboard) -> None:
        self._keyboard = keyboard


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
