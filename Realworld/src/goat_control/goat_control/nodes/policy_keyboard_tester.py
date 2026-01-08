#!/usr/bin/env python3
from __future__ import annotations

import sys
import math
import termios
import tty
import select
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


@dataclass
class KeyboardConfig:
    # GOAT: 8 actuators total
    #  - 0~5: joints (position target, rad)
    #  - 6~7: wheels (speed target, rad/s)
    num_actuators: int = 8
    joint_count: int = 6
    wheel_count: int = 2

    # Publish fast enough to avoid ControlNode action watchdog (e.g., 0.05s)
    publish_hz: float = 50.0

    # Default action topic (must match ControlNode subscription)
    action_topic: str = "goat/action"

    # Joint position step (deg)
    joint_step_deg: float = 20.0
    joint_step_deg_min: float = 1.0
    joint_step_deg_delta: float = 5.0  # W/S

    # Wheel speed step (deg/s)
    wheel_step_deg_per_sec: float = 60.0
    wheel_step_deg_per_sec_min: float = 1.0
    wheel_step_deg_per_sec_delta: float = 10.0  # E/D

    # Simple clamps for safer testing
    max_abs_joint_deg: float = 180.0
    max_abs_wheel_deg_per_sec: float = 720.0


class RawKeyboard:
    """Non-blocking raw keyboard reader for terminal."""

    def __init__(self) -> None:
        self._file_descriptor = sys.stdin.fileno()
        self._original_terminal_settings = termios.tcgetattr(self._file_descriptor)

    def __enter__(self) -> "RawKeyboard":
        tty.setcbreak(self._file_descriptor)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        termios.tcsetattr(
            self._file_descriptor, termios.TCSADRAIN, self._original_terminal_settings
        )

    def read_key(self) -> str | None:
        """Return a key string or None if no input.

        Arrow keys are escape sequences:
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

        # Escape sequence
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
    """Keyboard-based action publisher (replaces policy node for testing).

    Publishes Float32MultiArray on goat/action with layout(dim/stride) filled.

    Action format (len=8):
      - data[0:6] : desired joint position [rad]   (j0~j5)
      - data[6:8] : desired wheel speed [rad/s]    (wheel_l, wheel_r)
    """

    def __init__(self) -> None:
        # Name it "policy" to be drop-in compatible with your legacy policy node
        super().__init__("policy")

        # Keep parameters but default to goat/action like your policy_node.py
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

        self.action_publisher = self.create_publisher(
            Float32MultiArray, self.config.action_topic, 10
        )

        # Selected actuator index: 0~7
        self.selected_index = 0

        # Editable targets (human units)
        self.desired_joint_deg = [0.0] * self.config.joint_count              # j0~j5
        self.desired_wheel_deg_per_sec = [0.0] * self.config.wheel_count      # wheel_l, wheel_r

        period_sec = 1.0 / max(1e-6, self.config.publish_hz)
        self.timer = self.create_timer(period_sec, self._tick)

        self._keyboard: RawKeyboard | None = None
        self._print_help()

        self.get_logger().info(
            f"Keyboard tester started: publishing '{self.config.action_topic}' @ {self.config.publish_hz:.1f} Hz (len=8)"
        )

    def attach_keyboard(self, keyboard: RawKeyboard) -> None:
        self._keyboard = keyboard

    def _print_help(self) -> None:
        self.get_logger().info(
            "\n[Policy Keyboard Tester]\n"
            "Action(len=8): [j0..j5 position(rad), wheel_l/wheel_r speed(rad/s)]\n"
            "\nSelection:\n"
            "  0~5 : select joint\n"
            "  6   : select wheel_l\n"
            "  7   : select wheel_r\n"
            "  [ / ] : prev/next selection\n"
            "\nControl:\n"
            "  ↑/→ : +step\n"
            "  ↓/← : -step\n"
            "  W/S : joint step + / -\n"
            "  E/D : wheel step + / -\n"
            "\nOther:\n"
            "  r : reset all targets\n"
            "  q : quit\n"
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

        if key in ("q", "Q"):
            self.get_logger().info("Quit requested.")
            return False

        if key in ("r", "R"):
            self.desired_joint_deg = [0.0] * self.config.joint_count
            self.desired_wheel_deg_per_sec = [0.0] * self.config.wheel_count
            self.get_logger().info("Reset: joints=0 deg, wheels=0 deg/s.")
            return True

        if key == "[":
            self.selected_index = (self.selected_index - 1) % self.config.num_actuators
            self.get_logger().info(f"Selected index: {self.selected_index}")
            return True
        if key == "]":
            self.selected_index = (self.selected_index + 1) % self.config.num_actuators
            self.get_logger().info(f"Selected index: {self.selected_index}")
            return True

        if key.isdigit():
            idx = int(key)
            if 0 <= idx < self.config.num_actuators:
                self.selected_index = idx
                self.get_logger().info(f"Selected index: {self.selected_index}")
            return True

        # step size tuning
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

        # arrows
        if key in ("\x1b[A", "\x1b[C"):  # Up / Right
            self._apply_delta(+1.0)
            return True
        if key in ("\x1b[B", "\x1b[D"):  # Down / Left
            self._apply_delta(-1.0)
            return True

        return True

    def _apply_delta(self, direction_sign: float) -> None:
        idx = int(self.selected_index)

        if 0 <= idx < self.config.joint_count:
            new_deg = self.desired_joint_deg[idx] + direction_sign * self.config.joint_step_deg
            new_deg = self._clamp_joint_deg(new_deg)
            self.desired_joint_deg[idx] = new_deg
            self.get_logger().info(f"Joint j{idx}: {new_deg:.1f} deg")
            return

        if idx in (6, 7):
            wheel_array_index = idx - 6
            new_deg_per_sec = (
                self.desired_wheel_deg_per_sec[wheel_array_index]
                + direction_sign * self.config.wheel_step_deg_per_sec
            )
            new_deg_per_sec = self._clamp_wheel_deg_per_sec(new_deg_per_sec)
            self.desired_wheel_deg_per_sec[wheel_array_index] = new_deg_per_sec
            wheel_name = "wheel_l" if idx == 6 else "wheel_r"
            self.get_logger().info(f"{wheel_name}: {new_deg_per_sec:.1f} deg/s")
            return

    # ---------------------------
    # Publish
    # ---------------------------
    @staticmethod
    def _numpy_to_multiarray(array_value: np.ndarray) -> Float32MultiArray:
        """Same layout style as your policy_node.py."""
        array_value = np.asarray(array_value, dtype=np.float32)

        msg = Float32MultiArray()
        msg.layout.data_offset = 0
        msg.layout.dim = []

        shape = array_value.shape
        current_stride = 1
        strides = []
        for size in reversed(shape):
            strides.insert(0, current_stride)
            current_stride *= int(size)

        for dim_index, dim_size in enumerate(shape):
            dim = MultiArrayDimension()
            dim.label = f"dim_{dim_index}"
            dim.size = int(dim_size)
            dim.stride = int(strides[dim_index])
            msg.layout.dim.append(dim)

        msg.data = array_value.flatten().tolist()
        return msg

    def _publish_action(self) -> None:
        # Build len=8 vector:
        #   [0:6] joint position (rad), [6:8] wheel speed (rad/s)
        joint_position_rad = [
            float(math.radians(self._clamp_joint_deg(deg_value)))
            for deg_value in self.desired_joint_deg
        ]
        wheel_speed_rad_per_sec = [
            float(math.radians(self._clamp_wheel_deg_per_sec(deg_per_sec_value)))
            for deg_per_sec_value in self.desired_wheel_deg_per_sec
        ]

        action_vector = np.asarray(joint_position_rad + wheel_speed_rad_per_sec, dtype=np.float32)
        if action_vector.size != 8:
            self.get_logger().error(f"Internal error: action len={action_vector.size} (expected 8)")
            return

        action_msg = self._numpy_to_multiarray(action_vector)
        self.action_publisher.publish(action_msg)

    def _tick(self) -> None:
        # process keys (non-blocking), but publish continuously for watchdog safety
        if self._keyboard is not None:
            for _ in range(8):
                key = self._keyboard.read_key()
                if key is None:
                    break
                keep_running = self._handle_key(key)
                if not keep_running:
                    rclpy.shutdown()
                    return

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
