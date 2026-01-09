#!/usr/bin/env python3
from __future__ import annotations

import sys
import math
import termios
import tty
import select

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


# --- fixed step (as you asked) ---
JOINT_STEP_DEG = 20.0          # joint angle step (deg)
WHEEL_STEP_DEG_PER_SEC = 20.0  # wheel speed step (deg/s)

PUBLISH_HZ = 50.0
ACTION_TOPIC_DEFAULT = "goat/action"

NUM_ACTUATORS = 8
JOINT_COUNT = 6  # 0~5
WHEEL_COUNT = 2  # 6~7


class RawKeyboard:
    """Non-blocking raw keyboard reader for terminal (arrow keys supported)."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY. Run this in a real terminal.")
        self._fd = sys.stdin.fileno()
        self._orig = termios.tcgetattr(self._fd)

    def __enter__(self) -> "RawKeyboard":
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._orig)

    def read_key(self) -> str | None:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None

        ch1 = sys.stdin.read(1)
        if ch1 != "\x1b":
            return ch1

        # Escape sequence for arrows: \x1b [ A/B/C/D
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return ch1
        ch2 = sys.stdin.read(1)
        if ch2 != "[":
            return ch1 + ch2

        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return ch1 + ch2
        ch3 = sys.stdin.read(1)
        return ch1 + ch2 + ch3


class SimpleKeyboardPolicy(Node):
    """
    Publishes Float32MultiArray on goat/action (len=8)
      - data[0:6] : joint position [rad]
      - data[6:8] : wheel speed [rad/s]
    """

    def __init__(self, selected_index: int) -> None:
        super().__init__("policy")

        self.declare_parameter("action_topic", ACTION_TOPIC_DEFAULT)
        self.declare_parameter("publish_hz", float(PUBLISH_HZ))

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)

        self.selected_index = int(selected_index)

        # internal targets in human units
        self.joint_deg = [0.0] * JOINT_COUNT
        self.wheel_deg_s = [0.0] * WHEEL_COUNT

        self._keyboard: RawKeyboard | None = None

        self.get_logger().info(
            f"[Keyboard Policy] selected={self.selected_index}  "
            f"(0~5 joint deg, 6~7 wheel deg/s)\n"
            f"  arrows: +/-20   |   q: quit\n"
            f"Publishing '{self.action_topic}' @ {self.publish_hz:.1f} Hz"
        )

        period = 1.0 / max(1e-6, self.publish_hz)
        self.timer = self.create_timer(period, self._tick)

    def attach_keyboard(self, kb: RawKeyboard) -> None:
        self._keyboard = kb

    @staticmethod
    def _numpy_to_multiarray(array_value: np.ndarray) -> Float32MultiArray:
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

    def _apply_delta(self, sign: float) -> None:
        idx = self.selected_index

        if 0 <= idx < JOINT_COUNT:
            self.joint_deg[idx] += sign * JOINT_STEP_DEG
            self.get_logger().info(f"j{idx} = {self.joint_deg[idx]:.1f} deg")
            return

        if idx in (6, 7):
            w = idx - 6
            self.wheel_deg_s[w] += sign * WHEEL_STEP_DEG_PER_SEC
            name = "wheel_l" if idx == 6 else "wheel_r"
            self.get_logger().info(f"{name} = {self.wheel_deg_s[w]:.1f} deg/s")
            return

    def _handle_key(self, key: str | None) -> bool:
        if key is None:
            return True

        if key in ("q", "Q"):
            self.get_logger().info("Quit.")
            return False

        if key in ("\x1b[A", "\x1b[C"):  # Up / Right
            self._apply_delta(+1.0)
            return True

        if key in ("\x1b[B", "\x1b[D"):  # Down / Left
            self._apply_delta(-1.0)
            return True

        return True

    def _publish(self) -> None:
        joint_rad = [math.radians(v) for v in self.joint_deg]
        wheel_rad_s = [math.radians(v) for v in self.wheel_deg_s]

        action = np.asarray(joint_rad + wheel_rad_s, dtype=np.float32)
        msg = self._numpy_to_multiarray(action)
        self.pub.publish(msg)

    def _tick(self) -> None:
        if self._keyboard is not None:
            # drain a few keys per tick (non-blocking)
            for _ in range(8):
                key = self._keyboard.read_key()
                if key is None:
                    break
                if not self._handle_key(key):
                    rclpy.shutdown()
                    return

        # Always publish (watchdog-safe)
        self._publish()


def _select_actuator_blocking() -> int:
    """Select actuator index (0~7) in normal terminal mode before raw keyboard starts."""
    while True:
        try:
            s = input("Select actuator index (0~7) [0~5 joint, 6~7 wheel] : ").strip()
            idx = int(s)
            if 0 <= idx < NUM_ACTUATORS:
                return idx
        except Exception:
            pass
        print("Invalid. Enter an integer 0~7.")


def main(args=None) -> None:
    # 1) choose joint/wheel first (blocking, normal input)
    selected = _select_actuator_blocking()

    # 2) ROS2 start
    rclpy.init(args=args)
    node = SimpleKeyboardPolicy(selected_index=selected)

    try:
        # 3) raw keyboard control loop
        with RawKeyboard() as kb:
            node.attach_keyboard(kb)
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
