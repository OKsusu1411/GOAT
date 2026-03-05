#!/usr/bin/env python3
from __future__ import annotations

import sys
import math
import termios
import tty
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# ---- fixed config ----
NUM_ACTUATORS = 8
JOINT_COUNT = 6  # 0~5
WHEEL_COUNT = 2  # 6~7
STEP = 20.0      # joint: deg, wheel: deg/s
PUBLISH_HZ = 100.0
TOPIC = "goat/actions"


def _read_int_0_7() -> int:
    while True:
        try:
            s = input("Select actuator index (0~7) [0~5 joint, 6~7 wheel] : ").strip()
            v = int(s)
            if 0 <= v < 8:
                return v
        except Exception:
            pass
        print("Invalid. Enter 0~7.")


class RawKeyReader:
    """Keep terminal in raw mode and read 1 char blocking."""
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setraw(self.fd)  # raw mode ON
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)  # restore

    def getch(self) -> str:
        return sys.stdin.read(1)


class SimpleKeyboardPolicy(Node):
    """
    Publish Float32MultiArray(len=8) to goat/action
      - [0:6] joint position [rad]
      - [6:8] wheel speed [rad/s]
    """
    def __init__(self, selected_index: int):
        super().__init__("policy")

        self.publisher_ = self.create_publisher(Float32MultiArray, TOPIC, 10)

        self.selected_index = int(selected_index)

        # internal state (human units)
        self.joint_deg = [0.0] * JOINT_COUNT
        self.wheel_deg_s = [0.0] * WHEEL_COUNT

        # publish timer
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._publish_action)

        # print info
        self.initial_message = (
            f"[Keyboard Policy]\n"
            f"Selected index: {self.selected_index}  (0~5 joint deg, 6~7 wheel deg/s)\n"
            f"Controls: ↑/→ +{STEP}, ↓/← -{STEP}, q quit\n"
            f"Publishing '{TOPIC}' @ {PUBLISH_HZ} Hz"
        )
        print(self.initial_message)

        # key thread
        self._key_thread = threading.Thread(target=self._key_loop, daemon=True)
        self._key_thread.start()

        self._last_print_time = 0.0
        self._dirty = True  # print once

    def _apply_delta(self, sign: float):
        idx = self.selected_index
        if 0 <= idx < JOINT_COUNT:
            self.joint_deg[idx] += sign * STEP
            self._dirty = True
            return
        if idx in (6, 7):
            w = idx - 6
            self.wheel_deg_s[w] += sign * STEP
            self._dirty = True
            return

    def _key_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().error("stdin is not a TTY. Run this in a real terminal.")
            return

        try:
            with RawKeyReader() as kb:
                while rclpy.ok():
                    key = kb.getch()

                    if key in ("q", "Q"):
                        self.get_logger().info("Quit requested.")
                        rclpy.shutdown()
                        break

                    # Arrow keys: ESC [ A/B/C/D
                    if key == "\x1b":
                        k1 = kb.getch()
                        k2 = kb.getch()

                        if k1 == "[":
                            if k2 in ("A", "C"):      # Up / Right
                                self._apply_delta(+1.0)
                            elif k2 in ("B", "D"):    # Down / Left
                                self._apply_delta(-1.0)
        except Exception as e:
            self.get_logger().error(f"Keyboard thread error: {e}")

    def _publish_action(self):
        if not rclpy.ok():
            return

        # Convert to rad / rad/s
        joint_rad = [math.radians(v) for v in self.joint_deg]
        wheel_rad_s = [math.radians(v) for v in self.wheel_deg_s]

        msg = Float32MultiArray()
        msg.data = joint_rad + wheel_rad_s  # len=8
        self.publisher_.publish(msg)

        # print only when changed or every 0.3s
        now = time.time()
        if self._dirty or (now - self._last_print_time) > 0.3:
            self._last_print_time = now
            self._dirty = False
            sys.stdout.write("\033c")
            print(self.initial_message)
            print("joint_deg:", [f"{v:.1f}" for v in self.joint_deg])
            print("wheel_deg/s:", [f"{v:.1f}" for v in self.wheel_deg_s])


def main(args=None):
    # select first (normal terminal mode)
    selected = _read_int_0_7()

    rclpy.init(args=args)
    node = SimpleKeyboardPolicy(selected)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # on exit: publish zeros once (optional)
        if node is not None:
            node.joint_deg = [0.0] * JOINT_COUNT
            node.wheel_deg_s = [0.0] * WHEEL_COUNT
            node._publish_action()
            time.sleep(0.05)

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
