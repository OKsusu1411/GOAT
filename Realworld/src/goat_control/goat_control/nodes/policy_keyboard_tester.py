from __future__ import annotations

import sys
import time
import math
import select
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


@dataclass
class KeyboardConfig:
    num_joints: int = 8
    step_deg: float = 20.0
    publish_rate_hz: float = 30.0
    action_topic: str = "policy_action"


class _RawKeyboard:
    """Non-blocking raw keyboard reader (Linux terminal)."""

    def __init__(self):
        self._orig = termios.tcgetattr(sys.stdin)

    def __enter__(self):
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig)

    @staticmethod
    def read_key_nonblocking(timeout_sec: float = 0.0) -> Optional[str]:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout_sec)
        if not rlist:
            return None
        ch1 = sys.stdin.read(1)
        if ch1 != "\x1b":  # not ESC
            return ch1
        # ESC sequence (arrows): ESC [ A/B/C/D
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch2 = sys.stdin.read(1)
            if ch2 == "[" and select.select([sys.stdin], [], [], 0.0)[0]:
                ch3 = sys.stdin.read(1)
                return f"\x1b[{ch3}"
        return ch1


class PolicyKeyboardTester(Node):
    """
    Publishes test policy action (desired joint positions) for ControlNode.

    Output:
      topic: policy_action (Float32MultiArray)
      data: q_desired [rad] length = num_joints

    Interaction:
      - At start: choose joint id, mode
      - Mode 1 (manual): type number (deg) then Enter
      - Mode 2 (arrows): Up/Right increase +step_deg, Down/Left decrease -step_deg
      - 's' : select joint id again
      - 'm' : change mode again
      - 'r' : reset all targets to 0
      - 'q' : quit
    """

    def __init__(self):
        super().__init__("policy_keyboard_tester")

        self.declare_parameter("num_joints", 8)
        self.declare_parameter("step_deg", 20.0)
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("action_topic", "policy_action")

        self.cfg = KeyboardConfig(
            num_joints=int(self.get_parameter("num_joints").value),
            step_deg=float(self.get_parameter("step_deg").value),
            publish_rate_hz=float(self.get_parameter("publish_rate_hz").value),
            action_topic=str(self.get_parameter("action_topic").value),
        )

        self.publisher = self.create_publisher(Float32MultiArray, self.cfg.action_topic, 10)

        self.target_joint_id: int = 0
        self.mode: str = "arrows"  # "manual" or "arrows"
        self.target_rad: List[float] = [0.0] * self.cfg.num_joints

        self._next_prompt_time = 0.0
        self._print_help()

        # blocking prompts (outside timer)
        self._select_joint_id()
        self._select_mode()

        period = 1.0 / max(self.cfg.publish_rate_hz, 1.0)
        self.create_timer(period, self._tick)

    def _print_help(self):
        self.get_logger().info(
            "\n[PolicyKeyboardTester]\n"
            f"- Publishing '{self.cfg.action_topic}' as desired joint positions [rad], length={self.cfg.num_joints}\n"
            "- Keys:\n"
            "  Arrow keys: +/- step (in arrows mode)\n"
            "  s: select joint id\n"
            "  m: select mode (manual/arrows)\n"
            "  r: reset all targets to 0\n"
            "  q: quit\n"
        )

    def _select_joint_id(self):
        while True:
            try:
                text = input(f"Select target joint id [0..{self.cfg.num_joints-1}]: ").strip()
                joint_id = int(text)
                if 0 <= joint_id < self.cfg.num_joints:
                    self.target_joint_id = joint_id
                    self.get_logger().info(f"Target joint id set to {self.target_joint_id}")
                    return
                print("Out of range.")
            except (ValueError, EOFError):
                print("Invalid input. Try again.")

    def _select_mode(self):
        while True:
            text = input("Select mode: (1) manual deg input, (2) arrow +/- step : ").strip()
            if text == "1":
                self.mode = "manual"
                self.get_logger().info("Mode = manual (type degrees and press Enter)")
                return
            if text == "2":
                self.mode = "arrows"
                self.get_logger().info(f"Mode = arrows (step = {self.cfg.step_deg} deg)")
                return
            print("Please type 1 or 2.")

    def _manual_prompt_if_needed(self):
        # rate-limit prompts to avoid spamming
        now = time.time()
        if now < self._next_prompt_time:
            return
        self._next_prompt_time = now + 0.1

        # manual mode: only prompt when user presses Enter? -> we can't detect Enter cleanly here.
        # Instead, we prompt occasionally and allow user to type a line.
        try:
            text = input(f"[manual] joint {self.target_joint_id} deg = ").strip()
        except EOFError:
            return
        if text == "":
            return
        if text.lower() == "s":
            self._select_joint_id()
            return
        if text.lower() == "m":
            self._select_mode()
            return
        if text.lower() == "r":
            self.target_rad = [0.0] * self.cfg.num_joints
            self.get_logger().info("Reset all targets to 0.")
            return
        if text.lower() == "q":
            raise KeyboardInterrupt

        try:
            deg = float(text)
            self.target_rad[self.target_joint_id] = math.radians(deg)
            self.get_logger().info(
                f"Set joint {self.target_joint_id} = {deg:.2f} deg ({self.target_rad[self.target_joint_id]:.3f} rad)"
            )
        except ValueError:
            self.get_logger().warn("Invalid number.")

    def _apply_arrow_key(self, key: str):
        step_rad = math.radians(self.cfg.step_deg)
        if key in ("\x1b[A", "\x1b[C"):  # Up or Right
            self.target_rad[self.target_joint_id] += step_rad
        elif key in ("\x1b[B", "\x1b[D"):  # Down or Left
            self.target_rad[self.target_joint_id] -= step_rad

    def _tick(self):
        # Publish at fixed rate
        msg = Float32MultiArray()
        msg.data = [float(x) for x in self.target_rad]
        self.publisher.publish(msg)

        # Input handling
        if self.mode == "manual":
            # Manual uses blocking input, so do it in a controlled way
            self._manual_prompt_if_needed()
            return

        # Arrows mode: non-blocking key read
        key = _RawKeyboard.read_key_nonblocking(timeout_sec=0.0)
        if key is None:
            return

        if key in ("\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"):
            self._apply_arrow_key(key)
            current_deg = math.degrees(self.target_rad[self.target_joint_id])
            self.get_logger().info(f"[arrows] joint {self.target_joint_id} = {current_deg:.2f} deg")
            return

        if key == "s":
            # temporarily restore cooked input for blocking prompt
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
            self._select_joint_id()
            return
        if key == "m":
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
            self._select_mode()
            return
        if key == "r":
            self.target_rad = [0.0] * self.cfg.num_joints
            self.get_logger().info("Reset all targets to 0.")
            return
        if key == "q":
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = PolicyKeyboardTester()

    # Raw keyboard context for arrows mode
    try:
        with _RawKeyboard():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
