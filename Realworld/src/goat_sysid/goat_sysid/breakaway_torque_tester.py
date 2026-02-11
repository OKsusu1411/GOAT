#!/usr/bin/env python3
"""goat_sysid: breakaway_torque_tester

이 스크립트는 **정지 상태에서 관절이 처음 움직이기 시작하는 최소 토크**(breakaway torque)를
계단(step) 형태로 토크를 증가시키며 찾습니다.

요청하신 대로, 예전 버전의 `motor_states`(custom msg) 의존성을 제거하고,
`friction_id_node.py`가 쓰는 토픽 구성과 동일하게 동작하도록 수정했습니다.

사용 토픽 (friction_id_node.py와 동일 계열)
  - Sub:  joint_states (sensor_msgs/JointState)  : q, dq
  - Pub:  torque_commands (std_msgs/Float32MultiArray) : tau_cmd (Nm)

주의
  - `torque_commands`는 보통 goat_control 쪽에서도 publish 할 수 있으니,
    실험할 땐 **다른 publisher가 같은 토픽을 동시에 publish 하지 않도록**(컨트롤 노드 정지 등)
    환경을 정리해 주세요.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

JOINT_INDEX = 0


class BreakawayTorqueTester(Node):
    """Breakaway torque 자동 측정 노드."""

    def __init__(self) -> None:
        super().__init__("breakaway_torque_tester")

        # -----------------------------
        # Params (topics / sizes)
        # -----------------------------
        self.declare_parameter("num_joints", 8)
        self.declare_parameter("joint_index", JOINT_INDEX)
        self.declare_parameter("joint_name", "")  # optional
        self.declare_parameter("joint_states_topic", "joint_states")
        self.declare_parameter("torque_commands_topic", "torque_commands")

        self.num_joints = int(self.get_parameter("num_joints").value)
        self.joint_index_param = int(self.get_parameter("joint_index").value)
        self.joint_name_param = str(self.get_parameter("joint_name").value).strip()
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.torque_commands_topic = str(self.get_parameter("torque_commands_topic").value)

        # -----------------------------
        # Params (test profile)
        # -----------------------------
        self.start_torque = float(self.declare_parameter("start_torque", 0.0).value)
        self.torque_step = float(self.declare_parameter("torque_step", 0.01).value)
        self.max_torque = float(self.declare_parameter("max_torque", 4.5).value)
        self.step_duration = float(self.declare_parameter("step_duration", 0.5).value)  # [s]

        # direction: +1 / -1
        direction = float(self.declare_parameter("direction", 1.0).value)
        self.direction = 1.0 if direction >= 0.0 else -1.0

        # 움직임 판정 기준
        self.angle_threshold_deg = float(self.declare_parameter("angle_threshold_deg", 0.5).value)
        self.min_samples_over_threshold = int(
            self.declare_parameter("min_samples_over_threshold", 3).value
        )
        self.angle_threshold_rad = math.radians(self.angle_threshold_deg)

        # control loop
        self.control_frequency = float(self.declare_parameter("control_frequency", 200.0).value)
        self.dt = 1.0 / max(self.control_frequency, 1.0)

        # -----------------------------
        # Runtime state
        # -----------------------------
        self._joint_idx: Optional[int] = None
        self._latest_js: Optional[JointState] = None
        self.have_state = False

        self.state = "WAIT_FOR_STATE"  # WAIT_FOR_STATE -> START_STEP -> HOLD_STEP -> DONE
        self.current_step_torque = self.start_torque
        self.step_start_time: Optional[float] = None
        self.baseline_angle_rad = 0.0
        self.over_threshold_count = 0

        self.break_torque: Optional[float] = None
        self.break_time: Optional[float] = None

        # -----------------------------
        # ROS I/O
        # -----------------------------
        self.create_subscription(JointState, self.joint_states_topic, self._on_joint_state, 10)
        self.torque_pub = self.create_publisher(Float32MultiArray, self.torque_commands_topic, 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            "[BreakawayTester] Ready.\n"
            f"  joint_states_topic: {self.joint_states_topic}\n"
            f"  torque_commands_topic: {self.torque_commands_topic}\n"
            f"  num_joints: {self.num_joints}, joint_index: {self.joint_index_param}, joint_name: '{self.joint_name_param}'\n"
            f"  start_torque: {self.start_torque}, torque_step: {self.torque_step}, max_torque: {self.max_torque}, dir: {self.direction}\n"
            f"  step_duration: {self.step_duration}s, threshold: {self.angle_threshold_deg}deg, min_samples: {self.min_samples_over_threshold}\n"
            f"  control_frequency: {self.control_frequency}Hz"
        )

    # -------------------- callbacks --------------------
    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_js = msg
        self.have_state = True

        if self._joint_idx is None:
            self._joint_idx = self._resolve_joint_index(msg)
            if self._joint_idx is not None:
                self.get_logger().info(f"[BreakawayTester] Resolved joint index: {self._joint_idx}")

    def _resolve_joint_index(self, js: JointState) -> Optional[int]:
        # 1) by name
        if self.joint_name_param and js.name:
            try:
                return list(js.name).index(self.joint_name_param)
            except ValueError:
                self.get_logger().warn(
                    f"[BreakawayTester] joint_name '{self.joint_name_param}' not found in JointState.name. "
                    "Falling back to joint_index param."
                )

        # 2) by index
        if 0 <= self.joint_index_param < self.num_joints:
            return self.joint_index_param

        self.get_logger().error("[BreakawayTester] Invalid joint_index and joint_name not found.")
        return None

    # -------------------- control loop --------------------
    def control_loop(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        # joint state 아직 없으면 토크 0 유지
        if not self.have_state or self._latest_js is None or self._joint_idx is None:
            self.publish_torque_cmd(0.0)
            return

        # 안전: JointState에 position이 없으면 토크 0
        if not self._latest_js.position or self._joint_idx >= len(self._latest_js.position):
            self.publish_torque_cmd(0.0)
            return

        # current angle (rad)
        cur_angle = float(self._latest_js.position[self._joint_idx])

        if self.state == "WAIT_FOR_STATE":
            self.baseline_angle_rad = cur_angle
            self.over_threshold_count = 0
            self.step_start_time = now
            self.current_step_torque = self.start_torque
            self.state = "START_STEP"
            self.get_logger().info(
                f"[BreakawayTester] Start test. Baseline angle(rad)={self.baseline_angle_rad:.6f} "
                f"({math.degrees(self.baseline_angle_rad):.3f} deg)"
            )

        elif self.state == "START_STEP":
            # 새 step 시작
            self.baseline_angle_rad = cur_angle
            self.over_threshold_count = 0
            self.step_start_time = now
            self.state = "HOLD_STEP"
            self.get_logger().info(
                f"[BreakawayTester] Apply torque step: {self.direction * self.current_step_torque:.4f}"
            )

        elif self.state == "HOLD_STEP":
            diff = abs(cur_angle - self.baseline_angle_rad)
            if diff > self.angle_threshold_rad:
                self.over_threshold_count += 1

            # breakaway 판정
            if self.over_threshold_count >= self.min_samples_over_threshold:
                self.break_torque = self.direction * self.current_step_torque
                self.break_time = now
                self.state = "DONE"

                self.get_logger().info(
                    "[BreakawayTester] BREAKAWAY DETECTED! "
                    f"joint={self._joint_idx}, tau_break={self.break_torque:.4f} (torque_commands unit) "
                    f"| diff={diff:.6f} rad ({math.degrees(diff):.3f} deg)"
                )

            # step 유지 끝 -> 다음 step
            elif self.step_start_time is not None and (now - self.step_start_time) >= self.step_duration:
                self.current_step_torque += self.torque_step
                if abs(self.current_step_torque) > abs(self.max_torque):
                    self.get_logger().warn(
                        "[BreakawayTester] Reached max_torque without motion. Stop test for safety."
                    )
                    self.state = "DONE"
                else:
                    self.state = "START_STEP"

        elif self.state == "DONE":
            self.publish_torque_cmd(0.0)
            return

        # publish torque for this step
        tau_cmd_joint = 0.0
        if self.state in ("START_STEP", "HOLD_STEP"):
            tau_cmd_joint = self.direction * self.current_step_torque
        self.publish_torque_cmd(tau_cmd_joint)

    # -------------------- publish --------------------
    def publish_torque_cmd(self, tau_joint: float) -> None:
        """원하는 조인트 하나에만 토크를 인가, 나머지는 0."""
        cmd = np.zeros(self.num_joints, dtype=float)
        if self._joint_idx is not None and 0 <= self._joint_idx < cmd.size:
            cmd[self._joint_idx] = float(tau_joint)

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self.torque_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BreakawayTorqueTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
