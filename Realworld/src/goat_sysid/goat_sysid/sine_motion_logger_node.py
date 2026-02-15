# goat_sysid/goat_sysid/sine_motion_logger_node.py
'''
ros2 run goat_sysid sine_motion_logger_node --ros-args \
  -p num_joints:=0 \
  -p excite_indices:="[0, 1, 2, 3, 4, 5]" \
  -p amplitude_deg:=15.0 \
  -p frequency_hz:=0.5 \
  -p publish_rate_hz:=200.0 \
  -p duration_sec:=20.0 \
  -p settle_sec:=0.5 \
  -p log_dir:="/home/heachanlee/GOAT/GOAT/Realworld/logs" \
  -p log_filename:="joint_0_sine.csv"

'''
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


@dataclass
class LatestBuffers:
    joint_state: Optional[JointState] = None
    torque_cmd: Optional[np.ndarray] = None
    have_joint_state: bool = False
    have_torque: bool = False


class SineMotionLoggerNode(Node):
    """
    Publish:
      - goat/action (Float32MultiArray): desired joint positions [rad] length=num_joints

    Subscribe:
      - joint_states (sensor_msgs/JointState): measured position/velocity
      - goat/torque_commands (Float32MultiArray): commanded torque [Nm] length=num_joints

    Log CSV:
      - t_sec
      - for each joint i:
          q_ref_i(rad), q_meas_i(rad), dq_meas_i(rad/s), tau_cmd_i(Nm)
    """

    def __init__(self) -> None:
        super().__init__("sine_motion_logger_node")

        # -----------------------------
        # Parameters (topics / sizes)
        # -----------------------------
        self.declare_parameter("num_joints", 8)
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("joint_states_topic", "joint_states")
        self.declare_parameter("torque_commands_topic", "goat/torque_commands")

        # -----------------------------
        # Parameters (excitation)
        # -----------------------------
        # excite_indices: 사인파를 줄 조인트 인덱스들
        #   예: [0,1,2,3,4,5]  (다리 6개만), 또는 [6,7] (바퀴만), 또는 [0..7] 전부
        self.declare_parameter("excite_indices", [6])  # default: joint 6
        self.declare_parameter("amplitude_deg", 10.0)
        self.declare_parameter("frequency_hz", 0.3)
        self.declare_parameter("offset_deg", 0.0)

        # 각 조인트별 개별 amp/offset을 주고 싶으면 아래 리스트 파라미터를 쓰면 됨 (비어있으면 스칼라 사용)
        self.declare_parameter("amplitude_deg_list", [])  # length=num_joints
        self.declare_parameter("offset_deg_list", [])     # length=num_joints

        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("duration_sec", 30.0)
        self.declare_parameter("settle_sec", 0.5)
        self.declare_parameter("use_initial_positions", True)

        # -----------------------------
        # Parameters (logging)
        # -----------------------------
        self.declare_parameter("log_dir", "./test_log")
        self.declare_parameter("log_filename", "")

        # -----------------------------
        # Read parameters
        # -----------------------------
        self.num_joints = int(self.get_parameter("num_joints").value)

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.torque_commands_topic = str(self.get_parameter("torque_commands_topic").value)

        excite_indices = list(self.get_parameter("excite_indices").value)
        self.excite_indices = [int(i) for i in excite_indices if 0 <= int(i) < self.num_joints]

        amp_deg = float(self.get_parameter("amplitude_deg").value)
        freq_hz = float(self.get_parameter("frequency_hz").value)
        offset_deg = float(self.get_parameter("offset_deg").value)

        amp_list = list(self.get_parameter("amplitude_deg_list").value)
        off_list = list(self.get_parameter("offset_deg_list").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.use_initial_positions = bool(self.get_parameter("use_initial_positions").value)

        self.log_dir = Path(str(self.get_parameter("log_dir").value))
        self.log_filename = str(self.get_parameter("log_filename").value).strip()

        self.freq_hz = freq_hz

        # amp/offset: 리스트가 제대로 들어오면 조인트별 적용, 아니면 스칼라를 excite_indices에만 적용
        self.amp_rad = np.zeros(self.num_joints, dtype=float)
        self.offset_rad = np.zeros(self.num_joints, dtype=float)

        if len(amp_list) == self.num_joints:
            self.amp_rad[:] = np.radians(np.asarray(amp_list, dtype=float))
        else:
            for i in self.excite_indices:
                self.amp_rad[i] = math.radians(amp_deg)

        if len(off_list) == self.num_joints:
            self.offset_rad[:] = np.radians(np.asarray(off_list, dtype=float))
        else:
            for i in self.excite_indices:
                self.offset_rad[i] = math.radians(offset_deg)

        # -----------------------------
        # Pub/Sub
        # -----------------------------
        self.buffers = LatestBuffers()
        self.action_pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.create_subscription(JointState, self.joint_states_topic, self._on_joint_state, 50)
        self.create_subscription(Float32MultiArray, self.torque_commands_topic, self._on_torque_cmd, 50)

        # -----------------------------
        # Experiment state
        # -----------------------------
        self._started = False
        self._done = False
        self._t0 = None

        self._initial_q = np.zeros(self.num_joints, dtype=float)

        # CSV rows accumulate in memory; for very long duration you can stream-write instead
        self._rows: List[List[float]] = []
        self._header = self._build_header()

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "SineMotionLoggerNode started.\n"
            f"  action_topic: {self.action_topic}\n"
            f"  joint_states_topic: {self.joint_states_topic}\n"
            f"  torque_commands_topic: {self.torque_commands_topic}\n"
            f"  num_joints: {self.num_joints}\n"
            f"  excite_indices: {self.excite_indices}\n"
            f"  sine: freq={self.freq_hz:.3f} Hz, publish_rate={self.publish_rate_hz:.1f} Hz\n"
            f"  duration={self.duration_sec:.2f}s, settle={self.settle_sec:.2f}s\n"
            f"  use_initial_positions={self.use_initial_positions}\n"
            f"  log_dir={self.log_dir}, log_filename='{self.log_filename or '(auto)'}'\n"
        )

    def _build_header(self) -> List[str]:
        cols = ["t_sec"]
        for i in range(self.num_joints):
            cols += [
                f"q_ref_{i}_rad",
                f"q_meas_{i}_rad",
                f"dq_meas_{i}_rad_s",
                f"tau_cmd_{i}_nm",
            ]
        return cols

    def _on_joint_state(self, msg: JointState) -> None:
        self.buffers.joint_state = msg
        self.buffers.have_joint_state = True

        # 초기 자세 캡처(한 번만)
        if self.use_initial_positions and (not self._started) and msg.position:
            q = np.asarray(msg.position, dtype=float).flatten()
            if q.size >= self.num_joints:
                self._initial_q = q[: self.num_joints].copy()

    def _on_torque_cmd(self, msg: Float32MultiArray) -> None:
        vec = np.asarray(msg.data, dtype=float).flatten()
        self.buffers.torque_cmd = vec
        self.buffers.have_torque = True

    def _tick(self) -> None:
        if self._done:
            return
        if not self.buffers.have_joint_state or self.buffers.joint_state is None:
            return

        now = self.get_clock().now()
        if not self._started:
            self._started = True
            self._t0 = now
            self.get_logger().info("Experiment started.")

        t_sec = (now - self._t0).nanoseconds * 1e-9

        if t_sec >= self.duration_sec:
            self._finish()
            return

        # ---- build desired positions ----
        desired = self._initial_q.copy() if self.use_initial_positions else np.zeros(self.num_joints, dtype=float)

        # excite only indices: q_ref = q0 + offset + amp*sin(2πft)
        s = math.sin(2.0 * math.pi * self.freq_hz * t_sec)
        for i in self.excite_indices:
            desired[i] = desired[i] + self.offset_rad[i] + self.amp_rad[i] * s

        # publish
        msg = Float32MultiArray()
        msg.data = desired.astype(np.float32).tolist()
        self.action_pub.publish(msg)

        # ---- logging ----
        if t_sec < self.settle_sec:
            return
        if not self.buffers.have_torque or self.buffers.torque_cmd is None:
            return

        js = self.buffers.joint_state
        tau_vec = self.buffers.torque_cmd

        # measured
        q_meas = self._safe_vec(js.position, self.num_joints)
        dq_meas = self._safe_vec(js.velocity, self.num_joints)
        tau_cmd = self._safe_vec(tau_vec, self.num_joints)

        # q_ref는 방금 publish한 desired를 그대로 기록
        q_ref = desired

        row: List[float] = [float(t_sec)]
        for i in range(self.num_joints):
            row += [
                float(q_ref[i]),
                float(q_meas[i]),
                float(dq_meas[i]),
                float(tau_cmd[i]),
            ]
        self._rows.append(row)

    @staticmethod
    def _safe_vec(x, n: int) -> np.ndarray:
        if x is None:
            return np.zeros(n, dtype=float)
        arr = np.asarray(x, dtype=float).flatten()
        if arr.size >= n:
            return arr[:n]
        out = np.zeros(n, dtype=float)
        if arr.size > 0:
            out[: arr.size] = arr
        return out

    def _finish(self) -> None:
        self._done = True

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_filename:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = f"sine_log_{stamp}.csv"
        csv_path = self.log_dir / self.log_filename

        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._header)
            w.writerows(self._rows)

        self.get_logger().info(f"Saved CSV: {csv_path}")
        self.get_logger().info("Done. Shutting down.")
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SineMotionLoggerNode()
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
