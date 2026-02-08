# goat_control/nodes/motor_log_plotter_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from collections import deque
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

import matplotlib.pyplot as plt

# ===== Defaults (edit here) =====
DEFAULT_LOG_TOPIC = "motor_torque_log"
DEFAULT_NUM_JOINTS = 8
DEFAULT_MOTOR_INDICES = [0, 1, 2, 3, 4, 5]
DEFAULT_HISTORY_SEC = 10.0
DEFAULT_PLOT_DEGREES = True
DEFAULT_COMMAND_UNIT = "torque_nm"  # "torque_nm" or "amp"

# optional
DEFAULT_USE_JOINT_STATE_NAMES = True
DEFAULT_JOINT_STATE_TOPIC = "joint_states"
DEFAULT_EXPECTED_LOG_HZ = 200.0
DEFAULT_PLOT_RATE_HZ = 100.0
DEFAULT_AUTOSCALE_Y = True

@dataclass
class LatestLog:
    t_sec: Optional[float] = None
    vector: Optional[np.ndarray] = None


class MotorLogPlotter(Node):
    """
    Subscribes: motor_torque_log (Float32MultiArray)
      layout:
        (A) [q xN, dq xN, u xN]         => 3N
        (B) [q xN, dq xN, u xN, ref xN] => 4N (ref는 무시/옵션)

    Plots: selected motor_indices (default: 0..5) for:
      - angle q
      - velocity dq
      - command u (torque or current)
    """

    def __init__(self):
        super().__init__("motor_log_plotter")

        self.declare_parameter("log_topic", DEFAULT_LOG_TOPIC)
        self.declare_parameter("joint_state_topic", DEFAULT_JOINT_STATE_TOPIC)
        self.declare_parameter("use_joint_state_names", DEFAULT_USE_JOINT_STATE_NAMES)

        self.declare_parameter("num_joints", DEFAULT_NUM_JOINTS)

        # 기존 joint_names는 그대로 두거나, 필요하면 DEFAULT로 따로 빼도 됨
        self.declare_parameter(
            "joint_names",
            ["hip_L", "hip_R", "thigh_L", "thigh_R", "knee_L", "knee_R", "wheel_L", "wheel_R"],
        )

        self.declare_parameter("motor_indices", DEFAULT_MOTOR_INDICES)

        self.declare_parameter("plot_degrees", DEFAULT_PLOT_DEGREES)
        self.declare_parameter("command_unit", DEFAULT_COMMAND_UNIT)

        self.declare_parameter("history_sec", DEFAULT_HISTORY_SEC)
        self.declare_parameter("expected_log_hz", DEFAULT_EXPECTED_LOG_HZ)

        self.declare_parameter("plot_rate_hz", DEFAULT_PLOT_RATE_HZ)
        self.declare_parameter("autoscale_y", DEFAULT_AUTOSCALE_Y)

        # -------- Read params --------
        self.log_topic = str(self.get_parameter("log_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.use_joint_state_names = bool(self.get_parameter("use_joint_state_names").value)

        self.num_joints = int(self.get_parameter("num_joints").value)
        self.joint_names: List[str] = [str(x) for x in self.get_parameter("joint_names").value]

        self.motor_indices = [int(x) for x in self.get_parameter("motor_indices").value]
        self.motor_indices = [i for i in self.motor_indices if 0 <= i < self.num_joints]
        if len(self.motor_indices) == 0:
            self.motor_indices = list(range(min(6, self.num_joints)))

        self.plot_degrees = bool(self.get_parameter("plot_degrees").value)
        self.command_unit = str(self.get_parameter("command_unit").value)

        self.history_sec = float(self.get_parameter("history_sec").value)
        self.expected_log_hz = float(self.get_parameter("expected_log_hz").value)
        self.plot_rate_hz = float(self.get_parameter("plot_rate_hz").value)
        self.autoscale_y = bool(self.get_parameter("autoscale_y").value)

        if len(self.joint_names) != self.num_joints:
            self.get_logger().warn(
                f"joint_names length ({len(self.joint_names)}) != num_joints ({self.num_joints}). "
                "Falling back to generic names."
            )
            self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]

        # -------- Buffers --------
        # buffer length based on expected_log_hz * history_sec (min 50)
        self.maxlen = max(int(self.expected_log_hz * self.history_sec), 50)

        self.t_buf = deque(maxlen=self.maxlen)
        self.q_buf = {i: deque(maxlen=self.maxlen) for i in self.motor_indices}
        self.dq_buf = {i: deque(maxlen=self.maxlen) for i in self.motor_indices}
        self.u_buf = {i: deque(maxlen=self.maxlen) for i in self.motor_indices}

        self.latest = LatestLog()
        self._t0: Optional[float] = None
        self._last_plot_time = 0.0
        self._warn_last_time = 0.0

        # -------- ROS subscriptions --------
        self.create_subscription(Float32MultiArray, self.log_topic, self._on_log, 10)
        if self.use_joint_state_names:
            self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)

        # -------- Matplotlib figure --------
        plt.ion()
        self.fig, (self.ax_q, self.ax_dq, self.ax_u) = plt.subplots(3, 1, sharex=True)
        self.fig.canvas.manager.set_window_title("GOAT Motor Log Plotter")

        q_unit = "deg" if self.plot_degrees else "rad"
        dq_unit = "deg/s" if self.plot_degrees else "rad/s"
        u_unit = "Nm" if self.command_unit == "torque_nm" else "A"

        self.ax_q.set_ylabel(f"q [{q_unit}]")
        self.ax_dq.set_ylabel(f"dq [{dq_unit}]")
        self.ax_u.set_ylabel(f"u [{u_unit}]")
        self.ax_u.set_xlabel("time [s]")

        self.lines_q = {}
        self.lines_dq = {}
        self.lines_u = {}

        for i in self.motor_indices:
            name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
            (lq,) = self.ax_q.plot([], [], label=f"{i}:{name}")
            (ldq,) = self.ax_dq.plot([], [])
            (lu,) = self.ax_u.plot([], [])
            self.lines_q[i] = lq
            self.lines_dq[i] = ldq
            self.lines_u[i] = lu

        self.ax_q.legend(loc="upper right", ncol=2, fontsize="small")
        self.fig.tight_layout()

        self.get_logger().info(
            f"MotorLogPlotter started. topic='{self.log_topic}', motors={self.motor_indices}, "
            f"history={self.history_sec}s, plot_rate={self.plot_rate_hz}Hz"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_log(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=float).flatten()
        t = self._now_sec()
        self.latest.t_sec = t
        self.latest.vector = data

    def _on_joint_state(self, msg: JointState) -> None:
        if msg.name and len(msg.name) == self.num_joints:
            self.joint_names = list(msg.name)
            # legend labels 업데이트는 간단히 전체 다시 그리지 않고, q축 라인 label만 수정
            for i in self.motor_indices:
                name = self.joint_names[i]
                self.lines_q[i].set_label(f"{i}:{name}")
            self.ax_q.legend(loc="upper right", ncol=2, fontsize="small")

    def _parse_vector(self, vector: np.ndarray):
        expected_3n = 3 * self.num_joints
        expected_4n = 4 * self.num_joints
        if vector.size not in (expected_3n, expected_4n):
            # warn rate-limit (max 1Hz)
            now = time.time()
            if now - self._warn_last_time > 1.0:
                self.get_logger().warn(
                    f"log length mismatch: got {vector.size}, expected {expected_3n} (3N) or {expected_4n} (4N)"
                )
                self._warn_last_time = now
            return None

        q = vector[0:self.num_joints]
        dq = vector[self.num_joints:2 * self.num_joints]
        u = vector[2 * self.num_joints:3 * self.num_joints]
        # ref = vector[3 * self.num_joints:4 * self.num_joints] if vector.size == expected_4n else None
        return q, dq, u

    def push_sample(self) -> None:
        if self.latest.vector is None or self.latest.t_sec is None:
            return

        parsed = self._parse_vector(self.latest.vector)
        if parsed is None:
            return
        q_rad, dq_rad_s, u = parsed

        if self._t0 is None:
            self._t0 = self.latest.t_sec

        t_rel = self.latest.t_sec - self._t0

        if self.plot_degrees:
            q_val = np.rad2deg(q_rad)
            dq_val = np.rad2deg(dq_rad_s)
        else:
            q_val = q_rad
            dq_val = dq_rad_s

        self.t_buf.append(float(t_rel))
        for i in self.motor_indices:
            self.q_buf[i].append(float(q_val[i]))
            self.dq_buf[i].append(float(dq_val[i]))
            self.u_buf[i].append(float(u[i]))

    def update_plot(self) -> None:
        # rate-limit GUI updates
        now = time.time()
        if now - self._last_plot_time < (1.0 / max(self.plot_rate_hz, 1.0)):
            return
        self._last_plot_time = now

        if len(self.t_buf) < 2:
            return

        t = np.asarray(self.t_buf)
        # x축: 최근 history_sec만 보이도록
        t_max = float(t[-1])
        t_min = max(0.0, t_max - self.history_sec)

        for i in self.motor_indices:
            self.lines_q[i].set_data(t, np.asarray(self.q_buf[i]))
            self.lines_dq[i].set_data(t, np.asarray(self.dq_buf[i]))
            self.lines_u[i].set_data(t, np.asarray(self.u_buf[i]))

        for ax in (self.ax_q, self.ax_dq, self.ax_u):
            ax.set_xlim(t_min, t_max)

            if self.autoscale_y:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = MotorLogPlotter()

    try:
        # matplotlib GUI 루프를 위해 spin_once 기반으로 돌림
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.push_sample()
            node.update_plot()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
