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


# =========================
# Defaults (edit here)
# =========================
DEFAULT_LOG_TOPIC = "motor_torque_log"
DEFAULT_JOINT_STATE_TOPIC = "joint_states"
DEFAULT_USE_JOINT_STATE_NAMES = True

DEFAULT_NUM_JOINTS = 8
DEFAULT_JOINT_NAMES = [
    "hip_L", "hip_R", "thigh_L", "thigh_R", "knee_L", "knee_R", "wheel_L", "wheel_R"
]

# Indices
DEFAULT_JOINT_INDICES = [0, 1, 2, 3, 4, 5]  # 6 motors for joints
DEFAULT_WHEEL_INDICES = [6, 7]              # 2 wheels

# Units
DEFAULT_PLOT_DEGREES = True                 # joints: rad -> deg
DEFAULT_COMMAND_UNIT = "torque_nm"          # "torque_nm" or "amp"
DEFAULT_PLOT_WHEELS_SEPARATELY = True
DEFAULT_WHEEL_SPEED_UNIT = "deg_s"            # "rad_s", "deg_s", "rpm"

# Plot window / rates
DEFAULT_HISTORY_SEC = 10.0
DEFAULT_EXPECTED_LOG_HZ = 200.0
DEFAULT_PLOT_RATE_HZ = 30.0
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
        (B) [q xN, dq xN, u xN, ref xN] => 4N (ref는 여기서는 무시)

    Plots:
      - Joint window (3 axes): q / dq / u for joint_indices
      - Wheel window (2 axes): wheel speed / wheel u for wheel_indices
    """

    def __init__(self):
        super().__init__("motor_log_plotter")

        # -------- Parameters --------
        self.declare_parameter("log_topic", DEFAULT_LOG_TOPIC)
        self.declare_parameter("joint_state_topic", DEFAULT_JOINT_STATE_TOPIC)
        self.declare_parameter("use_joint_state_names", DEFAULT_USE_JOINT_STATE_NAMES)

        self.declare_parameter("num_joints", DEFAULT_NUM_JOINTS)
        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)

        self.declare_parameter("joint_indices", DEFAULT_JOINT_INDICES)
        self.declare_parameter("wheel_indices", DEFAULT_WHEEL_INDICES)

        self.declare_parameter("plot_degrees", DEFAULT_PLOT_DEGREES)
        self.declare_parameter("command_unit", DEFAULT_COMMAND_UNIT)

        self.declare_parameter("plot_wheels_separately", DEFAULT_PLOT_WHEELS_SEPARATELY)
        self.declare_parameter("wheel_speed_unit", DEFAULT_WHEEL_SPEED_UNIT)

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

        self.joint_indices = [int(x) for x in self.get_parameter("joint_indices").value]
        self.wheel_indices = [int(x) for x in self.get_parameter("wheel_indices").value]

        self.plot_degrees = bool(self.get_parameter("plot_degrees").value)
        self.command_unit = str(self.get_parameter("command_unit").value)

        self.plot_wheels_separately = bool(self.get_parameter("plot_wheels_separately").value)
        self.wheel_speed_unit = str(self.get_parameter("wheel_speed_unit").value)

        self.history_sec = float(self.get_parameter("history_sec").value)
        self.expected_log_hz = float(self.get_parameter("expected_log_hz").value)

        self.plot_rate_hz = float(self.get_parameter("plot_rate_hz").value)
        self.autoscale_y = bool(self.get_parameter("autoscale_y").value)

        # -------- Validate names / indices --------
        if len(self.joint_names) != self.num_joints:
            self.get_logger().warn(
                f"joint_names length ({len(self.joint_names)}) != num_joints ({self.num_joints}). "
                "Falling back to generic names."
            )
            self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]

        self.joint_indices = [i for i in self.joint_indices if 0 <= i < self.num_joints]
        self.wheel_indices = [i for i in self.wheel_indices if 0 <= i < self.num_joints]

        if len(self.joint_indices) == 0:
            self.joint_indices = list(range(min(6, self.num_joints)))
        # wheel_indices는 비어있어도 OK (그럼 wheel 창 안뜸)

        # -------- Buffers --------
        self.maxlen = max(int(self.expected_log_hz * self.history_sec), 50)

        # joints
        self.t_buf = deque(maxlen=self.maxlen)
        self.q_buf = {i: deque(maxlen=self.maxlen) for i in self.joint_indices}
        self.dq_buf = {i: deque(maxlen=self.maxlen) for i in self.joint_indices}
        self.u_buf = {i: deque(maxlen=self.maxlen) for i in self.joint_indices}

        # wheels (separate time buffer, but can share if you want)
        self.wt_buf = deque(maxlen=self.maxlen)
        self.wspd_buf = {i: deque(maxlen=self.maxlen) for i in self.wheel_indices}
        self.wu_buf = {i: deque(maxlen=self.maxlen) for i in self.wheel_indices}

        self.latest = LatestLog()
        self._t0: Optional[float] = None
        self._last_plot_time = 0.0
        self._warn_last_time = 0.0

        # -------- ROS subscriptions --------
        self.create_subscription(Float32MultiArray, self.log_topic, self._on_log, 10)
        if self.use_joint_state_names:
            self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)

        # -------- Matplotlib figures --------
        plt.ion()

        # Joint figure (3 axes)
        self.fig_j, (self.ax_q, self.ax_dq, self.ax_u) = plt.subplots(3, 1, sharex=True)
        self.fig_j.canvas.manager.set_window_title("GOAT Joint q/dq/u")

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

        for i in self.joint_indices:
            name = self._safe_name(i)
            (lq,) = self.ax_q.plot([], [], label=f"{i}:{name}")
            (ldq,) = self.ax_dq.plot([], [])
            (lu,) = self.ax_u.plot([], [])
            self.lines_q[i] = lq
            self.lines_dq[i] = ldq
            self.lines_u[i] = lu

        self.ax_q.legend(loc="upper right", ncol=2, fontsize="small")
        self.fig_j.tight_layout()

        # Wheel figure (2 axes) - optional
        self.fig_w = None
        self.ax_wspd = None
        self.ax_wu = None
        self.w_lines_spd = {}
        self.w_lines_u = {}

        if self.plot_wheels_separately and len(self.wheel_indices) > 0:
            self.fig_w, (self.ax_wspd, self.ax_wu) = plt.subplots(2, 1, sharex=True)
            self.fig_w.canvas.manager.set_window_title("GOAT Wheel speed/u")

            spd_unit = (
                "rad/s" if self.wheel_speed_unit == "rad_s"
                else ("deg/s" if self.wheel_speed_unit == "deg_s" else "rpm")
            )
            u_unit_w = "Nm" if self.command_unit == "torque_nm" else "A"

            self.ax_wspd.set_ylabel(f"wheel speed [{spd_unit}]")
            self.ax_wu.set_ylabel(f"wheel u [{u_unit_w}]")
            self.ax_wu.set_xlabel("time [s]")

            for i in self.wheel_indices:
                name = self._safe_name(i)
                (ls,) = self.ax_wspd.plot([], [], label=f"{i}:{name}")
                (lu,) = self.ax_wu.plot([], [])
                self.w_lines_spd[i] = ls
                self.w_lines_u[i] = lu

            self.ax_wspd.legend(loc="upper right", fontsize="small")
            self.fig_w.tight_layout()

        self.get_logger().info(
            f"MotorLogPlotter started. topic='{self.log_topic}', "
            f"joints={self.joint_indices}, wheels={self.wheel_indices}, "
            f"history={self.history_sec}s, plot_rate={self.plot_rate_hz}Hz"
        )

    def _safe_name(self, idx: int) -> str:
        if 0 <= idx < len(self.joint_names):
            return str(self.joint_names[idx])
        return f"joint_{idx}"

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_log(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=float).flatten()
        t = self._now_sec()
        self.latest.t_sec = t
        self.latest.vector = data

    def _on_joint_state(self, msg: JointState) -> None:
        # Use incoming names if they look valid and match num_joints
        if msg.name and len(msg.name) == self.num_joints:
            self.joint_names = list(msg.name)

            # update legend labels (joint fig)
            for i in self.joint_indices:
                self.lines_q[i].set_label(f"{i}:{self._safe_name(i)}")
            self.ax_q.legend(loc="upper right", ncol=2, fontsize="small")

            # update legend labels (wheel fig)
            if self.fig_w is not None:
                for i in self.wheel_indices:
                    self.w_lines_spd[i].set_label(f"{i}:{self._safe_name(i)}")
                self.ax_wspd.legend(loc="upper right", fontsize="small")

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
        return q, dq, u

    def _convert_wheel_speed(self, dq_rad_s: float) -> float:
        if self.wheel_speed_unit == "rad_s":
            return float(dq_rad_s)
        if self.wheel_speed_unit == "deg_s":
            return float(np.rad2deg(dq_rad_s))
        if self.wheel_speed_unit == "rpm":
            return float(dq_rad_s * 60.0 / (2.0 * np.pi))
        return float(dq_rad_s)

    def push_sample(self) -> None:
        if self.latest.vector is None or self.latest.t_sec is None:
            return

        parsed = self._parse_vector(self.latest.vector)
        if parsed is None:
            return

        q_rad, dq_rad_s, u = parsed

        if self._t0 is None:
            self._t0 = self.latest.t_sec

        t_rel = float(self.latest.t_sec - self._t0)

        # joints units
        if self.plot_degrees:
            q_val = np.rad2deg(q_rad)
            dq_val = np.rad2deg(dq_rad_s)
        else:
            q_val = q_rad
            dq_val = dq_rad_s

        # push joints
        self.t_buf.append(t_rel)
        for i in self.joint_indices:
            self.q_buf[i].append(float(q_val[i]))
            self.dq_buf[i].append(float(dq_val[i]))
            self.u_buf[i].append(float(u[i]))

        # push wheels
        if self.plot_wheels_separately and len(self.wheel_indices) > 0:
            self.wt_buf.append(t_rel)
            for i in self.wheel_indices:
                self.wspd_buf[i].append(self._convert_wheel_speed(float(dq_rad_s[i])))
                self.wu_buf[i].append(float(u[i]))

    def _update_joint_plot(self, t_min: float, t_max: float) -> None:
        if len(self.t_buf) < 2:
            return

        t = np.asarray(self.t_buf)
        for i in self.joint_indices:
            self.lines_q[i].set_data(t, np.asarray(self.q_buf[i]))
            self.lines_dq[i].set_data(t, np.asarray(self.dq_buf[i]))
            self.lines_u[i].set_data(t, np.asarray(self.u_buf[i]))

        for ax in (self.ax_q, self.ax_dq, self.ax_u):
            ax.set_xlim(t_min, t_max)
            if self.autoscale_y:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

        self.fig_j.canvas.draw_idle()
        self.fig_j.canvas.flush_events()

    def _update_wheel_plot(self, t_min: float, t_max: float) -> None:
        if self.fig_w is None or len(self.wt_buf) < 2:
            return

        wt = np.asarray(self.wt_buf)
        for i in self.wheel_indices:
            self.w_lines_spd[i].set_data(wt, np.asarray(self.wspd_buf[i]))
            self.w_lines_u[i].set_data(wt, np.asarray(self.wu_buf[i]))

        for ax in (self.ax_wspd, self.ax_wu):
            ax.set_xlim(t_min, t_max)
            if self.autoscale_y:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

        self.fig_w.canvas.draw_idle()
        self.fig_w.canvas.flush_events()

    def update_plot(self) -> None:
        # rate-limit GUI updates
        now = time.time()
        if now - self._last_plot_time < (1.0 / max(self.plot_rate_hz, 1.0)):
            return
        self._last_plot_time = now

        # x-axis range: last history_sec
        # (joint t_buf 기준으로 잡고, wheel도 동일 t_rel을 쓰므로 그대로 적용)
        if len(self.t_buf) < 2:
            return

        t_arr = np.asarray(self.t_buf)
        t_max = float(t_arr[-1])
        t_min = max(0.0, t_max - self.history_sec)

        # update plots
        self._update_joint_plot(t_min, t_max)
        self._update_wheel_plot(t_min, t_max)


def main(args=None):
    rclpy.init(args=args)
    node = MotorLogPlotter()

    try:
        # matplotlib GUI loop + ROS spin_once
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
