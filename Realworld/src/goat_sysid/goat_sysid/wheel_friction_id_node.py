#!/usr/bin/env python3
# goat_sysid/goat_sysid/wheel_friction_id_node.py

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


@dataclass
class LatestBuffers:
    """Latest subscribed messages."""
    joint_state: Optional[JointState] = None
    torque_cmd: Optional[np.ndarray] = None  # shape: (num_joints,)
    have_joint_state: bool = False
    have_torque: bool = False


@dataclass
class RefPiece:
    """One piece of a piecewise speed reference."""
    t0: float
    t1: float
    kind: str  # "const" or "ramp"
    w0: float
    w1: float

    def eval(self, t: float) -> float:
        if t <= self.t0:
            return float(self.w0)
        if t >= self.t1:
            return float(self.w1)

        if self.kind == "const":
            return float(self.w0)

        # ramp (linear interpolation)
        s = (t - self.t0) / max(self.t1 - self.t0, 1e-12)
        return float(self.w0 + s * (self.w1 - self.w0))


class PiecewiseSpeedRef:
    """Piecewise speed reference generator."""
    def __init__(self, pieces: List[RefPiece]):
        self.pieces = pieces
        self.duration = float(pieces[-1].t1) if pieces else 0.0

    def eval(self, t: float) -> float:
        if not self.pieces:
            return 0.0
        if t <= 0.0:
            return self.pieces[0].eval(0.0)
        if t >= self.duration:
            return self.pieces[-1].eval(self.duration)

        # linear scan is fine for short schedules
        for p in self.pieces:
            if p.t0 <= t < p.t1:
                return p.eval(t)
        return self.pieces[-1].eval(self.duration)


class WheelFrictionIdNode(Node):
    """
    Wheel dynamic friction identification node (PI wheel-speed controller).

    What this node does:
      1) Publishes /goat/action with:
           - first N: desired joint positions [rad] (held at initial positions)
           - second N: desired wheel speeds [rad/s] (excitation profile)
      2) Subscribes:
           - /joint_states         (measured wheel speed is JointState.velocity[idx])
           - /torque_commands      (measured torque command is Float32MultiArray.data[idx])
      3) Logs CSV:
           t_sec, wheel_index, omega_ref_rad_s, omega_meas_rad_s, tau_cmd_nm
      4) Runs LS estimation:
           - fric_only:     tau ≈ a*omega + b*sign(omega) using steady-state mask (|qdd| small)
           - motor_only:    tau* = tau - J_known*qdd; tau* ≈ a*omega + b*sign(omega)
           - full_joint:    tau ≈ J*qdd + a*omega + b*sign(omega)

    Important:
      - To excite the PI wheel controller, you MUST publish a 2N-length action vector.
      - Make sure /goat/action has only ONE publisher during sysid.
    """

    def __init__(self) -> None:
        super().__init__("wheel_friction_id_node")

        # -----------------------------
        # Parameters (topics / sizes)
        # -----------------------------
        self.declare_parameter("num_joints", 8)
        self.declare_parameter("wheel_indices", [6])  # e.g. [6] or [6,7]
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("joint_states_topic", "joint_states")
        self.declare_parameter("torque_commands_topic", "torque_commands")

        # -----------------------------
        # Parameters (publishing)
        # -----------------------------
        self.declare_parameter("publish_rate_hz", 200.0)
        self.declare_parameter("use_initial_positions", True)

        # -----------------------------
        # Parameters (excitation profile)
        # -----------------------------
        # profile: "plateau" or "trapezoid"
        self.declare_parameter("profile", "plateau")

        # plateau profile
        self.declare_parameter("speed_levels_rad_s", [20.0, 40.0, 60.0, 80.0, 100.0, 120.0])  # magnitudes
        self.declare_parameter("plateau_sec", 3.0)
        self.declare_parameter("rest_zero_sec", 1.0)
        self.declare_parameter("include_zero_between", True)
        self.declare_parameter("both_directions", True)  # + and - for each level

        # trapezoid profile
        self.declare_parameter("w_max_rad_s", 200.0)
        self.declare_parameter("accel_rad_s2", 30.0)
        self.declare_parameter("hold_sec", 4.0)
        self.declare_parameter("rest_sec", 6.0)
        self.declare_parameter("cycles", 1)  # number of (+ then -) cycles

        # total duration
        self.declare_parameter("duration_sec", 0.0)  # 0 => auto from profile
        self.declare_parameter("settle_sec", 5.0)    # initial logging skip

        # -----------------------------
        # Parameters (logging)
        # -----------------------------
        self.declare_parameter("log_dir", "./test_log")
        self.declare_parameter("log_filename", "")

        # -----------------------------
        # Parameters (estimation)
        # -----------------------------
        self.declare_parameter("estimate_ls", True)
        self.declare_parameter("estimate_mode", "full_joint")  # "fric_only" | "motor_only" | "full_joint"
        self.declare_parameter("J_known", 0.0)                # used in motor_only
        self.declare_parameter("v_min_rad_s", 0.5)            # filter |omega| < v_min
        self.declare_parameter("qdd_max_rad_s2", 0.5)         # used in fric_codeonly (steady-state mask)
        self.declare_parameter("smooth_window", 5)
        self.declare_parameter("result_path", "")             # if empty, auto next to CSV

        # -----------------------------
        # Read parameters
        # -----------------------------
        self.num_joints = int(self.get_parameter("num_joints").value)

        wheel_indices_raw = self.get_parameter("wheel_indices").value
        self.wheel_indices = [int(x) for x in list(wheel_indices_raw)]
        if not self.wheel_indices:
            raise ValueError("wheel_indices must not be empty.")

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.torque_commands_topic = str(self.get_parameter("torque_commands_topic").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.use_initial_positions = bool(self.get_parameter("use_initial_positions").value)

        self.profile = str(self.get_parameter("profile").value).strip().lower()

        self.speed_levels = [float(x) for x in list(self.get_parameter("speed_levels_rad_s").value)]
        self.plateau_sec = float(self.get_parameter("plateau_sec").value)
        self.rest_zero_sec = float(self.get_parameter("rest_zero_sec").value)
        self.include_zero_between = bool(self.get_parameter("include_zero_between").value)
        self.both_directions = bool(self.get_parameter("both_directions").value)

        self.w_max = float(self.get_parameter("w_max_rad_s").value)
        self.accel = float(self.get_parameter("accel_rad_s2").value)
        self.hold_sec = float(self.get_parameter("hold_sec").value)
        self.rest_sec = float(self.get_parameter("rest_sec").value)
        self.cycles = int(self.get_parameter("cycles").value)

        self.duration_sec_param = float(self.get_parameter("duration_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)

        self.log_dir = Path(str(self.get_parameter("log_dir").value))
        self.log_filename = str(self.get_parameter("log_filename").value).strip()

        self.estimate_ls = bool(self.get_parameter("estimate_ls").value)
        self.estimate_mode = str(self.get_parameter("estimate_mode").value).strip().lower()
        self.J_known = float(self.get_parameter("J_known").value)
        self.v_min = float(self.get_parameter("v_min_rad_s").value)
        self.qdd_max = float(self.get_parameter("qdd_max_rad_s2").value)
        self.smooth_window = int(self.get_parameter("smooth_window").value)
        self.result_path_param = str(self.get_parameter("result_path").value).strip()

        # -----------------------------
        # Build speed reference schedule
        # -----------------------------
        self.speed_ref = self._build_speed_reference()
        self.duration_sec = self.duration_sec_param if self.duration_sec_param > 0.0 else self.speed_ref.duration

        # -----------------------------
        # Pub/Sub
        # -----------------------------
        self.buffers = LatestBuffers()
        self.action_pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.create_subscription(JointState, self.joint_states_topic, self._on_joint_state, 10)
        self.create_subscription(Float32MultiArray, self.torque_commands_topic, self._on_torque_cmd, 10)

        # -----------------------------
        # Experiment state
        # -----------------------------
        self._started = False
        self._done = False
        self._t0 = None
        self._initial_q: Optional[np.ndarray] = None

        # Each row: [t, wheel_index, w_ref, w_meas, tau_cmd]
        self._log_rows: List[List[float]] = []

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "WheelFrictionIdNode started.\n"
            f"  action_topic: {self.action_topic}\n"
            f"  joint_states_topic: {self.joint_states_topic}\n"
            f"  torque_commands_topic: {self.torque_commands_topic}\n"
            f"  num_joints: {self.num_joints}, wheel_indices: {self.wheel_indices}\n"
            f"  profile: {self.profile}, duration_sec: {self.duration_sec:.2f}s, publish_rate: {self.publish_rate_hz:.1f} Hz\n"
            f"  settle_sec: {self.settle_sec:.2f}s\n"
            f"  estimate_ls: {self.estimate_ls}, estimate_mode: {self.estimate_mode}\n"
            f"  v_min: {self.v_min:.3f} rad/s, qdd_max(fric_only): {self.qdd_max:.3f} rad/s^2, smooth_window: {self.smooth_window}\n"
        )

    # -----------------------------
    # ROS callbacks
    # -----------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        self.buffers.joint_state = msg
        self.buffers.have_joint_state = True

        if self._initial_q is None and self.use_initial_positions and msg.position:
            q = np.asarray(msg.position, dtype=float).flatten()
            if q.size >= self.num_joints:
                self._initial_q = q[: self.num_joints].copy()
            else:
                self._initial_q = np.zeros(self.num_joints, dtype=float)

    def _on_torque_cmd(self, msg: Float32MultiArray) -> None:
        vec = np.asarray(msg.data, dtype=float).flatten()
        self.buffers.torque_cmd = vec
        self.buffers.have_torque = True

    # -----------------------------
    # Main loop
    # -----------------------------
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

        # Desired joint positions (PD part)
        if self.use_initial_positions and (self._initial_q is not None):
            desired_pos = self._initial_q.copy()
        else:
            desired_pos = np.zeros(self.num_joints, dtype=float)

        # Desired wheel speeds (PI part)
        w_ref = float(self.speed_ref.eval(t_sec))
        desired_w = np.zeros(self.num_joints, dtype=float)
        for wi in self.wheel_indices:
            if 0 <= wi < self.num_joints:
                desired_w[wi] = w_ref

        # Publish 2N action: [pos_ref (N), wheel_speed_ref (N)]
        action = np.concatenate([desired_pos, desired_w], axis=0)
        msg = Float32MultiArray()
        msg.data = action.astype(np.float32).tolist()
        self.action_pub.publish(msg)

        # Logging
        if t_sec < self.settle_sec:
            return
        if not self.buffers.have_torque or self.buffers.torque_cmd is None:
            return

        js = self.buffers.joint_state
        tau_vec = self.buffers.torque_cmd

        # Log per wheel index
        for wi in self.wheel_indices:
            w_meas = self._read_velocity(js, wi)
            tau_cmd = self._read_tau(tau_vec, wi)
            self._log_rows.append([t_sec, float(wi), w_ref, w_meas, tau_cmd])

    @staticmethod
    def _read_velocity(js: JointState, idx: int) -> float:
        if js.velocity and idx < len(js.velocity):
            return float(js.velocity[idx])
        return 0.0

    @staticmethod
    def _read_tau(tau_vec: np.ndarray, idx: int) -> float:
        if tau_vec.size == 0:
            return 0.0
        if idx < tau_vec.size:
            return float(tau_vec[idx])
        return float(tau_vec[-1])

    # -----------------------------
    # Reference builders
    # -----------------------------
    def _build_speed_reference(self) -> PiecewiseSpeedRef:
        if self.profile == "plateau":
            return self._build_plateau_ref()
        if self.profile == "trapezoid":
            return self._build_trapezoid_ref()
        raise ValueError(f"Unknown profile: {self.profile}")

    def _build_plateau_ref(self) -> PiecewiseSpeedRef:
        pieces: List[RefPiece] = []
        t = 0.0

        def add_const(dt: float, w: float):
            nonlocal t, pieces
            dt = max(float(dt), 0.0)
            pieces.append(RefPiece(t0=t, t1=t + dt, kind="const", w0=w, w1=w))
            t += dt

        # Start at zero for a short moment (optional: use rest_zero_sec)
        if self.rest_zero_sec > 0.0:
            add_const(self.rest_zero_sec, 0.0)

        for level in self.speed_levels:
            level = abs(float(level))
            targets = [level]
            if self.both_directions:
                targets = [level, -level]

            for w in targets:
                add_const(self.plateau_sec, w)
                if self.include_zero_between:
                    add_const(self.rest_zero_sec, 0.0)

        # End with zero
        add_const(max(self.rest_zero_sec, 0.5), 0.0)
        return PiecewiseSpeedRef(pieces)

    def _build_trapezoid_ref(self) -> PiecewiseSpeedRef:
        pieces: List[RefPiece] = []
        t = 0.0

        def add_const(dt: float, w: float):
            nonlocal t, pieces
            dt = max(float(dt), 0.0)
            pieces.append(RefPiece(t0=t, t1=t + dt, kind="const", w0=w, w1=w))
            t += dt

        def add_ramp(dt: float, w0: float, w1: float):
            nonlocal t, pieces
            dt = max(float(dt), 0.0)
            pieces.append(RefPiece(t0=t, t1=t + dt, kind="ramp", w0=w0, w1=w1))
            t += dt

        w_max = float(abs(self.w_max))
        accel = float(abs(self.accel))
        if accel < 1e-9:
            raise ValueError("accel_rad_s2 must be > 0 for trapezoid profile.")

        ramp_time = w_max / accel

        # Start at zero
        add_const(max(self.rest_sec, 0.5), 0.0)

        for _ in range(max(self.cycles, 1)):
            for sign in (1.0, -1.0):
                w_target = sign * w_max
                add_ramp(ramp_time, 0.0, w_target)
                add_const(self.hold_sec, w_target)
                add_ramp(ramp_time, w_target, 0.0)
                add_const(self.rest_sec, 0.0)

        # End at zero
        add_const(max(self.rest_sec, 0.5), 0.0)
        return PiecewiseSpeedRef(pieces)

    # -----------------------------
    # Finish + Save + Estimate
    # -----------------------------
    def _finish(self) -> None:
        self._done = True

        # Save CSV
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_filename:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wheels_str = "_".join(str(i) for i in self.wheel_indices)
            self.log_filename = f"wheel_sysid_w{wheels_str}_{stamp}.csv"

        csv_path = self.log_dir / self.log_filename
        self._save_csv(csv_path)
        self.get_logger().info(f"Saved log to: {csv_path}")

        # Estimate
        if self.estimate_ls and len(self._log_rows) >= 50:
            rows = np.asarray(self._log_rows, dtype=float)
            result_text, result_yaml = self._estimate_ls(rows)

            self.get_logger().info(result_text)
            result_path = self._resolve_result_path(csv_path)
            result_path.write_text(result_yaml, encoding="utf-8")
            self.get_logger().info(f"Saved result to: {result_path}")
        else:
            self.get_logger().warn("Skipping estimation (estimate_ls=false or not enough samples).")

        self.get_logger().info("Experiment finished. Shutting down node.")
        rclpy.shutdown()

    def _save_csv(self, path: Path) -> None:
        header = [
            "t_sec",
            "wheel_index",
            "omega_ref_rad_s",
            "omega_meas_rad_s",
            "tau_cmd_nm",
        ]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in self._log_rows:
                w.writerow(row)

    def _resolve_result_path(self, csv_path: Path) -> Path:
        if self.result_path_param:
            p = Path(self.result_path_param)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        return csv_path.with_suffix(".yaml")

    # -----------------------------
    # LS estimation (per wheel)
    # -----------------------------
    def _estimate_ls(self, rows: np.ndarray) -> Tuple[str, str]:
        """
        rows columns:
          t, wheel_index, omega_ref, omega_meas, tau_cmd

        Returns:
          (human_readable_text, yaml_text)
        """
        mode = self.estimate_mode
        wheels = sorted(set(int(x) for x in rows[:, 1].tolist()))

        out_lines: List[str] = []
        yaml_lines: List[str] = []

        out_lines.append(f"LS estimation mode: {mode}")
        yaml_lines.append(f"mode: {mode}")
        yaml_lines.append("units:")
        yaml_lines.append("  omega: rad/s")
        yaml_lines.append("  qdd: rad/s^2")
        yaml_lines.append("  tau: Nm")
        yaml_lines.append("results:")

        for wi in wheels:
            r = rows[rows[:, 1] == float(wi)]
            if r.shape[0] < 20:
                out_lines.append(f"  wheel {wi}: not enough samples ({r.shape[0]})")
                yaml_lines.append(f"  wheel_{wi}: {{error: not_enough_samples}}")
                continue

            t = r[:, 0].astype(float)
            omega = r[:, 3].astype(float)
            tau = r[:, 4].astype(float)

            omega_f = self._moving_average(omega, self.smooth_window)
            qdd = np.gradient(omega_f, t)

            # Basic mask
            mask_v = np.abs(omega_f) >= float(self.v_min)

            if mode == "fric_only":
                # "steady-state" mask: acceleration small
                mask = mask_v & (np.abs(qdd) <= float(self.qdd_max))
                Phi = np.column_stack([omega_f[mask], np.sign(omega_f[mask])])
                y = tau[mask]
                if y.size < 10:
                    out_lines.append(f"  wheel {wi}: not enough steady-state samples ({y.size})")
                    yaml_lines.append(f"  wheel_{wi}: {{error: not_enough_steady_state_samples}}")
                    continue

                theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                a_hat, b_hat = float(theta[0]), float(theta[1])

                y_hat = Phi @ theta
                rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))

                out_lines.append(
                    f"  wheel {wi} (fric_only): a={a_hat:.6e} Nm/(rad/s), b={b_hat:.6e} Nm, rmse={rmse:.4g}, used={y.size}/{len(t)}"
                )
                yaml_lines.append(f"  wheel_{wi}:")
                yaml_lines.append(f"    a_hat_nm_per_rad_s: {a_hat:.12f}")
                yaml_lines.append(f"    b_hat_nm: {b_hat:.12f}")
                yaml_lines.append(f"    rmse_nm: {rmse:.12f}")
                yaml_lines.append(f"    used_samples: {int(y.size)}")
                yaml_lines.append(f"    total_samples: {int(len(t))}")

            elif mode == "motor_only":
                # tau* = tau - J_known*qdd; tau* ≈ a*omega + b*sign(omega)
                mask = mask_v
                if mask.sum() < 10:
                    out_lines.append(f"  wheel {wi}: not enough samples after v_min ({int(mask.sum())})")
                    yaml_lines.append(f"  wheel_{wi}: {{error: not_enough_samples_after_vmin}}")
                    continue

                tau_star = tau[mask] - float(self.J_known) * qdd[mask]
                Phi = np.column_stack([omega_f[mask], np.sign(omega_f[mask])])
                theta, *_ = np.linalg.lstsq(Phi, tau_star, rcond=None)
                a_hat, b_hat = float(theta[0]), float(theta[1])

                y_hat = Phi @ theta
                rmse = float(np.sqrt(np.mean((y_hat - tau_star) ** 2)))

                out_lines.append(
                    f"  wheel {wi} (motor_only): J_known={self.J_known:.3e}, a={a_hat:.6e}, b={b_hat:.6e}, rmse={rmse:.4g}, used={int(mask.sum())}/{len(t)}"
                )
                yaml_lines.append(f"  wheel_{wi}:")
                yaml_lines.append(f"    J_known_nm_per_rad_s2: {float(self.J_known):.12f}")
                yaml_lines.append(f"    a_hat_nm_per_rad_s: {a_hat:.12f}")
                yaml_lines.append(f"    b_hat_nm: {b_hat:.12f}")
                yaml_lines.append(f"    rmse_nm: {rmse:.12f}")
                yaml_lines.append(f"    used_samples: {int(mask.sum())}")
                yaml_lines.append(f"    total_samples: {int(len(t))}")

            elif mode == "full_joint":
                # tau ≈ J*qdd + a*omega + b*sign(omega)
                mask = mask_v
                if mask.sum() < 10:
                    out_lines.append(f"  wheel {wi}: not enough samples after v_min ({int(mask.sum())})")
                    yaml_lines.append(f"  wheel_{wi}: {{error: not_enough_samples_after_vmin}}")
                    continue

                Phi = np.column_stack([qdd[mask], omega_f[mask], np.sign(omega_f[mask])])
                y = tau[mask]
                theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
                J_hat, a_hat, b_hat = float(theta[0]), float(theta[1]), float(theta[2])

                y_hat = Phi @ theta
                rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))

                out_lines.append(
                    f"  wheel {wi} (full_joint): J={J_hat:.6e} Nm/(rad/s^2), a={a_hat:.6e}, b={b_hat:.6e}, rmse={rmse:.4g}, used={int(mask.sum())}/{len(t)}"
                )
                yaml_lines.append(f"  wheel_{wi}:")
                yaml_lines.append(f"    J_hat_nm_per_rad_s2: {J_hat:.12f}")
                yaml_lines.append(f"    a_hat_nm_per_rad_s: {a_hat:.12f}")
                yaml_lines.append(f"    b_hat_nm: {b_hat:.12f}")
                yaml_lines.append(f"    rmse_nm: {rmse:.12f}")
                yaml_lines.append(f"    used_samples: {int(mask.sum())}")
                yaml_lines.append(f"    total_samples: {int(len(t))}")

            else:
                out_lines.append(f"  wheel {wi}: unknown estimate_mode '{mode}'")
                yaml_lines.append(f"  wheel_{wi}: {{error: unknown_mode}}")

        # Add a quick URDF hint (damping/friction)
        yaml_lines.append("urdf_hint:")
        yaml_lines.append("  # If you want to map to <dynamics> (very rough):")
        yaml_lines.append("  #   damping  ~= a_hat")
        yaml_lines.append("  #   friction ~= b_hat")

        text = "\n".join(out_lines)
        yml = "\n".join(yaml_lines) + "\n"
        return text, yml

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if window <= 1:
            return x
        window = int(window)
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(x, kernel, mode="same")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelFrictionIdNode()
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
