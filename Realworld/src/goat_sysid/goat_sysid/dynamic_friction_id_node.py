# goat_sysid/goat_sysid/friction_id_node.py
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


@dataclass
class LatestBuffers:
    """Buffers for latest subscribed messages."""
    joint_state: Optional[JointState] = None
    torque_cmd: Optional[np.ndarray] = None  # shape: (num_joints,)
    have_joint_state: bool = False
    have_torque: bool = False


class FrictionIdNode(Node):
    """
    Dynamic friction identification node (goat_sysid).

    This node:
      - Publishes a sine position target to goat_control via `goat/action`
      - Subscribes to `joint_states` for measured q, dq
      - Subscribes to `goat/torque_commands` for commanded torque tau
      - Logs {t, q_ref, q_meas, dq_meas, tau_cmd} to CSV
      - Optionally estimates friction parameters using LS with acceleration term:

        motor_only mode (known J_motor):
            tau* = tau - J_motor*qdd
            tau* ≈ a*dq + b*sign(dq)

        full_joint mode (estimate J too):
            tau ≈ J*qdd + a*dq + b*sign(dq)

    Notes:
      - qdd is computed numerically from dq using np.gradient(dq, t)
      - Optional moving-average smoothing on dq before differentiation
    """

    def __init__(self) -> None:
        super().__init__("friction_id_node")

        # -----------------------------
        # Parameters (topics / sizes)
        # -----------------------------
        self.declare_parameter("num_joints", 8)
        self.declare_parameter("joint_index", 6)
        self.declare_parameter("joint_name", "")  # optional
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("joint_states_topic", "joint_states")
        self.declare_parameter("torque_commands_topic", "goat/torque_commands")

        # -----------------------------
        # Parameters (excitation)
        # -----------------------------
        self.declare_parameter("amplitude_deg", 30.0)
        self.declare_parameter("frequency_hz", 0.3)
        self.declare_parameter("offset_deg", 0.0)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("settle_sec", 1.0)
        self.declare_parameter("use_initial_positions", True)

        # -----------------------------
        # Parameters (logging)
        # -----------------------------
        self.declare_parameter("log_dir", "./test_log")
        self.declare_parameter("log_filename", "")

        # -----------------------------
        # Parameters (estimation)
        # -----------------------------
        self.declare_parameter("estimate_ls", True)
        self.declare_parameter("estimate_mode", "motor_only")  # "simple_fric" | "motor_only" | "full_joint"
        self.declare_parameter("J_motor", 0.0)                 # used in motor_only
        self.declare_parameter("v_min_deg_s", 5.0)             # filter |dq| < v_min before LS
        self.declare_parameter("smooth_window", 1)             # moving average window on dq (samples)
        self.declare_parameter("result_path", "")              # if empty, auto next to CSV

        # Read parameters
        self.num_joints = int(self.get_parameter("num_joints").value)
        self.joint_index_param = int(self.get_parameter("joint_index").value)
        self.joint_name_param = str(self.get_parameter("joint_name").value).strip()

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.torque_commands_topic = str(self.get_parameter("goat/torque_commands_topic").value)

        amp_deg = float(self.get_parameter("amplitude_deg").value)
        freq_hz = float(self.get_parameter("frequency_hz").value)
        offset_deg = float(self.get_parameter("offset_deg").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.use_initial_positions = bool(self.get_parameter("use_initial_positions").value)

        self.amp_rad = math.radians(amp_deg)
        self.freq_hz = freq_hz
        self.offset_rad = math.radians(offset_deg)

        self.log_dir = Path(str(self.get_parameter("log_dir").value))
        self.log_filename = str(self.get_parameter("log_filename").value).strip()

        self.estimate_ls = bool(self.get_parameter("estimate_ls").value)
        self.estimate_mode = str(self.get_parameter("estimate_mode").value).strip() 
        self.J_motor = float(self.get_parameter("J_motor").value)
        self.v_min_rad_s = math.radians(float(self.get_parameter("v_min_deg_s").value))
        self.smooth_window = int(self.get_parameter("smooth_window").value)
        self.result_path_param = str(self.get_parameter("result_path").value).strip()

        # Pub/Sub
        self.buffers = LatestBuffers()
        self.action_pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.create_subscription(JointState, self.joint_states_topic, self._on_joint_state, 10)
        self.create_subscription(Float32MultiArray, self.torque_commands_topic, self._on_torque_cmd, 10)

        # Experiment state
        self._started = False
        self._done = False
        self._joint_idx: Optional[int] = None
        self._initial_q: Optional[np.ndarray] = None
        self._t0 = None
        self._log_rows: List[List[float]] = []  # [t, q_ref, q_meas, dq_meas, tau_cmd]

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "FrictionIdNode started.\n"
            f"  action_topic: {self.action_topic}\n"
            f"  joint_states_topic: {self.joint_states_topic}\n"
            f"  torque_commands_topic: {self.torque_commands_topic}\n"
            f"  num_joints: {self.num_joints}, joint_index: {self.joint_index_param}, joint_name: '{self.joint_name_param}'\n"
            f"  sine: amp={amp_deg:.2f} deg, freq={self.freq_hz:.3f} Hz, offset={offset_deg:.2f} deg\n"
            f"  duration={self.duration_sec:.2f}s, publish_rate={self.publish_rate_hz:.1f} Hz, settle={self.settle_sec:.2f}s\n"
            f"  estimate_ls={self.estimate_ls}, mode={self.estimate_mode}, v_min={math.degrees(self.v_min_rad_s):.2f} deg/s, smooth_window={self.smooth_window}\n"
            f"  J_motor (motor_only)={self.J_motor:.6e}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self.buffers.joint_state = msg
        self.buffers.have_joint_state = True

        if self._joint_idx is None:
            self._joint_idx = self._resolve_joint_index(msg)

            if self.use_initial_positions and msg.position:
                q = np.asarray(msg.position, dtype=float).flatten()
                if q.size >= self.num_joints:
                    self._initial_q = q[: self.num_joints].copy()
                else:
                    self._initial_q = np.zeros(self.num_joints, dtype=float)

            if self._joint_idx is None:
                self.get_logger().warn("Could not resolve joint index; will keep trying...")

    def _on_torque_cmd(self, msg: Float32MultiArray) -> None:
        vec = np.asarray(msg.data, dtype=float).flatten()
        self.buffers.torque_cmd = vec
        self.buffers.have_torque = True

    def _resolve_joint_index(self, js: JointState) -> Optional[int]:
        # 1) by name
        if self.joint_name_param and js.name:
            try:
                idx = list(js.name).index(self.joint_name_param)
                self.get_logger().info(f"Resolved joint index by name '{self.joint_name_param}': {idx}")
                return idx
            except ValueError:
                pass
        # 2) by index
        if 0 <= self.joint_index_param < self.num_joints:
            self.get_logger().info(f"Using joint_index param: {self.joint_index_param}")
            return self.joint_index_param
        self.get_logger().error("Invalid joint index and joint_name not found.")
        return None

    def _tick(self) -> None:
        if self._done:
            return
        if not self.buffers.have_joint_state or self.buffers.joint_state is None:
            return
        if self._joint_idx is None:
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

        # Build desired positions (rad)
        if self.use_initial_positions and (self._initial_q is not None):
            desired = self._initial_q.copy()
        else:
            desired = np.zeros(self.num_joints, dtype=float)

        base = float(desired[self._joint_idx]) if self._joint_idx < desired.size else 0.0
        q_ref = base + self.offset_rad + self.amp_rad * math.sin(2.0 * math.pi * self.freq_hz * t_sec)

        if self._joint_idx < desired.size:
            desired[self._joint_idx] = q_ref

        # Publish to goat/action
        action_msg = Float32MultiArray()
        action_msg.data = desired.astype(np.float32).tolist()
        self.action_pub.publish(action_msg)

        # Logging after settle time and when torque exists
        if t_sec < self.settle_sec:
            return
        if not self.buffers.have_torque or self.buffers.torque_cmd is None:
            return

        js = self.buffers.joint_state
        q_meas, dq_meas = self._read_joint_state(js, self._joint_idx)
        tau_cmd = self._read_tau(self.buffers.torque_cmd, self._joint_idx)

        self._log_rows.append([t_sec, q_ref, q_meas, dq_meas, tau_cmd])

    @staticmethod
    def _read_joint_state(js: JointState, idx: int) -> Tuple[float, float]:
        q = float(js.position[idx]) if (js.position and idx < len(js.position)) else 0.0
        dq = float(js.velocity[idx]) if (js.velocity and idx < len(js.velocity)) else 0.0
        return q, dq

    @staticmethod
    def _read_tau(tau_vec: np.ndarray, idx: int) -> float:
        if idx < tau_vec.size:
            return float(tau_vec[idx])
        return float(tau_vec[-1]) if tau_vec.size > 0 else 0.0

    def _finish(self) -> None:
        self._done = True

        # Save CSV
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_filename:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = f"sysid_joint{int(self._joint_idx)}_{stamp}.csv"
        csv_path = self.log_dir / self.log_filename
        self._save_csv(csv_path)
        self.get_logger().info(f"Saved log to: {csv_path}")

        # Estimate
        if self.estimate_ls and len(self._log_rows) >= 20:
            rows = np.asarray(self._log_rows, dtype=float)
            result_text, result_yaml = self._estimate_with_inertia_ls(rows)

            self.get_logger().info(result_text)
            result_path = self._resolve_result_path(csv_path)
            result_path.write_text(result_yaml, encoding="utf-8")
            self.get_logger().info(f"Saved result to: {result_path}")
        else:
            self.get_logger().warn("Skipping estimation (estimate_ls=false or not enough samples).")

        self.get_logger().info("Experiment finished. Shutting down node.")
        rclpy.shutdown()

    def _save_csv(self, path: Path) -> None:
        header = ["t_sec", "q_ref_rad", "q_meas_rad", "dq_meas_rad_s", "tau_cmd_nm"]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in self._log_rows:
                w.writerow(row)

    # -----------------------------
    # NEW: inertia-aware LS fitting
    # -----------------------------
    def _estimate_with_inertia_ls(self, rows: np.ndarray) -> Tuple[str, str]:
        """
        Inertia-aware LS based on your offline script logic.

        rows columns:
          t, q_ref, q_meas, dq_meas, tau_cmd

        Returns:
          (human_readable_text, yaml_text)
        """
        t = rows[:, 0].astype(float)
        dq = rows[:, 3].astype(float)   # rad/s
        tau = rows[:, 4].astype(float)  # Nm (or your torque unit)

        dq_filt = self._moving_average(dq, self.smooth_window)
        qdd = np.gradient(dq_filt, t)  # rad/s^2

        # Remove near-zero velocities (same idea as vmin in your script)
        mask = np.abs(dq_filt) >= self.v_min_rad_s
        dq_u = dq_filt[mask]
        qdd_u = qdd[mask]
        tau_u = tau[mask]

        if dq_u.size < 10:
            txt = f"Not enough samples after filtering: {dq_u.size}"
            yml = "error: not_enough_samples\n"
            return txt, yml

        mode = self.estimate_mode.lower()

        if mode == "simple_fric":
            # tau ≈ a*dq + b*sign(dq)
            Phi = np.column_stack([dq_u, np.sign(dq_u)])
            theta, *_ = np.linalg.lstsq(Phi, tau_u, rcond=None)
            a_hat, b_hat = float(theta[0]), float(theta[1])

            txt = (
                "LS result (simple_fric): tau ≈ a*dq + b*sign(dq)\n"
                f"  a_hat: {a_hat:.6e} [Nm/(rad/s)]\n"
                f"  b_hat: {b_hat:.6e} [Nm]\n"
                f"  used_samples: {dq_u.size}/{len(dq)}\n"
                f"  v_min: {self.v_min_rad_s:.6e} rad/s ({math.degrees(self.v_min_rad_s):.2f} deg/s)\n"
                f"  smooth_window: {self.smooth_window}\n"
            )
            yml = "\n".join([
                "mode: simple_fric",
                "model: tau = a*dq + b*sign(dq)",
                "units:",
                "  dq: rad/s",
                "  tau: Nm",
                f"a_hat_nm_per_rad_s: {a_hat:.12f}",
                f"b_hat_nm: {b_hat:.12f}",
                f"used_samples: {int(dq_u.size)}",
                f"total_samples: {int(len(dq))}",
                f"v_min_rad_s: {self.v_min_rad_s:.12f}",
                f"smooth_window: {int(self.smooth_window)}",
                "",
            ])
            return txt, yml

        if mode == "motor_only":
            # tau* = tau - J_motor*qdd
            # tau* ≈ a*dq + b*sign(dq)
            tau_star = tau_u - self.J_motor * qdd_u
            Phi = np.column_stack([dq_u, np.sign(dq_u)])
            theta, *_ = np.linalg.lstsq(Phi, tau_star, rcond=None)
            a_hat, b_hat = float(theta[0]), float(theta[1])

            txt = (
                "LS result (motor_only): tau* = tau - J_motor*qdd, tau* ≈ a*dq + b*sign(dq)\n"
                f"  J_motor: {self.J_motor:.6e} [Nm/(rad/s^2)]\n"
                f"  a_hat: {a_hat:.6e} [Nm/(rad/s)]\n"
                f"  b_hat: {b_hat:.6e} [Nm]\n"
                f"  used_samples: {dq_u.size}/{len(dq)}\n"
                f"  v_min: {self.v_min_rad_s:.6e} rad/s ({math.degrees(self.v_min_rad_s):.2f} deg/s)\n"
                f"  smooth_window: {self.smooth_window}\n"
            )
            yml = "\n".join([
                "mode: motor_only",
                "model: tau_star = tau - J_motor*qdd; tau_star = a*dq + b*sign(dq)",
                "units:",
                "  dq: rad/s",
                "  qdd: rad/s^2",
                "  tau: Nm",
                f"J_motor_nm_per_rad_s2: {self.J_motor:.12f}",
                f"a_hat_nm_per_rad_s: {a_hat:.12f}",
                f"b_hat_nm: {b_hat:.12f}",
                f"used_samples: {int(dq_u.size)}",
                f"total_samples: {int(len(dq))}",
                f"v_min_rad_s: {self.v_min_rad_s:.12f}",
                f"smooth_window: {int(self.smooth_window)}",
                "",
            ])
            return txt, yml

        if mode == "full_joint":
            # tau ≈ J*qdd + a*dq + b*sign(dq)
            Phi = np.column_stack([qdd_u, dq_u, np.sign(dq_u)])
            theta, *_ = np.linalg.lstsq(Phi, tau_u, rcond=None)
            J_hat, a_hat, b_hat = float(theta[0]), float(theta[1]), float(theta[2])

            # (Optional) friction-only torque for sanity check: tau - J*qdd
            tau_fric_only = tau_u - J_hat * qdd_u
            # not plotted here, but we keep computed value for potential debug

            txt = (
                "LS result (full_joint): tau ≈ J*qdd + a*dq + b*sign(dq)\n"
                f"  J_hat: {J_hat:.6e} [Nm/(rad/s^2)]\n"
                f"  a_hat: {a_hat:.6e} [Nm/(rad/s)]\n"
                f"  b_hat: {b_hat:.6e} [Nm]\n"
                f"  used_samples: {dq_u.size}/{len(dq)}\n"
                f"  v_min: {self.v_min_rad_s:.6e} rad/s ({math.degrees(self.v_min_rad_s):.2f} deg/s)\n"
                f"  smooth_window: {self.smooth_window}\n"
                f"  (computed tau_fric_only = tau - J_hat*qdd for validation)\n"
            )
            yml = "\n".join([
                "mode: full_joint",
                "model: tau = J*qdd + a*dq + b*sign(dq)",
                "units:",
                "  dq: rad/s",
                "  qdd: rad/s^2",
                "  tau: Nm",
                f"J_hat_nm_per_rad_s2: {J_hat:.12f}",
                f"a_hat_nm_per_rad_s: {a_hat:.12f}",
                f"b_hat_nm: {b_hat:.12f}",
                f"used_samples: {int(dq_u.size)}",
                f"total_samples: {int(len(dq))}",
                f"v_min_rad_s: {self.v_min_rad_s:.12f}",
                f"smooth_window: {int(self.smooth_window)}",
                "",
            ])
            return txt, yml

        # Unknown mode fallback
        txt = f"Unknown estimate_mode: {self.estimate_mode}"
        yml = "error: unknown_mode\n"
        return txt, yml

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """Simple moving average smoothing (same spirit as your offline script)."""
        x = np.asarray(x, dtype=float)
        if window <= 1:
            return x
        window = int(window)
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(x, kernel, mode="same")

    def _resolve_result_path(self, csv_path: Path) -> Path:
        if self.result_path_param:
            p = Path(self.result_path_param)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        return csv_path.with_suffix(".yaml")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrictionIdNode()
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
