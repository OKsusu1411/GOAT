#!/usr/bin/env python3
"""goat_sysid friction identification node (goat_control 'nodes' topic convention).

Why this exists
--------------
In the newer goat_control stack inside this repository, the main control node
(`goat_control/nodes/control_node.py`) consumes:
  - /joint_states (sensor_msgs/JointState)
  - /goat/action  (std_msgs/Float32MultiArray)
and outputs:
  - /torque_commands (std_msgs/Float32MultiArray)

So for system identification we should NOT depend on /motor_states (legacy).
This node publishes a sinusoidal joint position command on /goat/action,
subscribes to /joint_states and /torque_commands, logs data to CSV, and can
optionally fit a simple dynamic friction model via least squares.

Friction model (per joint)
--------------------------
    tau = b * dq + tau_c * sign(dq)

where:
  - dq is joint velocity [rad/s]
  - tau is torque [Nm]
  - b is viscous friction coefficient [Nms/rad]
  - tau_c is Coulomb friction torque [Nm]
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


@dataclass
class _Latest:
    joint_state: Optional[JointState] = None
    torque_cmd: Optional[np.ndarray] = None


def fit_dynamic_friction_ls(
    dq_rad_s: np.ndarray,
    tau_nm: np.ndarray,
    v_min_rad_s: float = 0.1,
) -> Tuple[float, float, int]:
    """Fit tau = b*dq + tau_c*sign(dq) using least squares.

    Returns: (b, tau_c, used_samples)
    """
    dq = np.asarray(dq_rad_s, dtype=float).reshape(-1)
    tau = np.asarray(tau_nm, dtype=float).reshape(-1)
    if dq.size != tau.size:
        n = min(dq.size, tau.size)
        dq = dq[:n]
        tau = tau[:n]

    mask = np.isfinite(dq) & np.isfinite(tau) & (np.abs(dq) >= float(v_min_rad_s))
    dq_f = dq[mask]
    tau_f = tau[mask]
    used = int(dq_f.size)
    if used < 10:
        raise ValueError(f"Too few samples after filtering (used={used}).")

    # Design matrix: [dq, sign(dq)]
    A = np.stack([dq_f, np.sign(dq_f)], axis=1)
    x, *_ = np.linalg.lstsq(A, tau_f, rcond=None)
    b = float(x[0])
    tau_c = float(x[1])
    return b, tau_c, used


class FrictionIdNode(Node):
    def __init__(self):
        super().__init__("friction_id_node")

        # --- Topics (match goat_control/nodes/control_node.py defaults) ---
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("joint_state_topic", "joint_states")
        self.declare_parameter("torque_commands_topic", "torque_commands")

        # --- Experiment params ---
        self.declare_parameter("num_joints", 8)
        self.declare_parameter("joint_index", 1)
        self.declare_parameter("amplitude_deg", 50.0)
        self.declare_parameter("frequency_hz", 0.5)
        self.declare_parameter("offset_deg", 0.0)
        self.declare_parameter("control_frequency_hz", 200.0)
        self.declare_parameter("duration_sec", 120.0)

        # --- Logging / estimation ---
        self.declare_parameter("save_path", "")
        self.declare_parameter("estimate_ls", True)
        self.declare_parameter("v_min_rad_s", 0.1)
        self.declare_parameter("result_path", "")

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.torque_commands_topic = str(self.get_parameter("torque_commands_topic").value)

        self.num_joints = int(self.get_parameter("num_joints").value)
        self.joint_index = int(self.get_parameter("joint_index").value)

        self.amplitude_rad = np.deg2rad(float(self.get_parameter("amplitude_deg").value))
        self.frequency_hz = float(self.get_parameter("frequency_hz").value)
        self.offset_rad = np.deg2rad(float(self.get_parameter("offset_deg").value))

        self.control_frequency_hz = float(self.get_parameter("control_frequency_hz").value)
        self.dt = 1.0 / max(self.control_frequency_hz, 1.0)
        self.duration_sec = float(self.get_parameter("duration_sec").value)

        save_path = str(self.get_parameter("save_path").value)
        if not save_path:
            ts = int(time.time())
            save_path = f"./test_log/friction_sysid_joint{self.joint_index}_{ts}.csv"
        self.save_path = save_path

        self.estimate_ls = bool(self.get_parameter("estimate_ls").value)
        self.v_min_rad_s = float(self.get_parameter("v_min_rad_s").value)
        self.result_path = str(self.get_parameter("result_path").value)
        if (not self.result_path) and self.estimate_ls:
            ts = int(time.time())
            self.result_path = f"./test_log/friction_params_joint{self.joint_index}_{ts}.yaml"

        if not (0 <= self.joint_index < self.num_joints):
            raise ValueError(f"joint_index must be in [0, {self.num_joints-1}] (got {self.joint_index}).")

        self.latest = _Latest()
        self._t0_wall = time.time()
        self._finished = False

        # Log buffer columns:
        #   t, q_ref_rad, q_meas_rad, dq_meas_rad_s, tau_cmd_nm, tau_meas_effort
        self._log_rows: list[list[float]] = []

        # Pub/Sub
        self.action_pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 50)
        self.create_subscription(Float32MultiArray, self.torque_commands_topic, self._on_torque_cmd, 50)

        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            "FrictionIdNode started.\n"
            f"  publish : {self.action_topic}\n"
            f"  subscribe: {self.joint_state_topic}, {self.torque_commands_topic}\n"
            f"  joint_index={self.joint_index}, A={np.rad2deg(self.amplitude_rad):.3f} deg, "
            f"f={self.frequency_hz:.3f} Hz, offset={np.rad2deg(self.offset_rad):.3f} deg\n"
            f"  duration={self.duration_sec:.1f}s, log={self.save_path}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self.latest.joint_state = msg

    def _on_torque_cmd(self, msg: Float32MultiArray) -> None:
        vec = np.asarray(msg.data, dtype=float).reshape(-1)
        if vec.size < self.num_joints:
            padded = np.zeros(self.num_joints, dtype=float)
            padded[: vec.size] = vec
            vec = padded
        elif vec.size > self.num_joints:
            vec = vec[: self.num_joints]
        self.latest.torque_cmd = vec

    def _publish_action(self, q_ref_rad: float) -> None:
        # control_node expects: first num_joints = desired joint position [rad]
        action = np.zeros(self.num_joints, dtype=np.float32)
        action[self.joint_index] = float(q_ref_rad)
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)

    def _tick(self) -> None:
        if self._finished:
            return

        t = time.time() - self._t0_wall
        if t >= self.duration_sec:
            self.get_logger().info("Duration reached. Saving log and (optionally) estimating friction...")
            self._finalize()
            self._finished = True
            return

        # 1) Reference (sinusoid) in radians
        q_ref = self.offset_rad + self.amplitude_rad * np.sin(2.0 * np.pi * self.frequency_hz * t)
        self._publish_action(q_ref)

        # 2) Log when we have both state and torque
        js = self.latest.joint_state
        tau_cmd = self.latest.torque_cmd
        if js is None or tau_cmd is None:
            return

        # Ensure arrays exist
        q = np.asarray(js.position or [], dtype=float)
        dq = np.asarray(js.velocity or [], dtype=float)
        eff = np.asarray(js.effort or [], dtype=float)

        if q.size <= self.joint_index or dq.size <= self.joint_index:
            return

        q_meas = float(q[self.joint_index])
        dq_meas = float(dq[self.joint_index])
        tau = float(tau_cmd[self.joint_index])
        tau_eff = float(eff[self.joint_index]) if eff.size > self.joint_index else float("nan")

        self._log_rows.append([t, float(q_ref), q_meas, dq_meas, tau, tau_eff])

    def _finalize(self) -> None:
        if not self._log_rows:
            self.get_logger().warn("No samples logged. Nothing to save.")
            return

        arr = np.asarray(self._log_rows, dtype=float)
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        header = "t_sec,q_ref_rad,q_meas_rad,dq_meas_rad_s,tau_cmd_nm,tau_meas_effort"
        np.savetxt(self.save_path, arr, delimiter=",", header=header, comments="")
        self.get_logger().info(f"Saved CSV: {self.save_path} (rows={arr.shape[0]}).")

        if not self.estimate_ls:
            return

        dq = arr[:, 3]
        tau = arr[:, 4]
        try:
            b, tau_c, used = fit_dynamic_friction_ls(dq, tau, v_min_rad_s=self.v_min_rad_s)
        except Exception as exc:
            self.get_logger().warn(f"LS estimation failed: {exc}")
            return

        self.get_logger().info(
            "Estimated dynamic friction (tau = b*dq + tau_c*sign(dq)):\n"
            f"  joint_index: {self.joint_index}\n"
            f"  b (viscous)  : {b:.6f}  [Nms/rad]\n"
            f"  tau_c (coulomb): {tau_c:.6f}  [Nm]\n"
            f"  used_samples : {used} (v_min={self.v_min_rad_s} rad/s)"
        )

        # Save YAML-like text (no external deps)
        if self.result_path:
            os.makedirs(os.path.dirname(self.result_path) or ".", exist_ok=True)
            with open(self.result_path, "w", encoding="utf-8") as f:
                f.write("# friction identification result\n")
                f.write(f"joint_index: {self.joint_index}\n")
                f.write("model: tau = b*dq + tau_c*sign(dq)\n")
                f.write(f"b_Nms_per_rad: {b:.12g}\n")
                f.write(f"tau_c_Nm: {tau_c:.12g}\n")
                f.write(f"v_min_rad_s: {self.v_min_rad_s}\n")
                f.write(f"log_csv: {self.save_path}\n")
            self.get_logger().info(f"Saved result: {self.result_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrictionIdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save partial logs on interrupt
        if hasattr(node, "_finished") and (not node._finished):
            node.get_logger().info("Interrupted. Saving partial log...")
            node._finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
