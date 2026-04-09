from __future__ import annotations

from typing import List, Optional, Tuple, Any

import numpy as np


class SafetyLimiter:
    """Unified safety gate: joint position check + velocity estop + torque LPF/clipping.

    Call apply() once per control cycle, after torque computation and before publishing.
    Receives pre-extracted numpy arrays from the node callback (no ROS2 message dependency).

    Block conditions (latching kill switch):
        Once triggered, is_blocked latches to True permanently until the process restarts.
        1. Any joint position is outside its soft limit (with margin).
        2. Any estop joint velocity exceeds the threshold.

    Torque filtering (always applied, regardless of is_blocked):
        LPF:      filtered = alpha * raw + (1 - alpha) * prev
        Clipping: output   = clip(filtered, -max, +max)

    YAML keys consumed:
        torque_lpf_alpha_per_joint   : list[float], length = num_joints
        max_torque_per_joint         : list[float], 0.0 = no clipping
        joint_pos_limit              : list[float], flat [lower0, upper0, ...], length = num_joints * 2
        joint_pos_limit_margin       : float [rad]
        joint_indices                : list[int], used as estop velocity check targets
        joint_vel_estop_threshold    : float [rad/s]
    """

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        joint_indices: List[int] = list(cfg["joint_indices"])
        self.num_joints: int = len(cfg["joint_names"])

        self.logger = logger

        # --- LPF alpha ---
        alpha_raw = np.asarray(cfg["torque_lpf_alpha_per_joint"], dtype=float).flatten()
        if alpha_raw.size != self.num_joints:
            raise ValueError("torque_lpf_alpha_per_joint length must equal num_joints.")
        if np.any(alpha_raw < 0.0) or np.any(alpha_raw > 1.0):
            raise ValueError("torque_lpf_alpha_per_joint values must be in [0, 1].")
        self._lpf_alpha = alpha_raw

        # --- Torque clipping ---
        max_raw = np.asarray(cfg["max_torque_per_joint"], dtype=float).flatten()
        if max_raw.size != self.num_joints:
            raise ValueError("max_torque_per_joint length must equal num_joints.")
        # 0 or negative -> treat as no clipping
        self._max_torque = np.where(max_raw > 0.0, max_raw, np.inf)

        # --- Joint position limits ---
        limits = np.asarray(cfg["joint_pos_limit"], dtype=float).flatten()
        if limits.size != self.num_joints * 2:
            raise ValueError("joint_pos_limit length must equal num_joints * 2.")
        margin = float(cfg.get("joint_pos_limit_margin", 0.0))
        margin_coeff = float(cfg.get("joint_pos_limit_margin_coeff", 1.0))

        # Coefficient-based margin processing
        self._pos_lower = limits[0::2] * margin_coeff
        self._pos_upper = limits[1::2] * margin_coeff
        self.logger.info(f"pos_lower : {self._pos_lower.tolist()}\r")
        self.logger.info(f"pos_upper : {self._pos_upper.tolist()}\r")
        # self._pos_lower = limits[0::2] + margin
        # self._pos_upper = limits[1::2] - margin

        # --- Velocity estop ---
        self._estop_indices = np.asarray(joint_indices, dtype=int)
        self._estop_threshold = float(cfg["joint_vel_estop_threshold"])

        # --- LPF state ---
        self._prev_torque = np.zeros(self.num_joints, dtype=float)

        # --- Kill switch (latching) ---
        # Once True, never resets. Requires process restart to clear.
        self._is_blocked = False

        # --- Motor gear ratio ---
        # Gear ratio is needed for checking safety margin with perspective of motor
        self.motor_gear_ratio = np.asarray(cfg["motor_gear_ratio"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset LPF state and kill switch. Call on mode switches or controller resets.
        NOTE: Reset the kill switch (_is_blocked) too.
        """
        self._prev_torque[:] = 0.0
        self._is_blocked = False

    def apply(self,
              raw_torque: np.ndarray,
              joint_pos: np.ndarray,
              joint_vel: np.ndarray,) -> Tuple[np.ndarray, bool]:
        """Apply safety checks and torque filtering.

        Args:
            raw_torque: Computed torque from the active controller (num_joints,).
            joint_pos:  Pre-extracted joint positions from callback [rad] (num_joints,).
            joint_vel:  Pre-extracted joint velocities from callback [rad/s] (num_joints,).

        Returns:
            out_torque:  Filtered torque, or zeros if blocked (num_joints,).
            is_blocked:  True if kill switch is active (latches permanently once triggered).
        """
        # Latching kill switch: once triggered, stays blocked forever
        if not self._is_blocked:
            self._is_blocked = (self._check_joint_pos(joint_pos) or self._check_joint_vel_estop(joint_vel))

        if self._is_blocked:
            self._prev_torque[:] = 0.0
            return np.zeros(self.num_joints, dtype=float), True

        # Normal path: LPF + clipping
        raw = np.asarray(raw_torque, dtype=float).flatten()
        filtered = self._lpf_alpha * raw + (1.0 - self._lpf_alpha) * self._prev_torque
        clipped = np.clip(filtered, -self._max_torque, self._max_torque)
        self._prev_torque[:] = clipped
        return clipped, False

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_joint_pos(self, pos: np.ndarray) -> bool:
        """Return True if any joint position is outside its allowed range."""
        motor_pos = pos / self.motor_gear_ratio
        result = bool(np.any(motor_pos < self._pos_lower) or np.any(motor_pos > self._pos_upper))
        if result:
            self.logger.info("Joint pos stop.\r")
            self.logger.info(f"Limiter Results: {np.logical_or((motor_pos < self._pos_lower), (motor_pos > self._pos_upper).tolist())}\r")
            self.logger.info(f"Joint pos : {motor_pos.tolist()}\r")
        return result

    def _check_joint_vel_estop(self, vel: np.ndarray) -> bool:
        """Return True if any estop joint exceeds the velocity threshold."""
        if self._estop_indices.size == 0:
            return False
        motor_vel = vel / self.motor_gear_ratio
        result = bool(np.any(np.abs(motor_vel[self._estop_indices]) > self._estop_threshold))
        if result:
            self.logger.info("Joint vel stop.\r")
            self.logger.info(f"Limiter Results: {np.abs(motor_vel[self._estop_indices]) > self._estop_threshold}.\r")
            self.logger.info(f"Joint vel: {motor_vel.tolist()}.\r")
        return result
