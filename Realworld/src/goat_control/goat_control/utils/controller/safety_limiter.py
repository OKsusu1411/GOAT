from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple, Any

import numpy as np


class SafetyLimiter:
    """Unified safety gate: joint position check + velocity estop + torque LPF/clipping.

    Call apply() once per control cycle, after torque computation and before publishing.
    Receives pre-extracted numpy arrays from the node callback (no ROS2 message dependency).

    Block conditions (latching kill switch):
        Once triggered, is_blocked latches to True permanently until the process restarts.
        1. Any joint position is outside its soft limit (with margin).
        2. Any estop joint's windowed mean *speed* exceeds the threshold. The window
           spans joint_vel_estop_sample_num ticks and averages |v|, so a single noisy
           sample cannot latch the switch while a sign-flipping oscillation still does.

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
        joint_vel_estop_sample_num   : int, estop averaging window length [ticks]
    """

    def __init__(self, cfg: dict, logger: Any | None) -> None:
        self.joint_indices: List[int] = list(cfg["self.joint_indices"])
        self.num_joints: int = len(cfg["joint_names"])

        self.logger = logger

        # --- Joint position limits ---
        limits = np.asarray(cfg["joint_pos_limit"], dtype=float).flatten()
        if limits.size != self.num_joints * 2:
            raise ValueError("joint_pos_limit length must equal num_joints * 2.")
        margin_coeff = float(cfg.get("joint_pos_limit_margin_coeff", 1.0))

        # Coefficient-based margin processing
        self._pos_lower = limits[0::2] * margin_coeff
        self._pos_upper = limits[1::2] * margin_coeff
        self.logger.info(f"pos_lower : {self._pos_lower.tolist()}\r")
        self.logger.info(f"pos_upper : {self._pos_upper.tolist()}\r")

        # --- Velocity estop ---
        self._estop_indices = np.asarray(self.joint_indices, dtype=int)
        self._estop_threshold = float(cfg["joint_vel_estop_threshold"])
        self._estop_sample_num = int(cfg.get("joint_vel_estop_sample_num", 1))
        self._vel_buffer: deque[np.ndarray] = deque(np.zeros((self._estop_sample_num, len(self.joint_indices))) ,maxlen=self._estop_sample_num)

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
        """Reset LPF state, kill switch, and the estop velocity window.
        Call on mode switches or controller resets.
        NOTE: Reset the kill switch (_is_blocked) too.

        Clearing the window here is what keeps a mode switch from judging the new
        controller with the previous one's velocity samples.
        """
        self._prev_torque[:] = 0.0
        self._is_blocked = False
        self._vel_buffer.clear()
        self._vel_buffer: deque[np.ndarray] = deque(np.zeros((self._estop_sample_num, len(self.joint_indices))) ,maxlen=self._estop_sample_num)

    def apply(self,
              raw_torque: np.ndarray,
              joint_pos: np.ndarray,
              joint_vel: np.ndarray,) -> Tuple[np.ndarray, bool]:
        """Apply safety checks and torque filtering.

        Args:
            raw_torque: Computed torque from the active controller (num_joints,).
            joint_pos:  Pre-extracted joint positions from callback [rad] (num_joints,).
            joint_vel:  Pre-extracted joint velocities from callback [rad/s] (num_joints,).
                        Buffered internally over joint_vel_estop_sample_num ticks.

        Returns:
            out_torque:  Filtered torque, or zeros if blocked (num_joints,).
            is_blocked:  True if kill switch is active (latches permanently once triggered).
        """
        # Latching kill switch: once triggered, stays blocked forever
        if not self._is_blocked:
            self._is_blocked = (self._check_joint_pos(joint_pos) or self._check_joint_vel_estop(joint_vel))
            # self._is_blocked = self._check_joint_vel_estop()

        if self._is_blocked:
            self._prev_torque[:] = 0.0
            return np.zeros(self.num_joints, dtype=float), True

        raw = raw_torque
        return raw, False

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_joint_pos(self, pos: np.ndarray) -> bool:
        """Return True if any joint position is outside its allowed range."""
        joint_pos = pos
        result = bool(np.any(joint_pos < self._pos_lower) or np.any(joint_pos > self._pos_upper))
        if result:
            self.logger.info("[SafetyLimiter] Position limiter activation.\r")
            self.logger.info(f"Limiter Results: {np.logical_or((joint_pos < self._pos_lower), (joint_pos > self._pos_upper).tolist())}\r")
            self.logger.info(f"Joint pos : {joint_pos.tolist()}\r")
        return result

    def _check_joint_vel_estop(self, vel: np.ndarray) -> bool:
        """Return True if any estop joint's windowed speed exceeds the threshold."""
        joint_vel = np.asarray(vel, dtype=float).flatten()
        self._vel_buffer.append(joint_vel)
        vel_buffer_np = np.asarray(self._vel_buffer, dtype=float)   # (sample_num, num_joints)
        vel_mean = np.mean(np.abs(vel_buffer_np), axis=0)         # [rad/s]
        over_threshold = vel_mean[self._estop_indices] > self._estop_threshold

        result = bool(np.any(over_threshold))
        if result:
            self.logger.info("[SafetyLimiter] Velocity limiter activation.\r")
            self.logger.info(f"Limiter Results: {over_threshold.tolist()}.\r")
            self.logger.info(f"Joint vel (windowed mean speed): "
                             f"{vel_mean[self._estop_indices].tolist()}.\r")
        return result
