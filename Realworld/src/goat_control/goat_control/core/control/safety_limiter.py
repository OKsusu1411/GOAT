# goat_control/core/control/safety_limiter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from ..estimation.state_types import RobotState


@dataclass
class SafetyLimiterConfig:
    """Safety layer for torque/effort command.

    - lpf_alpha_per_joint:
        * shape (num_joints,)
        * 0~1
        * 1.0 -> no filtering (raw passes through)
        * close to 1.0 -> light filtering
    - max_torque_per_joint:
        * shape (num_joints,)
        * positive values
    """
    num_joints: int
    lpf_alpha_per_joint: Optional[Sequence[float]] = None
    max_torque_per_joint: Optional[Sequence[float]] = None
    joint_pos_limit: Optional[Sequence[float]] = None
    joint_pos_limit_margin: float = 0.0
    joint_vel_limit: Optional[Sequence[float]] = None
    joint_vel_limit_margin: float = 0.0


class TorqueSafetyLimiter:
    """Apply LPF + per-joint torque clipping."""

    def __init__(self, config: SafetyLimiterConfig):
        self.num_joints = int(config.num_joints)

        if config.lpf_alpha_per_joint is None:
            # Default: no filtering
            self.lpf_alpha_per_joint = np.ones(self.num_joints, dtype=float)
        else:
            self.lpf_alpha_per_joint = np.asarray(config.lpf_alpha_per_joint, dtype=float).flatten()
            if self.lpf_alpha_per_joint.size != self.num_joints:
                raise ValueError("lpf_alpha_per_joint length must match num_joints.")
            if np.any(self.lpf_alpha_per_joint < 0.0) or np.any(self.lpf_alpha_per_joint > 1.0):
                raise ValueError("lpf_alpha_per_joint values must be in [0, 1].")

        if config.max_torque_per_joint is None:
            # Default: no clipping (use very large)
            self.max_torque_per_joint = np.full(self.num_joints, 1e9, dtype=float)
        else:
            self.max_torque_per_joint = np.asarray(config.max_torque_per_joint, dtype=float).flatten()
            if self.max_torque_per_joint.size != self.num_joints:
                raise ValueError("max_torque_per_joint length must match num_joints.")
            if np.any(self.max_torque_per_joint < 0.0):
                raise ValueError("max_torque_per_joint must be positive.")

        self.previous_torque_command = np.zeros(self.num_joints, dtype=float)

    def reset(self, previous_torque_command: Optional[np.ndarray] = None) -> None:
        if previous_torque_command is None:
            self.previous_torque_command[:] = 0.0
        else:
            previous = np.asarray(previous_torque_command, dtype=float).flatten()
            if previous.size != self.num_joints:
                raise ValueError("previous_torque_command length must match num_joints.")
            self.previous_torque_command[:] = previous

    def apply(self, raw_torque_command: np.ndarray) -> np.ndarray:
        """Apply LPF then per-joint clipping."""
        raw = np.asarray(raw_torque_command, dtype=float).flatten()
        if raw.size != self.num_joints:
            raise ValueError("raw_torque_command length must match num_joints.")

        # Low-pass filter (vector alpha)
        filtered = (
            self.lpf_alpha_per_joint * raw
            + (1.0 - self.lpf_alpha_per_joint) * self.previous_torque_command
        )
        self.previous_torque_command[:] = filtered

        # Per-joint clipping
        clipped = np.clip(filtered, -self.max_torque_per_joint, self.max_torque_per_joint)

        return clipped
    

# ---------------------------------------------------------------------
# Joint position + velocity safety lock limiter 
# ---------------------------------------------------------------------

class JointSafetyLimiter:
    """Apply joint position limit lock after calculate target position"""
    def __init__(self, config: SafetyLimiterConfig):
        self.num_joints = int(config.num_joints)

        # Joint limit
        self.joint_pos_limit = np.asarray(config.joint_pos_limit, dtype=float).flatten()
        self.joint_vel_limit = np.asarray(config.joint_vel_limit, dtype=float).flatten()

         # Exception
        if self.joint_pos_limit.size != self.num_joints * 2:
            raise ValueError("joint_pos_limit length must match num_joints * 2.")
        if self.joint_vel_limit.size != self.num_joints:
            raise ValueError("joint_vel_limit length must match num_joints.")
        
        # Margin considering
        joint_pos_limit_margin = config.joint_pos_limit_margin
        joint_vel_limit_margin = config.joint_vel_limit_margin

        self.joint_pos_lower_limits = self.joint_pos_limit[0::2] + joint_pos_limit_margin
        self.joint_pos_upper_limits = self.joint_pos_limit[1::2] - joint_pos_limit_margin
        self.joint_vel_limit -= joint_vel_limit_margin

        # Violation boolean
        self.has_violation = False

    def apply(self,
              robot_state: RobotState,
              target_joint_delta_position: np.array,
              target_joint_velocity: np.array,):
        
        current_joint_pos = robot_state.joint_position_rad

        safe_delta_pos = target_joint_delta_position.copy()
        safe_vel = target_joint_velocity.copy()

        # Examine violation
        pos_lower_violation_mask = (current_joint_pos <= self.joint_pos_lower_limits)# & (safe_delta_pos < 0)
        pos_upper_violation_mask = (current_joint_pos >= self.joint_pos_upper_limits)# & (safe_delta_pos > 0)
        vel_stop_mask = pos_lower_violation_mask | pos_upper_violation_mask         # Currently not used
        self.has_violation = bool((pos_lower_violation_mask | pos_upper_violation_mask).any())

        # Clipping
        safe_delta_pos[pos_lower_violation_mask] = 0.0
        safe_delta_pos[pos_upper_violation_mask] = 0.0
        safe_vel = np.clip(safe_vel, -self.joint_vel_limit, self.joint_vel_limit)
        safe_vel[vel_stop_mask] = 0.0                                               # Currently not used
        
        return safe_delta_pos, safe_vel, self.has_violation
        

# ---------------------------------------------------------------------
# Conditional integration anti-windup for PI controller (wheel indices)
# ---------------------------------------------------------------------

@dataclass
class ConditionalIntegratorConfig:
    """Conditional integration (anti-windup) configuration."""
    num_joints: int
    controlled_indices: Sequence[int]          # e.g., wheel indices [6,7]
    integrator_state_limit: float = 0.0        # 0 -> no clamp
    output_limit_per_joint: Optional[Sequence[float]] = None  # per-joint saturation limit


class ConditionalIntegratorAntiWindup:
    """Anti-windup helper implementing *conditional integration*.

    Rule:
    - Compute PI with candidate integrator update.
    - If output saturates AND the error would push further into saturation,
      reject the integrator update for that joint (freeze integrator).
    """

    def __init__(self, config: ConditionalIntegratorConfig):
        self.num_joints = int(config.num_joints)
        self.controlled_indices = list(config.controlled_indices)
        self.integrator_state_limit = float(config.integrator_state_limit)

        if config.output_limit_per_joint is None:
            # Default: practically no saturation
            self.output_limit_per_joint = np.full(self.num_joints, 1e9, dtype=float)
        else:
            raw_limits = np.asarray(config.output_limit_per_joint, dtype=float).flatten()
            wheel_count = len(self.controlled_indices)

            # Accept either:
            #  - full length vector (num_joints)
            #  - wheel-only vector (len(controlled_indices))
            if raw_limits.size == self.num_joints:
                full_limits = raw_limits.copy()
            elif raw_limits.size == wheel_count:
                # Expand wheel-only limits to full length
                full_limits = np.full(self.num_joints, 1e9, dtype=float)
                for joint_index, limit_value in zip(self.controlled_indices, raw_limits):
                    full_limits[int(joint_index)] = float(limit_value)
            else:
                raise ValueError(
                    f"output_limit_per_joint length must match num_joints({self.num_joints}) "
                    f"or wheel_count({wheel_count})."
                )

            # Validate only controlled joints (wheels)
            controlled_limits = full_limits[np.asarray(self.controlled_indices, dtype=int)]
            if np.any(controlled_limits <= 0.0):
                raise ValueError("output_limit_per_joint must be positive for controlled_indices.")

            self.output_limit_per_joint = full_limits

        self.integrator_state = np.zeros(self.num_joints, dtype=float)

    def reset(self) -> None:
        self.integrator_state[:] = 0.0

    def step(
        self,
        error: np.ndarray,
        proportional_gain_full: np.ndarray,
        integral_gain_full: np.ndarray,
        dt_sec: float,
    ) -> np.ndarray:
        """Return saturated PI output with conditional integration.

        Inputs are full-length arrays (num_joints,).
        Only controlled_indices will be active; others return 0.
        """
        error = np.asarray(error, dtype=float).flatten()
        proportional_gain_full = np.asarray(proportional_gain_full, dtype=float).flatten()
        integral_gain_full = np.asarray(integral_gain_full, dtype=float).flatten()

        if error.size != self.num_joints:
            raise ValueError("error length must match num_joints.")
        if proportional_gain_full.size != self.num_joints:
            raise ValueError("proportional_gain_full length must match num_joints.")
        if integral_gain_full.size != self.num_joints:
            raise ValueError("integral_gain_full length must match num_joints.")

        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")

        output = np.zeros(self.num_joints, dtype=float)

        for joint_index in self.controlled_indices:
            proportional_term = proportional_gain_full[joint_index] * error[joint_index]

            # Candidate integrator update
            candidate_integrator = self.integrator_state[joint_index] + error[joint_index] * dt_sec

            # Optional integrator clamp
            if self.integrator_state_limit > 0.0:
                candidate_integrator = float(
                    np.clip(candidate_integrator, -self.integrator_state_limit, self.integrator_state_limit)
                )

            candidate_output_unsat = proportional_term + integral_gain_full[joint_index] * candidate_integrator

            # Saturate output
            output_limit = self.output_limit_per_joint[joint_index]
            candidate_output_sat = float(np.clip(candidate_output_unsat, -output_limit, output_limit))

            is_saturated = (candidate_output_sat != candidate_output_unsat)

            if is_saturated:
                # If saturation happened, decide whether to accept integrator update.
                # Reject integration if error pushes further into saturation direction.
                pushing_positive_saturation = (candidate_output_unsat > output_limit) and (error[joint_index] > 0.0)
                pushing_negative_saturation = (candidate_output_unsat < -output_limit) and (error[joint_index] < 0.0)

                if pushing_positive_saturation or pushing_negative_saturation:
                    # Freeze integrator (reject update)
                    candidate_integrator = self.integrator_state[joint_index]
                    candidate_output_unsat = proportional_term + integral_gain_full[joint_index] * candidate_integrator
                    candidate_output_sat = float(np.clip(candidate_output_unsat, -output_limit, output_limit))

            # Commit integrator
            self.integrator_state[joint_index] = candidate_integrator
            output[joint_index] = candidate_output_sat

        return output
