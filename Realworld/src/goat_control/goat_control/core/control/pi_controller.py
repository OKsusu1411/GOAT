# goat_control/core/control/pi_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class WheelPIControllerConfig:
    """Wheel speed PI controller configuration (rad/s domain)."""
    proportional_gain: np.ndarray  # shape: (num_joints,) or (num_wheels,)
    integral_gain: np.ndarray      # shape: (num_joints,) or (num_wheels,)
    wheel_indices: Sequence[int]   # typically [6, 7]

    # Anti-windup: integrator state clamp
    # - 0.0 -> no clamp
    integrator_state_limit: float = 0.0

    # Output saturation per joint (used for conditional integration anti-windup)
    # Accept either:
    #  - full length vector (num_joints)
    #  - wheel-only vector (len(wheel_indices))
    # If None: practically no saturation.
    output_limit_per_joint: Optional[Sequence[float]] = None

    # Backward compatibility (older configs may set a single scalar limit)
    output_limit: Optional[float] = None


class WheelPIController:
    """Wheel speed PI controller for specified wheel indices.

    - Input units:
        wheel_speed_reference_rad_per_sec: [rad/s]
        wheel_speed_measured_rad_per_sec: [rad/s]
    - Output:
        torque_command: array (num_joints,) with PI applied only at wheel indices
    """

    def __init__(self, config: WheelPIControllerConfig, num_joints: int):
        self.config = config
        self.num_joints = int(num_joints)

        self.wheel_indices = list(config.wheel_indices)

        p_gain = np.asarray(config.proportional_gain, dtype=float).flatten()
        i_gain = np.asarray(config.integral_gain, dtype=float).flatten()

        # Allow either full-length (num_joints) arrays or wheel-only arrays (len == num_wheels)
        if p_gain.size == self.num_joints:
            self.p_gain_full = p_gain
        elif p_gain.size == len(self.wheel_indices):
            self.p_gain_full = np.zeros(self.num_joints, dtype=float)
            for wheel_local_index, wheel_global_index in enumerate(self.wheel_indices):
                self.p_gain_full[wheel_global_index] = p_gain[wheel_local_index]
        else:
            raise ValueError("p_gain must be length num_joints or num_wheels.")

        if i_gain.size == self.num_joints:
            self.i_gain_full = i_gain
        elif i_gain.size == len(self.wheel_indices):
            self.i_gain_full = np.zeros(self.num_joints, dtype=float)
            for wheel_local_index, wheel_global_index in enumerate(self.wheel_indices):
                self.i_gain_full[wheel_global_index] = i_gain[wheel_local_index]
        else:
            raise ValueError("i_gain must be length num_joints or num_wheels.")

        # Integrator state per joint (we will only use wheel indices)
        self.integrator_state = np.zeros(self.num_joints, dtype=float)

        self.integrator_state_limit = float(config.integrator_state_limit)

        # Build per-joint output limits (full length)
        if config.output_limit_per_joint is None:
            if config.output_limit is None:
                self.output_limit_per_joint = np.full(self.num_joints, 1e9, dtype=float)
            else:
                # Scalar limit -> apply only to wheels
                scalar = float(config.output_limit)
                if scalar <= 0.0:
                    raise ValueError("output_limit must be positive.")
                self.output_limit_per_joint = np.full(self.num_joints, 1e9, dtype=float)
                for idx in self.wheel_indices:
                    self.output_limit_per_joint[int(idx)] = scalar
        else:
            raw_limits = np.asarray(list(config.output_limit_per_joint), dtype=float).flatten()
            if raw_limits.size == self.num_joints:
                full_limits = raw_limits.copy()
            elif raw_limits.size == len(self.wheel_indices):
                full_limits = np.full(self.num_joints, 1e9, dtype=float)
                for idx, limit_value in zip(self.wheel_indices, raw_limits):
                    full_limits[int(idx)] = float(limit_value)
            else:
                raise ValueError(
                    f"output_limit_per_joint length must match num_joints({self.num_joints}) "
                    f"or wheel_count({len(self.wheel_indices)})."
                )

            wheel_limits = full_limits[np.asarray(self.wheel_indices, dtype=int)]
            if np.any(wheel_limits <= 0.0):
                raise ValueError("output_limit_per_joint must be positive for wheel_indices.")
            self.output_limit_per_joint = full_limits

    def reset(self) -> None:
        """Reset integrator state."""
        self.integrator_state[:] = 0.0

    def compute(
        self,
        wheel_speed_reference_rad_per_sec: np.ndarray,
        wheel_speed_measured_rad_per_sec: np.ndarray,
        dt_sec: float,
    ) -> np.ndarray:
        """Compute PI output torque/effort for wheel indices."""
        wheel_speed_reference_rad_per_sec = np.asarray(wheel_speed_reference_rad_per_sec, dtype=float).flatten()
        wheel_speed_measured_rad_per_sec = np.asarray(wheel_speed_measured_rad_per_sec, dtype=float).flatten()

        if wheel_speed_reference_rad_per_sec.size != self.num_joints or wheel_speed_measured_rad_per_sec.size != self.num_joints:
            raise ValueError("wheel_speed_reference_rad_per_sec and wheel_speed_measured_rad_per_sec must be length num_joints.")

        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")

        speed_error_rad_per_sec = wheel_speed_reference_rad_per_sec - wheel_speed_measured_rad_per_sec

        torque_command = np.zeros(self.num_joints, dtype=float)

        for wheel_index in self.wheel_indices:
            wheel_index = int(wheel_index)
            error = float(speed_error_rad_per_sec[wheel_index])

            proportional_term = float(self.p_gain_full[wheel_index] * error)

            # Candidate integrator update
            candidate_integrator = float(self.integrator_state[wheel_index] + error * dt_sec)

            # Optional integrator clamp
            if self.integrator_state_limit > 0.0:
                candidate_integrator = float(
                    np.clip(candidate_integrator, -self.integrator_state_limit, self.integrator_state_limit)
                )

            candidate_output_unsat = proportional_term + float(self.i_gain_full[wheel_index] * candidate_integrator)

            # Saturate output
            output_limit = float(self.output_limit_per_joint[wheel_index])
            candidate_output_sat = float(np.clip(candidate_output_unsat, -output_limit, output_limit))

            is_saturated = (candidate_output_sat != candidate_output_unsat)
            if is_saturated:
                # Conditional integration: freeze integrator if the error pushes further into saturation.
                pushing_positive = (candidate_output_unsat > output_limit) and (error > 0.0)
                pushing_negative = (candidate_output_unsat < -output_limit) and (error < 0.0)
                if pushing_positive or pushing_negative:
                    candidate_integrator = float(self.integrator_state[wheel_index])
                    candidate_output_unsat = proportional_term + float(self.i_gain_full[wheel_index] * candidate_integrator)
                    candidate_output_sat = float(np.clip(candidate_output_unsat, -output_limit, output_limit))

            # Commit integrator
            self.integrator_state[wheel_index] = candidate_integrator
            torque_command[wheel_index] = candidate_output_sat

        return torque_command
