# goat_control/core/control/pd_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class PDControllerConfig:
    """Per-joint PD gains configuration (rad / rad/s domain)."""
    proportional_gain: np.ndarray  # shape: (num_joints,)
    derivative_gain: np.ndarray    # shape: (num_joints,)
    joint_indices: Sequence[int]   # indices where PD is applied (e.g., [0..5])


class PDJointController:
    """Position PD controller for specified joint indices.

    - Input/Output units:
        target_position_rad: [rad]
        current_position_rad: [rad]
        current_velocity_rad_per_sec: [rad/s]
        desired_velocity_rad_per_sec: [rad/s] (optional, default 0)
        output_torque_command: controller effort (your system interprets it as torque)
    """

    def __init__(self, config: PDControllerConfig):
        self.config = config

        self.p_gain = np.asarray(config.proportional_gain, dtype=float).flatten()
        self.d_gain = np.asarray(config.derivative_gain, dtype=float).flatten()
        self.joint_indices = list(config.joint_indices)

        if self.p_gain.shape != self.d_gain.shape:
            raise ValueError("p_gain and d_gain must have the same shape.")

    def compute(
        self,
        target_position_rad: np.ndarray,
        current_position_rad: np.ndarray,
        current_velocity_rad_per_sec: np.ndarray,
        desired_velocity_rad_per_sec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute PD output torque/effort for the configured joint indices."""
        target_position_rad = np.asarray(target_position_rad, dtype=float).flatten()
        current_position_rad = np.asarray(current_position_rad, dtype=float).flatten()
        current_velocity_rad_per_sec = np.asarray(current_velocity_rad_per_sec, dtype=float).flatten()

        if desired_velocity_rad_per_sec is None:
            desired_velocity_rad_per_sec = np.zeros_like(current_velocity_rad_per_sec)
        else:
            desired_velocity_rad_per_sec = np.asarray(desired_velocity_rad_per_sec, dtype=float).flatten()

        if not (
            target_position_rad.shape == current_position_rad.shape == current_velocity_rad_per_sec.shape
            == desired_velocity_rad_per_sec.shape == self.p_gain.shape
        ):
            raise ValueError("All input arrays and gain arrays must have the same shape (num_joints,).")

        position_error_rad = target_position_rad - current_position_rad
        velocity_error_rad_per_sec = desired_velocity_rad_per_sec - current_velocity_rad_per_sec

        torque_command = np.zeros_like(target_position_rad, dtype=float)

        # Apply PD only to selected joint indices (e.g., 0~5)
        torque_command[self.joint_indices] = (
            self.p_gain[self.joint_indices] * position_error_rad[self.joint_indices]
            + self.d_gain[self.joint_indices] * velocity_error_rad_per_sec[self.joint_indices]
        )

        return torque_command
