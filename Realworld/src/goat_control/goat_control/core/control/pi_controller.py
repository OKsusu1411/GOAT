# goat_control/core/control/pi_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class WheelPIControllerConfig:
    """Wheel speed PI controller configuration (rad/s domain)."""
    p_gain: np.ndarray  # shape: (num_joints,) or (num_wheels,)
    i_gain: np.ndarray      # shape: (num_joints,) or (num_wheels,)
    wheel_indices: Sequence[int]   # typically [6, 7]

    # Anti-windup: integrator state clamp (same logic as your previous INT_LIMIT) :contentReference[oaicite:1]{index=1}
    integrator_state_limit: float = 0.0  # if 0 -> no clamp

    # Optional: output saturation (if you want PI 내부에서 anti-windup까지 같이 하려면 사용)
    output_limit: Optional[float] = None  # e.g., max torque/current


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

        p_gain = np.asarray(config.p_gain, dtype=float).flatten()
        i_gain = np.asarray(config.i_gain, dtype=float).flatten()

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
        self.output_limit = config.output_limit

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
            # Integrate error
            self.integrator_state[wheel_index] += speed_error_rad_per_sec[wheel_index] * dt_sec

            # Clamp integrator state (classic anti-windup like your INT_LIMIT) :contentReference[oaicite:2]{index=2}
            if self.integrator_state_limit > 0.0:
                self.integrator_state[wheel_index] = float(
                    np.clip(self.integrator_state[wheel_index], -self.integrator_state_limit, self.integrator_state_limit)
                )

            proportional_term = self.p_gain_full[wheel_index] * speed_error_rad_per_sec[wheel_index]
            integral_term = self.i_gain_full[wheel_index] * self.integrator_state[wheel_index]

            wheel_output = proportional_term + integral_term

            # Optional: output saturation
            if self.output_limit is not None:
                wheel_output = float(np.clip(wheel_output, -self.output_limit, self.output_limit))

            torque_command[wheel_index] = wheel_output

        return torque_command
