# goat_control/core/model/goat_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np

from ..control.pd_controller import PDControllerConfig
from ..control.pi_controller import WheelPIControllerConfig
from ..control.safety_limiter import SafetyLimiterConfig
from ..estimation.state_manager import StateManagerConfig


EffortOutputMode = Literal["current_amp", "torque_nm"]


@dataclass
class GoatModelConfig:
    """Single source of truth for GOAT indexing + parameters.

    Indices convention:
      - Joint motors: 0~5 (position PD)
      - Wheel motors: 6~7 (speed PI)
    """

    # -----------------------------
    # Basic indexing / naming
    # -----------------------------
    joint_names: List[str]                  # length = num_joints
    joint_indices: Sequence[int]            # typically [0,1,2,3,4,5]
    wheel_indices: Sequence[int]            # typically [6,7]
    knee_indices: Sequence[int]             # subset of joint_indices (if any)

    # -----------------------------
    # Motor -> joint mapping params
    # -----------------------------
    motor_torque_constant_nm_per_amp: List[float]   # length = num_joints
    motor_gear_ratio: List[float]                   # length = num_joints
    motor_direction: List[int]                      # length = num_joints (+1 or -1)

    # -----------------------------
    # Estimation scaling parameters
    # -----------------------------
    motor_current_amp_per_lsb: float = 66.0 / 4096.0
    angle_deg_per_lsb: float = 0.001
    speed_deg_per_sec_per_lsb: float = 0.01

    # -----------------------------
    # PD gains (full-length vectors)
    # -----------------------------
    pd_proportional_gain: List[float] = None       # length = num_joints
    pd_derivative_gain: List[float] = None         # length = num_joints

    # -----------------------------
    # Wheel PI gains
    # Can be full-length vectors or wheel-only vectors (WheelPIController supports both).
    # -----------------------------
    wheel_pi_proportional_gain: List[float] = None
    wheel_pi_integral_gain: List[float] = None

    # Anti-windup / saturation for wheel PI
    wheel_integrator_state_limit: float = 0.0
    wheel_output_limit_per_joint: Optional[List[float]] = None  # length = num_joints recommended

    # -----------------------------
    # Safety limiter (LPF + torque clip)
    # -----------------------------
    torque_lpf_alpha_per_joint: Optional[List[float]] = None     # length = num_joints
    max_torque_per_joint: Optional[List[float]] = None           # length = num_joints
    joint_pos_limit: Optional[List[float]] = None
    joint_pos_limit_margin: float = 0.0
    joint_vel_limit: Optional[List[float]] = None
    joint_vel_limit_margin: float = 0.0

    # -----------------------------
    # Optional filtering in state_manager
    # -----------------------------
    joint_velocity_lpf_alpha: Optional[float] = None
    joint_effort_like_lpf_alpha: Optional[float] = None


class GoatModel:
    """GOAT model: builds configs for estimation/control from a single config block."""

    def __init__(self, config: GoatModelConfig):
        self.config = config
        self.joint_names = list(config.joint_names)

        self.joint_indices = list(config.joint_indices)
        self.wheel_indices = list(config.wheel_indices)
        self.knee_indices = list(config.knee_indices)

        self.num_joints = len(self.joint_names)
        self._validate_lengths()

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def _validate_lengths(self) -> None:
        if self.num_joints <= 0:
            raise ValueError("joint_names must be non-empty.")

        def require_length(name: str, values: List, expected: int) -> None:
            if values is None or len(values) != expected:
                raise ValueError(f"{name} must have length == {expected} (got {None if values is None else len(values)}).")

        require_length("motor_torque_constant_nm_per_amp", self.config.motor_torque_constant_nm_per_amp, self.num_joints)
        require_length("motor_gear_ratio", self.config.motor_gear_ratio, self.num_joints)
        require_length("motor_direction", self.config.motor_direction, self.num_joints)

        for direction_value in self.config.motor_direction:
            if int(direction_value) not in (+1, -1):
                raise ValueError("motor_direction values must be +1 or -1.")

        if self.config.pd_proportional_gain is not None:
            require_length("pd_proportional_gain", self.config.pd_proportional_gain, self.num_joints)
        if self.config.pd_derivative_gain is not None:
            require_length("pd_derivative_gain", self.config.pd_derivative_gain, self.num_joints)

        if self.config.torque_lpf_alpha_per_joint is not None:
            require_length("torque_lpf_alpha_per_joint", self.config.torque_lpf_alpha_per_joint, self.num_joints)

        if self.config.max_torque_per_joint is not None:
            require_length("max_torque_per_joint", self.config.max_torque_per_joint, self.num_joints)

        if self.config.wheel_output_limit_per_joint is not None:
            require_length("wheel_output_limit_per_joint", self.config.wheel_output_limit_per_joint, self.num_joints)

    # ---------------------------------------------------------------------
    # Builders
    # ---------------------------------------------------------------------
    def build_state_manager_config(self, effort_output_mode: EffortOutputMode = "torque_nm") -> StateManagerConfig:
        return StateManagerConfig(
            joint_names=self.joint_names,
            knee_indices=self.knee_indices,
            motor_current_amp_per_lsb=self.config.motor_current_amp_per_lsb,
            angle_deg_per_lsb=self.config.angle_deg_per_lsb,
            speed_deg_per_sec_per_lsb=self.config.speed_deg_per_sec_per_lsb,
            joint_velocity_lpf_alpha=self.config.joint_velocity_lpf_alpha,
            joint_effort_like_lpf_alpha=self.config.joint_effort_like_lpf_alpha,
            effort_output_mode=effort_output_mode,
            motor_torque_constant_nm_per_amp=list(self.config.motor_torque_constant_nm_per_amp),
            motor_gear_ratio=list(self.config.motor_gear_ratio),
            motor_direction=list(self.config.motor_direction),
        )

    def build_pd_controller_config(self) -> PDControllerConfig:
        if self.config.pd_proportional_gain is None or self.config.pd_derivative_gain is None:
            raise ValueError("pd_proportional_gain and pd_derivative_gain must be provided.")

        return PDControllerConfig(
            proportional_gain=np.asarray(self.config.pd_proportional_gain, dtype=float),
            derivative_gain=np.asarray(self.config.pd_derivative_gain, dtype=float),
            joint_indices=self.joint_indices,
        )

    def build_wheel_pi_controller_config(self) -> WheelPIControllerConfig:
        if self.config.wheel_pi_proportional_gain is None or self.config.wheel_pi_integral_gain is None:
            raise ValueError("wheel_pi_proportional_gain and wheel_pi_integral_gain must be provided.")

        return WheelPIControllerConfig(
            proportional_gain=np.asarray(self.config.wheel_pi_proportional_gain, dtype=float),
            integral_gain=np.asarray(self.config.wheel_pi_integral_gain, dtype=float),
            wheel_indices=self.wheel_indices,
            integrator_state_limit=float(self.config.wheel_integrator_state_limit),
            output_limit_per_joint=self.config.wheel_output_limit_per_joint,
        )

    def build_safety_limiter_config(self) -> SafetyLimiterConfig:
        return SafetyLimiterConfig(
            num_joints=self.num_joints,
            lpf_alpha_per_joint=self.config.torque_lpf_alpha_per_joint,
            max_torque_per_joint=self.config.max_torque_per_joint,
            joint_pos_limit=self.config.joint_pos_limit,
            joint_pos_limit_margin=self.config.joint_pos_limit_margin,
            joint_vel_limit=self.config.joint_vel_limit,
            joint_vel_limit_margin=self.config.joint_vel_limit_margin
        )



    def convert_joint_torque_to_motor_current(self, joint_torque_nm: np.ndarray) -> np.ndarray:
        torque_constant = np.asarray(self.config.motor_torque_constant_nm_per_amp, dtype=float)
        gear_ratio = np.asarray(self.config.motor_gear_ratio, dtype=float)
        direction = np.asarray(self.config.motor_direction, dtype=float)

        denominator = gear_ratio * torque_constant  # direction *
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        current_command_amp = joint_torque_nm / denominator

        zero_mask = np.abs(direction * gear_ratio * torque_constant) < 1e-12
        current_command_amp = np.where(zero_mask, 0.0, current_command_amp)
        
        return current_command_amp