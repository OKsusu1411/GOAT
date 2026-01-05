# goat_control/core/control/control_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..estimation.state_manager import MotorStateCollector, StateManager
from ..estimation.state_types import ImuState, MotorStatesData, RobotState
from .pd_controller import PDJointController
from .safety_limiter import ConditionalIntegratorAntiWindup, TorqueSafetyLimiter


@dataclass
class ControlTargets:
    """Control references in full-length vectors (num_joints,).

    - desired_joint_position_rad:
        Used by PD controller on joint_indices (0~5).
    - desired_wheel_speed_rad_per_sec:
        Used by PI controller on wheel_indices (6~7).
    """
    desired_joint_position_rad: np.ndarray
    desired_wheel_speed_rad_per_sec: np.ndarray


@dataclass
class ControlPipelineOutput:
    """Outputs of one control step."""
    motor_states_data: MotorStatesData
    robot_state: RobotState

    raw_torque_command: np.ndarray
    safe_torque_command: np.ndarray


def _expand_gains_to_full_length(
    num_joints: int,
    controlled_indices: Sequence[int],
    gain_vector: np.ndarray,
) -> np.ndarray:
    """Expand gain vector to full length.

    - If gain_vector is length == num_joints: return as-is.
    - If gain_vector is length == len(controlled_indices): expand into full vector with zeros elsewhere.
    """
    gain_vector = np.asarray(gain_vector, dtype=float).flatten()

    if gain_vector.size == num_joints:
        return gain_vector

    if gain_vector.size == len(controlled_indices):
        full_gain_vector = np.zeros(num_joints, dtype=float)
        for local_index, global_index in enumerate(controlled_indices):
            full_gain_vector[int(global_index)] = float(gain_vector[local_index])
        return full_gain_vector

    raise ValueError(
        f"gain_vector length must be num_joints ({num_joints}) or len(controlled_indices) ({len(controlled_indices)})."
    )


class ControlPipeline:
    """Core-only control pipeline (no ROS2 dependency).

    Flow:
      1) MotorStateCollector.poll_all() -> MotorStatesData
      2) StateManager.build_robot_state(...) -> RobotState
      3) PD (joints) + conditional PI (wheels) -> raw torque
      4) TorqueSafetyLimiter (LPF + clipping) -> safe torque

    Notes:
      - PD controller acts only on joint_indices (e.g., 0~5).
      - PI controller (anti-windup) acts only on wheel_indices (e.g., 6~7).
      - The pipeline returns torque/effort command; sending to motors is handled outside (node/runner).
    """

    def __init__(
        self,
        motor_state_collector: MotorStateCollector,
        state_manager: StateManager,
        pd_joint_controller: PDJointController,
        wheel_antiwindup_controller: ConditionalIntegratorAntiWindup,
        wheel_proportional_gain_full: np.ndarray,
        wheel_integral_gain_full: np.ndarray,
        torque_safety_limiter: TorqueSafetyLimiter,
        num_joints: int,
        wheel_indices: Sequence[int],
    ):
        self.motor_state_collector = motor_state_collector
        self.state_manager = state_manager
        self.pd_joint_controller = pd_joint_controller

        self.wheel_antiwindup_controller = wheel_antiwindup_controller
        self.wheel_proportional_gain_full = np.asarray(wheel_proportional_gain_full, dtype=float).flatten()
        self.wheel_integral_gain_full = np.asarray(wheel_integral_gain_full, dtype=float).flatten()

        self.torque_safety_limiter = torque_safety_limiter

        self.num_joints = int(num_joints)
        self.wheel_indices = [int(index) for index in wheel_indices]

        if self.wheel_proportional_gain_full.size != self.num_joints:
            raise ValueError("wheel_proportional_gain_full must have length == num_joints.")
        if self.wheel_integral_gain_full.size != self.num_joints:
            raise ValueError("wheel_integral_gain_full must have length == num_joints.")

    # ---------------------------------------------------------------------
    # Factory helper: build pipeline from GoatModel + already-built objects
    # ---------------------------------------------------------------------
    @classmethod
    def build_from_goat_model(
        cls,
        goat_model,  # GoatModel (typed loosely to avoid circular import)
        motor_state_collector: MotorStateCollector,
        state_manager: StateManager,
        pd_joint_controller: PDJointController,
        wheel_antiwindup_controller: ConditionalIntegratorAntiWindup,
        torque_safety_limiter: TorqueSafetyLimiter,
        wheel_proportional_gain: np.ndarray,
        wheel_integral_gain: np.ndarray,
    ) -> "ControlPipeline":
        """Create ControlPipeline with gain expansion using GoatModel indices."""
        num_joints = int(goat_model.num_joints)
        wheel_indices = list(goat_model.wheel_indices)

        wheel_proportional_gain_full = _expand_gains_to_full_length(
            num_joints=num_joints,
            controlled_indices=wheel_indices,
            gain_vector=wheel_proportional_gain,
        )
        wheel_integral_gain_full = _expand_gains_to_full_length(
            num_joints=num_joints,
            controlled_indices=wheel_indices,
            gain_vector=wheel_integral_gain,
        )

        return cls(
            motor_state_collector=motor_state_collector,
            state_manager=state_manager,
            pd_joint_controller=pd_joint_controller,
            wheel_antiwindup_controller=wheel_antiwindup_controller,
            wheel_proportional_gain_full=wheel_proportional_gain_full,
            wheel_integral_gain_full=wheel_integral_gain_full,
            torque_safety_limiter=torque_safety_limiter,
            num_joints=num_joints,
            wheel_indices=wheel_indices,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal states (integrator + safety limiter memory)."""
        self.wheel_antiwindup_controller.reset()
        self.torque_safety_limiter.reset()

    def step(
        self,
        targets: ControlTargets,
        dt_sec: float,
        imu_state: Optional[ImuState] = None,
    ) -> ControlPipelineOutput:
        """Run one control cycle using fresh motor polling."""
        motor_states_data = self.motor_state_collector.poll_all()
        robot_state = self.state_manager.build_robot_state(motor_states_data, imu_state=imu_state)

        safe_torque_command, raw_torque_command = self.compute_control(
            robot_state=robot_state,
            targets=targets,
            dt_sec=dt_sec,
        )

        return ControlPipelineOutput(
            motor_states_data=motor_states_data,
            robot_state=robot_state,
            raw_torque_command=raw_torque_command,
            safe_torque_command=safe_torque_command,
        )

    def compute_control(
        self,
        robot_state: RobotState,
        targets: ControlTargets,
        dt_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (safe_torque, raw_torque) without polling (useful for testing)."""
        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")

        desired_joint_position_rad = np.asarray(targets.desired_joint_position_rad, dtype=float).flatten()
        desired_wheel_speed_rad_per_sec = np.asarray(targets.desired_wheel_speed_rad_per_sec, dtype=float).flatten()

        if desired_joint_position_rad.size != self.num_joints:
            raise ValueError("targets.desired_joint_position_rad must have length == num_joints.")
        if desired_wheel_speed_rad_per_sec.size != self.num_joints:
            raise ValueError("targets.desired_wheel_speed_rad_per_sec must have length == num_joints.")

        current_joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        current_joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()

        # 1) Joint PD (applies only to joint_indices configured inside PDJointController)
        pd_torque_command = self.pd_joint_controller.compute(
            target_position_rad=desired_joint_position_rad,
            current_position_rad=current_joint_position_rad,
            current_velocity_rad_per_sec=current_joint_velocity_rad_per_sec,
            desired_velocity_rad_per_sec=None,
        )

        # 2) Wheel PI with conditional integration (anti-windup)
        wheel_speed_error = desired_wheel_speed_rad_per_sec - current_joint_velocity_rad_per_sec

        pi_torque_command = self.wheel_antiwindup_controller.step(
            error=wheel_speed_error,
            proportional_gain_full=self.wheel_proportional_gain_full,
            integral_gain_full=self.wheel_integral_gain_full,
            dt_sec=dt_sec,
        )

        # 3) Sum raw command
        raw_torque_command = pd_torque_command + pi_torque_command

        # 4) Safety limiter (LPF + clipping)
        safe_torque_command = raw_torque_command  # Temporarily disabled safety limiter

        return safe_torque_command, raw_torque_command
