# goat_control/core/control/control_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import math

from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates
from ..estimation.state_manager import MotorStateCollector, StateManager
from ..estimation.state_types import MotorStatesData, RobotState
from ..estimation.calibration_manager import CalibrationManager
from .pd_controller import PDJointController
from .pi_controller import WheelPIController
from .safety_limiter import TorqueSafetyLimiter, JointSafetyLimiter


@dataclass
class ControlTargets:
    """Control references in full-length vectors (num_joints,).

    - desired_joint_position_rad:
        Used by PD controller on joint_indices (0~5).
    - desired_wheel_speed_rad_per_sec:
        Used by PI controller on wheel_indices (6~7).
    """
    desired_joint_delta_position_rad: np.ndarray
    desired_wheel_speed_rad_per_sec: np.ndarray


@dataclass
class ControlPipelineOutput:
    """Outputs of one control step."""
    motor_states_data: MotorStatesData
    robot_state: RobotState

    raw_torque_command: np.ndarray
    safe_torque_command: np.ndarray


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
        calibration_manager: CalibrationManager,
        pd_joint_controller: PDJointController,
        wheel_pi_controller: WheelPIController,
        torque_safety_limiter: TorqueSafetyLimiter,
        joint_safety_limiter: JointSafetyLimiter,
        num_joints: int,
        wheel_indices: Sequence[int],
    ):
        self.motor_state_collector = motor_state_collector
        self.state_manager = state_manager
        self.calibration_manager = calibration_manager
        self.pd_joint_controller = pd_joint_controller

        self.wheel_pi_controller = wheel_pi_controller

        self.torque_safety_limiter = torque_safety_limiter
        self.joint_safety_limiter = joint_safety_limiter

        self.num_joints = int(num_joints)
        self.wheel_indices = [int(index) for index in wheel_indices]

    # ---------------------------------------------------------------------
    # Factory helper: build pipeline from GoatModel + already-built objects
    # ---------------------------------------------------------------------
    @classmethod
    def build_from_goat_model(
        cls,
        goat_model,  # GoatModel (typed loosely to avoid circular import)
        motor_state_collector: MotorStateCollector,
        state_manager: StateManager,
        calibration_manager: CalibrationManager,
        pd_joint_controller: PDJointController,
        torque_safety_limiter: TorqueSafetyLimiter,
        joint_safety_limiter: JointSafetyLimiter,
        wheel_pi_controller: WheelPIController,
    ) -> "ControlPipeline":
        """Create ControlPipeline using GoatModel indices."""
        num_joints = int(goat_model.num_joints)
        wheel_indices = list(goat_model.wheel_indices)

        return cls(
            motor_state_collector=motor_state_collector,
            state_manager=state_manager,
            calibration_manager = calibration_manager,
            pd_joint_controller=pd_joint_controller,
            wheel_pi_controller=wheel_pi_controller,
            torque_safety_limiter=torque_safety_limiter,
            joint_safety_limiter=joint_safety_limiter,
            num_joints=num_joints,
            wheel_indices=wheel_indices,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal states (integrator + safety limiter memory)."""
        self.wheel_pi_controller.reset()
        self.torque_safety_limiter.reset()

    # def step(
    #     self,
    #     targets: ControlTargets,
    #     dt_sec: float,
    #     imu_state: Optional[ImuState] = None,
    # ) -> ControlPipelineOutput:
    #     """Run one control cycle using fresh motor polling."""
    #     motor_states_data = self.motor_state_collector.poll_all()
    #     robot_state = self.state_manager.build_robot_state(motor_states_data, imu_state=imu_state)

    #     safe_torque_command, safe_joint_targets, raw_torque_command = self.compute_control(
    #         robot_state=robot_state,
    #         targets=targets,
    #         dt_sec=dt_sec,
    #     )

    #     return ControlPipelineOutput(
    #         motor_states_data=motor_states_data,
    #         robot_state=robot_state,
    #         raw_torque_command=raw_torque_command,
    #         safe_torque_command=safe_torque_command,
    #     )

    def apply_calibrated_offset(self, joint_msg: JointState = None, imu_msg: BaseStates = None):
        """Apply calibrated offset to raw sensor data"""
        if joint_msg is not None:
            joint_msg = self.calibration_manager.apply_joint_offset(joint_msg)

        if imu_msg is not None:
            imu_msg = self.calibration_manager.apply_imu_offset(imu_msg)
        
        return joint_msg, imu_msg
    
    def compute_control(
        self,
        robot_state: RobotState,
        targets: ControlTargets,
        dt_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (safe_torque, raw_torque) without polling (useful for testing)."""
        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")

        desired_joint_delta_position_rad = np.asarray(targets.desired_joint_delta_position_rad, dtype=float).flatten()
        desired_wheel_speed_rad_per_sec = np.asarray(targets.desired_wheel_speed_rad_per_sec, dtype=float).flatten()

        if desired_joint_delta_position_rad.size != self.num_joints:
            raise ValueError("targets.desired_joint_delta_position_rad must have length == num_joints.")
        if desired_wheel_speed_rad_per_sec.size != self.num_joints:
            raise ValueError("targets.desired_wheel_speed_rad_per_sec must have length == num_joints.")

        natural_joint_position = np.asarray(robot_state.natural_joint_position, dtype=float).flatten()
        current_joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        current_joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()

        safe_joint_delta_position_rad, safe_wheel_speed_rad_per_sec, has_violation = self.joint_safety_limiter.apply(robot_state,
                                                                                                                     desired_joint_delta_position_rad,
                                                                                                                     desired_wheel_speed_rad_per_sec)
        
        # Delta position action space
        desired_joint_position_rad = natural_joint_position + safe_joint_delta_position_rad             # Reference = Default(Natural) + action
        desired_wheel_speed_rad_per_sec = safe_wheel_speed_rad_per_sec
        
        safe_joint_targets = np.array([desired_joint_position_rad, desired_wheel_speed_rad_per_sec])

        # 1) Joint PD (applies only to joint_indices configured inside PDJointController)
        pd_torque_command = self.pd_joint_controller.compute(
            target_position_rad=desired_joint_position_rad,
            current_position_rad=current_joint_position_rad,
            current_velocity_rad_per_sec=current_joint_velocity_rad_per_sec,
            desired_velocity_rad_per_sec=None,
        )

        # 2) Wheel PI (conditional integration anti-windup is implemented inside WheelPIController)
        pi_torque_command = self.wheel_pi_controller.compute(
            wheel_speed_reference_rad_per_sec=desired_wheel_speed_rad_per_sec,
            wheel_speed_measured_rad_per_sec=current_joint_velocity_rad_per_sec,
            dt_sec=dt_sec,
        )

        # 3) Sum raw command
        raw_torque_command = pd_torque_command + pi_torque_command

        # 4) Safety limiter (LPF + clipping)
        safe_torque_command = self.torque_safety_limiter.apply(raw_torque_command)

        return safe_torque_command, safe_joint_targets, has_violation

    def compute_natural_torque(
        self,
        robot_state: RobotState,
        dt_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Natural standing configuration controller"""

        # Exception
        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")
        
        # Extract quaternion
        quat_w = robot_state.imu_state.orientation_quat_w
        quat_x = robot_state.imu_state.orientation_quat_x
        quat_y = robot_state.imu_state.orientation_quat_y
        quat_z = robot_state.imu_state.orientation_quat_z
        
        gyroscope_y=float(robot_state.imu_state.gyroscope_y)

        ## ================================ Joint control ================================ ##
        # Reference input
        target_joint_pos = np.asarray(robot_state.natural_joint_position, dtype=float).flatten()
        target_joint_pos *= 0                                               # Reference joint position 
        target_joint_vel = np.zeros_like(target_joint_pos).flatten()

        # Current state
        current_joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        current_joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()

        # PD controller
        pd_torque_command = self.pd_joint_controller.compute(
            target_joint_pos,
            current_joint_position_rad,
            current_joint_velocity_rad_per_sec,
            target_joint_vel
        )
        
        ## ================================ Wheel control ================================ ##
        # Reference input
        target_wheel_position = 0
        target_wheel_velocity = 0
        target_pitch_velocity = 0

        # Pitch calculation
        pitch_sin = 2.0 * (quat_w * quat_y - quat_z * quat_x)

        if abs(pitch_sin) >= 1.0:
            # clipping
            pitch_rad = math.copysign(math.pi / 2.0, pitch_sin)
        else:
            # arcsin
            pitch_rad = math.asin(pitch_sin)

        # Current state
        current_pitch = pitch_rad
        current_wheel_position = np.asarray(robot_state.joint_position_rad[-2:], dtype=float).flatten()
        current_wheel_velocity = np.asarray(robot_state.joint_velocity_rad_per_sec[-2:], dtype=float).flatten()

        # Pitch controller
        wheel_position_error = target_wheel_position - current_wheel_position
        wheel_velocity_error = target_wheel_velocity - current_wheel_velocity

        # target_pitch = kp * wheel_position_error + kd * wheel_velocity_error
        target_pitch = np.zeros_like(wheel_position_error)                      # NOTE: test

        # Torque controller
        pitch_error = current_pitch - target_pitch
        pitch_velocity_error = gyroscope_y - target_pitch_velocity
        # pd_wheel_toque = kp * pitch_error + kd * pitch_velocity_error
        pd_wheel_toque = np.zeros_like(pitch_error)                             # NOTE: test

        # Safety limiter
        raw_torque_command = np.hstack((pd_torque_command[:-2], pd_wheel_toque))
        safe_torque_command = self.torque_safety_limiter.apply(raw_torque_command)

        safe_joint_targets = np.array([target_joint_pos, target_joint_vel])
        return safe_torque_command, safe_joint_targets