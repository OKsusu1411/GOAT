# goat_control/core/control/control_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from pinocchio.utils import *

import numpy as np
import math
import pinocchio as pin

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
        urdf_path,
        robot_state: RobotState,
        dt_sec: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Natural standing configuration controller"""

        # Exception
        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")
        
        # Extract base state
        quat_w = robot_state.imu_state.orientation_quat_w
        quat_x = robot_state.imu_state.orientation_quat_x
        quat_y = robot_state.imu_state.orientation_quat_y
        quat_z = robot_state.imu_state.orientation_quat_z

        vel_x = robot_state.imu_state.acceleration_x       # Despite variable name is accel, it's linear velocity actually
        vel_y = robot_state.imu_state.acceleration_y
        vel_z = robot_state.imu_state.acceleration_z

        gyro_x = robot_state.imu_state.gyroscope_x
        gyro_y = robot_state.imu_state.gyroscope_y
        gyro_z = robot_state.imu_state.gyroscope_z

        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.model_names = list(self.model.names)
        self.data = self.model.createData()

        # Robot parameters
        self.wheel_radius = 72.75E-03
        self.nv = self.model.nv                                                     # Velocity dim (6 + n)
        self.nq = self.model.nq                                                     # Position dim (7 + n)
        self.n_joints = self.nv - 6                                                 # Num of Motors
        self.joint_tau_limit = 4.5                                                  # Nm
        self.wheel_tau_limit = 2.5                                                  # Nm
        self.theta_cmd_limit =math.radians(5.0)

        # Parameters
        self.dt = 1/200
        self.Kp = np.eye(self.n_joints) * 0.1
        self.Kd = np.eye(self.n_joints) * 0.05
        self.Ko = np.eye(self.nv) * 1.0                                             # MOB gain (Base 6 + Joints n)
        self.wheel_Kp_att = 100.0
        self.wheel_Kd_att = 20.0
        self.wheel_Kp_pos = 0.05
        self.wheel_Kd_pos = 0.02

        # ========== Name Index Mapping =========== #
        self.ros_joint_names = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint',
            'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]

        self.pin_joint_names = [
            'hip_L_Joint', 'thigh_L_Joint', 'knee_L_Joint', 'wheel_L_Joint',
            'hip_R_Joint', 'thigh_R_Joint', 'knee_R_Joint', 'wheel_R_Joint'
        ]
        self.ros_name_to_idx = {name: i for i, name in enumerate(self.ros_joint_names)}
        self.pin_name_to_idx = {name: i for i, name in enumerate(self.pin_joint_names)}

        self.ros_to_pin_ids = [0, 2, 4, 6, 1, 3, 5, 7]                              # ROS[self.ros_to_pin_ids] = Pin Ids
        self.pin_to_ros_ids = [0, 4, 1, 5, 2, 6, 3, 7]                              # Pin[self.pin_to_ros_ids] = ROS Ids

        self.wheel_L_joint_id = self.pin_name_to_idx['wheel_L_Joint']               # 3 : Left wheel index in pinocchio-actuator-order
        self.wheel_R_joint_id = self.pin_name_to_idx['wheel_R_Joint']               # 7 : Right wheel index in pinocchio-actuator-order

        self.wheel_L_joint_pin_id = self.model_names.index('wheel_L_Joint')         # 5 : Left wheel index in pinocchio-joint-order
        self.wheel_R_joint_pin_id = self.model_names.index('wheel_R_Joint')         # 9 : Right wheel index in pinocchio-joint-order

        # State variables
        self.q_curr = np.zeros(self.nq)                                                                         # base + joint position
        self.v_curr = np.zeros(self.nv)                                                                         # base + joint velocity
        self.joint_q_curr = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()                   # Joint position
        self.joint_v_curr = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()           # Joint velocity
        self.base_q_curr = np.zeros(7)                                                                          # Base position state
        self.base_quat_curr = np.array([quat_x, quat_y, quat_z, quat_w])                                        # Base quaternion
        self.base_v_curr = np.array([vel_x, vel_y, vel_z, gyro_x, gyro_y, gyro_z])                              # Base velocity state
        self.q_ref = np.zeros(self.nq)                                                                          # Reference base + joint position
        self.a_ref = np.zeros(self.nv)                                                                          # Reference base + joint acceleration

        ## ================================ Joint control ================================ ##

        # Stack base + joint state
        self.base_q_curr = np.concatenate((np.zeros(3), self.base_quat_curr))                  # XYZ position fixed to 0
        self.q_curr = np.concatenate((self.base_q_curr, self.joint_q_curr))
        self.v_curr = np.concatenate((self.base_v_curr, self.joint_v_curr))

        # RNEA
        tau_rnea = pin.rnea(self.model, self.data, self.q_curr, self.v_curr, self.a_ref)
        tau_rnea_joint = tau_rnea[6:]                       # Extract joint torque

        # Compute Dynamics matrix
        pin.computeAllTerms(self.model, self.data, self.q_curr, self.v_curr)
        M = self.data.M                                     # Mass matrix
        C = self.data.C                                     # Coriolis matrix
        G = self.data.g                                     # Gravity vector

        # Generalized Momentum Observer ( tau_external = Ko * [Mv - int(tau + C.T*v - G + tau_external)dt] )
        tau_full = np.concatenate((np.zeros(6), self.tau_applied))              # base 6DOF unactuated
        integrand = tau_full + (C.T @ self.v_curr) - G + self.tau_external
        self.mob_integral += integrand * self.dt
        
        p_curr = M @ self.v_curr
        self.tau_external = self.Ko @ (p_curr - self.mob_integral)              # External torque for each joints
        self.joint_tau_external = self.tau_external[6:]                         # Extract joint torque
        
        # Error Feedback torque 
        q_err = self.q_ref[7:] - self.q_curr[7:]
        v_err = -self.v_curr[6:]
        tau_pd = self.Kp @ q_err + self.Kd @ v_err

        # Total torque (RNEA + Feedback + External)
        self.tau_cmd = tau_rnea_joint + tau_pd - self.joint_tau_external
        
        ## ================================ Wheel control ================================ ##
        # State
        theta, theta_dot, L = self.compute_com_and_theta(self.q_curr, self.v_curr)

        # print(f"theta : {theta}, theta_dot : {theta_dot}, L : {L}")

        # Control logic
        target_phi = 0
        theta_cmd = self.wheel_position_control(theta, theta_dot,  target_phi, L)
        wheel_tau = self.wheel_attitude_control(theta, theta_dot, theta_cmd)

        self.tau_cmd[self.wheel_L_joint_id] = wheel_tau
        self.tau_cmd[self.wheel_R_joint_id] = -wheel_tau

        # Safety limiter
        safe_torque_command = self.torque_safety_limiter.apply(self.tau_cmd)

        safe_joint_targets = np.array([self.q_ref[7:], np.zeros(self.num_joints)])             # Zero velocity reference
        return safe_torque_command, safe_joint_targets
    
    ### =============================== Auxilary Functions =============================== ###
    def compute_com_and_theta(self, q: np.ndarray, v: np.ndarray):
        # COM calculation
        pin.centerOfMass(self.model, self.data, q, v, compute_subtree_coms=True)
        
        # Robot property
        M_total = self.data.mass[0]       # Total mass
        com_total = self.data.com[0]      # Total mass position
        vcom_total = self.data.vcom[0]    # Total mass velocity 

        m_wheel_L = self.data.mass[self.wheel_L_joint_pin_id]
        com_wheel_L  = self.data.oMi[self.wheel_L_joint_pin_id].act(
                           self.data.com[self.wheel_L_joint_pin_id])     # world frame
        vcom_wheel_L = self.data.oMi[self.wheel_L_joint_pin_id].rotation @ \
                           self.data.vcom[self.wheel_L_joint_pin_id]     # world frame

        m_wheel_R = self.data.mass[self.wheel_R_joint_pin_id]
        com_wheel_R  = self.data.oMi[self.wheel_R_joint_pin_id].act(
                           self.data.com[self.wheel_R_joint_pin_id])     # world frame
        vcom_wheel_R = self.data.oMi[self.wheel_R_joint_pin_id].rotation @ \
                           self.data.vcom[self.wheel_R_joint_pin_id]     # world frame
        # all_body = []
        # for i, data in enumerate(self.data.oMi):
        #     if i == 0: continue
        #     com = data.act(self.data.com[i])
        #     all_body.append(com)
        # self.com_body.append(np.array(all_body))

        # Body's property exclude wheels
        M_body = M_total - m_wheel_L - m_wheel_R
        com_body = (M_total * com_total - m_wheel_L * com_wheel_L - m_wheel_R * com_wheel_R) / M_body
        vcom_body = (M_total * vcom_total - m_wheel_L * vcom_wheel_L - m_wheel_R * vcom_wheel_R) / M_body

        # Center of wheels
        com_wheel = (com_wheel_L + com_wheel_R) / 2.0
        vcom_wheel = (vcom_wheel_L + vcom_wheel_R) / 2.0
  
        P_rel = com_body - com_wheel
        V_rel = vcom_body - vcom_wheel
        
        # Pitch calculation
        theta = math.atan2(P_rel[0], P_rel[2])
        # Angular velocity calculation
        theta_dot = (V_rel[0] * P_rel[2] - V_rel[2] * P_rel[0]) / (P_rel[0]**2 + P_rel[2]**2 + 1e-6)
        # Pendulum length
        L = math.hypot(P_rel[0], P_rel[2])

        return theta, theta_dot, L
    

    def wheel_position_control(self, theta, theta_dot, target_phi, L):
        # Wheel state
        phi = (self.joint_q_curr[self.wheel_L_joint_id] - self.joint_q_curr[self.wheel_R_joint_id]) / 2.0
        phi_dot = (self.joint_v_curr[self.wheel_L_joint_id] - self.joint_v_curr[self.wheel_R_joint_id]) / 2.0
        ratio = L / self.wheel_radius
        
        # Dynamic decoupling
        phi_comp = phi + ratio * math.sin(theta)
        phi_comp_dot = phi_dot + ratio * math.cos(theta) * theta_dot
        phi_err = target_phi - phi_comp
        
        # Pitch PD controller
        theta_cmd = self.wheel_Kp_pos * phi_err - self.wheel_Kd_pos * phi_comp_dot
        
        return np.clip(theta_cmd, -self.theta_cmd_limit, self.theta_cmd_limit)
    

    def wheel_attitude_control(self, theta, theta_dot, theta_cmd):
        # PD controller
        theta_err = theta - theta_cmd
        wheel_tau = self.wheel_Kp_att * theta_err + self.wheel_Kd_att * theta_dot
        
        return np.clip(wheel_tau, -self.wheel_tau_limit, self.wheel_tau_limit)