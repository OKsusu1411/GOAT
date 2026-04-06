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

        # NSC trajectory
        self.num_traj_points = 1000
        
        # NSC mode states
        self._nsc_initialized = False
        self._nsc_active = False

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
        self._nsc_active = False

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
        self._nsc_active = False
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

        # Detect session state
        is_new_session = not self._nsc_active
        self._nsc_active = True

        # Lazy initialization: Load model and parameters only once
        if not self._nsc_initialized:
            print("NSC controller initialized!!")
            self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
            self.model_names = list(self.model.names)
            self.data = self.model.createData()

            # Robot parameters
            self.ros_to_pin_ids = [0, 2, 4, 6, 1, 3, 5, 7]                              # ROS[self.ros_to_pin_ids] = Pin Ids
            self.pin_to_ros_ids = [0, 4, 1, 5, 2, 6, 3, 7]                              # Pin[self.pin_to_ros_ids] = ROS Ids
            self.wheel_radius = 72.75E-03
            self.nv = self.model.nv                                                         # Velocity dim (6 + n)
            self.nq = self.model.nq                                                         # Position dim (7 + n)
            self.n_joints = self.nv - 6                                                     # Num of Motors
            self.theta_cmd_limit = math.radians(5.0)
            
            # State variables
            self.q_ref = np.concatenate((np.zeros(7), robot_state.natural_joint_position))  # Reference position
            self.q_ref = self.q_ref[self.ros_to_pin_ids]
            self.q_ref_traj = np.zeros((self.num_traj_points, self.n_joints))               # Reference joint position trajectory
            self.a_ref = np.zeros(self.nv)                                                  # Reference joint acceleration
            self.phi_ref = 0.0
            self.theta_ref = 0.0
            self.cur_theta_ref = 0.0

            # Control Parameters
            self.dt = 1/200
            self.Kp = np.eye(self.n_joints) * 100.0
            self.Kd = np.eye(self.n_joints) * 1.0
            # self.Ko = np.eye(self.nv) * 1.0                                             # MOB gain (Base 6 + Joints n)
            self.wheel_Kp_att = 7.0
            self.wheel_Kd_att = 2.0
            self.wheel_Kp_pos = 0.1
            self.wheel_Kd_pos = 0.1
            self.alpha = 1.0
            self.cascade_ratio = 1

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


            self.wheel_L_joint_id = self.pin_name_to_idx['wheel_L_Joint']               # 3 : Left wheel index in pinocchio-actuator-order
            self.wheel_R_joint_id = self.pin_name_to_idx['wheel_R_Joint']               # 7 : Right wheel index in pinocchio-actuator-order

            self.wheel_L_joint_pin_id = self.model_names.index('wheel_L_Joint')         # 5 : Left wheel index in pinocchio-joint-order
            self.wheel_R_joint_pin_id = self.model_names.index('wheel_R_Joint')         # 9 : Right wheel index in pinocchio-joint-order
            
            # Additional nv IDs needed for S_wheel
            self.wheel_L_joint_nv_id = self.model.joints[self.model.getJointId('wheel_L_Joint')].idx_v
            self.wheel_R_joint_nv_id = self.model.joints[self.model.getJointId('wheel_R_Joint')].idx_v

            # State Selection Matrices
            self.S_leg = np.zeros((6, self.nv))
            self.S_leg[0, 6]  = 1.0   # hip_L
            self.S_leg[1, 7]  = 1.0   # thigh_L
            self.S_leg[2, 8]  = 1.0   # knee_L
            self.S_leg[3, 10] = 1.0   # hip_R
            self.S_leg[4, 11] = 1.0   # thigh_R
            self.S_leg[5, 12] = 1.0   # knee_R

            self.S_wheel = np.zeros((2, self.nv))
            self.S_wheel[0, self.wheel_L_joint_nv_id] = 1.0
            self.S_wheel[1, self.wheel_R_joint_nv_id] = 1.0

            # Default contact geometry (can be updated dynamically if needed)
            self.contact_normal_world = np.array([0, 0, 1])
            self.contact_lateral_world = np.array([0, 1, 0])

            self._nsc_initialized = True


        # Exception
        dt_sec = float(dt_sec)
        if dt_sec <= 0.0:
            raise ValueError("dt_sec must be > 0.")
        
        # Extract base state
        quat_w = robot_state.imu_state.orientation_quat_w
        quat_x = robot_state.imu_state.orientation_quat_x
        quat_y = robot_state.imu_state.orientation_quat_y
        quat_z = robot_state.imu_state.orientation_quat_z

        vel_x = robot_state.imu_state.acceleration_x
        vel_y = robot_state.imu_state.acceleration_y
        vel_z = robot_state.imu_state.acceleration_z

        gyro_x = robot_state.imu_state.gyroscope_x
        gyro_y = robot_state.imu_state.gyroscope_y
        gyro_z = robot_state.imu_state.gyroscope_z

        # Current State
        self.joint_q_curr = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()                   # Joint position
        self.joint_v_curr = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()           # Joint velocity
        self.joint_q_curr = self.joint_q_curr[self.ros_to_pin_ids]
        self.joint_v_curr = self.joint_v_curr[self.ros_to_pin_ids]
        self.base_quat_curr = np.array([quat_x, quat_y, quat_z, quat_w])                                        # Base quaternion
        self.base_v_curr = np.array([vel_x, vel_y, vel_z, gyro_x, gyro_y, gyro_z])                              # Base velocity state
        self.base_q_curr = np.concatenate((np.zeros(3), self.base_quat_curr))                                   # base current position
        self.q_curr = np.concatenate((self.base_q_curr, self.joint_q_curr))                                     # base + joint position
        self.v_curr = np.concatenate((self.base_v_curr, self.joint_v_curr))                                     # base + joint velocity

        # Session initialization: Reset counters/trajectories when calling starts
        if is_new_session:
            # Reference assign
            for i, (start, end) in enumerate(zip(self.joint_q_curr, self.q_ref[7:])):
                self.q_ref_traj[:, i] = np.linspace(start, end, self.num_traj_points)
            
            self.count_tick = 0
            self.phi_ref = 0.0
            self.theta_ref = 0.0
            self.cur_theta_ref = 0.0

        # Compute Dynamics
        pin.computeAllTerms(self.model, self.data, self.q_curr, self.v_curr)
        M = self.data.M
        C = self.data.C
        G = self.data.g

        ## ================================ Wheel control ================================ ##
        # State
        theta, theta_dot, L = self.compute_com_and_theta(self.q_curr, self.v_curr)

        # Control logic
        if self.count_tick % self.cascade_ratio == 0:
            self.theta_ref = self.wheel_position_control(theta, theta_dot, self.phi_ref, L)
        self.cur_theta_ref = (1 - self.alpha) * self.cur_theta_ref + self.alpha * self.theta_ref
        wheel_tau = self.wheel_attitude_control(theta, theta_dot, self.cur_theta_ref)

        ## ================================ Joint control ================================ ##
        q_ref = np.zeros(self.nq)
        q_ref[7:] = self.q_ref_traj[min(self.count_tick, self.num_traj_points-1), :]
        
        # Error Feedback acceleration
        q_err = q_ref[7:] - self.q_curr[7:]
        v_err = -self.v_curr[6:]
        a_ref = np.zeros(self.nv)
        a_ref[6:] = self.Kp @ q_err + self.Kd @ v_err
        a_ref[-2:] = np.zeros(2)

        # Contact point Jacobian and projection
        Jc = self.compute_contact_jacobian()
        Jc_dot_v = self.compute_contact_jacobian_dot_times_v()
        Qu = self.compute_constraint_nullspace(Jc)
        a_ref_constrained = self.project_to_contact_consistent_acceleration(a_ref, Jc, Jc_dot_v)
        
        # Torque from reduced dynamics
        tau_constrained = self.solve_leg_torque_reduced_dynamics(M, C @ self.v_curr + G, a_ref_constrained, np.array([wheel_tau, -wheel_tau]), Qu)
        
        ## ================================ Combine torque command ================================ ##
        tau_constrained_full = self.S_leg.T @ tau_constrained + self.S_wheel.T @ np.array([wheel_tau, -wheel_tau])
        self.tau_cmd = tau_constrained_full[6:]
        self.tau_cmd = self.tau_cmd[self.pin_to_ros_ids].tolist()

        # Safety limiter
        safe_torque_command = self.torque_safety_limiter.apply(self.tau_cmd)

        self.count_tick += 1
        safe_joint_targets = np.array([q_ref[7:], np.zeros(self.num_joints)])             # Zero velocity reference
        return safe_torque_command, safe_joint_targets
    
    ### =============================== Auxilary Functions =============================== ###
    def compute_contact_jacobian(self) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, self.q_curr)
        pin.updateFramePlacements(self.model, self.data)

        rf = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Use wheel center (Identity placement) — offset [0,0,-r] would rotate
        # with the wheel joint, drifting away from the true ground contact point.
        placement_L = pin.SE3.Identity()
        placement_R = pin.SE3.Identity()

        J6_L = pin.getFrameJacobian(
            self.model,
            self.data,
            self.wheel_L_joint_pin_id,
            placement_L,
            rf,
        )
        J6_R = pin.getFrameJacobian(
            self.model,
            self.data,
            self.wheel_R_joint_pin_id,
            placement_R,
            rf,
        )

        Jv_L = J6_L[:3, :]
        Jv_R = J6_R[:3, :]

        n = self.contact_normal_world.reshape(1, 3)
        t = self.contact_lateral_world.reshape(1, 3)

        Jc = np.vstack([
            n @ Jv_L,
            n @ Jv_R,
        ])
        # Jc = np.vstack([
        #     n @ Jv_L,
        #     t @ Jv_L,
        #     n @ Jv_R,
        #     t @ Jv_R,
        # ])
        return Jc


    def compute_contact_jacobian_dot_times_v(self) -> np.ndarray:
        """
        Compute Jdot_c(q,v) * v in R^4
        """
        rf = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        placement_L = pin.SE3.Identity()
        placement_R = pin.SE3.Identity()

        a_zero = np.zeros(self.model.nv)

        # second-order forward kinematics
        pin.forwardKinematics(self.model, self.data, self.q_curr, self.v_curr, a_zero)
        pin.updateFramePlacements(self.model, self.data)

        # classical acceleration of the contact point
        acc_L = pin.getFrameClassicalAcceleration(
            self.model,
            self.data,
            self.wheel_L_joint_pin_id,
            placement_L,
            rf
        )
        acc_R = pin.getFrameClassicalAcceleration(
            self.model,
            self.data,
            self.wheel_R_joint_pin_id,
            placement_R,
            rf
        )

        # linear part = Jdot(q,v) * v   when a = 0
        a_lin_L = acc_L.linear
        a_lin_R = acc_R.linear

        n = self.contact_normal_world
        t = self.contact_lateral_world

        Jcdot_v = np.array([
            n @ a_lin_L,   # left normal
            n @ a_lin_R,   # right normal
        ])
        # Jcdot_v = np.array([
        #     n @ a_lin_L,   # left normal
        #     t @ a_lin_L,   # left lateral
        #     n @ a_lin_R,   # right normal
        #     t @ a_lin_R,   # right lateral
        # ])

        return Jcdot_v


    def compute_constraint_nullspace(self, Jc: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """
        Compute nullspace basis Q_u such that J_c @ Q_u = 0
        Q_u shape: (nv, nv-rank(Jc))
        """
        U, S, Vt = np.linalg.svd(Jc, full_matrices=True)
        rank = np.sum(S > tol)
        Qu = Vt.T[:, rank:]   # nullspace basis
        return Qu


    def project_to_contact_consistent_acceleration(self,
                                                   qdd_nom: np.ndarray,
                                                   Jc: np.ndarray,
                                                   Jcdot_v: np.ndarray,
                                                   damping: float = 1e-6) -> np.ndarray:
        """
        Equality-constrained least squares projection:
            min ||qdd - qdd_nom||^2
            s.t. Jc qdd = -Jcdot_v
        """
        JJt = Jc @ Jc.T
        JJt_reg = JJt + damping * np.eye(JJt.shape[0])

        rhs = Jc @ qdd_nom + Jcdot_v
        correction = Jc.T @ np.linalg.solve(JJt_reg, rhs)
        qdd_d = qdd_nom - correction
        return qdd_d


    def solve_leg_torque_reduced_dynamics(self,
                                          M: np.ndarray,
                                          h: np.ndarray,
                                          qdd_d: np.ndarray,
                                          tau_w: np.ndarray,
                                          Qu: np.ndarray) -> np.ndarray:
        """
        Solve leg torque from reduced constrained dynamics:
            tau_j = (Qu^T Sj^T)^dagger Qu^T (M qdd_d + h - Sw^T tau_w)
        """
        rhs_full = M @ qdd_d + h - self.S_wheel.T @ tau_w
        A = Qu.T @ self.S_leg.T
        b = Qu.T @ rhs_full

        tau_j = np.linalg.pinv(A) @ b
        return tau_j

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
        
        return wheel_tau