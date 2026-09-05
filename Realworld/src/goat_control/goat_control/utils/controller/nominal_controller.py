import numpy as np
import pinocchio as pin
import math

from typing import Any

from .base_controller import BaseController

from motor_interfaces.msg import ImuState
from sensor_msgs.msg import JointState

class NominalController(BaseController):
    """PD (legs) + PI (wheels) torque controller driven by policy actions.

    Action space (set via set_targets()):
        delta_pos:    Delta joint position [rad], shape (num_joints,).
                      Added to natural_joint_position to form the PD reference.
        wheel_speed:  Desired wheel speed [rad/s], shape (num_joints,).
                      PI controller tracks this on wheel_indices only.

    YAML keys consumed:
        joint_names, joint_indices, wheel_indices, natural_joint_position,
        policy_leg_proportional_gain, policy_leg_derivative_gain,
        policy_wheel_proportional_gain, policy_wheel_integral_gain,
        integrator_state_limit
    """
    def __init__(self, cfg:dict, logger: Any | None):
        # Logger
        self.logger = logger

        # Config
        self.cfg = cfg
        self.urdf_path = self.cfg.get("nsc_urdf_path", None)
        if self.urdf_path is None:
            raise ValueError("[Nominal Controller] URDF path is not provided.")
        
        # Pinocchio Model
        self.model = pin.buildModelFromUrdf(self.urdf_path, pin.JointModelFreeFlyer())
        self.model_names = list(self.model.names)
        self.data = self.model.createData()

        # ========== Pinocchio Name Index ========== #

        # model.names:
            # universe
            # root_joint
            # hip_L_Joint
            # thigh_L_Joint
            # knee_L_Joint
            # wheel_L_Joint
            # hip_R_Joint
            # thigh_R_Joint
            # knee_R_Joint
            # wheel_R_Joint
        
        # generalized positions: 
            # base_position_xyz(3),
            # base_quaternion_xyzw(4),
            # hip_L,
            # thigh_L,
            # knee_L,
            # wheel_L,
            # hip_R,
            # thigh_R,
            # knee_R,
            # wheel_R
        
        # generalized velocities:
            # base_twist(6),
            # hip_L_dot,
            # thigh_L_dot,
            # knee_L_dot,
            # wheel_L_dot,
            # hip_R_dot,
            # thigh_R_dot,
            # knee_R_dot,
            # wheel_R_dot

        # ========= ROS Topic Name Index ========== #
        # name:
            # hip_L_Joint
            # hip_R_Joint
            # thigh_L_Joint
            # thigh_R_Joint
            # knee_L_Joint
            # knee_R_Joint
            # wheel_L_Joint
            # wheel_R_Joint

        # ========== Name Index Mapping =========== #
        self.ros_joint_names = ['hip_L_Joint', 'hip_R_Joint',
                                'thigh_L_Joint', 'thigh_R_Joint',
                                'knee_L_Joint', 'knee_R_Joint', 
                                'wheel_L_Joint', 'wheel_R_Joint']

        self.pin_joint_names = ['thigh_L_Joint', 'knee_L_Joint', 'wheel_L_Joint',
                                'thigh_R_Joint', 'knee_R_Joint', 'wheel_R_Joint']
        self.ros_name_to_idx = {name: i for i, name in enumerate(self.ros_joint_names)}
        self.pin_name_to_idx = {name: i for i, name in enumerate(self.pin_joint_names)}

        self.ros_to_pin_ids = [2, 4, 6, 3, 5, 7] # ROS[self.ros_to_pin_ids] = Pin Ids
        # self.ros_to_pin_ids = [0, 2, 4, 6, 1, 3, 5, 7] # ROS[self.ros_to_pin_ids] = Pin Ids

        # actuator ordering
        self.wheel_L_joint_id = self.pin_name_to_idx['wheel_L_Joint']   
        self.wheel_R_joint_id = self.pin_name_to_idx['wheel_R_Joint']  

        # pinocchio model ordering
        self.wheel_L_joint_pin_id = self.model_names.index('wheel_L_Joint') 
        self.wheel_R_joint_pin_id = self.model_names.index('wheel_R_Joint') 

        # generalized velocity ordering inside nv=14
        self.wheel_L_joint_nv_id = 6 + self.wheel_L_joint_id                
        self.wheel_R_joint_nv_id = 6 + self.wheel_R_joint_id             

        # Robot parameters
        self.nq = self.model.nq                         # Position dim (7 + n)
        self.nv = self.model.nv                         # Velocity dim (6 + n)
        self.n_joints = self.nv - 6                     # Num of Motors

        # Config parameters
        self.wheel_radius = self.cfg.get("wheel_radius")
        self.num_traj_points = self.cfg.get("num_traj_points")
        self.leg_Kp = np.diag(np.asarray(self.cfg.get("nsc_leg_proportional_gain"), dtype=float)[self.ros_to_pin_ids])
        self.leg_Kd = np.diag(np.asarray(self.cfg.get("nsc_leg_derivative_gain"), dtype=float)[self.ros_to_pin_ids])
        self.wheel_lqr_K = np.asarray(self.cfg.get("nsc_wheel_lqr_gain",[-24.40576650, -3.92998946, -11.18582263, -9.42047823]), dtype=float).reshape(4)
        self.theta_eq = float(self.cfg.get("nsc_wheel_theta_equilibrium", 0.06923353258))
        self.wheel_s_ref = float(self.cfg.get("nsc_wheel_position_reference", 0.0))
        self.theta_anchor = 0.0
        self.wheel_lqr_state = np.zeros(4, dtype=float)
        self.wheel_tau = 0.0

        self.wheel_outer_Kd = self.cfg.get("nsc_wheel_outer_derivative_gain")  
        self.wheel_inner_Kp = self.cfg.get("nsc_wheel_inner_proportional_gain")  
        self.wheel_inner_Kd = self.cfg.get("nsc_wheel_inner_derivative_gain")
        self.theta_cmd_limit = math.radians(self.cfg.get("nsc_theta_cmd_limit"))  
        self.q_target = np.asarray([0.0, 0.0, 0.9756, -0.9756, 2.0944, -2.0944, 0.0, 0.0])[self.ros_to_pin_ids]  # Final target joint position

        # State variables
        self.S_leg = np.zeros((4, self.nv))
        self.S_leg[0, 6]  = 1.0   # thigh_L
        self.S_leg[1, 7]  = 1.0   # knee_L
        self.S_leg[2, 9]  = 1.0   # thigh_R
        self.S_leg[3, 10] = 1.0   # knee_R

        self.S_wheel = np.zeros((2, self.nv))
        self.S_wheel[0, self.wheel_L_joint_nv_id] = 1.0
        self.S_wheel[1, self.wheel_R_joint_nv_id] = 1.0

        self.q_curr = np.zeros(self.nq)                                     # Generalized position (Base 7 + Joints n)
        self.v_curr = np.zeros(self.nv)                                     # Generalized velocity (Base 6 + Joints n)
        
        self.joint_q_curr = np.zeros(self.n_joints)                         # Joint position
        self.joint_v_curr = np.zeros(self.n_joints)                         # Joint velocity
        
        self.base_q_curr = np.zeros(7)                                      # Base position state
        self.base_quat_curr = np.array([0.0, 0.0, 0.0, 1.0])                # Base quaternion (x, y, z, w)
        self.base_v_curr = np.zeros(6)                                      # Base velocity state
        self.base_lin_v_curr = np.zeros(3)                                  # Base linear velocity
        self.base_ang_v_curr = np.zeros(3)                                  # Base angular velocity
        self.base_a_curr = np.zeros(3)                                      # Base linear acceleration
        
        # Reference signals
        self.q_ref = np.zeros(self.nq)                                      # Reference joint position
        self.q_ref_traj = np.zeros((self.num_traj_points, self.n_joints))   # Reference joint position trajectory
        self.a_ref = np.zeros(self.nv)                                      # Reference joint acceleration
        self.phi_ref = 0.0
        self.theta_ref = 0.0

        # Contact jacobian variable
        self.contact_normal_world = np.array([0.0, 0.0, 1.0])   # ground normal

        # Count tick for trajectory tracking
        self.count_tick = 0
        self.wheel_anchor = np.zeros(2, dtype=np.float32)

        # Wheel torque limit for leg joint control
        self.wheel_tau_limit = self.cfg.get("max_torque_per_joint")[-1]

    def reset(self):
        """Reset only count tick for trajectory tracking control.
        [NOTE] For trajectory generation, this class needs lazy initialization.
        """
        self.count_tick = 0
    
    def compute(self, 
                joint_state: JointState, 
                imu_state: ImuState, 
                dt_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        # Data processing
        self.joint_q_curr = np.asarray(joint_state.position)[self.ros_to_pin_ids]
        self.joint_v_curr = np.asarray(joint_state.velocity)[self.ros_to_pin_ids]
        
        self.base_lin_v_curr = np.asarray([imu_state.vel.x, imu_state.vel.y, imu_state.vel.z])
        self.base_ang_v_curr = np.asarray([imu_state.gyro.x, imu_state.gyro.y, imu_state.gyro.z])
        self.base_quat_curr  = np.asarray([imu_state.quat.x, imu_state.quat.y, imu_state.quat.z, imu_state.quat.w])

        theta, theta_dot = self.compute_base_pitch_and_rate(self.base_quat_curr, self.base_ang_v_curr)
        # Lazy initialization
        if self.count_tick == 0:
            # Anchor assign
            self.wheel_anchor[0] = self.joint_q_curr[self.wheel_L_joint_id].copy()
            self.wheel_anchor[1] = self.joint_q_curr[self.wheel_R_joint_id].copy()
            self.theta_anchor = theta
            # Reference assign
            for i, (start, end) in enumerate(zip(self.joint_q_curr, self.q_target)):
                self.q_ref_traj[:, i] = np.linspace(start, end, self.num_traj_points)

        # Wheel anchoring
        self.joint_q_curr[self.wheel_L_joint_id] -= self.wheel_anchor[0]
        self.joint_q_curr[self.wheel_R_joint_id] -= self.wheel_anchor[1]

        # Control Logic
        self.base_q_curr = np.concatenate((np.zeros(3), self.base_quat_curr))        
        self.base_v_curr = np.concatenate((self.base_lin_v_curr, self.base_ang_v_curr))
        self.q_curr = np.concatenate((self.base_q_curr, self.joint_q_curr))
        self.v_curr = np.concatenate((self.base_v_curr, self.joint_v_curr)) 

        # Computed torque
        pos_cmd = np.zeros(len(self.ros_joint_names), dtype=np.float32)
        tau_cmd = np.zeros(len(self.ros_joint_names), dtype=np.float32)

        # Compute Dynamics matrix
        pin.computeAllTerms(self.model, self.data, self.q_curr, self.v_curr)
        M = self.data.M                                     # Mass matrix
        C = self.data.C                                     # Coriolis matrix
        G = self.data.g                                     # Gravity vector   

        ## ============== Wheel control ================ ##

        self.wheel_lqr_state = self.compute_wheel_lqr_state(theta, theta_dot)

        wheel_tau = self.wheel_lqr_control(self.wheel_lqr_state)
        self.wheel_tau = wheel_tau

        ## ============= Joint control ================ ##

        # Update reference
        self.q_ref[7:] = self.q_ref_traj[min(self.count_tick, self.num_traj_points-1), :]
        # self.q_ref[7:] = self.q_target

        # Error Feedback for desired generalized acceleration
        q_err = self.q_ref[7:] - self.q_curr[7:]
        v_err = -self.v_curr[6:].copy()
        self.a_ref[6:] = self.leg_Kp @ q_err + self.leg_Kd @ v_err
        self.a_ref[self.wheel_L_joint_nv_id] = 0.0
        self.a_ref[self.wheel_R_joint_nv_id] = 0.0

        # Contact point Jacobian, its time derivative, and nullspace basis
        Jc = self.compute_contact_jacobian()
        Jc_dot_v = self.compute_contact_jacobian_dot_times_v()
        Qu = self.compute_constraint_nullspace(Jc)

        # Constraints-consistent projection using nullspace method
        a_ref_constrained = self.project_to_contact_consistent_acceleration(self.a_ref, Jc, Jc_dot_v)

        # Torque from reduced dynamics
        tau_constrained = self.solve_leg_torque_reduced_dynamics(M, C @ self.v_curr + G, a_ref_constrained, np.array([wheel_tau, -wheel_tau]), Qu)
        tau_constrained_full = self.S_leg.T @ tau_constrained + self.S_wheel.T @ np.array([wheel_tau, -wheel_tau])

        # Index Mapping (Pin -> ROS)
        pos_cmd[self.ros_to_pin_ids] = self.q_ref[7:]
        tau_cmd[self.ros_to_pin_ids] = tau_constrained_full[6:]

        # Update tick
        self.count_tick += 1

        return tau_cmd, pos_cmd, None

    ### =============================== Auxilary Functions (Wheel) =============================== ###

    def compute_base_pitch_and_rate(self,
                                    quat_xyzw: np.ndarray,
                                    gyro_xyz: np.ndarray) -> tuple[float, float]:

        """Compute physical base pitch and pitch rate."""

        qx, qy, qz, qw = quat_xyzw

        # ZYX Euler pitch
        sin_theta = 2.0 * (qw * qy - qz * qx)
        sin_theta = np.clip(sin_theta, -1.0, +1.0)
        theta = math.asin(sin_theta)

        # Roll is needed for exact Euler pitch-rate conversion
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

        gyro_y = gyro_xyz[1]
        gyro_z = gyro_xyz[2]

        theta_dot = (gyro_y * math.cos(roll) - gyro_z * math.sin(roll))

        return float(theta), float(theta_dot)
    
    def compute_wheel_lqr_state(self, theta: float, theta_dot: float) -> np.ndarray:

        """
         WIP-LQR state.

        State:
            x = [
                delta,
                delta_dot,
                s - s_ref,
                s_dot
            ]

        where:
            delta = theta - theta_eq

        Wheel common relative angle:
            phi = (q_L - q_R) / 2

        No-slip axle position:
            s = r * [phi + (theta - theta_anchor)]

        """

        # Common wheel relative rotation
        phi = 0.5 * (self.joint_q_curr[self.wheel_L_joint_id] - self.joint_q_curr[self.wheel_R_joint_id])
        phi_dot = 0.5 * (self.joint_v_curr[self.wheel_L_joint_id] - self.joint_v_curr[self.wheel_R_joint_id])

        # Ground-relative wheel axle translation
        s = self.wheel_radius * (phi + theta - self.theta_anchor)
        s_dot = self.wheel_radius * (phi_dot + theta_dot)

        # Pitch state relative to equilibrium
        delta = theta - self.theta_eq
        delta_dot = theta_dot

        x_lqr = np.array([delta, delta_dot, s - self.wheel_s_ref, s_dot], dtype=float)

        return x_lqr
    
    def wheel_lqr_control(self, x_lqr: np.ndarray) -> float:

        """
        Discrete LQR state feedback

            u = -K x
        """

        wheel_tau = -float(self.wheel_lqr_K @ x_lqr)
        wheel_tau = np.clip(wheel_tau, -self.wheel_tau_limit, self.wheel_tau_limit)

        return float(wheel_tau)


    ### =============================== Auxilary Functions (Leg) =============================== ###

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

        Jc = np.vstack([
            n @ Jv_L,
            n @ Jv_R,
        ])

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

        Jcdot_v = np.array([
            n @ a_lin_L,   # left normal
            n @ a_lin_R,   # right normal
        ])

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
