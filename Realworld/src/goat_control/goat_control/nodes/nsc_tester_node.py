import rclpy
from rclpy.node import Node
from pinocchio.utils import *
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState, Imu

import numpy as np
import pinocchio as pin
import math


class NSCTesterNode(Node):
    def __init__(self):
        super().__init__('nsc_tester_node')

        # Pinocchio
        urdf_path = '/home/oksusu/Repos/GOAT/Realworld/src/goat_description/urdf/WF_GOAT.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.wheel_radius = 72.75E-03
        
        self.nv = self.model.nv                         # Velocity dim (6 + n)
        self.nq = self.model.nq                         # Position dim (7 + n)
        self.n_joints = self.nv - 6                     # Num of Motors
        self.wheel_L_joint_id = 6                       # Left wheel index
        self.wheel_R_joint_id = 7                       # Right wheel index
        self.joint_tau_limit = 4.5                      # Nm
        self.wheel_tau_limit = 2.5                      # Nm

        # Parameters
        self.dt = 0.01
        self.Kp = np.eye(self.n_joints) * 3.0
        self.Kd = np.eye(self.n_joints) * 1.0
        self.Ko = np.eye(self.nv) * 20.0                                    # MOB gain (Base 6 + Joints n)
        self.wheel_Kp_att = 0
        self.wheel_Kd_att = 0
        self.wheel_Kp_pos = 0
        self.wheel_Kd_pos = 0

        # State variables
        self.q_curr = np.zeros(self.nq)
        self.v_curr = np.zeros(self.nv)
        self.joint_q_curr = np.zeros(self.n_joints)                         # Joint position
        self.joint_v_curr = np.zeros(self.n_joints)                         # Joint velocity
        self.base_q_curr = np.zeros(7)                                      # Base position state
        self.base_quat_curr = np.array([0.0, 0.0, 0.0, 1.0])                # Base quaternion
        self.base_v_curr = np.zeros(6)                                      # Base velocity state
        self.base_linear_v_curr = np.zeros(3)                               # Base linear velocity
        self.base_w_curr = np.zeros(3)                                      # Base angular velocity
        self.base_a_curr = np.zeros(3)                                      # Base linear acceleration
        self.q_ref = np.zeros(self.nq)                                      # Reference joint position
        self.a_ref = np.zeros(self.nv)                                      # Reference joint acceleration
        
        self.tau_cmd = np.zeros(self.n_joints)
        self.tau_applied = np.zeros(self.n_joints)

        # MOB(Momentum Observer) parameters
        self.mob_integral = np.zeros(self.nv)
        self.tau_external = np.zeros(self.nv)                               # Residual

        # TF subscriber
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.command_publisher = self.create_publisher(JointState, '/joint_command', 10)

        # Subscriber
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_state', self.joint_callback, 10)
        self.imu_state_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

    def joint_callback(self, msg):
        self.joint_q_curr = np.array(msg.position)
        self.joint_v_curr = np.array(msg.velocity)
        self.torque_applied = np.array(msg.effort)

    def imu_callback(self, msg):
        self.base_linear_v_curr = np.zeros(3)               # NOTE: Linear velocity 받아와야됨
        self.base_a_curr = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.base_w_curr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.base_quat_curr = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def control_loop(self):
        # Stack base + joint state
        self.base_q_curr = np.concatenate((np.zeros(3), self.base_quat_curr))                  # XYZ position fixed to 0
        self.base_v_curr = np.concatenate((self.base_linear_v_curr, self.base_w_curr))
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

        # Clipping
        self.tau_cmd = np.clip(self.tau_cmd, -self.joint_tau_limit, self.joint_tau_limit)

        ## ============== Wheel control ============== ==
        # State
        theta, theta_dot, L = self.compute_com_and_theta(self.q_curr, self.v_curr)

        # Control logic
        target_phi = 0
        theta_cmd = self.wheel_position_control(theta, theta_dot,  target_phi, L)
        wheel_tau = self.wheel_attitude_control(theta, theta_dot, theta_cmd)

        self.tau_cmd[self.wheel_L_joint_id] = wheel_tau
        self.tau_cmd[self.wheel_R_joint_id] = -wheel_tau

        self.tau_cmd[self.wheel_L_joint_id:] = np.clip(self.tau_cmd[self.wheel_L_joint_id:], -self.wheel_tau_limit, self.wheel_tau_limit)

        # Publish joint command
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]
        joint_command.effort = self.tau_cmd.tolist()
        self.command_publisher.publish(joint_command)


### =============================== Auxilary Functions =============================== ###
    def compute_com_and_theta(self, q: np.ndarray, v: np.ndarray):
        # COM calculation
        pin.centerOfMass(self.model, self.data, q, v, compute_subtree_coms=True)
        
        # Robot property
        M_total = self.data.mass[0]       # Total mass
        com_total = self.data.com[0]      # Total mass position
        vcom_total = self.data.vcom[0]    # Total mass velocity 

        m_wheel_L = self.data.mass[self.wheel_L_joint_id]
        com_wheel_L = self.data.com[self.wheel_L_joint_id]
        vcom_wheel_L = self.data.vcom[self.wheel_L_joint_id]

        m_wheel_R = self.data.mass[self.wheel_R_joint_id]
        com_wheel_R = self.data.com[self.wheel_R_joint_id]
        vcom_wheel_R = self.data.vcom[self.wheel_R_joint_id]

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
        
        return theta_cmd
    

    def wheel_attitude_control(self, theta, theta_dot, theta_cmd):
        # PD controller
        theta_err = theta - theta_cmd
        wheel_tau = self.wheel_Kp_att * theta_err + self.wheel_Kd_att * theta_dot
        
        return wheel_tau

def main(args=None):
    rclpy.init(args=args)
    node = NSCTesterNode()

    node.control_loop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()