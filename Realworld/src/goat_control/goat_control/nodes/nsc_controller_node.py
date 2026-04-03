import rclpy
from rclpy.node import Node
from pinocchio.utils import *
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

import time
import numpy as np
import pinocchio as pin
import math
import csv

TARGET_JOINT_ANGLE = [0.0, 0.738, 1.462, 0.0, 0.0, -0.738, -1.462, 0.0] # Pinnochio convention : [hip_L, thigh_L, knee_L, wheel_L, hip_R, thigh_R, knee_R, wheel_R]

class NSCTesterNode(Node):
    def __init__(self):
        super().__init__('nsc_tester_node')

        # Pinocchio
        # urdf_path = '/home/oksusu/Repos/GOAT/Realworld/src/goat_description/urdf/WF_GOAT.urdf'
        urdf_path = '/home/grape4314/AISL-Isaac-ROS/src/goat_isaac_sim/assets/GOAT/WF_GOAT/urdf/WF_GOAT.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.model_names = list(self.model.names)
        self.data = self.model.createData()

        # ========== Pinocchio Name Index ========== #
        print("model.names:")
        for i, name in enumerate(self.model.names):
            print(i, name)

        print("\nidx_qs:", self.model.idx_qs)
        print("idx_vs:", self.model.idx_vs)

        print("\nJoint list:")
        for i, j in enumerate(self.model.joints):
            print(i, j)

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

        # State variables
        self.q_curr = np.zeros(self.nq)
        self.v_curr = np.zeros(self.nv)
        self.joint_q_curr = np.zeros(self.n_joints)                                 # Joint position
        self.joint_v_curr = np.zeros(self.n_joints)                                 # Joint velocity
        self.base_q_curr = np.zeros(7)                                              # Base position state
        self.base_quat_curr = np.array([0.0, 0.0, 0.0, 1.0])                        # Base quaternion
        self.base_v_curr = np.zeros(6)                                              # Base velocity state
        self.base_linear_v_curr = np.zeros(3)                                       # Base linear velocity
        self.base_w_curr = np.zeros(3)                                              # Base angular velocity
        self.base_a_curr = np.zeros(3)                                              # Base linear acceleration
        self.q_ref = np.zeros(self.nq)                                              # Reference base + joint position
        self.a_ref = np.zeros(self.nv)                                              # Reference base + joint acceleration

        # Reference assign
        self.q_ref[7:] = np.array(TARGET_JOINT_ANGLE)
        
        self.tau_cmd = np.zeros(self.n_joints)
        self.tau_applied = np.zeros(self.n_joints)

        # MOB(Momentum Observer) parameters
        self.mob_integral = np.zeros(self.nv)
        self.tau_external = np.zeros(self.nv)                                       # Residual

        # Publisher
        self.command_publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.debug_publisher = self.create_publisher(Float64MultiArray, '/nsc_debug', 10)

        # Subscriber
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.imu_state_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.velocity_state_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.com_body = []
        self.com_body_vel = []


    def joint_callback(self, msg):
        # ROS Ids -> Pinocchio Ids
        q = np.array(msg.position)
        v = np.array(msg.velocity)
        tau = np.array(msg.effort)
        
        self.joint_q_curr = q[self.ros_to_pin_ids]
        self.joint_v_curr = v[self.ros_to_pin_ids]
        self.tau_applied = tau[self.ros_to_pin_ids]


    def imu_callback(self, msg):
        self.base_a_curr = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.base_w_curr = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.base_quat_curr = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])


    def odom_callback(self, msg):
        self.base_linear_v_curr = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]) # In body frame


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

        # Coulomb Viscous friction
        coulomb_coeff = np.array([5.646268e-02, 5.646268e-02, 5.373143e-02])
        viscous_coeff = np.array([3.190248e-01, 3.190248e-01, 8.441387e-02])

        coulomb_friction = coulomb_coeff * np.sign(self.joint_v_curr[:self.wheel_L_joint_id])       # Ignore wheel friction
        viscous_friction = viscous_coeff * self.joint_v_curr[:self.wheel_L_joint_id]

        joint_friction = coulomb_friction + viscous_friction

        # Total torque (RNEA + Feedback + External)
        self.tau_cmd = tau_rnea_joint + tau_pd - self.joint_tau_external + joint_friction

        # Clipping
        self.tau_cmd = np.clip(self.tau_cmd, -self.joint_tau_limit, self.joint_tau_limit)

        # ## ============== Wheel control ================ ##
        # State
        theta, theta_dot, L = self.compute_com_and_theta(self.q_curr, self.v_curr)

        # print(f"theta : {theta}, theta_dot : {theta_dot}, L : {L}")

        # Control logic
        target_phi = 0
        theta_cmd = self.wheel_position_control(theta, theta_dot,  target_phi, L)
        wheel_tau = self.wheel_attitude_control(theta, theta_dot, theta_cmd)

        self.tau_cmd[self.wheel_L_joint_id] = wheel_tau
        self.tau_cmd[self.wheel_R_joint_id] = -wheel_tau

        self.tau_cmd[self.wheel_L_joint_id] = np.clip(self.tau_cmd[self.wheel_L_joint_id], -self.wheel_tau_limit, self.wheel_tau_limit)
        self.tau_cmd[self.wheel_R_joint_id] = np.clip(self.tau_cmd[self.wheel_R_joint_id], -self.wheel_tau_limit, self.wheel_tau_limit)

        # print(f"Torque Command : {self.tau_cmd}")

        # Publish joint command : Pin Ids -> ROS Ids
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 
            'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]
        joint_command.effort = self.tau_cmd[self.pin_to_ros_ids].tolist()
        self.command_publisher.publish(joint_command)

        # Publish debug signals for plotter
        # debug_msg = Float64MultiArray()
        # debug_msg.data = [
        #     float(theta), float(theta_dot), float(theta_cmd), float(L), float(wheel_tau),
        #     *self.tau_cmd.tolist(),
        #     *tau_rnea_joint.tolist(),
        #     *tau_pd.tolist(),
        #     *self.joint_tau_external.tolist(),
        # ]
        # self.debug_publisher.publish(debug_msg)


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

def main(args=None):
    rclpy.init(args=args)
    node = NSCTesterNode()
    
    # Automatic control loop using ROS timer callback
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()

    # Manual control loop to maintain fixed control frequency
    dt = node.dt
    next_t = time.monotonic()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)   
            node.control_loop()                      

            next_t += dt
            sleep_time = next_t - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        # print(f"Save")
        # data = [['name', 'value']]
        # for i, com in enumerate(node.com_body):
        #     data.append([f'body_{i}', f'{com}'])

        # with open('com_body_data.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data)
        node.destroy_node()
        rclpy.shutdown()
    
        

if __name__ == '__main__':
    main()