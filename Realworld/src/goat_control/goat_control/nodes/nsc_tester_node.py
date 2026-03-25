import rclpy
from rclpy.node import Node
from pinocchio.utils import *
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray 

import numpy as np
import pinocchio as pin


class FloatingBaseController():
    def __init__(self):

        # 1. Pinocchio 로봇 모델 로드 (FreeFlyer 조인트 필수 추가)
        urdf_path = '/home/oksusu/Repos/GOAT/Realworld/src/goat_description/urdf/WF_GOAT.urdf'
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        
        self.nv = self.model.nv # 속도 차원 (6 + n)
        self.nq = self.model.nq # 위치 차원 (7 + n)
        self.n_joints = self.nv - 6 # 실제 모터 개수

        # 2. 제어 주기 및 게인 설정
        self.dt = 0.01
        self.Kp = np.eye(self.n_joints) * 3.0
        self.Kd = np.eye(self.n_joints) * 1.0
        self.Ko = np.eye(self.nv) * 20.0 # MOB 게인 (Base 6 + Joints n)

        # State variables
        self.q_curr = np.zeros(self.nq)
        self.v_curr = np.zeros(self.nv)
        self.joint_q_curr = np.zeros(self.nq)
        self.joint_v_curr = np.zeros(self.nq)
        self.base_q_curr = np.zeros(7)
        self.base_v_curr = np.zeros(6)
        self.q_ref = np.zeros(self.nq)
        self.a_ref = np.zeros(self.nv)
        
        self.tau_cmd = np.zeros(self.n_joints)
        self.tau_applied = np.zeros(self.n_joints)

        # MOB(Momentum Observer)용 적분기 변수
        self.mob_integral = np.zeros(self.nv)
        self.tau_external = np.zeros(self.nv) # Residual

        # 접촉 프레임 ID (URDF상의 발바닥 프레임 이름)
        self.foot_frame_id = self.model.getFrameId("foot_link_name")

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
        self.base_q_curr = np.array(msg.angular_velocity)
        self.base_v_curr = np.array(msg.orientation)

    def control_loop(self):

        # Stack base + joint state
        self.q_curr = np.vstack(self.base_q_curr, self.joint_q_curr)
        self.v_curr = np.vstack(self.base_v_curr, self.joint_v_curr)

        # RNEA
        tau_rnea = pin.rnea(self.model, self.data, self.q_curr, self.v_curr, self.a_ref)
        tau_rnea_joint = tau_rnea[6:]                         # Extract joint torque

        # Compute Dynamics matrix
        pin.computeAllTerms(self.model, self.data, self.q_curr, self.v_curr)
        M = self.data.M                                     # Mass matrix
        C = self.data.C                                     # Coriolis matrix
        G = self.data.g                                     # Gravity vector

        # Generalized Momentum Observer ( tau_external = Ko * [Mv - int(tau + C.T*v - G + tau_external)dt] )
        integrand = self.tau_applied + (C.T @ self.v_curr) - G + self.tau_external
        self.mob_integral += integrand * self.dt
        
        p_curr = M @ self.v_curr
        self.tau_external = self.Ko @ (p_curr - self.mob_integral)              # External torque for each joints
        
        # --- [Step 3] 외력 투영 (Contact Jacobian Mapping) ---
        # Residual을 순수 지면 반력(Contact Force) 공간으로 맵핑한 뒤 다시 관절 토크로 변환합니다.
        # (원치 않을 경우 단순하게 r_joint = self.tau_external[6:] 로 사용해도 무방합니다)
        # pin.computeJointJacobians(self.model, self.data, self.q_curr)
        # pin.updateFramePlacements(self.model, self.data)
        
        # J_L_wheel = pin.getJointJacobian(self.model, self.data, 6, pin.LOCAL_WORLD_ALIGNED)
        # J_R_wheel = pin.getJointJacobian(self.model, self.data, 7, pin.LOCAL_WORLD_ALIGNED)

        # # Pseudo-inverse로 F_c 추정 후 관절 토크 공간으로 매핑
        # J_c_pinv = np.linalg.pinv(J_c.T)
        # F_c_hat = J_c_pinv @ self.tau_external
        
        # J_c_joint = J_c[:, 6:] # 조인트 부분 야코비안만 추출
        # tau_ext_hat = J_c_joint.T @ F_c_hat
        
        # Error Feedback torque 
        q_err = self.q_ref[7:] - self.q_curr[7:]
        v_err = -self.v_curr[6:]
        tau_pd = self.Kp @ q_err + self.Kd @ v_err

        # Total torque (RNEA + Feedback + External)
        self.tau_cmd = tau_rnea_joint + tau_pd - self.tau_external

        # Clipping
        tau_limit = 4.5 # Nm
        self.tau_cmd = np.clip(self.tau_cmd, -tau_limit, tau_limit)

        # Publish joint command
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = [
            'hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint', 'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint'
        ]
        joint_command.effort = 
        self.command_publisher.publish(joint_command)

def main(args=None):
    # rclpy.init(args=args)
    node = FloatingBaseController()

    node.control_loop()                 # NOTE: testing without ROS2
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()