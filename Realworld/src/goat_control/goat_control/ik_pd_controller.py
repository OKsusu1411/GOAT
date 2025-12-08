import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np

# Plot 
import matplotlib
matplotlib.use('Agg')  # 디스플레이 없는 환경에서도 PNG 저장 가능하게
import matplotlib.pyplot as plt
import os
from datetime import datetime


# URDF 순서 기준 조인트 이름 (index 0~7)
# 0:hip_L, 1:hip_R, 2:thigh_L, 3:thigh_R, 4:knee_L, 5:knee_R, 6:wheel_L, 7:wheel_R
JOINT_NAME_LIST = [
    "hip_L", "hip_R",
    "thigh_L", "thigh_R",
    "knee_L", "knee_R",
    "wheel_L", "wheel_R",
]

# --- MG Motor scale ---
ANGLE_LSB_TO_DEG = 0.001      # multi_turn_raw, single_turn_raw : 0.001 deg/LSB
SPEED_LSB_TO_DPS = 0.001      # speed_dps : 0.001 deg/s per LSB (모터 매뉴얼 기준)

# Controller frequency (Hz)
DEFAULT_CONTROL_FREQUENCY = 200.0  # 200 Hz (기본 제어 주파수)

# --- Robot size ---
NUM_JOINTS = 8         # 전체 모터 개수
MOTOR_INDEX = 1        # 테스트용으로 제어할 관절 index (0~7)

# 테스트용 기본 목표각 (deg) – 지금은 여기만 수정해서 인가
JOINT_DEGREE = 0       # degrees
KI_KP_ratio = 0.76
#KI_KP_ratio = 0.8
# 휠 목표 속도 (deg/s)
DEFAULT_WHEEL_KP = 1.8
DEFAULT_WHEEL_KI = 0.55
#DEFAULT_WHEEL_KI = 0.0
L_WHEEL_TARGET = 0.0  # 왼쪽 휠 목표 속도 (deg/s)
R_WHEEL_TARGET = 10.0  # 오른쪽 휠 목표 속도 (deg/s)
INT_TORQUE_LIMIT = 3.0  # 토크 중 적분항으로 허용할 최대 기여
# INT_LIMIT = INT_TORQUE_LIMIT / DEFAULT_WHEEL_KI

# --- Default gains (scalar) ---
DEFAULT_KP = 0.0061         # Proportional gain

DEFAULT_KD = 0.055          # Derivative gain

# LPF / Torque 기본값 (scalar)
DEFAULT_LPF_ALPHA = 1       # Low-pass filter alpha
DEFAULT_MAX_TORQUE = 4.5    # Maximum torque limit

# # --- Per-joint default lists ---> degree ---
# DEFAULT_KP_LIST           = [0.013, 0.015, 0.0061, 0.0061, 0.0161, 0.0161, 0.000061, 0.000061]
# DEFAULT_KD_LIST           = [0.055,  0.055,  0.055,  0.055,  0.055,  0.055,  0.055,  0.055]
# DEFAULT_LPF_ALPHA_LIST    = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
# DEFAULT_MAX_TORQUE_LIST   = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

# --- Per-joint default lists ---> rad ---
DEFAULT_KP_LIST           = [0.0, 0.70,  0.0,   0.516,  0.0,   2.2, 0.0,    0.0]
DEFAULT_KD_LIST           = [0.05,  0.004,  0.01,  0.0002,  0.1,  0.0001,    0.0,    0.0]
DEFAULT_LPF_ALPHA_LIST    = [0.951,  0.951,   0.951,   0.951,  0.951,   0.951,  0.951,  0.951]
DEFAULT_MAX_TORQUE_LIST   = [  0.0,    4.5,     0.0,     4.5,    0.0,    4.5,     0.0,    4.5]

# 기본 타겟 각도 [deg] 리스트: MOTOR_INDEX만 JOINuT_DEGREE, 나머지 0
# DEFAULT_TARGET_ANGLES_DEG = [-20.0, 30.0, 30.0, -20.0, 30.0, -30.0, 0.0, 0.0]
DEFAULT_TARGET_ANGLES_DEG = [0.0, 20.0, 0.0, -20.0, 0.0, -20.0, 0.0, 0.0]
# DEFAULT_TARGET_ANGLES_DEG = [0.0 for _ in range(NUM_JOINTS)]
#DEFAULT_TARGET_ANGLES_DEG[MOTOR_INDEX] = JOINT_DEGREE
     
# Topic names
MOTOR_STATES_TOPIC = 'motor_states'
TARGET_ANGLES_TOPIC = 'target_joint_angles'
TORQUE_COMMANDS_TOPIC = 'torque_commands'



class PDController(Node):
    """
    Multiple-joint PD controller.
    MotorStates에서 0.001 deg/LSB 값을 받아서 rad로 변환하여 내부 계산을 수행하고,
    플롯/로그 출력 시에는 degree/deg/s 단위로 변환해서 확인 가능하게 하는
    여러 관절 동시 제어용 PD 컨트롤러.

    - 모터별로 서로 다른 Kp, Kd, LPF alpha, Max torque를 사용할 수 있게
      kp_gains, kd_gains, lpf_alpha_list, max_torque_list 파라미터 지원.
    - target_angles_deg 파라미터 + target_joint_angles 토픽으로 목표 각도 설정.
    """
    def __init__(self):
        super().__init__('ik_pd_controller')

        self.rotation_axis: np.array = np.array([                # Screw axies for each joints
            [-1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 1, 0]
        ])

        # gain
        self.kp = KP_GAiN                                          # P gain
        self.kd = KD_GAIN                                           # D gain
        
        # Link frames
        self.base_frame = 'base_Link'
        self.joint_frames = [
            'hip_L_Link',
            'hip_R_Link',
            'thigh_L_Link',
            'thigh_R_Link',
            'calf_L_Link',
            'calf_R_Link',
            'wheel_L_Link',
            'wheel_R_Link'
        ]

        # Joint names
        self.joint_names = [
            'hip_L_Joint',
            'hip_R_Joint',
            'thigh_L_Joint',
            'thigh_R_Joint',
            'knee_L_Joint',
            'knee_R_Joint',
            'wheel_L_Joint',
            'wheel_R_Joint'
        ]

        # TF subscriber
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Joint(Motor) state subscriber
        self.multi_angles_deg = None          # latest multi-turn angles [deg] from MotorStatePublisher
        self.multi_vel_deg_s = None           # latest multi-turn angular velocities [deg/s]
        self.prev_multi_angles_deg = None     # previous angles for velocity estimation
        self.prev_angle_time = None           # timestamp of previous angle update
        self.motor_states_sub = self.create_subscription(
            MotorStates,
            'motor_states',                   # topic published by MotorStatePublisher
            self.motor_states_callback,
            10,
        )

        # Target position subscriber
        self.policy_action = np.zeros((2, 3))   # Target position from Policy node
        self.policy_subscriber = self.create_subscription(
            Float32MultiArray,
            'policy_action',                # topic published by Policy node
            self.policy_subscriber_callback,
            10
        )


        # Torque command publisher for motor_torque_controller (Float32MultiArray)
        self.torque_publisher = self.create_publisher(
            Float32MultiArray,
            'torque_commands',                # topic subscribed by MotorTorqueController
            10,
        )

        # Controller timer
        self.timer = self.create_timer(
            0.01,
            self.controller_callback
        )

    # Transformation matrix from tf
    def transformation_matrix(self, tf: TransformStamped) -> np.array:
        """
        tf message to Transformation matrix
        """
        
        t = tf.transform.translation        # translation vector
        r = tf.transform.rotation           # quaternion
        T = np.array([
            [1 - 2*(r.y**2 + r.z**2), 2*(r.x*r.y - r.z*r.w), 2*(r.x*r.z + r.y*r.w), t.x],
            [2*(r.x*r.y + r.z*r.w), 1 - 2*(r.x**2 + r.z**2), 2*(r.y*r.z - r.x*r.w), t.y],
            [2*(r.x*r.z - r.y*r.w), 2*(r.y*r.z + r.x*r.w), 1 - 2*(r.x**2 + r.y**2), t.z],
            [0,                    0,                     0,                          1]
        ])
        return T

    def quaternion_to_rot(self, q):
        """
        Quaternion (w, x, y, z) → Rotation Matrix (3x3)
        """

        w, x, y, z = q
        R = np.array([
            [1 - 2*(x**2 + y**2),  2*(w*x - y*z),        2*(w*y + x*z)],
            [2*(w*x + y*z),        1 - 2*(w**2 + y**2),  2*(x*y - w*z)],
            [2*(w*y - x*z),        2*(x*y + w*z),        1 - 2*(w**2 + x**2)]
        ])
        return R
    
    def DLS_pinv(self, matrix, damping_constant = 0.01):
        """
        Damped Least Squares Pseudoinverse
        """

        matrix_T = np.transpose(matrix, (1, 0))                           # Matrix transpose
        lambda_matrix = (damping_constant**2) * np.eye(matrix.shape[0])   
        inv_term = np.linalg.inv(matrix @ matrix_T + lambda_matrix)
        matrix_pinv = matrix_T @ (inv_term)
        return matrix_pinv
    
    def multiarray_to_numpy(msg: Float32MultiArray) -> np.ndarray:
        """
        Converts a ROS2 Float32MultiArray message back into a NumPy array
        with its original shape.

        Args:
            msg (Float32MultiArray): The received ROS2 message.

        Returns:
            np.ndarray: The reconstructed NumPy array (float32).
        """
        
        # Check if data is empty
        if not msg.data:
            return np.array([], dtype=np.float32)

        # Extract the shape from the layout dimensions
        if msg.layout.dim:
            shape = [d.size for d in msg.layout.dim]
        else:
            # Fallback if layout is empty: assume 1D array
            shape = (len(msg.data), )

        # Convert data to NumPy array and reshape
        try:
            restored_array = np.array(msg.data, dtype=np.float32).reshape(shape)
        except ValueError as e:
            # Handle mismatch between data length and shape
            print(f"Error reshaping array: Data length ({len(msg.data)}) "
                f"does not match layout shape ({shape}). Error: {e}")
            # Return 1D array as a fallback
            return np.array(msg.data, dtype=np.float32)

        return restored_array

    def inverse_kinematics(self,
                           target_pose: np.array,
                           current_pos: np.array,
                           current_rot: np.array,
                           jacobian: np.array,
                           joint_pos: np.array,
                           mode: str = "translation") -> np.array:
        """
        Inverse Kinematics Algorithm

        Args:
            target_pose (np.array): Target end effector pose (7x1) [qx, qy, qz, qw, x, y, z]
            current_pos (np.array): Current end effector position (3x1)
            current_rot (np.array): Current end effector rotation matrix (3x3)
            jacobian (np.array): Jacobian matrix (6xn)
            joint_pos (np.array): Current joint positions (nx1)
            mode (str): "position" or "translation" for IK calculation
        """

        if mode == "position":
            target_rot = self.quaternion_to_rot(target_pose[0:4, 0])
            target_pos = target_pose[4:, 0]

            error_rot = logm(np.transpose(current_rot) @ target_rot)         # Logarithm of the rotation error
            error_rot = error_rot.real
            
            error_angle = np.array([error_rot[2, 1], error_rot[0, 2], error_rot[1, 0]]).reshape(3, 1)  # Extract angular error
            error_pos = target_pos - current_pos                                     # Extract position error                           
            error_pose = np.concatenate((error_angle, error_pos))

            # Inverse kinematics
            J_inv = self.DLS_pinv(jacobian)                                    # Pseudoinverse of the jacobian
            delta_joint_pos = J_inv @ error_pose
            joint_command = joint_pos + delta_joint_pos

        else:
            target_pos = target_pose[4:, 0].reshape(3, 1)
            current_pos = current_pos.reshape(3, 1)
            
            error_pos = target_pos - current_pos                                     # Extract position error

            # Inverse kinematics
            J_inv = self.DLS_pinv(jacobian[3:, :])                                    # Pseudoinverse of the jacobian
            delta_joint_pos = J_inv @ error_pos
            joint_command = joint_pos + delta_joint_pos

        return joint_command
    
    def motor_states_callback(self, msg: MotorStates):
        """
        Callback for MotorStates; stores multi-turn angles and estimates velocities.
        """

        # multi_turn_raw is in 0.01 deg/LSB (signed)
        try:
            raw = list(msg.multi_turn_raw)
        except AttributeError:
            self.get_logger().warn("MotorStates message has no 'multi_turn_raw' field.")
            return

        if not raw:
            return

        # Convert to degrees
        angles_deg = [float(v) * 0.01 for v in raw]

        # Match number of joints
        n_joints = len(self.joint_names)
        if len(angles_deg) > n_joints:
            angles_deg = angles_deg[:n_joints]
        elif len(angles_deg) < n_joints:
            angles_deg.extend([0.0] * (n_joints - len(angles_deg)))

        now = self.get_clock().now()

        # Estimate angular velocity (deg/s) using finite difference
        if self.multi_angles_deg is not None and self.prev_angle_time is not None:
            dt = (now.nanoseconds - self.prev_angle_time.nanoseconds) / 1e9
            if dt > 1e-6:
                prev = self.multi_angles_deg
                self.multi_vel_deg_s = [
                    (a - b) / dt for a, b in zip(angles_deg, prev)
                ]
        else:
            self.multi_vel_deg_s = [0.0] * n_joints

        # Update stored angles and timestamp
        self.prev_multi_angles_deg = self.multi_angles_deg
        self.multi_angles_deg = angles_deg
        self.prev_angle_time = now

    def policy_subscriber_callback(self, msg: Float32MultiArray):
        """
        Callback for receiving target positions from the Policy node.
        """
        if len(msg.data) != 6:
            self.get_logger().warn(f"policy_action length {len(msg.data)} != 6, data={msg.data}")
            return

        arr = np.array(msg.data, dtype=np.float32).reshape(2, 3)
        self.policy_action = arr

    # ==================== Controller ==================== #
    def controller_callback(self):
        # Transform matrices for each joint
        # J = np.zeros((6, len(self.joint_names)))      # Initialize Jacobian matrix
        # i = 0                                         # Joint index
        # L_T_matrix = np.eye(4)                        # Left foot transformation matrix
        # R_T_matrix = np.eye(4)                        # Right foot transformation matrix
        # J = np.zeros((6, len(self.joint_names)))      # Initialize Jacobian matrix

        # for joint_frame in self.joint_frames:
        #     try:
        #         transform: TransformStamped = self.tf_buffer.lookup_transform(
        #             self.base_frame, 
        #             joint_frame, 
        #             rclpy.time.Time()
        #         )
        #         T = self.transformation_matrix(transform)

        #         # Calculate Jacobian
        #         R = T[:3, :3]                       # Rotation matrix
        #         p = T[:3, 3]                        # Position vector
        #         w = R @ self.rotation_axis[i]       # J_w
        #         v = -np.cross(w, p)                 # J_v
        #         J[:, i] = np.concatenate([w, v])

        #         if joint_frame == 'wheel_L_Link':
        #             L_T_matrix = T                  # Left foot T matrix
                
        #         elif joint_frame == 'wheel_R_Link':
        #             R_T_matrix = T                  # Right foot T matrix

        #         i += 1

        #     except Exception as e:
        #         self.get_logger().error(f'Error looking up transform for {joint_frame}: {e}')

        # Current state
        L_leg_indices = np.array([0, 2, 4])                 # Indices for left leg joints
        R_leg_indices = np.array([1, 3, 5])                 # Indices for right leg joints
        # L_current_pos = L_T_matrix[:3, 3]                   # Left foot state 
        # L_current_rot = L_T_matrix[:3, :3]
        # R_current_pos = R_T_matrix[:3, 3]                   # Right foot state
        # R_current_rot = R_T_matrix[:3, :3]
        # foot_current_pos = np.array([L_current_pos, R_current_pos])
        # L_J = J[:, L_leg_indices]                           # Jacobian for left leg
        # R_J = J[:, R_leg_indices]                           # Jacobian for right leg 

        # Joint state from multi-turn motor angles (if available)
        n_joints = len(self.joint_names)
        joint_pos = np.zeros((n_joints, 1))
        joint_vel = np.zeros((n_joints, 1))

        if self.multi_angles_deg is not None:
            # Angles from MotorStatePublisher are in degrees; convert/scale here if needed.
            angles = np.array(self.multi_angles_deg, dtype=float).reshape(n_joints, 1)
            # 필요하면 여기에서 np.deg2rad(angles)로 바꿔서 사용
            joint_pos = angles

        if self.multi_vel_deg_s is not None:
            velocity = np.array(self.multi_vel_deg_s, dtype=float).reshape(n_joints, 1)
            joint_vel = velocity

        # Target feet position
        # target_pos = foot_current_pos + self.policy_action
        # self.get_logger().info(f"Target position:\n{target_pos}")

        # ======================= Left leg control ======================= #   
        # Target foot pose
        #L_target_pose = np.zeros((7, 1))
        #L_target_pose[4:, 0] = target_pos[0, :]

        # Current joint state
        L_joint_pos = joint_pos[L_leg_indices, 0].reshape(3, 1)
        L_joint_vel = joint_vel[L_leg_indices, 0].reshape(3, 1)

        # Inverse kinematics
        # L_joint_command = self.inverse_kinematics(target_pose=L_target_pose,
        #                                           current_pos=L_current_pos,
        #                                           current_rot=L_current_rot,
        #                                           jacobian=L_J,
        #                                           joint_pos=L_joint_pos,
        #                                           mode="translation")
        L_joint_command = np.array(L_J_C).reshape(3, 1)
        L_joint_pos_error = L_joint_command - L_joint_pos
        L_joint_vel_error = - L_joint_vel
        L_torque_command = self.kp * L_joint_pos_error + self.kd * L_joint_vel_error


        # ======================= Right leg control ======================= #
        # Target foot pose
        #R_target_pose = np.zeros((7, 1))
        #R_target_pose[4:, 0] = target_pos[1, :]

        # Current joint state
        R_joint_pos = joint_pos[R_leg_indices, 0].reshape(3, 1)
        R_joint_vel = joint_vel[R_leg_indices, 0].reshape(3, 1)

        # Inverse kinematics
        # R_joint_command = self.inverse_kinematics(target_pose=R_target_pose,
        #                                           current_pos=R_current_pos,
        #                                           current_rot=R_current_rot,
        #                                           jacobian=R_J,
        #                                           joint_pos=R_joint_pos,
        #                                           mode="translation")
        R_joint_command = np.array(R_J_C).reshape(3, 1)
        R_joint_pos_error = R_joint_command - R_joint_pos
        R_joint_vel_error = - R_joint_vel
        R_torque_command = self.kp * R_joint_pos_error + self.kd * R_joint_vel_error

        # Combine torque commands
        torque_command = np.zeros((len(self.joint_names), 1))
        # Apply torque only to the third joint of the left leg (knee_L_Joint)
        # L_leg_indices are [0, 2, 4], so L_leg_indices[2] is the global index for knee_L_Joint (which is 4)
        # L_torque_command is a (3,1) array, where L_torque_command[2] corresponds to the knee_L_Joint torque
        torque_command[L_leg_indices[2]] = L_torque_command[2]

        # Publish torque command to MotorTorqueController (Float32MultiArray)
        torque_msg = Float32MultiArray()
        torque_msg.data = torque_command.flatten().astype(float).tolist()
        self.torque_publisher.publish(torque_msg)

def main(args=None):
    rclpy.init(args=args)
    ik_pd_controller = IKPDcontroller()
    rclpy.spin(ik_pd_controller)
    ik_pd_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()