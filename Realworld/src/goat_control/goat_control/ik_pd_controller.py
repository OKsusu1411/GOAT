import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np
#from __future__ import annotations
from scipy.linalg import logm


class IKPDcontroller(Node):
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
        self.kp = 300.0                                          # P gain
        self.kd = 20.0                                           # D gain
        
        # Link frames
        self.base_frame = 'base_link'
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
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        lambda_matrix = (damping_constant**2) * np.eye(matrix.shape[0])   # 
        inv_term = np.linalg.inv(matrix @ matrix_T + lambda_matrix)
        matrix_pinv= matrix_T @ (inv_term)

        return matrix_pinv
    
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
            target_pos = target_pose[4:, 0]
            
            error_pos = target_pos - current_pos                                     # Extract position error                           
            error_pose = np.concatenate((error_angle, error_pos))

            # Inverse kinematics
            J_inv = self.DLS_pinv(jacobian[4:, :])                                    # Pseudoinverse of the jacobian
            delta_joint_pos = J_inv @ error_pose
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

    # ==================== Controller ==================== #
    def controller_callback(self):
        # Transform matrices for each joint
        J = np.zeros((6, len(self.joint_names)))      # Initialize Jacobian matrix
        i = 0                                         # Joint index

        for joint_frame in self.joint_frames:
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    self.base_frame, 
                    joint_frame, 
                    rclpy.time.Time()
                )
                T = self.transformation_matrix(transform)

                # Calculate Jacobian
                R = T[:3, :3]                       # Rotation matrix
                p = T[:3, 3]                        # Position vector
                w = R @ self.rotation_axis[i]       # J_w
                v = -np.cross(w, p)                 # J_v
                J[:, i] = np.concatenate([w, v])

                if joint_frame == 'wheel_L_Link':
                    L_T_matrix = T                  # Left foot T matrix
                
                elif joint_frame == 'wheel_R_Link':
                    R_T_matrix = T                  # Right foot T matrix

                i += 1

            except Exception as e:
                self.get_logger().error(f'Error looking up transform for {joint_frame}: {e}')

        # Current state
        L_leg_indices = np.array([0, 2, 4])                 # Indices for left leg joints
        R_leg_indices = np.array([1, 3, 5])                 # Indices for right leg joints
        L_current_pos = L_T_matrix[:3, 3]                   # Left foot state 
        L_current_rot = L_T_matrix[:3, :3]
        R_current_pos = R_T_matrix[:3, 3]                   # Right foot state
        R_current_rot = R_T_matrix[:3, :3]
        L_J = J[:, L_leg_indices]                           # Jacobian for left leg
        R_J = J[:, R_leg_indices]                           # Jacobian for right leg  
        joint_vel = np.zeros((len(self.joint_names), 1))            # TODO: Get current joint velocities
        joint_pos = np.zeros((len(self.joint_names), 1))            # TODO: Get current joint positions

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
            v = np.array(self.multi_vel_deg_s, dtype=float).reshape(n_joints, 1)
            joint_vel = v

        # Target feet position
        target_pose    # TODO: setting target_pose

        # ======================= Left leg control ======================= #   
        # Target foot pose
        L_target_pose = target_pose[0,:].reshape(7, 1)

        # Current joint state
        L_joint_pos = joint_pos[L_leg_indices, 0].reshape(3, 1)
        L_joint_vel = joint_vel[L_leg_indices, 0].reshape(3, 1)

        # Inverse kinematics
        L_joint_command = self.inverse_kinematics(target_pose=L_target_pose,
                                                  current_pos=L_current_pos,
                                                  current_rot=L_current_rot,
                                                  jacobian=L_J,
                                                  joint_pos=L_joint_pos,
                                                  mode="translation")

        L_joint_pos_error = L_joint_command - L_joint_pos
        L_joint_vel_error = - L_joint_vel
        L_torque_command = self.kp * L_joint_pos_error + self.kd * L_joint_vel_error


        # ======================= Right leg control ======================= #
        # Target foot pose
        R_target_pose = target_pose[1,:].reshape(7, 1)

        # Current joint state
        R_joint_pos = joint_pos[R_leg_indices, 0].reshape(3, 1)
        R_joint_vel = joint_vel[R_leg_indices, 0].reshape(3, 1)

        # Inverse kinematics
        R_joint_command = self.inverse_kinematics(target_pose=R_target_pose,
                                                  current_pos=R_current_pos,
                                                  current_rot=R_current_rot,
                                                  jacobian=R_J,
                                                  joint_pos=R_joint_pos,
                                                  mode="translation")
        
        R_joint_pos_error = R_joint_command - R_joint_pos
        R_joint_vel_error = - R_joint_vel
        R_torque_command = self.kp * R_joint_pos_error + self.kd * R_joint_vel_error

        # Combine torque commands
        torque_command = np.zeros((len(self.joint_names), 1))
        torque_command[L_leg_indices] = L_torque_command
        torque_command[R_leg_indices] = R_torque_command

        # Publish torque command to MotorTorqueController (Float32MultiArray)
        torque_msg = Float32MultiArray()
        # Flatten to 1D list: one element per joint/motor
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