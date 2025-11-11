import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import numpy as np
from scipy.linalg import logm, pinv


class PDcontroller(Node):
    def __init__(self):
        super().__init__('pd_controller')

        self.trajectory: np.array = np.array([                  # Trajectory points for the end effector
            [-0.8931, 0.10237, 0.08916]
        ])

        self.rotation_axis: np.array = np.array([               # Screw axies for each joints
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ])
        self.count = 0

        self.Kp = 5.0                                          # P gain
        self.Ki = 3.0                                          # I gain
        self.error_integral = np.zeros((6, 1))                 # Integral of error for PI control

        # TF subscriber
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Joint torque command publisher
        self.command_publisher = self.create_publisher(JointState, '/joint_command', 10)
        
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

        # Controller timer
        self.timer = self.create_timer(0.01, self.controller_callback)

    # Transformation matrix from tf
    def transformation_matrix(self, tf: TransformStamped) -> np.array:
        t = tf.transform.translation        # translation vector
        r = tf.transform.rotation           # quaternion
        T = np.array([
            [1 - 2*(r.y**2 + r.z**2), 2*(r.x*r.y - r.z*r.w), 2*(r.x*r.z + r.y*r.w), t.x],
            [2*(r.x*r.y + r.z*r.w), 1 - 2*(r.x**2 + r.z**2), 2*(r.y*r.z - r.x*r.w), t.y],
            [2*(r.x*r.z - r.y*r.w), 2*(r.y*r.z + r.x*r.w), 1 - 2*(r.x**2 + r.y**2), t.z],
            [0,                    0,                     0,                          1]
        ])
        return T

    def rpy_to_R(self, roll, pitch, yaw):
        """
        Roll, Pitch, Yaw (in radians) → Rotation Matrix (3x3)
        Rotation order: ZYX (yaw → pitch → roll)
        """
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rx = np.array([
            [1, 0,            0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        R = Rz @ Ry @ Rx
        return R
    
    def quaternion_to_rot(self, q):
        """
        Quaternion (x, y, z, w) → Rotation Matrix (3x3)
        """
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        return R
    
    # Controller
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
        L_leg_indices = np.array([0, 2, 4])                # Indices for left leg joints
        R_leg_indices = np.array([1, 3, 5])               # Indices for right leg joints
        L_current_pos = L_T_matrix[:3, 3]             # Left foot  
        L_current_rot = L_T_matrix[:3, :3]
        R_current_pos = R_T_matrix[:3, 3]             # Right foot
        R_current_rot = R_T_matrix[:3, :3]
        L_J = J[:, L_leg_indices]
        R_J = J[:, R_leg_indices]
        w = np.zeros((len(self.joint_names), 1))    # TODO: Get current joint velocities
        q = np.zeros((len(self.joint_names), 1))    # TODO: Get current joint positions

        # Target state
        target_pose

        # ======================= Left leg control ======================= #
        L_target_pose = target_pose[0,:].reshape(7, 1)
        L_target_rot = self.quaternion_to_rot(L_target_pose[0:4, 0])
        L_target_pos = L_target_pose[4:, 0]
        L_q = q[L_leg_indices, 0].reshape(3, 1)
        L_J_inv = pinv(L_J)                             # Pseudoinverse of the Jacobian

        L_error_rot = logm(np.transpose(L_current_rot) @ L_target_rot)         # Logarithm of the rotation error
        L_error_rot = L_error_rot.real

        L_error_angle = np.array([L_error_rot[2, 1], L_error_rot[0, 2], L_error_rot[1, 0]]).reshape(3, 1)  # Extract angular error
        L_error_pos = L_target_pos - L_current_pos                                     # Extract position error                           
        L_error_pose = np.concatenate((L_error_angle, L_error_pos))

        L_delta_q = L_J_inv @ L_error_pose
        L_q = L_q + L_delta_q


        # ======================= Right leg control ======================= #
        R_target_pose = target_pose[0,:].reshape(7, 1)
        R_target_rot = self.quaternion_to_rot(R_target_pose[0:4, 0])
        R_target_pos = R_target_pose[4:, 0]
        R_q = q[R_leg_indices, 0].reshape(3, 1)
        R_J_inv = pinv(R_J)                             # Pseudoinverse of the Jacobian

        R_error_rot = logm(np.transpose(R_current_rot) @ R_target_rot)         # Logarithm of the rotation error
        R_error_rot = R_error_rot.real

        R_error_angle = np.array([R_error_rot[2, 1], R_error_rot[0, 2], R_error_rot[1, 0]]).reshape(3, 1)  # Extract angular error
        R_error_pos = R_target_pos - R_current_pos                                     # Extract position error                           
        R_error_pose = np.concatenate((R_error_angle, R_error_pos))

        R_delta_q = R_J_inv @ R_error_pose
        R_q = R_q + R_delta_q

        # Publish joint command
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = self.joint_names
        joint_command.velocity = [float(val) for val in w.flatten()]
        self.command_publisher.publish(joint_command)

def main(args=None):
    rclpy.init(args=args)
    pd_controller = PDcontroller()
    rclpy.spin(pd_controller)
    pd_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()