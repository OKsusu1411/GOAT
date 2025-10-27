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

    # def rpy_to_R(self, roll, pitch, yaw):
    #     """
    #     Roll, Pitch, Yaw (in radians) → Rotation Matrix (3x3)
    #     Rotation order: ZYX (yaw → pitch → roll)
    #     """
    #     Rz = np.array([
    #         [np.cos(yaw), -np.sin(yaw), 0],
    #         [np.sin(yaw),  np.cos(yaw), 0],
    #         [0,            0,           1]
    #     ])

    #     Ry = np.array([
    #         [np.cos(pitch), 0, np.sin(pitch)],
    #         [0,             1, 0],
    #         [-np.sin(pitch), 0, np.cos(pitch)]
    #     ])

    #     Rx = np.array([
    #         [1, 0,            0],
    #         [0, np.cos(roll), -np.sin(roll)],
    #         [0, np.sin(roll),  np.cos(roll)]
    #     ])

    #     R = Rz @ Ry @ Rx
    #     return R
    
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
        L_current_p = L_T_matrix[:3, 3]             # Left foot  
        L_current_R = L_T_matrix[:3, :3]
        R_current_p = R_T_matrix[:3, 3]             # Right foot
        R_current_R = R_T_matrix[:3, :3]
        w = np.zeros((len(self.joint_names), 1))    # TODO: Get current joint velocities

        J_inv = pinv(J)                               # Pseudoinverse of the Jacobian 
        target_position 
        target_R 
        error_R = logm(np.transpose(current_R) @ target_R)         # Logarithm of the rotation error
        error_R = error_R.real

        error_w = np.array([error_R[2, 1], error_R[0, 2], error_R[1, 0]]).reshape(3, 1)  # Extract angular error

        error_p = target_position - current_position
        error = np.concatenate((error_w, error_p))

        twist = self.Kp * error + self.Ki * self.error_integral       # PI control
        q = J_inv @ twist

        # Publish joint 
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