import rclpy
from rclpy.node import Node
from tf2_ros import Bufr, TransformListener
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

import numpy as np
from scipy.linalg import logm, pinv


class PDcontroller(Node):
    def __init__(self):
        super().__init__('pd_controller')

        self.trajectory: np.array = np.array([  # End-effector target position(s)
            [-0.8931, 0.10237, 0.08916]
        ])

        self.rotation_axis: np.array = np.array([  # Screw axes for each joint
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

        self.Kp = 5.0
        self.Ki = 3.0
        self.error_integral = np.zeros((6, 1))

        # TF subscriber
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # [MOD] 두 퍼블리셔를 분리 (타입 충돌 방지)
        self.joint_pub = self.create_publisher(JointState, '/joint_command', 10)            # 기존 JointState 명령(유지)
        self.current_pub = self.create_publisher(Float32MultiArray, '/current_command', 10)  # q 발행용

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

        # [MOD] dt 계산용
        self.prev_time = self.get_clock().now()

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

    def controller_callback(self):
        # Jacobian
        J = np.zeros((6, len(self.joint_names)))
        i = 0

        L_T_matrix = None  # [MOD] 가드
        R_T_matrix = None

        for joint_frame in self.joint_frames:
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    joint_frame,
                    rclpy.time.Time()
                )
                T = self.transformation_matrix(transform)

                # Jacobian column
                R = T[:3, :3]
                p = T[:3, 3]
                w_axis = R @ self.rotation_axis[i]     # angular part
                v_axis = -np.cross(w_axis, p)          # linear part
                J[:, i] = np.concatenate([w_axis, v_axis])

                if joint_frame == 'wheel_L_Link':
                    L_T_matrix = T
                elif joint_frame == 'wheel_R_Link':
                    R_T_matrix = T

                i += 1

            except Exception as e:
                self.get_logger().error(f'Error looking up transform for {joint_frame}: {e}')

        # [MOD] 엔드이펙터 현재 상태 (왼쪽 바퀴 링크 기준)
        if L_T_matrix is None:
            self.get_logger().warn('Left wheel transform not available yet; skip control step')
            return

        current_position = L_T_matrix[:3, 3].reshape(3, 1)
        current_R = L_T_matrix[:3, :3]

        # [MOD] 목표 설정: 주어진 목표 위치, 방향은 현재 유지(각도 오차 0으로)
        target_position = self.trajectory[min(self.count, len(self.trajectory) - 1)].reshape(3, 1)
        target_R = current_R  # orientation tracking은 비활성(필요 시 별도 목표 회전행렬 지정)

        # Orientation error (zero here since target_R == current_R)
        error_R = logm(current_R.T @ target_R).real
        error_w = np.array([error_R[2, 1], error_R[0, 2], error_R[1, 0]]).reshape(3, 1)

        # Position error
        error_p = target_position - current_position

        error = np.vstack((error_w, error_p))

        # [MOD] 적분항 업데이트
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now
        if dt <= 0 or dt > 0.2:  # 큰 시간 점프 가드
            dt = 0.01
        self.error_integral += error * dt

        # PI twist
        twist = self.Kp * error + self.Ki * self.error_integral  # 6x1

        # Joint-space command
        J_inv = pinv(J)
        q = J_inv @ twist  # (n_joints x 1)

        # [MOD] q를 /current_command(Float32MultiArray)로 발행
        cur_msg = Float32MultiArray()
        cur_msg.data = [float(v) for v in q.flatten()]
        self.current_pub.publish(cur_msg)

        # 기존 JointState 퍼블리시(유지). 여기서 velocity는 현재 0으로 둠.
        w = np.zeros((len(self.joint_names), 1))
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = self.joint_names
        joint_command.velocity = [float(val) for val in w.flatten()]
        self.joint_pub.publish(joint_command)


def main(args=None):
    rclpy.init(args=args)
    pd_controller = PDcontroller()
    rclpy.spin(pd_controller)
    pd_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
