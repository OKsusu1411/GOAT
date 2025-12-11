import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

NUM_JOINTS = 8  # 0~5: 관절, 6~7: 휠 속도용

class StepTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('step_trajectory_publisher')

        # ===== 파라미터 =====
        # 몇 번 조인트를 움직일지 (0~5만 사용하는 것을 추천)
        self.joint_index = int(
            self.declare_parameter('joint_index', 1).value
        )

        # 시작 각도 [deg]
        self.start_angle_deg = float(
            self.declare_parameter('start_angle_deg', 0.0).value
        )

        # 한 번에 증가할 각도 [deg] (스텝 크기)
        self.step_angle_deg = float(
            self.declare_parameter('step_angle_deg', 20.0).value
        )

        # 몇 번 스텝을 줄지 (예: 5 → 0, 20, 40, 60, 80, 100)
        self.num_steps = int(
            self.declare_parameter('num_steps', 1000).value
        )

        # 스텝 간 간격 [s] (느리게/빠르게 조절)
        self.step_interval_sec = float(
            self.declare_parameter('step_interval_sec', 3.0).value
        )

        # target을 어느 주기로 publish 할지 [Hz]
        self.publish_frequency = float(
            self.declare_parameter('publish_frequency', 200.0).value
        )

        if not (0 <= self.joint_index <= 5):
            self.get_logger().warn(
                f"joint_index={self.joint_index} 이(가) 0~5 범위를 벗어났습니다. "
                "0번 조인트로 강제 설정합니다."
            )
            self.joint_index = 0

        # ===== 타겟 데이터 초기화 =====
        # data[0:6] : 관절 각도 [deg]
        # data[6:8] : 휠 목표 속도 [deg/s]
        self.target_data = [0.0] * NUM_JOINTS
        self.target_data[self.joint_index] = self.start_angle_deg

        self.current_step = 0

        # ===== Publisher =====
        self.pub = self.create_publisher(
            Float32MultiArray,
            'target_joint_angles',
            10
        )

        # 스텝을 업데이트하는 타이머 (느리게)
        self.step_timer = self.create_timer(
            self.step_interval_sec,
            self.step_callback
        )

        # 현재 target을 계속 내보내는 타이머 (빠르게, PD 주기랑 비슷하게)
        self.publish_timer = self.create_timer(
            1.0 / self.publish_frequency,
            self.publish_callback
        )

        self.get_logger().info(
            f"StepTrajectoryPublisher 시작: joint_index={self.joint_index}, "
            f"start={self.start_angle_deg} deg, step={self.step_angle_deg} deg, "
            f"num_steps={self.num_steps}, step_interval={self.step_interval_sec}s"
        )

    def step_callback(self):
        """주기적으로 호출되어 목표 각도를 20도씩 증가시키는 콜백"""
        if self.current_step >= self.num_steps:
            # 더 이상 스텝을 올리지 않고 마지막 각도에서 유지
            return

        self.current_step += 1
        new_angle = self.start_angle_deg + self.current_step * self.step_angle_deg
        self.target_data[self.joint_index] = new_angle

        self.get_logger().info(
            f"[STEP] joint {self.joint_index}: target = {new_angle:.1f} deg "
            f"(step {self.current_step}/{self.num_steps})"
        )

    def publish_callback(self):
        """현재 target_data를 계속 publish (다른 조인트는 0도 고정)"""
        msg = Float32MultiArray()
        msg.data = self.target_data[:]  # 리스트 복사
        self.pub.publish(msg)
        # 너무 많이 찍히면 주석 처리 가능
        # self.get_logger().info(f"Published target: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = StepTrajectoryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
