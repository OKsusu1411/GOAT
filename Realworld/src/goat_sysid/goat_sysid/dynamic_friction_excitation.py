#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time

NUM_JOINTS = 8
TARGET_TOPIC = 'target_joint_angles'  # pd_controller가 구독하는 토픽 이름

class DynamicFrictionExcitation(Node):
    """
    1개 조인트에 사인 궤적을 걸어주는 노드.
    q_ref(t) = offset + A * sin(2π f t)

    - 나머지 조인트는 0도 유지
    - PD 컨트롤러가 이 각도를 추종하면서 torque_commands를 발생
    - motor_states + torque_commands를 bag에 기록해서 나중에 friction 식별에 사용
    """

    def __init__(self):
        super().__init__('dynamic_friction_excitation')

        # ------- 파라미터 -------
        # 실험할 조인트 index
        self.joint_index = int(self.declare_parameter('joint_index', 0).value)

        # 사인 궤적 파라미터 (deg, Hz)
        self.amplitude_deg = float(self.declare_parameter('amplitude_deg', 20.0).value)
        self.frequency_hz = float(self.declare_parameter('frequency_hz', 0.5).value)
        self.offset_deg = float(self.declare_parameter('offset_deg', 0.0).value)

        # 컨트롤 주파수
        self.control_frequency = float(
            self.declare_parameter('control_frequency', 200.0).value
        )
        self.dt = 1.0 / self.control_frequency

        self.get_logger().info(
            f"[DynExcitation] joint_index={self.joint_index}, "
            f"A={self.amplitude_deg} deg, f={self.frequency_hz} Hz, "
            f"offset={self.offset_deg} deg"
        )

        self.pub = self.create_publisher(
            Float32MultiArray,
            TARGET_TOPIC,
            10
        )

        self.t0 = time.time()
        self.timer = self.create_timer(self.dt, self.timer_cb)

    def timer_cb(self):
        t = time.time() - self.t0

        # q_ref(t) = offset + A sin(2π f t)
        q_ref = self.offset_deg + self.amplitude_deg * np.sin(2 * np.pi * self.frequency_hz * t)

        targets = [0.0] * NUM_JOINTS
        targets[self.joint_index] = float(q_ref)

        msg = Float32MultiArray()
        msg.data = targets
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicFrictionExcitation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
