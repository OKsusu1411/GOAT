# goat_control/nodes/policy_node.py
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates


class PolicyNode(Node):
    """
    Legacy-compatible policy I/O:
      - Subscribes:
          * imu_data (motor_interfaces/BaseStates)
          * joint_states (sensor_msgs/JointState)   <-- policy input requirement
      - Publishes:
          * policy_action (std_msgs/Float32MultiArray) with layout populated
    """

    def __init__(self):
        super().__init__("policy")

        action_frequency_param = float(self.declare_parameter("action_frequency", 50).value)

        # Compatibility: interpret <=1.0 as period(sec), >1.0 as frequency(Hz)
        if action_frequency_param <= 1.0:
            self.action_period_sec = max(1e-4, action_frequency_param)
            self.action_frequency_hz = 1.0 / self.action_period_sec
        else:
            self.action_frequency_hz = action_frequency_param
            self.action_period_sec = 1.0 / max(1e-6, self.action_frequency_hz)

        self.get_logger().info(
            f"Policy tick: {self.action_frequency_hz:.1f} Hz ({self.action_period_sec:.4f} s)"
        )

        self.latest_imu: BaseStates | None = None
        self.latest_joint_state: JointState | None = None

        self.create_subscription(BaseStates, "imu_data", self._on_imu, 10)
        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)

        self.action_publisher = self.create_publisher(Float32MultiArray, "policy_action", 10)
        self.timer = self.create_timer(self.action_period_sec, self._tick)

    def _on_imu(self, msg: BaseStates) -> None:
        self.latest_imu = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _tick(self) -> None:
        if self.latest_joint_state is None:
            return

        # TODO: 실제 policy inference 붙이면 됨.
        # 너의 기존 policy.py처럼 (2,3) 텐서 구조를 유지하고 싶으면 그대로 유지 가능.
        # 여기서는 예시로 (2,3) 형태의 0 action을 보냄.
        action_array = np.zeros((2, 3), dtype=np.float32)

        action_msg = self._numpy_to_multiarray(action_array)
        self.action_publisher.publish(action_msg)

    @staticmethod
    def _numpy_to_multiarray(array_value: np.ndarray) -> Float32MultiArray:
        array_value = np.asarray(array_value, dtype=np.float32)

        msg = Float32MultiArray()
        msg.layout.data_offset = 0
        msg.layout.dim = []

        shape = array_value.shape
        current_stride = 1
        strides = []
        for size in reversed(shape):
            strides.insert(0, current_stride)
            current_stride *= int(size)

        for dim_index, dim_size in enumerate(shape):
            dim = MultiArrayDimension()
            dim.label = f"dim_{dim_index}"
            dim.size = int(dim_size)
            dim.stride = int(strides[dim_index])
            msg.layout.dim.append(dim)

        msg.data = array_value.flatten().tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
