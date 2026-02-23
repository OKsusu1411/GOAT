#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

# 너 메시지 패키지에 맞게 import 경로 수정!
from motor_interfaces.msg import BaseStates


class ImuTfBroadcaster(Node):
    """
    Subscribe: /imu_data (motor_interfaces/msg/BaseStates)
    Publish TF: odom -> base_link (rotation = imu quat)
    """
    def __init__(self):
        super().__init__("imu_tf_broadcaster")

        self.declare_parameter("imu_topic", "/imu_data")
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("use_translation_zero", True)

        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        self.parent_frame = self.get_parameter("parent_frame").get_parameter_value().string_value
        self.child_frame = self.get_parameter("child_frame").get_parameter_value().string_value
        self.use_translation_zero = self.get_parameter("use_translation_zero").get_parameter_value().bool_value

        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(BaseStates, imu_topic, self.cb, 10)

        self.get_logger().info(
            f"Listening {imu_topic}, publishing TF {self.parent_frame} -> {self.child_frame}"
        )

    def cb(self, msg: BaseStates):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        if self.use_translation_zero:
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0

        # BaseStates.quat가 geometry_msgs/Quaternion 타입이면 그대로 가능
        t.transform.rotation = msg.quat

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = ImuTfBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()