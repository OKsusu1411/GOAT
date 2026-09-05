#!/usr/bin/env python3
from __future__ import annotations

import rclpy

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from motor_interfaces.msg import ImuState


class ImuTfPublisher(Node):
    """
    Transform custom imu msg type into Rviz imu msg type and publish.

    Subscribe: /imu_data (motor_interfaces/msg/ImuState)
    Publish TF: odom -> base_link (rotation = imu quat)
    """
    def __init__(self):
        super().__init__("imu_tf_publisher")

        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_link")
        self.declare_parameter("use_translation_zero", True)

        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        self.parent_frame = self.get_parameter("parent_frame").get_parameter_value().string_value
        self.child_frame = self.get_parameter("child_frame").get_parameter_value().string_value
        self.use_translation_zero = self.get_parameter("use_translation_zero").get_parameter_value().bool_value
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(ImuState, imu_topic, self.cb, qos_profile)

        self.get_logger().info(
            f"Listening {imu_topic}, publishing TF {self.parent_frame} -> {self.child_frame}")

    def cb(self, msg: ImuState):
        t = TransformStamped()
        # Use the source message stamp (sim clock) so the TF timeline stays
        # consistent with /joint_states under use_sim_time. Falls back to the
        # current clock only if the incoming header has no stamp.
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            t.header.stamp = self.get_clock().now().to_msg()
        else:
            t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        if self.use_translation_zero:
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0

        t.transform.rotation = msg.quat

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = ImuTfPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()