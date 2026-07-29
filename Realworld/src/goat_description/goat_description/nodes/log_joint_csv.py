#!/usr/bin/env python3

import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class SimJointStateLogger(Node):
    def __init__(self):
        super().__init__("sim_joint_state_logger")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("topic_name", "/sim_joint_states")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("use_msg_time", True)

        self.topic_name = (
            self.get_parameter("topic_name")
            .get_parameter_value()
            .string_value
        )

        csv_path_param = (
            self.get_parameter("csv_path")
            .get_parameter_value()
            .string_value
        )

        self.use_msg_time = (
            self.get_parameter("use_msg_time")
            .get_parameter_value()
            .bool_value
        )

        # -----------------------------
        # CSV path
        # -----------------------------
        if csv_path_param:
            self.csv_path = os.path.abspath(os.path.expanduser(csv_path_param))
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_joint_states_{timestamp}.csv"
            self.csv_path = os.path.abspath(filename)

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.header_written = False
        self.joint_names = []

        # -----------------------------
        # Subscriber
        # -----------------------------
        self.subscription = self.create_subscription(
            JointState,
            self.topic_name,
            self.joint_state_callback,
            10,
        )

        self.get_logger().info(f"Subscribing to topic: {self.topic_name}")
        self.get_logger().info(f"CSV logging path: {self.csv_path}")

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) == 0:
            self.get_logger().warn("Received JointState with empty position array.")
            return

        # Write CSV header once
        if not self.header_written:
            if len(msg.name) == len(msg.position):
                self.joint_names = list(msg.name)
            else:
                self.get_logger().warn(
                    "JointState name and position lengths differ. "
                    "Using generic joint names."
                )
                self.joint_names = [
                    f"joint_{i}" for i in range(len(msg.position))
                ]

            header = ["time_sec"] + self.joint_names
            self.csv_writer.writerow(header)
            self.header_written = True

            self.get_logger().info(
                f"CSV header written with {len(self.joint_names)} joints."
            )

        # Time column
        if self.use_msg_time:
            time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            now = self.get_clock().now().nanoseconds
            time_sec = now * 1e-9

        row = [time_sec] + list(msg.position)
        self.csv_writer.writerow(row)

        # Optional: ensure data is written frequently
        self.csv_file.flush()

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"CSV file closed: {self.csv_path}")


def main(args=None):
    rclpy.init(args=args)

    node = SimJointStateLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()