#!/usr/bin/env python3

import csv
import os
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from message_filters import ApproximateTimeSynchronizer, Subscriber


class SimStateLogger(Node):
    """Log synchronized base pose and non-wheel joint positions to CSV."""

    def __init__(self):
        super().__init__("sim_state_logger")

        # ============================================================
        # Parameters
        # ============================================================
        self.declare_parameter(
            "joint_topic_name",
            "/sim_joint_states",
        )
        self.declare_parameter(
            "pose_topic_name",
            "/sim_pose",
        )
        self.declare_parameter(
            "csv_path",
            "",
        )
        self.declare_parameter(
            "use_msg_time",
            True,
        )

        # Message synchronization
        self.declare_parameter(
            "sync_queue_size",
            50,
        )
        self.declare_parameter(
            "sync_slop_sec",
            0.02,
        )

        # CSV logging period
        self.declare_parameter(
            "log_interval_sec",
            0.02,
        )

        self.joint_topic_name = str(
            self.get_parameter("joint_topic_name").value
        )
        self.pose_topic_name = str(
            self.get_parameter("pose_topic_name").value
        )
        self.use_msg_time = bool(
            self.get_parameter("use_msg_time").value
        )
        self.sync_queue_size = int(
            self.get_parameter("sync_queue_size").value
        )
        self.sync_slop_sec = float(
            self.get_parameter("sync_slop_sec").value
        )
        self.log_interval_sec = float(
            self.get_parameter("log_interval_sec").value
        )

        if self.log_interval_sec <= 0.0:
            raise ValueError(
                "log_interval_sec must be greater than zero."
            )

        csv_path_param = str(
            self.get_parameter("csv_path").value
        )

        # ============================================================
        # Wheel joints excluded from CSV
        # ============================================================
        self.excluded_joint_names = {
            "wheel_L_Joint",
            "wheel_R_Joint",
        }

        # ============================================================
        # CSV path
        # ============================================================
        if csv_path_param:
            self.csv_path = os.path.abspath(
                os.path.expanduser(csv_path_param)
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_states_{timestamp}.csv"
            self.csv_path = os.path.abspath(filename)

        csv_directory = os.path.dirname(self.csv_path)

        if csv_directory:
            os.makedirs(csv_directory, exist_ok=True)

        self.csv_file = open(
            self.csv_path,
            mode="w",
            newline="",
            encoding="utf-8",
        )
        self.csv_writer = csv.writer(self.csv_file)

        # ============================================================
        # Internal state
        # ============================================================
        self.header_written = False

        # Joint names actually written to CSV
        self.logged_joint_names: list[str] = []

        # Indices corresponding to logged_joint_names when the incoming
        # JointState ordering remains unchanged
        self.logged_joint_indices: list[int] = []

        # Full joint order from the first valid JointState
        self.reference_joint_names: list[str] = []

        self.row_count = 0

        # Last simulation timestamp written to CSV
        self.last_logged_time_sec: Optional[float] = None

        # ============================================================
        # Subscribers and synchronizer
        # ============================================================
        qos_profile = QoSProfile(depth=50)

        self.pose_subscription = Subscriber(
            self,
            PoseStamped,
            self.pose_topic_name,
            qos_profile=qos_profile,
        )

        self.joint_subscription = Subscriber(
            self,
            JointState,
            self.joint_topic_name,
            qos_profile=qos_profile,
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [
                self.pose_subscription,
                self.joint_subscription,
            ],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
            allow_headerless=False,
        )

        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            f"Pose topic: {self.pose_topic_name}"
        )
        self.get_logger().info(
            f"Joint topic: {self.joint_topic_name}"
        )
        self.get_logger().info(
            f"Synchronization slop: {self.sync_slop_sec:.6f} sec"
        )
        self.get_logger().info(
            f"CSV logging interval: {self.log_interval_sec:.3f} sec"
        )
        self.get_logger().info(
            f"Excluded joints: {sorted(self.excluded_joint_names)}"
        )
        self.get_logger().info(
            f"CSV logging path: {self.csv_path}"
        )

    def synchronized_callback(
        self,
        pose_msg: PoseStamped,
        joint_msg: JointState,
    ) -> None:
        """Write one synchronized pose/joint pair every 0.1 seconds."""

        if len(joint_msg.position) == 0:
            self.get_logger().warning(
                "Received JointState with an empty position array."
            )
            return

        if len(joint_msg.name) != len(joint_msg.position):
            self.get_logger().error(
                "JointState name and position lengths differ. "
                "Wheel joints cannot be excluded reliably."
            )
            return

        # ------------------------------------------------------------
        # Representative timestamp
        # ------------------------------------------------------------
        if self.use_msg_time:
            time_sec = self._stamp_to_sec(
                joint_msg.header.stamp
            )
        else:
            time_sec = (
                self.get_clock().now().nanoseconds * 1.0e-9
            )

        # ------------------------------------------------------------
        # Log only once every log_interval_sec
        # ------------------------------------------------------------
        if self.last_logged_time_sec is not None:
            time_delta = time_sec - self.last_logged_time_sec

            # Simulation time may reset when Stop/Play is repeated.
            if time_delta < -1.0e-9:
                self.get_logger().warning(
                    "Simulation time moved backwards. "
                    "Resetting the CSV logging timer."
                )
                self.last_logged_time_sec = None

            elif time_delta + 1.0e-9 < self.log_interval_sec:
                return

        # ------------------------------------------------------------
        # Initialize CSV header
        # ------------------------------------------------------------
        if not self.header_written:
            if not self._write_header(joint_msg):
                return

        # Keep joint positions in the same order as the CSV header.
        joint_positions = self._get_ordered_joint_positions(
            joint_msg
        )

        if joint_positions is None:
            return

        # ------------------------------------------------------------
        # Base world pose
        # ------------------------------------------------------------
        position = pose_msg.pose.position
        orientation = pose_msg.pose.orientation

        row = [
            time_sec,

            # Base world position
            position.x,
            position.y,
            position.z,

            # ROS quaternion order: x, y, z, w
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,

            # Non-wheel joint positions
            *joint_positions,
        ]

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.row_count += 1
        self.last_logged_time_sec = time_sec

    def _write_header(
        self,
        joint_msg: JointState,
    ) -> bool:
        """Create the CSV header while excluding wheel joints."""

        self.reference_joint_names = list(joint_msg.name)

        self.logged_joint_indices = [
            index
            for index, joint_name in enumerate(joint_msg.name)
            if joint_name not in self.excluded_joint_names
        ]

        self.logged_joint_names = [
            joint_msg.name[index]
            for index in self.logged_joint_indices
        ]

        excluded_found = {
            name
            for name in joint_msg.name
            if name in self.excluded_joint_names
        }

        missing_excluded_names = (
            self.excluded_joint_names - excluded_found
        )

        if missing_excluded_names:
            self.get_logger().warning(
                "The following configured wheel joints were not found: "
                f"{sorted(missing_excluded_names)}"
            )

        if not self.logged_joint_names:
            self.get_logger().error(
                "No joints remain after wheel-joint filtering."
            )
            return False

        header = [
            "time_sec",

            "base_pos_x",
            "base_pos_y",
            "base_pos_z",

            "base_quat_x",
            "base_quat_y",
            "base_quat_z",
            "base_quat_w",

            *self.logged_joint_names,
        ]

        self.csv_writer.writerow(header)
        self.csv_file.flush()

        self.header_written = True

        self.get_logger().info(
            "CSV header written: "
            f"base pose + {len(self.logged_joint_names)} non-wheel joints."
        )

        return True

    def _get_ordered_joint_positions(
        self,
        joint_msg: JointState,
    ) -> Optional[list[float]]:
        """Return non-wheel joint positions in CSV-header order."""

        current_joint_names = list(joint_msg.name)

        # Fast path: incoming order has not changed.
        if current_joint_names == self.reference_joint_names:
            return [
                float(joint_msg.position[index])
                for index in self.logged_joint_indices
            ]

        # Handle changed JointState ordering by joint name.
        position_by_name = {
            name: float(position)
            for name, position in zip(
                joint_msg.name,
                joint_msg.position,
            )
        }

        missing_joint_names = [
            name
            for name in self.logged_joint_names
            if name not in position_by_name
        ]

        if missing_joint_names:
            self.get_logger().error(
                "Required joints are missing from JointState: "
                f"{missing_joint_names}. Skipping this row."
            )
            return None

        return [
            position_by_name[name]
            for name in self.logged_joint_names
        ]

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        """Convert ROS Time to floating-point seconds."""

        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1.0e-9
        )

    def close(self) -> tuple[str, int]:
        """Close the CSV file without using the ROS logger."""

        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

        return self.csv_path, self.row_count


def main(args=None):
    rclpy.init(args=args)

    node = SimStateLogger()
    interrupted = False

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        interrupted = True

    finally:
        csv_path, row_count = node.close()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        if interrupted:
            print("\nKeyboard interrupt received. Shutting down.")

        print(f"CSV file closed: {csv_path}")
        print(f"Total logged rows: {row_count}")


if __name__ == "__main__":
    main()