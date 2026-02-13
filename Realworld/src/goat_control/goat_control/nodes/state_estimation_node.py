# src/goat_control/goat_control/nodes/state_estimation_node.py
from __future__ import annotations

import rclpy
from rclpy.node import Node
from motor_interfaces.msg import BaseStates

from goat_control.core.estimation.imu import ImuSerialReader, ImuConfig


class StateEstimationNode(Node):
    """IMU-only state publisher.

    Refactor note
    -------------
    The previous refactor version of this node opened the CAN device to publish
    `joint_states` *and* MotorCommandNode opened CAN again to write commands.
    On real hardware this often causes unnecessary bus contention and makes
    timestamp alignment harder.

    With the new structure:
      - MotorIONode owns CAN (read+write) and publishes `joint_states`.
      - StateEstimationNode owns IMU serial and publishes `goat/imu_data`.
    """

    def __init__(self):
        super().__init__("state_estimation_node")

        # Parameters
        self.declare_parameter("estimation_rate_hz", 200.0)
        self.declare_parameter("imu_port", "/dev/ttyUSB0")
        self.declare_parameter("imu_baudrate", 115200)
        self.declare_parameter("imu_timeout", 1.0)

        estimation_rate_hz = float(self.get_parameter("estimation_rate_hz").value)
        imu_port = str(self.get_parameter("imu_port").value)
        imu_baudrate = int(self.get_parameter("imu_baudrate").value)
        imu_timeout = float(self.get_parameter("imu_timeout").value)

        # Publisher
        self.imu_publisher = self.create_publisher(BaseStates, "/goat/imu_data", 10)

        # IMU Reader
        imu_config = ImuConfig(port=imu_port, baudrate=imu_baudrate, timeout=imu_timeout)
        self.imu_reader = ImuSerialReader(config=imu_config, logger=self.get_logger())
        self.imu_reader.open()

        # Timer
        estimation_period_sec = 1.0 / max(estimation_rate_hz, 1.0)
        self.estimation_timer = self.create_timer(estimation_period_sec, self._tick)

        self.get_logger().info(
            "StateEstimationNode started (IMU-only). Publishing 'goat/imu_data'. "
            "Motor states are published by MotorIONode."
        )

    def _tick(self) -> None:
        packet = self.imu_reader.get_latest_packet()
        if packet is None:
            return

        msg = BaseStates()
        msg.quat.w = float(packet.quat_w)
        msg.quat.x = float(packet.quat_x)
        msg.quat.y = float(packet.quat_y)
        msg.quat.z = float(packet.quat_z)

        msg.gyro.x = float(packet.gyro_x)
        msg.gyro.y = float(packet.gyro_y)
        msg.gyro.z = float(packet.gyro_z)

        msg.acc.x = float(packet.acc_x)
        msg.acc.y = float(packet.acc_y)
        msg.acc.z = float(packet.acc_z)

        msg.mag.x = float(packet.mag_x)
        msg.mag.y = float(packet.mag_y)
        msg.mag.z = float(packet.mag_z)

        msg.time_ms = float(packet.time_ms)
        self.imu_publisher.publish(msg)

    def destroy_node(self):
        try:
            self.imu_reader.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
