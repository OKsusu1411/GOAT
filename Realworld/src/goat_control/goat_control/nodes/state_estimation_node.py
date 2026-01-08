# src/goat_control/goat_control/nodes/state_estimation_node.py
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates

from goat_control.core.comm import CanInterface, MotorDriver, MotorParams
from goat_control.core.estimation.imu import ImuSerialReader, ImuConfig
from goat_control.core.estimation.state_manager import MotorStateCollector


class StateEstimationNode(Node):
    """
    This node is responsible for estimating the state of the robot.
    It reads the motor states from the CAN bus and the IMU data from the serial port.
    It then publishes the robot's state as ROS2 messages.
    """
    def __init__(self):
        super().__init__("state_estimation_node")

        # Parameters
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("motor_node_ids", [1, 2, 3, 4, 5, 6, 7, 8])
        self.declare_parameter("estimation_rate_hz", 200.0)
        self.declare_parameter("imu_port", "/dev/ttyUSB0")
        self.declare_parameter("imu_baudrate", 115200)

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        estimation_rate_hz = float(self.get_parameter("estimation_rate_hz").value)
        imu_port = str(self.get_parameter("imu_port").value)
        imu_baudrate = int(self.get_parameter("imu_baudrate").value)

        # Publishers
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.imu_publisher = self.create_publisher(BaseStates, "imu_data", 10)

        # CAN Interface
        self.can_interface = CanInterface(channel=can_channel, interface=can_interface)
        self.can_interface.open()

        # Motor Drivers
        self.motor_drivers: list[MotorDriver] = []
        for node_id in motor_node_ids:
            params = MotorParams(node_id=int(node_id))
            self.motor_drivers.append(MotorDriver(self.can_interface, params))

        self.motor_state_collector = MotorStateCollector(self.motor_drivers)

        # IMU Reader
        imu_config = ImuConfig(port=imu_port, baudrate=imu_baudrate)
        self.imu_reader = ImuSerialReader(config=imu_config, logger=self.get_logger())
        self.imu_reader.open()

        # Estimation loop timer
        estimation_period_sec = 1.0 / estimation_rate_hz
        self.estimation_timer = self.create_timer(estimation_period_sec, self._estimation_loop)

        self.get_logger().info("StateEstimationNode started.")

    def _estimation_loop(self):
        now_time = self.get_clock().now().to_msg()

        # 1. Poll motor states
        motor_states_data = self.motor_state_collector.poll_all()

        # 2. Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = now_time
        joint_state_msg.name = [f"joint_{i}" for i in range(len(motor_states_data.positions_rad))]
        joint_state_msg.position = motor_states_data.positions_rad.tolist()
        joint_state_msg.velocity = motor_states_data.velocities_rad_per_sec.tolist()
        joint_state_msg.effort = motor_states_data.torques_nm.tolist()
        self.joint_state_publisher.publish(joint_state_msg)

        # 3. Poll IMU data
        imu_packet = self.imu_reader.get_latest_packet()

        # 4. Publish IMU data
        if imu_packet:
            imu_msg = BaseStates()
            imu_msg.header.stamp = now_time
            imu_msg.quat.w = imu_packet.quat_w
            imu_msg.quat.x = imu_packet.quat_x
            imu_msg.quat.y = imu_packet.quat_y
            imu_msg.quat.z = imu_packet.quat_z
            imu_msg.gyro.x = imu_packet.gyro_x
            imu_msg.gyro.y = imu_packet.gyro_y
            imu_msg.gyro.z = imu_packet.gyro_z
            imu_msg.acc.x = imu_packet.acc_x
            imu_msg.acc.y = imu_packet.acc_y
            imu_msg.acc.z = imu_packet.acc_z
            imu_msg.mag.x = imu_packet.mag_x
            imu_msg.mag.y = imu_packet.mag_y
            imu_msg.mag.z = imu_packet.mag_z
            imu_msg.time_ms = imu_packet.time_ms
            self.imu_publisher.publish(imu_msg)

    def destroy_node(self):
        self.imu_reader.close()
        self.can_interface.close()
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
