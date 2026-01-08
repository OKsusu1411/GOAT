# src/goat_control/goat_control/nodes/state_estimation_node.py
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates

from goat_control.core.comm import CanInterface, MotorDriver, MotorParams
from goat_control.core.estimation.imu import ImuSerialReader, ImuConfig
from goat_control.core.estimation.state_manager import MotorStateCollector
from goat_control.core import launch_core_control_system


class StateEstimationNode(Node):
    """
    This node is responsible for estimating the state of the robot.
    It reads the motor states from the CAN bus and the IMU data from the serial port.
    It converts raw data into physical units (rad, rad/s, Nm) and publishes ROS2 messages.
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
        self.declare_parameter("yaml_path", "goat_config.yaml")

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        estimation_rate_hz = float(self.get_parameter("estimation_rate_hz").value)
        imu_port = str(self.get_parameter("imu_port").value)
        imu_baudrate = int(self.get_parameter("imu_baudrate").value)
        yaml_path = str(self.get_parameter("yaml_path").value)

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

        # [수정] ControlNode와 동일한 방식으로 Core 시스템 로드
        # pipeline을 받아와서 내부의 state_manager를 공유 사용 (중복 생성 방지)
        self.goat_model, self.control_pipeline = launch_core_control_system(
            yaml_path=yaml_path,
            motor_drivers=self.motor_drivers,
            effort_output_mode="torque_nm",
        )
        
        # Pipeline 내부의 manager 사용
        self.state_manager = self.control_pipeline.state_manager
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

        # 1. Poll motor states (Raw Data)
        motor_states_data = self.motor_state_collector.poll_all()

        # 2. Convert Raw Data -> RobotState (Physical Units)
        # StateManager를 사용하여 raw 데이터를 rad, rad/s, Nm 단위로 변환합니다.
        robot_state = self.state_manager.build_robot_state(motor_states_data)

        # 3. Publish joint states
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = now_time
        
        # 모델에 정의된 관절 이름 사용
        self.get_logger().info(f"Joint names: {self.goat_model.joint_names}")
        joint_state_msg.name = self.goat_model.joint_names 
        
        # RobotState에서 변환된 값 사용
        joint_state_msg.position = robot_state.joint_position_rad
        joint_state_msg.velocity = robot_state.joint_velocity_rad_per_sec
        joint_state_msg.effort = robot_state.joint_effort_like
        
        self.joint_state_publisher.publish(joint_state_msg)

        # 4. Poll IMU data
        imu_packet = self.imu_reader.get_latest_packet()

        # 5. Publish IMU data
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