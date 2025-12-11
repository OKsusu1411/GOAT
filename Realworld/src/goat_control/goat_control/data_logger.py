import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray # for target_joint_angles
from motor_interfaces.msg import MotorStates # Import MotorStates message type
import csv
import os
from datetime import datetime
import numpy as np

# Assuming a fixed number of joints for consistency with pd_controller and states_pub
NUM_JOINTS = 8

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        
        # 데이터 저장을 위한 디렉토리 경로
        self.log_dir = os.path.join('src', 'goat_control', 'datalogs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 파일명에 타임스탬프 추가
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file_path = os.path.join(self.log_dir, f'datalog_{timestamp}.csv')
        
        # joint_states 구독
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_states_callback,
            10)
        
        # motor_states 구독
        self.motor_states_subscription = self.create_subscription(
            MotorStates,
            'motor_states',
            self.motor_states_callback,
            10)
        
        # target_joint_angles 구독
        self.target_angles_subscription = self.create_subscription(
            Float32MultiArray,
            'target_joint_angles', # pd_controller가 사용하는 토픽 이름
            self.target_angles_callback,
            10)
        
        # torque_commands 구독 (pd_controller가 발행)
        self.torque_commands_subscription = self.create_subscription(
            Float32MultiArray,
            'torque_commands',
            self.torque_commands_callback,
            10)
        
        self.data_buffer = []
        self.header_written = False
        
        # 현재 목표 각도 (topic으로부터 받음)
        # pd_controller의 target_angles_deg와 wheel_speed_ref_deg_s가 Float32MultiArray로 발행됨
        # 0~5는 각도(deg), 6~7은 휠 속도(deg/s)
        self.current_target_values = [float('nan')] * NUM_JOINTS 
        
        # 현재 토크 명령 (pd_controller로부터 받음)
        self.current_torque_commands = [float('nan')] * NUM_JOINTS

        # 현재 모터 속도 (motor_states로부터 받음, dps 단위)
        self.current_motor_speed_dps = [float('nan')] * NUM_JOINTS

        self.get_logger().info(f"Subscribing to 'joint_states', 'motor_states', 'target_joint_angles', and 'torque_commands'.")
        self.get_logger().info(f"Logging to {self.log_file_path}")

    def joint_states_callback(self, msg):
        """joint_states 토픽을 받을 때마다 호출되는 콜백"""
        
        # 헤더 생성 (첫 메시지 수신 시)
        if not self.header_written:
            header = ['time_sec', 'time_nanosec']
            header.extend([f'pos_{i}' for i in range(NUM_JOINTS)])
            header.extend([f'motor_speed_dps_{i}' for i in range(NUM_JOINTS)]) # Log motor_speed_dps instead of joint_states velocity
            header.extend([f'eff_{i}' for i in range(NUM_JOINTS)])
            header.extend([f'target_value_{i}' for i in range(NUM_JOINTS)])
            header.extend([f'torque_command_{i}' for i in range(NUM_JOINTS)])
            self.csv_header = header
            self.header_written = True

        # 데이터 행 생성
        timestamp = msg.header.stamp
        row = [timestamp.sec, timestamp.nanosec]
        row.extend(msg.position)
        row.extend(self.current_motor_speed_dps) # Log the motor speed in dps
        row.extend(msg.effort)
        row.extend(self.current_target_values) # 최신 목표 값 추가
        row.extend(self.current_torque_commands) # 최신 토크 명령 추가
        
        self.data_buffer.append(row)
        
    def motor_states_callback(self, msg: MotorStates):
        """motor_states 토픽 callback"""
        if len(msg.speed_dps) == NUM_JOINTS:
            self.current_motor_speed_dps = list(msg.speed_dps)
        else:
            self.get_logger().warn(
                f"Received motor_states speed_dps with unexpected length: {len(msg.speed_dps)}. Expected {NUM_JOINTS}."
            )

    def target_angles_callback(self, msg: Float32MultiArray):
        """target_joint_angles 토픽을 받을 때마다 호출되는 콜백"""
        if len(msg.data) == NUM_JOINTS:
            # pd_controller에서 발행하는 값은 deg 단위이므로 그대로 저장
            self.current_target_values = list(msg.data)
        else:
            self.get_logger().warn(
                f"Received target_joint_angles with unexpected length: {len(msg.data)}. Expected {NUM_JOINTS}."
            )

    def torque_commands_callback(self, msg: Float32MultiArray):
        """torque_commands 토픽 callback"""
        if len(msg.data) == NUM_JOINTS:
            self.current_torque_commands = list(msg.data)
        else:
            self.get_logger().warn(
                f"Received torque_commands with unexpected length: {len(msg.data)}. Expected {NUM_JOINTS}."
            )

    def save_data_to_csv(self):
        """수집된 데이터를 CSV 파일로 저장"""
        if not self.data_buffer:
            self.get_logger().warn("No data collected, skipping file save.")
            return
            
        self.get_logger().info(f"Saving {len(self.data_buffer)} data points to {self.log_file_path}...")
        
        try:
            with open(self.log_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if self.header_written:
                    writer.writerow(self.csv_header)
                writer.writerows(self.data_buffer)
            self.get_logger().info("Data saved successfully.")
        except IOError as e:
            self.get_logger().error(f"Failed to write to file: {e}")

    def destroy_node(self):
        self.save_data_to_csv()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    data_logger = DataLogger()
    
    try:
        rclpy.spin(data_logger)
    except KeyboardInterrupt:
        data_logger.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        # spin()이 반환되면 노드를 명시적으로 파괴하여
        # destroy_node()가 호출되도록 함
        data_logger.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
