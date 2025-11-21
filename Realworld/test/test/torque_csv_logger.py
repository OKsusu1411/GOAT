import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pynput import keyboard

import csv
import os
from datetime import datetime
import math

class TorqueTestPublisher(Node):
    def __init__(self):
        super().__init__('torque_test_publisher')
        # ===== Publisher =====
        self.publisher = self.create_publisher(Float32MultiArray, 'torque_commands', 10)
        timer_period = 0.1  # 10Hz (100ms)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # ===== Runtime state =====
        self.torque = 0.0
        self.command = [0.0]*8
        self.mass = 0.0
        self.joint_length = 0.252  # m
        self.wheel_radius = 0.252  # m

        # ===== Parameters for temperature subscription =====
        self.declare_parameter('temp_topic', '/motor_state2')   # Float32MultiArray 기대
        self.declare_parameter('temp_index', 0)                 # 온도가 담긴 인덱스
        self.declare_parameter('log_dir', 'logs')               # CSV 저장 폴더

        self.temp_topic = self.get_parameter('temp_topic').get_parameter_value().string_value
        self.temp_index = int(self.get_parameter('temp_index').get_parameter_value().integer_value)
        self.log_dir   = self.get_parameter('log_dir').get_parameter_value().string_value

        # ===== Subscribe motor feedback (temperature etc.) =====
        # 기대형태: Float32MultiArray; data[self.temp_index]가 온도(℃)
        self.last_temp_c = math.nan
        self.last_state  = None
        self.state_sub = self.create_subscription(
            Float32MultiArray, self.temp_topic, self.state_callback, 20
        )

        # ===== CSV logger =====
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.log_dir, f'motor_temperature_{ts}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        # 헤더: 그래프 그릴 때 편한 최소구성 + 부가정보
        self.csv_writer.writerow([
            'iso_time', 'ros_sec', 'ros_nsec',
            'temperature_c',
            'cmd0','cmd1','cmd2','cmd3','cmd4','cmd5','cmd6','cmd7',
            'torque_Nm','mass_g'
        ])
        self.get_logger().info(f'CSV logging to: {self.csv_path}')

    # ===== conversions =====
    def joint_torque2current(self, torque):
        if torque != 0.0:
            current = (torque - 0.0081) / 0.2552  # for joint
        else:
            current = 0.0
        return current

    def wheel_torque2current(self, torque):
        if torque != 0.0:
            current = (torque - 0.0052) / 0.2569  # for wheel
        else:
            current = 0.0
        return current

    def current2mass(self, current, jointorwheel):
        if jointorwheel == self.joint_length:
            self.mass = (0.2552*current+0.0081)*1000/(self.joint_length*9.81)
        elif jointorwheel == self.wheel_radius:
            self.mass = (0.2569*current+0.0052)*1000/(self.wheel_radius*9.81)

    # ===== feedback callback =====
    def state_callback(self, msg: Float32MultiArray):
        self.last_state = list(msg.data) if msg.data is not None else None
        if self.last_state is not None and len(self.last_state) > self.temp_index:
            self.last_temp_c = float(self.last_state[self.temp_index])

    # ===== command update (외부에서 호출 가정) =====
    def command_update(self, comm):
        comm = [
            self.joint_torque2current(comm[0]),
            self.joint_torque2current(comm[1]),
            self.joint_torque2current(comm[2]),
            self.wheel_torque2current(comm[3]),
            self.joint_torque2current(comm[4]),
            self.joint_torque2current(comm[5]),
            self.joint_torque2current(comm[6]),
            self.wheel_torque2current(comm[7]),
        ]
        self.command = comm

    # ===== periodic publish + log =====
    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = self.command
        self.publisher.publish(msg)

        # 로깅: ROS 시간 + ISO 벽시계 + 최근 온도(없으면 NaN)
        now = self.get_clock().now().to_msg()  # builtin_interfaces/Time
        iso = datetime.now().isoformat(timespec='milliseconds')
        row = [
            iso, now.sec, now.nanosec,
            self.last_temp_c,
            *self.command,
            self.torque, self.mass
        ]
        self.csv_writer.writerow(row)

        self.get_logger().info(
            f'Published torque: {self.torque:.2f} Nm, '
            f'commands: {self.command} '
            f'mass: {self.mass:.1f} g, '
            f'T[°C]: {self.last_temp_c if not math.isnan(self.last_temp_c) else float("nan")}'
        )

    # ===== graceful close =====
    def close(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

# 키보드 컨트롤(필요하면 살려서 사용)
def on_press(key, node):
    try:
        if key == keyboard.Key.up:
            node.torque += 0.2
        elif key == keyboard.Key.down:
            node.torque -= 0.2
        elif hasattr(key, 'char') and key.char == 's':
            node.torque = 0.0
        # 커맨드 갱신 예시(4번 조인트를 토크→전류 매핑)
        node.command[3] = node.wheel_torque2current(node.torque)
    except Exception as e:
        print(f"Error on key press: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TorqueTestPublisher()

    listener = keyboard.Listener(on_press=lambda key: on_press(key, node))
    listener.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        node.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
