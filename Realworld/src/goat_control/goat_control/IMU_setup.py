import serial
import threading
import time


import rclpy
from rclpy.node import Node

from motor_interfaces.msg import BaseStates  


PORT = "/dev/ttyUSB0"   
BAUD = 115200

class IMUsetup(Node):
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1):
        super().__init__('imu_setup')
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)   # Open serial port
        self.imu_data = [0.0]*14  # Initialize with zeros
        self.lock = threading.Lock()

        # Topic publisher
        self.pub = self.create_publisher(BaseStates, 'imusetup', 50)
        
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

        self.timer = self.create_timer(0.01, self.publish_timer_callback)

    def read_loop(self):
            while rclpy.ok():
                try:
                    raw_data = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if not raw_data:
                        continue
                    if not raw_data.startswith('*'):
                        continue

                    data_string = raw_data[1:].split(',')

                    try:
                        data = list(map(float, data_string))
                    except ValueError:
                        continue

                    if len(data) != 14:
                        continue

                    with self.lock:
                        self.imu_data = data

                except Exception as e:
                    self.get_logger().warn(f"IMU data read error: {e}")
                    time.sleep(0.01)
    def split_packet(self, data_list):
        """Refine raw data vector into IMU data dictionary."""
        if len(data_list) != 14:
            # Wrong packet size
            return None
        
        w, x, y, z = data_list[0:4]       # Quaternions
        gx, gy, gz = data_list[4:7]       # Angular velocity
        ax, ay, az = data_list[7:10]      # Linear acceleration
        mx, my, mz = data_list[10:13]     # Magnetic field
        t_ms = data_list[13]              # Timestamp in milliseconds

        imu_data = {
            "quat": {"w": w, "x": x, "y": y, "z": z},
            "gyro": {"x": gx, "y": gy, "z": gz},   # 보통 deg/s(설정 확인)
            "acc":  {"x": ax, "y": ay, "z": az},   # m/s^2 또는 g(설정 확인)
            "mag":  {"x": mx, "y": my, "z": mz},   # 보통 µT
            "time_ms": t_ms
        }
        return imu_data
    def publish_timer_callback(self):
        # 최신 데이터 복사
        with self.lock:
            data_list = self.imu_data.copy()

        pkt = self.split_packet(data_list)
        if not pkt:
            return

        q = pkt["quat"]
        g = pkt["gyro"]
        a = pkt["acc"]
        m = pkt["mag"]
        t = pkt["time_ms"]

        # 메시지 생성
        msg = BaseStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        msg.quat.w = q["w"]
        msg.quat.x = q["x"]
        msg.quat.y = q["y"]
        msg.quat.z = q["z"]

        msg.gyro.x = g["x"]
        msg.gyro.y = g["y"]
        msg.gyro.z = g["z"]

        msg.acc.x = a["x"]
        msg.acc.y = a["y"]
        msg.acc.z = a["z"]

        msg.mag.x = m["x"]
        msg.mag.y = m["y"]
        msg.mag.z = m["z"]

        msg.time_ms = float(t)

        # 퍼블리시
        self.pub.publish(msg)

        # # 기존처럼 터미널에 출력 (원하면 유지, 아니면 지워도 됨)
        # print(
        #     f"[{t:5.0f} ms] "
        #     f"quat=({q['w']:8.4f},{q['x']:8.4f},{q['y']:8.4f},{q['z']:8.4f}) "
        #     f"gyro=({g['x']:7.3f},{g['y']:7.3f},{g['z']:7.3f}) "
        #     f"acc=({a['x']:7.3f},{a['y']:7.3f},{a['z']:7.3f}) "
        #     f"mag=({m['x']:6.1f},{m['y']:6.1f},{m['z']:6.1f})"
        # )


    # def imu_parser(self, lock):
    #     while True:
    #         try:
    #             raw_data = self.serial.readline().decode('utf-8', errors='ignore').strip()
    #             if not raw_data:
    #                 continue
    #             if not raw_data.startswith('*'):
    #                 continue
    #             data_string = raw_data[1:].split(',')
                
    #             # Turn to float list
    #             try:
    #                 data = list(map(float, data_string))
    #             except ValueError:
    #                 continue
    #             with lock:
    #                 self.imu_data = data
    #         except Exception:
    #             print("IMU data read error")
    #         continue

def main(args=None):
    rclpy.init(args=args)
    node = IMUsetup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down IMU node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()