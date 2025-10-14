import serial
import threading
import time

PORT = "/dev/ttyUSB0"   # 또는 "COM3"
BAUD = 115200

class IMUsetup:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1):
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)   # Open serial port
        self.imu_data = [0.0]*14  # Initialize with zeros

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

    def imu_parser(self, lock):
        while True:
            try:
                raw_data = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if not raw_data:
                    continue
                if not raw_data.startswith('*'):
                    continue
                data_string = raw_data[1:].split(',')
                
                # Turn to float list
                try:
                    data = list(map(float, data_string))
                except ValueError:
                    continue
                with lock:
                    self.imu_data = data
            except Exception:
                print("IMU data read error")
            continue

# def main():
#     imu = IMUsetup(PORT, BAUD)
#     ser = imu.serial
#     lock = threading.Lock()

#     thread = threading.Thread(target=imu.imu_parser, daemon=True)
#     thread.start()
#     time.sleep(0.3)

#     try:
#         while True:
#             with lock:
#                 data_list = imu.imu_data.copy()
                
#             pkt = imu.split_packet(data_list)
#             if pkt:
#                 # 보기 좋게 한 줄 요약 출력
#                 q = pkt["quat"]; g = pkt["gyro"]; a = pkt["acc"]; m = pkt["mag"]; t = pkt["time_ms"]
#                 print(f"[{t:.0f} ms] "
#                     f"quat=({q['w']:.4f},{q['x']:.4f},{q['y']:.4f},{q['z']:.4f}) "
#                     f"gyro=({g['x']:.3f},{g['y']:.3f},{g['z']:.3f}) "
#                     f"acc=({a['x']:.3f},{a['y']:.3f},{a['z']:.3f}) "
#                     f"mag=({m['x']:.1f},{m['y']:.1f},{m['z']:.1f})")
#             time.sleep(0.01)

#     except KeyboardInterrupt:
#         pass
#     finally:
#         try:
#             ser.close()
#         except Exception:
#             pass
