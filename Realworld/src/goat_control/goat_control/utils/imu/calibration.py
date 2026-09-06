from __future__ import annotations

import time
import serial

# def wait_for_response(imu_serial: serial.Serial, timeout: float = 3.0) -> str | None:
#     """Wait for EBIMU response such as <ok> or <er>."""

#     deadline = time.monotonic() + timeout
#     buffer = bytearray()

#     while time.monotonic() < deadline:
#         byte = imu_serial.read(1)

#         if not byte:
#             continue

#         buffer.extend(byte)

#         # Keep buffer from growing indefinitely
#         if len(buffer) > 1024:
#             buffer = buffer[-1024:]

#         if b"<ok>" in buffer:
#             return "<ok>"

#         if b"<er>" in buffer:
#             return "<er>"

#     return None


def simple_accelerometer_calibration(port: str = "/dev/ttyUSB0",
                                     baudrate: int = 115200,
                                     serial_timeout: float = 0.1,
                                     command_timeout: float = 3.0) -> bool:

    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=serial_timeout) as imu_serial:

            time.sleep(0.2)

            imu_serial.reset_input_buffer()

            imu_serial.write(b"<cas>")
            imu_serial.flush()

            deadline = time.monotonic() + command_timeout
            buffer = bytearray()

            while time.monotonic() < deadline:
                byte = imu_serial.read(1)

                if not byte:
                    continue

                buffer.extend(byte)

                if b"<ok>" in buffer:
                    return True

                if b"<er>" in buffer:
                    return False

            return False

    except serial.SerialException:
        return False

def magnetometer_calibration(port: str = "/dev/ttyUSB0",
                             baudrate: int = 115200,
                             serial_timeout: float = 0.1,
                             command_timeout: float = 3.0) -> bool:

    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=serial_timeout) as imu_serial:

            time.sleep(0.2)

            imu_serial.reset_input_buffer()

            imu_serial.write(b"<cg>")
            imu_serial.flush()

            deadline = time.monotonic() + command_timeout
            buffer = bytearray()

            while time.monotonic() < deadline:
                byte = imu_serial.read(1)

                if not byte:
                    continue

                buffer.extend(byte)

                if b"<ok>" in buffer:
                    return True

                if b"<er>" in buffer:
                    return False

            return False

    except serial.SerialException:
        return False

def main():
    print("[IMU] Opening /dev/ttyUSB0 @ 115200 bps")
    print()
    print("============================================")
    print(" IMU Accelerometer, Magnetometer Calibration")
    print("============================================")
    print("Keep the robot Upright and Steady.")
    print()

    input("Press ENTER to start calibration...")

    print("[IMU] Calibrating ...")

    success_acc = simple_accelerometer_calibration()

    if success_acc:
        print("[SUCCESS] Accelerometer calibration completed.")
    else:
        print("[FAILED] Accelerometer calibration failed.")

    success_mag = magnetometer_calibration()

    if success_mag:
        print("[SUCCESS] Magnetometer calibration completed.")
    else:
        print("[FAILED] Magnetometer calibration failed.")


if __name__ == "__main__":
    main()