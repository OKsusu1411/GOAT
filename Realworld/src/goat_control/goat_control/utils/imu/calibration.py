from __future__ import annotations

import time
import serial


IMU_PORT = "/dev/ttyUSB0"
IMU_BAUDRATE = 115200
IMU_TIMEOUT = 0.1

COMMAND_TIMEOUT = 3.0


def wait_for_response(ser: serial.Serial, timeout: float = COMMAND_TIMEOUT) -> str | None:
    """Wait for EBIMU response such as <ok> or <er>."""

    deadline = time.monotonic() + timeout
    buffer = bytearray()

    while time.monotonic() < deadline:
        byte = ser.read(1)

        if not byte:
            continue

        buffer.extend(byte)

        # Keep buffer from growing indefinitely
        if len(buffer) > 1024:
            buffer = buffer[-1024:]

        if b"<ok>" in buffer:
            return "<ok>"

        if b"<er>" in buffer:
            return "<er>"

    return None


def simple_accelerometer_calibration(port: str = "/dev/ttyUSB0",
                                     baudrate: int = 115200,
                                     serial_timeout: float = 0.1,
                                     command_timeout: float = 3.0) -> bool:

    try:
        with serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=serial_timeout,
        ) as ser:

            time.sleep(0.2)

            ser.reset_input_buffer()

            ser.write(b"<cas>")
            ser.flush()

            deadline = time.monotonic() + command_timeout
            buffer = bytearray()

            while time.monotonic() < deadline:
                byte = ser.read(1)

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
    print(" Simple Accelerometer Calibration")
    print("============================================")
    print("Keep the robot BASE LEVEL and STATIONARY.")
    print()

    input("Press ENTER to start calibration...")

    print("[IMU] Sending <cas> ...")

    success = simple_accelerometer_calibration()

    if success:
        print("[IMU] <ok> received.")
        print("[SUCCESS] Accelerometer calibration completed.")
    else:
        print("[FAILED] Accelerometer calibration failed.")



if __name__ == "__main__":
    main()