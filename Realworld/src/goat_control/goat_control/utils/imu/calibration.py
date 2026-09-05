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


def simple_accelerometer_calibration(ser: serial.Serial) -> bool:
    """
    Run EBIMU-9DOFV5 Simple Accelerometer Calibration.

    IMPORTANT:
        - Robot base must be mechanically level.
        - Robot must remain completely stationary.
    """

    print()
    print("============================================")
    print(" Simple Accelerometer Calibration")
    print("============================================")
    print("Keep the robot BASE LEVEL and STATIONARY.")
    print()

    input("Press ENTER to start calibration...")

    # Remove already-buffered streaming packets.
    ser.reset_input_buffer()

    print("[IMU] Sending <cas> ...")

    ser.write(b"<cas>")
    ser.flush()

    response = wait_for_response(ser)

    if response == "<ok>":
        print("[IMU] <ok> received.")
        print("[SUCCESS] Accelerometer calibration completed.")

        # Allow AHRS to settle after calibration
        time.sleep(1.0)

        return True

    if response == "<er>":
        print("[FAILED] IMU returned <er>.")
        return False

    print("[FAILED] Response timeout.")
    return False


def main():
    print(f"[IMU] Opening {IMU_PORT} @ {IMU_BAUDRATE} bps")

    try:
        with serial.Serial(port=IMU_PORT, baudrate=IMU_BAUDRATE, timeout=IMU_TIMEOUT) as ser:

            # Give serial device a little time after opening.
            time.sleep(0.2)

            success = simple_accelerometer_calibration(ser)

            if success:
                print()
                print("Calibration data is stored inside the IMU.")
                print("You may now close this program and restart the controller.")
            else:
                print()
                print("Calibration failed.")

    except serial.SerialException as e:
        print(f"[ERROR] Failed to open IMU serial port: {e}")

    except KeyboardInterrupt:
        print("\n[IMU] Calibration cancelled.")


if __name__ == "__main__":
    main()