# src/goat_control/goat_control/nodes/imu_io.py
from __future__ import annotations

from motor_interfaces.msg import ImuState

from goat_control.utils.imu.imu_manager import ImuSerialReader, ImuConfig


class ImuIO:
    """In-process IMU reader (no longer a ROS2 Node).

    Owns the IMU serial port. Instantiated and driven directly by
    ControllerNode: every control tick calls `read_imu()`, which returns the
    latest decoded ImuState. Mirrors MotorIO's pattern.

    Replaces the old imu_io_node + `/imu` subscription with a direct function
    call, eliminating the cross-process DDS latency.
    """

    def __init__(self, cfg: dict, logger,
                 imu_port: str, imu_baudrate: int, imu_timeout: float):
        # Config + logger are owned by ControllerNode and passed in.
        self.cfg = cfg
        self.logger = logger

        # IMU Reader (background thread reads serial continuously).
        imu_config = ImuConfig(
            port=imu_port,
            baudrate=imu_baudrate,
            timeout=imu_timeout,
            yaml=self.cfg,
        )
        self.imu_reader = ImuSerialReader(config=imu_config, logger=self.logger)
        self.imu_reader.open()

        # Initial poll so `latest_imu_state` is non-None on tick 0.
        self.latest_imu_state = ImuState()

        self.logger.info("[ImuIO] initialized — owns IMU serial (in-process).")

    def read_imu(self) -> ImuState:
        """Fetch the latest IMU packet, cache as `latest_imu_state`, return it."""
        msg = ImuState()
        packet = self.imu_reader.get_latest_packet()
        if packet is None:
            return msg

        # Quaternion - unitless
        msg.quat.w = float(packet.quat_w)
        msg.quat.x = float(packet.quat_x)
        msg.quat.y = float(packet.quat_y)
        msg.quat.z = float(packet.quat_z)

        # Gyroscope (angular velocity) - rad/s
        msg.gyro.x = float(packet.gyro_x)
        msg.gyro.y = float(packet.gyro_y)
        msg.gyro.z = float(packet.gyro_z)

        # Linear velocity - m/s
        msg.acc.x = float(packet.acc_x)
        msg.acc.y = float(packet.acc_y)
        msg.acc.z = float(packet.acc_z)

        # Magnetometer
        msg.mag.x = float(packet.mag_x)
        msg.mag.y = float(packet.mag_y)
        msg.mag.z = float(packet.mag_z)

        msg.time_ms = float(packet.time_ms)

        self.latest_imu_state = msg
        return self.latest_imu_state

    def close(self) -> None:
        """Stop the reader thread and close the serial port."""
        try:
            self.imu_reader.close()
        except Exception:
            pass
