# goat_control/core/estimation/imu.py
from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import serial


@dataclass(frozen=True)
class ImuConfig:
    """IMU serial configuration."""
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 1.0

    # Input format
    start_char: str = "*"
    expected_length: int = 14

    # Read loop behavior
    read_sleep_sec_on_error: float = 0.01


@dataclass
class ImuPacket:
    """Decoded IMU packet (same semantic fields as your split_packet)."""
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    acc_x: float
    acc_y: float
    acc_z: float

    mag_x: float
    mag_y: float
    mag_z: float

    time_ms: float

    def as_dict(self) -> Dict[str, Any]:
        """Compatibility helper: returns the original dict-style structure."""
        return {
            "quat": {"w": self.quat_w, "x": self.quat_x, "y": self.quat_y, "z": self.quat_z},
            "gyro": {"x": self.gyro_x, "y": self.gyro_y, "z": self.gyro_z},
            "acc":  {"x": self.acc_x,  "y": self.acc_y,  "z": self.acc_z},
            "mag":  {"x": self.mag_x,  "y": self.mag_y,  "z": self.mag_z},
            "time_ms": self.time_ms,
        }


class ImuSerialReader:
    """
    IMU serial reader and decoder.
    Responsibilities:
      - Open/close serial port
      - Background thread reads lines continuously
      - Keep the latest valid raw vector (length=14)
      - Provide decoded packet via get_latest_packet()

    Input line format (from your imu_publisher.py):
      *w,x,y,z,gx,gy,gz,ax,ay,az,mx,my,mz,t_ms
    """
    def __init__(self, config: ImuConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.serial_port: Optional[serial.Serial] = None

        self._latest_raw_vector: List[float] = [0.0] * self.config.expected_length
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # --------------------------
    # Lifecycle
    def open(self) -> None:
        """Open the serial port and start the read thread."""
        if self.serial_port is not None and self.serial_port.is_open:
            return

        self.serial_port = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self.logger.info(
            f"[IMU] opened serial: {self.config.port} @ {self.config.baudrate} bps"
        )

    def close(self) -> None:
        """Stop the read thread and close the serial port."""
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except Exception:
                pass
            self.serial_port = None

        self.logger.info("[IMU] closed")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    # --------------------------
    # Public API
    def get_latest_raw_vector(self) -> List[float]:
        """Return the latest raw 14-float vector (thread-safe copy)."""
        with self._lock:
            return self._latest_raw_vector.copy()

    def get_latest_packet(self) -> ImuPacket | None:
        """Return the latest decoded packet. Returns None if vector is invalid."""
        raw_vector = self.get_latest_raw_vector()
        return self.decode_vector(raw_vector)

    # --------------------------
    # Decoding (ported from split_packet)
    def decode_vector(self, data_list: List[float]) -> ImuPacket | None:
        """Decode a raw float vector into an ImuPacket."""
        if len(data_list) != self.config.expected_length:
            self.logger.warning(f"[IMU] wrong packet size: {len(data_list)}")
            return None

        quat_w, quat_x, quat_y, quat_z = data_list[0:4]
        gyro_x, gyro_y, gyro_z = data_list[4:7]
        acc_x, acc_y, acc_z = data_list[7:10]
        mag_x, mag_y, mag_z = data_list[10:13]
        time_ms = data_list[13]

        return ImuPacket(
            quat_w=quat_w, quat_x=quat_x, quat_y=quat_y, quat_z=quat_z,
            gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
            acc_x=acc_x, acc_y=acc_y, acc_z=acc_z,
            mag_x=mag_x, mag_y=mag_y, mag_z=mag_z,
            time_ms=float(time_ms),
        )

    # --------------------------
    # Internal read loop (ported from read_loop)
    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.serial_port is None:
                    time.sleep(0.05)
                    continue

                raw_line = self.serial_port.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                if not raw_line.startswith(self.config.start_char):
                    continue

                raw_payload = raw_line[1:]  # remove leading '*'
                string_fields = raw_payload.split(",")

                try:
                    float_values = list(map(float, string_fields))
                except ValueError:
                    continue

                if len(float_values) != self.config.expected_length:
                    continue

                with self._lock:
                    self._latest_raw_vector = float_values

            except Exception as exception:
                self.logger.warning(f"[IMU] read error: {exception}")
                time.sleep(self.config.read_sleep_sec_on_error)
