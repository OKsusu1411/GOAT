# goat_control/core/estimation/state_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MotorStatesData:
    """Motor state snapshot (ROS-independent equivalent of your MotorStates message).

    Naming rules:
      - Use explicit physical meaning whenever possible.
      - Include units in the field name when it helps clarity.
    """
    motor_temperature_c: List[float]                   # [degC]
    motor_phase_current_amp: List[float]               # [A] (motor IQ current)
    motor_speed_deg_per_sec: List[float]               # [deg/s]
    motor_encoder_count: List[int]                     # [count]

    motor_single_turn_angle_raw_0p001deg: List[int]    # [0.001 deg/LSB] unsigned raw
    motor_multi_turn_angle_raw_0p001deg: List[int]     # [0.001 deg/LSB] signed raw (int64 reconstructed)

    motor_error_flags: List[int]                       # bitfield / error codes (device-defined)
    motor_operating_state: List[int]                   # motor mode/state (device-defined)

    timestamp_sec: Optional[float] = None


@dataclass
class ImuState:
    """IMU state (ROS-independent).

    Units:
      - gyro: rad/s or deg/s depends on your sensor output. Keep it consistent in your pipeline.
      - accel: m/s^2 or g depends on your sensor output.
      - magnetometer: uT or arbitrary, depends on your sensor output.

    If you later standardize units (recommended), rename fields accordingly.
    """

    # Quaternion (unitless)
    orientation_quat_w: float
    orientation_quat_x: float
    orientation_quat_y: float
    orientation_quat_z: float

    # Angular velocity
    angular_velocity_x: float
    angular_velocity_y: float
    angular_velocity_z: float

    # Linear acceleration
    linear_acceleration_x: float
    linear_acceleration_y: float
    linear_acceleration_z: float

    # Magnetic field
    magnetic_field_x: float
    magnetic_field_y: float
    magnetic_field_z: float

    # Sensor timestamp from IMU packet
    sensor_time_ms: float


@dataclass
class RobotState:
    """Robot state in SI-like units for control/policy.

    Notes:
      - joint_effort_like can be motor current(A) or torque(Nm), depending on your choice.
      - Keep a clear convention and document it here.
    """
    joint_names: List[str]

    joint_position_rad: List[float]                    # [rad]
    joint_velocity_rad_per_sec: List[float]            # [rad/s]
    joint_effort_like: List[float]                     # [A] or [Nm] depending on your convention

    motor_temperature_c: List[float]                   # [degC]
    motor_error_flags: List[int]
    motor_operating_state: List[int]

    imu_state: Optional[ImuState] = None
    timestamp_sec: Optional[float] = None
