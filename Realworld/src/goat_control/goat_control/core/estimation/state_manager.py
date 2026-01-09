# goat_control/core/estimation/state_manager.py
from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Literal

from ..comm.motor_driver import MotorDriver
from .state_types import MotorStatesData, ImuState, RobotState
from .filters import FirstOrderLowPassFilter


# Motor current resolution used previously:
# motor_current_amp = iq_raw_lsb * (66.0 / 4096.0)
DEFAULT_MOTOR_CURRENT_AMP_PER_LSB = 66.0 / 4096.0

# Angle raw unit: 0.001 deg / LSB
DEFAULT_ANGLE_DEG_PER_LSB = 0.001

# Speed scaling: if already in deg/s, keep 1.0. If 0.01 deg/s per LSB, set 0.01.
DEFAULT_SPEED_DEG_PER_SEC_PER_LSB = 0.01


@dataclass
class StateManagerConfig:
    joint_names: List[str]

    # NEW: joint order <-> motor index mapping from YAML
    joint_indices: List[int] | None = None   # e.g. [0,1,2,3,4,5]
    wheel_indices: List[int] | None = None   # e.g. [6,7]

    # (기존 유지)
    knee_indices: List[int] = None

    motor_current_amp_per_lsb: float = DEFAULT_MOTOR_CURRENT_AMP_PER_LSB
    angle_deg_per_lsb: float = DEFAULT_ANGLE_DEG_PER_LSB
    speed_deg_per_sec_per_lsb: float = DEFAULT_SPEED_DEG_PER_SEC_PER_LSB

    joint_velocity_lpf_alpha: float | None = None
    joint_effort_like_lpf_alpha: float | None = None

    effort_output_mode: Literal["current_amp", "torque_nm"] = "torque_nm"

    motor_torque_constant_nm_per_amp: List[float] | None = None
    motor_gear_ratio: List[float] | None = None
    motor_direction: List[float] | None = None


def format_motor_states(
    motor_states_data: MotorStatesData,
    max_motor_count: int | None = 8,
    show_angles_in_degrees: bool = True,
    angle_deg_per_lsb: float = DEFAULT_ANGLE_DEG_PER_LSB,
) -> str:
    """Human-readable formatter (core-friendly replacement of motor_states_echo.py)."""

    motor_limit = max_motor_count if (max_motor_count and max_motor_count > 0) else None

    def clip(values: List):
        return values if (motor_limit is None) else values[:motor_limit]

    motor_temperature_c = clip(motor_states_data.motor_temperature_c)
    motor_phase_current_amp = clip(motor_states_data.motor_phase_current_amp)
    motor_speed_deg_per_sec = clip(motor_states_data.motor_speed_deg_per_sec)
    motor_encoder_count = clip(motor_states_data.motor_encoder_count)

    motor_single_turn_angle_raw_0p001deg = clip(motor_states_data.motor_single_turn_angle_raw_0p001deg)
    motor_multi_turn_angle_raw_0p001deg = clip(motor_states_data.motor_multi_turn_angle_raw_0p001deg)

    motor_error_flags = clip(motor_states_data.motor_error_flags)
    motor_operating_state = clip(motor_states_data.motor_operating_state)

    if show_angles_in_degrees:
        motor_single_turn_angle_deg = [
            raw_value * angle_deg_per_lsb for raw_value in motor_single_turn_angle_raw_0p001deg
        ]
        motor_multi_turn_angle_deg = [
            raw_value * angle_deg_per_lsb for raw_value in motor_multi_turn_angle_raw_0p001deg
        ]
    else:
        motor_single_turn_angle_deg = None
        motor_multi_turn_angle_deg = None

    timestamp_string = (
        f"{motor_states_data.timestamp_sec:.6f}"
        if motor_states_data.timestamp_sec is not None
        else "N/A"
    )

    lines: List[str] = []
    lines.append(f"\n=== MotorStates @ {timestamp_string} ===")
    lines.append(f"  motor_temperature_c        : {['{:.1f}'.format(v) for v in motor_temperature_c]} (degC)")
    lines.append(f"  motor_phase_current_amp    : {['{:.3f}'.format(v) for v in motor_phase_current_amp]} (A)")
    lines.append(f"  motor_speed_deg_per_sec    : {['{:.1f}'.format(v) for v in motor_speed_deg_per_sec]} (deg/s)")
    lines.append(f"  motor_encoder_count        : {motor_encoder_count}")

    if show_angles_in_degrees:
        lines.append(
            "  motor_single_turn_angle    : "
            f"{motor_single_turn_angle_raw_0p001deg} (raw, 0.001deg/LSB) -> "
            f"{['{:.2f}'.format(v) for v in motor_single_turn_angle_deg]} (deg)"
        )
        lines.append(
            "  motor_multi_turn_angle     : "
            f"{motor_multi_turn_angle_raw_0p001deg} (raw, 0.001deg/LSB) -> "
            f"{['{:.2f}'.format(v) for v in motor_multi_turn_angle_deg]} (deg)"
        )
    else:
        lines.append(f"  motor_single_turn_angle    : {motor_single_turn_angle_raw_0p001deg} (raw)")
        lines.append(f"  motor_multi_turn_angle     : {motor_multi_turn_angle_raw_0p001deg} (raw)")

    lines.append(f"  motor_error_flags          : {motor_error_flags}")
    lines.append(f"  motor_operating_state      : {motor_operating_state}")

    return "\n".join(lines)


class MotorStateCollector:
    """ROS-independent full-scan motor polling (ported from states_pub.py).

    It polls for each motor:
      - state2 (0x9C): temperature, iq(current), speed, encoder
      - state1 (0x9A): operating_state, error_flags
      - single/multi turn angle (0x94 / 0x92)
    """

    def __init__(
        self,
        motor_drivers: Sequence[MotorDriver],
        motor_current_amp_per_lsb: float = DEFAULT_MOTOR_CURRENT_AMP_PER_LSB,
        angle_deg_per_lsb: float = DEFAULT_ANGLE_DEG_PER_LSB,
        speed_deg_per_sec_per_lsb: float = DEFAULT_SPEED_DEG_PER_SEC_PER_LSB,
        single_turn_motor_indices: Optional[Sequence[int]] = None,
        multi_turn_motor_indices: Optional[Sequence[int]] = None,
    ):
        self.motor_drivers = list(motor_drivers)
        self.motor_count = len(self.motor_drivers)

        self.motor_current_amp_per_lsb = float(motor_current_amp_per_lsb)
        self.angle_deg_per_lsb = float(angle_deg_per_lsb)
        self.speed_deg_per_sec_per_lsb = float(speed_deg_per_sec_per_lsb)

        self.single_turn_motor_indices = set(single_turn_motor_indices or [])
        self.multi_turn_motor_indices = set(multi_turn_motor_indices or [])

        # Internal buffers (all length = motor_count)
        self.motor_temperature_c: List[float] = [float("nan")] * self.motor_count
        self.motor_phase_current_amp: List[float] = [float("nan")] * self.motor_count
        self.motor_speed_deg_per_sec: List[float] = [float("nan")] * self.motor_count
        self.motor_encoder_count: List[int] = [0] * self.motor_count

        self.motor_single_turn_angle_raw_0p001deg: List[int] = [0] * self.motor_count
        self.motor_multi_turn_angle_raw_0p001deg: List[int] = [0] * self.motor_count

        self.motor_error_flags: List[int] = [0] * self.motor_count
        self.motor_operating_state: List[int] = [0] * self.motor_count

    def poll_state2(self, motor_index: int, timeout: float = 0.05) -> None:
        response_message = self.motor_drivers[motor_index].read_state2(timeout=timeout)
        if response_message is None:
            return

        response_data = response_message.data

        # temperature: int8 [degC]
        self.motor_temperature_c[motor_index] = float(struct.unpack("<b", response_data[1:2])[0])

        # iq(current): int16 -> [A]
        motor_current_raw_lsb = struct.unpack("<h", response_data[2:4])[0]
        self.motor_phase_current_amp[motor_index] = float(motor_current_raw_lsb) * self.motor_current_amp_per_lsb

        # speed: int16 -> [deg/s]
        speed_raw_lsb = struct.unpack("<h", response_data[4:6])[0]
        self.motor_speed_deg_per_sec[motor_index] = float(speed_raw_lsb) * self.speed_deg_per_sec_per_lsb

        # encoder: uint16 [count]
        self.motor_encoder_count[motor_index] = int(struct.unpack("<H", response_data[6:8])[0])

    def poll_state1(self, motor_index: int, timeout: float = 0.05) -> None:
        response_message = self.motor_drivers[motor_index].read_state1(timeout=timeout)
        if response_message is None:
            return

        response_data = response_message.data
        self.motor_operating_state[motor_index] = int(response_data[6])
        self.motor_error_flags[motor_index] = int(response_data[7])

    def poll_single_or_multi_turn(self, motor_index: int, timeout: float = 0.25) -> None:
        motor_driver = self.motor_drivers[motor_index]

        def read_single_turn_angle() -> None:
            response_message = motor_driver.read_single_turn(timeout=timeout)
            if response_message is None:
                return
            response_data = response_message.data
            self.motor_single_turn_angle_raw_0p001deg[motor_index] = int.from_bytes(
                response_data[4:8],
                byteorder="little",
                signed=False
            )

        def read_multi_turn_angle() -> None:
            response_message = motor_driver.read_multi_turn(timeout=timeout)
            if response_message is None:
                return
            response_data = response_message.data

            # multi-turn uses 7 bytes signed (little-endian) in data[1:8]
            raw_7bytes = response_data[1:8]
            sign_extension_byte = b"\x00" if raw_7bytes[-1] < 0x80 else b"\xff"

            signed_int64 = int.from_bytes(
                raw_7bytes + sign_extension_byte,
                byteorder="little",
                signed=True
            )
            self.motor_multi_turn_angle_raw_0p001deg[motor_index] = int(signed_int64)

        # If both sets are empty, read both angles for all motors.
        if not self.single_turn_motor_indices and not self.multi_turn_motor_indices:
            read_single_turn_angle()
            read_multi_turn_angle()
            return

        if motor_index in self.single_turn_motor_indices:
            read_single_turn_angle()
        if motor_index in self.multi_turn_motor_indices:
            read_multi_turn_angle()

    def poll_all(self) -> MotorStatesData:
        for motor_index in range(self.motor_count):
            self.poll_state2(motor_index)
            self.poll_state1(motor_index)
            self.poll_single_or_multi_turn(motor_index)

        return MotorStatesData(
            motor_temperature_c=self.motor_temperature_c.copy(),
            motor_phase_current_amp=self.motor_phase_current_amp.copy(),
            motor_speed_deg_per_sec=self.motor_speed_deg_per_sec.copy(),
            motor_encoder_count=self.motor_encoder_count.copy(),
            motor_single_turn_angle_raw_0p001deg=self.motor_single_turn_angle_raw_0p001deg.copy(),
            motor_multi_turn_angle_raw_0p001deg=self.motor_multi_turn_angle_raw_0p001deg.copy(),
            motor_error_flags=self.motor_error_flags.copy(),
            motor_operating_state=self.motor_operating_state.copy(),
            timestamp_sec=time.time(),
        )


class StateManager:
    """Build RobotState (control/policy friendly) from MotorStatesData + ImuState."""

    def __init__(self, config: StateManagerConfig):
        self.config = config

        # --- (2) joint_names 순서대로 motor_states_data의 몇 번 모터를 쓸지 "YAML 기반"으로 결정
        # joint_indices + wheel_indices를 이어 붙여서 joint_names 길이와 맞춰줌
        mapped: List[int] = []
        if config.joint_indices:
            mapped.extend(list(config.joint_indices))
        if config.wheel_indices:
            mapped.extend(list(config.wheel_indices))

        self.motor_index_for_joint: List[int] | None = mapped if mapped else None

        self.joint_velocity_low_pass_filter = (
            FirstOrderLowPassFilter(alpha=config.joint_velocity_lpf_alpha)
            if config.joint_velocity_lpf_alpha is not None
            else None
        )
        self.joint_effort_like_low_pass_filter = (
            FirstOrderLowPassFilter(alpha=config.joint_effort_like_lpf_alpha)
            if config.joint_effort_like_lpf_alpha is not None
            else None
        )

    def _get_joint_torque_nm(self, motor_current_amp: float, motor_index: int) -> float:
        """Convert motor phase current [A] to joint torque [Nm] using config lists."""
        if self.config.motor_torque_constant_nm_per_amp is None:
            raise ValueError("motor_torque_constant_nm_per_amp is required for torque_nm output mode.")
        if self.config.motor_gear_ratio is None:
            raise ValueError("motor_gear_ratio is required for torque_nm output mode.")
        if self.config.motor_direction is None:
            raise ValueError("motor_direction is required for torque_nm output mode.")

        torque_constant_nm_per_amp = float(self.config.motor_torque_constant_nm_per_amp[motor_index])
        # gear_ratio = float(self.config.motor_gear_ratio[motor_index])
        # direction = float(self.config.motor_direction[motor_index])  # +1/-1

        motor_shaft_torque_nm = motor_current_amp * torque_constant_nm_per_amp
        joint_torque_nm = motor_shaft_torque_nm
        return joint_torque_nm

    def build_robot_state(
        self,
        motor_states_data: MotorStatesData,
        imu_state: Optional[ImuState] = None,
    ) -> RobotState:
        motor_count = len(motor_states_data.motor_temperature_c)
        joint_count = len(self.config.joint_names)

        # --- (3) 매핑 검증 (joint_indices+wheel_indices 사용 시)
        if self.motor_index_for_joint is not None:
            if len(self.motor_index_for_joint) != joint_count:
                raise ValueError(
                    f"joint_indices+wheel_indices length({len(self.motor_index_for_joint)}) "
                    f"must match joint_names length({joint_count})."
                )
        else:
            # 매핑이 없으면 joint_count==motor_count라고 가정하고 0..joint_count-1 사용
            if joint_count != motor_count:
                raise ValueError(
                    f"joint_names length({joint_count}) != motor_count({motor_count}). "
                    "Provide joint_indices/wheel_indices in YAML."
                )

        # --- (4) gear/direction list 길이 검증
        if self.config.motor_gear_ratio is None or len(self.config.motor_gear_ratio) != motor_count:
            raise ValueError("motor_gear_ratio must be a list with length == motor_count.")
        if self.config.motor_direction is None or len(self.config.motor_direction) != motor_count:
            raise ValueError("motor_direction must be a list with length == motor_count.")

        if self.config.effort_output_mode == "torque_nm":
            if self.config.motor_torque_constant_nm_per_amp is None or len(self.config.motor_torque_constant_nm_per_amp) != motor_count:
                raise ValueError("motor_torque_constant_nm_per_amp must be a list with length == motor_count.")

        joint_position_rad: List[float] = [0.0] * joint_count
        joint_velocity_rad_per_sec: List[float] = [0.0] * joint_count
        joint_effort_like: List[float] = [0.0] * joint_count  # current[A] or torque[Nm]

        # --- (5) 핵심 변경:
        # "knee만 따로" 같은 하드코딩 제거
        # 모든 관절에 대해: (motor angle/speed) * motor_gear_ratio * motor_direction 적용
        for joint_i in range(joint_count):
            motor_i = self.motor_index_for_joint[joint_i] if self.motor_index_for_joint is not None else joint_i

            gear = float(self.config.motor_gear_ratio[motor_i])
            direction = float(self.config.motor_direction[motor_i])

            # Position: multi-turn 우선, (필요하면 single-turn로 fallback)
            # NOTE: 0도도 정상값일 수 있어서 "!=0" 센티넬은 위험함.
            # 일단은 multi-turn을 기본으로 쓰고, 둘 다 0이면 0으로 둠.
            raw_multi = motor_states_data.motor_multi_turn_angle_raw_0p001deg[motor_i]
            raw_single = motor_states_data.motor_single_turn_angle_raw_0p001deg[motor_i]

            if raw_multi != 0:
                motor_angle_deg = raw_multi * self.config.angle_deg_per_lsb
            elif raw_single != 0:
                motor_angle_deg = raw_single * self.config.angle_deg_per_lsb
            else:
                motor_angle_deg = 0.0

            joint_angle_deg = motor_angle_deg * gear * direction
            joint_position_rad[joint_i] = joint_angle_deg * math.pi / 180.0

            # Velocity: deg/s -> rad/s, gear+direction 적용
            motor_speed_deg_s = motor_states_data.motor_speed_deg_per_sec[motor_i]
            joint_speed_deg_s = motor_speed_deg_s * gear * direction
            joint_velocity_rad_per_sec[joint_i] = joint_speed_deg_s * math.pi / 180.0

            # Effort-like
            motor_current_amp = motor_states_data.motor_phase_current_amp[motor_i]
            if self.config.effort_output_mode == "torque_nm":
                joint_effort_like[joint_i] = self._get_joint_torque_nm(motor_current_amp, motor_i)
            else:
                joint_effort_like[joint_i] = motor_current_amp

        # Optional filtering
        if self.joint_velocity_low_pass_filter is not None:
            joint_velocity_rad_per_sec = self.joint_velocity_low_pass_filter.apply(joint_velocity_rad_per_sec)  # type: ignore[assignment]
        if self.joint_effort_like_low_pass_filter is not None:
            joint_effort_like = self.joint_effort_like_low_pass_filter.apply(joint_effort_like)  # type: ignore[assignment]

        return RobotState(
            joint_names=self.config.joint_names,
            joint_position_rad=list(joint_position_rad),
            joint_velocity_rad_per_sec=list(joint_velocity_rad_per_sec),
            joint_effort_like=list(joint_effort_like),
            motor_temperature_c=motor_states_data.motor_temperature_c,
            motor_error_flags=motor_states_data.motor_error_flags,
            motor_operating_state=motor_states_data.motor_operating_state,
            imu_state=imu_state,
            timestamp_sec=motor_states_data.timestamp_sec,
        )
