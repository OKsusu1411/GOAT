# goat_control/core/estimation/state_manager.py
from __future__ import annotations

import math
import struct
import time
import numpy as np
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Sequence

from goat_control.utils.motor.motor_driver import MotorDriver
from goat_control.utils.motor.filters import FirstOrderLowPassFilter


# Motor current resolution used previously:
# motor_current_amp = iq_raw_lsb * (66.0 / 4096.0)
DEFAULT_MOTOR_CURRENT_AMP_PER_LSB = 66.0 / 4096.0

# Angle raw unit: 0.001 deg / LSB
DEFAULT_ANGLE_DEG_PER_LSB = 0.001

# Speed scaling: if already in deg/s, keep 1.0. If 0.01 deg/s per LSB, set 0.01.
DEFAULT_SPEED_DEG_PER_SEC_PER_LSB = 0.01

@dataclass
class MotorStatesData:
    """Motor state snapshot (ROS-independent equivalent of your MotorStates message).

    Naming rules:
      - Use explicit physical meaning whenever possible.
      - Include units in the field name when it helps clarity.
    """
    joint_names: List[str]

    joint_position_rad: List[float]                    # [rad]
    joint_velocity_rad_per_sec: List[float]            # [rad/s]
    joint_effort_like: List[float]                     # [Nm]

    motor_temperature_c: List[float]                   # [degC]
    motor_error_flags: List[int]                       # bitfield / error codes (device-defined)
    motor_operating_state: List[int]                   # motor mode/state (device-defined)

    timestamp_sec: Optional[float] = None

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


class MotorManager:
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
        cfg:dict = None
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

        # YAML cfg
        self.cfg = cfg
        # joint_velocity_lpf_alpha = self.cfg["joint_velocity_lpf_alpha"]
        # joint_effort_like_lpf_alpha = self.cfg["joint_effort_like_lpf_alpha"]
        self.motor_gear_ratio = self.cfg["motor_gear_ratio"]
        self.motor_direction = self.cfg["motor_direction"]
        self.motor_torque_constant_nm_per_amp = self.cfg["motor_torque_constant_nm_per_amp"]
        self.angle_deg_per_lsb = self.cfg["angle_deg_per_lsb"]
        self.joint_names = self.cfg["joint_names"]
        self.num_joints = len(self.joint_names)
        joint_offsets = self.cfg["joint_offsets"]
        self.joint_offsets = np.asarray(joint_offsets, dtype=float).flatten()
        lpf_alpha = self.cfg["torque_lpf_alpha_per_joint"]
        self.lpf_alpha = np.asarray(lpf_alpha, dtype=float).flatten()
        hw_max_torque_per_joint = self.cfg["hw_max_torque_per_joint"]
        sw_max_torque_per_joint = self.cfg["sw_max_torque_per_joint"]
        self.hw_max_torque_per_joint = np.asarray(hw_max_torque_per_joint, dtype=float).flatten()
        self.sw_max_torque_per_joint = np.asarray(sw_max_torque_per_joint, dtype=float).flatten()

        mapped: List[int] = []
        if self.cfg["joint_indices"]:
            mapped.extend(list(self.cfg["joint_indices"]))
        if self.cfg["wheel_indices"]:
            mapped.extend(list(self.cfg["wheel_indices"]))

        self.motor_index_for_joint: List[int] | None = mapped if mapped else None
        
        self.prev_torque = np.zeros(self.num_joints, dtype=float)

        # self.joint_velocity_low_pass_filter = (
        #     FirstOrderLowPassFilter(alpha=joint_velocity_lpf_alpha)
        #     if joint_velocity_lpf_alpha is not None
        #     else None
        # )

        # self.joint_effort_like_low_pass_filter = (
        #     FirstOrderLowPassFilter(alpha=joint_effort_like_lpf_alpha)
        #     if joint_effort_like_lpf_alpha is not None
        #     else None
        # )

    def torque_clipping(self, torque_cmd:np.ndarray) -> np.ndarray:
        clipped_torque = np.clip(torque_cmd, -self.hw_max_torque_per_joint, self.hw_max_torque_per_joint) # [4.5 and 2.5]
        clipped_torque = np.clip(clipped_torque, -self.sw_max_torque_per_joint, self.sw_max_torque_per_joint) # [Arbitrary]
        
        return clipped_torque
    
    def torque_lpf(self, torque_cmd:np.ndarray) -> np.ndarray:
        filtered = self.lpf_alpha * torque_cmd + (1.0 - self.lpf_alpha) * self.prev_torque
        self.prev_torque[:] = filtered

        return filtered

    def torque_to_current(self, torque_cmd:np.ndarray) -> np.ndarray:
        # Convert torque into current
        torque_constant = np.asarray(self.motor_torque_constant_nm_per_amp, dtype=float)
        gear_ratio = np.asarray(self.motor_gear_ratio, dtype=float)
        direction = np.asarray(self.motor_direction, dtype=float)

        denominator = gear_ratio * torque_constant  # gear ratio
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        current_command_amp = torque_cmd / denominator

        zero_mask = np.abs(direction * gear_ratio * torque_constant) < 1e-12
        current_command_amp = np.where(zero_mask, 0.0, current_command_amp)

        return current_command_amp

    def poll_state1(self, motor_index: int, timeout: float = 0.05) -> None:
        response_message = self.motor_drivers[motor_index].read_state1(timeout=timeout)
        if response_message is None:
            return

        response_data = response_message.data
        self.motor_operating_state[motor_index] = int(response_data[6])
        self.motor_error_flags[motor_index] = int(response_data[7])

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
    
    def _get_joint_torque_nm(self, motor_current_amp: float, motor_index: int) -> float:
        """Convert motor phase current [A] to joint torque [Nm] using config lists."""
        if self.motor_torque_constant_nm_per_amp is None:
            raise ValueError("motor_torque_constant_nm_per_amp is required for torque_nm output mode.")
        if self.motor_gear_ratio is None:
            raise ValueError("motor_gear_ratio is required for torque_nm output mode.")
        if self.motor_direction is None:
            raise ValueError("motor_direction is required for torque_nm output mode.")

        torque_constant_nm_per_amp = float(self.motor_torque_constant_nm_per_amp[motor_index])
        motor_shaft_torque_nm = motor_current_amp * torque_constant_nm_per_amp
        joint_torque_nm = motor_shaft_torque_nm
        return joint_torque_nm

    # def decode_motor_encoder(self) -> MotorStatesData:
    #     # for motor_index in range(self.motor_count):
    #     # for motor_index in [0]:
    #     #     self.poll_state2(motor_index)
    #     #     self.poll_state1(motor_index)
    #     #     self.poll_single_or_multi_turn(motor_index)

    #     def fetch_motor_data(motor_index: int):
    #         self.poll_state2(motor_index)
    #         self.poll_state1(motor_index)
    #         self.poll_single_or_multi_turn(motor_index)

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=self.motor_count) as executor:
    #         executor.map(fetch_motor_data, range(self.motor_count))

    #     motor_count = len(self.motor_temperature_c)
    #     joint_count = len(self.joint_names)

    #     # Index mapping validation
    #     if self.motor_index_for_joint is not None:
    #         if len(self.motor_index_for_joint) != joint_count:
    #             raise ValueError(
    #                 f"joint_indices+wheel_indices length({len(self.motor_index_for_joint)}) "
    #                 f"must match joint_names length({joint_count})."
    #             )
    #     else:
    #         if joint_count != motor_count:
    #             raise ValueError(
    #                 f"joint_names length({joint_count}) != motor_count({motor_count}). "
    #                 "Provide joint_indices/wheel_indices in YAML."
    #             )
        
    #     # Joint variables
    #     joint_position_rad: List[float] = [0.0] * joint_count
    #     joint_velocity_rad_per_sec: List[float] = [0.0] * joint_count
    #     joint_effort_like: List[float] = [0.0] * joint_count  # current[A] or torque[Nm]
        
    #     # gear, direction index validation
    #     if self.motor_gear_ratio is None or len(self.motor_gear_ratio) != motor_count:
    #         raise ValueError("motor_gear_ratio must be a list with length == motor_count.")
    #     if self.motor_direction is None or len(self.motor_direction) != motor_count:
    #         raise ValueError("motor_direction must be a list with length == motor_count.")

    #     if self.motor_torque_constant_nm_per_amp is None or len(self.motor_torque_constant_nm_per_amp) != motor_count:
    #         raise ValueError("motor_torque_constant_nm_per_amp must be a list with length == motor_count.")

    #     # Main state computation logic
    #     for joint_i in range(joint_count):
    #         motor_i = self.motor_index_for_joint[joint_i] if self.motor_index_for_joint is not None else joint_i

    #         gear = float(self.motor_gear_ratio[motor_i])
    #         direction = float(self.motor_direction[motor_i])

    #         raw_multi = self.motor_multi_turn_angle_raw_0p001deg[motor_i]
    #         raw_single = self.motor_single_turn_angle_raw_0p001deg[motor_i]

    #         # Encoder fail safe logic
    #         if raw_multi != 0:
    #             motor_angle_deg = raw_multi * self.angle_deg_per_lsb
    #         elif raw_single != 0:
    #             motor_angle_deg = raw_single * self.angle_deg_per_lsb
    #         else:
    #             motor_angle_deg = 0.0

    #         # Convert motor position into joint position
    #         joint_angle_deg = motor_angle_deg * gear * direction
    #         joint_position_rad[joint_i] = joint_angle_deg * math.pi / 180.0                 # degree to radian

    #         # Convert motor velocity into joint velocity
    #         motor_speed_deg_s = self.motor_speed_deg_per_sec[motor_i]
    #         joint_speed_deg_s = motor_speed_deg_s * gear * direction
    #         joint_velocity_rad_per_sec[joint_i] = joint_speed_deg_s * math.pi / 180.0       # degree to radian

    #         # Effort-like
    #         motor_current_amp = self.motor_phase_current_amp[motor_i]
    #         joint_effort_like[joint_i] = self._get_joint_torque_nm(motor_current_amp, motor_i)

    #     # Apply joint position offset
    #     joint_position_rad = np.array(joint_position_rad) - np.asarray(self.joint_offsets, dtype=float)
        
    #     return MotorStatesData(
    #         joint_names=self.joint_names,
    #         joint_position_rad=list(joint_position_rad).copy(),
    #         joint_velocity_rad_per_sec=list(joint_velocity_rad_per_sec).copy(),
    #         joint_effort_like=list(joint_effort_like).copy(),
    #         motor_temperature_c=self.motor_temperature_c.copy(),
    #         motor_error_flags=self.motor_error_flags.copy(),
    #         motor_operating_state=self.motor_operating_state.copy(),
    #         timestamp_sec=time.time()
    #     )

    # =========================================================================
    # [공통 수학 연산 로직] 원시 데이터(Raw data)를 MotorStatesData로 패키징
    # =========================================================================
    def _package_motor_states(self) -> MotorStatesData:
        """내부 버퍼의 Raw 데이터를 이용해 물리량으로 변환하고 MotorStatesData를 생성합니다."""
        joint_count = len(self.joint_names)
        
        joint_position_rad: List[float] = [0.0] * joint_count
        joint_velocity_rad_per_sec: List[float] = [0.0] * joint_count
        joint_effort_like: List[float] = [0.0] * joint_count
        
        for joint_i in range(joint_count):
            motor_i = self.motor_index_for_joint[joint_i] if self.motor_index_for_joint is not None else joint_i
            gear = float(self.motor_gear_ratio[motor_i])
            direction = float(self.motor_direction[motor_i])

            raw_multi = self.motor_multi_turn_angle_raw_0p001deg[motor_i]
            raw_single = self.motor_single_turn_angle_raw_0p001deg[motor_i]

            if raw_multi != 0: motor_angle_deg = raw_multi * self.angle_deg_per_lsb
            elif raw_single != 0: motor_angle_deg = raw_single * self.angle_deg_per_lsb
            else: motor_angle_deg = 0.0

            joint_angle_deg = motor_angle_deg * gear * direction
            joint_position_rad[joint_i] = joint_angle_deg * math.pi / 180.0

            motor_speed_deg_s = self.motor_speed_deg_per_sec[motor_i]
            joint_speed_deg_s = motor_speed_deg_s * gear * direction
            joint_velocity_rad_per_sec[joint_i] = joint_speed_deg_s * math.pi / 180.0

            motor_current_amp = self.motor_phase_current_amp[motor_i]
            joint_effort_like[joint_i] = self._get_joint_torque_nm(motor_current_amp, motor_i)

        joint_position_rad = np.array(joint_position_rad) - np.asarray(self.joint_offsets, dtype=float)
        
        return MotorStatesData(
            joint_names=self.joint_names,
            joint_position_rad=list(joint_position_rad).copy(),
            joint_velocity_rad_per_sec=list(joint_velocity_rad_per_sec).copy(),
            joint_effort_like=list(joint_effort_like).copy(),
            motor_temperature_c=self.motor_temperature_c.copy(),
            motor_error_flags=self.motor_error_flags.copy(),
            motor_operating_state=self.motor_operating_state.copy(),
            timestamp_sec=time.time()
        )

    # =========================================================================
    # [Read Only] 기존과 동일한 읽기 함수 (이제 공통 로직을 재사용합니다)
    # =========================================================================
    def decode_motor_encoder(self, perform_slow_poll: bool = True) -> MotorStatesData:
        """모터의 상태만 읽어옵니다. (주로 초기화나 대기 상태일 때 사용)"""
        def fetch_motor_data(motor_index: int):
            self.poll_state2(motor_index)
            if perform_slow_poll:
                self.poll_state1(motor_index)
                self.poll_single_or_multi_turn(motor_index)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.motor_count) as executor:
            executor.map(fetch_motor_data, range(self.motor_count))
            
        # 데이터를 읽어온 후 패키징하여 반환합니다.
        return self._package_motor_states()

    # =========================================================================
    # [Write + Read] 토크 명령과 상태 읽기를 동시에 처리하는 함수
    # =========================================================================
    def write_torques_and_read_states(self, current_cmd_amp: Sequence[float], timeout: float = 0.05, perform_slow_poll: bool = False) -> MotorStatesData:
        """
        모든 모터에 토크 명령을 내림과 동시에 최신 상태를 읽어오고, 
        그 결과를 MotorStatesData 형식으로 반환합니다.
        """
        def control_and_read(motor_index: int):
            command_amp = float(current_cmd_amp[motor_index])
            # 명령 송신 + 상태(State2) 즉시 업데이트
            # self.send_torque_and_update_state(motor_index, command_amp, timeout=timeout)
            response_message = self.motor_drivers[motor_index].torque_mode_amp(command_amp, timeout=timeout)
        
            if response_message is None:
                return
                
            response_data = response_message.data       # TODO: 변경
            
            # 매뉴얼에 명시된 0xA1 응답 데이터(State2 포맷) 파싱[cite: 1]
            self.motor_temperature_c[motor_index] = float(struct.unpack("<b", response_data[1:2])[0])
            
            motor_current_raw_lsb = struct.unpack("<h", response_data[2:4])[0]
            self.motor_phase_current_amp[motor_index] = float(motor_current_raw_lsb) * self.motor_current_amp_per_lsb
            
            speed_raw_lsb = struct.unpack("<h", response_data[4:6])[0]
            self.motor_speed_deg_per_sec[motor_index] = float(speed_raw_lsb) * self.speed_deg_per_sec_per_lsb
            
            self.motor_encoder_count[motor_index] = int(struct.unpack("<H", response_data[6:8])[0])
            # 에러 플래그 등은 매번 읽을 필요가 없으므로 옵션으로 처리합니다.
            if perform_slow_poll:
                self.poll_state1(motor_index)

        # 8개 모터에 동시에 명령 송신 및 상태 수신
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.motor_count) as executor:
            executor.map(control_and_read, range(self.motor_count))
            
        # 최신화된 버퍼를 이용해 수학 연산을 수행하고 결과를 반환합니다.
        return self._package_motor_states()