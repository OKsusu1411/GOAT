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
# [DEPRECATED] LPF logic was moved into MotorManager.torque_lpf() — this generic filter is no longer wired into the pipeline.
# from goat_control.utils.motor.filters import FirstOrderLowPassFilter


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
        # [DEPRECATED] Velocity / effort LPFs were removed in favor of a single
        # torque-side LPF (see torque_lpf_alpha_per_joint below).
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

        # ------------------------------------------------------------------
        # Encoder integration (Option B): convert the per-tick uint16 encoder
        # field of the 0xA1 reply into a multi-turn motor angle, anchored to
        # the absolute multi-turn read taken once at init.
        # ------------------------------------------------------------------
        self.motor_encoder_counts_per_rev = int(self.cfg.get("motor_encoder_counts_per_rev", 16384))
        # Per-motor anchor state. `None` until _seed_position_anchor() runs.
        self.motor_anchor_motor_angle_deg: List[Optional[float]] = [None] * self.motor_count
        self.motor_anchor_encoder_count:   List[Optional[int]]   = [None] * self.motor_count
        self.motor_prev_encoder_count:     List[Optional[int]]   = [None] * self.motor_count
        self.motor_encoder_wrap_count:     List[int]             = [0]    * self.motor_count

        # Persistent thread pool reused across every tick. Recreating an
        # 8-thread pool at 200 Hz was costing ~800 thread spawns/sec for no
        # parallelism benefit (per-bus txrx_lock already serializes the bus).
        self._io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.motor_count,
            thread_name_prefix="motor_io",
        )

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
        """Read 0x9A error flags. Auto-picks blocking txrx (init, no reader)
        vs fire-and-forget + cache read (after reader thread starts)."""
        driver = self.motor_drivers[motor_index]
        if driver.can_interface.is_reader_running():
            # Reader thread mode: fire request, read prior reply from cache.
            # 1 Hz call rate means worst-case staleness is ~1 s — fine for
            # safety telemetry. First call returns None (no cached reply yet).
            driver.send_state1_request()
            response_message = driver.latest_state1()
        else:
            # Init mode: blocking round-trip is safe; reader not yet running.
            response_message = driver.read_state1(timeout=timeout)

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
    # Encoder integration helpers (Option B)
    # =========================================================================
    def _seed_position_anchor(self) -> None:
        """Capture absolute motor angle + matching encoder count as the
        per-motor anchor. Called once after the init multi-turn poll."""
        for motor_index in range(self.motor_count):
            raw_multi = self.motor_multi_turn_angle_raw_0p001deg[motor_index]
            raw_single = self.motor_single_turn_angle_raw_0p001deg[motor_index]
            if raw_multi != 0:
                anchor_motor_angle_deg = raw_multi * self.angle_deg_per_lsb
            elif raw_single != 0:
                anchor_motor_angle_deg = raw_single * self.angle_deg_per_lsb
            else:
                anchor_motor_angle_deg = 0.0

            self.motor_anchor_motor_angle_deg[motor_index] = anchor_motor_angle_deg
            self.motor_anchor_encoder_count[motor_index]   = int(self.motor_encoder_count[motor_index])
            self.motor_prev_encoder_count[motor_index]     = int(self.motor_encoder_count[motor_index])
            self.motor_encoder_wrap_count[motor_index]     = 0

    def _update_motor_angle_from_encoder(self, motor_index: int) -> None:
        """Integrate the latest uint16 encoder count into a multi-turn motor
        angle and write it back into motor_multi_turn_angle_raw_0p001deg so
        _package_motor_states picks it up without further changes."""
        anchor_angle_deg = self.motor_anchor_motor_angle_deg[motor_index]
        if anchor_angle_deg is None:
            return  # anchor not seeded yet (init not done) — leave buffer alone

        counts_per_rev = self.motor_encoder_counts_per_rev
        half_range = counts_per_rev // 2

        current_count = int(self.motor_encoder_count[motor_index])
        prev_count = self.motor_prev_encoder_count[motor_index]
        delta_count = current_count - prev_count

        if motor_index == 0:
            print(f"Motor {motor_index}: current_count={current_count}\r")

        # Wrap detection — a single-tick step larger than half range means
        # the uint encoder field wrapped, not that the motor moved that fast.
        if delta_count > half_range:
            self.motor_encoder_wrap_count[motor_index] -= 1
        elif delta_count < -half_range:
            self.motor_encoder_wrap_count[motor_index] += 1
        self.motor_prev_encoder_count[motor_index] = current_count

        anchor_count = self.motor_anchor_encoder_count[motor_index]
        total_count_delta = (
            (current_count - anchor_count)
            + self.motor_encoder_wrap_count[motor_index] * counts_per_rev
        )
        motor_angle_deg = anchor_angle_deg + total_count_delta * (360.0 / counts_per_rev)

        # Write back in the same 0.001 deg/LSB units _package_motor_states reads.
        self.motor_multi_turn_angle_raw_0p001deg[motor_index] = int(
            round(motor_angle_deg / self.angle_deg_per_lsb)
        )

    # =========================================================================
    # [Read Only] 기존과 동일한 읽기 함수 (이제 공통 로직을 재사용합니다)
    # =========================================================================
    def decode_motor_encoder(self, perform_slow_poll: bool = True) -> MotorStatesData:
        """모터의 상태만 읽어옵니다. (주로 초기화나 대기 상태일 때 사용)"""
        def fetch_motor_data(motor_index: int):
            # Order matters: 0x92 multi-turn (anchor_motor_angle_deg) and
            # 0x9C state2 (anchor_encoder_count) must be adjacent in time
            # so the anchor pair refers to nearly the same physical motor
            # position. Slow-poll first, then multi-turn, then state2 last.
            if perform_slow_poll:
                self.poll_state1(motor_index)                # 0x9A error flags
                self.poll_single_or_multi_turn(motor_index)  # 0x92 / 0x94
            self.poll_state2(motor_index)                    # 0x9C — pairs with 0x92

        # Submit on the persistent pool and drain via f.result() so any
        # exceptions surface instead of being silently swallowed by a
        # discarded executor.map() iterator.
        futures = [self._io_pool.submit(fetch_motor_data, i) for i in range(self.motor_count)]
        for f in futures:
            f.result()

        # Seed (or re-seed) the encoder anchor exactly when we have a fresh
        # multi-turn read available. Safe to call multiple times — it always
        # re-aligns to the latest absolute angle.
        if perform_slow_poll:
            self._seed_position_anchor()

        # 데이터를 읽어온 후 패키징하여 반환합니다.
        return self._package_motor_states()

    # =========================================================================
    # [Write + Read] 토크 명령과 상태 읽기를 동시에 처리하는 함수
    # =========================================================================
    def write_torques_and_read_states(
        self,
        current_cmd_amp: Sequence[float],
        timeout: float = 0.003,
        perform_slow_poll: bool = False,
    ) -> MotorStatesData:
        """Synchronous fire-all-then-wait-all torque + state2 pass.

        Phase 1 — clear each motor's 0xA1 reply event, then send its torque
                  command. TXs are non-blocking (kernel ring); both buses
                  run in parallel by virtue of separate CanInterface
                  instances.
        Phase 2 — for each motor, wait on its arrival event until the shared
                  deadline. On timeout, inject NaN so the kill switch fires
                  instead of letting stale data drive the controller.

        Δt between successive ticks is now constant (the control timer
        period), so the encoder wrap detector in
        _update_motor_angle_from_encoder works as originally designed
        without any speed-sign disambiguation.

        `perform_slow_poll` is kept for API compatibility but ignored —
        state1 polling lives on a separate 1 Hz timer.
        """
        # Phase 1 — arm + fire. Sequential is fine: bus.send() goes into the
        # kernel TX ring without blocking on the wire.
        t_submit = time.perf_counter()                                           # [timing]
        for motor_index, amp in enumerate(current_cmd_amp):
            driver = self.motor_drivers[motor_index]
            driver.clear_state2_event()
            driver.send_torque_only(float(amp))
        t_fired = time.perf_counter()                                            # [timing]

        # Phase 2 — bounded wait, one shared deadline so total RX time is
        # bounded by `timeout` rather than 8 × per-motor timeout.
        deadline_monotonic = time.monotonic() + float(timeout)
        for motor_index in range(self.motor_count):
            driver = self.motor_drivers[motor_index]
            response_message = driver.await_state2(deadline_monotonic)
            if response_message is None:
                # Motor silent this tick. Mark its slot NaN so
                # controller_node._sensor_data_has_nan() trips the kill
                # switch — far safer than re-using stale state for a
                # balance loop.
                self.motor_speed_deg_per_sec[motor_index] = float("nan")
                self.motor_phase_current_amp[motor_index] = float("nan")
                continue

            response_data = response_message.data

            # 0xA1 reply has the same byte layout as 0x9C state2.
            self.motor_temperature_c[motor_index] = float(struct.unpack("<b", response_data[1:2])[0])

            motor_current_raw_lsb = struct.unpack("<h", response_data[2:4])[0]
            self.motor_phase_current_amp[motor_index] = float(motor_current_raw_lsb) * self.motor_current_amp_per_lsb

            speed_raw_lsb = struct.unpack("<h", response_data[4:6])[0]
            self.motor_speed_deg_per_sec[motor_index] = float(speed_raw_lsb) * self.speed_deg_per_sec_per_lsb

            self.motor_encoder_count[motor_index] = int(struct.unpack("<H", response_data[6:8])[0])

            self._update_motor_angle_from_encoder(motor_index)
        t_done = time.perf_counter()                                             # [timing]

        # Surface timings (read by controller_node timing log).
        self._last_tx_submit_ms = (t_fired - t_submit) * 1e3
        self._last_tx_wait_ms = (t_done - t_fired) * 1e3

        return self._package_motor_states()