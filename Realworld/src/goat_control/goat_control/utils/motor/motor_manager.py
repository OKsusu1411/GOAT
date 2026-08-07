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

    def __init__(self,
                 motor_drivers: Sequence[MotorDriver],
                 cfg:dict = None):
        self.cfg = cfg
        self.motor_drivers = list(motor_drivers)
        self.motor_count = len(self.motor_drivers)

        # ------------------------------------------------------------------
        # Scalar configuration
        # ------------------------------------------------------------------
        self.angle_deg_per_lsb = float(self.cfg["angle_deg_per_lsb"])
        self.motor_current_amp_per_lsb = float(self.cfg["motor_current_amp_per_lsb"])
        self.speed_deg_per_sec_per_lsb = float(self.cfg["speed_deg_per_sec_per_lsb"])
        self.max_current_lsb = int(self.cfg["max_current_lsb"])
        self.motor_encoder_counts_per_rev = int(self.cfg.get("motor_encoder_counts_per_rev", 65536)) # Encoder integration : convert the per-tick uint16 encoder

        # ------------------------------------------------------------------
        # Names and mapping
        # ------------------------------------------------------------------
        mapped: List[int] = []
        self.joint_names = self.cfg["joint_names"]
        self.num_joints = len(self.joint_names)

        if self.cfg["joint_indices"]: mapped.extend(list(self.cfg["joint_indices"]))
        if self.cfg["wheel_indices"]: mapped.extend(list(self.cfg["wheel_indices"]))

        self.motor_index_for_joint: List[int] | None = mapped if mapped else None

        # ------------------------------------------------------------------
        # Numeric vector configuration
        # ------------------------------------------------------------------
        self.motor_torque_constant_nm_per_amp = np.asarray(self.cfg["motor_torque_constant_nm_per_amp"], dtype=np.float64).flatten()
        self.max_torque_per_joint = np.asarray(self.cfg["max_torque_per_joint"], dtype=np.float64).flatten()
        self.motor_gear_ratio = np.asarray(self.cfg["motor_gear_ratio"], dtype=np.float64).flatten()
        self.motor_direction = np.asarray(self.cfg["motor_direction"], dtype=np.float64).flatten()
        self.joint_offsets = np.asarray(self.cfg["joint_offsets"], dtype=np.float64).flatten()
        self.torque_to_current_denominator = self.motor_gear_ratio * self.motor_torque_constant_nm_per_amp
        if np.any(np.abs(self.torque_to_current_denominator) < 1e-12):
            raise ValueError("gear ratio x torque constant must not be zero.")

        # ------------------------------------------------------------------
        # Motor states via CAN
        # ------------------------------------------------------------------

        self.motor_temperature_c: List[float] = [float("nan")] * self.motor_count
        self.motor_phase_current_amp: List[float] = [float("nan")] * self.motor_count
        self.motor_speed_deg_per_sec: List[float] = [float("nan")] * self.motor_count
        self.motor_encoder_count: List[int] = [0] * self.motor_count
        self.motor_pi_gain: List[List[float]] = [[0, 0]] * self.motor_count

        self.motor_single_turn_angle_raw_0p001deg: List[Optional[int]] = [None] * self.motor_count
        self.motor_multi_turn_angle_raw_0p001deg: List[Optional[int]] = [None] * self.motor_count

        self.motor_error_flags: List[int] = [0] * self.motor_count
        self.motor_operating_state: List[int] = [0] * self.motor_count

        # Per-motor anchor state. `None` until _set_position_anchor() runs.
        self.motor_anchor_angle_deg:       List[Optional[float]] = [None] * self.motor_count
        self.motor_anchor_encoder_count:   List[Optional[int]]   = [None] * self.motor_count
        self.motor_prev_encoder_count:     List[Optional[int]]   = [None] * self.motor_count
        self.motor_encoder_wrap_count:     List[int]             = [0]    * self.motor_count

        # Boot anchor fold window (per motor, motor degrees)
        self.motor_fold_center_deg: List[Optional[float]] = [None] * self.motor_count

        joint_pos_limit = self.cfg["joint_pos_limit"]
        for joint_i in range(self.num_joints):
            motor_i = self.motor_index_for_joint[joint_i] if self.motor_index_for_joint is not None else joint_i

            limit_lo_rad = float(joint_pos_limit[2 * joint_i])
            limit_hi_rad = float(joint_pos_limit[2 * joint_i + 1])
            if not (math.isfinite(limit_lo_rad) and math.isfinite(limit_hi_rad)):
                continue  # wheel — nothing to fold into

            gear = float(self.motor_gear_ratio[motor_i])
            direction = float(self.motor_direction[motor_i])
            offset_rad = float(self.joint_offsets[joint_i])

            # Inverse of _package_motor_states: calibrated joint rad -> raw motor deg.
            raw_lo_deg = math.degrees(limit_lo_rad + offset_rad) * gear / direction
            raw_hi_deg = math.degrees(limit_hi_rad + offset_rad) * gear / direction
            self.motor_fold_center_deg[motor_i] = 0.5 * (raw_lo_deg + raw_hi_deg)

        # Persistent thread pool reused across every tick. 
        self._io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.motor_count, thread_name_prefix="motor_io")


    # =========================================================================
    # Initialization
    # =========================================================================
    def _set_position_anchor(self) -> None:
        """Capture absolute motor angle + matching encoder count as the
        per-motor anchor. Called once after the init multi-turn poll."""
        for motor_index in range(self.motor_count):
            raw_multi = self.motor_multi_turn_angle_raw_0p001deg[motor_index]
            raw_single = self.motor_single_turn_angle_raw_0p001deg[motor_index]
            if raw_multi is not None:
                anchor_motor_angle_deg = raw_multi * self.angle_deg_per_lsb
            elif raw_single is not None:
                anchor_motor_angle_deg = raw_single * self.angle_deg_per_lsb
            else:
                raise RuntimeError(f"Motor {motor_index}: no absolute-angle response during initialization.")

            fold_center_deg = self.motor_fold_center_deg[motor_index]
            if fold_center_deg is not None:
                # Pull the boot reading into (center-180, center+180]
                anchor_motor_angle_deg = fold_center_deg + ((anchor_motor_angle_deg - fold_center_deg + 180.0) % 360.0 - 180.0)

            self.motor_anchor_angle_deg[motor_index]       = anchor_motor_angle_deg
            self.motor_anchor_encoder_count[motor_index]   = self.motor_encoder_count[motor_index]
            self.motor_prev_encoder_count[motor_index]     = self.motor_encoder_count[motor_index]
            self.motor_encoder_wrap_count[motor_index]     = 0

    def read_initial_motor_state(self) -> MotorStatesData:
        def fetch_motor_data(motor_index: int) -> tuple[int, bool]:
            # Order matters: 0x92 multi-turn (anchor_motor_angle_deg) and
            # 0x9C state2 (anchor_encoder_count) must be adjacent in time
            # so the anchor pair refers to nearly the same physical motor
            # position.
            is_valid_flag   = self.poll_state1(motor_index)                 # 0x9A error flags
            is_valid_angle  = self.poll_single_and_multi_turn(motor_index)  # 0x92 / 0x94
            is_valid_state2 = self.poll_state2(motor_index)                 # 0x9C — pairs with 0x92
            is_valid_pi_gain = self.poll_pi_gain(motor_index)               # 0x30

            return motor_index, (is_valid_flag and is_valid_angle and is_valid_state2 and is_valid_pi_gain)

        # Submit on the persistent pool and drain via f.result().
        futures = [self._io_pool.submit(fetch_motor_data, i) for i in range(self.motor_count)]
        results = [future.result() for future in futures]
        failed = [motor_index for motor_index, valid in results if not valid]

        # Check Initializing
        if failed:
            raise RuntimeError(f"Initial motor-state read failed: motors={failed}")

        # Seed (or re-seed) the encoder anchor exactly when we have a fresh multi-turn read available. 
        self._set_position_anchor()

        # Update folded multi-turn angle
        for motor_index in range(self.motor_count):
            self._update_motor_angle_from_encoder(motor_index)

        return self._package_motor_states()


    # ==================================
    # Motor control input 
    # ==================================
    def torque_clipping(self, torque_cmd:np.ndarray) -> np.ndarray:
        # Joint Space
        clipped_torque = np.clip(torque_cmd, -self.max_torque_per_joint, self.max_torque_per_joint)
        return clipped_torque

    def torque_to_current(self, torque_cmd:np.ndarray) -> np.ndarray:
        # Convert torque into current (Joint -> Motor)
        current_command_amp = torque_cmd / self.torque_to_current_denominator # Joint -> Motor
        return current_command_amp

    # ==================================
    # Read byte signal 
    # ==================================    
    def poll_state1(self, motor_index: int, timeout: float = 0.05) -> bool:
        """Read 0x9A error flags."""
        driver = self.motor_drivers[motor_index]
        if driver.can_interface.is_reader_running():
            # Reader thread mode: fire request, read prior reply from cache.
            driver.send_state1_request()
            response_message = driver.latest_state1()
        else:
            # Init mode: blocking round-trip is safe; reader not yet running.
            response_message = driver.read_state1(timeout=timeout)

        if response_message is None:
            return False

        response_data = response_message.data
        self.motor_operating_state[motor_index] = int(response_data[6])
        self.motor_error_flags[motor_index] = int(response_data[7])

        return True

    def poll_state2(self, motor_index: int, timeout: float = 0.05) -> bool:
        response_message = self.motor_drivers[motor_index].read_state2(timeout=timeout)
        if response_message is None:
            return False

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

        return True

    def poll_pi_gain(self, motor_index: int, timeout: float = 0.05) -> bool:
        response_message = self.motor_drivers[motor_index].read_pi_gain(timeout=timeout)
        if response_message is None:
            return False

        response_message = bytes(response_message.data)
        self.motor_pi_gain[motor_index][0] = response_message[6]
        self.motor_pi_gain[motor_index][1] = response_message[7]

        return True
        
    def poll_single_and_multi_turn(self, motor_index: int, timeout: float = 0.25) -> bool:
        response_single = self.motor_drivers[motor_index].read_single_turn(timeout=timeout)
        if response_single is not None:
            self.motor_single_turn_angle_raw_0p001deg[motor_index] = int.from_bytes(response_single.data[4:8], byteorder="little", signed=False)

        response_multi = self.motor_drivers[motor_index].read_multi_turn(timeout=timeout)
        if response_multi is not None:
            raw_7bytes = response_multi.data[1:8]
            sign_extension_byte = b"\x00" if raw_7bytes[-1] < 0x80 else b"\xff"      
            self.motor_multi_turn_angle_raw_0p001deg[motor_index] = int.from_bytes(raw_7bytes + sign_extension_byte, byteorder="little", signed=True)

        return response_multi is not None or response_single is not None
        
    # =========================================================================
    # MotorStatesData Packaging
    # =========================================================================
    def _package_motor_states(self) -> MotorStatesData:
        """Raw Data -> MotorStatesData."""
        joint_count = len(self.joint_names)
        
        joint_position_rad: List[float] = [0.0] * joint_count
        joint_velocity_rad_per_sec: List[float] = [0.0] * joint_count
        joint_effort_like: List[float] = [0.0] * joint_count
        
        for joint_i in range(joint_count):
            motor_i = self.motor_index_for_joint[joint_i] if self.motor_index_for_joint is not None else joint_i
            gear = self.motor_gear_ratio[motor_i]
            direction = self.motor_direction[motor_i]

            raw_multi = self.motor_multi_turn_angle_raw_0p001deg[motor_i]
            raw_single = self.motor_single_turn_angle_raw_0p001deg[motor_i]

            if raw_multi != 0: motor_angle_deg = raw_multi * self.angle_deg_per_lsb
            elif raw_single != 0: motor_angle_deg = raw_single * self.angle_deg_per_lsb
            else: motor_angle_deg = 0.0

            joint_angle_deg = (motor_angle_deg * direction) / gear # Motor -> Joint
            joint_position_rad[joint_i] = joint_angle_deg * math.pi / 180.0

            motor_speed_deg_s = self.motor_speed_deg_per_sec[motor_i]
            joint_speed_deg_s = (motor_speed_deg_s * direction) / gear # Motor -> Joint
            joint_velocity_rad_per_sec[joint_i] = joint_speed_deg_s * math.pi / 180.0

            motor_current_amp = self.motor_phase_current_amp[motor_i]
            joint_effort_like[joint_i] = motor_current_amp * self.torque_to_current_denominator[motor_i]

        joint_position_rad = np.array(joint_position_rad) - self.joint_offsets
        
        return MotorStatesData(
            joint_names=self.joint_names,
            joint_position_rad=list(joint_position_rad),
            joint_velocity_rad_per_sec=list(joint_velocity_rad_per_sec),
            joint_effort_like=list(joint_effort_like),
            motor_temperature_c=self.motor_temperature_c.copy(),
            motor_error_flags=self.motor_error_flags.copy(),
            motor_operating_state=self.motor_operating_state.copy(),
            timestamp_sec=time.time())

    def _update_motor_angle_from_encoder(self, motor_index: int) -> None:
        """Integrate the latest uint16 encoder count into a multi-turn motor
        angle and write it back into motor_multi_turn_angle_raw_0p001deg so
        _package_motor_states picks it up without further changes."""
        anchor_angle_deg = self.motor_anchor_angle_deg[motor_index]
        if anchor_angle_deg is None:
            return 

        counts_per_rev = self.motor_encoder_counts_per_rev
        half_range = counts_per_rev // 2

        current_count = self.motor_encoder_count[motor_index]
        prev_count = self.motor_prev_encoder_count[motor_index]
        delta_count = current_count - prev_count

        # Wrap detection — a single-tick step larger than half range means
        # the uint encoder field wrapped, not that the motor moved that fast.
        if delta_count > half_range:
            self.motor_encoder_wrap_count[motor_index] -= 1
        elif delta_count < -half_range:
            self.motor_encoder_wrap_count[motor_index] += 1
        self.motor_prev_encoder_count[motor_index] = current_count

        anchor_count = self.motor_anchor_encoder_count[motor_index]
        total_count_delta = (current_count - anchor_count) + self.motor_encoder_wrap_count[motor_index] * counts_per_rev
        motor_angle_deg = anchor_angle_deg + total_count_delta * (360.0 / counts_per_rev)

        # Write back in the same 0.001 deg/LSB units _package_motor_states reads.
        self.motor_multi_turn_angle_raw_0p001deg[motor_index] = int(round(motor_angle_deg / self.angle_deg_per_lsb))

    # =========================================================================
    # [Write + Read]
    # =========================================================================
    def write_torques_and_read_states(self, current_cmd_amp: Sequence[float], timeout: float = 0.003) -> MotorStatesData:
        """Synchronous fire-all-then-wait-all torque + state2 pass.

        Phase 1 — clear each motor's 0xA1 reply event, then send its torque
                  command. TXs are non-blocking (kernel ring); both buses
                  run in parallel by virtue of separate CanInterface
                  instances.
        Phase 2 — for each motor, wait on its arrival event until the shared
                  deadline. On timeout, inject NaN so the kill switch fires
                  instead of letting stale data drive the controller.
        """
        # Phase 1 — arm + fire. Sequential is fine: bus.send() goes into the
        # kernel TX ring without blocking on the wire.
        t_submit = time.perf_counter()                                           # [timing]
        for motor_index, amp in enumerate(current_cmd_amp):
            driver = self.motor_drivers[motor_index]
            driver.clear_state2_event()
            driver.send_torque_only(float(amp), self.max_current_lsb, self.motor_current_amp_per_lsb)
        t_fired = time.perf_counter()                                            # [timing]

        # Phase 2 — bounded wait, one shared deadline so total RX time is
        # bounded by `timeout` rather than 8 × per-motor timeout.
        deadline_monotonic = time.monotonic() + timeout
        for motor_index in range(self.motor_count):
            driver = self.motor_drivers[motor_index]
            response_message = driver.await_state2(deadline_monotonic)
            if response_message is None:
                # Motor silent this tick. Mark its slot NaN so
                # controller_node._sensor_data_has_nan() trips the kill
                # switch — far safer than re-using stale state for a balance loop.
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