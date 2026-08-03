#!/usr/bin/env python3
# goat_sysid/goat_sysid/motor_id.py
"""Motor unit-scale verification node (READ-ONLY).

Verifies that the unit scales in `goat_sysid/config/config.yaml` match the real
hardware:

  position test -> angle_deg_per_lsb
  velocity test -> speed_deg_per_sec_per_lsb

`motor_gear_ratio` is treated as ground truth (it comes from the mechanical
design), so the position test has exactly one unknown and reports a single
conclusion for `angle_deg_per_lsb`.

This node NEVER commands torque — it only issues 0x9C / 0x92 read requests, so
the motor cannot move on its own. No E-stop procedure is required.

What this node canNOT verify:
  - motor_current_amp_per_lsb: the command path converts LSB -> A -> LSB with
    the same constant, so it cancels out; a loopback test is blind to it.
  - motor_torque_constant_nm_per_amp: needs a load cell or a known load.
"""

from __future__ import annotations

import signal
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node

from goat_control.utils.motor import CanInterface, MotorDriver, MotorParams


# The 0x9C/0xA1 speed field is int16. Once it pins at the limit the reported
# value stops tracking the real speed, so those samples must leave the fit.
SPEED_RAW_SATURATION = 32767


# =============================================================================
# CAN frame decoding
# =============================================================================
def parse_state2(data: bytes) -> dict:
    """Decode a 0x9C / 0xA1 reply payload."""
    return {
        "temp_c": struct.unpack("<b", data[1:2])[0],
        "iq_raw": struct.unpack("<h", data[2:4])[0],
        "speed_raw": struct.unpack("<h", data[4:6])[0],
        "enc_raw": struct.unpack("<H", data[6:8])[0],
    }


def parse_multi_turn(data: bytes) -> int:
    """Decode a 0x92 reply payload (7-byte signed little-endian)."""
    raw_7bytes = data[1:8]
    sign_extension_byte = b"\x00" if raw_7bytes[-1] < 0x80 else b"\xff"
    return int.from_bytes(raw_7bytes + sign_extension_byte, byteorder="little", signed=True)


class MotorIdNode(Node):
    """Interactive, single-motor unit-scale verification node."""

    def __init__(self) -> None:
        super().__init__("motor_id_node")

        # -----------------------------------------------------------------
        # Parameters (launch-overridable)
        # -----------------------------------------------------------------
        self.declare_parameter("yaml_path", "config.yaml")
        self.declare_parameter("joint_name", "")
        self.declare_parameter("joint_index", 0)
        self.declare_parameter("test", "position")
        self.declare_parameter("expected_turns", 1.0)
        self.declare_parameter("velocity_duration_sec", 30.0)
        self.declare_parameter("sample_rate_hz", 200.0)
        self.declare_parameter("stream_rate_hz", 10.0)

        self.yaml_path = str(self.get_parameter("yaml_path").value)
        self.test_name = str(self.get_parameter("test").value).strip().lower()
        self.expected_turns = float(self.get_parameter("expected_turns").value)
        self.velocity_duration_sec = float(self.get_parameter("velocity_duration_sec").value)
        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self.stream_rate_hz = float(self.get_parameter("stream_rate_hz").value)

        self.log = self.get_logger()

        if self.test_name not in ("position", "velocity"):
            raise ValueError(f"test must be 'position' or 'velocity' (got '{self.test_name}')")

        # -----------------------------------------------------------------
        # Config — the single source of truth for every scale under test
        # -----------------------------------------------------------------
        with open(self.yaml_path, "r", encoding="utf-8") as file_handle:
            self.cfg = yaml.safe_load(file_handle)
        if not isinstance(self.cfg, dict):
            raise ValueError("YAML root must be a mapping/dict.")

        self.angle_deg_per_lsb = float(self.cfg["angle_deg_per_lsb"])
        self.speed_deg_per_sec_per_lsb = float(self.cfg["speed_deg_per_sec_per_lsb"])
        # The live controller derives position from this encoder field, NOT
        # from 0x92 (motor_manager._update_motor_angle_from_encoder), so it
        # needs its own ground-truth check.
        self.encoder_counts_per_rev = int(self.cfg.get("motor_encoder_counts_per_rev", 65536))

        self.joint_index = self._resolve_joint_index()
        self.joint_name = str(self.cfg["joint_names"][self.joint_index])
        # Ground truth from the mechanical design — NOT under test here.
        self.gear_ratio = float(self.cfg["motor_gear_ratio"][self.joint_index])
        self.direction = float(self.cfg["motor_direction"][self.joint_index])

        # -----------------------------------------------------------------
        # CAN — open only the bus this motor sits on
        # -----------------------------------------------------------------
        bus_index = int(self.cfg["motor_bus_index"][self.joint_index])
        self.can_channel = str(self.cfg["can_channels"][bus_index])
        self.node_id = int(self.cfg["motor_node_ids"][self.joint_index])

        self.can = CanInterface(channel=self.can_channel, interface="socketcan")
        self.can.open()
        self.driver = MotorDriver(
            self.can,
            MotorParams(
                node_id=self.node_id,
                direction=int(self.direction),
                gear_ratio=self.gear_ratio,
            ),
        )
        self._open_operator_terminal()

        # -----------------------------------------------------------------
        # Streaming state
        # -----------------------------------------------------------------
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._velocity_status: dict = {}

        self.log.info("=" * 66)
        self.log.info(f" motor_id_node — test '{self.test_name}'")
        self.log.info(f" joint          : [{self.joint_index}] {self.joint_name}")
        self.log.info(f" CAN            : {self.can_channel} node_id={self.node_id}")
        self.log.info(f" gear ratio     : {self.gear_ratio}")
        self.log.info(f" direction      : {self.direction:+.0f}")
        self.log.info(f" angle_deg_per_lsb          : {self.angle_deg_per_lsb}")
        self.log.info(f" speed_deg_per_sec_per_lsb  : {self.speed_deg_per_sec_per_lsb}")
        self.log.info(f" config         : {self.yaml_path}")
        self.log.info("=" * 66)

        self._worker = threading.Thread(target=self._run, daemon=True, name="motor_id_worker")
        self._worker.start()

    # =====================================================================
    # Setup helpers
    # =====================================================================
    def _resolve_joint_index(self) -> int:
        """joint_name wins over joint_index when non-empty."""
        joint_names = list(self.cfg["joint_names"])

        requested_name = str(self.get_parameter("joint_name").value).strip()
        if requested_name:
            if requested_name not in joint_names:
                raise ValueError(f"joint_name '{requested_name}' not in config joint_names {joint_names}")
            return joint_names.index(requested_name)

        joint_index = int(self.get_parameter("joint_index").value)
        if not (0 <= joint_index < len(joint_names)):
            raise ValueError(f"joint_index {joint_index} out of range (0..{len(joint_names) - 1})")
        return joint_index

    # =====================================================================
    # Terminal I/O
    # =====================================================================
    def _open_operator_terminal(self) -> None:
        """Pick the streams the operator can actually read and type on.

        Order matters: a real stdin (i.e. `ros2 run` from a shell) is the most
        reliable, and only when stdin is not a terminal do we reach for
        /dev/tty — the `ros2 launch` case, where stdin is a dead pipe.
        `self.tty_in` stays None when neither works; _prompt() then fails
        loudly instead of hanging.
        """
        self.tty_owned = False  # True only when we opened /dev/tty ourselves

        if sys.stdin.isatty():
            self.tty_in = sys.stdin
            self.tty_out = sys.stdout
            self.log.info("[terminal] 입력 경로: stdin (tty)")
            return
        try:
            terminal_in = open("/dev/tty", "r")
            terminal_out = open("/dev/tty", "w")
        except OSError as exc:
            self.tty_in = None
            self.tty_out = sys.stdout
            self.log.warn(
                f"[terminal] stdin이 tty가 아니고 /dev/tty도 열 수 없습니다 ({exc}). "
                "제어 터미널이 없는 환경(systemd 서비스 등)으로 보입니다. "
                "터미널에서 직접 실행하세요."
            )
            return

        self.tty_in = terminal_in
        self.tty_out = terminal_out
        self.tty_owned = True
        self.log.info("[terminal] 입력 경로: /dev/tty (stdin이 tty가 아님 — ros2 launch)")

    def _prompt(self, message: str) -> str:
        """Blocking prompt on the operator's terminal."""
        if self.tty_in is None:
            raise RuntimeError(
                "대화형 입력 경로가 없습니다 (stdin이 tty가 아니고 /dev/tty도 없음). "
                "제어 터미널이 있는 셸에서 실행하세요."
            )

        # Always flush: the prompt has no trailing newline, so a buffered
        # stream would swallow it and leave the operator staring at nothing.
        self.tty_out.write(message)
        self.tty_out.flush()

        try:
            line = self.tty_in.readline()
        except OSError as exc:
            # EIO here means the read was refused because this process is not
            # in the terminal's foreground group (see the SIGTTIN guard).
            raise RuntimeError(
                f"터미널 읽기 실패({exc}). 이 노드는 포그라운드 터미널에서 실행해야 합니다."
            ) from exc

        if line == "":  # EOF — input closed or redirected from /dev/null
            raise RuntimeError("입력이 EOF로 닫혔습니다. 터미널에서 직접 실행하세요.")
        return line.strip()

    def _say(self, message: str = "") -> None:
        """Procedural text for the operator (not mirrored into rosout)."""
        self.tty_out.write(message + "\n")
        self.tty_out.flush()

    # =====================================================================
    # Real-time streaming
    # =====================================================================
    def _start_stream(self, line_builder) -> None:
        """Emit one status line per 1/stream_rate_hz until stopped."""
        self._stream_stop.clear()

        def stream_loop() -> None:
            period_sec = 1.0 / max(self.stream_rate_hz, 0.1)
            while not self._stream_stop.is_set():
                try:
                    line = line_builder()
                except Exception as exc:  # never let the stream kill the test
                    line = f"[stream] error: {exc}"
                if line:
                    self.log.info(line)
                self._stream_stop.wait(period_sec)

        self._stream_thread = threading.Thread(target=stream_loop, daemon=True, name="motor_id_stream")
        self._stream_thread.start()

    def _start_position_collector(self, start_snapshot: dict) -> dict:
        """Sample during the hand rotation, accumulating the encoder count and
        the speed integral alongside the 0x92 angle. Returns the accumulator
        dict, filled in by the thread until _stop_stream()."""
        accumulator = {
            "encoder_counts": 0,   # unwrapped, signed
            "speed_path_deg": 0.0,  # integral of |reported speed|
            "samples": 0,
            "misses": 0,
        }
        self._stream_stop.clear()

        def collector_loop() -> None:
            counts_per_rev = self.encoder_counts_per_rev
            half_range = counts_per_rev // 2
            previous_encoder = start_snapshot["enc_raw"]
            previous_time = start_snapshot["t"]
            t0 = time.perf_counter()

            sample_period_sec = 1.0 / max(self.sample_rate_hz, 1.0)
            print_period_sec = 1.0 / max(self.stream_rate_hz, 0.1)
            next_print_time = t0

            while not self._stream_stop.is_set():
                snapshot = self._read_snapshot(timeout=0.02)
                if snapshot is None:
                    accumulator["misses"] += 1
                else:
                    # Same wrap rule the live controller uses.
                    delta_counts = snapshot["enc_raw"] - previous_encoder
                    if delta_counts > half_range:
                        delta_counts -= counts_per_rev
                    elif delta_counts < -half_range:
                        delta_counts += counts_per_rev
                    accumulator["encoder_counts"] += delta_counts
                    previous_encoder = snapshot["enc_raw"]

                    dt = snapshot["t"] - previous_time
                    previous_time = snapshot["t"]
                    if 0.0 < dt < 1.0:
                        accumulator["speed_path_deg"] += (
                            abs(snapshot["speed_raw"] * self.speed_deg_per_sec_per_lsb) * dt
                        )

                    accumulator["samples"] += 1
                    accumulator["last"] = snapshot

                now = time.perf_counter()
                if now >= next_print_time:
                    next_print_time = now + print_period_sec
                    last = accumulator.get("last")
                    if last is None:
                        self.log.info("[pos] 응답 없음 — CAN 확인")
                    else:
                        raw_delta = last["multi_raw"] - start_snapshot["multi_raw"]
                        output_deg = self._output_deg_from_raw(raw_delta)
                        self.log.info(
                            f"[pos] t={now - t0:5.1f}s | "
                            f"raw={last['multi_raw']:>10,} | "
                            f"motor={raw_delta * self.angle_deg_per_lsb:9.3f} deg | "
                            f"누적={output_deg / 360.0:+.3f} turn | 목표 {self.expected_turns:.3f} turn | "
                            f"enc={accumulator['encoder_counts']:+,} | "
                            f"∫v={accumulator['speed_path_deg']:8.1f} deg"
                        )

                sleep_sec = sample_period_sec - (time.perf_counter() - now)
                if sleep_sec > 0.0:
                    self._stream_stop.wait(sleep_sec)

        self._stream_thread = threading.Thread(
            target=collector_loop, daemon=True, name="motor_id_stream")
        self._stream_thread.start()
        return accumulator

    def _stop_stream(self) -> None:
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
        self._stream_thread = None

    # =====================================================================
    # Sampling
    # =====================================================================
    def _read_snapshot(self, timeout: float = 0.05) -> dict | None:
        """One 0x9C + 0x92 pair. Returns None if either read times out."""
        state2_msg = self.driver.read_state2(timeout=timeout)
        multi_turn_msg = self.driver.read_multi_turn(timeout=timeout)
        if state2_msg is None or multi_turn_msg is None:
            return None
        snapshot = parse_state2(state2_msg.data)
        snapshot["multi_raw"] = parse_multi_turn(multi_turn_msg.data)
        snapshot["t"] = time.perf_counter()
        return snapshot

    def _output_deg_from_raw(self, raw_delta: int) -> float:
        """Raw multi-turn LSB delta -> output-shaft degrees (config scales)."""
        motor_deg = raw_delta * self.angle_deg_per_lsb
        return motor_deg * self.direction / self.gear_ratio

    # =====================================================================
    # Worker
    # =====================================================================
    def _run(self) -> None:
        try:
            if self._read_snapshot() is None:
                self.log.error("모터가 응답하지 않습니다. 전원과 CAN 연결을 확인하세요.")
            elif self.test_name == "position":
                self._test_position()
            else:
                self._test_velocity()
        except Exception as exc:
            self.log.error(f"테스트 중 예외 발생: {exc}")
        finally:
            self._stop_stream()
            self.close()
            if rclpy.ok():
                rclpy.shutdown()

    # =====================================================================
    # Position test — verifies angle_deg_per_lsb
    # =====================================================================
    def _test_position(self) -> None:
        self._say()
        self._say("=" * 66)
        self._say(" Position Test — angle_deg_per_lsb 검증")
        self._say("=" * 66)
        self._say()
        self._say(" 준비:  출력축(관절 쪽)에 유성펜으로 표시를 하나 그리세요.")
        self._say("        그 표시가 가리키는 고정된 기준점도 정하세요.")
        self._say()
        self._prompt(" 준비되면 Enter... ")

        start_snapshot = self._read_snapshot()
        if start_snapshot is None:
            self.log.error("[실패] 시작 상태를 읽지 못했습니다.")
            return
        self.log.info(f"시작 원시값: {start_snapshot['multi_raw']:,} (enc={start_snapshot['enc_raw']})")

        self._say()
        self._say(f" 이제 출력축을 손으로 천천히 정확히 '{self.expected_turns:g} 바퀴' 돌리세요.")
        self._say(" 아래에 누적 회전수가 실시간으로 표시됩니다.")
        self._say()

        accumulator = self._start_position_collector(start_snapshot)
        self._prompt(" 다 돌렸으면 Enter... ")
        self._stop_stream()

        end_snapshot = self._read_snapshot()
        if end_snapshot is None:
            self.log.error("[실패] 종료 상태를 읽지 못했습니다.")
            return

        raw_delta = end_snapshot["multi_raw"] - start_snapshot["multi_raw"]
        measured_output_deg = self._output_deg_from_raw(raw_delta)
        measured_turns = measured_output_deg / 360.0

        if abs(self.expected_turns) < 1e-9:
            self.log.error("[실패] expected_turns가 0입니다.")
            return
        ratio = measured_turns / self.expected_turns

        self.log.info("-" * 66)
        self.log.info(f" 원시값 변화량        : {raw_delta:,}")
        self.log.info(f" 코드가 인식한 회전수 : {measured_turns:+.4f} 바퀴")
        self.log.info(f" 실제 회전수          : {self.expected_turns:+.4f} 바퀴")
        self.log.info(f" >>> 오차 배수        : {ratio:+.4f}")
        self.log.info("-" * 66)

        if ratio < 0.0:
            self.log.warn(
                "측정 방향이 config의 양(+)의 방향과 반대입니다. "
                f"motor_direction[{self.joint_index}]={self.direction:+.0f} 를 확인하세요 "
                "(크기 비교는 계속 진행합니다)."
            )

        magnitude_ratio = abs(ratio)
        if magnitude_ratio < 1e-6:
            self.log.error(
                "[실패] 위치가 전혀 변하지 않았습니다. 축을 실제로 돌렸는지, "
                "그리고 이 관절이 맞는지 (joint_index/joint_name) 확인하세요."
            )
            return

        if abs(magnitude_ratio - 1.0) < 0.02:
            self.log.info(f"[통과] angle_deg_per_lsb = {self.angle_deg_per_lsb} 는 정확합니다.")
        else:
            corrected = self.angle_deg_per_lsb / magnitude_ratio
            self.log.warn(f"[불일치] 위치가 실제의 {magnitude_ratio:.4f}배로 계산되고 있습니다.")
            self.log.warn(
                f"         angle_deg_per_lsb: {self.angle_deg_per_lsb} -> {corrected:.6f} 로 바꾸면 맞습니다."
            )
            # gear ratio is assumed correct, but an exact integer multiple is a
            # strong hint that the config gear entry is the real culprit.
            for candidate in (magnitude_ratio, 1.0 / magnitude_ratio):
                if abs(candidate - round(candidate)) < 0.02 and round(candidate) >= 2:
                    self.log.warn(
                        f"         [참고] 오차가 정확히 {round(candidate)}배입니다. "
                        f"motor_gear_ratio[{self.joint_index}]={self.gear_ratio:g} 오기입 가능성도 확인하세요."
                    )
                    break

        self._report_position_cross_checks(accumulator)
        self._report_unverifiable()

    def _report_position_cross_checks(self, accumulator: dict) -> None:
        """Check the encoder path and the speed field against the SAME physical
        rotation, so neither depends on 0x92 being right."""
        self.log.info("")
        self.log.info("=" * 66)
        self.log.info(" 교차검증 — 실제 회전 1건에 세 신호를 대조")
        self.log.info("=" * 66)
        self.log.info(f" 샘플 {accumulator['samples']} (miss={accumulator['misses']})")

        turns = abs(self.expected_turns)
        if accumulator["samples"] < 20 or turns < 1e-9:
            self.log.warn(" [건너뜀] 샘플이 부족합니다.")
            return

        # ---- 1) Encoder path — this is what the live controller integrates.
        measured_counts_per_output_turn = abs(accumulator["encoder_counts"]) / turns
        expected_counts_per_output_turn = self.encoder_counts_per_rev * self.gear_ratio
        self.log.info("")
        self.log.info(" [1] 엔코더 경로 (실제 컨트롤러가 위치 적분에 쓰는 신호)")
        self.log.info(f"     출력축 1바퀴당 측정 카운트 : {measured_counts_per_output_turn:,.0f}")
        self.log.info(f"     config 기준 기대 카운트    : {expected_counts_per_output_turn:,.0f} "
                      f"(= {self.encoder_counts_per_rev:,} x gear {self.gear_ratio:g})")
        if expected_counts_per_output_turn > 0:
            encoder_ratio = measured_counts_per_output_turn / expected_counts_per_output_turn
            self.log.info(f"     >>> 배수 : {encoder_ratio:.4f}")
            if abs(encoder_ratio - 1.0) < 0.05:
                self.log.info("     [통과] 엔코더는 출력축 기준입니다.")
            else:
                self.log.warn(f"     [불일치] 엔코더가 출력축 1바퀴에 기대값의 {encoder_ratio:.2f}배를 셉니다.")
                if abs(encoder_ratio - round(encoder_ratio)) < 0.05 and round(encoder_ratio) >= 2:
                    self.log.warn(f"     → 모터 내부 감속비 1:{round(encoder_ratio)} 가 있고 엔코더는 "
                                  "로터(감속 전) 기준이라는 뜻입니다.")
                    self.log.warn("       이 경우 speed 필드도 로터 기준일 가능성이 높습니다.")

        # ---- 2) Speed field vs the physical turn (independent of 0x92).
        # Motor-side degrees implied by the rotation the operator performed.
        expected_motor_deg = turns * 360.0 * self.gear_ratio
        self.log.info("")
        self.log.info(" [2] 속도 필드 vs 실제 회전 (0x92를 거치지 않는 독립 검증)")
        self.log.info(f"     속도 적분값        : {accumulator['speed_path_deg']:.1f} deg (현재 설정 기준)")
        self.log.info(f"     실제 회전량        : {expected_motor_deg:.1f} deg (모터축 환산)")
        if accumulator["speed_path_deg"] > 1e-6:
            speed_ratio = expected_motor_deg / accumulator["speed_path_deg"]
            corrected = self.speed_deg_per_sec_per_lsb * speed_ratio
            self.log.info(f"     >>> 배수 : {speed_ratio:.4f}")
            if abs(speed_ratio - 1.0) < 0.05:
                self.log.info(f"     [통과] speed_deg_per_sec_per_lsb = "
                              f"{self.speed_deg_per_sec_per_lsb} 가 실제 회전과 일치합니다.")
            else:
                self.log.warn(f"     [불일치] speed_deg_per_sec_per_lsb: "
                              f"{self.speed_deg_per_sec_per_lsb} -> {corrected:.6f}")
                self.log.warn("     이 값은 velocity 테스트(0x92 기준)와 반드시 일치해야 합니다. "
                              "두 값이 다르면 어느 쪽도 신뢰하지 마세요.")
        else:
            self.log.warn("     [경고] 속도 적분이 0입니다. 손으로 돌리는 동안 속도 필드가 "
                          "갱신되지 않았다는 뜻이므로 velocity 테스트 결과도 의심하세요.")

        self.log.info("")
        self.log.info(" 주의: 천천히 한 방향으로만 돌렸을 때만 [2]가 유효합니다 "
                      "(왕복하면 적분 경로가 실제 회전량보다 커집니다).")

    # =====================================================================
    # Velocity test — verifies speed_deg_per_sec_per_lsb
    # =====================================================================
    def _test_velocity(self) -> None:
        self._say()
        self._say("=" * 66)
        self._say(" Velocity Test — speed_deg_per_sec_per_lsb 검증")
        self._say("=" * 66)
        self._say()
        self._say(" 전제: position 테스트가 먼저 통과해야 이 결과가 의미 있습니다.")
        self._say(f" {self.velocity_duration_sec:.0f}초 동안 출력축을 손으로 자유롭게 돌리세요.")
        self._say(" 한 방향으로만 일정하게 돌리면 결과가 부정확합니다.")
        self._say()
        self._prompt(" 준비되면 Enter... ")

        pairs: list[tuple[float, float]] = []
        sample_count = 0
        miss_count = 0
        saturated_count = 0
        position_path_deg = 0.0
        reported_path_deg = 0.0
        reported_dps_min = 0.0
        reported_dps_max = 0.0
        self._velocity_status = {"reported_dps": 0.0, "derived_dps": 0.0}

        def velocity_line() -> str | None:
            status = self._velocity_status
            elapsed = status.get("elapsed", 0.0)
            sample_hz = sample_count / elapsed if elapsed > 0.0 else 0.0
            miss_text = f" | miss={miss_count}" if miss_count else ""
            if saturated_count:
                miss_text += f" | 포화={saturated_count} (너무 빠름)"
            return (
                f"[vel] t={elapsed:5.1f}s/{self.velocity_duration_sec:.0f}s | "
                f"n={sample_count} ({sample_hz:5.1f} Hz) | "
                f"보고={status['reported_dps']:+8.2f} dps | "
                f"미분={status['derived_dps']:+8.2f} dps | "
                f"유효샘플={len(pairs)}{miss_text}"
            )

        self._start_stream(velocity_line)

        # Differentiated position is the reference (angle_deg_per_lsb is
        # verified by the position test); the reported speed field is the
        # quantity under test. Both are motor-side deg/s, so gear ratio and
        # direction cancel out and play no part in this test.
        sample_period_sec = 1.0 / max(self.sample_rate_hz, 1.0)
        previous_snapshot: dict | None = None
        t0 = time.perf_counter()
        next_sample_time = t0

        while time.perf_counter() - t0 < self.velocity_duration_sec:
            snapshot = self._read_snapshot(timeout=0.02)
            if snapshot is None:
                miss_count += 1
            else:
                sample_count += 1
                if previous_snapshot is not None:
                    dt = snapshot["t"] - previous_snapshot["t"]
                    if dt >= 1e-4:
                        raw_delta = snapshot["multi_raw"] - previous_snapshot["multi_raw"]
                        speed_from_position_dps = raw_delta * self.angle_deg_per_lsb / dt
                        speed_reported_dps = snapshot["speed_raw"] * self.speed_deg_per_sec_per_lsb
                        self._velocity_status["derived_dps"] = speed_from_position_dps
                        self._velocity_status["reported_dps"] = speed_reported_dps

                        # Path lengths — sign-independent, so back-and-forth
                        # hand motion accumulates instead of cancelling.
                        position_path_deg += abs(raw_delta) * self.angle_deg_per_lsb
                        reported_path_deg += abs(speed_reported_dps) * dt
                        reported_dps_min = min(reported_dps_min, speed_reported_dps)
                        reported_dps_max = max(reported_dps_max, speed_reported_dps)
                        if abs(snapshot["speed_raw"]) >= SPEED_RAW_SATURATION:
                            # int16 speed field pinned at its limit — the
                            # reported value is no longer the real speed, so it
                            # would drag the fitted slope down. Drop and warn.
                            saturated_count += 1
                        elif abs(snapshot["speed_raw"]) >= 3:
                            # Near standstill the reported field carries no scale
                            # information — including it would bias the fit to 0.
                            pairs.append((speed_reported_dps, speed_from_position_dps))
                previous_snapshot = snapshot

            self._velocity_status["elapsed"] = time.perf_counter() - t0

            # Pace the loop so dt stays well-conditioned: a free-running loop
            # makes dt small enough that 1-LSB position quantisation dominates
            # the derivative and destroys the fit.
            next_sample_time += sample_period_sec
            sleep_sec = next_sample_time - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_sample_time = time.perf_counter()  # CAN is the bottleneck

        self._stop_stream()

        if sample_count < 100:
            self.log.error(f"[실패] 샘플이 {sample_count}개뿐입니다 (miss={miss_count}). CAN 통신을 확인하세요.")
            return

        if len(pairs) < 50:
            self.log.error(f"[실패] 유효 구간이 {len(pairs)}개뿐입니다. 더 많이 움직여주세요.")
            return

        # Slope of the best-fit line through the origin.
        sum_xy = sum(x * y for x, y in pairs)
        sum_xx = sum(x * x for x, _ in pairs)
        if sum_xx <= 0.0:
            self.log.error("[실패] 보고된 속도가 전부 0입니다.")
            return
        slope = sum_xy / sum_xx

        y_mean = sum(y for _, y in pairs) / len(pairs)
        ss_res = sum((y - slope * x) ** 2 for x, y in pairs)
        ss_tot = sum((y - y_mean) ** 2 for _, y in pairs)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

        self.log.info("-" * 66)
        self.log.info(f" 전체 샘플 : {sample_count} (miss={miss_count})")
        self.log.info(f" 유효 샘플 : {len(pairs)}")
        if saturated_count:
            saturation_dps = SPEED_RAW_SATURATION * self.speed_deg_per_sec_per_lsb
            self.log.warn(
                f" 포화 샘플 : {saturated_count} — 보고 속도가 ±{saturation_dps:.1f} dps(int16 한계)에 "
                "붙어 피팅에서 제외했습니다. 더 천천히 돌리세요."
            )
        self.log.info(f" 직선 적합도(R2) : {r_squared:.4f}")
        self.log.info(f" 측정 속도 구간  : {reported_dps_min:+.1f} ~ {reported_dps_max:+.1f} dps (모터축, 현재 설정 기준)")
        self.log.info(f" >>> 오차 배수 (기울기 방식) : {slope:.4f}")

        # Cross-check with the integral estimator. Two independent estimates
        # agreeing is what makes the number safe to put into the config.
        if reported_path_deg > 1e-6:
            slope_path = position_path_deg / reported_path_deg
            self.log.info(f" >>> 오차 배수 (적분 방식) : {slope_path:.4f}")
            disagreement = abs(slope_path - slope) / max(abs(slope), 1e-9)
            if disagreement > 0.10:
                self.log.warn(
                    f" [경고] 두 추정치가 {disagreement * 100:.1f}% 어긋납니다. "
                    "드라이버 내부 속도 필터/지연 또는 샘플링 문제가 의심되므로 "
                    "이 값을 config에 반영하지 마세요."
                )
        else:
            slope_path = float("nan")
            self.log.warn(" [경고] 적분 방식 교차검증 불가 (보고 속도가 거의 0)")
        self.log.info("-" * 66)

        if r_squared < 0.95:
            self.log.warn("[주의] 적합도가 낮습니다. 결론을 내리지 마세요.")
            self.log.warn("       원인: 샘플 부족, 시간 측정 문제, 또는 드라이버 내부 속도 필터")
            return

        if abs(slope - 1.0) < 0.05:
            self.log.info(
                f"[통과] speed_deg_per_sec_per_lsb = {self.speed_deg_per_sec_per_lsb} 는 정확합니다."
            )
        else:
            corrected = self.speed_deg_per_sec_per_lsb * slope
            self.log.warn(f"[불일치] 속도가 실제의 1/{slope:.4f}로 계산되고 있습니다.")
            self.log.warn(
                f"         speed_deg_per_sec_per_lsb: {self.speed_deg_per_sec_per_lsb} -> "
                f"{corrected:.6f} 로 바꾸면 맞습니다."
            )
            self.log.warn("")
            self.log.warn(f" [!!! 중요 !!!] 이 값만 바꾸면 로봇이 발산할 수 있습니다.")
            self.log.warn(f"   측정 속도가 {slope:.2f}배로 커지므로, 이 값을 소비하는 쪽도 함께 조정해야")
            self.log.warn(f"   기존과 동일한 폐루프 거동이 유지됩니다:")
            self.log.warn(f"   - nsc_leg_derivative_gain / policy_leg_derivative_gain / "
                          f"nsc_wheel_*_derivative_gain : {1.0 / slope:.4f} 배로 조정 "
                          f"(그대로 두면 D항 토크가 {slope:.2f}배가 됨)")
            self.log.warn(f"   - joint_vel_estop_threshold : 지금까지 실제 임계치는 설정값의 {slope:.2f}배였음")
            self.log.warn(f"   - 정책 관측(joint_vel) : 학습 시 단위와 일치하는지 확인 후 무부하 상태에서 먼저 시험")

        self._report_unverifiable()

    # =====================================================================
    # Reporting / teardown
    # =====================================================================
    def _report_unverifiable(self) -> None:
        self.log.info("")
        self.log.info("[이 테스트로 검증되지 않는 값]")
        self.log.info(" - motor_current_amp_per_lsb : 지령 LSB->A->LSB 변환에서 상수가 상쇄되어")
        self.log.info("                               루프백으로는 원리적으로 검증 불가")
        self.log.info(" - motor_torque_constant_nm_per_amp : 로드셀 또는 기지 부하 필요")
        self.log.info(" - 드라이버 내부 전류 제한 : 별도 실부하 시험 필요")

    def close(self) -> None:
        try:
            self.can.close()
        except Exception as exc:
            self.log.warn(f"CAN close 실패: {exc}")
        # Only close /dev/tty if we opened it — never close sys.stdin/stdout.
        if getattr(self, "tty_owned", False):
            for handle in (self.tty_in, self.tty_out):
                try:
                    handle.close()
                except Exception:
                    pass
            self.tty_in = None
            self.tty_out = sys.stdout
            self.tty_owned = False


def main(args=None) -> None:
    # If this process ends up in a background process group, a terminal read
    # would raise SIGTTIN and stop the whole node — indistinguishable from a
    # hang. Ignoring it turns that into an EIO the prompt can report.
    try:
        signal.signal(signal.SIGTTIN, signal.SIG_IGN)
    except (AttributeError, ValueError, OSError):
        pass

    rclpy.init(args=args)
    node = MotorIdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
