from __future__ import annotations

import time
from enum import Enum
from typing import Optional

import numpy as np
import yaml

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState

import mujoco.viewer
from mujoco_ros_bridge import (
    BridgeConfig, MujocoRosBridge, msg_to_torque, seq_to_stamp, stamp_to_seq, state_to_msgs,
)

#: 루프가 콜백을 기다리며 블로킹할 수 있는 최대 시간. 이보다 길게 자면
#: 뷰어 닫기 · Ctrl-C 반응이 눈에 띄게 늦는다.
MAX_WAIT = 0.01


class TorqueSource(Enum):
    PRESTART = "prestart"  # 명령을 한 번도 못 받았다 -> zero
    FRESH = "fresh"        # 최신 명령 사용
    STALE = "stale"        # 받다가 끊겼다 -> watchdog 정책


class MujocoSimNode(Node):

    def __init__(self) -> None:
        super().__init__("mujoco_sim_node")

        self.declare_parameter("xml_path", "config/WF_GOAT/xml/goat_on_stand.xml")
        self.declare_parameter("yaml_path", "../../../Realworld/src/goat_control/config/goat_config.yaml")
        self.declare_parameter("mode", "unsync")                  # Mode sync or unsync
        # self.declare_parameter("physics_substeps", 5)
        # self.declare_parameter("control_dt", 0.005)
        # self.declare_parameter("keyframe", "home")
        # self.declare_parameter("watchdog", "hold")
        # self.declare_parameter("watchdog_timeout_sec", 0.05)
        # self.declare_parameter("watchdog_kd", 0.5)
        self.declare_parameter("lockstep_timeout_sec", 0.5)
        self.declare_parameter("use_viewer", True)
        self.declare_parameter("viewer_hz", 60.0)

        xml_path = str(self.get_parameter("xml_path").value)
        yaml_path = str(self.get_parameter("yaml_path").value)
        with open(yaml_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.joint_names = [str(n) for n in cfg["joint_names"]]             # Our Joint order
        self.control_dt = cfg[""]                                           # TODO: control_dt 어디감?

        self.mode = str(self.get_parameter("mode").value)
        # self.watchdog = str(self.get_parameter("watchdog").value)
        # self.watchdog_timeout = float(self.get_parameter("watchdog_timeout_sec").value)
        # self.watchdog_kd = float(self.get_parameter("watchdog_kd").value)
        self.lockstep_timeout = float(self.get_parameter("lockstep_timeout_sec").value)
        self.use_viewer = bool(self.get_parameter("use_viewer").value)
        self.viewer_period = 1.0 / float(self.get_parameter("viewer_hz").value)

        # ---------------- MuJoCo ROS Bridge ----------------
        self.bridge_cfg = BridgeConfig(xml_path=xml_path,
                                       joint_names=self.joint_names,
                                       physics_substeps=int(self.get_parameter("physics_substeps").value),
                                       control_dt=float(self.get_parameter("control_dt").value),
                                       keyframe=str(self.get_parameter("keyframe").value) or None,)
        self.bridge = MujocoRosBridge(self.bridge_cfg)
        self.control_dt = self.bridge.control_dt
        self.njoint = self.bridge.njoint

        # ---------------- Data Buffer ----------------
        self.latest_tau = np.zeros(self.njoint, dtype=np.float64)
        self.latest_seq = -1
        self.last_cmd_wall: Optional[float] = None
        self.cmd_accepted = 0
        self.cmd_dropped = 0

        # ---------------- 루프 상태 (루프 스레드 전용) ----------------
        self.expected_seq = 0        # 지금 응답을 기다리는 seq
        self._t0 = 0.0               # 워밍업 종료 시점 = seq 격자의 원점
        self._step0 = 0
        self.stale_ticks = 0
        self.timeout_ticks = 0
        self._last_source = TorqueSource.PRESTART
        self._viewer = None
        self._viewer_next = 0.0
        self._reset_requested = False
        self._quit_requested = False
        self._paused = False

        # ---------------- ROS2 Topic ----------------
        reliability = (ReliabilityPolicy.RELIABLE if self.mode is "sync"
                       else ReliabilityPolicy.BEST_EFFORT)
        io_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=reliability)
        clock_qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                               reliability=ReliabilityPolicy.BEST_EFFORT)

        self.clock_pub = self.create_publisher(Clock, "/clock", clock_qos)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", io_qos)
        self.imu_pub = self.create_publisher(ImuState, "/imu", io_qos)
        self.create_subscription(JointState, "/commands", self._on_command, io_qos)

        self.get_logger().info(
            f"mujoco_sim_node: mode={self.mode} "
            f"physics_dt={self.bridge.physics_dt} x{self.bridge.physics_substeps} "
            f"-> control_dt={self.control_dt} | watchdog={self.watchdog} "
            f"(timeout {self.watchdog_timeout}s) | joints={self.njoint}"
        )

    # ------------------------------------------------------------------ #
    # ROS 콜백 — 버퍼 쓰기 두 줄이 전부다. 블로킹 금지.
    # ------------------------------------------------------------------ #
    def _on_command(self, msg: JointState) -> None:
        seq = stamp_to_seq(msg.header.stamp, self.control_dt)
        if seq is None:
            self.cmd_dropped += 1
            self.get_logger().warn("명령 stamp가 제어 격자에 안 맞는다 (폐기)")
            return
        if seq < self.latest_seq:  # 순서 역전
            self.cmd_dropped += 1
            return
        reason = msg_to_torque(msg, self.bridge.joint_names, self.latest_tau)
        if reason is not None:
            self.cmd_dropped += 1
            self.get_logger().warn(f"명령 폐기: {reason}")
            return
        self.latest_seq = seq
        self.last_cmd_wall = time.monotonic()
        self.cmd_accepted += 1

    # ------------------------------------------------------------------ #
    # 두 모드의 술어. 물리 코드 경로는 분기하지 않는다.
    # ------------------------------------------------------------------ #
    def _wait_budget(self, now: float, deadline: float) -> float:
        return max(0.0, min(deadline - now, MAX_WAIT))

    def _should_step(self, now: float, deadline: float) -> bool:
        if self.mode is "sync":
            # '=='가 아니라 '>='. '=='면 패킷 한 번 유실에 영구 데드락.
            return self.latest_seq >= self.expected_seq or now >= deadline
        return now >= deadline

    def _torque_source(self, now: float) -> TorqueSource:
        if self.last_cmd_wall is None:
            return TorqueSource.PRESTART
        if now - self.last_cmd_wall <= self.watchdog_timeout:
            return TorqueSource.FRESH
        return TorqueSource.STALE

    def _select_torque(self, now: float, qvel: np.ndarray) -> np.ndarray:
        """이 tick에 적용할 토크. 여기가 워치독의 전부다.

        - PRESTART: 명령을 한 번도 못 받았다 -> 0. 구독자가 붙기 전에도 시뮬은 돈다.
        - FRESH   : 최신 명령.
        - STALE   : 받다가 끊겼다 -> 기본 hold (마지막 명령 유지).
        """
        source = self._torque_source(now)
        if source is not self._last_source:
            self._on_source_change(self._last_source, source)
            self._last_source = source

        if source is TorqueSource.PRESTART:
            return np.zeros(self.njoint)
        if source is TorqueSource.FRESH:
            return self.latest_tau
        self.stale_ticks += 1
        if self.watchdog == "hold":
            return self.latest_tau
        if self.watchdog == "zero":
            return np.zeros(self.njoint)
        return -self.watchdog_kd * qvel  # damping

    def _on_source_change(self, old: TorqueSource, new: TorqueSource) -> None:
        if new is TorqueSource.FRESH:
            self.get_logger().info(
                f"명령 수신 {'재개' if old is TorqueSource.STALE else '시작'} "
                f"(seq={self.latest_seq})"
            )
        elif new is TorqueSource.STALE:
            self.get_logger().warn(
                f"명령 끊김 ({self.watchdog_timeout}s 초과). watchdog={self.watchdog} 적용. "
                f"마지막 seq={self.latest_seq}"
            )

    # ------------------------------------------------------------------ #
    # 메인 루프
    # ------------------------------------------------------------------ #
    def run(self, executor: SingleThreadedExecutor) -> None:
        if self.use_viewer:
            self._open_viewer()

        # 지연선을 먼저 채운다. 안 하면 첫 발행이 "모든 관절 0"이 된다.
        self._start_epoch()
        # 부트스트랩: seq=0 상태를 능동 발행한다. 안 하면 서로 기다리다 멈춘다.
        self._publish(self.expected_seq)
        deadline = self._next_deadline(time.monotonic())

        while rclpy.ok() and not self._quit_requested:
            now = time.monotonic()
            executor.spin_once(timeout_sec=self._wait_budget(now, deadline))
            self._sync_viewer()

            now = time.monotonic()
            if not self._should_step(now, deadline):
                continue

            if self.mode is "sync" and self.latest_seq < self.expected_seq:
                self.timeout_ticks += 1
                self.get_logger().warn(
                    f"lockstep 타임아웃: seq={self.expected_seq} 응답 없음. "
                    f"워치독 토크로 진행한다"
                )

            if self._consume_reset():
                deadline = self._next_deadline(time.monotonic())
                continue
            if self._paused:
                deadline = self._next_deadline(time.monotonic())
                continue

            state = self.bridge.read_state()
            self.bridge.write_torque(self._select_torque(now, state.qvel))
            self.bridge.step()
            self.expected_seq += 1

            # 불변식. 깨지면 명령 중복 유입이거나 실행기 재진입이다.
            assert (self.bridge.step_count - self._step0
                    == self.expected_seq * self.bridge.physics_substeps)
            assert abs(self.bridge.sim_time - self._t0
                       - self.expected_seq * self.control_dt) < 1e-6

            self._publish(self.expected_seq)
            deadline = self._next_deadline(deadline if self.mode is not "sync"
                                           else time.monotonic())

        self._log_summary()

    def _start_epoch(self) -> None:
        """지연선 워밍업 후 seq 격자의 원점을 잡는다.

        `data.time`을 0으로 되돌리지 않는다 (지연 버퍼가 시각으로 조회된다).
        발행하는 stamp는 원점 기준 `seq x control_dt`이므로 `/clock`은 0에서 시작한다.
        """
        self.bridge.warmup()
        self._t0 = self.bridge.sim_time
        self._step0 = self.bridge.step_count
        if self.bridge.warmup_ticks:
            self.get_logger().info(
                f"센서/액추에이터 지연선 워밍업 {self.bridge.warmup_ticks} tick 완료"
            )

    def _next_deadline(self, base: float) -> float:
        """다음 tick의 기한.

        UNSYNC: **절대 격자**로 누적한다 (`now + control_dt`로 하면 드리프트가 쌓인다).
                0.5초 이상 밀리면 따라잡기를 포기하고 기준을 재설정한다 (몰아치기 방지).
        SYNC  : 격자가 아니라 '이만큼 기다려도 안 오면 진행' 타임아웃이다.
        """
        if self.mode is not "sync":
            deadline = base + self.control_dt
            now = time.monotonic()
            if now - deadline > 0.5:
                self.get_logger().warn("free-run이 0.5s 이상 밀렸다. 기준 재설정")
                deadline = now + self.control_dt
            return deadline
        return base + self.lockstep_timeout

    def _publish(self, seq: int) -> None:
        """/clock을 상태보다 **먼저** 보낸다. 두 상태 메시지의 stamp는 동일해야 한다."""
        stamp = seq_to_stamp(seq, self.control_dt)
        self.clock_pub.publish(Clock(clock=stamp))
        js, imu = state_to_msgs(self.bridge.read_state(), self.bridge.joint_names, stamp)
        self.joint_pub.publish(js)
        self.imu_pub.publish(imu)

    # ------------------------------------------------------------------ #
    # 뷰어 — 키 콜백은 뷰어 스레드에서 돈다. 플래그만 세운다.
    # ------------------------------------------------------------------ #
    def _open_viewer(self) -> None:
        self._viewer = mujoco.viewer.launch_passive(
            self.bridge.model, self.bridge.data,
            key_callback=self._on_viewer_key,
            show_left_ui=False, show_right_ui=False,
        )

    def _on_viewer_key(self, keycode: int) -> None:
        char = chr(keycode) if 0 <= keycode < 0x110000 else ""
        if char in ("r", "R"):
            self._reset_requested = True
        elif char in ("q", "Q"):
            self._quit_requested = True
        elif keycode == 32:  # space
            self._paused = not self._paused

    def _sync_viewer(self) -> None:
        """60 Hz 제한. 매 물리 스텝 부르면 렌더가 물리 주기를 잡아먹는다."""
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            self._quit_requested = True
            return
        now = time.monotonic()
        if now >= self._viewer_next:
            self._viewer.sync()
            self._viewer_next = now + self.viewer_period

    def _consume_reset(self) -> bool:
        if not self._reset_requested:
            return False
        self._reset_requested = False
        self.bridge.reset()
        self._start_epoch()
        self.expected_seq = 0
        self.latest_seq = -1
        self.latest_tau[:] = 0.0
        self.last_cmd_wall = None
        self._last_source = TorqueSource.PRESTART
        self.get_logger().info("reset. seq=0을 다시 부트스트랩 발행한다")
        self._publish(self.expected_seq)
        return True

    def _log_summary(self) -> None:
        self.get_logger().info(
            f"종료: seq={self.expected_seq} steps={self.bridge.step_count - self._step0} "
            f"sim_time={self.bridge.sim_time - self._t0:.3f}s | 명령 accepted={self.cmd_accepted} "
            f"dropped={self.cmd_dropped} | stale_ticks={self.stale_ticks} "
            f"lockstep_timeouts={self.timeout_ticks}"
        )

    def destroy_node(self) -> bool:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoSimNode()
    executor = SingleThreadedExecutor()  # MultiThreaded는 mj_step 재진입 -> 세그폴트
    executor.add_node(node)
    try:
        node.run(executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
