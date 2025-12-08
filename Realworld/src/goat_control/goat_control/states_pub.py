#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from motor_interfaces.msg import MotorStates
from sensor_msgs.msg import JointState
import can
import struct
import time
import math

from .utils.can_mixin import CanMixin  # 🔹 CanMixin 상속

"""
motor_state_publisher.py (MG 전용, CanMixin + full-scan 폴링 버전)

- CanMixin을 상속받아 공통 CAN TX/RX 유틸 사용
- 타이머 한 번 호출마다 모든 모터에 대해:
    1) state2(0x9C) 읽기: 온도, iq, 속도, 엔코더
    2) state1(0x9A) 읽기: motor_state, error_flags
    3) angle(0x94/0x92) 읽기: single/multi-turn 각도
  를 순서대로 읽은 뒤 MotorStates + JointState를 퍼블리시
- poll_hz = 전체 모터 상태 갱신 주기(Hz) 로 해석
"""

# === MG 전용 iq 해상도 (A/LSB) ===
IQ_RES_A_PER_LSB = 66.0 / 4096.0   # MG: ≈ 0.01611 A/LSB


class MotorStatePublisher(Node, CanMixin):
    def __init__(self):
        super().__init__('motor_state_publisher')

        # 파라미터
        self.channel    = self.declare_parameter('channel', 'can0').value
        self.interface  = self.declare_parameter('interface', 'socketcan').value
        self.num_motors = int(self.declare_parameter('num_motors', 8).value)
        self.poll_hz    = float(self.declare_parameter('poll_hz', 200.0).value)
        # 각도 읽을 모터 인덱스(0-based) 집합
        self.single_turn_indices = set(self.declare_parameter('single_turn_indices', []).value)
        self.multi_turn_indices  = set(self.declare_parameter('multi_turn_indices', []).value)

        # MG 해상도 고정
        self.iq_res = IQ_RES_A_PER_LSB

        # CAN 버스 열기 (CanMixin은 self.bus만 있으면 됨)
        self.get_logger().info(f"[MotorStatePublisher] Open CAN bus: channel={self.channel}, interface={self.interface}")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            self.get_logger().error(f"Failed to open CAN bus: {e}")
            raise

        # 퍼블리셔
        self.publisher = self.create_publisher(MotorStates, 'motor_states', 10)
        self.js_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # joint name: joint_1, joint_2, ...
        self.joint_names = [
            'hip_L_Joint',
            'hip_R_Joint',
            'thigh_L_Joint',
            'thigh_R_Joint',
            'knee_L_Joint',
            'knee_R_Joint',
            'wheel_L_Joint',
            'wheel_R_Joint',
        ]

        # 데이터 버퍼
        N = self.num_motors
        self.temp_arr   = [float('nan')] * N     # °C
        self.iq_amp_arr = [float('nan')] * N     # A
        self.speed_arr  = [float('nan')] * N     # dps
        self.enc_arr    = [0] * N                # raw encoder
        self.st_arr     = [0] * N                # single-turn raw (0.001°/LSB)
        self.mt_arr     = [0] * N                # multi-turn  raw (0.001°/LSB, signed)
        self.err_arr    = [0] * N                # error flags
        self.mstat_arr  = [0] * N                # motor state

        # 타이머: 이제 poll_hz = "모든 모터 한 번씩 읽는 주기(Hz)"
        self.timer = self.create_timer(1.0 / self.poll_hz, self.timer_cb)
        self.get_logger().info(
            f"[MotorStatePublisher] Started with full-scan polling at {self.poll_hz:.1f} Hz "
            f"({self.num_motors} motors, MG only)."
        )

    def destroy_node(self):
        self.get_logger().info("Shutting down CAN bus.")
        try:
            if hasattr(self, 'bus') and self.bus:
                self.bus.shutdown()
        finally:
            super().destroy_node()

    # ---------------- 폴링 함수 (모터 1개 기준) ----------------
    def poll_state2(self, motor_idx: int):
        """0x9C: 상태2 (온도, iq(or power), 속도, 엔코더)"""
        node_id = motor_idx + 1
        rep = self.cmd_read_state2(node_id, timeout=0.05)
        if not rep:
            return
        d = rep.data
        # temp: int8 (°C)
        self.temp_arr[motor_idx] = float(struct.unpack('<b', d[1:2])[0])
        # iq(or power): int16 → A (MG 해상도)
        iq_or_power = struct.unpack('<h', d[2:4])[0]
        self.iq_amp_arr[motor_idx] = float(iq_or_power) * self.iq_res
        # speed: int16 (dps)
        self.speed_arr[motor_idx] = float(struct.unpack('<h', d[4:6])[0])
        # encoder: uint16
        self.enc_arr[motor_idx] = int(struct.unpack('<H', d[6:8])[0])

    def poll_state1(self, motor_idx: int):
        """0x9A: 상태1 (motor_state, error_flags 등)"""
        node_id = motor_idx + 1
        rep = self.cmd_read_state1(node_id, timeout=0.05)
        if not rep:
            return
        d = rep.data
        self.mstat_arr[motor_idx] = int(d[6])
        self.err_arr[motor_idx]   = int(d[7])

    def poll_single_or_multi_turn(self, motor_idx: int):
        """0x94/0x92: 각도 (raw 0.001°/LSB)"""
        node_id = motor_idx + 1
        READ_TO = 0.25

        def read_single():
            rep = self.cmd_read_single_turn(node_id, timeout=READ_TO)
            if not rep:
                return False
            d = rep.data
            val = int.from_bytes(d[4:8], byteorder='little', signed=False)  # 0.001°/LSB
            self.st_arr[motor_idx] = val
            return True

        def read_multi():
            rep = self.cmd_read_multi_turn(node_id, timeout=READ_TO)
            if not rep:
                return False
            d = rep.data
            raw7 = d[1:8]
            # 마지막 바이트의 최상위 비트(0x80)를 sign bit로 사용
            sign = b'\x00' if raw7[-1] < 0x80 else b'\xff'
            s64  = int.from_bytes(raw7 + sign, byteorder='little', signed=True)  # 0.001°/LSB
            self.mt_arr[motor_idx] = s64
            return True

        # 인덱스 세트가 비어 있으면 '둘 다' 읽기
        if not self.single_turn_indices and not self.multi_turn_indices:
            _ = read_single()
            _ = read_multi()
            return

        if motor_idx in self.single_turn_indices:
            _ = read_single()
        if motor_idx in self.multi_turn_indices:
            _ = read_multi()

    # ---------------- Timer & Publish ----------------
    def timer_cb(self):
        """
        한 타이머 주기마다:
        - 모든 모터에 대해 state2 → state1 → angle 순으로 CAN 요청
        - 그 후 MotorStates + JointState 한 번씩 publish
        """
        for i in range(self.num_motors):
            try:
                self.poll_state2(i)
            except Exception as e:
                self.get_logger().warn(f"poll_state2({i}) failed: {e}")
            try:
                self.poll_state1(i)
            except Exception as e:
                self.get_logger().warn(f"poll_state1({i}) failed: {e}")
            try:
                self.poll_single_or_multi_turn(i)
            except Exception as e:
                self.get_logger().warn(f"poll_single_or_multi_turn({i}) failed: {e}")

        self.publish_all()
        self.publish_joint_state()

    def publish_all(self):
        msg = MotorStates()
        try:
            msg.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        msg.temperature_c   = self.temp_arr
        msg.iq_amp          = self.iq_amp_arr
        msg.speed_dps       = self.speed_arr
        msg.encoder_raw     = self.enc_arr
        msg.single_turn_raw = self.st_arr       # 0.001°/LSB
        msg.multi_turn_raw  = self.mt_arr       # 0.001°/LSB (signed)
        msg.error_flags     = self.err_arr
        msg.motor_state     = self.mstat_arr

        self.publisher.publish(msg)

    def publish_joint_state(self):
        js = JointState()
        try:
            js.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        js.name = self.joint_names

        positions = []
        velocities = []
        efforts = []

        for i in range(self.num_motors):
            # 1순위: multi-turn(누적각)  2순위: single-turn  3순위: encoder_raw
            angle_rad = 0.0

            if self.mt_arr[i] != 0:
                # mt_arr: 0.001 deg/LSB
                deg = self.mt_arr[i] * 0.001
                # 무릎 기어비 보정 (예: 1:2)
                if i in [4, 5]:  # knee_L, knee_R
                    angle_rad = deg * math.pi / 90.0
                else:
                    angle_rad = deg * math.pi / 180.0

            positions.append(angle_rad)

            # speed_arr: dps → rad/s
            if isinstance(self.speed_arr[i], float) and not math.isnan(self.speed_arr[i]):
                vel = self.speed_arr[i] * math.pi / 180.0
            else:
                vel = 0.0
            velocities.append(vel)

            # effort: 토크 전류(A)
            if isinstance(self.iq_amp_arr[i], float) and not math.isnan(self.iq_amp_arr[i]):
                eff = self.iq_amp_arr[i]
            else:
                eff = 0.0
            efforts.append(eff)

        js.position = positions
        js.velocity = velocities
        js.effort   = efforts

        self.js_publisher.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = MotorStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
