#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from motor_interfaces.msg import MotorStates
from sensor_msgs.msg import JointState          # === JointState 추가 ===
import can
import struct
import time
import math                                     # === JointState 추가 ===

"""
motor_state_publisher.py (MG 전용, read_angle_tx_only.py 방식 각도 안정화)
- 기존 동작(0x9C/0x9A 등)은 그대로 유지
- 각도 읽기(0x94: single-turn / 0x92: multi-turn)만 tx-only 에코 필터 적용
- MF 관련 로직/스케일 제거 (MG만 사용)
"""

# === MG 전용 iq 해상도 (A/LSB) ===
IQ_RES_A_PER_LSB = 66.0 / 4096.0   # MG: ≈ 0.01611 A/LSB

class MotorStatePublisher(Node):
    def __init__(self):
        super().__init__('motor_state_publisher')

        # 파라미터
        self.channel   = self.declare_parameter('channel', 'can0').value
        self.interface = self.declare_parameter('interface', 'socketcan').value
        self.num_motors = int(self.declare_parameter('num_motors', 8).value)
        self.poll_hz    = float(self.declare_parameter('poll_hz', 100.0).value)
        # 각도 읽을 모터 인덱스(0-based) 집합
        self.single_turn_indices = set(self.declare_parameter('single_turn_indices', []).value)
        self.multi_turn_indices  = set(self.declare_parameter('multi_turn_indices', []).value)

        # MG 해상도 고정
        self.iq_res = IQ_RES_A_PER_LSB

        # CAN 버스 열기
        self.get_logger().info(f"Open CAN bus: channel={self.channel}, interface={self.interface}")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            self.get_logger().error(f"Failed to open CAN bus: {e}")
            raise

        # 퍼블리셔
        self.publisher = self.create_publisher(MotorStates, 'motor_states', 10)
        # === JointState 퍼블리셔 추가 ===
        # ROS 관습상 토픽 이름은 'joint_states'를 사용
        self.js_publisher = self.create_publisher(JointState, 'joint_states', 10)
        # joint name: joint_1, joint_2, ...
        self.joint_names = [f"joint_{i+1}" for i in range(self.num_motors)]
        # =================================

        # 데이터 버퍼
        N = self.num_motors
        self.temp_arr   = [float('nan')] * N     # °C
        self.iq_amp_arr = [float('nan')] * N     # A
        self.speed_arr  = [float('nan')] * N     # dps (문서 기준)
        self.enc_arr    = [0] * N                # raw encoder (0x9C[6..7])
        self.st_arr     = [0] * N                # single-turn raw (0.01°/LSB)
        self.mt_arr     = [0] * N                # multi-turn  raw (0.01°/LSB, signed)
        self.err_arr    = [0] * N                # error flags (0x9A[7])
        self.mstat_arr  = [0] * N                # motor state  (0x9A[6])

        # 라운드 로빈 폴링 상태
        self.poll_motor_idx = 0
        self.poll_type_idx  = 0
        self.poll_sequence  = [self.poll_state2, self.poll_state1, self.poll_single_or_multi_turn]

        # 타이머
        self.timer = self.create_timer(1.0 / self.poll_hz, self.timer_cb)
        self.get_logger().info(f"Motor state publisher started with {self.poll_hz:.1f} Hz polling rate (MG only).")

    def destroy_node(self):
        self.get_logger().info("Shutting down CAN bus.")
        try:
            if hasattr(self, 'bus') and self.bus:
                self.bus.shutdown()
        finally:
            super().destroy_node()

    # ---------------- CAN 유틸 (CanMixin 대체) ----------------
    def _txrx(self, node_id: int, cmd: int, payload7: bytes=b'\x00'*7,
              timeout: float = 0.05, accept_rx: bool=False, verbose: bool=False):
        """
        read_angle_tx_only.py와 동일한 TX-echo 무시 방식.
        - TX ID(0x140+ID)로 되돌아오는 에코 프레임은 '내용이 같으면' 무시
        - 같은 TX ID라도 '내용이 다른' 프레임은 유효 응답으로 인정
        - accept_rx=True이면 RX ID(0x180+ID) 응답도 허용
        """
        tx_id = 0x140 + int(node_id)
        rx_id = 0x180 + int(node_id)
        data  = bytes([cmd]) + payload7

        try:
            self.bus.send(can.Message(arbitration_id=tx_id, data=data, is_extended_id=False))
        except can.CanError as e:
            self.get_logger().error(f"CAN send failed (node {node_id}, cmd 0x{cmd:02X}): {e}")
            return None

        t_end = time.time() + timeout
        while time.time() < t_end:
            m = self.bus.recv(timeout=min(0.05, timeout))
            if not m or len(m.data) != 8 or m.data[0] != cmd:
                continue

            if accept_rx and m.arbitration_id == rx_id:
                if verbose:
                    self.get_logger().info(f"[RX] 0x{m.arbitration_id:X} <- 0x{cmd:02X} {m.data.hex(' ').upper()}")
                return m

            if m.arbitration_id == tx_id and m.data != data:
                if verbose:
                    self.get_logger().info(f"[RX-TXID] 0x{m.arbitration_id:X} <- 0x{cmd:02X} {m.data.hex(' ').upper()}")
                return m
        return None

    # ---- 개별 커맨드 래퍼 (기존 CanMixin 대체) ----
    def cmd_read_state2(self, node_id: int, cmd: int = 0x9C, timeout: float = 0.05):
        return self._txrx(node_id, cmd, timeout=timeout)

    def cmd_read_state1(self, node_id: int, cmd: int = 0x9A, timeout: float = 0.05):
        return self._txrx(node_id, cmd, timeout=timeout)

    def cmd_read_single_turn(self, node_id: int, timeout: float = 0.25):
        return self._txrx(node_id, 0x94, timeout=timeout)

    def cmd_read_multi_turn(self, node_id: int, timeout: float = 0.25):
        return self._txrx(node_id, 0x92, timeout=timeout)

    # ---------------- 폴링 함수 (라운드 로빈) ----------------
    def poll_state2(self, motor_idx: int):
        """0x9C: 상태2 (온도, iq(or power), 속도, 엔코더)"""
        node_id = motor_idx + 1
        rep = self.cmd_read_state2(node_id, 0x9C)
        if not rep:
            return
        d = rep.data
        # temp: int8 (°C)
        self.temp_arr[motor_idx] = float(struct.unpack('<b', d[1:2])[0])
        # iq(or power): int16 → A (MG 해상도)
        iq_or_power = struct.unpack('<h', d[2:4])[0]
        self.iq_amp_arr[motor_idx] = float(iq_or_power) * self.iq_res
        # speed: int16 (dps, 문서 표기)
        self.speed_arr[motor_idx] = float(struct.unpack('<h', d[4:6])[0])
        # encoder: uint16
        self.enc_arr[motor_idx] = int(struct.unpack('<H', d[6:8])[0])

    def poll_state1(self, motor_idx: int):
        """0x9A: 상태1 (motor_state, error_flags 등)"""
        node_id = motor_idx + 1
        rep = self.cmd_read_state1(node_id)
        if not rep:
            return
        d = rep.data
        self.mstat_arr[motor_idx] = int(d[6])
        self.err_arr[motor_idx]   = int(d[7])

    def poll_single_or_multi_turn(self, motor_idx: int):
        """0x94/0x92: 각도 (raw 0.01°/LSB)만 안정적으로 읽기"""
        node_id = motor_idx + 1
        READ_TO = 0.25  # read_angle_tx_only.py 기본값과 동일

        def read_single():
            rep = self.cmd_read_single_turn(node_id, timeout=READ_TO)
            if not rep:
                return False
            d = rep.data
            val = int.from_bytes(d[4:8], byteorder='little', signed=False)  # 0.01°/LSB
            self.st_arr[motor_idx] = val
            return True

        def read_multi():
            rep = self.cmd_read_multi_turn(node_id, timeout=READ_TO)
            if not rep:
                return False
            d = rep.data
            raw7 = d[1:8]
            sign = b'\x00' if raw7[-1] < 0.80 else b'\xff'
            s64  = int.from_bytes(raw7 + sign, byteorder='little', signed=True)  # 0.01°/LSB
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
        # 현재 모터에 대해 현재 폴링 종류 실행
        self.poll_sequence[self.poll_type_idx](self.poll_motor_idx)

        # 한 바퀴 끝이면 퍼블리시
        if self.poll_motor_idx == self.num_motors - 1 and self.poll_type_idx == len(self.poll_sequence) - 1:
            self.publish_all()
            self.publish_joint_state()   # === JointState도 함께 퍼블리시 ===

        # 다음 폴링 종류로
        self.poll_type_idx += 1
        if self.poll_type_idx >= len(self.poll_sequence):
            self.poll_type_idx = 0
            # 다음 모터로
            self.poll_motor_idx += 1
            if self.poll_motor_idx >= self.num_motors:
                self.poll_motor_idx = 0

    def publish_all(self):
        msg = MotorStates()
        try:
            msg.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        try: msg.temperature_c   = self.temp_arr
        except Exception: pass
        try: msg.iq_amp          = self.iq_amp_arr
        except Exception: pass
        try: msg.speed_dps       = self.speed_arr
        except Exception: pass
        try: msg.encoder_raw     = self.enc_arr
        except Exception: pass
        try: msg.single_turn_raw = self.st_arr       # 0.01°/LSB
        except Exception: pass
        try: msg.multi_turn_raw  = self.mt_arr       # 0.01°/LSB (signed)
        except Exception: pass
        try: msg.error_flags     = self.err_arr
        except Exception: pass
        try: msg.motor_state     = self.mstat_arr
        except Exception: pass

        self.publisher.publish(msg)

    # === JointState 퍼블리시 함수 추가 ===
    def publish_joint_state(self):
        js = JointState()
        try:
            js.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        # joint name 세팅
        js.name = self.joint_names

        positions = []
        velocities = []
        efforts = []

        for i in range(self.num_motors):
            # 1순위: multi-turn (누적각)  2순위: single-turn  3순위: encoder_raw
            angle_rad = 0.0

            if self.mt_arr[i] != 0:
                # mt_arr: 0.01 deg/LSB
                deg = self.mt_arr[i] * 0.01
                angle_rad = deg * math.pi / 180.0
            elif self.st_arr[i] != 0:
                # st_arr: 0.01 deg/LSB
                deg = self.st_arr[i] * 0.01
                angle_rad = deg * math.pi / 180.0
            else:
                # encoder_raw: 0~(2^N-1), N=14/15/16 중 하나
                # 여기서는 대략 16bit로 가정해서 0~2π로 매핑 (필요하면 나중에 수정)
                angle_rad = (self.enc_arr[i] / 65535.0) * 2.0 * math.pi

            positions.append(angle_rad)

            # speed_arr: dps → rad/s
            if isinstance(self.speed_arr[i], float) and not math.isnan(self.speed_arr[i]):
                vel = self.speed_arr[i] * math.pi / 180.0
            else:
                vel = 0.0
            velocities.append(vel)

            # effort에는 토크 전류(A) 그대로 넣어둠 (필요하면 토크로 변환해서 사용)
            if isinstance(self.iq_amp_arr[i], float) and not math.isnan(self.iq_amp_arr[i]):
                eff = self.iq_amp_arr[i]
            else:
                eff = 0.0
            efforts.append(eff)

        js.position = positions
        js.velocity = velocities
        js.effort   = efforts

        self.js_publisher.publish(js)
    # ==================================

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
