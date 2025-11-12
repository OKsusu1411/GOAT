#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from motor_interfaces.msg import MotorStates  # 사용자 정의 메시지
import can
import struct
import time

# 0x9C 응답에서 iq(또는 출력) 필드가 "iq"일 때의 해상도 (A/LSB)
IQ_RES_A_PER_LSB = {       # 매뉴얼: MF=33/4096 A/LSB, MG=66/4096 A/LSB
    "MF": 33.0 / 4096.0,
    "MG": 66.0 / 4096.0,
}

class MotorStatePublisher(Node):
    def __init__(self):
        super().__init__('motor_state_publisher')

        # ----- 파라미터 -----
        self.channel = self.declare_parameter('channel', 'can0').value
        self.interface = self.declare_parameter('interface', 'socketcan').value
        self.series = self.declare_parameter('series', 'MG').value   # 'MF' or 'MG'
        self.num_motors = int(self.declare_parameter('num_motors', 8).value)
        self.poll_hz = float(self.declare_parameter('poll_hz', 20.0).value)

        # 시리즈별 iq 해상도
        self.iq_res = IQ_RES_A_PER_LSB.get(self.series, IQ_RES_A_PER_LSB['MG'])

        # ----- CAN 오픈 -----
        self.get_logger().info(f"Open CAN bus: channel={self.channel}, interface={self.interface}")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            self.get_logger().error(f"Failed to open CAN bus: {e}")
            raise

        # 퍼블리셔
        self.publisher = self.create_publisher(MotorStates, 'motor_states', 10)

        # ----- 버퍼 초기화 -----
        N = self.num_motors
        self.temp_arr   = [float('nan')] * N     # °C
        self.iq_amp_arr = [float('nan')] * N     # A
        self.speed_arr  = [float('nan')] * N     # dps
        self.enc_arr    = [0] * N                # raw 0..65535
        # -1은 아직 데이터가 수신되지 않았음을 의미하는 센티널 값입니다.
        self.st_arr     = [-1] * N               # single-turn angle (0.01 deg units)
        self.mt_arr     = [-1] * N               # multi-turn angle (0.01 deg units)
        self.err_arr    = [0] * N
        self.mstat_arr  = [0] * N

        # 송신 에코 필터를 위한 “최근 보낸 프레임 집합”
        self._last_tx_frames = set()

        # 폴링 진행 상태
        self.curr_motor_idx = 0

        # 타이머 시작
        self.timer = self.create_timer(1.0 / self.poll_hz, self.timer_cb)
        self.get_logger().info(f"Motor state publisher started with {self.poll_hz} Hz.")

    def destroy_node(self):
        self.get_logger().info("Shutting down CAN bus.")
        if hasattr(self, 'bus') and self.bus:
            self.bus.shutdown()
        super().destroy_node()

    # =========================
    # CAN 송수신 유틸
    # =========================
    def _send(self, node_id: int, cmd: int, payload7: bytes=b'\x00'*7):
        """TX-ID(0x140+ID)로 8바이트 프레임 송신하고, 에코 필터를 위해 보낸 프레임을 기록."""
        tx_id = 0x140 + node_id
        data = bytes([cmd]) + payload7
        msg = can.Message(arbitration_id=tx_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
            # 에코 필터용으로 key 저장 (ID+data)
            self._last_tx_frames.add((tx_id, data))
        except can.CanError as e:
            self.get_logger().error(f"CAN send failed (node {node_id}, cmd 0x{cmd:02X}): {e}")

    def _recv_route_for_node(self, node_id: int, recv_window: float = 0.02):
        """
        일정 시간 동안 해당 node_id의 TX-ID 수신 프레임을 읽어서
        DATA[0] = cmd 바이트로 라우팅/파싱한다.
        - TX 에코(보낸 것과 동일한 프레임)는 무시
        - 필요한 명령(0x9C, 0x9A, 0x92, 0x94)만 파싱
        """
        tx_id = 0x140 + node_id
        end_t = time.time() + recv_window
        needed_cmds = {0x9C, 0x9A, 0x92, 0x94}

        while time.time() < end_t:
            m = self.bus.recv(timeout=0.005)
            if not m or m.arbitration_id != tx_id or len(m.data) != 8:
                continue
            # 에코 프레임 무시
            key = (m.arbitration_id, bytes(m.data))
            if key in self._last_tx_frames:
                # 한번 인식했으면 바로 지워서 set이 무한히 커지지 않도록
                self._last_tx_frames.discard(key)
                continue

            cmd = m.data[0]
            if cmd not in needed_cmds:
                continue  # 우리가 관심없는 프레임

            # ==== 라우팅 & 파싱 ====
            idx = node_id - 1
            d = m.data

            if cmd == 0x9C:
                # 상태2: temp(i8), iq(int16) or power, speed(int16), encoder(uint16)
                self.temp_arr[idx] = float(struct.unpack('<b', d[1:2])[0])
                iq_or_power = struct.unpack('<h', d[2:4])[0]
                self.iq_amp_arr[idx] = float(iq_or_power) * self.iq_res
                self.speed_arr[idx] = float(struct.unpack('<h', d[4:6])[0])
                self.enc_arr[idx] = int(struct.unpack('<H', d[6:8])[0])

            elif cmd == 0x9A:
                # 상태1: [6]=motorState, [7]=errorFlags
                self.mstat_arr[idx] = int(d[6])
                self.err_arr[idx] = int(d[7])

            elif cmd == 0x94:
                # 단일턴: DATA[4..7] = uint32, 매뉴얼 단위 0.01 degree
                raw = int.from_bytes(d[4:8], byteorder='little', signed=False)
                self.st_arr[idx] = raw  # 0.01 deg units

            elif cmd == 0x92:
                # 멀티턴: DATA[1..7] = int64 하위 7바이트, 0.01 degree
                raw7 = d[1:8]
                sign = b'\x00' if raw7[-1] < 0x80 else b'\xff'
                raw = int.from_bytes(raw7 + sign, byteorder='little', signed=True)
                self.mt_arr[idx] = raw  # 0.01 deg units

    # =========================
    # 주기 루프
    # =========================
    def timer_cb(self):
        i = self.curr_motor_idx
        node_id = i + 1

        # 이 모터에 대해 필요한 명령들을 모두 송신
        self._send(node_id, 0x9C)  # 상태2
        self._send(node_id, 0x9A)  # 상태1
        self._send(node_id, 0x94)  # 단일턴 각도
        self._send(node_id, 0x92)  # 멀티턴 각도

        # 짧은 수신 윈도우 동안 들어오는 응답을 명령 바이트로 라우팅/파싱
        self._recv_route_for_node(node_id, recv_window=0.02)

        # 라운드 종료 시 퍼블리시(모든 모터 한 바퀴가 끝난 시점)
        if self.curr_motor_idx == self.num_motors - 1:
            self.publish_all()

        # 다음 모터
        self.curr_motor_idx += 1
        if self.curr_motor_idx >= self.num_motors:
            self.curr_motor_idx = 0

    def publish_all(self):
        msg = MotorStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.temperature_c   = self.temp_arr
        msg.iq_amp          = self.iq_amp_arr
        msg.speed_dps       = self.speed_arr
        msg.encoder_raw     = self.enc_arr
        msg.single_turn_raw = self.st_arr     # 0.01 deg units
        msg.multi_turn_raw  = self.mt_arr     # 0.01 deg units
        msg.error_flags     = self.err_arr
        msg.motor_state     = self.mstat_arr
        self.publisher.publish(msg)

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
