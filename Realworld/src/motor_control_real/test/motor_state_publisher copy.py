#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from motor_interfaces.msg import MotorStates  # 사용자 정의 메시지 import
import can
import struct
import time


# 모터 시리즈별 전류 스케일 (A/LSB) for 0xA1 명령 및 iq 환산 참고
SCALE_A_PER_LSB = {
    "MF": 16.5 / 2048.0,   # ≈ 0.00806 A/LSB
    "MG": 33.0 / 2048.0,   # ≈ 0.01611 A/LSB
}
# 0x9C 응답에서 iq(또는 출력) 필드가 "iq"일 때의 해상도 (A/LSB)
IQ_RES_A_PER_LSB = {       # 문서 기준: MF=33/4096, MG=66/4096
    "MF": 33.0 / 4096.0,
    "MG": 66.0 / 4096.0,
}

class MotorStatePublisher(Node):
    def __init__(self):
        super().__init__('motor_state_publisher')

        # 파라미터
        self.channel = self.declare_parameter('channel', 'can0').value
        self.interface = self.declare_parameter('interface', 'socketcan').value
        self.series = self.declare_parameter('series', 'MG').value   # 'MF' or 'MG'
        self.num_motors = int(self.declare_parameter('num_motors', 8).value)
        self.poll_hz = float(self.declare_parameter('poll_hz', 100.0).value)
        self.single_turn_indices = set(self.declare_parameter('single_turn_indices', []).value)
        self.multi_turn_indices  = set(self.declare_parameter('multi_turn_indices', []).value)

        # 시리즈별 iq 해상도 선택
        self.iq_res = IQ_RES_A_PER_LSB.get(self.series, IQ_RES_A_PER_LSB['MG'])

        # CAN 버스 열기
        self.get_logger().info(f"Open CAN bus: channel={self.channel}, interface={self.interface}")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            self.get_logger().error(f"Failed to open CAN bus: {e}")
            raise

        # 단일 퍼블리셔 준비
        self.publisher = self.create_publisher(MotorStates, 'motor_states', 10)

        # 데이터 저장용 버퍼 초기화
        N = self.num_motors
        self.temp_arr   = [float('nan')] * N
        self.iq_amp_arr = [float('nan')] * N
        self.speed_arr  = [float('nan')] * N
        self.enc_arr    = [0] * N
        self.st_arr     = [0] * N
        self.mt_arr     = [0] * N
        self.err_arr    = [0] * N
        self.mstat_arr  = [0] * N

        # 라운드 로빈 폴링을 위한 상태 변수
        self.poll_motor_idx = 0
        self.poll_type_idx = 0
        self.poll_sequence = [self.poll_state2, self.poll_state1, self.poll_single_or_multi_turn]

        # 폴링 타이머
        self.timer = self.create_timer(1.0 / self.poll_hz, self.timer_cb)
        self.get_logger().info(f"Motor state publisher started with {self.poll_hz} Hz polling rate.")

    def destroy_node(self):
        self.get_logger().info("Shutting down CAN bus.")
        if self.bus:
            self.bus.shutdown()
        super().destroy_node()

    # ---- CAN 유틸 ----
    def _txrx(self, node_id: int, cmd: int, payload7: bytes=b'\x00'*7, timeout=0.05):
        tx_id = 0x140 + node_id
        rx_id = 0x180 + node_id
        data = bytes([cmd]) + payload7
        try:
            self.bus.send(can.Message(arbitration_id=tx_id, data=data, is_extended_id=False))
        except can.CanError as e:
            self.get_logger().error(f"CAN send failed (node {node_id}, cmd 0x{cmd:02X}): {e}")
            return None

        t_end = time.time() + timeout
        while time.time() < t_end:
            m = self.bus.recv(timeout=0.02)
            if not m or len(m.data) != 8 or m.data[0] != cmd:
                continue
            # 1) 정상 프로토콜 (0x180+ID) 응답
            # if m.arbitration_id == rx_id:
            #     return m
            # 2) 같은 ID(0x140+ID)로 응답하는 경우 허용
            if m.arbitration_id == tx_id:
                # loopback으로 보낸 프레임(요청)과 "동일한 데이터"는 스킵
                if m.data == data:
                    continue
                # 내용이 다르면 실제 응답으로 간주
                return m
        return None


    # ---- 폴링 함수 (타이머에서 순차적으로 호출) ----
    def poll_state2(self, motor_idx):
        node_id = motor_idx + 1
        rep = self._txrx(node_id, 0x9C)
        if rep:
            d = rep.data
            self.temp_arr[motor_idx] = float(struct.unpack('<b', d[1:2])[0])
            iq_or_power = struct.unpack('<h', d[2:4])[0]
            self.iq_amp_arr[motor_idx] = float(iq_or_power) * self.iq_res
            self.speed_arr[motor_idx] = float(struct.unpack('<h', d[4:6])[0])
            self.enc_arr[motor_idx] = int(struct.unpack('<H', d[6:8])[0])

    def poll_state1(self, motor_idx):
        node_id = motor_idx + 1
        rep = self._txrx(node_id, 0x9A)
        if rep:
            self.mstat_arr[motor_idx] = int(rep.data[6])
            self.err_arr[motor_idx] = int(rep.data[7])

    def poll_single_or_multi_turn(self, motor_idx):
        node_id = motor_idx + 1
        if motor_idx in self.single_turn_indices:
            rep = self._txrx(node_id, 0x94)
            if rep:
                d = rep.data
                # circleAngle: uint32 LE at DATA[4..7], unit = 0.01°
                val = int.from_bytes(d[4:8], byteorder='little', signed=False)
                self.st_arr[motor_idx] = val
        elif motor_idx in self.multi_turn_indices:
            rep = self._txrx(node_id, 0x92)
            if rep:
                d = rep.data
                # motorAngle: int64, but only 7 bytes sent at DATA[1..7]
                raw7 = d[1:8]                       # 7 bytes
                sign = b'\x00' if raw7[-1] < 0x80 else b'\xff'
                val = int.from_bytes(raw7 + sign, byteorder='little', signed=True)
                self.mt_arr[motor_idx] = val


    # ---- Timer & Publish ----
    def timer_cb(self):
        poll_function = self.poll_sequence[self.poll_type_idx]
        poll_function(self.poll_motor_idx)

        if self.poll_motor_idx == self.num_motors - 1 and self.poll_type_idx == len(self.poll_sequence) - 1:
            self.publish_all()

        self.poll_type_idx += 1
        if self.poll_type_idx >= len(self.poll_sequence):
            self.poll_type_idx = 0
            self.poll_motor_idx += 1
            if self.poll_motor_idx >= self.num_motors:
                self.poll_motor_idx = 0

    def publish_all(self):
        msg = MotorStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.temperature_c = self.temp_arr
        msg.iq_amp = self.iq_amp_arr
        msg.speed_dps = self.speed_arr
        msg.encoder_raw = self.enc_arr
        msg.single_turn_raw = self.st_arr
        msg.multi_turn_raw = self.mt_arr
        msg.error_flags = self.err_arr
        msg.motor_state = self.mstat_arr
        
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
