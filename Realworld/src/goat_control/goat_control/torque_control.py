import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from .utils.can_mixin import CanMixin
import struct
import time
#from utils.can_mixin import CanMixin

# 모터 시리즈별 전류 -> LSB 변환 비율 (데이터시트 기반)
# SCALE_A_PER_LSB = {
#     "MG": 33.0 / 2048.0,   # MG 시리즈: 약 0.01611 A/LSB (±33.0A 범위)
# }
SCALE_A_PER_LSB = 66.0 / 4096.0   # MG: ≈ 0.01611 A/LSB

class MotorTorqueController(Node, CanMixin):
    def __init__(self):
        super().__init__('motor_torque_controller')
        # 파라미터 선언 및 가져오기
        self.channel = self.declare_parameter('channel', 'can0').value
        self.bitrate = self.declare_parameter('bitrate', 1000000).value
        self.interface = self.declare_parameter('interface', 'socketcan').value                 # CAN 인터페이스 유형
        #self.series = self.declare_parameter('series', 'MG').value
        self.num_motors = self.declare_parameter('num_motors', 8).value
        self.control_frequency = self.declare_parameter('control_frequency', 50.0).value
        self.timeout_sec = self.declare_parameter('timeout_sec', 0.5).value
        self.scale = SCALE_A_PER_LSB                                                            # 모터 시리즈에 따른 전류 스케일 설정 

        # CAN 버스 열기
        # 참고: 이 노드를 실행하기 전에 사용자가 직접 CAN 인터페이스를 활성화해야 합니다.
        # 예: sudo ip link set can0 up type can bitrate 1000000
        self.get_logger().info(f"Attempting to open CAN bus on channel '{self.channel}' with interface '{self.interface}'...")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            self.get_logger().error(f"Failed to open CAN bus on {self.channel}: {e}")
            raise e

        # 모터별 초기화: 오류 클리어 및 구동 활성화 명령 전송
        for node_id in range(1, self.num_motors + 1):
            self._send_command_expect(node_id, 0x9B)      # CLEAR ERRORS (0x9B)
            response = self._send_command_expect(node_id, 0x88)  # RUN (Enable motor, 0x88)
            if response:
                self.get_logger().info(f"Motor {node_id:02d}: RUN command acknowledged.")
            else:
                self.get_logger().warn(f"Motor {node_id:02d}: No response to RUN command.")

        # 현재 명령 저장용 변수 및 상태 플래그 초기화
        self.current_commands = [0.0] * self.num_motors
        self.last_command_time = None
        self.got_command = False
        self.safe_mode = False

        # 토크 명령 토픽 구독 (Float32MultiArray 형식)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'torque_commands',   # 토픽 이름 (필요시 변경 가능)
            self.command_callback,
            10
        )

        # 제어 루프용 타이머 생성 (지속적 주기적 명령 전송)
        timer_period = 1.0 / self.control_frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _send_command_expect(self, node_id: int, cmd_byte: int, payload7: bytes = b"\x00" * 7):
        """지정한 CAN 명령을 보내고 짧은 시간 응답을 대기하는 함수"""
        tx_id = 0x140 + node_id
        rx_id = 0x180 + node_id
        data = bytes([cmd_byte]) + payload7
        msg = can.Message(arbitration_id=tx_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
        except can.CanError as e:
            self.get_logger().error(f"CAN send failed for ID {node_id}: {e}")
            return None
        # 최대 0.3초 대기하여 응답 프레임 수신 확인
        end_time = time.time() + 0.3
        while time.time() < end_time:
            rx_msg = self.bus.recv(timeout=0.1)
            if rx_msg is None:
                continue
            if rx_msg.arbitration_id == tx_id and len(rx_msg.data) == 8 and rx_msg.data[0] == cmd_byte:
                return rx_msg
        return None

    def command_callback(self, msg: Float32MultiArray):
        # 새로운 명령 메시지 수신 시 콜백
        # 수신된 Float32MultiArray에서 명령 값 리스트 추출
        commands = list(msg.data)
        # 모터 개수에 맞춰 리스트 크기 조정 (부족하면 0으로 패딩, 많으면 잘라냄)
        if len(commands) < self.num_motors:
            commands.extend([0.0] * (self.num_motors - len(commands)))
        elif len(commands) > self.num_motors:
            commands = commands[:self.num_motors]
        # 현재 명령과 시간 업데이트
        self.current_commands = commands
        self.last_command_time = time.time()
        self.got_command = True
        # 안전 모드였다면 해제
        if self.safe_mode:
            self.get_logger().info("Received new commands. Exiting safe mode.")
            self.safe_mode = False

    def timer_callback(self):
        # 주기적으로 호출되어 모터에 CAN 명령을 보내는 타이머 콜백
        now = time.time()
        # 마지막 명령 수신 이후 timeout_sec 경과 시 안전 모드 진입
        if self.got_command and (now - self.last_command_time > self.timeout_sec):
            if not self.safe_mode:
                # 안전 모드 진입 (최초 1회만 실행)
                self.get_logger().warn(f"No command received for {self.timeout_sec:.2f} seconds. Entering safe mode (sending zero torque).")
                self.current_commands = [0.0] * self.num_motors  # 모든 모터 0A 명령
                self.safe_mode = True

        # 각 모터에 대한 토크(CAN 현재 명령) 전송
        for i in range(self.num_motors):
            node_id = i + 1  # 모터 ID (리스트 인덱스 0->ID1, 1->ID2, ...)
            amps = self.current_commands[i]
            # 전류(A)를 기기 명령 단위로 변환 및 한계값 클램핑
            # iq = int(round(amps / self.scale))
            # if iq > 2048:
            #     iq = 2048
            # if iq < -2048:
            #     iq = -2048
            # # CAN 데이터 프레임 생성 (명령 0xA1 + 7바이트 payload 구성)
            # iq_bytes = struct.pack("<h", iq)  # 2바이트 (little-endian)로 변환
            # data = bytes([0xA1]) + b"\x00\x00\x00" + iq_bytes + b"\x00\x00"
            # msg = can.Message(arbitration_id=(0x140 + node_id), data=data, is_extended_id=False)
            # try:
            #     self.bus.send(msg)
            # except can.CanError as e:
            #     self.get_logger().error(f"Failed to send torque to motor {node_id}: {e}")
            # CanMixin이 제공하는 iq 포장 + TX/RX를 사용
            resp = self.cmd_torque_mode(node_id, amps, timeout=0.02)
            if not resp:
                # 필요시 더 자세한 로그
                self._log().debug(f"[CAN] torque cmd no resp (id={node_id}, A={amps:.3f})")
        # (옵션) 주기적으로 상태 읽기 (디버그용)
        # ### ADDED: 예시 — 상태 읽기 또한 CanMixin 래퍼 사용
        # state = self.cmd_read_state2(node_id=1, timeout=0.02)
        # if state:
        #     self.get_logger().debug(f"state2: {state.data.hex(' ')}")
    def destroy_node(self):
        # 노드가 종료될 때 CAN 버스 정리
        self.get_logger().info("Shutting down CAN bus.")
        if self.bus:
            self.bus.shutdown()
        super().destroy_node()
__main__ = '__main__'
def main(args=None):
    rclpy.init(args=args)
    node = MotorTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()