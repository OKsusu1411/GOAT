#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np

NUM_JOINTS = 8
# === pd_controller.py와 맞추기 ===
JOINT_INDEX = 7
DIRECTION = 1
MOTOR_STATES_TOPIC = 'motor_states'
TORQUE_COMMANDS_TOPIC = 'torque_commands'

# MG 모터 angle scale (0.001 deg/LSB)
ANGLE_LSB_TO_DEG = 0.001

class BreakawayTorqueTester(Node):
    """
    특정 조인트의 '최소 구동 토크(breakaway torque)'를 자동으로 찾는 노드.

    동작 개요:
    1) motor_states를 받아서 현재 각도를 실시간으로 기록
    2) 토크 명령을 0 -> step -> 2*step -> ... 형태로 계단처럼 증가
    3) 각 스텝마다 일정 시간 동안 관절이 안 움직이는지 확인
    4) 처음으로 일정 각도 이상 움직인 스텝의 토크를 breakaway torque로 기록

    - 토크 명령은 torque_commands 토픽으로 발행 (pd_controller와 동일 토픽)
    - breakaway torque를 찾으면 토크를 0으로 돌리고 결과를 로그로 출력
    """

    def __init__(self):
        super().__init__('breakaway_torque_tester')

        # -------- 파라미터 / 설정 --------
        # 몇 번 조인트를 테스트할지 (URDF 인덱스 기준)
        self.joint_index = JOINT_INDEX

        # 토크 스텝 세부 설정
        self.start_torque = float(self.declare_parameter('start_torque', 0.0).value)   # [torque_commands 단위]
        self.torque_step  = float(self.declare_parameter('torque_step', 0.01).value)  # 한 스텝당 증가량
        self.max_torque   = 4.5    # 안전 상한 (torque_commands 단위)

        # 한 스텝을 유지하는 시간
        self.step_duration = float(self.declare_parameter('step_duration', 0.5).value)  # [s]

        # 움직임 판정 기준
        self.angle_threshold_deg = float(
            self.declare_parameter('angle_threshold_deg', 0.5).value
        )  # 기준 각도 [deg]
        self.min_samples_over_threshold = int(
            self.declare_parameter('min_samples_over_threshold', 3).value
        )  # 연속 샘플 수

        # 토크 방향 (+1이면 +방향, -1이면 -방향 실험)
        self.direction = DIRECTION
        if self.direction >= 0:
            self.direction = 1.0
        else:
            self.direction = -1.0

        # 컨트롤 루프 주파수
        self.control_frequency = float(
            self.declare_parameter('control_frequency', 200.0).value
        )
        self.dt = 1.0 / self.control_frequency

        self.get_logger().info(
            f"[BreakawayTester] joint_index={self.joint_index}, "
            f"start_torque={self.start_torque}, step={self.torque_step}, "
            f"max_torque={self.max_torque}, dir={self.direction}"
        )

        # -------- 상태 변수 --------
        self.current_angles_deg = np.zeros(NUM_JOINTS, dtype=float)
        self.have_state = False

        self.state = 'WAIT_FOR_STATE'  # WAIT_FOR_STATE -> START_STEP -> HOLD_STEP -> DONE
        self.current_step_torque = self.start_torque
        self.step_start_time = None
        self.baseline_angle_deg = 0.0
        self.over_threshold_count = 0

        self.break_torque = None
        self.break_time = None

        # -------- ROS 통신 설정 --------
        self.create_subscription(
            MotorStates,
            MOTOR_STATES_TOPIC,
            self.motor_states_callback,
            100
        )

        self.torque_pub = self.create_publisher(
            Float32MultiArray,
            TORQUE_COMMANDS_TOPIC,
            100
        )

        self.timer = self.create_timer(self.dt, self.control_loop)

    # -------------------- 콜백: MotorStates --------------------
    def motor_states_callback(self, msg: MotorStates):
        """
        motor_states로부터 현재 각도를 degree 단위로 업데이트.
        """
        raw_angles_deg = np.array(msg.multi_turn_raw, dtype=float) * ANGLE_LSB_TO_DEG

        if raw_angles_deg.size != NUM_JOINTS:
            self.get_logger().error(
                f"motor_states_callback: received {raw_angles_deg.size} angles, "
                f"expected {NUM_JOINTS}."
            )
            return

        self.current_angles_deg = raw_angles_deg
        self.have_state = True

    # -------------------- 제어 루프 --------------------
    def control_loop(self):
        """
        주기적으로 호출되는 제어 루프.
        상태 머신 기반으로 토크를 조금씩 올리면서 움직임이 발생하는지 검사한다.
        """
        now = self.get_clock().now().nanoseconds / 1e9

        # 아직 상태 데이터가 없으면 아무것도 안 함
        if not self.have_state:
            # 토크 0 유지
            self.publish_torque_cmd(0.0)
            return

        # 상태 머신
        if self.state == 'WAIT_FOR_STATE':
            # 첫 상태를 받았으니 바로 첫 스텝 시작
            self.baseline_angle_deg = float(self.current_angles_deg[self.joint_index])
            self.over_threshold_count = 0
            self.step_start_time = now
            self.current_step_torque = self.start_torque
            self.state = 'START_STEP'
            self.get_logger().info(
                f"[BreakawayTester] Start test. Baseline angle(deg)={self.baseline_angle_deg:.4f}"
            )

        elif self.state == 'START_STEP':
            # 새로운 스텝 시작: 기준각도 리셋
            self.baseline_angle_deg = float(self.current_angles_deg[self.joint_index])
            self.over_threshold_count = 0
            self.step_start_time = now
            self.state = 'HOLD_STEP'
            self.get_logger().info(
                f"[BreakawayTester] Apply torque step: "
                f"{self.direction * self.current_step_torque:.4f}"
            )

        elif self.state == 'HOLD_STEP':
            # 1) 움직임 검사
            cur_angle = float(self.current_angles_deg[self.joint_index])
            diff = abs(cur_angle - self.baseline_angle_deg)

            if diff > self.angle_threshold_deg:
                self.over_threshold_count += 1
            else:
                # 필요하면 연속성 강하게 보려면 reset해도 됨
                pass

            # 움직임이 충분히 관측되면 breakaway torque로 판정
            if self.over_threshold_count >= self.min_samples_over_threshold:
                self.break_torque = self.direction * self.current_step_torque
                self.break_time = now
                self.state = 'DONE'

                self.get_logger().info(
                    f"[BreakawayTester] BREAKAWAY DETECTED! "
                    f"joint {self.joint_index}, "
                    f"tau_break={self.break_torque:.4f} "
                    f"(same unit as torque_commands)."
                )

            # 2) 스텝 유지 시간이 끝났는데도 안 움직였으면 다음 스텝으로
            elif (now - self.step_start_time) >= self.step_duration:
                # 다음 스텝으로 토크 증가
                self.current_step_torque += self.torque_step

                if abs(self.current_step_torque) > abs(self.max_torque):
                    self.get_logger().warn(
                        "[BreakawayTester] Reached max_torque without motion. "
                        "Stop test for safety."
                    )
                    self.state = 'DONE'
                else:
                    self.state = 'START_STEP'

        elif self.state == 'DONE':
            # 테스트 완료: 토크 0 출력만 유지
            self.publish_torque_cmd(0.0)
            return

        # 해당 스텝에서 실제로 줄 토크 값 계산
        if self.state in ['START_STEP', 'HOLD_STEP']:
            tau_cmd_joint = self.direction * self.current_step_torque
        else:
            tau_cmd_joint = 0.0

        self.publish_torque_cmd(tau_cmd_joint)

    # -------------------- 토크 명령 발행 --------------------
    def publish_torque_cmd(self, tau_joint: float):
        """
        원하는 조인트 하나에만 토크를 인가하고,
        나머지는 0으로 두어서 안전하게 테스트한다.
        """
        cmd = np.zeros(NUM_JOINTS, dtype=float)
        cmd[self.joint_index] = tau_joint

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self.torque_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BreakawayTorqueTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
