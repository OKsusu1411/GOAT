#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np
import time
import os

NUM_JOINTS = 8
TARGET_TOPIC = 'target_joint_angles'   # pd_controller가 구독
MOTOR_STATES_TOPIC = 'motor_states'
TORQUE_COMMANDS_TOPIC = 'torque_commands'

ANGLE_LSB_TO_DEG = 0.001  # multi_turn_raw -> deg

class DynamicFrictionSysID(Node):
    """
    1개 조인트에 사인 궤적을 인가하면서,
    같은 루프에서 motor_states와 torque_commands를 함께 로깅해서
    CSV로 저장하는 SysID용 노드.

    - q_ref(t) = offset + A * sin(2π f t)
    - 나머지 조인트는 0도 유지
    - 주기마다 [t, q_ref, q, dq, tau]를 기록
    - duration초가 지나면 자동으로 CSV 저장 후 종료
    """

    def __init__(self):
        super().__init__('dynamic_friction_sysid')

        # ====== 파라미터 ======
        self.joint_index = int(self.declare_parameter('joint_index', 1).value)

        # 사인 파라미터 (deg, Hz)
        self.amplitude_deg = float(self.declare_parameter('amplitude_deg', 20.0).value)
        self.frequency_hz = float(self.declare_parameter('frequency_hz', 0.3).value)
        self.offset_deg = float(self.declare_parameter('offset_deg', 0.0).value)

        # 컨트롤 주파수
        self.control_frequency = float(
            self.declare_parameter('control_frequency', 200.0).value
        )
        self.dt = 1.0 / self.control_frequency

        # 실험 시간 (초)
        self.duration = float(self.declare_parameter('duration', 200.0).value)

        # 로그 저장 경로
        default_filename = f"sysid_joint{self.joint_index}_{int(time.time())}.csv"
        self.save_path = str(
            self.declare_parameter('save_path', f'./test_log/{default_filename}').value
        )

        self.get_logger().info(
            f"[DynSysID] joint_index={self.joint_index}, "
            f"A={self.amplitude_deg} deg, f={self.frequency_hz} Hz, "
            f"offset={self.offset_deg} deg, duration={self.duration} s, "
            f"save_path={self.save_path}"
        )

        # ====== 상태 변수 ======
        self.current_angles_deg = np.zeros(NUM_JOINTS, dtype=float)
        self.current_speed_dps = np.zeros(NUM_JOINTS, dtype=float)
        self.have_state = False

        self.last_tau_cmd = np.zeros(NUM_JOINTS, dtype=float)
        self.have_tau = False

        self.t0 = time.time()
        self.finished = False

        # 로깅용 버퍼: [t, q_ref, q_meas, dq_meas, tau_cmd]
        self.log_data = []

        # ====== ROS 통신 ======
        self.pub_target = self.create_publisher(
            Float32MultiArray,
            TARGET_TOPIC,
            10
        )
        self.sub_states = self.create_subscription(
            MotorStates,
            MOTOR_STATES_TOPIC,
            self.motor_states_callback,
            100
        )
        self.sub_tau = self.create_subscription(
            Float32MultiArray,
            TORQUE_COMMANDS_TOPIC,
            self.torque_commands_callback,
            100
        )

        self.timer = self.create_timer(self.dt, self.timer_cb)

    # -------- 콜백: MotorStates --------
    def motor_states_callback(self, msg: MotorStates):
        """
        MotorStates에서 각도/속도 업데이트.
        """
        angles_deg = np.array(msg.multi_turn_raw, dtype=float) * ANGLE_LSB_TO_DEG

        # speed_dps 필드가 있다고 가정 (없으면 angle로부터 나중에 미분)
        if hasattr(msg, 'speed_dps'):
            speed_dps = np.array(msg.speed_dps, dtype=float)
        else:
            # 속도 정보가 없다면 0으로 두고, 나중에 angle 미분해서 쓰는 식으로 대체 가능
            speed_dps = np.zeros_like(angles_deg)

        if angles_deg.size != NUM_JOINTS:
            self.get_logger().error(
                f"motor_states: got {angles_deg.size} angles, expected {NUM_JOINTS}"
            )
            return

        self.current_angles_deg = angles_deg
        self.current_speed_dps = speed_dps
        self.have_state = True

    # -------- 콜백: torque_commands --------
    def torque_commands_callback(self, msg: Float32MultiArray):
        """
        torque_commands에서 마지막 토크 명령을 저장.
        """
        data = np.array(msg.data, dtype=float)
        if data.size != NUM_JOINTS:
            # 필요하면 여기서도 사이즈 체크
            self.get_logger().warn(
                f"torque_commands: got {data.size} cmds, expected {NUM_JOINTS}"
            )
        # 길이 맞춰서 저장
        if data.size < NUM_JOINTS:
            padded = np.zeros(NUM_JOINTS, dtype=float)
            padded[:data.size] = data
            data = padded
        elif data.size > NUM_JOINTS:
            data = data[:NUM_JOINTS]

        self.last_tau_cmd = data
        self.have_tau = True

    # -------- 메인 타이머 루프 --------
    def timer_cb(self):
        if self.finished:
            # 이미 실험 종료 후면 아무 것도 안 함
            return

        now = time.time()
        t = now - self.t0

        # 1) duration 지나면 종료
        if t >= self.duration:
            self.get_logger().info(
                f"[DynSysID] duration {self.duration}s reached. Saving log and stopping..."
            )
            self.finish_and_save()
            self.finished = True
            # rclpy.shutdown()은 main()에서 처리
            return

        # 2) 사인 궤적 생성 및 퍼블리시
        q_ref = self.offset_deg + self.amplitude_deg * np.sin(2 * np.pi * self.frequency_hz * t)

        targets = [0.0] * NUM_JOINTS
        targets[self.joint_index] = float(q_ref)

        msg = Float32MultiArray()
        msg.data = targets
        self.pub_target.publish(msg)

        # 3) 로깅 (state/torque 둘 다 있을 때만)
        if self.have_state and self.have_tau:
            q_meas = float(self.current_angles_deg[self.joint_index])
            dq_meas = float(self.current_speed_dps[self.joint_index])
            tau = float(self.last_tau_cmd[self.joint_index])

            self.log_data.append([t, q_ref, q_meas, dq_meas, tau])

    # -------- 종료 & 저장 --------
    def finish_and_save(self):
        if len(self.log_data) == 0:
            self.get_logger().warn("[DynSysID] No data logged. Skip saving.")
            return

        data_arr = np.array(self.log_data, dtype=float)

        # 디렉토리 없으면 생성
        save_dir = os.path.dirname(self.save_path)
        if save_dir != '' and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        header = "t, q_ref_deg, q_meas_deg, dq_meas_deg_per_s, tau_cmd"
        np.savetxt(self.save_path, data_arr, delimiter=",", header=header, comments='')

        self.get_logger().info(
            f"[DynSysID] Log saved to {self.save_path} "
            f"(rows={data_arr.shape[0]})."
        )


def main(args=None):
    rclpy.init(args=args)
    node = DynamicFrictionSysID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 혹시 duration 전에 종료해도 남은 로그 저장
        if not node.finished:
            node.get_logger().info("[DynSysID] Interrupted. Saving partial log...")
            node.finish_and_save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
