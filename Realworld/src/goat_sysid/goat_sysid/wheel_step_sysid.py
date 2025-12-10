#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np
import time
import os

# --- 상수 설정 ---
NUM_JOINTS = 8
WHEEL_INDEX = 7  # 휠 조인트 index 고정
MOTOR_STATES_TOPIC = 'motor_states'
TORQUE_COMMANDS_TOPIC = 'torque_commands'
TARGET_TOPIC = 'target_joint_angles'  # PD 컨트롤러가 구독하는 토픽


class WheelStepSysID(Node):
    """
    휠 속도 SysID용 스텝 스윕 노드.

    - WHEEL_INDEX(7번) 조인트에 대해,
      v_ref(속도 참조, deg/s)를 step 형태로 바꿔가며 인가.
    - PD 컨트롤러는 각도 명령만 받으므로,
      v_ref 를 dt로 적분해서 q_ref[deg]를 만들어
      target_joint_angles (길이 8)로 퍼블리시.
    - motor_states.speed_dps 는 LSB 이므로 0.01을 곱해 deg/s로 사용.
    - 매 루프마다: [t, step_index, v_ref_dps, v_meas_dps, tau_cmd] 로깅.
    - 모든 스텝이 끝나면 CSV 저장 후 자동 종료.
    """

    def __init__(self):
        super().__init__('wheel_step_sysid')

        # ===== 파라미터 =====
        # 스텝 속도 리스트 [deg/s]
        default_step_speeds = [
            0.0,
            200.0, 400.0,
            600.0, 400.0, 200.0, 0.0,
            -200.0, -400.0, 
            -600.0, -400.0, -200.0, 0.0,
        ]
        step_param = self.declare_parameter(
            'step_speeds_dps', default_step_speeds
        ).value
        self.step_speeds_dps = [float(v) for v in step_param]
        self.num_steps = len(self.step_speeds_dps)

        # 각 스텝 유지 시간 [s]
        self.step_duration = float(
            self.declare_parameter('step_duration', 3.0).value
        )

        # 제어 주기/주파수
        self.control_frequency = float(
            self.declare_parameter('control_frequency', 200.0).value
        )
        self.dt_nominal = 1.0 / self.control_frequency

        # 로그 파일 경로
        timestamp = int(time.time())
        default_filename = f"wheel_sysid_w{WHEEL_INDEX}_{timestamp}.csv"
        self.save_path = str(
            self.declare_parameter('save_path', f'./test_log/{default_filename}').value
        )

        self.get_logger().info(
            f"[WheelStepSysID] wheel_index={WHEEL_INDEX}, "
            f"step_speeds_dps={self.step_speeds_dps}, "
            f"step_duration={self.step_duration}s, "
            f"control_freq={self.control_frequency}Hz, "
            f"save_path={self.save_path}"
        )

        # ===== 상태 변수 =====
        # 현재 휠 속도 (deg/s)
        self.current_speed_dps = np.zeros(NUM_JOINTS, dtype=float)
        self.have_state = False

        # 현재 토크 명령
        self.last_tau_cmd = np.zeros(NUM_JOINTS, dtype=float)
        self.have_tau = False

        # 휠 각도 참조 [deg] – v_ref 적분해서 업데이트
        self.ref_angles_deg = np.zeros(NUM_JOINTS, dtype=float)

        self.t0 = time.time()
        self.last_time = self.t0
        self.finished = False

        # 스텝 진행 상태
        self.current_step_idx = 0
        self.step_start_time = None

        # 로깅 버퍼: [t, step_index, v_ref_dps, v_meas_dps, tau_cmd]
        self.log_data = []

        # ===== ROS 통신 =====
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
        self.pub_target = self.create_publisher(
            Float32MultiArray,
            TARGET_TOPIC,
            10
        )

        # 타이머 – dt_nominal 주기
        self.timer = self.create_timer(self.dt_nominal, self.timer_cb)

    # -------- MotorStates 콜백 --------
    def motor_states_callback(self, msg: MotorStates):
        """
        MotorStates에서 휠 속도 읽기.
        msg.speed_dps 는 LSB 이므로 0.01 곱해서 deg/s로 변환.
        """
        speed_raw = np.array(msg.speed_dps, dtype=float)  # LSB (0.01 deg/s per LSB)
        if speed_raw.size != NUM_JOINTS:
            self.get_logger().error(
                f"[WheelStepSysID] motor_states: got {speed_raw.size} speeds, expected {NUM_JOINTS}"
            )
            return

        # LSB -> deg/s
        self.current_speed_dps = speed_raw * 0.01
        self.have_state = True

    # -------- torque_commands 콜백 --------
    def torque_commands_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=float)

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
        # 종료 후에는 각도 0 유지
        now = time.time()
        if self.finished:
            self.publish_angle_cmd_zero()
            self.last_time = now
            return

        # 상태, 토크 아직 안 들어왔으면 0도 명령만
        if not (self.have_state and self.have_tau):
            self.publish_angle_cmd_zero()
            self.last_time = now
            return

        t = now - self.t0
        dt = now - self.last_time
        if dt <= 0.0:
            dt = self.dt_nominal
        self.last_time = now

        # 스텝 시작 시점 초기화
        if self.step_start_time is None:
            self.step_start_time = now
            self.current_step_idx = 0
            self.get_logger().info(
                f"[WheelStepSysID] Start step 0: "
                f"v_ref={self.step_speeds_dps[0]} deg/s"
            )

        # 현재 스텝 유지 시간이 끝났는지 확인
        elapsed_step = now - self.step_start_time
        if elapsed_step >= self.step_duration:
            self.current_step_idx += 1
            if self.current_step_idx >= self.num_steps:
                # 모든 스텝 종료
                self.get_logger().info(
                    "[WheelStepSysID] All steps finished. Saving log and stopping..."
                )
                self.finish_and_save()
                self.finished = True
                self.publish_angle_cmd_zero()
                return
            else:
                self.step_start_time = now
                self.get_logger().info(
                    f"[WheelStepSysID] Step {self.current_step_idx}/{self.num_steps - 1}: "
                    f"v_ref={self.step_speeds_dps[self.current_step_idx]} deg/s"
                )

        # 현재 스텝에서 속도 참조 [deg/s]
        v_ref = float(self.step_speeds_dps[self.current_step_idx])

        # 🔹 속도 -> 각도: q_ref += v_ref * dt
        self.ref_angles_deg[WHEEL_INDEX] += v_ref * dt

        # PD가 쓰는 각도 명령 퍼블리시 (8개 중 7번만 움직임)
        self.publish_angle_cmd_array()

        # 로깅 (휠 조인트 7번만 기록)
        v_meas = float(self.current_speed_dps[WHEEL_INDEX])  # deg/s
        tau = float(self.last_tau_cmd[WHEEL_INDEX])
        self.log_data.append([t, self.current_step_idx, v_ref, v_meas, tau])

    # -------- 각도 명령 퍼블리시 --------
    def publish_angle_cmd_zero(self):
        """
        전체 조인트 각도 0도로 명령.
        """
        cmd = np.zeros(NUM_JOINTS, dtype=float)
        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self.pub_target.publish(msg)

    def publish_angle_cmd_array(self):
        """
        ref_angles_deg를 이용해서 target_joint_angles 퍼블리시.
        7번 조인트만 ref_angles_deg에서 가져오고,
        나머지는 0으로 둔다 (안전하게 가기 위해).
        """
        cmd = np.zeros(NUM_JOINTS, dtype=float)
        cmd[WHEEL_INDEX] = self.ref_angles_deg[WHEEL_INDEX]

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self.pub_target.publish(msg)

    # -------- 로그 저장 --------
    def finish_and_save(self):
        if len(self.log_data) == 0:
            self.get_logger().warn("[WheelStepSysID] No data logged. Skip saving.")
            return

        data_arr = np.array(self.log_data, dtype=float)

        # 디렉토리 없으면 생성
        save_dir = os.path.dirname(self.save_path)
        if save_dir != '' and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        header = "t, step_index, v_ref_dps, v_meas_dps, tau_cmd"
        np.savetxt(self.save_path, data_arr, delimiter=",", header=header, comments='')

        self.get_logger().info(
            f"[WheelStepSysID] Log saved to {self.save_path} "
            f"(rows={data_arr.shape[0]})."
        )


def main(args=None):
    rclpy.init(args=args)
    node = WheelStepSysID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 중간에 끊겨도 로그는 최대한 저장
        if not node.finished:
            node.get_logger().info(
                "[WheelStepSysID] Interrupted. Saving partial log..."
            )
            node.finish_and_save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
