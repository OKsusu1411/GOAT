import rclpy
from rclpy.node import Node
#from rclpy.exceptions import RCLError
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np

# Plot 
import matplotlib
matplotlib.use('Agg')  # 디스플레이 없는 환경에서도 PNG 저장 가능하게
import matplotlib.pyplot as plt
import os
from datetime import datetime


# URDF 순서 기준 조인트 이름 (index 0~7)
# 0:hip_L, 1:hip_R, 2:thigh_L, 3:thigh_R, 4:knee_L, 5:knee_R, 6:wheel_L, 7:wheel_R
JOINT_NAME_LIST = [
    "hip_L", "hip_R",
    "thigh_L", "thigh_R",
    "knee_L", "knee_R",
    "wheel_L", "wheel_R",
]

# --- MG Motor scale ---
ANGLE_LSB_TO_DEG = 0.001      # multi_turn_raw, single_turn_raw : 0.001 deg/LSB
SPEED_LSB_TO_DPS = 0.001      # speed_dps : 0.001 deg/s per LSB (모터 매뉴얼 기준)

# Controller frequency (Hz)
DEFAULT_CONTROL_FREQUENCY = 200.0  # 200 Hz (기본 제어 주파수)

# --- Robot size ---
NUM_JOINTS = 8         # 전체 모터 개수
MOTOR_INDEX = 1        # 테스트용으로 제어할 관절 index (0~7)

# 테스트용 기본 목표각 (deg) – 지금은 여기만 수정해서 인가
JOINT_DEGREE = 0       # degrees
# KI_KP_ratio = 0.75
KI_KP_ratio = 0.8
# 휠 목표 속도 (deg/s)
DEFAULT_WHEEL_KP = 0.03
# DEFAULT_WHEEL_KI = KI_KP_ratio * DEFAULT_WHEEL_KP
DEFAULT_WHEEL_KI = 0.0
L_WHEEL_TARGET = 10.0  # 왼쪽 휠 목표 속도 (deg/s)
R_WHEEL_TARGET = 10.0  # 오른쪽 휠 목표 속도 (deg/s)
INT_TORQUE_LIMIT = 3.0  # 토크 중 적분항으로 허용할 최대 기여
INT_LIMIT = INT_TORQUE_LIMIT / DEFAULT_WHEEL_KI if DEFAULT_WHEEL_KI != 0 else 0.0

# --- Default gains (scalar) ---
DEFAULT_KP = 0.0061         # Proportional gain
DEFAULT_KD = 0.055          # Derivative gain

# LPF / Torque 기본값 (scalar)
DEFAULT_LPF_ALPHA = 1       # Low-pass filter alpha
DEFAULT_MAX_TORQUE = 4.5    # Maximum torque limit

# # --- Per-joint default lists ---> degree ---
# DEFAULT_KP_LIST           = [0.013, 0.015, 0.0061, 0.0061, 0.0161, 0.0161, 0.000061, 0.000061]
# DEFAULT_KD_LIST           = [0.055,  0.055,  0.055,  0.055,  0.055,  0.055,  0.055,  0.055]
# DEFAULT_LPF_ALPHA_LIST    = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
# DEFAULT_MAX_TORQUE_LIST   = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

# --- Per-joint default lists ---> rad --- hip = 30   thigh = 45
DEFAULT_KP_LIST           = [0.15, 0.01, 0.27,   0.27,  0.0,   2.4, 0.0,    0.0]
DEFAULT_KD_LIST           = [0.00,  0.0,  0.01,  0.001,  0.1,  0.0001,    0.0,    0.0]
DEFAULT_LPF_ALPHA_LIST    = [0.951,  0.951,   0.951,   0.951,  0.951,   0.951,  0.951,  0.951]
DEFAULT_MAX_TORQUE_LIST   = [  4.5,    0.0,     0.0,     0.0,    0.0,    0.0,     0.0,    0.0]

# 기본 타겟 각도 [deg] 리스트: MOTOR_INDEX만 JOINuT_DEGREE, 나머지 0
# DEFAULT_TARGET_ANGLES_DEG = [-20.0, 30.0, 30.0, -20.0, 30.0, -30.0, 0.0, 0.0]
DEFAULT_TARGET_ANGLES_DEG = [-45.0, 0.0, 0.0, -45.0, 50.0, -50.0, 0.0, 0.0]
# DEFAULT_TARGET_ANGLES_DEG = [0.0 for _ in range(NUM_JOINTS)]
#DEFAULT_TARGET_ANGLES_DEG[MOTOR_INDEX] = JOINT_DEGREE
     
# Topic names
MOTOR_STATES_TOPIC = 'motor_states'
TARGET_ANGLES_TOPIC = 'target_joint_angles'
TORQUE_COMMANDS_TOPIC = 'torque_commands'



class PDController(Node):
    """
    Multiple-joint PD controller.
    MotorStates에서 0.001 deg/LSB 값을 받아서 rad로 변환하여 내부 계산을 수행하고,
    플롯/로그 출력 시에는 degree/deg/s 단위로 변환해서 확인 가능하게 하는
    여러 관절 동시 제어용 PD 컨트롤러.

    - 모터별로 서로 다른 Kp, Kd, LPF alpha, Max torque를 사용할 수 있게
      kp_gains, kd_gains, lpf_alpha_list, max_torque_list 파라미터 지원.
    - target_angles_deg 파라미터 + target_joint_angles 토픽으로 목표 각도 설정.
    """
    def __init__(self):
        super().__init__('pd_controller')

        # --- Control loop frequency parameter ---
        self.control_frequency = float(
            self.declare_parameter('control_frequency', DEFAULT_CONTROL_FREQUENCY).value
        )
        self.get_logger().info(f"PD control frequency: {self.control_frequency} Hz")

        # --- Gains / Limits ---

        # 1) Kp, Kd (per-joint)  - 내부 계산은 rad 기준 gain
        kp_param = self.declare_parameter('kp_gains', DEFAULT_KP_LIST).value
        kd_param = self.declare_parameter('kd_gains', DEFAULT_KD_LIST).value

        self.kp = self._build_array_param(kp_param, DEFAULT_KP, 'Kp')
        self.kd = self._build_array_param(kd_param, DEFAULT_KD, 'Kd')

        # 2) LPF alpha, Max torque (per-joint)
        lpf_param = self.declare_parameter('lpf_alpha_list', DEFAULT_LPF_ALPHA_LIST).value
        max_torque_param = self.declare_parameter('max_torque_list', DEFAULT_MAX_TORQUE_LIST).value

        self.lpf_alpha  = self._build_array_param(lpf_param, DEFAULT_LPF_ALPHA, 'LPF alpha')
        self.max_torque = self._build_array_param(max_torque_param, DEFAULT_MAX_TORQUE, 'Max torque')

        # 3) 초기 타겟 각도 (deg) – 테스트용 기본값, 이후 토픽으로 override 가능
        target_param = self.declare_parameter(
            'target_angles_deg', DEFAULT_TARGET_ANGLES_DEG
        ).value
        self.target_angles_deg = self._build_array_param(
            target_param, 0.0, 'target angles (deg)'
        )
        self.target_angles_rad = np.deg2rad(self.target_angles_deg)
    

        self.get_logger().info(f"Kp gains per joint        : {self.kp.tolist()}")
        self.get_logger().info(f"Kd gains per joint        : {self.kd.tolist()}")
        self.get_logger().info(f"LPF alpha per joint       : {self.lpf_alpha.tolist()}")
        self.get_logger().info(f"Max torque per joint      : {self.max_torque.tolist()}")
        # 로그는 사람이 보기 좋게 deg로 변환해서 출력
        self.get_logger().info(
            f"Initial target angles (deg): {np.rad2deg(self.target_angles_rad).tolist()}"
        )

        # --- State Variables (rad 기준, 내부 계산용) ---
        self.current_angles_rad = np.zeros(NUM_JOINTS)        # 현재 관절 각도 [rad]
        self.current_velocities_rad_s = np.zeros(NUM_JOINTS)  # 현재 관절 각속도 [rad/s]
        self.current_angles_deg = np.zeros(NUM_JOINTS)        # 현재 관절 각도 [rad] (변수명 유지)
        self.current_velocities_deg_s = np.zeros(NUM_JOINTS)  # 현재 관절 각속도 [rad/s] (변수명 유지)
    
        self.previous_torque_command = np.zeros(NUM_JOINTS)

        self.last_angle_update_time = None
        self.last_angles_deg = None  # [rad]

        # wheel control variables
        self.wheel_indices = [6, 7]

        # 기본값을 "0 리스트"가 아니라 "None"으로 두고, 없을 때만 default_scalar 적용
        wheel_kp_param = self.declare_parameter('wheel_kp_gains', None).value
        wheel_ki_param = self.declare_parameter('wheel_ki_gains', None).value

        if wheel_kp_param is None:
            self.wheel_kp = np.full(NUM_JOINTS, DEFAULT_WHEEL_KP, dtype=float)
        else:
            self.wheel_kp = self._build_array_param(wheel_kp_param, DEFAULT_WHEEL_KP, 'wheel Kp')

        if wheel_ki_param is None:
            self.wheel_ki = np.full(NUM_JOINTS, DEFAULT_WHEEL_KI, dtype=float)
        else:
            self.wheel_ki = self._build_array_param(wheel_ki_param, DEFAULT_WHEEL_KI, 'wheel Ki')

        self.wheel_int = np.zeros(NUM_JOINTS, dtype=float)
        self.wheel_speed_ref = np.zeros(NUM_JOINTS, dtype=float)  # rad/s
        # 실험용: 코드 내부에서 휠 목표 속도 설정 (deg/s → rad/s)
        self.wheel_speed_ref[6] = np.deg2rad(L_WHEEL_TARGET)
        self.wheel_speed_ref[7] = np.deg2rad(R_WHEEL_TARGET)


        # --- ROS2 Communications ---
        self.create_subscription(MotorStates, MOTOR_STATES_TOPIC, self.motor_states_callback, 100)

        # 토픽으로 목표각 업데이트 받을 준비 (없으면 파라미터 값으로만 동작)
        self.create_subscription(Float32MultiArray, TARGET_ANGLES_TOPIC, self.target_angles_callback, 100)

        self.torque_publisher = self.create_publisher(Float32MultiArray, TORQUE_COMMANDS_TOPIC, 100)

        # --- Controller Timer ---
        timer_period = 1.0 / self.control_frequency
        self.timer = self.create_timer(timer_period, self.controller_callback)

        # --- Logging buffers for plotting (single joint) ---
        self.time_log = []
        self.current_angle_log = []   # [deg]로 변환해서 저장
        self.target_angle_log = []    # [deg]로 변환해서 저장
        self.torque_log = []
        self.velocity_log = []        # [deg/s]로 변환해서 저장
        self.start_time = None  # 첫 샘플 시간 (0초 기준 맞추기용)
        self.logging_started = False  # 토크가 실제로 나오기 시작한 이후부터 로깅 ### <<< 추가

        # --- Logging buffers for all joints ---
        self.current_angles_all_log = []   # shape: [N, NUM_JOINTS], [deg]로 변환해서 저장
        self.velocities_all_log = []       # shape: [N, NUM_JOINTS], [deg/s]로 변환해서 저장
        self.torques_all_log = []          # shape: [N, NUM_JOINTS]

        # --- Wheel logging (speed tracking) ---
        # 휠 속도 추종 그래프용 로그 (rad/s → 저장 시 deg/s로 변환)
        self.wheel_time_log = []
        self.wheel_speed_meas_log = []   # shape: [N, 2]
        self.wheel_speed_ref_log  = []   # shape: [N, 2]

    # === 내부 유틸: 리스트 파라미터 → 길이 NUM_JOINTS인 np.array ===
    def _build_array_param(self, value, default_scalar, name: str) -> np.ndarray:
        """
        리스트/스칼라 파라미터를 받아서 길이 NUM_JOINTS인 np.array로 변환.
        - value가 비었거나 파싱에 실패하면 default_scalar로 채움
        - 길이가 짧으면 default_scalar로 패딩
        - 길이가 길면 자르면서 warn 출력
        """
        try:
            arr = np.array(value, dtype=float).flatten()
        except Exception:
            self.get_logger().warn(
                f"{name} parameter invalid, using default scalar {default_scalar} for all joints."
            )
            return np.full(NUM_JOINTS, default_scalar, dtype=float)

        if arr.size == 0:
            self.get_logger().warn(
                f"{name} list empty, using default scalar {default_scalar} for all joints."
            )
            return np.full(NUM_JOINTS, default_scalar, dtype=float)

        if arr.size < NUM_JOINTS:
            padded = np.full(NUM_JOINTS, default_scalar, dtype=float)
            padded[:arr.size] = arr
            self.get_logger().warn(
                f"{name} length {arr.size} < NUM_JOINTS={NUM_JOINTS}, "
                f"padding remaining entries with {default_scalar}."
            )
            return padded

        if arr.size > NUM_JOINTS:
            self.get_logger().warn(
                f"{name} length {arr.size} > NUM_JOINTS={NUM_JOINTS}, extra values will be ignored."
            )
            arr = arr[:NUM_JOINTS]

        return arr

    def motor_states_callback(self, msg: MotorStates):
        """
        MotorStates로부터 multi_turn_raw / speed_dps 를 받아 현재 각도/각속도를 업데이트.
        - multi_turn_raw : 0.001 deg / LSB  → ANGLE_LSB_TO_DEG 배율로 [deg] 계산 후 rad로 변환
        - speed_dps      : 0.001 deg/s /LSB → SPEED_LSB_TO_DPS 배율로 [deg/s] 계산 후 rad/s로 변환
        """
        now = self.get_clock().now()

        # === 1) 각도 [deg] 및 [rad] ===
        raw_angles_deg = np.array(msg.multi_turn_raw, dtype=float) * ANGLE_LSB_TO_DEG

        if raw_angles_deg.size != NUM_JOINTS:
            self.get_logger().error(
                f"motor_states_callback: received {raw_angles_deg.size} angles, "
                f"expected {NUM_JOINTS}."
            )
            return

        # degree / rad 둘 다 유지
        self.current_angles_deg = raw_angles_deg
        self.current_angles_rad = np.deg2rad(raw_angles_deg)

        # === 2) 각속도 [deg/s] 및 [rad/s] ===
        raw_speed_deg_s = np.array(msg.speed_dps, dtype=float) * SPEED_LSB_TO_DPS

        if raw_speed_deg_s.size != NUM_JOINTS:
            self.get_logger().error(
                f"motor_states_callback: received {raw_speed_deg_s.size} speeds, "
                f"expected {NUM_JOINTS}."
            )
            return

        self.current_velocities_deg_s = raw_speed_deg_s
        self.current_velocities_rad_s = np.deg2rad(raw_speed_deg_s)

        # 마지막 업데이트 기록
        self.last_angles_rad = self.current_angles_rad.copy()
        self.last_angle_update_time = now

    def target_angles_callback(self, msg: Float32MultiArray):
        """
        목표 각도 업데이트 (토픽 입력은 deg 단위).
        - msg.data 길이가 NUM_JOINTS와 다르면 에러만 찍고 무시.
        - 파라미터로 설정한 초기 target_angles_deg 위에 override.
        """
        arr_deg = np.array(msg.data, dtype=float).flatten()

        if arr_deg.size != NUM_JOINTS:
            self.get_logger().error(
                f"Received {arr_deg.size} target angles, expected {NUM_JOINTS}. "
                "Target angles update ignored."
            )
            return

        # deg / rad 둘 다 유지
        self.target_angles_deg = arr_deg
        self.target_angles_rad = np.deg2rad(arr_deg)

        # 로그는 degree 기준으로 출력
        self.get_logger().info(f"Updated target angles (deg): {self.target_angles_deg.tolist()}")

    def controller_callback(self):
        """
        PD 제어 루프 (내부 계산은 rad / rad/s 단위).
        - position_error: [rad]
        - velocity_error: [rad/s]
        - torque: [arb. unit] (실제 튜닝에 따라 해석)
        모터별 Kp, Kd, LPF, Max torque가 벡터로 적용됨.
        """
        # dt 계산
        now = self.get_clock().now().nanoseconds / 1e9
        if not hasattr(self, "last_ctrl_time"):
            dt = 1.0 / self.control_frequency
        else:
            dt = now - self.last_ctrl_time
        self.last_ctrl_time = now

        # --- 인덱스 분리: 0~5 = 관절, 6~7 = 휠 ---
        joint_indices = [i for i in range(NUM_JOINTS) if i not in self.wheel_indices]

        # --- PD Control Law (모터별 개별 gain, 내부 rad 기준) ---
        position_error = self.target_angles_rad - self.current_angles_rad   # [rad]
        desired_vel_rad_s = np.zeros(NUM_JOINTS)   # 일단 0 가정
        velocity_error = desired_vel_rad_s - self.current_velocities_rad_s
        velocity_error *= 0.001  # 스케일 조절은 이 다음에

        # 전체 토크 벡터 초기화
        raw_torque_command = np.zeros(NUM_JOINTS, dtype=float)

        # 1) 조인트(0~5)에는 PD 제어만 적용
        raw_torque_command[joint_indices] = (
            self.kp[joint_indices] * position_error[joint_indices]
            + self.kd[joint_indices] * velocity_error[joint_indices]
        )

        # 2) 휠(6,7)은 속도 PI 제어 (rad/s 기준)로 덮어쓰기
        for idx in self.wheel_indices:
            omega_ref = self.wheel_speed_ref[idx]               # [rad/s]
            omega_meas = self.current_velocities_rad_s[idx]     # [rad/s]

            e_w = omega_ref - omega_meas

            self.wheel_int[idx] += e_w * dt
            self.wheel_int[idx] = np.clip(self.wheel_int[idx], -INT_LIMIT, INT_LIMIT)
            tau_pi = self.wheel_kp[idx] * e_w + self.wheel_ki[idx] * self.wheel_int[idx]
            raw_torque_command[idx] = tau_pi

        # 디버깅용 로그 (필요 없으면 주석)
        # 로그는 사람이 보기 편하게 deg / deg/s로 변환해서 출력
        pos_err_deg = np.rad2deg(position_error)
        vel_err_deg_s = np.rad2deg(velocity_error)
        # self.get_logger().info_throttle(1.0, "...")
        # self.get_logger().info(f"position_error(deg): {np.round(pos_err_deg, 4)}")
        # self.get_logger().info(f"velocity_error(deg/s): {np.round(vel_err_deg_s, 4)}")
        # self.get_logger().info(f"raw_torque_command: {np.round(raw_torque_command, 4)}")


        # --- Low-Pass Filter (LPF) ---
        filtered_torque = (
            self.lpf_alpha * raw_torque_command
            + (1.0 - self.lpf_alpha) * self.previous_torque_command
        )

        # --- Torque Clipping (per-joint limit) ---
        clipped_torque = np.clip(filtered_torque, -self.max_torque, self.max_torque)

        # 모든 조인트에 토크 사용 (이미 per-joint clip 적용됨)
        torque_output = clipped_torque

        # 다음 LPF를 위해 저장
        self.previous_torque_command = torque_output

        # === 여기서부터: 토크가 의미 있게 나오기 전까지는 로깅 안 함 ===
        TORQUE_START_THRESH = 1e-3  # 이 값보다 크면 "입력이 들어왔다"고 판단 ### <<<
        max_tau = float(np.max(np.abs(torque_output)))

        if not self.logging_started:
            if max_tau > TORQUE_START_THRESH:
                # 이 시점을 로깅 시작 기준으로 사용
                self.logging_started = True
                self.start_time = now
                self.get_logger().info(
                    f"Logging started at t={now:.3f}s (|tau|max={max_tau:.4f})"
                )
            else:
                # 아직 입력 안 들어왔다고 보고, 토크만 퍼블리시하고 return
                torque_msg = Float32MultiArray()
                torque_msg.data = torque_output.flatten().tolist()
                self.torque_publisher.publish(torque_msg)
                self.get_logger().info(f"Published Torque Command: {torque_msg.data}")
                return
        # === 로깅 시작 이후 ===

        # --- Logging for plotting ---
        now = self.get_clock().now().nanoseconds / 1e9  # [s]
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time

        # 단일 조인트(MOTOR_INDEX) 로그 (deg, deg/s로 변환해서 저장)
        self.time_log.append(t)
        self.current_angle_log.append(float(self.current_angles_deg[MOTOR_INDEX]))          # [deg]
        self.target_angle_log.append(float(self.target_angles_deg[MOTOR_INDEX]))            # [deg]
        self.torque_log.append(float(torque_output[MOTOR_INDEX]))
        self.velocity_log.append(float(self.current_velocities_deg_s[MOTOR_INDEX]))         # [deg/s]

        # 전체 조인트 로그 (deg, deg/s로 변환해서 저장)
        self.current_angles_all_log.append(self.current_angles_deg.copy())                  # [deg]
        self.velocities_all_log.append(self.current_velocities_deg_s.copy())                # [deg/s]
        self.torques_all_log.append(torque_output.copy())

        # 휠 속도 추종 로그
        wheel_meas = self.current_velocities_deg_s[self.wheel_indices].copy()               # [deg/s]
        wheel_ref  = np.rad2deg(self.wheel_speed_ref[self.wheel_indices].copy())            # [deg/s]

        self.wheel_time_log.append(t)
        self.wheel_speed_meas_log.append(wheel_meas)    # [deg/s]
        self.wheel_speed_ref_log.append(wheel_ref)      # [deg/s]

        # --- Publish Command ---
        torque_msg = Float32MultiArray()
        torque_msg.data = torque_output.flatten().tolist()
        self.torque_publisher.publish(torque_msg)

        # # --- Publish Command: MOTOR_INDEX만 토크 인가 ---
        # torque_output_cmd = np.zeros(NUM_JOINTS)
        # torque_output_cmd[MOTOR_INDEX] = torque_output[MOTOR_INDEX]     
        # self.previous_torque_command = torque_output_cmd  # LPF도 이 값 기준으로
        # torque_msg = Float32MultiArray()
        # torque_msg.data = torque_output_cmd.flatten().tolist()
        # self.torque_publisher.publish(torque_msg)

        self.get_logger().info(f"Published Torque Command: {torque_msg.data}")

    def save_plots(self):
        """노드 종료 시 누적된 로그를 바탕으로 그래프 PNG로 저장 (0~100초 구간만, 단위는 deg / deg/s)
        - 단일 조인트 플롯은 제거
        - 왼쪽/오른쪽 관절(0~5번) 각각에 대해 Angle / Velocity / Torque 플롯 생성
        - 휠 속도 추종 그래프는 기존처럼 유지
        """
        if not self.time_log:
            self.get_logger().warn("No logged data, skip plotting.")
            return

        t   = np.array(self.time_log)
        cur = np.array(self.current_angle_log)   # [deg]
        tgt = np.array(self.target_angle_log)    # [deg]
        tq  = np.array(self.torque_log)
        vel = np.array(self.velocity_log)        # [deg/s]

        # all-joint 로그 (없으면 None) - 이미 deg/deg/s 단위로 저장됨
        angles_all = np.array(self.current_angles_all_log) if self.current_angles_all_log else None  # [deg]
        vel_all    = np.array(self.velocities_all_log)     if self.velocities_all_log     else None  # [deg/s]
        tq_all     = np.array(self.torques_all_log)        if self.torques_all_log        else None

        # ===== 1) n초까지만 사용 =====
        max_time = 100.0  # [s]
        mask = t <= max_time

        if np.any(mask):
            t   = t[mask]
            cur = cur[mask]
            tgt = tgt[mask]
            tq  = tq[mask]
            vel = vel[mask]

            if angles_all is not None and angles_all.shape[0] == mask.size:
                angles_all = angles_all[mask]
                vel_all    = vel_all[mask]
                tq_all     = tq_all[mask]
        else:
            self.get_logger().warn("No samples within 100s window, plotting all data.")

        t_end = float(np.max(t))
        view_end = min(max_time, t_end)

        # x축 major/minor tick 설정
        major_step = 1.0   # 1초 단위 라벨
        minor_step = 0.1   # 0.1초 단위 그리드

        major_ticks = np.arange(0.0, view_end + 1e-9, major_step)
        minor_ticks = np.arange(0.0, view_end + 1e-9, minor_step)

        out_dir = os.path.expanduser("./src/goat_control/pd_logs")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dpi_val = 200
        figsize_val = (10, 4)

        angle_left_path = angle_right_path = None
        vel_left_path = vel_right_path = None
        torque_left_path = torque_right_path = None

        # ====================== 왼/오른쪽 관절(0~5번) 그래프 ======================
        if angles_all is not None and vel_all is not None and tq_all is not None:
            # URDF 순서 기준:
            # 0:hip_L, 1:hip_R, 2:thigh_L, 3:thigh_R, 4:knee_L, 5:knee_R, 6:wheel_L, 7:wheel_R
            left_indices  = [0, 2, 4]  # hip_L, thigh_L, knee_L
            right_indices = [1, 3, 5]  # hip_R, thigh_R, knee_R

            # ---- A-L) 왼쪽 각도 ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in left_indices:
                ax.plot(t, angles_all[:, j], label=JOINT_NAME_LIST[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angle [deg]")
            ax.set_title("Left leg joint angles (deg)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            angle_left_path = os.path.join(out_dir, f"pd_angle_left_{stamp}.png")
            fig.savefig(angle_left_path, dpi=dpi_val)
            plt.close(fig)

            # ---- A-R) 오른쪽 각도 ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in right_indices:
                ax.plot(t, angles_all[:, j], label=JOINT_NAME_LIST[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angle [deg]")
            ax.set_title("Right leg joint angles (deg)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            angle_right_path = os.path.join(out_dir, f"pd_angle_right_{stamp}.png")
            fig.savefig(angle_right_path, dpi=dpi_val)
            plt.close(fig)

            # ---- B-L) 왼쪽 속도 ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in left_indices:
                ax.plot(t, vel_all[:, j], label=JOINT_NAME_LIST[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Velocity [deg/s]")
            ax.set_title("Left leg joint velocities (deg/s)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            vel_left_path = os.path.join(out_dir, f"pd_velocity_left_{stamp}.png")
            fig.savefig(vel_left_path, dpi=dpi_val)
            plt.close(fig)

            # ---- B-R) 오른쪽 속도 ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in right_indices:
                ax.plot(t, vel_all[:, j], label=JOINT_NAME_LIST[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Velocity [deg/s]")
            ax.set_title("Right leg joint velocities (deg/s)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            vel_right_path = os.path.join(out_dir, f"pd_velocity_right_{stamp}.png")
            fig.savefig(vel_right_path, dpi=dpi_val)
            plt.close(fig)

            # ---- C-L) 왼쪽 토크 ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in left_indices:
                ax.plot(t, tq_all[:, j], label=JOINT_NAME_LIST[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Torque [arb. unit]")
            ax.set_title("Left leg joint torques")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            torque_left_path = os.path.join(out_dir, f"pd_torque_left_{stamp}.png")
            fig.savefig(torque_left_path, dpi=dpi_val)
            plt.close(fig)

            # ---- C-R) 오른쪽 토크 ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in right_indices:
                ax.plot(t, tq_all[:, j], label=JOINT_NAME_LIST[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Torque [arb. unit]")
            ax.set_title("Right leg joint torques")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            torque_right_path = os.path.join(out_dir, f"pd_torque_right_{stamp}.png")
            fig.savefig(torque_right_path, dpi=dpi_val)
            plt.close(fig)

            self.get_logger().info(
                "Saved LEFT/RIGHT leg PD plots:\n"
                f"  angle_left   : {angle_left_path}\n"
                f"  angle_right  : {angle_right_path}\n"
                f"  vel_left     : {vel_left_path}\n"
                f"  vel_right    : {vel_right_path}\n"
                f"  torque_left  : {torque_left_path}\n"
                f"  torque_right : {torque_right_path}"
            )

        # ====================== 휠 속도 추종 그래프 (기존 유지) ======================
        wheel_speed_path = None
        if self.wheel_time_log:
            wt = np.array(self.wheel_time_log)
            w_meas = np.array(self.wheel_speed_meas_log)  # [deg/s], shape [N, 2]
            w_ref  = np.array(self.wheel_speed_ref_log)   # [deg/s], shape [N, 2]

            mask_w = wt <= max_time
            if np.any(mask_w):
                wt = wt[mask_w]
                w_meas = w_meas[mask_w]
                w_ref  = w_ref[mask_w]

            fig, ax = plt.subplots(figsize=figsize_val)
            ax.plot(wt, w_meas[:, 0], label="wheel_L_meas_deg_s")
            ax.plot(wt, w_ref[:, 0],  '--', label="wheel_L_ref_deg_s")
            ax.plot(wt, w_meas[:, 1], label="wheel_R_meas_deg_s")
            ax.plot(wt, w_ref[:, 1],  '--', label="wheel_R_ref_deg_s")

            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Wheel speed [deg/s]")
            ax.set_title("Wheel Speed Tracking (L/R)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            wheel_speed_path = os.path.join(out_dir, f"wheel_speed_{stamp}.png")
            fig.savefig(wheel_speed_path, dpi=dpi_val)
            plt.close(fig)

            self.get_logger().info(f"Saved wheel speed plot: {wheel_speed_path}")

    def destroy_node(self):
        # 노드 종료시 플롯 저장
        try:
            self.save_plots()
        except Exception as e:
            try:
                self.get_logger().error(f"Failed to save plots: {e}")
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    pd_controller = PDController()

    try:
        rclpy.spin(pd_controller)
    except KeyboardInterrupt:
        pass
    finally:
        pd_controller.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            # 이미 shutdown된 경우 등 에러는 그냥 무시
            pass


if __name__ == '__main__':
    main()
