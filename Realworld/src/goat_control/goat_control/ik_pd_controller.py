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

# --- MG Motor scale ---
ANGLE_LSB_TO_DEG = 0.001      # multi_turn_raw, single_turn_raw : 0.001 deg/LSB
SPEED_LSB_TO_DPS = 0.001      # speed_dps : 0.001 deg/s per LSB (모터 매뉴얼 기준)

# Controller frequency (Hz)
DEFAULT_CONTROL_FREQUENCY = 200.0  # 100 Hz

# --- Robot size ---
NUM_JOINTS = 8         # 전체 모터 개수
MOTOR_INDEX = 1      # 테스트용으로 제어할 관절 index (0~7)

# 테스트용 기본 목표각 (deg) – 지금은 여기만 수정해서 인가
JOINT_DEGREE = 0    # degrees

# --- Default gains (scalar) ---
DEFAULT_KP = 0.0061         # Proportional gain
DEFAULT_KD = 0.055          # Derivative gain

# LPF / Torque 기본값 (scalar)
DEFAULT_LPF_ALPHA = 0.8     # Low-pass filter alpha
DEFAULT_MAX_TORQUE = 4.5    # Maximum torque limit

# # --- Per-joint default lists ---> degree ---
# DEFAULT_KP_LIST           = [0.013, 0.015, 0.0061, 0.0061, 0.0161, 0.0161, 0.000061, 0.000061]
# DEFAULT_KD_LIST           = [0.055,  0.055,  0.055,  0.055,  0.055,  0.055,  0.055,  0.055]
# DEFAULT_LPF_ALPHA_LIST    = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
# DEFAULT_MAX_TORQUE_LIST   = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

# --- Per-joint default lists ---> rad ---
DEFAULT_KP_LIST           = [0.01, 0.0156,   0.016,   0.016,   0.028,   0.028, 0.000061, 0.000061]
DEFAULT_KD_LIST           = [0.005,  0.001,  0.0001,  0.0001,  0.001,  0.0001,    0.055,    0.055]
DEFAULT_LPF_ALPHA_LIST    = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
DEFAULT_MAX_TORQUE_LIST   = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

# 기본 타겟 각도 [deg] 리스트: MOTOR_INDEX만 JOINT_DEGREE, 나머지 0
DEFAULT_TARGET_ANGLES_DEG = [-30.0, 30.0, 30.0, -20.0, 30.0, -30.0, 0.0, 0.0]
#DEFAULT_TARGET_ANGLES_DEG[MOTOR_INDEX] = JOINT_DEGREE

# Topic names
MOTOR_STATES_TOPIC = 'motor_states'
TARGET_ANGLES_TOPIC = 'target_joint_angles'
TORQUE_COMMANDS_TOPIC = 'torque_commands'

# Controller frequency
CONTROLLER_TIMER_PERIOD = 0.01  # seconds (100Hz)


class PDController(Node):
    """
    Multiple-joint PD controller (deg 단위 사용).
    MotorStates에서 0.001 deg/LSB 값을 받아서 degree로 계산하고,
    기본적으로 여러 관절을 동시에 제어하는 PD 컨트롤러.

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

        # 1) Kp, Kd (per-joint)
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

        self.get_logger().info(f"Kp gains per joint        : {self.kp.tolist()}")
        self.get_logger().info(f"Kd gains per joint        : {self.kd.tolist()}")
        self.get_logger().info(f"LPF alpha per joint       : {self.lpf_alpha.tolist()}")
        self.get_logger().info(f"Max torque per joint      : {self.max_torque.tolist()}")
        self.get_logger().info(f"Initial target angles (deg): {self.target_angles_deg.tolist()}")

        # --- State Variables (deg 기준) ---
        self.current_angles_deg = np.zeros(NUM_JOINTS)        # 현재 관절 각도 [deg]
        self.current_velocities_deg_s = np.zeros(NUM_JOINTS)  # 현재 관절 각속도 [deg/s]

        self.previous_torque_command = np.zeros(NUM_JOINTS)

        self.last_angle_update_time = None
        self.last_angles_deg = None

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
        self.current_angle_log = []
        self.target_angle_log = []
        self.torque_log = []
        self.velocity_log = []
        self.start_time = None  # 첫 샘플 시간 (0초 기준 맞추기용)

        # --- Logging buffers for all joints ---
        self.current_angles_all_log = []   # shape: [N, NUM_JOINTS]
        self.velocities_all_log = []       # shape: [N, NUM_JOINTS]
        self.torques_all_log = []          # shape: [N, NUM_JOINTS]

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
        - multi_turn_raw : 0.001 deg / LSB  → ANGLE_LSB_TO_DEG 배율로 [deg]
        - speed_dps      : 0.001 deg/s /LSB → SPEED_LSB_TO_DPS 배율로 [deg/s]
        """
        now = self.get_clock().now()

        # === 1) 각도 [deg] ===
        try:
            raw_angles_deg = np.array(msg.multi_turn_raw, dtype=float) * ANGLE_LSB_TO_DEG
        except Exception:
            raw_angles_deg = np.zeros(NUM_JOINTS)

        if len(raw_angles_deg) != NUM_JOINTS:
            self.get_logger().warn(
                f"Received {len(raw_angles_deg)} joint states, expected {NUM_JOINTS}. Padding/truncating."
            )
            padded_angles = np.zeros(NUM_JOINTS)
            num_to_copy = min(len(raw_angles_deg), NUM_JOINTS)
            padded_angles[:num_to_copy] = raw_angles_deg[:num_to_copy]
            self.current_angles_deg = np.deg2rad(padded_angles)
        else:
            self.current_angles_deg = np.deg2rad(raw_angles_deg)

        # === 2) 각속도 [deg/s] ===
        try:
            raw_speed_deg_s = np.array(msg.speed_dps, dtype=float) * SPEED_LSB_TO_DPS
        except Exception:
            raw_speed_deg_s = None

        if raw_speed_deg_s is not None and raw_speed_deg_s.size > 0:
            if len(raw_speed_deg_s) != NUM_JOINTS:
                self.get_logger().warn(
                    f"Received {len(raw_speed_deg_s)} speed_dps, expected {NUM_JOINTS}. Padding/truncating."
                )
                padded_speed = np.zeros(NUM_JOINTS)
                num_to_copy = min(len(raw_speed_deg_s), NUM_JOINTS)
                padded_speed[:num_to_copy] = raw_speed_deg_s[:num_to_copy]
                self.current_velocities_deg_s = np.deg2rad(padded_speed)
            else:
                self.current_velocities_deg_s = np.deg2rad(raw_speed_deg_s)
        else:
            self.get_logger().warn("No valid speed_dps in MotorStates, keep previous velocities.")

        self.last_angles_deg = self.current_angles_deg.copy()
        self.last_angle_update_time = now

    def target_angles_callback(self, msg: Float32MultiArray):
        """
        목표 각도 업데이트 (deg 단위).
        - msg.data 길이가 NUM_JOINTS와 다르면 padding/truncating.
        - 파라미터로 설정한 초기 target_angles_deg 위에 override.
        """
        arr = np.array(msg.data, dtype=float).flatten()
        if arr.size != NUM_JOINTS:
            self.get_logger().warn(
                f"Received {arr.size} target angles, expected {NUM_JOINTS}. Padding/truncating."
            )
            padded = np.zeros(NUM_JOINTS, dtype=float)
            num_to_copy = min(arr.size, NUM_JOINTS)
            padded[:num_to_copy] = arr[:num_to_copy]
            self.target_angles_deg = np.deg2rad(padded)
        else:
            self.target_angles_deg = np.deg2rad(arr)

        self.get_logger().info(f"Updated target angles (deg): {self.target_angles_deg.tolist()}")

    def controller_callback(self):
        """
        PD 제어 루프 (deg 단위).
        - position_error: [deg]
        - velocity_error: [deg/s]
        - torque: [arb. unit] (실제 튜닝에 따라 해석)
        모터별 Kp, Kd, LPF, Max torque가 벡터로 적용됨.
        """
        position_error = self.target_angles_deg - self.current_angles_deg
        velocity_error = -self.current_velocities_deg_s * 0.001

        # --- PD Control Law (모터별 개별 gain) ---
        raw_torque_command = self.kp * position_error + self.kd * velocity_error

        # 디버깅용 로그 (필요 없으면 주석)
        self.get_logger().info(f"position_error(deg): {position_error}")
        self.get_logger().info(f"velocity_error(deg/s): {velocity_error}")
        self.get_logger().info(f"raw_torque_command: {raw_torque_command}")

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
        

        # --- Logging for plotting ---
        now = self.get_clock().now().nanoseconds / 1e9  # [s]
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time

        # 단일 조인트(MOTOR_INDEX) 로그
        self.time_log.append(t)
        self.current_angle_log.append(float(self.current_angles_deg[MOTOR_INDEX]))
        self.target_angle_log.append(float(self.target_angles_deg[MOTOR_INDEX]))
        self.torque_log.append(float(torque_output[MOTOR_INDEX]))
        self.velocity_log.append(float(self.current_velocities_deg_s[MOTOR_INDEX]))

        # 전체 조인트 로그
        self.current_angles_all_log.append(self.current_angles_deg.copy())
        self.velocities_all_log.append(self.current_velocities_deg_s.copy())
        self.torques_all_log.append(torque_output.copy())

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
        """노드 종료 시 누적된 로그를 바탕으로 그래프 PNG로 저장 (0~7초 구간만)"""
        if not self.time_log:
            self.get_logger().warn("No logged data, skip plotting.")
            return

        t   = np.array(self.time_log)
        cur = np.array(self.current_angle_log)
        tgt = np.array(self.target_angle_log)
        tq  = np.array(self.torque_log)
        vel = np.array(self.velocity_log)

        # all-joint 로그 (없으면 None)
        angles_all = np.array(self.current_angles_all_log) if self.current_angles_all_log else None
        vel_all    = np.array(self.velocities_all_log)     if self.velocities_all_log     else None
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
                angles_all = np.rad2deg(angles_all[mask])
                vel_all    = np.rad2deg(vel_all[mask])
                tq_all     = tq_all[mask]
        else:
            self.get_logger().warn("No samples within 7s window, plotting all data.")

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

        # ====================== 기존: 단일 조인트(MOTOR_INDEX) 그래프 3개 ======================

        # -------- 1) 각도 그래프 (single joint) --------
        fig, ax = plt.subplots(figsize=figsize_val)
        ax.plot(t, cur, label="current_angle_deg")
        ax.plot(t, tgt, label="target_angle_deg", linestyle="--")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [deg]")
        ax.set_title(f"Joint {MOTOR_INDEX} Angle Tracking\n(Kp={self.kp[MOTOR_INDEX]:.4f}, Kd={self.kd[MOTOR_INDEX]:.4f})")

        ax.set_xlim(0.0, view_end)
        ax.set_xticks(major_ticks)               # 굵은 눈금 + 라벨
        ax.set_xticks(minor_ticks, minor=True)   # 가는 눈금 (라벨 없음)

        ax.grid(True, which='major', linewidth=0.8)
        ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)

        ax.legend()
        fig.tight_layout()
        angle_path = os.path.join(out_dir, f"pd_angle_{stamp}.png")
        fig.savefig(angle_path, dpi=dpi_val)
        plt.close(fig)

        # -------- 2) 속도 그래프 (single joint) --------
        fig, ax = plt.subplots(figsize=figsize_val)
        ax.plot(t, vel, label="velocity_deg_s")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [deg/s]")
        ax.set_title(f"Joint {MOTOR_INDEX} Velocity\n(Kp={self.kp[MOTOR_INDEX]:.4f}, Kd={self.kd[MOTOR_INDEX]:.4f})")

        ax.set_xlim(0.0, view_end)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        ax.grid(True, which='major', linewidth=0.8)
        ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)

        fig.tight_layout()
        vel_path = os.path.join(out_dir, f"pd_velocity_{stamp}.png")
        fig.savefig(vel_path, dpi=dpi_val)
        plt.close(fig)

        # -------- 3) 토크 그래프 (single joint) --------
        fig, ax = plt.subplots(figsize=figsize_val)
        ax.plot(t, tq, label="torque_command")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque [arb. unit]")
        ax.set_title(f"Joint {MOTOR_INDEX} Torque Command\n(Kp={self.kp[MOTOR_INDEX]:.4f}, Kd={self.kd[MOTOR_INDEX]:.4f})")

        ax.set_xlim(0.0, view_end)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        ax.grid(True, which='major', linewidth=0.8)
        ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)

        fig.tight_layout()
        torque_path = os.path.join(out_dir, f"pd_torque_{stamp}.png")
        fig.savefig(torque_path, dpi=dpi_val)
        plt.close(fig)

        # ====================== 추가: 조인트 6개(0~5) 그래프 3개 ======================
        if angles_all is not None and vel_all is not None and tq_all is not None:
            joints_to_plot = range(min(6, NUM_JOINTS))  # 0~5까지 6개 조인트

            # ---- A) 각도 (all joints) ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in joints_to_plot:
                ax.plot(t, angles_all[:, j], label=f"joint{j}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angle [deg]")
            ax.set_title("Joint 0-5 Angle (deg)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            angle_all_path = os.path.join(out_dir, f"pd_angle_all_{stamp}.png")
            fig.savefig(angle_all_path, dpi=dpi_val)
            plt.close(fig)

            # ---- B) 속도 (all joints) ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in joints_to_plot:
                ax.plot(t, vel_all[:, j], label=f"joint{j}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Velocity [deg/s]")
            ax.set_title("Joint 0-5 Velocity (deg/s)")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            vel_all_path = os.path.join(out_dir, f"pd_velocity_all_{stamp}.png")
            fig.savefig(vel_all_path, dpi=dpi_val)
            plt.close(fig)

            # ---- C) 토크 (all joints) ----
            fig, ax = plt.subplots(figsize=figsize_val)
            for j in joints_to_plot:
                ax.plot(t, tq_all[:, j], label=f"joint{j}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Torque [arb. unit]")
            ax.set_title("Joint 0-5 Torque Command")
            ax.set_xlim(0.0, view_end)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True, which='major', linewidth=0.8)
            ax.grid(True, which='minor', linewidth=0.3, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            torque_all_path = os.path.join(out_dir, f"pd_torque_all_{stamp}.png")
            fig.savefig(torque_all_path, dpi=dpi_val)
            plt.close(fig)

            self.get_logger().info(
                f"Saved ALL-JOINT PD plots:\n"
                f"  angle_all   : {angle_all_path}\n"
                f"  velocity_all: {vel_all_path}\n"
                f"  torque_all  : {torque_all_path}"
            )

        self.get_logger().info(
            f"Saved PD plots:\n"
            f"  angle   : {angle_path}\n"
            f"  velocity: {vel_path}\n"
            f"  torque  : {torque_path}"
        )

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