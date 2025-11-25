import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np

# Controller frequency
# Controller frequency (Hz)
DEFAULT_CONTROL_FREQUENCY = 100.0  # 100 Hz

# --- Constants ---
# Default PD gains
DEFAULT_KP = 0.03         # Proportional gain
DEFAULT_KD = 0.001       # Derivative gain
DEFAULT_LPF_ALPHA = 0.8  # Low-pass filter alpha
DEFAULT_MAX_TORQUE = 0.5 # Maximum torque limit
JOINT_DEGREE = 20.0  # degrees

# Number of joints
NUM_JOINTS = 8

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
    하나의 관절(여기서는 index 3)만 토크를 출력하는 테스트용 컨트롤러.
    """
    def __init__(self):
        super().__init__('pd_controller')
        # --- Control loop frequency parameter ---
        self.control_frequency = float(
            self.declare_parameter('control_frequency', DEFAULT_CONTROL_FREQUENCY).value
        )
        self.get_logger().info(f"PD control frequency: {self.control_frequency} Hz")

        # --- Gains / Limits (전역 상수 그대로 사용) ---
        self.kp = DEFAULT_KP
        self.kd = DEFAULT_KD
        self.lpf_alpha = DEFAULT_LPF_ALPHA
        self.max_torque = DEFAULT_MAX_TORQUE

        self.get_logger().info(f"Using Gains: Kp={self.kp}, Kd={self.kd}")
        self.get_logger().info(f"LPF Alpha: {self.lpf_alpha}, Max Torque: {self.max_torque}")

        # --- State Variables (deg 기준) ---
        self.current_angles_deg = np.zeros(NUM_JOINTS)        # 현재 관절 각도 [deg]
        self.current_velocities_deg_s = np.zeros(NUM_JOINTS)  # 현재 관절 각속도 [deg/s]
        self.target_angles_deg = np.zeros(NUM_JOINTS)         # 목표 각도 [deg]

        self.previous_torque_command = np.zeros(NUM_JOINTS)

        self.last_angle_update_time = None
        self.last_angles_deg = None

        # --- ROS2 Communications ---
        self.create_subscription(MotorStates, MOTOR_STATES_TOPIC, self.motor_states_callback, 100)
        self.create_subscription(Float32MultiArray, TARGET_ANGLES_TOPIC, self.target_angles_callback, 100)
        self.torque_publisher = self.create_publisher(Float32MultiArray, TORQUE_COMMANDS_TOPIC, 100)

        # --- Controller Timer ---
        # --- Controller Timer ---
        timer_period = 1.0 / self.control_frequency
        self.timer = self.create_timer(timer_period, self.controller_callback)

    def motor_states_callback(self, msg: MotorStates):
        """
        MotorStates로부터 multi_turn_raw를 받아 현재 각도/각속도를 업데이트.
        - 입력 단위: 0.001 deg / LSB
        - 내부 단위: degree, deg/s
        """
        # Data is received in 0.001 degrees per LSB
        raw_angles_deg = np.array(msg.multi_turn_raw, dtype=float) * 0.001

        # 개수 맞추기 (padding / truncating)
        if len(raw_angles_deg) != NUM_JOINTS:
            self.get_logger().warn(
                f"Received {len(raw_angles_deg)} joint states, expected {NUM_JOINTS}. Padding/truncating."
            )
            padded_angles = np.zeros(NUM_JOINTS)
            num_to_copy = min(len(raw_angles_deg), NUM_JOINTS)
            padded_angles[:num_to_copy] = raw_angles_deg[:num_to_copy]
            self.current_angles_deg = padded_angles
        else:
            self.current_angles_deg = raw_angles_deg

        # --- Velocity Estimation (Finite Difference, deg/s) ---
        now = self.get_clock().now()
        if self.last_angles_deg is not None and self.last_angle_update_time is not None:
            dt = (now.nanoseconds - self.last_angle_update_time.nanoseconds) / 1e9
            if dt > 1e-6:
                self.current_velocities_deg_s = (self.current_angles_deg - self.last_angles_deg) / dt

        self.last_angles_deg = self.current_angles_deg.copy()
        self.last_angle_update_time = now

    def target_angles_callback(self, msg: Float32MultiArray):
        """
        목표 각도 업데이트.
        현재는 관절 테스트용으로 4번 관절만 30 [deg]로 고정.
        (msg는 무시)
        """
        target_angles_deg = np.zeros(NUM_JOINTS)
        target_angles_deg[3] = JOINT_DEGREE  # 4번 관절만 30도
        self.target_angles_deg = target_angles_deg

    def controller_callback(self):
        """
        PD 제어 루프 (deg 단위).
        - position_error: [deg]
        - velocity_error: [deg/s]
        - torque: [Nm] (단위는 실제 튜닝에 따라 해석)
        """
        # --- PD Control Law ---
        position_error = self.target_angles_deg - self.current_angles_deg
        velocity_error = -self.current_velocities_deg_s

        raw_torque_command = self.kp * position_error + self.kd * velocity_error

        # 디버깅용 로그 (원하면 주석 처리)
        self.get_logger().info(f"position_error(deg): {position_error}")
        self.get_logger().info(f"velocity_error(deg/s): {velocity_error}")

        # --- Low-Pass Filter (LPF) ---
        # smoothed_torque = alpha * new_value + (1 - alpha) * old_value
        filtered_torque = (
            self.lpf_alpha * raw_torque_command
            + (1.0 - self.lpf_alpha) * self.previous_torque_command
        )

        # --- Torque Clipping ---
        clipped_torque = np.clip(filtered_torque, -self.max_torque, self.max_torque)

        # 하나의 관절(여기선 index 3)만 토크 사용, 나머지는 0
        torque_output = np.zeros(NUM_JOINTS)
        torque_output[3] = clipped_torque[3]

        # 다음 LPF를 위해 저장
        self.previous_torque_command = torque_output

        # --- Publish Command ---
        torque_msg = Float32MultiArray()
        torque_msg.data = torque_output.flatten().tolist()
        self.torque_publisher.publish(torque_msg)

        self.get_logger().info(f"Published Torque Command: {torque_msg.data}")


def main(args=None):
    rclpy.init(args=args)
    pd_controller = PDController()

    try:
        rclpy.spin(pd_controller)
    except KeyboardInterrupt:
        pass
    finally:
        pd_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
