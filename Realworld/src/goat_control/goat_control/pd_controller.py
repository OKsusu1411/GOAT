import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from motor_interfaces.msg import MotorStates
import numpy as np

# --- Constants ---
# Default PD gains
DEFAULT_KP = 0.0000000000005  # Proportional gain
DEFAULT_KD = 0.2   # Derivative gain
DEFAULT_LPF_ALPHA = 0.8 # Low-pass filter alpha
DEFAULT_MAX_TORQUE = 10.0 # Maximum torque limit
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
    A simple PD controller for multiple joints.
    This node subscribes to the current joint states (position and velocity)
    and target joint positions. It calculates the required torques using a
    PD control law, applies a low-pass filter, clips the output, and
    then publishes the final command.
    """
    def __init__(self):
        super().__init__('pd_controller')

        # --- Parameters ---
        self.declare_parameter('kp', DEFAULT_KP)
        self.declare_parameter('kd', DEFAULT_KD)
        self.declare_parameter('lpf_alpha', DEFAULT_LPF_ALPHA)
        self.declare_parameter('max_torque', DEFAULT_MAX_TORQUE)
        
        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        self.lpf_alpha = self.get_parameter('lpf_alpha').get_parameter_value().double_value
        self.max_torque = self.get_parameter('max_torque').get_parameter_value().double_value
        
        self.get_logger().info(f"Using Gains: Kp={self.kp}, Kd={self.kd}")
        self.get_logger().info(f"LPF Alpha: {self.lpf_alpha}, Max Torque: {self.max_torque}")

        # --- State Variables ---
        self.current_angles_rad = np.zeros(NUM_JOINTS)
        self.current_velocities_rad_s = np.zeros(NUM_JOINTS)
        self.target_angles_rad = np.zeros(NUM_JOINTS)
        self.previous_torque_command = np.zeros(NUM_JOINTS)
        
        self.last_angle_update_time = None
        self.last_angles_rad = None

        # --- ROS2 Communications ---
        self.create_subscription(MotorStates, MOTOR_STATES_TOPIC, self.motor_states_callback, 10)
        self.create_subscription(Float32MultiArray, TARGET_ANGLES_TOPIC, self.target_angles_callback, 10)
        self.torque_publisher = self.create_publisher(Float32MultiArray, TORQUE_COMMANDS_TOPIC, 10)

        # --- Controller Timer ---
        self.timer = self.create_timer(CONTROLLER_TIMER_PERIOD, self.controller_callback)

    def motor_states_callback(self, msg: MotorStates):
        """
        Processes incoming joint states, converting them to radians
        and estimating velocity.
        """
        # Data is received in 0.01 degrees per LSB
        raw_angles_deg = np.array(msg.multi_turn_raw) * 0.01
        
        if len(raw_angles_deg) != NUM_JOINTS:
            self.get_logger().warn(f"Received {len(raw_angles_deg)} joint states, expected {NUM_JOINTS}. Padding/truncating.")
            padded_angles = np.zeros(NUM_JOINTS)
            num_to_copy = min(len(raw_angles_deg), NUM_JOINTS)
            padded_angles[:num_to_copy] = raw_angles_deg[:num_to_copy]
            self.current_angles_rad = np.deg2rad(padded_angles)
        else:
            self.current_angles_rad = np.deg2rad(raw_angles_deg)

        # --- Velocity Estimation (Finite Difference) ---
        now = self.get_clock().now()
        if self.last_angles_rad is not None and self.last_angle_update_time is not None:
            dt = (now.nanoseconds - self.last_angle_update_time.nanoseconds) / 1e9
            if dt > 1e-6:
                self.current_velocities_rad_s = (self.current_angles_rad - self.last_angles_rad) / dt
        
        self.last_angles_rad = self.current_angles_rad
        self.last_angle_update_time = now

    def target_angles_callback(self, msg: Float32MultiArray):
        """
        Processes incoming target joint angles.
        Assumes target angles are provided in radians.
        """
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(f"Received {len(msg.data)} target angles, expected {NUM_JOINTS}. Ignoring.")
            return
        self.target_angles_rad = np.array(msg.data, dtype=np.float32)

    def controller_callback(self):
        """
        The main PD control loop with LPF and clipping.
        """
        # --- PD Control Law ---
        position_error = self.target_angles_rad - self.current_angles_rad
        velocity_error = -self.current_velocities_rad_s
        raw_torque_command = self.kp * position_error + self.kd * velocity_error
        
        # --- Low-Pass Filter (LPF) ---
        # smoothed_torque = alpha * new_value + (1 - alpha) * old_value
        filtered_torque = self.lpf_alpha * raw_torque_command + (1 - self.lpf_alpha) * self.previous_torque_command
        
        # --- Torque Clipping ---
        clipped_torque = np.clip(filtered_torque, -self.max_torque, self.max_torque)
        
        # --- Update State for next iteration ---
        self.previous_torque_command = clipped_torque
        
        # --- Publish Command ---
        torque_msg = Float32MultiArray()
        torque_msg.data = clipped_torque.flatten().tolist()
        self.torque_publisher.publish(torque_msg)
        
        # Optional: Log for debugging
        # self.get_logger().info(f"Torque Out: {clipped_torque}")

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
