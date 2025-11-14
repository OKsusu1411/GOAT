import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from .utils.can_mixin import CanMixin
import struct
import time
import can

# Current-to-LSB conversion ratio per motor series 
# SCALE_A_PER_LSB = {
#     "MG": 33.0 / 2048.0,   # MG series: ≈ 0.01611 A/LSB (±33.0A range)
# }
SCALE_A_PER_LSB = 66.0 / 4096.0   # MG: ≈ 0.01611 A/LSB


class MotorTorqueController(Node, CanMixin):
    def __init__(self):
        super().__init__('motor_torque_controller')
        
        # Declare and load parameters
        self.channel = self.declare_parameter('channel', 'can0').value
        self.bitrate = self.declare_parameter('bitrate', 1000000).value
        self.interface = self.declare_parameter('interface', 'socketcan').value   # CAN interface type
        # self.series = self.declare_parameter('series', 'MG').value
        self.num_motors = self.declare_parameter('num_motors', 8).value
        self.control_frequency = self.declare_parameter('control_frequency', 50.0).value
        self.timeout_sec = self.declare_parameter('timeout_sec', 0.5).value
        self.scale = SCALE_A_PER_LSB  # Current scaling factor depending on motor series

        # Open CAN bus
        # NOTE: The user must activate the CAN interface before running this node.
        # Example: sudo ip link set can0 up type can bitrate 1000000
        self.get_logger().info(f"Attempting to open CAN bus on channel '{self.channel}' with interface '{self.interface}'...")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        except Exception as e:
            self.get_logger().error(f"Failed to open CAN bus on {self.channel}: {e}")
            raise e

        # Motor initialization: clear errors and enable drive
        for node_id in range(1, self.num_motors + 1):
            self._send_command_expect(node_id, 0x9B)      # CLEAR ERRORS (0x9B)
            response = self._send_command_expect(node_id, 0x88)  # RUN (Enable motor, 0x88)
            if response:
                self.get_logger().info(f"Motor {node_id:02d}: RUN command acknowledged.")
            else:
                self.get_logger().warn(f"Motor {node_id:02d}: No response to RUN command.")

        # Initialize current command storage and safety flags
        self.current_commands = [0.0] * self.num_motors
        self.last_command_time = None
        self.got_command = False
        self.safe_mode = False

        # Subscribe to torque command topic (Float32MultiArray)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'torque_commands',   # Topic name (modifiable if needed)
            self.command_callback,
            10
        )

        # Create control loop timer (periodic torque transmission)
        timer_period = 1.0 / self.control_frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _send_command_expect(self, node_id: int, cmd_byte: int, payload7: bytes = b"\x00" * 7):
        """Send a specific CAN command and wait briefly for a response."""
        tx_id = 0x140 + node_id
        rx_id = 0x180 + node_id
        data = bytes([cmd_byte]) + payload7
        msg = can.Message(arbitration_id=tx_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
        except can.CanError as e:
            self.get_logger().error(f"CAN send failed for ID {node_id}: {e}")
            return None

        # Wait up to 0.3 seconds for a response frame
        end_time = time.time() + 0.3
        while time.time() < end_time:
            rx_msg = self.bus.recv(timeout=0.1)
            if rx_msg is None:
                continue
            if rx_msg.arbitration_id == tx_id and len(rx_msg.data) == 8 and rx_msg.data[0] == cmd_byte:
                return rx_msg
        return None

    def command_callback(self, msg: Float32MultiArray):
        """Callback when a new torque command message is received."""
        commands = list(msg.data)
        # Adjust list size to match number of motors (pad or truncate as needed)
        if len(commands) < self.num_motors:
            commands.extend([0.0] * (self.num_motors - len(commands)))
        elif len(commands) > self.num_motors:
            commands = commands[:self.num_motors]

        # Update current commands and timestamp
        self.current_commands = commands
        self.last_command_time = time.time()
        self.got_command = True

        # Exit safe mode if previously triggered
        if self.safe_mode:
            self.get_logger().info("Received new commands. Exiting safe mode.")
            self.safe_mode = False

    def timer_callback(self):
        """Periodic timer callback — sends torque (current) commands over CAN."""
        now = time.time()

        # Enter safe mode if no command received within timeout_sec
        if self.got_command and (now - self.last_command_time > self.timeout_sec):
            if not self.safe_mode:
                # Enter safe mode once
                self.get_logger().warn(f"No command received for {self.timeout_sec:.2f} seconds. Entering safe mode (sending zero torque).")
                self.current_commands = [0.0] * self.num_motors
                self.safe_mode = True

        # Send torque/current commands to each motor
        for i in range(self.num_motors):
            node_id = i + 1  # Motor ID (index 0 → ID1, 1 → ID2, ...)
            amps = self.current_commands[i]
            # Example (manual frame composition, replaced by CanMixin):
            # iq = int(round(amps / self.scale))
            # iq = max(min(iq, 2048), -2048)
            # iq_bytes = struct.pack("<h", iq)
            # data = bytes([0xA1]) + b"\x00\x00\x00" + iq_bytes + b"\x00\x00"
            # msg = can.Message(arbitration_id=(0x140 + node_id), data=data, is_extended_id=False)
            # try:
            #     self.bus.send(msg)
            # except can.CanError as e:
            #     self.get_logger().error(f"Failed to send torque to motor {node_id}: {e}")

            # Use CanMixin-provided torque command and TX/RX logic
            resp = self.cmd_torque_mode(node_id, amps, timeout=0.02)
            if not resp:
                self._log().debug(f"[CAN] No response to torque cmd (id={node_id}, A={amps:.3f})")

        # (Optional) Periodic state read for debugging
        # Example: using CanMixin state read wrapper
        # state = self.cmd_read_state2(node_id=1, timeout=0.02)
        # if state:
        #     self.get_logger().debug(f"state2: {state.data.hex(' ')}")

    def destroy_node(self):
        """Cleanup CAN bus when shutting down the node."""
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
