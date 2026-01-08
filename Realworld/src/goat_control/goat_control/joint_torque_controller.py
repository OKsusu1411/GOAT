import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import termios
import tty
import sys
import threading
import time

JOINT_NUM=5

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

class JointTorqueController(Node):
    def __init__(self):
        super().__init__('joint_torque_controller')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'goat/action', 10)
        self.num_motors = self.declare_parameter('num_motors', 8).value
        self.joint_to_control = self.declare_parameter('joint_to_control', JOINT_NUM).value
        self.torque_step = self.declare_parameter('torque_step', 0.1).value
        self.torques = [0.0] * self.num_motors
        self.initial_message = (
            f"Controlling joint: {self.joint_to_control}\n"
            f"Use arrow keys to control torque. 'q' to quit."
        )
        print(self.initial_message)

        # Create a timer to publish torques periodically
        self.timer = self.create_timer(0.1, self.publish_torques) # 10Hz

        # Start a thread to listen for keyboard input
        self.key_thread = threading.Thread(target=self.key_listener_thread)
        self.key_thread.daemon = True
        self.key_thread.start()

    def key_listener_thread(self):
        while rclpy.ok():
            key = getch()
            if key == 'q':
                self.get_logger().info("'q' pressed, initiating shutdown.")
                rclpy.shutdown()
                break
            elif key == '\x1b':  # Arrow key prefix
                next_key_1 = getch()
                next_key_2 = getch()
                if next_key_1 == '[':
                    if next_key_2 == 'A':  # Up arrow
                        self.torques[self.joint_to_control] += self.torque_step
                    elif next_key_2 == 'B':  # Down arrow
                        self.torques[self.joint_to_control] -= self.torque_step

    def publish_torques(self):
        if not rclpy.ok():
            return
        msg = Float32MultiArray()
        msg.data = self.torques
        self.publisher_.publish(msg)
        sys.stdout.write("\033c")
        print(self.initial_message)
        print(f"Torques: {[f'{t:.2f}' for t in self.torques]}")

def main(args=None):
    rclpy.init(args=args)
    node = JointTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    
    # Cleanup, executed after rclpy.spin() returns
    node.get_logger().info('Resetting torques to 0.')
    # Set torques to zero
    node.torques = [0.0] * node.num_motors
    # Publish zero torques one last time
    node.publish_torques()
    # Give a moment for the message to go out
    time.sleep(0.1)

    node.destroy_node()
    # This is important to ensure rclpy context is properly cleaned up
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()