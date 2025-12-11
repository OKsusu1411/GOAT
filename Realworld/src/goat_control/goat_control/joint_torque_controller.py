import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import termios
import tty
import sys

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
        self.publisher_ = self.create_publisher(Float32MultiArray, 'torque_commands', 10)
        self.num_motors = self.declare_parameter('num_motors', 8).value
        self.joint_to_control = self.declare_parameter('joint_to_control', 0).value
        self.torque_step = self.declare_parameter('torque_step', 0.1).value
        self.torques = [0.0] * self.num_motors
        self.initial_message = (
            f"Controlling joint: {self.joint_to_control}\n"
            f"Use arrow keys to control torque. 'q' to quit."
        )
        print(self.initial_message)
        self.run()

    def run(self):
        while rclpy.ok():
            key = getch()
            if key == 'q':
                break
            elif key == '\x1b':  # Arrow key prefix
                next_key_1 = getch()
                next_key_2 = getch()
                if next_key_1 == '[':
                    if next_key_2 == 'A':  # Up arrow
                        self.torques[self.joint_to_control] += self.torque_step
                    elif next_key_2 == 'B':  # Down arrow
                        self.torques[self.joint_to_control] -= self.torque_step
            
            self.publish_torques()
            sys.stdout.write("\033c")
            print(self.initial_message)
            print(f"Torques: {[f'{t:.2f}' for t in self.torques]}")

    def publish_torques(self):
        msg = Float32MultiArray()
        msg.data = self.torques
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
