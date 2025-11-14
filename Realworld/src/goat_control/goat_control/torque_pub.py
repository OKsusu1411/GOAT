import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pynput import keyboard

class TorqueTestPublisher(Node):
    def __init__(self):
        super().__init__('torque_test_publisher')
        self.publisher = self.create_publisher(Float32MultiArray, 'torque_commands', 10)
        timer_period = 0.1  # 10Hz (100ms)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.torque = 0.0
        self.command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mass = 0.0
        self.joint_length = 0.252  # m
        self.wheel_radius = 0.252  # m
    
    def joint_torque2current(self, torque):
        if torque != 0.0:
            current = (torque-0.0081)/0.2552  # for joint

        else:
            current = 0.0
        return current
    def wheel_torque2current(self, torque):
        if torque != 0.0:
            current = (torque-0.0052)/(0.2569)  # for wheel
        else:
            current = 0.0
        return current
    
    def current2mass(self, current, jointorwheel):
        if jointorwheel == self.joint_length:
            self.mass = (0.2552*current+0.0081)*1000/(self.joint_length*9.81)
        elif jointorwheel == self.wheel_radius:
            self.mass = (0.2569*current+0.0052)*1000/(self.wheel_radius*9.81)
    #모터 테스트용 키보드 입력 함수들
    def increase_torque(self):
        self.torque += 0.2
        self.update_command()

    def decrease_torque(self):
        self.torque -= 0.2
        self.update_command()

    def reset_torque(self):
        self.torque = 0.0
        self.update_command()

    def update_command(self):
        self.command[4] = self.joint_torque2current(self.torque)
        #self.command[1] = self.torque
        self.current2mass(abs(self.command[1]), self.joint_length)
    def command_update(self, comm):
        comm = [self.joint_torque2current(comm[0]), self.joint_torque2current(comm[1]), self.joint_torque2current(comm[2]), self.wheel_torque2current(comm[3]), \
                self.joint_torque2current(comm[4]), self.joint_torque2current(comm[5]), self.joint_torque2current(comm[6]), self.wheel_torque2current(comm[7])]
        self.command = comm

    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = self.command
        self.publisher.publish(msg)
        self.get_logger().info(f'Published torque: {self.torque:.2f} Nm, commands: {self.command} mass : {self.mass} g')

def on_press(key, node):
    try:
        if key == keyboard.Key.up:
            node.increase_torque()
        elif key == keyboard.Key.down:
            node.decrease_torque()
        elif hasattr(key, 'char') and key.char == 's':
            node.reset_torque()
    except Exception as e:
        print(f"Error on key press: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TorqueTestPublisher()

    listener = keyboard.Listener(on_press=lambda key: on_press(key, node))
    listener.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
