import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from motor_interfaces.msg import ImuState

from message_filters import Subscriber, ApproximateTimeSynchronizer

class TopicConverterNode(Node):
    """
    Virtual Topic Converter for HIL Test

        Sim:
            JointState : /sim_joint_states
            Imu        : /sim_imu
            Odometry   : /odom

        Real:
            JointState : /joint_states
            ImuState   : /imu

    """
    def __init__(self):
        super().__init__("topic_converter_node")

        # Message
        self.joint_msg = None
        self.imu_msg = None
        self.odom_msg = None

        # Subscriber
        self.sim_joint_state_subscriber = Subscriber(self, JointState, '/sim_joint_states', 10)
        self.sim_imu_state_subscriber = Subscriber(self, Imu, '/sim_imu', 10)
        self.sim_velocity_state_subscriber = Subscriber(self, Odometry, '/odom', 10)

        # Publisher
        self.real_joint_state_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.real_imu_state_publisher = self.create_publisher(ImuState, '/imu', 10)

        # Syncronizer
        self.time_sync = ApproximateTimeSynchronizer([self.sim_joint_state_subscriber, self.sim_imu_state_subscriber, self.sim_velocity_state_subscriber], 10, 0.01)
        self.time_sync.registerCallback(self.sync_callback)

    def sync_callback(self, joint_msg, imu_msg, odom_msg):
        self.joint_callback(joint_msg)
        self.imu_callback(imu_msg)
        self.odom_callback(odom_msg)
        self.convert()

    def joint_callback(self, msg):
        self.joint_msg = msg

    def imu_callback(self, msg):
        self.imu_msg = msg

    def odom_callback(self, msg):
        self.odom_msg = msg
    
    def convert(self):
        if self.joint_msg is None or self.imu_msg is None or self.odom_msg is None:
            return
        # Header
        time_stamp = self.get_clock().now().to_msg()
        # Joint msg
        joint_msg: JointState = self.joint_msg
        joint_msg.header.stamp = time_stamp
        joint_msg.header.frame_id = 'joint_states'
        # Imu msg
        imu_msg = ImuState()
        imu_msg.header.stamp = time_stamp
        imu_msg.header.frame_id = 'imu'
        imu_msg.quat = self.imu_msg.orientation
        imu_msg.gyro = self.imu_msg.angular_velocity
        imu_msg.vel  = self.odom_msg.twist.twist.linear
        # Publish
        self.real_joint_state_publisher.publish(joint_msg)
        self.real_imu_state_publisher.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TopicConverterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()