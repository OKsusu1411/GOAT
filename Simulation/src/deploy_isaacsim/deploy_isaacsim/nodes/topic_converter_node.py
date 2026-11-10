import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from motor_interfaces.msg import ImuState
from message_filters import Subscriber, TimeSynchronizer


class TopicConverterNode(Node):
    """
    Bridge between simulation and controller.

    Sim -> Controller:
        /sim_joint_states + /sim_imu + /sim_odom
            -> strict TimeSynchronizer
            -> /joint_states + /imu

        - Source headers are preserved.
        - Since ActionGraph already guarantees equal stamps:
            sim_joint_states.header.stamp == sim_imu.header.stamp == sim_odom.header.stamp
          the converted /joint_states and /imu will also have equal stamps.

    Controller -> Sim:
        /commands -> /sim_commands
        - Immediate relay.
        - Command header is preserved.
        - Command is not synchronized with sensor messages.
    """

    def __init__(self):
        super().__init__("topic_converter_node")

        # ------------------------------------------------------------------
        # Config
        # ------------------------------------------------------------------
        self.declare_parameter("sync_queue_size", 10)

        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        # Isaac Sim ROS2 bridge commonly uses RELIABLE + VOLATILE.
        # KEEP_LAST is safer than KEEP_ALL for control pipelines because old
        # packets should not accumulate.
        sensor_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=self.sync_queue_size,
        )

        command_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ------------------------------------------------------------------
        # Synchronized subscribers: simulation sensors
        # ------------------------------------------------------------------
        self.sim_joint_state_subscriber = Subscriber(
            self,
            JointState,
            "/sim_joint_states",
            qos_profile=sensor_qos_profile,
        )

        self.sim_imu_state_subscriber = Subscriber(
            self,
            Imu,
            "/sim_imu",
            qos_profile=sensor_qos_profile,
        )

        self.sim_odom_subscriber = Subscriber(
            self,
            Odometry,
            "/sim_odom",
            qos_profile=sensor_qos_profile,
        )

        self.sensor_sync = TimeSynchronizer(
            [
                self.sim_joint_state_subscriber,
                self.sim_imu_state_subscriber,
                self.sim_odom_subscriber,
            ],
            self.sync_queue_size,
        )
        self.sensor_sync.registerCallback(self.synced_sensor_callback)

        # ------------------------------------------------------------------
        # Subscriber: controller command
        # ------------------------------------------------------------------
        self.command_subscriber = self.create_subscription(
            JointState,
            "/commands",
            self.command_callback,
            qos_profile=command_qos_profile,
        )

        # ------------------------------------------------------------------
        # Publishers: controller-side sensor topics
        # ------------------------------------------------------------------
        self.real_joint_state_publisher = self.create_publisher(
            JointState,
            "/joint_states",
            qos_profile=sensor_qos_profile,
        )

        self.real_imu_state_publisher = self.create_publisher(
            ImuState,
            "/imu",
            qos_profile=sensor_qos_profile,
        )

        # ------------------------------------------------------------------
        # Publisher: simulation-side command topic
        # ------------------------------------------------------------------
        self.sim_joint_command_publisher = self.create_publisher(
            JointState,
            "/sim_commands",
            qos_profile=command_qos_profile,
        )

        self.get_logger().info(
            "TopicConverterNode started with strict TimeSynchronizer: "
            "/sim_joint_states + /sim_imu + /sim_odom -> /joint_states + /imu"
        )

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _same_stamp(a, b) -> bool:
        return a.sec == b.sec and a.nanosec == b.nanosec

    # ----------------------------------------------------------------------
    # Synchronized sensor callback
    # ----------------------------------------------------------------------

    def synced_sensor_callback(
        self,
        joint_msg: JointState,
        imu_msg: Imu,
        odom_msg: Odometry,
    ) -> None:
        """
        Convert synchronized simulation sensor messages.

        Important:
            - Triggered only when /sim_joint_states, /sim_imu, /sim_odom
              have identical header.stamp.
            - Preserve full source headers.
            - One synchronized input bundle produces one /joint_states and one /imu.
        """

        # This should always hold under the confirmed ActionGraph setup.
        # Keep the check as a guard/logging aid.
        if not (
            self._same_stamp(joint_msg.header.stamp, imu_msg.header.stamp)
            and self._same_stamp(joint_msg.header.stamp, odom_msg.header.stamp)
        ):
            self.get_logger().error(
                "[CONVERTER_SYNC_ERROR] TimeSynchronizer callback received "
                "non-identical stamps. This should not happen with strict sync.\r"
            )
            return

        # --------------------------------------------------------------
        # /sim_joint_states -> /joint_states
        # Preserve the full source header exactly.
        # --------------------------------------------------------------
        joint_out = JointState()
        joint_out.header = joint_msg.header
        joint_out.name = ["hip_L_Joint", "hip_R_Joint"] + list(joint_msg.name)
        joint_out.position = [0.0, 0.0] + list(joint_msg.position)
        joint_out.velocity = [0.0, 0.0] + list(joint_msg.velocity)
        joint_out.effort = [0.0, 0.0] + list(joint_msg.effort)

        # --------------------------------------------------------------
        # /sim_imu + /sim_odom -> /imu
        # Preserve the full /sim_imu header exactly.
        # Because the input bundle is strictly synchronized, imu_out.header.stamp
        # is identical to joint_out.header.stamp.
        # --------------------------------------------------------------
        imu_out = ImuState()
        imu_out.header = imu_msg.header
        imu_out.quat = imu_msg.orientation
        imu_out.gyro = imu_msg.angular_velocity
        imu_out.vel = odom_msg.twist.twist.linear

        self.real_joint_state_publisher.publish(joint_out)
        self.real_imu_state_publisher.publish(imu_out)

    # ----------------------------------------------------------------------
    # Command callback: relay immediately
    # ----------------------------------------------------------------------

    def command_callback(self, msg: JointState) -> None:
        """
        Relay /commands -> /sim_commands immediately.

        Important:
            - Preserve controller command header exactly.
            - Do not synchronize command with sensor messages.
            - Do not create another control clock here.
        """
        sim_cmd_msg = JointState()
        sim_cmd_msg.header = msg.header
        sim_cmd_msg.name = list(msg.name[2:])
        sim_cmd_msg.position = list(msg.position[2:])
        sim_cmd_msg.velocity = list(msg.velocity[2:])
        sim_cmd_msg.effort = list(msg.effort[2:])

        self.sim_joint_command_publisher.publish(sim_cmd_msg)


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


if __name__ == "__main__":
    main()