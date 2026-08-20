from __future__ import annotations

import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

from .utils import ros_bridge
from .utils.mujoco_sim import MujocoSim, SimConfig


class GoatMujocoNode(Node):
    def __init__(self) -> None:
        super().__init__('goat_mujoco_node')

        # Route mujoco_sim/ros_bridge stdlib logging into rosout so the load-time
        # model inspection and resolver warnings are visible under ros2 launch.
        ros_bridge.install_ros_logging_bridge(self)

        self.declare_parameter('model_path', '')
        self.declare_parameter('timestep', 0.0)
        self.declare_parameter('steps_per_cmd', 1)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('render_sleep', True)
        self.declare_parameter('home_keyframe', '')
        self.declare_parameter('joint_order', [], ParameterDescriptor(dynamic_typing=True))

        # Parameters
        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise RuntimeError('Parameter "model_path" must be set.')
        timestep = float(self.get_parameter('timestep').value)
        if timestep <= 0.0:
            raise RuntimeError('Parameter "timestep" must be set (> 0) from yaml/launch.')
        self._steps_per_cmd = max(1, int(self.get_parameter('steps_per_cmd').value))
        self._use_viewer = bool(self.get_parameter('use_viewer').value)
        self._render_sleep = bool(self.get_parameter('render_sleep').value)
        home_keyframe = self.get_parameter('home_keyframe').value or None
        joint_order = list(self.get_parameter('joint_order').value) or None

        # MuJoCo instance
        self.sim = MujocoSim(SimConfig(
            model_path=model_path,
            use_viewer=self._use_viewer,
            home_keyframe=home_keyframe,
            timestep=timestep,
            joint_order=joint_order,
        ))
        self.sim.reset()  # MujocoSim logs the model summary at load time
        if self._use_viewer:
            self.sim.open_viewer()

        # Wall-clock target for optional render pacing.
        self._control_period = self._steps_per_cmd * self.sim.timestep
        self._shutting_down = False

        # Subscriber
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_sub = self.create_subscription(JointState, 'commands', self._on_cmd, cmd_qos)

        # Publisher
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joint_pub = self.create_publisher(JointState, 'sim_joint_states', state_qos)
        self.imu_pub = self.create_publisher(Imu, 'sim_imu', state_qos)
        self.odom_pub = self.create_publisher(Odometry, 'sim_odom', state_qos)
        self.clock_pub = self.create_publisher(Clock, 'clock', state_qos)

        # Initial publish
        self._publish_state()

        # Housekeeping only: watch quit/viewer-close. Never advances the sim.
        self._housekeep_timer = self.create_timer(0.1, self._on_housekeep)

        self.get_logger().info(
            f'goat_mujoco_node up: model={model_path}, '
            f'steps_per_cmd={self._steps_per_cmd}, use_viewer={self._use_viewer}'
        )

    # ------------------------------------------------------------------ #
    # Main loop: one /cmd == one control step
    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: JointState) -> None:
        if self.sim.is_quit_requested:
            self._shutdown()
            return

        if self.sim.consume_reset_request():
            self.sim.reset()

        self.sim.set_ctrl(ros_bridge.cmd_to_ctrl(msg, self.sim))

        wall_start = time.monotonic()
        if not self.sim.is_paused:
            self.sim.step(self._steps_per_cmd)

        self._publish_state()

        if self._use_viewer:
            if not self.sim.is_viewer_running:
                self._shutdown()
                return
            self.sim.sync()

        if self._render_sleep and self._use_viewer:
            remaining = self._control_period - (time.monotonic() - wall_start)
            if remaining > 0:
                time.sleep(remaining)

    def _publish_state(self) -> None:
        """Publish clock + joint/imu state for the current MjData.

        Called both after the initial reset (the loop seed) and after every
        control step, so the wire format is identical in both cases.
        """
        stamp = ros_bridge.sim_time_to_msg(self.sim.sim_time)
        self.clock_pub.publish(Clock(clock=stamp))
        self.joint_pub.publish(ros_bridge.joint_state_msg(self.sim, stamp))
        self.imu_pub.publish(ros_bridge.imu_msg(self.sim, stamp))
        self.odom_pub.publish(ros_bridge.odom_msg(self.sim, stamp))

    # ------------------------------------------------------------------ #
    # Housekeeping: quit / viewer close (no stepping)
    # ------------------------------------------------------------------ #
    def _on_housekeep(self) -> None:
        """Keep the viewer usable while NO commands flow -- never steps.

        Lockstep means /commands is the only trigger that advances physics, but
        quit / reset / viewer refresh must still respond when the controller is
        idle (otherwise ``r`` and the rendered state look dead until the next
        command). This consumes quit and reset (reset + republish, no step) and
        re-syncs the viewer. It never calls ``sim.step()``, so sim-time and
        determinism are unchanged.
        """
        if self.sim.is_quit_requested:
            self._shutdown()
            return
        if self._use_viewer and not self.sim.is_viewer_running:
            self._shutdown()
            return

        if self.sim.consume_reset_request():
            self.sim.reset()

        if self._use_viewer:
            self.sim.sync()

        self._publish_state()

    @property
    def shutdown_requested(self) -> bool:
        return self._shutting_down

    def _shutdown(self) -> None:
        """Request shutdown: close the viewer and flag the spin loop to exit.

        This only flips state. The actual context teardown (destroy_node then
        rclpy.shutdown, in that order) is owned by main(), so we never tear the
        context down from inside a callback the context is still driving.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info('shutting down goat_mujoco_node')
        self.sim.close_viewer()

    def destroy_node(self) -> bool:
        self.sim.close_viewer()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoatMujocoNode()
    try:
        # Manual spin so a viewer-close / quit (which only sets a flag) breaks
        # the loop; teardown below then runs in the correct order.
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
