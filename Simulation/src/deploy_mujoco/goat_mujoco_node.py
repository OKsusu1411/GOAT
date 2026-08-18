"""MuJoCo simulator node with physics on its own thread, ROS I/O on others.

Physics no longer runs inside a subscription callback. ``SimWorker`` owns a
dedicated sim thread that ticks at the control period; the ROS side only ever
touches two lock-guarded mailboxes:

    /commands  --> _on_cmd ----(cmd lock)---->  [latest ctrl]  --> sim thread
    sim thread --(data lock: set_ctrl/step/snapshot/sync)--> [latest snapshot]
                                --(snap lock)--> _on_publish --> /sim_*, /clock

So a slow DDS publish can no longer stall the integrator, a burst of commands
can no longer pile up a backlog of steps, and MjData is touched by exactly one
thread at a time. Locks are held only for a reference swap; nothing that can
block (publishing, sleeping, rendering) happens inside one. See solution.md for
the full lock ordering and the invariants each critical section keeps.
"""
from __future__ import annotations

import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

from utils import ros_bridge
from utils.mujoco_sim import MujocoSim, SimConfig
from utils.sim_worker import SimWorker


class GoatMujocoNode(Node):
    def __init__(self) -> None:
        super().__init__('goat_mujoco_node')

        # Route mujoco_sim/sim_worker/ros_bridge stdlib logging into rosout so
        # the load-time model inspection, the resolver warnings and any sim
        # thread traceback are visible under ros2 launch.
        ros_bridge.install_ros_logging_bridge(self)

        self.declare_parameter('model_path', '')
        self.declare_parameter('timestep', 0.0)
        self.declare_parameter('steps_per_cmd', 1)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('realtime', True)
        self.declare_parameter('publish_rate_hz', 0.0)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('home_keyframe', '')
        self.declare_parameter('joint_order', [], ParameterDescriptor(dynamic_typing=True))

        # Parameters
        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise RuntimeError('Parameter "model_path" must be set.')
        timestep = float(self.get_parameter('timestep').value)
        if timestep <= 0.0:
            raise RuntimeError('Parameter "timestep" must be set (> 0) from yaml/launch.')
        steps_per_cmd = max(1, int(self.get_parameter('steps_per_cmd').value))
        self._use_viewer = bool(self.get_parameter('use_viewer').value)
        realtime = bool(self.get_parameter('realtime').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        home_keyframe = self.get_parameter('home_keyframe').value or None
        joint_order = list(self.get_parameter('joint_order').value) or None

        # MuJoCo instance. Loaded here (so a bad model fails at construction,
        # on this thread), but from now on MjData belongs to the sim thread and
        # is reachable only through the worker's lock.
        self.sim = MujocoSim(SimConfig(
            model_path=model_path,
            use_viewer=self._use_viewer,
            home_keyframe=home_keyframe,
            timestep=timestep,
            joint_order=joint_order,
        ))
        self.sim.reset()  # MujocoSim logs the model summary at load time

        self.worker = SimWorker(
            self.sim,
            steps_per_cmd=steps_per_cmd,
            use_viewer=self._use_viewer,
            realtime=realtime,
            cmd_timeout=cmd_timeout,
        )
        self._shutting_down = False
        self._last_pub_seq = -1

        # Separate callback groups so the command callback and the publish
        # timer can run at the same time on the MultiThreadedExecutor: a
        # publish in flight must not delay accepting the next command.
        self._cmd_group = MutuallyExclusiveCallbackGroup()
        self._pub_group = MutuallyExclusiveCallbackGroup()
        self._housekeep_group = MutuallyExclusiveCallbackGroup()

        # Subscriber
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_sub = self.create_subscription(
            JointState, 'commands', self._on_cmd, cmd_qos,
            callback_group=self._cmd_group,
        )

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

        # Start physics, then wait for the seed snapshot so the first publish
        # never races the model load / viewer open.
        self.worker.start()
        if not self.worker.wait_ready(timeout=10.0):
            err = self.worker.error
            raise RuntimeError(f'sim thread failed to start: {err!r}'
                               if err else 'sim thread failed to start')

        # Publish timer: reads the latest snapshot, never MjData.
        publish_period = (1.0 / publish_rate_hz if publish_rate_hz > 0.0
                          else self.worker.control_period)
        self._publish_timer = self.create_timer(
            publish_period, self._on_publish, callback_group=self._pub_group)

        # Housekeeping: watch the sim thread, log throughput. Never steps.
        self._housekeep_timer = self.create_timer(
            0.5, self._on_housekeep, callback_group=self._housekeep_group)
        self._last_stats_ticks = 0

        self.get_logger().info(
            f'goat_mujoco_node up: model={model_path}, '
            f'steps_per_cmd={steps_per_cmd}, '
            f'control_dt={self.worker.control_period:.4f}s, '
            f'publish_dt={publish_period:.4f}s, use_viewer={self._use_viewer}, '
            f'realtime={realtime}, cmd_timeout={cmd_timeout}s'
        )

    # ------------------------------------------------------------------ #
    # ROS thread: command in
    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: JointState) -> None:
        """Convert and hand the command to the sim thread. Never steps.

        ``cmd_to_ctrl`` reads MjModel only (immutable after load), so the
        conversion happens with no lock held; only the resulting vector goes
        into the command mailbox. Newest command wins -- see SimWorker.
        """
        if self._shutting_down:
            return
        try:
            ctrl = ros_bridge.cmd_to_ctrl(msg, self.sim)
        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'ignoring malformed command: {exc}')
            return
        self.worker.submit_ctrl(ctrl)

    # ------------------------------------------------------------------ #
    # ROS thread: state out
    # ------------------------------------------------------------------ #
    def _on_publish(self) -> None:
        """Publish the latest completed tick, at most once per tick.

        Skipping when ``seq`` is unchanged keeps a publish timer that runs
        faster than physics from re-sending the same state (and re-stamping
        it with the same sim time).
        """
        seq, snap = self.worker.latest_snapshot()
        if snap is None or seq == self._last_pub_seq:
            return
        self._last_pub_seq = seq

        stamp = ros_bridge.sim_time_to_msg(snap.time)
        self.clock_pub.publish(Clock(clock=stamp))
        self.joint_pub.publish(ros_bridge.joint_state_msg(snap, stamp))
        self.imu_pub.publish(ros_bridge.imu_msg(snap, stamp))
        self.odom_pub.publish(ros_bridge.odom_msg(snap, stamp))

    # ------------------------------------------------------------------ #
    # Housekeeping: sim thread health (no stepping, no MjData access)
    # ------------------------------------------------------------------ #
    def _on_housekeep(self) -> None:
        if self.worker.finished:
            err = self.worker.error
            if err is not None:
                self.get_logger().error(f'sim thread died: {err!r}')
            else:
                self.get_logger().info(f'sim thread stopped: {self.worker.exit_reason}')
            self._shutdown()
            return

        stats = self.worker.stats
        ticks = stats['ticks']
        if ticks == self._last_stats_ticks:
            self.get_logger().warn('sim thread produced no tick in the last 0.5s')
        self._last_stats_ticks = ticks
        self.get_logger().debug(
            f"ticks={ticks} cmd_received={stats['cmd_received']} "
            f"cmd_used={stats['cmd_used']} stale_ticks={stats['stale_ticks']}"
        )

    @property
    def shutdown_requested(self) -> bool:
        return self._shutting_down

    def _shutdown(self) -> None:
        """Request shutdown: stop the sim thread and flag the spin loop.

        This only flips state and signals the worker (whose thread closes the
        viewer it opened). The context teardown (destroy_node then
        rclpy.shutdown, in that order) is owned by main(), so we never tear the
        context down from inside a callback the context is still driving.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info('shutting down goat_mujoco_node')
        self.worker.stop('node shutdown')

    def destroy_node(self) -> bool:
        self.worker.stop('node destroyed')
        self.worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = GoatMujocoNode()
    except BaseException:
        rclpy.shutdown()
        raise

    # Multi-threaded so the command callback, the publish timer and
    # housekeeping can overlap. spin() (not spin_once) is what actually hands
    # callbacks to the pool, so it runs on its own thread while the main
    # thread waits for the shutdown flag -- teardown then happens here, never
    # inside a callback the executor is still driving.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name='ros_spin', daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok() and not node.shutdown_requested:
            time.sleep(0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        node.destroy_node()          # stops and joins the sim thread
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
