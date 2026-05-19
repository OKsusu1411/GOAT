import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import os
import time
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import JointState
from imu_interface.msg import ImuState

class MujocoRos2Bridge(Node):
    # MuJoCo orders qpos[7:]/qvel[6:] by tree traversal (per-leg):
    #   hip_L, thigh_L, knee_L, wheel_L, hip_R, thigh_R, knee_R, wheel_R
    # The controller and yaml index joints per-joint-type:
    #   hip_L, hip_R, thigh_L, thigh_R, knee_L, knee_R, wheel_L, wheel_R
    # MJ_TO_CTRL[k] = MuJoCo index that holds the joint at controller index k.
    MJ_TO_CTRL = [0, 4, 1, 5, 2, 6, 3, 7]

    def __init__(self):
        super().__init__('mujoco_ros2_bridge')

        pkg_share_path = get_package_share_directory('mujoco_sim2real')
        default_scene_path = os.path.join(pkg_share_path, 'xml', 'WF_GOAT_on_jig.xml')

        # Parameters
        self.declare_parameter("simulation_rate_hz", 200.0)
        self.declare_parameter("scene_path", default_scene_path)

        self.simulation_rate_hz = float(self.get_parameter("simulation_rate_hz").value)
        self.scene_path = str(self.get_parameter("scene_path").value)

        self.mujoco_model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        # Publisher
        self.imu_pub = self.create_publisher(ImuState, '/imu', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Subscriber
        self.cmd_sub = self.create_subscription(JointState, '/commands', self.control_callback, 10)

        # Sensor ID parsing
        self.acc_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_acc')
        self.gyro_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
        self.mag_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_mag')

        self.viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)

        # Simulation loop
        simulation_period_sec = 1.0 / max(self.simulation_rate_hz, 1.0)
        self.timer = self.create_timer(simulation_period_sec, self.simulation_step)

        # Topic-traffic monitor: per-second rate report for each topic this node touches.
        # Counters are reset every time the monitor timer fires.
        self._imu_pub_count = 0          # /imu publishes since last report
        self._js_pub_count = 0           # /joint_states publishes since last report
        self._cmd_recv_count = 0         # /commands receives since last report

        # First-seen latches so we log the first message on each topic at INFO once.
        self._first_imu_logged = False
        self._first_js_logged = False
        self._first_cmd_logged = False

        # Wall-clock anchor for the rate report (so Hz math is honest even if sim rate changes).
        self._monitor_last_time = self.get_clock().now()

        # 1 Hz monitor timer — decoupled from simulation_rate_hz on purpose.
        self._monitor_timer = self.create_timer(1.0, self._log_topic_rates)

        # Real-time pacing anchors. simulation_step will run mj_step in a while loop
        # until data.time catches up to (wall-elapsed since this point). Cap below
        # keeps a transient pause from triggering a spiral-of-death catch-up.
        self._wallclock_anchor = time.monotonic()
        self._sim_time_anchor = float(self.mujoco_data.time)
        self._max_substeps_per_tick = 20

    def control_callback(self, msg):
        """ Write command to Mujoco actuator, and tally for the topic monitor."""
        np.copyto(self.mujoco_data.ctrl, msg.effort)

        # Topic monitor: count this receive and log the first one at INFO.
        self._cmd_recv_count += 1
        if not self._first_cmd_logged:
            self.get_logger().info("first /commands received")
            self._first_cmd_logged = True

    def simulation_step(self):
        if not self.viewer.is_running():
            rclpy.shutdown()
            return

        # Catch sim time up to wall clock. mj_step always advances by model.opt.timestep,
        # so we loop until data.time has reached the target or the cap is hit.
        target_sim_time = self._sim_time_anchor + (time.monotonic() - self._wallclock_anchor)
        substeps = 0
        while (
            self.mujoco_data.time < target_sim_time
            and substeps < self._max_substeps_per_tick
        ):
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            substeps += 1

        self.viewer.sync()

        # Publish physics step (rate is still driven by the ROS timer, not the substep loop).
        self.publish_sensor_data()

    def publish_sensor_data(self):
        time_msg = self.get_clock().now().to_msg()

        # IMU data
        imu_msg = ImuState()
        imu_msg.header.stamp = time_msg

        # Sensor data indexing
        # acc_adr = self.mujoco_model.sensor_adr[self.acc_id]
        gyro_adr = self.mujoco_model.sensor_adr[self.gyro_id]
        mag_adr = self.mujoco_model.sensor_adr[self.mag_id]

        # Quaternion - unitless
        imu_msg.quat.w = self.mujoco_data.qpos[3]
        imu_msg.quat.x = self.mujoco_data.qpos[4]
        imu_msg.quat.y = self.mujoco_data.qpos[5]
        imu_msg.quat.z = self.mujoco_data.qpos[6]

        # Gyroscope(Angular velocity) - rad/s
        imu_msg.gyro.x = self.mujoco_data.sensordata[gyro_adr]
        imu_msg.gyro.y = self.mujoco_data.sensordata[gyro_adr+1]
        imu_msg.gyro.z = self.mujoco_data.sensordata[gyro_adr+2]

        # Linear velocity - m/s
        imu_msg.vel.x = self.mujoco_data.qvel[0]
        imu_msg.vel.y = self.mujoco_data.qvel[1]
        imu_msg.vel.z = self.mujoco_data.qvel[2]

        # Magnetometer - ?
        imu_msg.mag.x = self.mujoco_data.sensordata[mag_adr]
        imu_msg.mag.y = self.mujoco_data.sensordata[mag_adr+1]
        imu_msg.mag.z = self.mujoco_data.sensordata[mag_adr+2]

        # IMU accel (currently not used)
        # imu_msg.linear_acceleration.x = self.mujoco_data.sensordata[acc_adr]
        # imu_msg.linear_acceleration.y = self.mujoco_data.sensordata[acc_adr+1]
        # imu_msg.linear_acceleration.z = self.mujoco_data.sensordata[acc_adr+2]

        # IMU data publish
        self.imu_pub.publish(imu_msg)

        # Topic monitor: count this publish and log the first one at INFO.
        self._imu_pub_count += 1
        if not self._first_imu_logged:
            self.get_logger().info("first /imu published")
            self._first_imu_logged = True

        # Joint_states
        js_msg = JointState()
        js_msg.header.stamp = time_msg
        js_msg.name = ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint',
                       'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        # Reorder MuJoCo's per-leg joint layout into the per-joint-type order the controller expects.
        mj_qpos_joints = np.asarray(self.mujoco_data.qpos[7:])      # [:7] is base's free-joint pose
        mj_qvel_joints = np.asarray(self.mujoco_data.qvel[6:])      # [:6] is base's free-joint vel
        js_msg.position = mj_qpos_joints[self.MJ_TO_CTRL].tolist()
        js_msg.velocity = mj_qvel_joints[self.MJ_TO_CTRL].tolist()
        js_msg.effort = self.mujoco_data.actuator_force.tolist()

        # Joint_state publish
        self.joint_state_pub.publish(js_msg)

        # Topic monitor: count this publish and log the first one at INFO.
        self._js_pub_count += 1
        if not self._first_js_logged:
            self.get_logger().info("first /joint_states published")
            self._first_js_logged = True

    def _log_topic_rates(self):
        """1 Hz reporter: print observed Hz for each topic, then reset counters."""
        now = self.get_clock().now()
        # Elapsed seconds since the previous report (wall-clock from rclpy clock).
        elapsed = (now - self._monitor_last_time).nanoseconds * 1e-9
        if elapsed <= 0.0:
            return  # Guard against a zero/negative interval on the very first tick.

        # Compute observed rates over the elapsed window.
        imu_hz = self._imu_pub_count / elapsed
        js_hz = self._js_pub_count / elapsed
        cmd_hz = self._cmd_recv_count / elapsed

        self.get_logger().info(
            f"rates (Hz) — /imu: {imu_hz:.1f}  "
            f"/joint_states: {js_hz:.1f}  "
            f"/commands: {cmd_hz:.1f}"
        )

        # Reset counters and re-anchor the wall-clock window.
        self._imu_pub_count = 0
        self._js_pub_count = 0
        self._cmd_recv_count = 0
        self._monitor_last_time = now

def main(args=None):
    rclpy.init(args=args)
    node = MujocoRos2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
