import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import JointState
from motor_interfaces.msg import ImuState

class MujocoRos2Bridge(Node):
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

    def control_callback(self, msg):
        """ Write command to Mujoco actuator"""
        np.copyto(self.mujoco_data.ctrl, msg.effort)

    def simulation_step(self):
        if not self.viewer.is_running():
            rclpy.shutdown()
            return

        mujoco.mj_step(self.mujoco_model, self.mujoco_data)
        self.viewer.sync()

        # Publish physics step
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

        # Joint_states
        js_msg = JointState()
        js_msg.header.stamp = time_msg
        js_msg.name = ['hip_L_Joint', 'hip_R_Joint', 'thigh_L_Joint', 'thigh_R_Joint',
                       'knee_L_Joint', 'knee_R_Joint', 'wheel_L_Joint', 'wheel_R_Joint']
        js_msg.position = self.mujoco_data.qpos[7:].tolist()        # [:7] is base's position
        js_msg.velocity = self.mujoco_data.qvel[6:].tolist()        # []:6] is base's velocity
        js_msg.effort = self.mujoco_data.actuator_force.tolist()

        # Joint_state publish
        self.joint_state_pub.publish(js_msg)

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
