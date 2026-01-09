# goat_control/nodes/control_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

from motor_interfaces.msg import BaseStates
from goat_control.core.control.control_pipeline import ControlTargets
from goat_control.core.estimation.state_types import RobotState, ImuState
from goat_control.core.model import build_control_pipeline_from_yaml


@dataclass
class LatestBuffers:
    """Thread-safe buffers for incoming messages."""
    joint_state_msg: Optional[JointState] = None
    action_msg: Optional[Float32MultiArray] = None
    imu_msg: Optional[BaseStates] = None


class GoatControlNode(Node):
    """
    Main control loop node.
    - Subscribes to robot state (joint_states, imu_data) and policy actions.
    - Computes control commands using the core control pipeline.
    - Publishes the final torque command.
    - Publishes observation and debug topics.
    """

    def __init__(self):
        super().__init__("goat_control_node")

        # Parameters
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("action_timeout_sec", 0.05)
        self.declare_parameter("debug_print_period_sec", 0.2)
        self.declare_parameter("log_topic", "motor_torque_log")
        self.declare_parameter("policy_action", "goat/action")
        self.declare_parameter("observation_topic", "goat/observation")
        self.declare_parameter("torque_command_topic", "torque_commands")

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        yaml_path = str(self.get_parameter("yaml_path").value)
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self.debug_print_period_sec = float(self.get_parameter("debug_print_period_sec").value)
        log_topic = str(self.get_parameter("log_topic").value)
        action_topic = str(self.get_parameter("policy_action").value)
        observation_topic = str(self.get_parameter("observation_topic").value)
        torque_command_topic = str(self.get_parameter("torque_command_topic").value)

        self.buffers = LatestBuffers()

        # Pub/Sub
        self.action_subscriber = self.create_subscription(
            Float32MultiArray, action_topic, self._on_action_msg, 10
        )
        self.joint_state_subscriber = self.create_subscription(
            JointState, "joint_states", self._on_joint_state_msg, 10
        )
        self.imu_subscriber = self.create_subscription(
            BaseStates, "imu_data", self._on_imu_msg, 10
        )
        self.observation_publisher = self.create_publisher(
            Float32MultiArray, observation_topic, 10
        )
        self.motor_torque_log_publisher = self.create_publisher(
            Float32MultiArray, log_topic, 10
        )
        self.torque_command_publisher = self.create_publisher(
            Float32MultiArray, torque_command_topic, 10
        )
        
        # Build core system (Model + Pipeline)
        # NOTE: ControlNode does not talk to hardware. We still build the pipeline
        # to reuse the exact same StateManager + controllers as real hardware.
        self.goat_model, self.control_pipeline = build_control_pipeline_from_yaml(
            yaml_path=yaml_path,
            motor_drivers=[],
            effort_output_mode="torque_nm",
        )
        self.control_pipeline.reset()
        self.num_joints = int(self.goat_model.num_joints)

        # Default targets
        self.default_desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
        # self.default_desired_joint_position_rad[2] = np.deg2rad(-20.0)
        self.default_desired_wheel_speed_rad_per_sec = np.zeros(self.num_joints, dtype=float)

        # Timers
        self.last_control_time = self.get_clock().now()
        self._last_timeout_warn_time_sec = 0.0
        self._last_debug_print_time_sec = 0.0
        self.last_action_time = None
        
        control_period_sec = 1.0 / max(control_rate_hz, 1.0)
        self.control_timer = self.create_timer(control_period_sec, self._control_loop)

        self.get_logger().info("GoatControlNode started.")

    def _on_action_msg(self, msg: Float32MultiArray):
        self.buffers.action_msg = msg
        self.last_action_time = self.get_clock().now()

    def _on_joint_state_msg(self, msg: JointState):
        self.buffers.joint_state_msg = msg

    def _on_imu_msg(self, msg: BaseStates):
        self.buffers.imu_msg = msg

    def _is_action_timed_out(self, now_time) -> bool:
        if self.last_action_time is None:
            return True
        age_sec = (now_time - self.last_action_time).nanoseconds * 1e-9
        return age_sec > self.action_timeout_sec

    def _control_loop(self):
        now_time = self.get_clock().now()
        dt_sec = (now_time - self.last_control_time).nanoseconds * 1e-9
        if dt_sec <= 0.0:
            dt_sec = 1e-3
        self.last_control_time = now_time

        if self.buffers.joint_state_msg is None:
            self.get_logger().warn("No joint states received yet, skipping control loop.")
            return

        # 1. Decode action to targets
        action_msg = self.buffers.action_msg
        action_timed_out = self._is_action_timed_out(now_time)

        if (action_msg is None) or action_timed_out:
            desired_joint_position_rad = self.default_desired_joint_position_rad.copy()
            desired_wheel_speed_rad_per_sec = self.default_desired_wheel_speed_rad_per_sec.copy()
        else:
            desired_joint_position_rad, desired_wheel_speed_rad_per_sec = self._decode_action_to_targets(action_msg)

        targets = ControlTargets(
            desired_joint_position_rad=desired_joint_position_rad,
            desired_wheel_speed_rad_per_sec=desired_wheel_speed_rad_per_sec,
        )

        # 2. Construct RobotState from subscribed messages
        joint_state_msg = self.buffers.joint_state_msg
        imu_msg = self.buffers.imu_msg
        
        imu_state = None
        if imu_msg:
            imu_state = ImuState(
                orientation_quat_w=float(imu_msg.quat.w),
                orientation_quat_x=float(imu_msg.quat.x),
                orientation_quat_y=float(imu_msg.quat.y),
                orientation_quat_z=float(imu_msg.quat.z),
                angular_velocity_x=float(imu_msg.gyro.x),
                angular_velocity_y=float(imu_msg.gyro.y),
                angular_velocity_z=float(imu_msg.gyro.z),
                linear_acceleration_x=float(imu_msg.acc.x),
                linear_acceleration_y=float(imu_msg.acc.y),
                linear_acceleration_z=float(imu_msg.acc.z),
                magnetic_field_x=float(imu_msg.mag.x),
                magnetic_field_y=float(imu_msg.mag.y),
                magnetic_field_z=float(imu_msg.mag.z),
                sensor_time_ms=float(imu_msg.time_ms)
            )

        robot_state = RobotState(
            joint_names=joint_state_msg.name,
            joint_position_rad=np.array(joint_state_msg.position),
            joint_velocity_rad_per_sec=np.array(joint_state_msg.velocity),
            joint_effort_like=np.array(joint_state_msg.effort),
            motor_temperature_c=[],
            motor_error_flags=[],
            motor_operating_state=[],
            imu_state=imu_state,
            timestamp_sec=now_time.nanoseconds * 1e-9
        )
        
        # 3. Compute control command
        safe_command, _ = self.control_pipeline.compute_control(
            robot_state=robot_state,
            targets=targets,
            dt_sec=dt_sec
        )
        self.get_logger().debug(f"Computed safe command: {safe_command}")
        # 4. Apply action watchdog
        if action_timed_out:
            safe_command[:] = 0.0
            now_sec = now_time.nanoseconds * 1e-9
            if now_sec - self._last_timeout_warn_time_sec > 1.0:
                self.get_logger().warn(
                    f"Policy action timeout (> {self.action_timeout_sec:.3f}s) -> FORCE ZERO TORQUE"
                )
                self._last_timeout_warn_time_sec = now_sec

        # 5. Publish torque command
        self._publish_torque_command(safe_command)

        # 6. Publish observation and debug topics
        self._publish_observation(robot_state)
        self._publish_motor_torque_log(robot_state, safe_command, targets)
        
    def _decode_action_to_targets(self, action_msg: Float32MultiArray) -> Tuple[np.ndarray, np.ndarray]:
        action_array = np.asarray(action_msg.data, dtype=float).flatten()
        desired_joint_position_rad = self.default_desired_joint_position_rad.copy()
        desired_wheel_speed_rad_per_sec = self.default_desired_wheel_speed_rad_per_sec.copy()

        if action_array.size >= self.num_joints:
            desired_joint_position_rad[:] = action_array[:self.num_joints]
        if action_array.size >= (2 * self.num_joints):
            desired_wheel_speed_rad_per_sec[:] = action_array[self.num_joints: 2 * self.num_joints]
        
        return desired_joint_position_rad, desired_wheel_speed_rad_per_sec

    def _publish_torque_command(self, safe_command: np.ndarray):
        msg = Float32MultiArray()
        msg.data = safe_command.tolist()
        self.torque_command_publisher.publish(msg)

    def _publish_observation(self, robot_state):
        q = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        dq = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        effort_like = np.asarray(robot_state.joint_effort_like, dtype=float).flatten()
        
        obs = np.concatenate([q, dq, effort_like], axis=0)
        msg = Float32MultiArray()
        msg.data = obs.astype(np.float32).tolist()
        self.observation_publisher.publish(msg)

    def _publish_motor_torque_log(self, robot_state, command_vector: np.ndarray, targets: ControlTargets):
        joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        command_vector = np.asarray(command_vector, dtype=float).flatten()

        # Ref vector convention:
        #   - joints: position reference [rad]
        #   - wheels: speed reference [rad/s]
        ref_vector = np.asarray(targets.desired_joint_position_rad, dtype=float).flatten().copy()
        wheel_ref = np.asarray(targets.desired_wheel_speed_rad_per_sec, dtype=float).flatten()
        for wi in getattr(self.goat_model, "wheel_indices", []):
            wi = int(wi)
            if 0 <= wi < ref_vector.size and 0 <= wi < wheel_ref.size:
                ref_vector[wi] = float(wheel_ref[wi])

        log_vector = np.concatenate([joint_position_rad, joint_velocity_rad_per_sec, command_vector, ref_vector], axis=0)
        msg = Float32MultiArray()
        msg.data = log_vector.astype(np.float32).tolist()
        self.motor_torque_log_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GoatControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()