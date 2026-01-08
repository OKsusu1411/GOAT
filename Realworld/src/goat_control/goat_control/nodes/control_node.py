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
from goat_control.core.comm import CanInterface, MotorDriver, MotorParams
from goat_control.core.control.control_pipeline import ControlTargets
from goat_control.core.estimation.state_types import RobotState, MotorStatesData, ImuState
from goat_control.core import launch_core_control_system


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
    - Sends commands to the motors.
    - Publishes observation and debug topics.
    """

    def __init__(self):
        super().__init__("goat_control_node")

        # Parameters
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("motor_node_ids", [1, 2, 3, 4, 5, 6, 7, 8])
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("command_unit", "torque_nm")
        self.declare_parameter("action_timeout_sec", 0.05)
        self.declare_parameter("debug_print_period_sec", 0.2)
        self.declare_parameter("log_topic", "motor_torque_log")
        self.declare_parameter("policy_action", "goat/action")
        self.declare_parameter("observation_topic", "goat/observation")

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        yaml_path = str(self.get_parameter("yaml_path").value)
        self.command_unit = str(self.get_parameter("command_unit").value)
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self.debug_print_period_sec = float(self.get_parameter("debug_print_period_sec").value)
        log_topic = str(self.get_parameter("log_topic").value)
        action_topic = str(self.get_parameter("policy_action").value)
        observation_topic = str(self.get_parameter("observation_topic").value)

        self.buffers = LatestBuffers()

        # CAN Interface and Motor Drivers
        self.can_interface = CanInterface(channel=can_channel, interface=can_interface)
        self.can_interface.open()
        self.motor_drivers: list[MotorDriver] = []
        for node_id in motor_node_ids:
            params = MotorParams(node_id=int(node_id))
            self.motor_drivers.append(MotorDriver(self.can_interface, params))

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
        
        # Build core system (Model + Pipeline)
        self.goat_model, self.control_pipeline = launch_core_control_system(
            yaml_path=yaml_path,
            motor_drivers=self.motor_drivers,
            effort_output_mode="torque_nm",
        )
        self.control_pipeline.reset()
        self.num_joints = int(self.goat_model.num_joints)

        # Default targets
        self.default_desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
        self.default_desired_joint_position_rad[2] = -20.0
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

        motor_states_data = MotorStatesData(
            positions_rad=np.array(joint_state_msg.position),
            velocities_rad_per_sec=np.array(joint_state_msg.velocity),
            torques_nm=np.array(joint_state_msg.effort),
            motor_temperature_c=np.zeros(self.num_joints),
            motor_phase_current_amp=np.zeros(self.num_joints),
            motor_speed_deg_per_sec=np.zeros(self.num_joints),
            motor_encoder_count=np.zeros(self.num_joints),
            motor_single_turn_angle_raw_0p001deg=np.zeros(self.num_joints),
            motor_multi_turn_angle_raw_0p001deg=np.zeros(self.num_joints),
            motor_error_flags=np.zeros(self.num_joints),
            motor_operating_state=np.zeros(self.num_joints),
            timestamp_sec=now_time.nanoseconds * 1e-9,
        )
        
        imu_state = None
        if imu_msg:
            imu_state = ImuState(
                roll=float(imu_msg.rpy.x), pitch=float(imu_msg.rpy.y), yaw=float(imu_msg.rpy.z),
                gyro_x=float(imu_msg.gyro.x), gyro_y=float(imu_msg.gyro.y), gyro_z=float(imu_msg.gyro.z),
                acc_x=float(imu_msg.acc.x), acc_y=float(imu_msg.acc.y), acc_z=float(imu_msg.acc.z),
                time_ms=float(imu_msg.time_ms)
            )

        robot_state = self.control_pipeline.state_manager.build_robot_state(
            motor_states_data=motor_states_data,
            imu_state=imu_state
        )

        # 3. Compute control command
        safe_command, _ = self.control_pipeline.compute_control(
            robot_state=robot_state,
            targets=targets,
            dt_sec=dt_sec
        )

        # 4. Apply action watchdog
        if action_timed_out:
            safe_command[:] = 0.0
            now_sec = now_time.nanoseconds * 1e-9
            if now_sec - self._last_timeout_warn_time_sec > 1.0:
                self.get_logger().warn(
                    f"Policy action timeout (> {self.action_timeout_sec:.3f}s) -> FORCE ZERO TORQUE"
                )
                self._last_timeout_warn_time_sec = now_sec

        # 5. Send command to motors
        self._send_command_to_motors(safe_command)

        # 6. Publish observation and debug topics
        self._publish_observation(robot_state)
        self._publish_motor_torque_log(robot_state, safe_command)
        
    def _decode_action_to_targets(self, action_msg: Float32MultiArray) -> Tuple[np.ndarray, np.ndarray]:
        action_array = np.asarray(action_msg.data, dtype=float).flatten()
        desired_joint_position_rad = self.default_desired_joint_position_rad.copy()
        desired_wheel_speed_rad_per_sec = self.default_desired_wheel_speed_rad_per_sec.copy()

        if action_array.size >= self.num_joints:
            desired_joint_position_rad[:] = action_array[:self.num_joints]
        if action_array.size >= (2 * self.num_joints):
            desired_wheel_speed_rad_per_sec[:] = action_array[self.num_joints: 2 * self.num_joints]
        
        return desired_joint_position_rad, desired_wheel_speed_rad_per_sec

    def _send_command_to_motors(self, safe_command: np.ndarray):
        if self.command_unit == "torque_nm":
            current_command_amp = self._convert_torque_to_current_amp(safe_command)
        else:
            current_command_amp = safe_command

        for motor_index, motor_driver in enumerate(self.motor_drivers):
            command_amp = float(current_command_amp[motor_index])
            motor_driver.torque_mode_amp(command_amp, timeout=0.02)

    def _convert_torque_to_current_amp(self, joint_torque_command_nm: np.ndarray) -> np.ndarray:
        torque_constant = np.asarray(self.goat_model.config.motor_torque_constant_nm_per_amp, dtype=float)
        gear_ratio = np.asarray(self.goat_model.config.motor_gear_ratio, dtype=float)
        direction = np.asarray(self.goat_model.config.motor_direction, dtype=float)

        denominator = direction * gear_ratio * torque_constant
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        current_command_amp = joint_torque_command_nm / denominator

        zero_mask = np.abs(direction * gear_ratio * torque_constant) < 1e-12
        current_command_amp = np.where(zero_mask, 0.0, current_command_amp)
        return current_command_amp

    def _publish_observation(self, robot_state):
        q = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        dq = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        effort_like = np.asarray(robot_state.joint_effort_like, dtype=float).flatten()
        
        obs = np.concatenate([q, dq, effort_like], axis=0)
        msg = Float32MultiArray()
        msg.data = obs.astype(np.float32).tolist()
        self.observation_publisher.publish(msg)

    def _publish_motor_torque_log(self, robot_state, command_vector: np.ndarray):
        joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        command_vector = np.asarray(command_vector, dtype=float).flatten()

        log_vector = np.concatenate([joint_position_rad, joint_velocity_rad_per_sec, command_vector], axis=0)
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