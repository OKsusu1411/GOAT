# goat_control/nodes/control_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates  # IMU message

from goat_control.core.comm import CanInterface, MotorDriver, MotorParams
from goat_control.core.control.control_pipeline import ControlTargets
from goat_control.core.build_system import launch_core_control_system
from goat_control.core.estimation.state_types import ImuState


@dataclass
class LatestBuffers:
    imu_msg: Optional[BaseStates] = None
    action_msg: Optional[Float32MultiArray] = None


class GoatControlNode(Node):
    """
    - Subscribes:
        * imu_data (BaseStates)
        * policy_action (Float32MultiArray)
    - Publishes:
        * joint_states (sensor_msgs/JointState)          <-- policy input
        * motor_torque_log (Float32MultiArray)           <-- logging (motor state + torque) single topic
    """

    def __init__(self):
        super().__init__("goat_control_node")

        # Parameters
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("motor_node_ids", [1, 2, 3, 4, 5, 6, 7, 8])
        self.declare_parameter("command_unit", "torque_nm")  # "amp" or "torque_nm"
        
        # watchdog: if policy_action is stale -> force zero torque
        self.declare_parameter("action_timeout_sec", 0.05)  # 50 ms
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)

        self.last_action_time = None  # rclpy.time.Time | None
        self._last_timeout_warn_time_sec = 0.0  # rate-limit warning log

        # Topic names (legacy-friendly)
        self.declare_parameter("imu_topic", "imu_data")
        self.declare_parameter("action_topic", "policy_action")
        self.declare_parameter("log_topic", "motor_torque_log")

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        yaml_path = str(self.get_parameter("yaml_path").value)
        motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        self.command_unit = str(self.get_parameter("command_unit").value)

        imu_topic = str(self.get_parameter("imu_topic").value)
        action_topic = str(self.get_parameter("action_topic").value)
        log_topic = str(self.get_parameter("log_topic").value)

        self.buffers = LatestBuffers()

        # Subscribers
        self.imu_subscriber = self.create_subscription(
            BaseStates, imu_topic, self._on_imu_msg, 20
        )
        self.action_subscriber = self.create_subscription(
            Float32MultiArray, action_topic, self._on_action_msg, 10
        )

        # Publishers
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.log_publisher = self.create_publisher(Float32MultiArray, log_topic, 10)

        # Build core system (CAN + MotorDrivers + Pipeline)
        self.can_interface = CanInterface(
            channel=can_channel,
            interface=can_interface,
            bitrate=None,
            receive_own_messages=False,
        )
        self.can_interface.open()

        self.motor_drivers: List[MotorDriver] = []
        for node_id in motor_node_ids:
            motor_params = MotorParams(node_id=int(node_id))
            self.motor_drivers.append(MotorDriver(self.can_interface, motor_params))

        self.goat_model, self.control_pipeline = launch_core_control_system(
            yaml_path=yaml_path,
            motor_drivers=self.motor_drivers,
            effort_output_mode="torque_nm",
        )
        self.control_pipeline.reset()

        self.num_joints = int(self.goat_model.num_joints)

        # Default targets (when policy_action is missing)
        self.default_desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
        self.default_desired_wheel_speed_rad_per_sec = np.zeros(self.num_joints, dtype=float)

        self.last_control_time = self.get_clock().now()
        self.control_timer = self.create_timer(1.0 / max(control_rate_hz, 1.0), self._control_loop)

        self.get_logger().info("GoatControlNode started (JointState->Policy, single log topic).")

    # Callbacks
    def _is_action_timed_out(self, now_time) -> bool:
        if self.last_action_time is None:
            return True
        age_sec = (now_time - self.last_action_time).nanoseconds * 1e-9
        return age_sec > self.action_timeout_sec

    def _on_imu_msg(self, msg: BaseStates) -> None:
        self.buffers.imu_msg = msg

    def _on_action_msg(self, msg: Float32MultiArray) -> None:
        self.buffers.action_msg = msg
        self.last_action_time = self.get_clock().now()

    # Main loop
    def _control_loop(self) -> None:
        now_time = self.get_clock().now()
        dt_sec = (now_time - self.last_control_time).nanoseconds * 1e-9
        if dt_sec <= 0.0:
            dt_sec = 1e-3
        self.last_control_time = now_time

        # 1) policy_action -> targets
        action_msg = self.buffers.action_msg
        if action_msg is None:
            desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
            desired_wheel_speed_rad_per_sec = np.zeros(self.num_joints, dtype=float)
        else:
            desired_joint_position_rad, desired_wheel_speed_rad_per_sec = self._decode_action_to_targets(action_msg)
            #NOTE: debug logs
        self.get_logger().info(f"Decoded desired_joint_position_rad: {desired_joint_position_rad}")
        self.get_logger().info(f"Decoded desired_wheel_speed_rad_per_sec: {desired_wheel_speed_rad_per_sec}")
        targets = ControlTargets(
            desired_joint_position_rad=desired_joint_position_rad,
            desired_wheel_speed_rad_per_sec=desired_wheel_speed_rad_per_sec,
        )

        # 2) IMU -> core ImuState
        imu_state = None
        if self.buffers.imu_msg is not None:
            imu_state = self._convert_base_states_to_core_imu(self.buffers.imu_msg)

        # 3) step core pipeline
        pipeline_output = self.control_pipeline.step(
            targets=targets,
            dt_sec=dt_sec,
            imu_state=imu_state,
        )

        safe_command = np.asarray(pipeline_output.safe_torque_command, dtype=float).flatten()
        #NOTE: debug logs
        self.get_logger().info(f"Raw safe_command: {safe_command}")
        # WATCHDOG: if policy_action is stale -> force zero command
        if self._is_action_timed_out(now_time):
            safe_command = np.zeros(self.num_joints, dtype=float)

            # rate-limited warning (1 Hz)
            now_sec = now_time.nanoseconds * 1e-9
            if now_sec - self._last_timeout_warn_time_sec > 1.0:
                self.get_logger().warn(
                    f"policy_action timeout (> {self.action_timeout_sec:.3f}s). Forcing ZERO torque/current."
                )
                self._last_timeout_warn_time_sec = now_sec

        # 4) send to motors
        self._send_command_to_motors(safe_command)

        # 5) publish JointState for policy input
        self._publish_joint_state(pipeline_output.robot_state)

        # 6) publish single log topic: motor state + torque
        self._publish_motor_torque_log(pipeline_output.robot_state, safe_command)

    # Action -> Targets
    def _decode_action_to_targets(self, action_msg: Float32MultiArray):
        """Decode policy action message into desired joint positions and wheel speeds.

        Supported formats:
        - size == num_joints (8): desired_joint_position_rad only, wheel speeds = 0
        - size >= 2*num_joints (16): [q_des(8), wheel_speed(8)]
        (you can still set only wheel indices [6,7] meaningful)
        """
        action_array = np.asarray(action_msg.data, dtype=float).flatten()

        # Defaults
        desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
        desired_wheel_speed_rad_per_sec = np.zeros(self.num_joints, dtype=float)

        if action_array.size >= self.num_joints:
            desired_joint_position_rad = action_array[: self.num_joints].copy()

        if action_array.size >= 2 * self.num_joints:
            desired_wheel_speed_rad_per_sec = action_array[self.num_joints : 2 * self.num_joints].copy()

        return desired_joint_position_rad, desired_wheel_speed_rad_per_sec


    # IMU conversion
    def _convert_base_states_to_core_imu(self, msg: BaseStates):
        return ImuState(
            quat_w=float(msg.quat.w),
            quat_x=float(msg.quat.x),
            quat_y=float(msg.quat.y),
            quat_z=float(msg.quat.z),
            gyro_x=float(msg.gyro.x),
            gyro_y=float(msg.gyro.y),
            gyro_z=float(msg.gyro.z),
            acc_x=float(msg.acc.x),
            acc_y=float(msg.acc.y),
            acc_z=float(msg.acc.z),
            time_ms=float(msg.time_ms),
        )

    # Send command
    def _send_command_to_motors(self, safe_command: np.ndarray) -> None:
        if safe_command.size != self.num_joints:
            self.get_logger().warn("safe_command size mismatch.")
            return

        if self.command_unit == "torque_nm":
            current_command_amp = self._convert_torque_to_current_amp(safe_command)
        else:
            current_command_amp = safe_command

        for motor_index, motor_driver in enumerate(self.motor_drivers):
            motor_driver.torque_mode_amp(float(current_command_amp[motor_index]), timeout=0.02)

    def _convert_torque_to_current_amp(self, joint_torque_command_nm: np.ndarray) -> np.ndarray:
        torque_constant = np.asarray(self.goat_model.config.motor_torque_constant_nm_per_amp, dtype=float)
        gear_ratio = np.asarray(self.goat_model.config.motor_gear_ratio, dtype=float)
        direction = np.asarray(self.goat_model.config.motor_direction, dtype=float)

        denominator = direction * gear_ratio * torque_constant
        current_command_amp = np.zeros_like(joint_torque_command_nm, dtype=float)

        valid_mask = np.abs(denominator) > 1e-9
        current_command_amp[valid_mask] = joint_torque_command_nm[valid_mask] / denominator[valid_mask]
        return current_command_amp

    # Publish: JointState
    def _publish_joint_state(self, robot_state) -> None:
        joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.goat_model.joint_names)
        msg.position = joint_position_rad.astype(float).tolist()
        msg.velocity = joint_velocity_rad_per_sec.astype(float).tolist()
        self.joint_state_publisher.publish(msg)

    # Publish: single log topic (motor state + torque)
    def _publish_motor_torque_log(self, robot_state, safe_command: np.ndarray) -> None:
        """
        motor_torque_log.data = [q(rad) ... , dq(rad/s) ... , u(cmd) ...]
          length = 3*num_joints
        This is for rosbag/rqt_plot logging with a single topic.
        """
        joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        torque_or_current_command = np.asarray(safe_command, dtype=float).flatten()

        log_vector = np.concatenate(
            [joint_position_rad, joint_velocity_rad_per_sec, torque_or_current_command],
            axis=0,
        )

        msg = Float32MultiArray()
        msg.data = log_vector.astype(float).tolist()
        self.log_publisher.publish(msg)

    # Shutdown
    def destroy_node(self):
        try:
            self.can_interface.close()
        except Exception:
            pass
        super().destroy_node()


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
