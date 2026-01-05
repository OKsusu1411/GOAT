# goat_control/nodes/control_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

# 너 프로젝트 메시지 타입(이미 사용 중이면 이걸 쓰는 게 제일 좋음)
from motor_interfaces.msg import BaseStates  # imu_publisher.py에서 쓰던 타입

from goat_control.core.comm import CanInterface, MotorDriver, MotorParams
from goat_control.core.control.control_pipeline import ControlTargets
from goat_control.core import launch_core_control_system


@dataclass
class LatestBuffers:
    """Thread-safe 하게 쓰려면 lock을 쓰는게 정석이지만,
    ROS2 single-thread executor면 콜백/타이머가 같은 스레드에서 돌아서 일단 단순 버퍼로도 안정적임.
    """
    imu_msg: Optional[BaseStates] = None
    action_msg: Optional[Float32MultiArray] = None


class GoatControlNode(Node):
    """Main control loop node (ROS2-only).
    - Reads IMU + policy action from ROS topics
    - Polls motor states and computes control command (core pipeline)
    - Sends torque/current command to motors
    - Publishes observation/debug topics
    """

    def __init__(self):
        super().__init__("goat_control_node")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("motor_node_ids", [1, 2, 3, 4, 5, 6, 7, 8])

        # control command unit
        # - "amp": pipeline output을 모터 전류[A]로 간주해서 torque_mode_amp로 전송
        # - "torque_nm": pipeline output을 joint torque[Nm]로 간주해서 A로 변환 후 전송
        self.declare_parameter("command_unit", "amp")

        # action watchdog
        # - action_topic이 일정 시간 이상 업데이트 안되면 토크 0으로 강제하는 안전장치
        self.declare_parameter("action_timeout_sec", 0.05)

        # debug print rate limit
        # - 너무 많이 찍히면 보기 힘드니까 일정 주기마다만 출력
        self.declare_parameter("debug_print_period_sec", 0.2)

        # topic names
        self.declare_parameter("imu_topic", "imu_data")
        self.declare_parameter("action_topic", "goat/action")
        self.declare_parameter("observation_topic", "goat/observation")
        self.declare_parameter("command_topic", "goat/command_safe")

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        yaml_path = str(self.get_parameter("yaml_path").value)
        motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        command_unit = str(self.get_parameter("command_unit").value)

        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self.debug_print_period_sec = float(self.get_parameter("debug_print_period_sec").value)

        imu_topic = str(self.get_parameter("imu_topic").value)
        action_topic = str(self.get_parameter("action_topic").value)
        observation_topic = str(self.get_parameter("observation_topic").value)
        command_topic = str(self.get_parameter("command_topic").value)

        self.get_logger().info(f"CAN: {can_interface}:{can_channel}")
        self.get_logger().info(f"Control rate: {control_rate_hz} Hz")
        self.get_logger().info(f"Config YAML: {yaml_path}")
        self.get_logger().info(f"Motor node IDs: {motor_node_ids}")
        self.get_logger().info(f"Command unit: {command_unit}")
        self.get_logger().info(f"Action timeout: {self.action_timeout_sec:.3f} sec")

        self.buffers = LatestBuffers()
        self.command_unit = command_unit

        # action watchdog timestamps
        self.last_action_time = None  # rclpy.time.Time | None
        self._last_timeout_warn_time_sec = 0.0
        self._last_debug_print_time_sec = 0.0

        # -------------------------
        # ROS2 Pub/Sub
        # -------------------------
        self.imu_subscriber = self.create_subscription(
            BaseStates, imu_topic, self._on_imu_msg, 20
        )
        self.action_subscriber = self.create_subscription(
            Float32MultiArray, action_topic, self._on_action_msg, 10
        )

        self.observation_publisher = self.create_publisher(
            Float32MultiArray, observation_topic, 10
        )
        self.command_publisher = self.create_publisher(
            Float32MultiArray, command_topic, 10
        )
        self.joint_state_publisher = self.create_publisher(
            JointState, "joint_states", 10
        )

        # -------------------------
        # Build core system (CAN + MotorDrivers + Pipeline)
        # -------------------------
        self.can_interface = CanInterface(channel=can_channel, interface=can_interface)
        self.can_interface.open()

        # motor drivers (node id ordering must match config vectors ordering)
        self.motor_drivers: List[MotorDriver] = []
        for node_id in motor_node_ids:
            params = MotorParams(node_id=int(node_id))
            self.motor_drivers.append(MotorDriver(self.can_interface, params))

        # build model + control pipeline from YAML
        self.goat_model, self.control_pipeline = launch_core_control_system(
            yaml_path=yaml_path,
            motor_drivers=self.motor_drivers,
            effort_output_mode="torque_nm",  # pipeline 내부 effort-like 해석 모드(필요시 변경)
        )
        self.control_pipeline.reset()

        self.num_joints = int(self.goat_model.num_joints)

        # action이 안 들어올 때 기본 목표값
        self.default_desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
        self.default_desired_wheel_speed_rad_per_sec = np.zeros(self.num_joints, dtype=float)

        # last timestamp for dt
        self.last_control_time = self.get_clock().now()

        control_period_sec = 1.0 / max(control_rate_hz, 1.0)
        self.control_timer = self.create_timer(control_period_sec, self._control_loop)

        self.get_logger().info("GoatControlNode started.")

    # -------------------------
    # Callbacks
    # -------------------------
    def _on_imu_msg(self, msg: BaseStates) -> None:
        # IMU는 action watchdog과 무관 (IMU 들어온다고 안전하다고 보면 안 됨)
        self.buffers.imu_msg = msg

    def _on_action_msg(self, msg: Float32MultiArray) -> None:
        # policy action 수신 시각을 기록해 watchdog에 사용
        self.buffers.action_msg = msg
        self.last_action_time = self.get_clock().now()

    # -------------------------
    # Watchdog helper
    # -------------------------
    def _is_action_timed_out(self, now_time) -> bool:
        """Return True if policy action is missing or older than action_timeout_sec."""
        if self.last_action_time is None:
            return True
        age_sec = (now_time - self.last_action_time).nanoseconds * 1e-9
        return age_sec > self.action_timeout_sec

    # -------------------------
    # Main loop
    # -------------------------
    def _control_loop(self) -> None:
        now_time = self.get_clock().now()

        dt_sec = (now_time - self.last_control_time).nanoseconds * 1e-9
        if dt_sec <= 0.0:
            dt_sec = 1e-3
        self.last_control_time = now_time

        # 1) action -> targets (watchdog 포함)
        action_msg = self.buffers.action_msg
        action_timed_out = self._is_action_timed_out(now_time)

        if (action_msg is None) or action_timed_out:
            # action이 없거나 오래되면 안전하게 0 타겟
            desired_joint_position_rad = self.default_desired_joint_position_rad.copy()
            desired_wheel_speed_rad_per_sec = self.default_desired_wheel_speed_rad_per_sec.copy()
        else:
            desired_joint_position_rad, desired_wheel_speed_rad_per_sec = self._decode_action_to_targets(action_msg)

        targets = ControlTargets(
            desired_joint_position_rad=desired_joint_position_rad,
            desired_wheel_speed_rad_per_sec=desired_wheel_speed_rad_per_sec,
        )

        # 2) imu conversion (optional)
        imu_state = None
        if self.buffers.imu_msg is not None:
            imu_state = self._convert_base_states_to_core_imu(self.buffers.imu_msg)

        # 3) pipeline step: poll motors + compute torque/current command
        pipeline_output = self.control_pipeline.step(
            targets=targets,
            dt_sec=dt_sec,
            imu_state=imu_state,
        )

        # pipeline output은 기본적으로 torque[Nm]라고 가정 (safe_torque_command)
        safe_command = np.asarray(pipeline_output.safe_torque_command, dtype=float).flatten()
        if safe_command.size != self.num_joints:
            self.get_logger().warn("safe_torque_command size mismatch. Force zero.")
            safe_command = np.zeros(self.num_joints, dtype=float)

        # 4) Action watchdog 최종 강제 (가장 중요)
        # - 정책 명령이 일정 시간 이상 안 오면 토크 0으로 강제
        if action_timed_out:
            safe_command[:] = 0.0

            now_sec = now_time.nanoseconds * 1e-9
            if now_sec - self._last_timeout_warn_time_sec > 1.0:
                self.get_logger().warn(
                    f"policy action timeout (> {self.action_timeout_sec:.3f}s) -> FORCE ZERO TORQUE"
                )
                self._last_timeout_warn_time_sec = now_sec

        # 5) send to motors
        self.get_logger().info(f"Safe command to motors: {safe_command.tolist()}")
        self._send_command_to_motors(safe_command)

        # 6) publish observation + debug
        self._publish_observation(pipeline_output.robot_state)
        self._publish_joint_state(pipeline_output.robot_state)
        self._publish_command(safe_command)

        # 7) rate-limited debug print (너가 보던 decode 출력이 0인지 확인하기 좋게)
        now_sec = now_time.nanoseconds * 1e-9
        if now_sec - self._last_debug_print_time_sec >= self.debug_print_period_sec:
            self._last_debug_print_time_sec = now_sec

            action_size = 0 if action_msg is None else len(action_msg.data)
            self.get_logger().info(
                f"[DEBUG] action_size={action_size}, timeout={action_timed_out}, "
                f"q_des(rad)={np.round(desired_joint_position_rad, 4).tolist()}, "
                f"wheel_des(rad/s)={np.round(desired_wheel_speed_rad_per_sec, 4).tolist()}"
            )

    # -------------------------
    # Action -> Targets
    # -------------------------
    def _decode_action_to_targets(self, action_msg: Float32MultiArray) -> Tuple[np.ndarray, np.ndarray]:
        """Action message format (테스트/추천):
        - action.data length == 8  : desired joint positions [rad] (0~7)
        - action.data length == 16 : [0:8]=q_des[rad], [8:16]=dq_des[rad/s] (wheel 6~7 유효)

        나중에 policy 포맷이 정해지면 여기만 바꾸면 됨.
        """
        action_array = np.asarray(action_msg.data, dtype=float).flatten()

        # 기본은 0
        desired_joint_position_rad = self.default_desired_joint_position_rad.copy()
        desired_wheel_speed_rad_per_sec = self.default_desired_wheel_speed_rad_per_sec.copy()

        if action_array.size >= self.num_joints:
            desired_joint_position_rad[:] = action_array[: self.num_joints]

        if action_array.size >= (2 * self.num_joints):
            desired_wheel_speed_rad_per_sec[:] = action_array[self.num_joints : 2 * self.num_joints]

        return desired_joint_position_rad, desired_wheel_speed_rad_per_sec

    # -------------------------
    # BaseStates -> core IMU state
    # -------------------------
    def _convert_base_states_to_core_imu(self, msg: BaseStates):
        """Convert BaseStates(imu_publisher) -> core ImuState.
        core쪽 ImuState 타입/필드에 맞게 변환해줌.
        """
        imu_state = self.control_pipeline.make_imu_state(
            roll=float(msg.rpy.x),
            pitch=float(msg.rpy.y),
            yaw=float(msg.rpy.z),
            gyro_x=float(msg.gyro.x),
            gyro_y=float(msg.gyro.y),
            gyro_z=float(msg.gyro.z),
            acc_x=float(msg.acc.x),
            acc_y=float(msg.acc.y),
            acc_z=float(msg.acc.z),
            time_ms=float(msg.time_ms),
        )
        return imu_state

    # -------------------------
    # Motor command sending
    # -------------------------
    def _send_command_to_motors(self, safe_command: np.ndarray) -> None:
        """Send safe command to each motor.
        - command_unit == "amp": safe_command is current[A]
        - command_unit == "torque_nm": convert torque -> current using GoatModel config
        """
        if safe_command.size != self.num_joints:
            self.get_logger().warn("safe_command size mismatch.")
            return

        if self.command_unit == "torque_nm":
            current_command_amp = self._convert_torque_to_current_amp(safe_command)
        else:
            current_command_amp = safe_command

        # send to all motors (node_id ordering must match motor_drivers ordering)
        for motor_index, motor_driver in enumerate(self.motor_drivers):
            command_amp = float(current_command_amp[motor_index])
            motor_driver.torque_mode_amp(command_amp, timeout=0.02)

    def _convert_torque_to_current_amp(self, joint_torque_command_nm: np.ndarray) -> np.ndarray:
        """Convert joint torque -> motor current.
        current = torque / (direction * gear_ratio * torque_constant)

        (Kt=0인 경우 division error 방지로 0 처리)
        """
        torque_constant = np.asarray(self.goat_model.config.motor_torque_constant_nm_per_amp, dtype=float)
        gear_ratio = np.asarray(self.goat_model.config.motor_gear_ratio, dtype=float)
        direction = np.asarray(self.goat_model.config.motor_direction, dtype=float)

        denominator = direction * gear_ratio * torque_constant
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        current_command_amp = joint_torque_command_nm / denominator

        # Kt가 0이었던 곳은 0으로 강제
        zero_mask = np.abs(direction * gear_ratio * torque_constant) < 1e-12
        current_command_amp = np.where(zero_mask, 0.0, current_command_amp)
        return current_command_amp

    # -------------------------
    # Publishers
    # -------------------------
    def _publish_observation(self, robot_state) -> None:
        """Publish observation as Float32MultiArray.
        지금은 예시로 [q, dq, effort_like]를 이어붙임.
        """
        q = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        dq = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        effort_like = np.asarray(robot_state.joint_effort_like, dtype=float).flatten()

        obs = np.concatenate([q, dq, effort_like], axis=0)

        msg = Float32MultiArray()
        msg.data = obs.astype(np.float32).tolist()
        self.observation_publisher.publish(msg)

    def _publish_joint_state(self, robot_state) -> None:
        """Publish JointState for policy input/debug."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.goat_model.config.joint_names)
        msg.position = np.asarray(robot_state.joint_position_rad, dtype=float).flatten().tolist()
        msg.velocity = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten().tolist()
        msg.effort = np.asarray(robot_state.joint_effort_like, dtype=float).flatten().tolist()
        self.joint_state_publisher.publish(msg)

    def _publish_command(self, safe_command: np.ndarray) -> None:
        """Publish final safe command (after limiter + watchdog)."""
        msg = Float32MultiArray()
        msg.data = np.asarray(safe_command, dtype=float).flatten().astype(np.float32).tolist()
        self.command_publisher.publish(msg)


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
