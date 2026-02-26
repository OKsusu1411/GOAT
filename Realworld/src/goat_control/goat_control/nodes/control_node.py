# goat_control/nodes/control_node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

import os
import csv
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
    - Subscribes to robot state (joint_states, goat/imu_data) and policy actions.
    - Computes control commands using the core control pipeline.
    - Publishes the final torque command.
    - Publishes observation and debug topics.
    """

    def __init__(self):
        super().__init__("goat_control_node")

        # Parameters
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("policy_decimation", 2.0)
        self.declare_parameter("yaml_path", "goat_config.yaml")
        self.declare_parameter("action_timeout_sec", 0.05)
        self.declare_parameter("debug_print_period_sec", 0.2)
        self.declare_parameter("action_space_len", 8)
        self.declare_parameter("observation_space_len", 28)
        self.declare_parameter("log_topic", "goat/torque_log")
        self.declare_parameter("policy_action", "goat/actions")
        self.declare_parameter("observation_topic", "goat/observations")
        self.declare_parameter("torque_command_topic", "goat/torque_commands")
        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter("csv_log_dir", "csv_logs")

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        policy_decimation = float(self.get_parameter("policy_decimation").value)
        yaml_path = str(self.get_parameter("yaml_path").value)
        self.action_timeout_sec = float(self.get_parameter("action_timeout_sec").value)
        self.debug_print_period_sec = float(self.get_parameter("debug_print_period_sec").value)
        log_topic = str(self.get_parameter("log_topic").value)
        action_topic = str(self.get_parameter("policy_action").value)
        observation_topic = str(self.get_parameter("observation_topic").value)
        torque_command_topic = str(self.get_parameter("torque_command_topic").value)
        self.enable_csv_log = bool(self.get_parameter("enable_csv_log").value)
        self.csv_log_dir = str(self.get_parameter("csv_log_dir").value)

        # Buffer for observation, action
        self.buffers = LatestBuffers()

        # Pub/Sub
        self.action_subscriber = self.create_subscription(
            Float32MultiArray, action_topic, self._on_action_msg, 10
        )
        self.joint_state_subscriber = self.create_subscription(
            JointState, "joint_states", self._on_joint_state_msg, 10
        )
        self.imu_subscriber = self.create_subscription(
            BaseStates, "/goat/imu_data", self._on_imu_msg, 10
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
        self.goat_model, self.control_pipeline = build_control_pipeline_from_yaml(
            yaml_path=yaml_path,
            motor_drivers=[],
            effort_output_mode="torque_nm", # Use current output for sim control
        )
        self.control_pipeline.reset()
        self.num_joints = int(self.goat_model.num_joints)

        # Default targets
        self.default_desired_joint_position_rad = np.zeros(self.num_joints, dtype=float)
        self.default_desired_joint_velocity_rad_per_sec = np.zeros(self.num_joints, dtype=float)

        # Timers
        self.last_control_time = self.get_clock().now()
        self._last_timeout_warn_time_sec = 0.0
        self._last_debug_print_time_sec = 0.0
        self.last_action_time = None
        self.agent_start_time: Optional[rclpy.time.Time] = None
        self.agent_timestep = 0.0
        
        # Controller loop timer
        control_period_sec = 1.0 / max(control_rate_hz, 1.0)                                # Low-level controller frequency
        self.control_timer = self.create_timer(control_period_sec, self._control_loop)
        
        # Decimation(Policy) counter
        self.policy_decimation = policy_decimation
        self.policy_decimation_counter = 0

        # Initialize CSV Logger
        self.csv_file = None
        self.csv_writer = None
        if self.enable_csv_log:
            self._init_csv_logger()

        self.get_logger().info("GoatControlNode started.")

    def _on_action_msg(self, msg: Float32MultiArray):
        """Action msg subscriber"""
        now_time = self.get_clock().now()

        # Record first action recived time
        if self.agent_start_time is None:
            self.agent_start_time = now_time
            self.agent_timestep = 0
            self.get_logger().info("Agent node's first action detected! Starting policy timer.")
        
        self.buffers.action_msg = msg
        self.last_action_time = now_time

    def _on_joint_state_msg(self, msg: JointState):
        self.buffers.joint_state_msg = msg

    def _on_imu_msg(self, msg: BaseStates):
        self.buffers.imu_msg = msg

    def _is_action_timed_out(self, now_time) -> bool:
        if self.last_action_time is None:
            return True
        age_sec = (now_time - self.last_action_time).nanoseconds * 1e-9
        
        if age_sec > self.action_timeout_sec:
            # Reset agent timer
            self.agent_start_time = None
            self.agent_timestep = -1
            return True
        
        return False

    def _control_loop(self):
        now_time = self.get_clock().now()
        dt_sec = (now_time - self.last_control_time).nanoseconds * 1e-9
        if dt_sec <= 0.0:
            dt_sec = 1e-3
        self.last_control_time = now_time

        if self.buffers.joint_state_msg is None:
            self.get_logger().warn("No joint states received yet, skipping control loop.")
            return

        # 1. Construct RobotState from subscribed messages
        joint_state_msg = self.buffers.joint_state_msg
        imu_msg = self.buffers.imu_msg
        
        imu_state = None
        if imu_msg:
            imu_state = ImuState(
                orientation_quat_w=float(imu_msg.quat.w),
                orientation_quat_x=float(imu_msg.quat.x),
                orientation_quat_y=float(imu_msg.quat.y),
                orientation_quat_z=float(imu_msg.quat.z),
                gyroscope_x=float(imu_msg.gyro.x),
                gyroscope_y=float(imu_msg.gyro.y),
                gyroscope_z=float(imu_msg.gyro.z),
                acceleration_x=float(imu_msg.acc.x),
                acceleration_y=float(imu_msg.acc.y),
                acceleration_z=float(imu_msg.acc.z),
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

        # 2. Decode action to targets
        action_msg = self.buffers.action_msg
        action_timed_out = self._is_action_timed_out(now_time)


        ###### =========================== Safety stop =========================== ###### 
        if (robot_state.joint_velocity_rad_per_sec[0:6] >= 0.436).any():
            self.get_logger().info("!!!!Emergency Stopped!!!!")
            self.control_timer.cancel()
        ###### =========================== Safety stop =========================== ###### 


        # Action timeout exception
        if (action_msg is None) or action_timed_out:
            desired_joint_position_rad = self.default_desired_joint_position_rad.copy()
            desired_joint_delta_position_rad = self.default_desired_joint_position_rad.copy()
            desired_wheel_speed_rad_per_sec = self.default_desired_joint_velocity_rad_per_sec.copy()
        else:
            desired_joint_delta_position_rad, desired_wheel_speed_rad_per_sec = self._decode_action_to_targets(action_msg, robot_state)

        targets = ControlTargets(
            desired_joint_delta_position_rad=desired_joint_delta_position_rad,
            desired_wheel_speed_rad_per_sec=desired_wheel_speed_rad_per_sec,
        )
        
        # 3. Compute command torque
        safe_command, safe_joint_targets, _ = self.control_pipeline.compute_control(
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
        
        # 5. Publish torque command, log
        self._publish_torque_command(safe_command)
        self._publish_motor_torque_log(robot_state, safe_command, targets)

        # 6. Decimation counting
        self.policy_decimation_counter += 1
        
        # 7. Publish observation (Execute agent node)
        if self.policy_decimation_counter >= self.policy_decimation:
            self._publish_observation(robot_state)
            self.policy_decimation_counter = 0                              # Reset counter
    
    # Refining action vector
    def _decode_action_to_targets(self, action_msg: Float32MultiArray, robot_state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        # Delta pos action
        action_array = np.asarray(action_msg.data, dtype=float).flatten()
        # current_joint_pos = robot_state.joint_position_rad

        # Initial(zero) pos, vel
        desired_joint_delta_position_rad = self.default_desired_joint_position_rad.copy()
        desired_wheel_speed_rad_per_sec = self.default_desired_joint_velocity_rad_per_sec.copy()

        # Indices for slicing
        joint_indices = [int(i) for i in self.goat_model.joint_indices]
        wheel_indices = [int(i) for i in self.goat_model.wheel_indices]

        expected_len = len(joint_indices) + len(wheel_indices)

        # Only 8 bytes
        if action_array.size != expected_len:
            if not hasattr(self, "_warned_action_len"):
                self._warned_action_len = False
            if not self._warned_action_len:
                self.get_logger().warn(
                    f"Action length mismatch: expected {expected_len} (=len(joint_indices)+len(wheel_indices)), "
                    f"got {action_array.size}. Using default targets."
                )
                self._warned_action_len = True
            return desired_joint_delta_position_rad, desired_wheel_speed_rad_per_sec

        # 1) Delta position action
        if len(joint_indices) > 0:
            ji = np.asarray(joint_indices, dtype=int)
            desired_joint_delta_position_rad[ji] = action_array[:len(joint_indices)]

        # 2) Reference velocity action
        if len(wheel_indices) > 0:
            wi = np.asarray(wheel_indices, dtype=int)
            desired_wheel_speed_rad_per_sec[wi] = action_array[len(joint_indices) : expected_len]

        return desired_joint_delta_position_rad, desired_wheel_speed_rad_per_sec

    def _publish_torque_command(self, safe_command: np.ndarray):
        msg = Float32MultiArray()
        msg.data = safe_command.tolist()
        self.torque_command_publisher.publish(msg)

    ## =============================== Observation vector =============================== ## 
    def _publish_observation(self, robot_state):
        # TODO: Noise filtering

        acceleration = [robot_state.imu_state.acceleration_x,
                        robot_state.imu_state.acceleration_y,
                        robot_state.imu_state.acceleration_z]
        gyroscope = [robot_state.imu_state.gyroscope_x,
                     robot_state.imu_state.gyroscope_y,
                     robot_state.imu_state.gyroscope_z]
        orientation_quat = [robot_state.imu_state.orientation_quat_w,
                            robot_state.imu_state.orientation_quat_x,
                            robot_state.imu_state.orientation_quat_y,
                            robot_state.imu_state.orientation_quat_z]
        magnetic_field = [robot_state.imu_state.magnetic_field_x,               # Currently not used
                          robot_state.imu_state.magnetic_field_y,
                          robot_state.imu_state.magnetic_field_z]
        # sensor_time_ms = float(imu_msg.time_ms)

        base_acc = np.asarray(acceleration, dtype=float).flatten()
        base_angular_vel = np.asarray(gyroscope, dtype=float).flatten()
        base_quat = np.asarray(orientation_quat, dtype=float).flatten()
        joint_q = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        joint_dq = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        # effort_like = np.asarray(robot_state.joint_effort_like, dtype=float).flatten()

        # Initial agent time
        agent_operating_time = 0.0

        # Agent operation detected
        if self.agent_start_time is not None:
            now_time = self.get_clock().now()
            agent_operating_time = (now_time - self.agent_start_time).nanoseconds * 1e-9            # Currently not used (for time specific task later)
            self.agent_timestep += 1
        
        # Agent time information
        time_info = np.array([self.agent_timestep, agent_operating_time], dtype=float)
        
        # Build observation vector
        obs = np.concatenate([base_acc, base_angular_vel, base_quat, joint_q, joint_dq, time_info], dtype=float, axis=0)
        msg = Float32MultiArray()
        msg.data = obs.astype(np.float32).tolist()
        self.observation_publisher.publish(msg)

    def _publish_motor_torque_log(self, robot_state, command_vector: np.ndarray, safe_joint_targets: np.ndarray, targets: ControlTargets):
        joint_position_rad = np.asarray(robot_state.joint_position_rad, dtype=float).flatten()
        joint_velocity_rad_per_sec = np.asarray(robot_state.joint_velocity_rad_per_sec, dtype=float).flatten()
        command_vector = np.asarray(command_vector, dtype=float).flatten()
        safe_joint_targets = np.asarray(safe_joint_targets, dtype=float).flatten()

        # Ref vector convention:
        #   - joints: position reference [rad]
        #   - wheels: speed reference [rad/s]
        ref_vector = np.asarray(targets.desired_joint_delta_position_rad + robot_state.joint_position_rad, dtype=float).flatten().copy()
        wheel_ref = np.asarray(targets.desired_wheel_speed_rad_per_sec, dtype=float).flatten()

        # Safe target command
        safe_joint_pos_targets = np.asarray(safe_joint_targets[:self.num_joints], dtype=float).flatten()
        safe_joint_vel_targets = np.asarray(safe_joint_targets[self.num_joints:], dtype=float).flatten()
        
        for wi in getattr(self.goat_model, "wheel_indices", []):
            wi = int(wi)
            if 0 <= wi < ref_vector.size and 0 <= wi < wheel_ref.size:
                ref_vector[wi] = float(wheel_ref[wi])
                safe_joint_pos_targets[wi] = float(safe_joint_vel_targets[wi])              # Combine into one vector

        log_vector = np.concatenate([joint_position_rad, joint_velocity_rad_per_sec, command_vector, safe_joint_pos_targets, ref_vector], axis=0)
        msg = Float32MultiArray()
        msg.data = log_vector.astype(np.float32).tolist()
        self.motor_torque_log_publisher.publish(msg)

    def _init_csv_logger(self):
        """Create CSV file and write headers."""
        if not os.path.exists(self.csv_log_dir):
            os.makedirs(self.csv_log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"goat_log_{timestamp}.csv"
        filepath = os.path.join(self.csv_log_dir, filename)
        
        try:
            self.csv_file = open(filepath, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 1. Define Headers
            headers = ["timestamp_sec", "agent_step", "agent_time"]
            
            # Observation headers
            obs_headers = []
            obs_headers += ["acc_x", "acc_y", "acc_z"]
            obs_headers += ["gyro_x", "gyro_y", "gyro_z"]
            obs_headers += ["quat_w", "quat_x", "quat_y", "quat_z"]
            # obs_headers += ["mag_x", "mag_y", "mag_z"] # Uncomment if mag is used in obs
            
            for i in range(self.num_joints):
                obs_headers.append(f"joint_pos_{i}")
            for i in range(self.num_joints):
                obs_headers.append(f"joint_vel_{i}")
            
            # Action headers
            action_headers = []
            for i in range(self.action_dim):
                action_headers.append(f"action_{i}")
            
            # Torque Command headers (Optional but useful)
            cmd_headers = []
            for i in range(self.num_joints):
                cmd_headers.append(f"tau_cmd_{i}")

            full_header = headers + obs_headers + action_headers + cmd_headers
            self.csv_writer.writerow(full_header)
            
            self.get_logger().info(f"CSV Logging enabled: {filepath}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to open CSV log file: {e}")
            self.enable_csv_log = False
    
    def _log_data_to_csv(self, robot_state, obs_vector, action, command):
        """Writes a row to the CSV file."""
        try:
            # Extract basic info
            timestamp = robot_state.timestamp_sec
            agent_step = self.agent_timestep
            
            # Agent operating time
            agent_time = 0.0
            if self.agent_start_time:
                agent_time = (self.get_clock().now() - self.agent_start_time).nanoseconds * 1e-9

            # Prepare row data
            # Format: [timestamp, step, agent_time, ...obs(minus time info), ...action, ...command]
            
            # NOTE: obs_vector has [..., step, time] at the end. We remove them to avoid duplication if needed,
            # or keep them. Here I strictly follow header definition.
            # Header defined: 
            # [acc(3), gyro(3), quat(4), q(N), dq(N)] -> length = 10 + 2*N
            # obs_vector has extra 2 at end. Let's slice them off for CSV body to match explicit headers.
            
            obs_core = obs_vector[:-2] 
            
            row = [timestamp, agent_step, agent_time]
            row.extend(obs_core.tolist())
            row.extend(action.tolist())
            row.extend(command.tolist())
            
            self.csv_writer.writerow(row)
            
        except Exception as e:
            self.get_logger().warn(f"CSV write failed: {e}")

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