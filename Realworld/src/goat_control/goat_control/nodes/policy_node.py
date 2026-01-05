# goat_control/nodes/policy_node.py
from __future__ import annotations

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates


class PolicyNode(Node):
    """
    Legacy-compatible policy I/O:
      - Subscribes:
          * imu_data (motor_interfaces/BaseStates)
          * joint_states (sensor_msgs/JointState)   <-- policy input requirement
      - Publishes:
          * policy_action (std_msgs/Float32MultiArray) with layout populated
    """

    def __init__(self):
        super().__init__("policy")

        self.policy_checkpoint = str(self.declare_parameter("policy_checkpoint", "").value)
        policy_device_param = str(self.declare_parameter("policy_device", "cpu").value)
        self.policy_device = self._resolve_device(policy_device_param)
        self.action_shape = tuple(int(x) for x in self.declare_parameter("action_shape", [8]).value)
        self.policy_model = self._load_policy(self.policy_checkpoint, self.policy_device)
        self._last_inference_error_log_time_sec = 0.0

        action_frequency_param = float(self.declare_parameter("action_frequency", 50.0).value)

        # Compatibility: interpret <=1.0 as period(sec), >1.0 as frequency(Hz)
        if action_frequency_param <= 1.0:
            self.action_period_sec = max(1e-4, action_frequency_param)
            self.action_frequency_hz = 1.0 / self.action_period_sec
        else:
            self.action_frequency_hz = action_frequency_param
            self.action_period_sec = 1.0 / max(1e-6, self.action_frequency_hz)

        self.get_logger().info(
            f"Policy tick: {self.action_frequency_hz:.1f} Hz ({self.action_period_sec:.4f} s)"
        )

        self.latest_imu: BaseStates | None = None
        self.latest_joint_state: JointState | None = None

        self.create_subscription(BaseStates, "imu_data", self._on_imu, 10)
        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)

        self.action_publisher = self.create_publisher(Float32MultiArray, "policy_action", 10)
        self.timer = self.create_timer(self.action_period_sec, self._tick)

    def _on_imu(self, msg: BaseStates) -> None:
        self.latest_imu = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _tick(self) -> None:
        if self.latest_joint_state is None:
            return

        observation = self._build_observation(self.latest_joint_state, self.latest_imu)
        action_array = self._infer_action(observation)

        action_msg = self._numpy_to_multiarray(action_array)
        self.action_publisher.publish(action_msg)

    def _resolve_device(self, device_name: str) -> torch.device:
        try:
            device = torch.device(device_name)
        except Exception:
            self.get_logger().warn(f"Invalid policy_device '{device_name}'. Falling back to CPU.")
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested for policy_device but unavailable. Falling back to CPU.")
            return torch.device("cpu")

        return device

    def _build_observation(self, joint_state: JointState, imu_state: BaseStates | None) -> np.ndarray:
        positions = np.asarray(joint_state.position or [], dtype=np.float32)

        velocities = np.asarray(joint_state.velocity or [], dtype=np.float32)
        if velocities.size < positions.size:
            velocities = np.pad(velocities, (0, positions.size - velocities.size), mode="constant")
        elif velocities.size > positions.size:
            velocities = velocities[: positions.size]

        imu_vector = np.zeros(10, dtype=np.float32)
        if imu_state is not None:
            imu_vector = np.asarray(
                [
                    imu_state.quat.w,
                    imu_state.quat.x,
                    imu_state.quat.y,
                    imu_state.quat.z,
                    imu_state.gyro.x,
                    imu_state.gyro.y,
                    imu_state.gyro.z,
                    imu_state.acc.x,
                    imu_state.acc.y,
                    imu_state.acc.z,
                ],
                dtype=np.float32,
            )

        return np.concatenate([positions, velocities, imu_vector], axis=0, dtype=np.float32)

    def _infer_action(self, observation: np.ndarray) -> np.ndarray:
        target_elements = int(np.prod(self.action_shape)) if self.action_shape else observation.size
        zero_action = np.zeros(target_elements, dtype=np.float32)

        if self.policy_model is None:
            return zero_action.reshape(self.action_shape) if self.action_shape else zero_action

        try:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.policy_device).unsqueeze(0)
            with torch.no_grad():
                output = self.policy_model(obs_tensor)

            if isinstance(output, (tuple, list)):
                output = output[0]

            output_flat = output.detach().cpu().numpy().astype(np.float32).reshape(-1)

            if output_flat.size < target_elements:
                output_flat = np.pad(output_flat, (0, target_elements - output_flat.size), mode="constant")
            elif output_flat.size > target_elements:
                output_flat = output_flat[:target_elements]

        except Exception as exc:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_inference_error_log_time_sec > 1.0:
                self.get_logger().warn(f"Policy inference failed: {exc}")
                self._last_inference_error_log_time_sec = now_sec
            output_flat = zero_action

        return output_flat.reshape(self.action_shape) if self.action_shape else output_flat

    def _load_policy(self, checkpoint_path: str, device: torch.device):
        if not checkpoint_path:
            self.get_logger().info("No policy checkpoint provided; publishing zero actions.")
            return None

        try:
            model = torch.jit.load(checkpoint_path, map_location=device)
            model.eval()
            self.get_logger().info(f"Loaded policy checkpoint from '{checkpoint_path}' on {device}.")
            return model
        except Exception as exc:
            self.get_logger().error(f"Failed to load policy checkpoint '{checkpoint_path}': {exc}")
            return None

    @staticmethod
    def _numpy_to_multiarray(array_value: np.ndarray) -> Float32MultiArray:
        array_value = np.asarray(array_value, dtype=np.float32)

        msg = Float32MultiArray()
        msg.layout.data_offset = 0
        msg.layout.dim = []

        shape = array_value.shape
        current_stride = 1
        strides = []
        for size in reversed(shape):
            strides.insert(0, current_stride)
            current_stride *= int(size)

        for dim_index, dim_size in enumerate(shape):
            dim = MultiArrayDimension()
            dim.label = f"dim_{dim_index}"
            dim.size = int(dim_size)
            dim.stride = int(strides[dim_index])
            msg.layout.dim.append(dim)

        msg.data = array_value.flatten().tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
