# goat_control/nodes/policy_node.py
from __future__ import annotations

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class AgentNode(Node):
    """
    Event-driven policy I/O:
      - Subscribes:
          * goat/observation (std_msgs/Float32MultiArray)
            -> Format: [pure_observations, timestep, agent_operating_time]
      - Publishes:
          * goat/action (std_msgs/Float32MultiArray)
    """

    def __init__(self):
        super().__init__("agent")

        # Parameters
        self.checkpoint = str(self.declare_parameter("checkpoint", "").value)
        torch_device_param = str(self.declare_parameter("torch_device", "cuda").value)
        self.torch_device = self._resolve_device(torch_device_param)
        self.action_shape = tuple(int(x) for x in self.declare_parameter("action_shape", [8]).value)
        self.declare_parameter("observation_topic", "goat/observations")
        self.declare_parameter("action_topic", "goat/actions")
        
        # Agent model load
        self.agent = self._load_agent(self.checkpoint, self.torch_device)
        self._last_inference_error_log_time_sec = 0.0

        # Topic name
        observation_topic = str(self.get_parameter("observation_topic").value)
        action_topic = str(self.get_parameter("action_topic").value)

        # Observation subscriber
        self.create_subscription(Float32MultiArray, observation_topic, self._on_observation, 10)
        
        # Action publisher
        self.action_publisher = self.create_publisher(Float32MultiArray, action_topic, 10)

        self.get_logger().info(f"!! Agent Node started on device '{self.torch_device}' !!")

    def _on_observation(self, msg: Float32MultiArray) -> None:
        """Observation subscriber"""
        obs_array = np.asarray(msg.data, dtype=np.float32).flatten()

        # 1. Extract time info
        agent_timestep = int(obs_array[-2])
        agent_operating_time = float(obs_array[-1])             # Currently not used (for time specific task later)

        # 2. Extract observation
        pure_observation = obs_array[:-2]

        # 3. Extract action
        action_array = self._policy(pure_observation, agent_timestep)

        # 4. Publish action
        action_msg = self._numpy_to_multiarray(action_array)
        self.action_publisher.publish(action_msg)

    def _resolve_device(self, device_name: str) -> torch.device:
        try:
            device = torch.device(device_name)
        except Exception:
            self.get_logger().warn(f"Invalid torch_device '{device_name}'. Falling back to CPU.")
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested for torch_device but unavailable. Falling back to CPU.")
            return torch.device("cpu")

        return device

    def _policy(self, observation: np.ndarray, agent_timestep: int) -> np.ndarray:
        """Policy method: Observation -> Action"""
        target_elements = int(np.prod(self.action_shape)) if self.action_shape else observation.size
        zero_action = np.zeros(target_elements, dtype=np.float32)

        # Execption when agent is not defined
        if self.agent is None:
            return zero_action.reshape(self.action_shape) if self.action_shape else zero_action

        
        try:
            # Load observation on cuda
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.torch_device).unsqueeze(0)
            
            # Action
            with torch.no_grad():
                action, _, _ , _ = self.agent.act(obs_tensor, timestep=agent_timestep, deterministic=True)          # Ignore timestep

            action_flat = action.detach().cpu().numpy().astype(np.float32).reshape(-1)

            if action_flat.size < target_elements:
                action_flat = np.pad(action_flat, (0, target_elements - action_flat.size), mode="constant")
            elif action_flat.size > target_elements:
                action_flat = action_flat[:target_elements]

        except Exception as exc:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_inference_error_log_time_sec > 1.0:
                self.get_logger().warn(f"Policy inference failed: {exc}")
                self._last_inference_error_log_time_sec = now_sec
            action_flat = zero_action

        return action_flat.reshape(self.action_shape) if self.action_shape else action_flat

    def _load_agent(self, checkpoint_path: str, device: torch.device):
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
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()