import time
import rclpy
import torch
import numpy as np
from rclpy.node import Node
from motor_interfaces.msg import BaseStates, MotorStates
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

class Policy(Node):
    def __init__(self):
        super().__init__('policy')
        self.imu_data_subscriber = self.create_subscription(
            BaseStates,
            'imu_data',
            self.imu_data_callback,
            10
        )

        self.action_publisher = self.create_publisher(
            Float32MultiArray,
            'policy_action',
            10
        )

        self.timer = self.create_timer(
            0.02,
            self.action_callback
        )

        self.latest_base_states = None

    def tensor_to_multiarray(self, tensor_action: torch.Tensor) -> Float32MultiArray:
        """
        Converts a PyTorch Tensor into a ROS2 Float3TMultiArray message.
        
        It automatically populates the 'layout' field (dimensions and strides)
        so the receiver can reconstruct the original shape.

        Args:
            tensor_action (torch.Tensor): The input tensor (e.g., from a policy).

        Returns:
            Float32MultiArray: ROS2 message.
        """
        
        # Handle the Tensor: Detach from graph, move to CPU, and convert to NumPy
        if isinstance(tensor_action, torch.Tensor):
            np_array = tensor_action.detach().cpu().numpy()
        else:
            # Handle if the input is already a numpy array
            np_array = np.array(tensor_action)

        # Ensure data type is float32 (as required by the message type)
        np_array = np_array.astype(np.float32)
        
        # Create the ROS2 message
        msg = Float32MultiArray()
        
        # Set up the layout
        msg.layout.data_offset = 0
        msg.layout.dim = []
        
        shape = np_array.shape
        current_stride = 1
        strides = []

        # Calculate strides (C-style / row-major order)
        # E.g., for shape (2, 3): strides will be (3, 1)
        for size in reversed(shape):
            strides.insert(0, current_stride)
            current_stride *= size

        # Populate the MultiArrayDimension fields
        for i, size in enumerate(shape):
            dim = MultiArrayDimension()
            dim.label = f"dim_{i}"
            dim.size = size
            dim.stride = strides[i]
            msg.layout.dim.append(dim)

        # Flatten the array and assign it to the data field
        msg.data = np_array.flatten().tolist()
        
        return msg
    
    def imu_data_callback(self, msg):
        self.latest_base_states = msg

    def action_callback(self):
        # TODO: Apply policy later

        target_position = torch.zeros(2, 3, device= "cuda:0")
        target_position_msg = self.tensor_to_multiarray(target_position)

        if self.latest_base_states is None:
            return
        
        self.action_publisher.publish(target_position_msg)

def main(args=None):
    rclpy.init(args=args)
    policy = Policy()
    rclpy.spin(policy)
    policy.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()