# src/goat_control/goat_control/nodes/motor_command_node.py
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

from goat_control.core.comm import CanInterface, MotorDriver, MotorParams
from goat_control.core.model import build_goat_model_from_yaml

class MotorCommandNode(Node):
    """
    DEPRECATED.

    This legacy node subscribes to torque commands, converts them to current commands,
    and sends them to the motors via the CAN bus.

    In the refactored real-hardware flow, prefer **MotorIONode** which owns CAN
    (read+write) in a single process.
    """

    def __init__(self):
        super().__init__("motor_command_node")

        # Safety: keep CAN disabled by default to avoid accidentally opening
        # a second CAN socket when MotorIONode is running.
        self.declare_parameter("enable_can", False)
        if not bool(self.get_parameter("enable_can").value):
            self.get_logger().error(
                "MotorCommandNode is deprecated and CAN is disabled by default. "
                "Use 'motor_io_node' instead. If you really need this legacy node, "
                "run with enable_can:=true (not recommended with MotorIONode)."
            )
            self.can_interface = None
            return

        # Parameters
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("motor_node_ids", [1, 2, 3, 4, 5, 6, 7, 8])
        self.declare_parameter("torque_command_topic", "goat/torque_commands")
        self.declare_parameter("yaml_path", "goat_config.yaml")

        can_channel = str(self.get_parameter("can_channel").value)
        can_interface = str(self.get_parameter("can_interface").value)
        motor_node_ids = list(self.get_parameter("motor_node_ids").value)
        torque_command_topic = str(self.get_parameter("torque_command_topic").value)
        yaml_path = str(self.get_parameter("yaml_path").value)

        # CAN Interface
        self.can_interface = CanInterface(channel=can_channel, interface=can_interface)
        self.can_interface.open()

        # Motor Drivers
        self.motor_drivers: list[MotorDriver] = []
        for node_id in motor_node_ids:
            params = MotorParams(node_id=int(node_id))
            self.motor_drivers.append(MotorDriver(self.can_interface, params))
            
        self.num_joints = len(self.motor_drivers)

        # Goat Model for torque to current conversion
        self.goat_model = build_goat_model_from_yaml(yaml_path)

        # Subscriber
        self.command_subscriber = self.create_subscription(
            Float32MultiArray, torque_command_topic, self._on_command, 10
        )

        self.get_logger().info("MotorCommandNode started.")

    def _on_command(self, msg: Float32MultiArray):
        joint_torque_command_nm = np.asarray(msg.data, dtype=float).flatten()

        if joint_torque_command_nm.size != self.num_joints:
            self.get_logger().warn("Torque command size mismatch.")
            return

        current_command_amp = self.goat_model.convert_joint_torque_to_motor_current(joint_torque_command_nm)
            
        for motor_index, motor_driver in enumerate(self.motor_drivers):
            command_amp = float(current_command_amp[motor_index])
            motor_driver.torque_mode_amp(command_amp, timeout=0.02)

    def destroy_node(self):
        if self.can_interface is not None:
            try:
                self.can_interface.close()
            except Exception:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MotorCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
