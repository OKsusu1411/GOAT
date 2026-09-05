# goat_control/nodes/calibration_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from goat_control.utils.imu.quaternion_utils import *
from goat_control.utils.imu.calibration import simple_accelerometer_calibration
from motor_interfaces.msg import ImuState

import yaml
import os
import sys
import termios
import tty
import time
import threading
import numpy as np

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        
        # Parameters
        self.declare_parameter("yaml_path", "src/goat_control/config/goat_config.yaml")
        self.declare_parameter("sample_count", 20)

        self.yaml_path = str(self.get_parameter("yaml_path").value)
        self.sample_count = int(self.get_parameter("sample_count").value)

        # Subscriber
        self.joint_state_subscriber = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state_msg, 10
        )
        self.imu_subscriber = self.create_subscription(
            ImuState, "/imu", self._on_imu_msg, 10
        )

        # YAML file
        with open(self.yaml_path, 'r', encoding='utf-8') as file_handle:
            self.cfg = yaml.safe_load(file_handle)

        self.old_joint_offsets = self.cfg["joint_offsets"]
        self.old_imu_offsets = self.cfg["imu_offsets"]

        # Data buffers
        self.latest_joint_state = None
        self.latest_imu_state = None

        # Save terminal settings (to restore on exit)
        self.settings = termios.tcgetattr(sys.stdin)

        # Start keyboard listener thread
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

        # Print UI
        self.get_logger().info("Calibration Node Started.\r")
        self.get_logger().info("=============================================")
        self.get_logger().info("[CONTROLS]")
        self.get_logger().info("'j': All Joint Position Calibration")
        self.get_logger().info("'w': Wheel Position Calibration")
        self.get_logger().info("'i': IMU Calibration")
        self.get_logger().info("'q': Quit")
        self.get_logger().info("=============================================\r")

    def _on_joint_state_msg(self, msg: JointState):
        self.latest_joint_state = msg

    def _on_imu_msg(self, msg: ImuState):
        self.latest_imu_state = msg

    def _get_key(self):
        """Read a single character from the terminal immediately (Blocking)."""
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def _is_controller_running(self) -> bool:
        """Check whether controller_node exists in the current ROS graph."""
        node_names_and_namespaces = self.get_node_names_and_namespaces()

        for node_name, namespace in node_names_and_namespaces:
            if node_name == "controller_node":
                return True

        return False

    def _keyboard_listener_loop(self):
        """Main loop to monitor keyboard input."""
        while rclpy.ok():
            key = self._get_key()
            
            if key == 'j':
                self.get_logger().info("Key 'j' pressed: Starting Joint Calibration\r")
                self._joint_calibration()
            
            elif key == 'w':
                self.get_logger().info("Key 'w' pressed: Starting Wheel Calibration\r")
                self._joint_calibration()

            elif key == 'i':
                if self._is_controller_running():
                    self.get_logger().warning("[IMU Calibration] controller_node is running.")
                    self.get_logger().warning("Stop controller_node before IMU calibration.")
                    continue
                self.get_logger().info("Key 'i' pressed: Starting IMU Calibration\r")
                self._imu_calibration()
                
            elif key == 'q':
                self.get_logger().info("Key 'q' pressed: Shutting down node...\r")
                rclpy.shutdown()
                break
            
            # Handle Ctrl+C
            elif key == '\x03':
                rclpy.shutdown()
                break
            
            # Execption
            else:
                self.get_logger().info("Wrong key! Please enter the right key")
                self.get_logger().info("=============================================")
                self.get_logger().info("[CONTROLS]")
                self.get_logger().info("'j': All Joint Position Calibration")
                self.get_logger().info("'w': Wheel Position Calibration")
                self.get_logger().info("'i': IMU Calibration")
                self.get_logger().info("'q': Quit")
                self.get_logger().info("=============================================\r")
                continue

    def _joint_calibration(self, is_wheel_mode:bool = False):
        """Collect N samples, calculate average, and save offsets to YAML."""
        if self.latest_joint_state is None:
            self.get_logger().warn("No joint states received yet! Cannot calibrate joints.")
            return

        # Settings for sampling
        sleep_interval = 0.05  # 20 * 0.05 = 1.0 second total duration

        self.get_logger().info(f"Collecting {self.sample_count} samples (approx 1 sec)... Keep robot still.")
        
        # Joint position buffer list
        position_samples = []
        joint_names = None

        # Sampling Loop
        for i in range(self.sample_count):
            
            # Exception
            if self.latest_joint_state is None:
                self.get_logger().warn("Joint state lost during sampling!")
                return
            
            if joint_names is None:
                joint_names = self.latest_joint_state.name
            
            # Store current positions
            current_pos = np.array(self.latest_joint_state.position, dtype=float)
            current_pos += self.old_joint_offsets                                         # Restore original position
            position_samples.append(current_pos)
            
            # Wait for next update
            time.sleep(sleep_interval)

        # Calculate Average
        avg_positions = np.mean(position_samples, axis=0)

        # Calculate offsets (Average)
        joint_offsets = avg_positions

        self.get_logger().info(f"Averaged Positions ({self.sample_count} samples): {avg_positions}")
        self.get_logger().info(f"Calculated Joint Offsets: {joint_offsets}")

        # Save to YAML
        self._save_joint_offsets_to_yaml(joint_offsets)

    def _imu_calibration(self):
        """Run EBIMU simple accelerometer calibration (<cas>)."""

        self.get_logger().info("=============================================")
        self.get_logger().info("Simple Accelerometer Calibration")
        self.get_logger().info("Keep the robot BASE LEVEL and STATIONARY.")
        self.get_logger().info("=============================================")

        success = simple_accelerometer_calibration(port="/dev/ttyUSB0", baudrate=115200) 

        if success:
            self.get_logger().info("[SUCCESS] IMU accelerometer calibration completed.")
            self.get_logger().info("Calibration data is stored inside the IMU.")
        else:
            self.get_logger().error("[FAILED] IMU accelerometer calibration failed.")


    def _save_joint_offsets_to_yaml(self, offsets):
        # Read existing file
        data = {}
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                self.get_logger().error(f"Failed to load existing YAML: {e}")
                return
        
        # Update data structure
        formatted_offsets = [float(val) for val in offsets]
        
        if 'joint_offsets' not in data:
            data['joint_offsets'] = {}
            
        data['joint_offsets'] = formatted_offsets

        # Write to file
        try:
            os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            self.get_logger().info(f"\n[SUCCESS] Joint offsets saved to '{self.yaml_path}'\n")
            
        except Exception as e:
            self.get_logger().error(f"Failed to write YAML file: {e}")

    # def _imu_calibration(self):
    #     """IMU Calibration without Yaw"""
    #     # Settings for sampling
    #     sleep_interval = 0.05                           # 20 * 0.05 = 1.0 second total duration

    #     self.get_logger().info(f"Collecting {self.sample_count} samples (approx 1 sec)... Keep robot still.")
        
    #     # IMU buffer list
    #     quat_samples = []
    #     old_imu_offsets_inv = inverse_quat(self.old_imu_offsets)

    #     # Sampling Loop
    #     for i in range(self.sample_count):
            
    #         # Exception
    #         if self.latest_imu_state is None:
    #             self.get_logger().warn("IMU data lost during sampling!")
    #             return
            
    #         # Store current positions
    #         current_quat = np.array([self.latest_imu_state.quat.w,
    #                                  self.latest_imu_state.quat.x,
    #                                  self.latest_imu_state.quat.y,
    #                                  self.latest_imu_state.quat.z], dtype=float)
    #         current_quat = multiply_quat(old_imu_offsets_inv, current_quat)               # Restore original quaternion

    #         # q and -q are the same rotation
    #         if quat_samples and np.dot(quat_samples[0], current_quat) < 0.0:
    #             current_quat = -current_quat

    #         quat_samples.append(current_quat)
            
    #         # Wait for next update
    #         time.sleep(sleep_interval)

    #     # Calculate Average
    #     avg_quat = np.mean(quat_samples, axis=0)
    #     avg_norm = np.linalg.norm(avg_quat)
    #     if avg_norm < 1e-8:
    #         self.get_logger().info("Averaged quaternion is 0. Calibration failed.")
    #         return
    #     avg_quat /= avg_norm

    #     # Local Z axis
    #     z_axis_local = np.array([0.0, 0.0, 1.0])
    #     v_up = rotate_vector_by_quat(avg_quat, z_axis_local)

    #     # Target Z axis
    #     v_target = np.array([0.0, 0.0, 1.0])

    #     # Cross axis between local, target z axis
    #     axis = np.cross(v_up, v_target)
    #     axis_norm = np.linalg.norm(axis)
    #     axis = axis / axis_norm

    #     # Dot product
    #     dot_prod = np.clip(np.dot(v_up, v_target), -1.0, 1.0)
    #     angle = np.arccos(dot_prod)

    #     # Already robot is upright
    #     if axis_norm < 1e-8:
    #         quat_offsets = np.array([1.0, 0.0, 0.0, 0.0])

    #     # Offset quaternion
    #     else:
    #         quat_offsets = axis_angle_to_quat(axis, angle)

    #     self.get_logger().info(f"Calculated IMU quaternion Offsets: {quat_offsets}")

    #     # Save to YAML
    #     self._save_imu_offsets_to_yaml(quat_offsets)

    # def _save_imu_offsets_to_yaml(self, offsets):
    #     # Read existing file
    #     data = {}
    #     if os.path.exists(self.yaml_path):
    #         try:
    #             with open(self.yaml_path, 'r') as f:
    #                 data = yaml.safe_load(f) or {}
    #         except Exception as e:
    #             self.get_logger().error(f"Failed to load existing YAML: {e}")
    #             return
        
    #     # Update data structure
    #     formatted_offsets = [float(val) for val in offsets]
        
    #     if 'imu_offsets' not in data:
    #         data['imu_offsets'] = {}
            
    #     data['imu_offsets'] = formatted_offsets

    #     # Write to file
    #     try:
    #         os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
    #         with open(self.yaml_path, 'w') as f:
    #             yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
    #         self.get_logger().info(f"\n[SUCCESS] IMU offsets saved to '{self.yaml_path}'\n !! Restart all nodes !!")
            
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to write YAML file: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    
    # Try-finally structure to restore terminal settings
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Node exception: {e}")
    finally:
        # Restore terminal settings on exit (Very important)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()