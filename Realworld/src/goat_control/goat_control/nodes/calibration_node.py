# goat_control/nodes/calibration_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from motor_interfaces.msg import BaseStates

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
            JointState, "joint_states", self._on_joint_state_msg, 10
        )
        self.imu_subscriber = self.create_subscription(
            BaseStates, "/goat/imu_data", self._on_imu_msg, 10
        )

        # Data buffers
        self.latest_joint_state = None
        self.latest_imu_state = None

        # Save terminal settings (to restore on exit)
        self.settings = termios.tcgetattr(sys.stdin)

        # Start keyboard listener thread
        self.input_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self.input_thread.start()

        # Print UI
        self.get_logger().info("Calibration Node Started.")
        self.get_logger().info(f"Target YAML: {self.yaml_path}")
        print("\n" + "="*30)
        print(" [CONTROLS]\n")
        print("  'j': Joint Calibration\n")
        print("  'i': IMU Calibration\n")
        print("  'q': Quit\n")
        print("="*30 + "\n")

    def _on_joint_state_msg(self, msg: JointState):
        self.latest_joint_state = msg

    def _on_imu_msg(self, msg: BaseStates):
        self.latest_imu_state = msg

    def _get_key(self):
        """Read a single character from the terminal immediately (Blocking)."""
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def _keyboard_listener_loop(self):
        """Main loop to monitor keyboard input."""
        while rclpy.ok():
            key = self._get_key()
            
            if key == 'j':
                self.get_logger().info("Key 'j' pressed: Starting Joint Calibration")
                self._joint_calibration()
                
            elif key == 'i':
                self.get_logger().info("Key 'i' pressed: Starting IMU Calibration")
                self._imu_calibration()
                
            elif key == 'q':
                self.get_logger().info("Key 'q' pressed: Shutting down node...")
                rclpy.shutdown()
                break
            
            # Handle Ctrl+C
            elif key == '\x03':
                rclpy.shutdown()
                break
            
            # Execption
            else:
                self.get_logger().info("Wrong key! Please enter the right key")
                print("\n" + "="*30)
                print(" [CONTROLS]\n")
                print("  'j': Joint Calibration\n")
                print("  'i': IMU Calibration\n")
                print("  'q': Quit\n")
                print("="*30 + "\n")
                continue

    def _joint_calibration(self):
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
            
            # 1. Store current positions
            current_pos = np.array(self.latest_joint_state.position, dtype=float)
            position_samples.append(current_pos)
            
            # Wait for next update
            time.sleep(sleep_interval)

        # 2. Calculate Average
        avg_positions = np.mean(position_samples, axis=0)

        # 3. Calculate offsets (Average)
        joint_offsets = avg_positions

        self.get_logger().info(f"Averaged Positions ({self.sample_count} samples): {avg_positions}")
        self.get_logger().info(f"Calculated Joint Offsets: {joint_offsets}")

        # 4. Save to YAML
        self._save_joint_offsets_to_yaml(joint_offsets)
    
    def _imu_calibration(self):
        """IMU Calibration Logic (Placeholder)."""
        # TODO: Implement IMU calibration logic here
        pass

    def _save_joint_offsets_to_yaml(self, offsets):
        # 1. Read existing file
        data = {}
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                self.get_logger().error(f"Failed to load existing YAML: {e}")
                return
        
        # 2. Update data structure
        formatted_offsets = [float(val) for val in offsets]
        
        if 'calibration' not in data:
            data['calibration'] = {}
            
        data['calibration']['joint_offsets'] = formatted_offsets

        # 3. Write to file
        try:
            os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            print(f"\n[SUCCESS] Joint offsets saved to '{self.yaml_path}'\n")
            
        except Exception as e:
            self.get_logger().error(f"Failed to write YAML file: {e}")

    def _save_imu_offsets_to_yaml(self, offsets):
        # 1. Read existing file
        data = {}
        if os.path.exists(self.yaml_path):
            try:
                with open(self.yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                self.get_logger().error(f"Failed to load existing YAML: {e}")
                return
        
        # 2. Update data structure
        formatted_offsets = [float(val) for val in offsets]
        
        if 'calibration' not in data:
            data['calibration'] = {}
            
        data['calibration']['imu_offsets'] = formatted_offsets

        # 3. Write to file
        try:
            os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            print(f"\n[SUCCESS] IMU offsets saved to '{self.yaml_path}'\n !! Restart all nodes !!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to write YAML file: {e}")


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