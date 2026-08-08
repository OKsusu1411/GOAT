# actuator_test.launch.py
#
# Launches motor_friction_id (ActuatorTargetTestNode), which reads keyboard
# input from /dev/tty. Same caveat as motor_id.launch.py: run from a foreground
# terminal — not a systemd unit — because the node needs a controlling terminal.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Default paths (mirror goat_control_system.launch.py)
    default_yaml_path = PathJoinSubstitution([
        FindPackageShare("goat_control"),
        "config",
        "goat_config.yaml",
    ])
    default_urdf_path = PathJoinSubstitution([
        FindPackageShare("goat_description"),
        "urdf",
        "WF_GOAT.urdf",
    ])

    # Arguments
    yaml_path_arg = DeclareLaunchArgument(
        "yaml_path",
        default_value=default_yaml_path,
        description="Path to goat YAML config (default: goat_control share/config/goat_config.yaml).",
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=default_urdf_path,
        description="Path to goat URDF (default: goat_description share/urdf/WF_GOAT.urdf).",
    )
    control_rate_arg = DeclareLaunchArgument(
        "control_rate_hz",
        default_value="200.0",
        description="Control loop rate for ActuatorTargetTestNode.",
    )
    imu_port_arg = DeclareLaunchArgument(
        "imu_port",
        default_value="/dev/ttyUSB0",
        description="Serial port for the IMU.",
    )
    imu_baudrate_arg = DeclareLaunchArgument(
        "imu_baudrate",
        default_value="115200",
        description="Baudrate for the IMU.",
    )
    imu_timeout_arg = DeclareLaunchArgument(
        "imu_timeout",
        default_value="1.0",
        description="Serial read timeout for the IMU [s].",
    )
    action_timeout_arg = DeclareLaunchArgument(
        "action_timeout_sec",
        default_value="0.05",
        description="Per-tick CAN action timeout [s].",
    )

    # Node
    friction_id_node = Node(
        package="goat_sysid",
        executable="motor_friction_id",
        name="actuator_target_test_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "yaml_path": ParameterValue(LaunchConfiguration("yaml_path"), value_type=str),
            "urdf_path": ParameterValue(LaunchConfiguration("urdf_path"), value_type=str),
            "control_rate_hz": ParameterValue(LaunchConfiguration("control_rate_hz"), value_type=float),
            "imu_port": ParameterValue(LaunchConfiguration("imu_port"), value_type=str),
            "imu_baudrate": ParameterValue(LaunchConfiguration("imu_baudrate"), value_type=int),
            "imu_timeout": ParameterValue(LaunchConfiguration("imu_timeout"), value_type=float),
            "action_timeout_sec": ParameterValue(LaunchConfiguration("action_timeout_sec"), value_type=float),
        }],
    )

    return LaunchDescription([
        yaml_path_arg,
        urdf_path_arg,
        control_rate_arg,
        imu_port_arg,
        imu_baudrate_arg,
        imu_timeout_arg,
        action_timeout_arg,
        friction_id_node,
    ])
