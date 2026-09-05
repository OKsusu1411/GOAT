# goat_control_system.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Default YAML path = <share/goat_control>/config/goat_config.yaml
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
        description="Path to goat YAML config (default: package share/config/goat_config.yaml).",
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=default_urdf_path,
        description="Path to goat URDF (default: package share/urdf/WF_GOAT.urdf).",
    )
    can_channel_arg = DeclareLaunchArgument(
        "can_channel",
        default_value="can0",
        description="SocketCAN channel name.",
    )
    can_interface_arg = DeclareLaunchArgument(
        "can_interface",
        default_value="socketcan",
        description="python-can interface type.",
    )
    control_rate_arg = DeclareLaunchArgument(
        "control_rate_hz",
        default_value="200.0",
        description="Control loop rate for GoatControlNode.",
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

    # Nodes
    controller_node = Node(
        package="goat_sysid",
        executable="actuator_target_test",
        name="actuator_target_test_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
            "urdf_path": LaunchConfiguration("urdf_path"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "imu_port": LaunchConfiguration("imu_port"),
            "imu_baudrate": LaunchConfiguration("imu_baudrate"),
        }],
    )

    return LaunchDescription([
        yaml_path_arg,
        urdf_path_arg,
        can_channel_arg,
        can_interface_arg,
        control_rate_arg,
        imu_port_arg,
        imu_baudrate_arg,
        controller_node,
    ])