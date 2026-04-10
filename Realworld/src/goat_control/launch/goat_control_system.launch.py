# goat_control_system.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
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
        FindPackageShare("goat_control"),
        "urdf",
        "WF_GOAT.urdf",
    ])

    # default_ckeckpoint_path = PathJoinSubstitution([
    #     FindPackageShare("goat_control"),
    #     "checkpoint",
    #     "?",                    # TODO: fill it out
    # ])

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
    # checkpoint_path_arg = DeclareLaunchArgument(
    #     "checkpoint_path",
    #     default_value=default_ckeckpoint_path,
    #     description="Path to goat policy checkpoint (default: package share/checkpoint/).",
    # )
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
    imu_io_node = Node(
        package="goat_control",
        executable="imu_io_node",
        name="imu_io_node",
        output="screen",
        parameters=[{
            "imu_port": LaunchConfiguration("imu_port"),
            "imu_baudrate": LaunchConfiguration("imu_baudrate"),
            "yaml_path": LaunchConfiguration("yaml_path"),
        }],
    )

    controller_node = Node(
        package="goat_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        parameters=[{
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "yaml_path": LaunchConfiguration("yaml_path"),
            "urdf_path": LaunchConfiguration("urdf_path"),
            # "checkpoint_path": LaunchConfiguration("checkpoint_path"),
        }],
    )

    motor_io_node = Node(
        package="goat_control",
        executable="motor_io_node",
        name="motor_io_node",
        output="screen",
        parameters=[{
            "can_channel": LaunchConfiguration("can_channel"),
            "can_interface": LaunchConfiguration("can_interface"),
            "yaml_path": LaunchConfiguration("yaml_path"),
            "io_rate_hz": LaunchConfiguration("control_rate_hz"),
        }],
    )
    
    return LaunchDescription([
        yaml_path_arg,
        urdf_path_arg,
        # checkpoint_path_arg,
        can_channel_arg,
        can_interface_arg,
        control_rate_arg,
        imu_port_arg,
        imu_baudrate_arg,
        imu_io_node,
        # controller_node,
        motor_io_node,
    ])