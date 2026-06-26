# goat_control_system_sim.launch.py

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
    control_rate_arg = DeclareLaunchArgument(
        "control_rate_hz",
        default_value="200.0",
        description="Control loop rate for GoatControlNode.",
    )
    estimation_rate_arg = DeclareLaunchArgument(
        "estimation_rate_hz",
        default_value="200.0",
        description="Estimation loop rate for StateEstimationNode.",
    )
    command_unit_arg = DeclareLaunchArgument(
        "command_unit",
        default_value="torque_nm",
        description="Command unit for motors: 'torque_nm' or 'amp'.",
    )

    controller_node = Node(
        package="goat_control",
        executable="sim_controller_node",
        name="sim_controller_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
            "urdf_path": LaunchConfiguration("urdf_path"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "use_sim_time": True,
        }],
    )

    topic_converter_node = Node(
        package="goat_control",
        executable="topic_converter_node",
        name="topic_converter_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
        }],
    )
    
    return LaunchDescription([
        yaml_path_arg,
        urdf_path_arg,
        control_rate_arg,
        estimation_rate_arg,
        command_unit_arg,
        controller_node,
        topic_converter_node,
    ])