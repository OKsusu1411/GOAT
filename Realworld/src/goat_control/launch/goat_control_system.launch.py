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

    # Arguments
    yaml_path_arg = DeclareLaunchArgument(
        "yaml_path",
        default_value=default_yaml_path,
        description="Path to goat YAML config (default: package share/config/goat_config.yaml).",
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
    command_unit_arg = DeclareLaunchArgument(
        "command_unit",
        default_value="torque_nm",
        description="Command unit for motors: 'torque_nm' or 'amp'.",
    )
    launch_log_viewer_arg = DeclareLaunchArgument(
        "launch_log_viewer",
        default_value="false",
        description="If true, launch log_viewer_node.",
    )
    print_rate_arg = DeclareLaunchArgument(
        "print_rate_hz",
        default_value="50.0",
        description="Print rate for log viewer.",
    )

    # Nodes
    control_node = Node(
        package="goat_control",
        executable="goat_control_node",
        name="goat_control_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
            "can_channel": LaunchConfiguration("can_channel"),
            "can_interface": LaunchConfiguration("can_interface"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "command_unit": LaunchConfiguration("command_unit"),
        }],
    )

    policy_node = Node(
        package="goat_control",
        executable="policy_node",
        name="policy",
        output="screen",
        parameters=[{
            "action_frequency": 0.02,
        }],
    )

    log_viewer_node = Node(
        package="goat_control",
        executable="log_viewer_node",
        name="motor_torque_log_viewer",
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_log_viewer")),
        parameters=[{
            "log_topic": "motor_torque_log",
            "joint_state_topic": "joint_states",
            "use_joint_state_names": True,
            "print_rate_hz": LaunchConfiguration("print_rate_hz"),
            "command_unit": LaunchConfiguration("command_unit"),
            "print_degrees": False,
        }],
    )

    return LaunchDescription([
        yaml_path_arg,
        can_channel_arg,
        can_interface_arg,
        control_rate_arg,
        command_unit_arg,
        launch_log_viewer_arg,
        print_rate_arg,
        control_node,
        policy_node,
        log_viewer_node,
    ])
