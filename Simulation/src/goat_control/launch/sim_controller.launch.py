# sim_controller.launch.py — launches sim_controller_node with default yaml/urdf/checkpoint.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Default paths point at the Simulation goat_control install share/ tree.
    pkg_share = FindPackageShare("goat_control")

    default_yaml_path = PathJoinSubstitution(
        [pkg_share, "config", "goat_config.yaml"]
    )
    default_urdf_path = PathJoinSubstitution(
        [pkg_share, "urdf", "WF_GOAT.urdf"]
    )
    default_checkpoint_path = PathJoinSubstitution(
        [pkg_share, "checkpoint", "agent_jit_56000.pt"]
    )

    # Launch args — overridable from the CLI with `key:=value`.
    yaml_path_arg = DeclareLaunchArgument(
        "yaml_path",
        default_value=default_yaml_path,
        description="Path to goat YAML config.",
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=default_urdf_path,
        description="Path to goat URDF.",
    )
    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value=default_checkpoint_path,
        description="Path to policy checkpoint (overrides yaml's relative path).",
    )

    # The single node this launch starts.
    controller_node = Node(
        package="goat_control",
        executable="sim_controller_node",
        name="sim_controller_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
            "urdf_path": LaunchConfiguration("urdf_path"),
            "checkpoint_path": LaunchConfiguration("checkpoint_path"),
        }],
    )

    return LaunchDescription([
        yaml_path_arg,
        urdf_path_arg,
        checkpoint_path_arg,
        controller_node,
    ])
