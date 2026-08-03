# motor_id.launch.py
#
# WARNING: motor_id_node is interactive (it prompts the operator to rotate the
# shaft by hand). `ros2 launch` spawns nodes with stdin=PIPE and never writes
# to it, so the node can only reach the operator through /dev/tty — and if
# /dev/tty is unavailable (systemd unit, no controlling terminal) the node
# aborts with an explanatory error instead of prompting.
#
# Prefer running it directly from a shell:
#   ros2 run goat_sysid motor_id --ros-args \
#     -p yaml_path:=src/goat_sysid/config/config.yaml \
#     -p joint_index:=0 -p test:=position
#
# This launch file is kept for parameter documentation and for setups where
# /dev/tty is available.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    default_yaml_path = PathJoinSubstitution([
        FindPackageShare("goat_sysid"),
        "config",
        "config.yaml",
    ])

    # Arguments
    yaml_path_arg = DeclareLaunchArgument(
        "yaml_path",
        default_value=default_yaml_path,
        description="Path to the sysid YAML config (default: package share/config/config.yaml).",
    )

    joint_name_arg = DeclareLaunchArgument(
        "joint_name",
        default_value="",
        description="Joint to test by name. Overrides joint_index when non-empty.",
    )

    joint_index_arg = DeclareLaunchArgument(
        "joint_index",
        default_value="0",
        description="Joint to test by index into joint_names (0..7).",
    )

    test_arg = DeclareLaunchArgument(
        "test",
        default_value="position",
        description="Which test to run: 'position' (angle_deg_per_lsb) or "
                    "'velocity' (speed_deg_per_sec_per_lsb).",
    )

    expected_turns_arg = DeclareLaunchArgument(
        "expected_turns",
        default_value="1.0",
        description="position test: how many output-shaft turns you rotate by hand.",
    )

    velocity_duration_arg = DeclareLaunchArgument(
        "velocity_duration_sec",
        default_value="30.0",
        description="velocity test: recording duration in seconds.",
    )

    sample_rate_arg = DeclareLaunchArgument(
        "sample_rate_hz",
        default_value="200.0",
        description="velocity test: target CAN sampling rate (upper bound; CAN may be slower).",
    )

    stream_rate_arg = DeclareLaunchArgument(
        "stream_rate_hz",
        default_value="10.0",
        description="Real-time terminal status line update rate.",
    )

    csv_dir_arg = DeclareLaunchArgument(
        "csv_dir",
        default_value="",
        description="velocity test: CSV output directory (empty = current working dir).",
    )

    # Nodes
    test_node = Node(
        package="goat_sysid",
        executable="motor_id",
        name="motor_id_node",
        output="screen",
        emulate_tty=True,
        # Launch substitutions resolve to strings, so numeric parameters must
        # declare their type explicitly or they arrive as str and fail the
        # node's int()/float() declarations.
        parameters=[{
            "yaml_path": ParameterValue(LaunchConfiguration("yaml_path"), value_type=str),
            "joint_name": ParameterValue(LaunchConfiguration("joint_name"), value_type=str),
            "joint_index": ParameterValue(LaunchConfiguration("joint_index"), value_type=int),
            "test": ParameterValue(LaunchConfiguration("test"), value_type=str),
            "expected_turns": ParameterValue(LaunchConfiguration("expected_turns"), value_type=float),
            "velocity_duration_sec": ParameterValue(
                LaunchConfiguration("velocity_duration_sec"), value_type=float),
            "sample_rate_hz": ParameterValue(LaunchConfiguration("sample_rate_hz"), value_type=float),
            "stream_rate_hz": ParameterValue(LaunchConfiguration("stream_rate_hz"), value_type=float),
            "csv_dir": ParameterValue(LaunchConfiguration("csv_dir"), value_type=str),
        }],
    )

    return LaunchDescription([
        yaml_path_arg,
        joint_name_arg,
        joint_index_arg,
        test_arg,
        expected_turns_arg,
        velocity_duration_arg,
        sample_rate_arg,
        stream_rate_arg,
        csv_dir_arg,
        test_node,
    ])
