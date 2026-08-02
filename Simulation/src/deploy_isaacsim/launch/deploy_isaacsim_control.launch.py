from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Config lives in the Realworld workspace: single source of truth shared with
    # the real robot, so sim and real cannot drift apart.
    default_yaml_path = PathJoinSubstitution([
        FindPackageShare("goat_control"),
        "config",
        "goat_config.yaml",
    ])

    # URDF is owned by goat_description (used by NominalController via pinocchio).
    default_urdf_path = PathJoinSubstitution([
        FindPackageShare("goat_description"),
        "urdf",
        "WF_GOAT.urdf",
    ])

    # Absolute path to the installed policy checkpoint. The YAML value is
    # CWD-relative and only resolves when launched from the Realworld workspace
    # root, so it is overridden here.
    default_checkpoint_path = PathJoinSubstitution([
        FindPackageShare("goat_control"),
        "checkpoint",
        "stand.onnx",
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
    checkpoint_path_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value=default_checkpoint_path,
        description="Absolute path to the ONNX policy checkpoint. Overrides the YAML value.",
    )
    control_rate_arg = DeclareLaunchArgument(
        "control_rate_hz",
        default_value="200.0",
        description="Control loop rate for SimControllerNode.",
    )

    controller_node = Node(
        package="deploy_isaacsim",
        executable="sim_controller_node",
        name="sim_controller_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
            "urdf_path": LaunchConfiguration("urdf_path"),
            "checkpoint_path": LaunchConfiguration("checkpoint_path"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "use_sim_time": True,
        }],
    )

    topic_converter_node = Node(
        package="deploy_isaacsim",
        executable="topic_converter_node",
        name="topic_converter_node",
        output="screen",
    )

    return LaunchDescription([
        yaml_path_arg,
        urdf_path_arg,
        checkpoint_path_arg,
        control_rate_arg,
        controller_node,
        topic_converter_node,
    ])
