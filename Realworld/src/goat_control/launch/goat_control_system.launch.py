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
    launch_log_viewer_arg = DeclareLaunchArgument(
        "launch_log_viewer",
        default_value="false",
        description="If true, launch log_viewer_node.",
    )
    print_rate_arg = DeclareLaunchArgument(
        "print_rate_hz",
        default_value="100.0",
        description="Print rate for log viewer.",
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
    state_estimation_node = Node(
        package="goat_control",
        executable="state_estimation_node",
        name="state_estimation_node",
        output="screen",
        parameters=[{
            "can_channel": LaunchConfiguration("can_channel"),
            "can_interface": LaunchConfiguration("can_interface"),
            "estimation_rate_hz": LaunchConfiguration("estimation_rate_hz"),
            "imu_port": LaunchConfiguration("imu_port"),
            "imu_baudrate": LaunchConfiguration("imu_baudrate"),
            "yaml_path": LaunchConfiguration("yaml_path"),
        }],
    )

    control_node = Node(
        package="goat_control",
        executable="goat_control_node",
        name="goat_control_node",
        output="screen",
        parameters=[{
            "yaml_path": LaunchConfiguration("yaml_path"),
            "urdf_path": LaunchConfiguration("urdf_path"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "command_unit": LaunchConfiguration("command_unit"),
        }],
    )

    # agent_node = Node(
    #     package="goat_control",
    #     executable="agent_node",
    #     name="agent_node",
    #     output="screen",
    #     parameters=[{
    #     }],
    # )

    # log_viewer_node = Node(
    #     package="goat_control",
    #     executable="log_viewer_node",
    #     name="log_viewer_node",
    #     output="screen",
    #     condition=IfCondition(LaunchConfiguration("launch_log_viewer")),
    #     parameters=[{
    #         "log_topic": "goat/torque_log",
    #         "joint_state_topic": "joint_states",
    #         "use_joint_state_names": True,
    #         "print_rate_hz": LaunchConfiguration("print_rate_hz"),
    #         "command_unit": LaunchConfiguration("command_unit"),
    #         "print_degrees": True,
    #     }],
    # )

    # motor_command_node = Node(
    #     package="goat_control",
    #     executable="motor_command_node",
    #     name="motor_command_node",
    #     output="screen",
    #     parameters=[{
    #         "can_channel": LaunchConfiguration("can_channel"),
    #         "can_interface": LaunchConfiguration("can_interface"),
    #         "yaml_path": LaunchConfiguration("yaml_path"),
    #     }],
    # )

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
        can_channel_arg,
        can_interface_arg,
        control_rate_arg,
        estimation_rate_arg,
        command_unit_arg,
        launch_log_viewer_arg,
        print_rate_arg,
        imu_port_arg,
        imu_baudrate_arg,
        state_estimation_node,
        control_node,
        # policy_node,
        # log_viewer_node,
        motor_io_node,
        # motor_command_node,
    ])