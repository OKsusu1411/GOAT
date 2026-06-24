# goat_sim_bringup.launch.py
#
# Single-command bringup for Isaac Sim visualization:
#   - goat_control_system_sim.launch.py : topic_converter_node + sim_controller_node (control)
#   - display.launch.py (goat_description) : robot_state_publisher + imu_tf_publisher + rviz2 (viz)
#
# Prerequisite: Isaac Sim must already be publishing /sim_joint_states, /sim_imu, /sim_odom.
# Usage:
#   ros2 launch goat_control goat_sim_bringup.launch.py
#   ros2 launch goat_control goat_sim_bringup.launch.py use_sim_time:=true

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    # -------------------------------
    # Launch Arguments
    # -------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Isaac Sim) clock across control + visualization nodes.",
    )

    # -------------------------------
    # Control side: converter + sim controller
    # -------------------------------
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("goat_control"),
                "launch",
                "goat_control_system_sim.launch.py",
            ])
        ),
    )

    # -------------------------------
    # Visualization side: RSP + imu_tf + rviz (from goat_description)
    # jsp_gui stays OFF (external /joint_states is the source of truth).
    # -------------------------------
    display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("goat_description"),
                "launch",
                "display.launch.py",
            ])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_gui": "false",
        }.items(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        control,
        display,
    ])
