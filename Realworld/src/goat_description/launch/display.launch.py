from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_share = get_package_share_directory('goat_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'WF_GOAT.urdf')

    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    params = {'robot_description': robot_description}

    # -------------------------------
    # Launch Arguments
    # -------------------------------
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="false",
        description="Run joint_state_publisher_gui (ONLY when no external /joint_states publisher).",
    )

    # -------------------------------
    # Robot State Publisher
    # -------------------------------
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params],
    )

    # -------------------------------
    # Joint State Publisher GUI
    # -------------------------------
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_gui")),
    )

    # -------------------------------
    # IMU TF Broadcaster (추가!)
    # -------------------------------
    imu_tf_publisher_node = Node(
        package='goat_description',
        executable='imu_tf_publisher',
        name='imu_tf_publisher',
        output='screen',
        parameters=[
            {
                "imu_topic": "/imu_data",
                "parent_frame": "odom",
                "child_frame": "base_link",
                "use_translation_zero": True,
            }
        ],
    )

    # -------------------------------
    # RViz
    # -------------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
    )

    return LaunchDescription([
        use_gui_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        imu_tf_publisher_node,
        rviz_node,
    ])