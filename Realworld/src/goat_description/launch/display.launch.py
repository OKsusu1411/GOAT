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

    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="false",   # 하드웨어 기본은 false 추천
        description="Run joint_state_publisher_gui (ONLY when no external /joint_states publisher).",
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params],
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_gui")),
        # parameters=[params],  # 굳이 없어도 됨(있어도 OK)
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
    )

    return LaunchDescription([
        use_gui_arg,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
    ])
