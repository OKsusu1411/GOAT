from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot = LaunchConfiguration('robot')
    model_path = LaunchConfiguration('model_path')
    config_file = LaunchConfiguration('config_file')

    pkg = FindPackageShare('goat')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value='WF_GOAT',
            description='Robot name; selects config/<robot>/.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution([pkg, 'config', robot, 'xml', 'goat_on_stand.xml']),
            description='MJCF model path (defaults to config/<robot>/xml/goat_floating.xml).',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([pkg, 'config', robot, 'yaml', 'goat_mujoco.yaml']),
            description='YAML parameter file for the simulator node.',
        ),
        Node(
            package='goat',
            executable='goat_mujoco_node',
            name='goat_mujoco_node',
            output='screen',
            parameters=[config_file, {'model_path': model_path}],
        ),
    ])
