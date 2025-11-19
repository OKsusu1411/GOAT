import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def prepare_launch(context, *args, **kwargs):

    urdf_path = LaunchConfiguration("model").perform(context)
    with open(urdf_path, "r") as infp:
        robot_description = infp.read()

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name="robot_state_publisher",
            parameters=[{"robot_description": robot_description}]
        ),

        # Node(                            # 이름 바꿔야됨
        #     package='goat_control',
        #     executable='states_pub',
        #     name='states_pub'        
        # ),

        # Node(
        #     package='goat_control',
        #     executable='torque_converter',
        #     name='torque_converter'
        # ),

        # Node(
        #     package='goat_control',
        #     executable='imu_publisher',
        #     name='imu_publisher'
        # )
    ]

def generate_launch_description():

    goat_control_share_dir = get_package_share_directory('goat_control')
    urdf_path = os.path.join(goat_control_share_dir, 'assets', 'WF_GOAT', 'urdf', 'WF_GOAT.urdf')

    return LaunchDescription([
        DeclareLaunchArgument(
            name="model",
            default_value=urdf_path,
            description="URDF file path"
        ),

        OpaqueFunction(function=prepare_launch)
    ])