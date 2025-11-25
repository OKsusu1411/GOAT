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
    # 🔹 goat_control 패키지에서 YAML 파라미터 파일 경로 얻기
    
    goat_control_share_dir = get_package_share_directory('goat_control')
    timing_yaml = os.path.join(goat_control_share_dir, 'config', 'goat_timing.yaml')

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name="robot_state_publisher",
            parameters=[{"robot_description": robot_description}]
        ),
        Node(
            package='goat_control',
            executable='states_pub',
            name='motor_state_publisher',          # ← # 이름 바꿔야됨 이 부분 반영
            parameters=[timing_yaml]               # ← YAML 적용
        ),

        Node(
            package='goat_control',
            executable='torque_converter',
            name='torque_converter',
            parameters=[timing_yaml]               # ← YAML 적용
        ),

        Node(
            package='goat_control',
            executable='imu_publisher',
            name='imu_publisher',
            parameters=[timing_yaml]               # ← YAML 적용
        )
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