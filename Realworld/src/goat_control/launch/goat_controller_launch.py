import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # URDF file path
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    goat_path = os.path.normpath(os.path.join(launch_file_dir, '..', '..', '..', '..', '..', '..'))
    urdf_file = os.path.join(goat_path, 'lib/assets/GOAT/WF_GOAT/urdf', 'WF_GOAT.urdf')
    
    # Read URDF
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()
    robot_description_param = {'robot_description': robot_description_content}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description_param]
    )

    states_pub = Node(                            # 이름 바꿔야됨
        package='goat_control',
        executable='states_pub'
    )

    torque_converter = Node(
        package='goat_control',
        executable='torque_converter'
    )

    imu_publisher = Node(
        package='goat_control',
        executable='imu_publisher'
    )

    return LaunchDescription([
        robot_state_publisher,
        states_pub,
        torque_converter,
        imu_publisher
    ])