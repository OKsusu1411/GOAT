import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # URDF file path
    pkg_path = get_package_share_directory('goat_control')
    urdf_file = os.path.join(pkg_path, 'assets/WF_GOAT/urdf', 'WF_GOAT.urdf')
    
    # Read URDF
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()
    robot_description_param = {'robot_description': robot_description_content}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description_param]
    )

    motor_states_publisher = Node(
        package='goat_control',
        executable='motor_states_publisher'
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
        motor_states_publisher,
        torque_converter,
        imu_publisher
    ])