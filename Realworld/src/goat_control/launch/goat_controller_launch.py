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

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description_param]
    )

    node_hardware_driver = Node(
        package='goat_control',
        executable='states_pub',
        name='goat_hardware_driver'
    )

    return LaunchDescription([
        node_robot_state_publisher,
        node_hardware_driver
    ])