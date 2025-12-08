from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # goat_description 패키지 안에 설치된 urdf 파일 경로
    pkg_share = get_package_share_directory('goat_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'WF_GOAT.urdf')

    # URDF 파일 내용 읽어서 robot_description 파라미터로 넣기
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    params = {'robot_description': robot_description}

    # 노드들 정의
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[params],
    )

#    joint_state_publisher_gui_node = Node(
#        package='joint_state_publisher_gui',
#        executable='joint_state_publisher_gui',
#        name='joint_state_publisher_gui',
#        output='screen',
#        parameters=[params],
#    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )

    # LaunchDescription 리턴 필수!!
    return LaunchDescription([
        robot_state_publisher_node,
#        joint_state_publisher_gui_node,
        rviz_node,
    ])
