import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'goat_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index + package.xml
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # launch files -> share/goat_control/launch
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),

        # config files -> share/goat_control/config
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml')) + glob(os.path.join('config', '*.yml'))),

        # assets (existing)
        (os.path.join('share', package_name, 'assets', 'WF_GOAT', 'urdf'),
         glob('../../../lib/assets/GOAT/WF_GOAT/urdf/*.urdf')),

        # (os.path.join('share', package_name, 'assets', 'PF_GOAT', 'urdf'),
        #  glob('../../../lib/assets/TRON/PF_TRON1A/urdf/*.urdf')),
        # (os.path.join('share', package_name, 'assets/PF_GOAT/meshes'),
        #  glob('meshes/*.STL')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='heachanlee',
    maintainer_email='eojin333c@gmail.com',
    description='GOAT ROS2 control package (refactored core + nodes)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # --- existing scripts ---
            'states_pub = goat_control.states_pub:main',
            'torque_converter = goat_control.torque_converter:main',
            'torque_test_publisher = goat_control.torque_test_publisher:main',
            'motor_states_echo = goat_control.motor_states_echo:main',
            'policy = goat_control.policy:main',
            'pd_controller = goat_control.pd_controller:main',
            'joint_torque_controller = goat_control.joint_torque_controller:main',
            'data_logger = goat_control.data_logger:main',

            # --- NEW refactored ROS2 nodes ---
            'goat_control_node = goat_control.nodes.control_node:main',
            'state_estimation_node = goat_control.nodes.state_estimation_node:main',
            'policy_node = goat_control.nodes.policy_node:main',
            'log_viewer_node = goat_control.nodes.log_viewer_node:main',
            'policy_keyboard_tester = goat_control.nodes.policy_keyboard_tester:main',
        ],
    },
)
