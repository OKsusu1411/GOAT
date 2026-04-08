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

        # URDF
        (os.path.join("share", package_name, "urdf"),
         glob(os.path.join("urdf/*"))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haechanlee',
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
            'goat_control_node = goat_control.nodes.control_node:main',
            'state_estimation_node = goat_control.nodes.state_estimation_node:main',
            'motor_io_node = goat_control.nodes.motor_io_node:main',
            'agent_node = goat_control.nodes.agent_node:main',
            'log_viewer_node = goat_control.nodes.log_viewer_node:main',
            'policy_keyboard_tester = goat_control.nodes.policy_keyboard_tester:main',
            'plot_node = goat_control.nodes.plot_node:main',
            'calibration_node = goat_control.nodes.calibration_node:main',
            'nsc_tester_node = goat_control.nodes.nsc_tester_node:main',

            'calibration_node = goat_control.nodes_develop.calibration_node:main',
            'controller_node = goat_control.nodes_develop.controller_node:main',
            'imu_io_node = goat_control.nodes_develop.imu_io_node:main',
            'log_viewer_node = goat_control.nodes_develop.log_viewer_node:main',
            'motor_io_node = goat_control.nodes_develop.motor_io_node:main',
            'topic_converter_node = goat_control.nodes_develop.topic_converter_node:main',
        ],
    },
)
