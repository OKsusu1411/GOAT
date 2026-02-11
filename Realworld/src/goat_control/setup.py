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
            'goat_control_node = goat_control.nodes.control_node:main',
            'state_estimation_node = goat_control.nodes.state_estimation_node:main',
            'motor_command_node = goat_control.nodes.motor_command_node:main',
            'motor_io_node = goat_control.nodes.motor_io_node:main',
            'policy_node = goat_control.nodes.policy_node:main',
            'log_viewer_node = goat_control.nodes.log_viewer_node:main',
            'policy_keyboard_tester = test.policy_keyboard_tester:main',
            'plot_node = goat_control.nodes.plot_node:main',
        ],
    },
)
