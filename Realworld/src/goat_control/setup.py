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

        # YAML
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml')) + glob(os.path.join('config', '*.yml'))),

        # URDF
        (os.path.join("share", package_name, "urdf"),
         glob(os.path.join("urdf/*"))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HansuKim',
    maintainer_email='hansusu1411@gmail.com',
    description='GOAT ROS2 control package (refactored core + nodes)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'calibration_node = goat_control.nodes.calibration_node:main',
            'controller_node = goat_control.nodes.controller_node:main',
            'log_viewer_node = goat_control.nodes.log_viewer_node:main',
            'sim_controller_node = goat_control.nodes.sim_controller_node:main',
            'topic_converter_node = goat_control.nodes.topic_converter_node:main',
        ],
    },
)
