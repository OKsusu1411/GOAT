from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'goat_sysid'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),

        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HongryeolYoon',
    maintainer_email='eojin333c@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'actuator_torque_test = goat_sysid.actuator_torque_test:main',
            'actuator_target_test = goat_sysid.actuator_target_test:main',
            'motor_id = goat_sysid.motor_id:main',
            'breakaway_torque_tester = goat_sysid.breakaway_torque_tester:main',
            'dynamic_friction_sysid = goat_sysid.dynamic_friction_sysid:main',
            'friction_id_node = goat_sysid.dynamic_friction_id_node:main',
            'wheel_friction_id = goat_sysid.wheel_friction_id_node:main',
            'sine_motion_logger_node = goat_sysid.sine_motion_logger_node:main',
        ],
    },
)
