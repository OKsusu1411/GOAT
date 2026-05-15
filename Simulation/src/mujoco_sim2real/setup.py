import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mujoco_sim2real'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'xml'),
            glob(os.path.join('xml/*'))),
        (os.path.join('share', package_name, 'meshes'),
            glob(os.path.join('meshes/*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oksusu',
    maintainer_email='hansusu1411@gmail.com',
    description='MuJoCo ROS2 bridge for the GOAT robot (sim2real).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mujoco_ros2_bridge = mujoco_sim2real.mujoco_ros2_bridge:main',
        ],
    },
)
