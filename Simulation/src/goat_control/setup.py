import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'goat_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        # YAML configs
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
        # URDF
        (os.path.join('share', package_name, 'urdf'),
         glob(os.path.join('urdf', '*'))),
        # Policy checkpoints
        (os.path.join('share', package_name, 'checkpoint'),
         glob(os.path.join('checkpoint', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oksusu',
    maintainer_email='hansusu1411@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim_controller_node = goat_control.nodes.sim_controller_node:main',
        ],
    },
)
