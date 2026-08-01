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

        # Policy checkpoints -> share/goat_control/checkpoint
        # NOTE: these are ONNX, not .pt. The previous "*.pt" glob matched nothing,
        # so no checkpoint was ever installed and configs had to use a
        # CWD-relative path. Launch files now inject the installed absolute path.
        (os.path.join("share", package_name, "checkpoint"),
         glob(os.path.join("checkpoint", "*.onnx"))),

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
        ],
    },
)
