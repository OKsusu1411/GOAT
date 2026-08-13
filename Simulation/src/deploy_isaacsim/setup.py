import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'deploy_isaacsim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index + package.xml
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # launch files -> share/deploy_isaacsim/launch
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),

        # Isaac Sim scene assets -> share/deploy_isaacsim/usd
        (os.path.join('share', package_name, 'usd'),
         glob(os.path.join('usd', '*.usd'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HansuKim',
    maintainer_email='hansusu1411@gmail.com',
    description='Isaac Sim deployment bridge for the GOAT robot (sim-side nodes only).',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim_controller_node = deploy_isaacsim.nodes.sim_controller_node:main',
            'topic_converter_node = deploy_isaacsim.nodes.topic_converter_node:main',
            'sim_log_viewer_node = deploy_isaacsim.nodes.sim_log_viewer_node:main',
            'sim_actuator_target_test_node = deploy_isaacsim.nodes.sim_actuator_target_test_node:main',
            'sim_friction_id_node = deploy_isaacsim.nodes.sim_friction_id_node:main',
            'log_joint_csv = deploy_isaacsim.nodes.log_joint_csv:main',
            'log_state_csv = deploy_isaacsim.nodes.log_state_csv:main',
        ],
    },
)
