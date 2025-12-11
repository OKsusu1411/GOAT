from setuptools import find_packages, setup

package_name = 'goat_sysid'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='heachanlee',
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
            'breakaway_torque_tester = goat_sysid.breakaway_torque_tester:main',
            'dynamic_friction_sysid = goat_sysid.dynamic_friction_sysid:main',
            'wheel_step_sysid = goat_sysid.wheel_step_sysid:main',
            'step_trajectory_publisher = goat_sysid.step_trajectory_publisher:main',
        ],
    },
)
