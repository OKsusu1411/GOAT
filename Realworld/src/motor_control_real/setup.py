from setuptools import find_packages, setup

package_name = 'motor_control_real'

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
    maintainer_email='heachanlee@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_state_publisher = motor_control_real.motor_state_publisher:main',
            'motor_torque_control = motor_control_real.motor_torque_control:main',
            'motor_torque_command_publisher = motor_control_real.motor_torque_command_publisher:main',
            'motor_states_echo = motor_control_real.motor_states_echo:main',
            'motor_temp_monitor = motor_control_real.motor_temp_logger:main',

        ],
    },
)
