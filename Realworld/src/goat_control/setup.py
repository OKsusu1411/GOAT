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
            'states_pub = goat_control.states_pub:main',                    # TODO: change name later
            'torque_converter = goat_control.torque_converter:main',
            'torque_test_publisher = goat_control.torque_test_publisher:main',
            'motor_states_echo = goat_control.motor_states_echo:main',
        ],
    },
)
