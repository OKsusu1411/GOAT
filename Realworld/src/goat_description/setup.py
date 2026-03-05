from setuptools import find_packages, setup
import os
from glob import glob

package_name = "goat_description"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch/*.launch.py"))),
        (os.path.join("share", package_name, "urdf"), glob(os.path.join("urdf/*"))),
        (os.path.join("share", package_name, "meshes"), glob(os.path.join("meshes/*"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="heachanlee",
    maintainer_email="eojin333c@gmail.com",
    description="GOAT Rviz display + helper nodes",
    license="TODO: License declaration",
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'imu_tf_publisher = goat_description.nodes.imu_tf_publisher:main',
        ],
    },
)