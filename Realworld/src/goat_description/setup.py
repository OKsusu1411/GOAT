from setuptools import find_packages, setup
import os
from glob import glob

package_name = "goat_description"

def safe_glob(pattern):
    return glob(pattern, recursive=True)

data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
]

if os.path.isdir("launch"):
    data_files.append((os.path.join("share", package_name, "launch"), safe_glob("launch/*.py")))
if os.path.isdir("urdf"):
    data_files.append((os.path.join("share", package_name, "urdf"), safe_glob("urdf/*")))
if os.path.isdir("config"):
    data_files.append((os.path.join("share", package_name, "config"), safe_glob("config/*")))
if os.path.isdir("meshes"):
    data_files.append((os.path.join("share", package_name, "meshes"), safe_glob("meshes/**/*")))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
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
        "console_scripts": [
            "imu_tf_publisher = goat_description.nodes.imu_tf_publisher:main",
        ],
    },
)