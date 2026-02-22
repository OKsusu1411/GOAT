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

# 있으면 설치, 없어도 에러 안 나게
if os.path.isdir("launch"):
    data_files.append((os.path.join("share", package_name, "launch"), safe_glob("launch/*.py")))
if os.path.isdir("urdf"):
    data_files.append((os.path.join("share", package_name, "urdf"), safe_glob("urdf/*")))
if os.path.isdir("config"):
    data_files.append((os.path.join("share", package_name, "config"), safe_glob("config/*")))
if os.path.isdir("meshes"):
    # meshes는 하위 폴더까지 전부 설치
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
    description="GOAT description + helper nodes",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ros2 run goat_description imu_tf_broadcaster
            "imu_tf_broadcaster = goat_description.nodes.imu_tf_broadcaster:main",
        ],
    },
)