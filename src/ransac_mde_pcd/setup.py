from glob import glob
from setuptools import setup

package_name = "ransac_mde_pcd"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", glob("launch/*.py")),
    ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ("share/" + package_name + "/config", glob("config/*.yaml")),
]

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="junho",
    maintainer_email="snowpoet@naver.com",
    description="Generate point clouds from MDE depth and align them with a RANSAC plane per frame.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ransac_mde_pcd_node = ransac_mde_pcd.ransac_mde_pcd_node:main",
        ],
    },
)
