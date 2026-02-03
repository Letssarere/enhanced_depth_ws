from glob import glob
from setuptools import setup

package_name = "da3_depth_warp"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", glob("launch/*.py")),
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
    description="Postprocess DA3 depth with perspective warp and adaptive threshold.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "da3_depth_warp_node = da3_depth_warp.da3_depth_warp_node:main",
        ],
    },
)
