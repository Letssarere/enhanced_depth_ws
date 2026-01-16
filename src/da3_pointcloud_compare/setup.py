from glob import glob
from setuptools import setup

package_name = 'da3_pointcloud_compare'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob('launch/*.py')),
    ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ('share/' + package_name + '/config', glob('config/*.yaml')),
]

trt_files = glob('models/da3_small/tensorrt/*')
if trt_files:
    data_files.append(
        ('share/' + package_name + '/models/da3_small/tensorrt', trt_files)
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junho',
    maintainer_email='snowpoet@naver.com',
    description='Compare point clouds using model intrinsics and RealSense intrinsics.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'da3_pointcloud_compare_node = da3_pointcloud_compare.da3_pointcloud_compare_node:main',
        ],
    },
)
