from glob import glob
from setuptools import setup

package_name = 'table_depth_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.npz') + glob('config/*.yaml')),
        (
            'share/' + package_name + '/models/depth_anything_v3_small_onnx',
            ['models/depth_anything_v3_small_onnx/config.json'],
        ),
        (
            'share/' + package_name + '/models/depth_anything_v3_small_onnx/onnx',
            glob('models/depth_anything_v3_small_onnx/onnx/*'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junho',
    maintainer_email='snowpoet@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_depth_calibration_node = table_depth_fusion.fusion_depth_calibration_node:main',
            'fusion_depth_node = table_depth_fusion.fusion_depth_node:main',
        ],
    },
)
