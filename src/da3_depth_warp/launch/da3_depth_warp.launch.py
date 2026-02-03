from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("da3_depth_warp")
    params_file = PathJoinSubstitution(
        [pkg_share, "config", "da3_depth_warp_params.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=params_file),
            Node(
                package="da3_depth_warp",
                executable="da3_depth_warp_node",
                name="da3_depth_warp_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
