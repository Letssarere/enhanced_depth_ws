from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("table_depth_fusion"), "rviz", "table_depth_fusion.rviz"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz_config", default_value=rviz_config),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument(
                "parent_frame", default_value="camera_color_optical_frame"
            ),
            DeclareLaunchArgument("child_frame", default_value="table_link"),
            DeclareLaunchArgument("translation_scale", default_value="1.0"),
            Node(
                package="table_depth_fusion",
                executable="table_config_tf_node",
                name="table_config_tf_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "parent_frame": LaunchConfiguration("parent_frame"),
                        "child_frame": LaunchConfiguration("child_frame"),
                        "translation_scale": LaunchConfiguration("translation_scale"),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            ),
        ]
    )
