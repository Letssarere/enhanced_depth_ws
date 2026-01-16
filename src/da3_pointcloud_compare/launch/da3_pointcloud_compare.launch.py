from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("da3_pointcloud_compare")
    params_file = PathJoinSubstitution([pkg_share, "config", "compare_params.yaml"])
    rviz_both = PathJoinSubstitution(
        [pkg_share, "rviz", "da3_pointcloud_compare_both.rviz"]
    )
    rviz_model = PathJoinSubstitution(
        [pkg_share, "rviz", "da3_pointcloud_compare_model.rviz"]
    )
    rviz_camera = PathJoinSubstitution(
        [pkg_share, "rviz", "da3_pointcloud_compare_camera.rviz"]
    )

    use_rviz = LaunchConfiguration("use_rviz")
    rviz_view = LaunchConfiguration("rviz_view")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=params_file),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("rviz_view", default_value="both"),
            Node(
                package="da3_pointcloud_compare",
                executable="da3_pointcloud_compare_node",
                name="da3_pointcloud_compare_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_da3_both",
                output="screen",
                arguments=["-d", rviz_both],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            use_rviz,
                            "' == 'true' and '",
                            rviz_view,
                            "' == 'both'",
                        ]
                    )
                ),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_da3_model",
                output="screen",
                arguments=["-d", rviz_model],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            use_rviz,
                            "' == 'true' and '",
                            rviz_view,
                            "' == 'model'",
                        ]
                    )
                ),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_da3_camera",
                output="screen",
                arguments=["-d", rviz_camera],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            use_rviz,
                            "' == 'true' and '",
                            rviz_view,
                            "' == 'camera'",
                        ]
                    )
                ),
            ),
        ]
    )
