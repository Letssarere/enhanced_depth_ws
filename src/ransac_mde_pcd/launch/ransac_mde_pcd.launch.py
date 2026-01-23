from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("ransac_mde_pcd")
    params_file = PathJoinSubstitution(
        [pkg_share, "config", "ransac_mde_pcd_params.yaml"]
    )
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "ransac_mde_pcd.rviz"])

    use_rviz = LaunchConfiguration("use_rviz")
    use_table_roi = LaunchConfiguration("use_table_roi")
    roi_yaml_path = LaunchConfiguration("roi_yaml_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=params_file),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_table_roi", default_value="true"),
            DeclareLaunchArgument("roi_yaml_path", default_value=""),
            Node(
                package="ransac_mde_pcd",
                executable="ransac_mde_pcd_node",
                name="ransac_mde_pcd_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "use_table_roi": use_table_roi,
                        "roi_yaml_path": roi_yaml_path,
                    },
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_ransac_mde_pcd",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
