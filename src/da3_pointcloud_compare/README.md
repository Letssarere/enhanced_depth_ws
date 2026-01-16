# DA3 Point Cloud Compare

Compare point clouds generated from the same TensorRT depth output using two sets of intrinsics:
- Model-predicted intrinsics.
- RealSense intrinsics from camera_info.

## Topics
- Input:
  - /camera/color/image_raw (sensor_msgs/Image)
  - /camera/color/camera_info (sensor_msgs/CameraInfo)
- Output:
  - /da3/points_model_k (sensor_msgs/PointCloud2)
  - /da3/points_camera_k (sensor_msgs/PointCloud2)

## Launch
- Both clouds:
  - ros2 launch da3_pointcloud_compare da3_pointcloud_compare.launch.py rviz_view:=both
- Model only:
  - ros2 launch da3_pointcloud_compare da3_pointcloud_compare.launch.py rviz_view:=model
- Camera only:
  - ros2 launch da3_pointcloud_compare da3_pointcloud_compare.launch.py rviz_view:=camera

## Validation Checklist
- RViz fixed frame matches the point cloud frame_id (default: camera_color_optical_frame).
- Depth scale looks reasonable (set depth_scale if the model outputs relative depth).
- Two point clouds are aligned when using the same depth and correct intrinsics.
