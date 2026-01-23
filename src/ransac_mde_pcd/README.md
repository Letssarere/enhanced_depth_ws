# ransac_mde_pcd

Generate a table-aligned point cloud by running per-frame RANSAC on MDE depth.

## Launch
```bash
ros2 launch ransac_mde_pcd ransac_mde_pcd.launch.py
```

## Notes
- Uses TensorRT engine from `table_depth_fusion/models` by default.
- ROI is loaded from `table_depth_fusion/config/calibration_params.yaml` (`roi_corners_2d`).
- Output topic default: `/ransac_mde_pcd/points_table` (frame: `table_link`).
