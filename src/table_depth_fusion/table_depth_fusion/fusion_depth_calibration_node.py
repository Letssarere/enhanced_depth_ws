import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def resolve_default_config_path() -> Path:
    candidates = []
    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
        candidates.append(share_dir / "config" / "table_config.npz")
    except Exception:
        pass

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(package_root / "config" / "table_config.npz")

    for candidate in candidates:
        if candidate.parent.exists() and os.access(candidate.parent, os.W_OK):
            return candidate

    return candidates[-1]


class FusionDepthCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_depth_calibration_node")
        self.bridge = CvBridge()

        self.declare_parameter("roi_corners_2d", [0.0] * 8)
        self.declare_parameter("roi_size_mm", [0.0, 0.0])
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("num_frames", 30)
        self.declare_parameter("safe_zone_margin_ratio", 0.25)
        self.declare_parameter("depth_unit_scale", 1.0)
        self.declare_parameter("output_config", str(resolve_default_config_path()))

        self.num_frames = int(self.get_parameter("num_frames").value)
        self.safe_zone_margin_ratio = float(self.get_parameter("safe_zone_margin_ratio").value)
        self.depth_unit_scale = float(self.get_parameter("depth_unit_scale").value)
        self.output_config = Path(self.get_parameter("output_config").value)

        self.roi_corners_2d = self._parse_roi_corners(
            self.get_parameter("roi_corners_2d").value
        )
        self.roi_size_mm = self._parse_roi_size(self.get_parameter("roi_size_mm").value)

        self.camera_info = None
        self.crop_roi = None
        self.intrinsics_cropped = None
        self.safe_zone_mask = None
        self.roi_polygon_mask = None
        self.sum_depth = None
        self.count_depth = None
        self.frames_collected = 0
        self.calibrated = False

        camera_info_topic = self.get_parameter("camera_info_topic").value
        depth_topic = self.get_parameter("depth_topic").value

        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)

        self.get_logger().info("Fusion depth calibration node started.")

    def _parse_roi_corners(self, value) -> np.ndarray:
        if isinstance(value, (list, tuple)) and len(value) == 4 and all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in value
        ):
            points = np.array(value, dtype=np.float32)
        elif isinstance(value, (list, tuple)) and len(value) == 8:
            points = np.array(value, dtype=np.float32).reshape(4, 2)
        else:
            raise ValueError(
                "roi_corners_2d must be a list of 4 [u,v] pairs or 8 numbers"
            )
        return points

    def _parse_roi_size(self, value) -> np.ndarray:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("roi_size_mm must be [width_mm, height_mm]")
        return np.array(value, dtype=np.float32)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_info is not None:
            return
        self.camera_info = msg
        self._prepare_roi_and_intrinsics()
        self.get_logger().info("Camera info received.")

    def _prepare_roi_and_intrinsics(self) -> None:
        if self.camera_info is None:
            return

        corners = self.roi_corners_2d
        left_margin = 10
        right_margin = 10
        top_margin = 10
        bottom_margin = 30

        x_min = int(np.floor(np.min(corners[:, 0]))) - left_margin
        x_max = int(np.ceil(np.max(corners[:, 0]))) + right_margin
        y_min = int(np.floor(np.min(corners[:, 1]))) - top_margin
        y_max = int(np.ceil(np.max(corners[:, 1]))) + bottom_margin

        width = int(self.camera_info.width)
        height = int(self.camera_info.height)
        x_min = max(0, min(x_min, width - 1))
        x_max = max(1, min(x_max, width))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(1, min(y_max, height))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("ROI bounds are invalid; check roi_corners_2d")

        crop_width = x_max - x_min
        crop_height = y_max - y_min
        self.crop_roi = np.array([x_min, y_min, crop_width, crop_height], dtype=np.int32)

        k = np.array(self.camera_info.k, dtype=np.float32).reshape(3, 3)
        k_cropped = k.copy()
        k_cropped[0, 2] -= x_min
        k_cropped[1, 2] -= y_min
        self.intrinsics_cropped = k_cropped

        roi_corners = self.roi_corners_2d.copy()
        roi_corners[:, 0] -= x_min
        roi_corners[:, 1] -= y_min
        roi_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.round(roi_corners).astype(np.int32)], 1)
        self.roi_polygon_mask = roi_mask.astype(bool)

        margin_x = int(crop_width * self.safe_zone_margin_ratio)
        margin_y = int(crop_height * self.safe_zone_margin_ratio)
        if margin_x * 2 >= crop_width:
            margin_x = max(0, crop_width // 4)
        if margin_y * 2 >= crop_height:
            margin_y = max(0, crop_height // 4)
        mask = np.zeros((crop_height, crop_width), dtype=bool)
        mask[margin_y : crop_height - margin_y, margin_x : crop_width - margin_x] = True
        self.safe_zone_mask = mask & self.roi_polygon_mask

        self.get_logger().info(
            "ROI prepared: x=%d y=%d w=%d h=%d", x_min, y_min, crop_width, crop_height
        )

    def _depth_cb(self, msg: Image) -> None:
        if self.calibrated or self.camera_info is None or self.crop_roi is None:
            return

        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32) * self.depth_unit_scale

        x_min, y_min, crop_w, crop_h = self.crop_roi
        depth_crop = depth[y_min : y_min + crop_h, x_min : x_min + crop_w]

        if self.sum_depth is None:
            self.sum_depth = np.zeros_like(depth_crop, dtype=np.float64)
            self.count_depth = np.zeros_like(depth_crop, dtype=np.int32)

        valid = depth_crop > 0.0
        self.sum_depth[valid] += depth_crop[valid]
        self.count_depth[valid] += 1
        self.frames_collected += 1

        if self.frames_collected < self.num_frames:
            if self.frames_collected % 5 == 0:
                self.get_logger().info(
                    "Collecting depth frames: %d/%d",
                    self.frames_collected,
                    self.num_frames,
                )
            return

        self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        valid_counts = self.count_depth > 0
        avg_depth = np.zeros_like(self.sum_depth, dtype=np.float32)
        avg_depth[valid_counts] = (
            self.sum_depth[valid_counts] / self.count_depth[valid_counts]
        ).astype(np.float32)

        safe_mask = self.safe_zone_mask & valid_counts
        safe_depth_values = avg_depth[safe_mask]
        if safe_depth_values.size == 0:
            self.get_logger().error("Safe zone has no valid depth values.")
            return

        z_real = float(np.median(safe_depth_values))

        image_points = self.roi_corners_2d.copy()
        image_points[:, 0] -= self.crop_roi[0]
        image_points[:, 1] -= self.crop_roi[1]

        width_mm, height_mm = self.roi_size_mm
        object_points = np.array(
            [
                [0.0, height_mm, 0.0],
                [width_mm, height_mm, 0.0],
                [width_mm, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        dist_coeffs = np.array(self.camera_info.d, dtype=np.float32)
        if dist_coeffs.size == 0:
            dist_coeffs = None

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.intrinsics_cropped,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self.get_logger().error("solvePnP failed.")
            return

        rotation, _ = cv2.Rodrigues(rvec)
        center_obj = np.array([[width_mm / 2.0, height_mm / 2.0, 0.0]], dtype=np.float32)
        center_cam = rotation @ center_obj.T + tvec
        z_pnp = float(center_cam[2, 0])

        if z_pnp <= 0.0:
            self.get_logger().error("PnP returned invalid Z depth: %.3f", z_pnp)
            return

        scale = z_real / z_pnp
        tvec *= scale

        rotation_ct = rotation.T
        translation_ct = -rotation_ct @ tvec
        t_matrix = np.eye(4, dtype=np.float32)
        t_matrix[:3, :3] = rotation_ct.astype(np.float32)
        t_matrix[:3, 3] = translation_ct.flatten().astype(np.float32)

        self.output_config.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_config,
            crop_roi=self.crop_roi,
            intrinsics_cropped=self.intrinsics_cropped,
            T_matrix=t_matrix,
            safe_zone_mask=self.safe_zone_mask.astype(np.uint8),
            roi_polygon_mask=self.roi_polygon_mask.astype(np.uint8),
        )

        self.calibrated = True
        self.get_logger().info("Calibration saved to %s", str(self.output_config))
        self.get_logger().info("Z_real=%.2f, Z_pnp=%.2f, scale=%.4f", z_real, z_pnp, scale)


def main() -> None:
    rclpy.init()
    node = FusionDepthCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
