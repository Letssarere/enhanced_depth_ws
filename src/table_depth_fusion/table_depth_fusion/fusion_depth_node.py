from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - handled at runtime
    ort = None


def resolve_default_config_path() -> Path:
    candidates = []
    workspace_candidate = (
        Path.cwd() / "src" / "table_depth_fusion" / "config" / "table_config.npz"
    )
    candidates.append(workspace_candidate)
    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
        candidates.append(share_dir / "config" / "table_config.npz")
    except Exception:
        pass

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(package_root / "config" / "table_config.npz")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_default_model_path() -> Path:
    candidates = []
    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
        candidates.append(
            share_dir
            / "models"
            / "depth_anything_v3_small_onnx"
            / "onnx"
            / "model.onnx"
        )
    except Exception:
        pass

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(
        package_root
        / "models"
        / "depth_anything_v3_small_onnx"
        / "onnx"
        / "model.onnx"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


class FusionDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_depth_node")
        self.bridge = CvBridge()

        self.declare_parameter("config_path", str(resolve_default_config_path()))
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("depth_unit_scale", 1.0)
        self.declare_parameter("frame_id", "table_link")
        self.declare_parameter("point_stride", 1)
        self.declare_parameter("safe_zone_min_points", 50)
        self.declare_parameter("use_mde", False)
        self.declare_parameter("mde_model_path", "")
        self.declare_parameter("mde_input_width", 0)
        self.declare_parameter("mde_input_height", 0)

        self.config_path = Path(self.get_parameter("config_path").value)
        self.depth_unit_scale = float(self.get_parameter("depth_unit_scale").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.point_stride = max(1, int(self.get_parameter("point_stride").value))
        self.safe_zone_min_points = int(self.get_parameter("safe_zone_min_points").value)
        self.use_mde = bool(self.get_parameter("use_mde").value)
        self.mde_warned = False
        self.mde_input_width = int(self.get_parameter("mde_input_width").value)
        self.mde_input_height = int(self.get_parameter("mde_input_height").value)
        self.mde_model_path = self._resolve_model_path(
            self.get_parameter("mde_model_path").value
        )
        self.ort_session = None
        self.ort_input_name = None
        self.ort_output_name = None
        self.ort_input_shape = None

        self._load_config()
        self._prepare_grids()

        self.fused_pub = self.create_publisher(Image, "/table/fused_depth_image", 10)
        self.points_pub = self.create_publisher(PointCloud2, "/table/roi_points", 10)

        depth_topic = self.get_parameter("depth_topic").value
        if self.use_mde:
            if ort is None:
                self.get_logger().error(
                    "onnxruntime is not available; disabling MDE inference."
                )
                self.use_mde = False
            else:
                if not self._init_mde_session():
                    self.use_mde = False

        if self.use_mde:
            color_topic = self.get_parameter("color_topic").value
            self.color_sub = Subscriber(self, Image, color_topic)
            self.depth_sub = Subscriber(self, Image, depth_topic)
            self.sync = ApproximateTimeSynchronizer(
                [self.color_sub, self.depth_sub], queue_size=5, slop=0.1
            )
            self.sync.registerCallback(self._synced_cb)
        else:
            self.create_subscription(Image, depth_topic, self._depth_only_cb, 10)

        self.get_logger().info("Fusion depth runtime node started.")

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        data = np.load(self.config_path)
        self.crop_roi = data["crop_roi"].astype(np.int32)
        self.intrinsics = data["intrinsics_cropped"].astype(np.float32)
        self.t_matrix = data["T_matrix"].astype(np.float32)
        self.safe_zone_mask = data["safe_zone_mask"].astype(bool)
        if "roi_polygon_mask" in data.files:
            self.roi_polygon_mask = data["roi_polygon_mask"].astype(bool)
        else:
            width = int(self.crop_roi[2])
            height = int(self.crop_roi[3])
            self.roi_polygon_mask = np.ones((height, width), dtype=bool)
            self.get_logger().warning(
                "roi_polygon_mask missing in config; using full crop mask."
            )

        self.fx = float(self.intrinsics[0, 0])
        self.fy = float(self.intrinsics[1, 1])
        self.cx = float(self.intrinsics[0, 2])
        self.cy = float(self.intrinsics[1, 2])

    def _resolve_model_path(self, raw_path: str) -> Path:
        if raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                try:
                    share_dir = Path(get_package_share_directory("table_depth_fusion"))
                    share_candidate = share_dir / path
                    if share_candidate.exists():
                        return share_candidate
                except Exception:
                    pass

                package_root = Path(__file__).resolve().parents[1]
                root_candidate = package_root / path
                if root_candidate.exists():
                    return root_candidate
            return path
        return resolve_default_model_path()

    def _init_mde_session(self) -> bool:
        model_path = self.mde_model_path
        if not model_path.exists():
            self.get_logger().error(f"MDE model not found: {model_path}")
            return False

        self.ort_session = ort.InferenceSession(
            str(model_path), providers=ort.get_available_providers()
        )
        inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.get_outputs()
        if not inputs or not outputs:
            self.get_logger().error("MDE model has invalid inputs/outputs.")
            return False

        self.ort_input_name = inputs[0].name
        self.ort_input_shape = inputs[0].shape
        self.ort_output_name = outputs[0].name
        self.get_logger().info(
            f"MDE model loaded: {model_path} (input shape: {self.ort_input_shape})"
        )
        return True

    def _prepare_grids(self) -> None:
        _, _, width, height = self.crop_roi
        u = np.arange(0, width, self.point_stride, dtype=np.float32)
        v = np.arange(0, height, self.point_stride, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)
        self.x_norm = (u_grid - self.cx) / self.fx
        self.y_norm = (v_grid - self.cy) / self.fy
        if self.roi_polygon_mask.shape != (height, width):
            self.get_logger().warning(
                "roi_polygon_mask shape mismatch; using full crop mask."
            )
            self.roi_polygon_mask = np.ones((height, width), dtype=bool)
        self.roi_mask_sampled = self.roi_polygon_mask[
            :: self.point_stride, :: self.point_stride
        ]

    def _synced_cb(self, color_msg: Image, depth_msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32) * self.depth_unit_scale

        x_min, y_min, crop_w, crop_h = self.crop_roi
        depth_crop = depth[y_min : y_min + crop_h, x_min : x_min + crop_w]

        fused_depth = depth_crop
        if self.use_mde:
            mde_depth = self._infer_mde(color_msg, x_min, y_min, crop_w, crop_h)
            if mde_depth is not None:
                fused_depth = self._scale_recover(depth_crop, mde_depth)

        self._publish_fused_depth(fused_depth, depth_msg)
        self._publish_point_cloud(fused_depth, depth_msg.header.stamp)

    def _depth_only_cb(self, depth_msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32) * self.depth_unit_scale

        x_min, y_min, crop_w, crop_h = self.crop_roi
        depth_crop = depth[y_min : y_min + crop_h, x_min : x_min + crop_w]

        self._publish_fused_depth(depth_crop, depth_msg)
        self._publish_point_cloud(depth_crop, depth_msg.header.stamp)

    def _infer_mde(
        self, color_msg: Image, x_min: int, y_min: int, crop_w: int, crop_h: int
    ) -> Optional[np.ndarray]:
        if self.ort_session is None:
            return None

        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        crop = color[y_min : y_min + crop_h, x_min : x_min + crop_w]

        input_h, input_w = self._resolve_mde_input_size(crop_h, crop_w)
        if (input_h, input_w) != (crop_h, crop_w):
            resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = crop

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

        output = self.ort_session.run(
            [self.ort_output_name], {self.ort_input_name: input_tensor}
        )[0]
        depth = self._squeeze_depth_output(output)
        if depth is None:
            if not self.mde_warned:
                self.get_logger().warning(
                    "Unexpected MDE output shape; skipping MDE fusion."
                )
                self.mde_warned = True
            return None

        depth = depth.astype(np.float32)
        if depth.shape != (crop_h, crop_w):
            depth = cv2.resize(depth, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        return depth

    def _resolve_mde_input_size(self, crop_h: int, crop_w: int) -> Tuple[int, int]:
        if self.mde_input_height > 0 and self.mde_input_width > 0:
            return self.mde_input_height, self.mde_input_width

        if self.ort_input_shape and len(self.ort_input_shape) >= 4:
            h = self.ort_input_shape[2]
            w = self.ort_input_shape[3]
            if isinstance(h, int) and isinstance(w, int):
                return h, w

        return crop_h, crop_w

    @staticmethod
    def _squeeze_depth_output(output: np.ndarray) -> Optional[np.ndarray]:
        if output.ndim == 4:
            if output.shape[1] == 1:
                return output[0, 0]
            return output[0]
        if output.ndim == 3:
            return output[0]
        if output.ndim == 2:
            return output
        return None

    def _scale_recover(self, depth_metric: np.ndarray, depth_relative: np.ndarray) -> np.ndarray:
        if depth_metric.shape != depth_relative.shape:
            self.get_logger().warning("MDE output shape mismatch; using metric depth.")
            return depth_metric

        safe_mask = self.safe_zone_mask
        if safe_mask.shape != depth_metric.shape:
            self.get_logger().warning("Safe zone mask mismatch; using metric depth.")
            return depth_metric

        valid = (
            safe_mask
            & np.isfinite(depth_metric)
            & np.isfinite(depth_relative)
            & (depth_metric > 0.0)
        )
        sample_count = int(np.sum(valid))
        mde_vals = depth_relative[valid]
        real_vals = depth_metric[valid]
        if sample_count < self.safe_zone_min_points:
            self.get_logger().warning("Insufficient samples for scale recovery.")
            if sample_count == 0:
                return depth_metric
            median_mde = float(np.median(mde_vals))
            if abs(median_mde) < 1e-6:
                return depth_metric
            scale = float(np.median(real_vals)) / median_mde
            return depth_relative * scale
        design = np.column_stack([mde_vals, np.ones_like(mde_vals)])
        solution, _, _, _ = np.linalg.lstsq(design, real_vals, rcond=None)
        scale, bias = solution
        return depth_relative * scale + bias

    def _publish_fused_depth(self, fused_depth: np.ndarray, depth_msg: Image) -> None:
        fused_msg = self.bridge.cv2_to_imgmsg(
            fused_depth.astype(np.float32), encoding="32FC1"
        )
        fused_msg.header = depth_msg.header
        self.fused_pub.publish(fused_msg)

    def _publish_point_cloud(self, fused_depth: np.ndarray, stamp) -> None:
        sampled = fused_depth[:: self.point_stride, :: self.point_stride]
        valid = np.isfinite(sampled) & (sampled > 0.0)
        valid &= self.roi_mask_sampled
        if not np.any(valid):
            return

        x = self.x_norm * sampled
        y = self.y_norm * sampled
        z = sampled
        points = np.stack([x, y, z], axis=-1)[valid]

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points.astype(np.float32), ones], axis=1)
        points_table = (self.t_matrix @ points_h.T).T[:, :3]

        header = self._create_header(stamp)
        cloud_msg = point_cloud2.create_cloud_xyz32(
            header, points_table.astype(np.float32)
        )
        self.points_pub.publish(cloud_msg)

    def _create_header(self, stamp):
        return Header(stamp=stamp, frame_id=self.frame_id)


def main() -> None:
    rclpy.init()
    node = FusionDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
