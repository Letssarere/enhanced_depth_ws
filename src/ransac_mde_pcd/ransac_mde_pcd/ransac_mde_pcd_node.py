from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - handled at runtime
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:  # pragma: no cover - handled at runtime
    cuda = None

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


def resolve_default_engine_path() -> Path:
    candidates = []
    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
        candidates.append(
            share_dir
            / "models"
            / "depth_anything_v3_small"
            / "tensorrt"
            / "model_fp16_640x480.engine"
        )
        candidates.append(
            share_dir
            / "models"
            / "depth_anything_v3_small"
            / "tensorrt"
            / "model_fp16.engine"
        )
    except Exception:
        pass

    candidates.append(
        Path.cwd()
        / "src"
        / "table_depth_fusion"
        / "models"
        / "depth_anything_v3_small"
        / "tensorrt"
        / "model_fp16_640x480.engine"
    )
    candidates.append(
        Path.cwd()
        / "src"
        / "table_depth_fusion"
        / "models"
        / "depth_anything_v3_small"
        / "tensorrt"
        / "model_fp16.engine"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_default_roi_yaml_path() -> Path:
    candidates = []
    candidates.append(
        Path.cwd()
        / "src"
        / "table_depth_fusion"
        / "config"
        / "calibration_params.yaml"
    )
    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
        candidates.append(share_dir / "config" / "calibration_params.yaml")
    except Exception:
        pass

    package_root = Path(__file__).resolve().parents[2]
    candidates.append(
        package_root.parent
        / "table_depth_fusion"
        / "config"
        / "calibration_params.yaml"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


class RansacMdePcdNode(Node):
    def __init__(self) -> None:
        super().__init__("ransac_mde_pcd_node")
        self.bridge = CvBridge()
        self.rng = np.random.default_rng()

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("engine_path", str(resolve_default_engine_path()))
        self.declare_parameter("input_width", 0)
        self.declare_parameter("input_height", 0)
        self.declare_parameter("letterbox", True)
        self.declare_parameter("depth_scale", 1.0)
        self.declare_parameter("min_depth_m", 0.1)
        self.declare_parameter("max_depth_m", 5.0)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("frame_id", "table_link")
        self.declare_parameter("publish_rgb", True)
        self.declare_parameter("sync_slop", 0.1)
        self.declare_parameter("queue_size", 5)
        self.declare_parameter("points_topic", "/ransac_mde_pcd/points_table")
        self.declare_parameter("use_table_roi", True)
        self.declare_parameter("roi_yaml_path", str(resolve_default_roi_yaml_path()))
        self.declare_parameter("model_depth_output", "")
        self.declare_parameter("ransac_max_iterations", 150)
        self.declare_parameter("ransac_distance_thresh_m", 0.01)
        self.declare_parameter("ransac_min_inliers", 500)
        self.declare_parameter("ransac_stop_ratio", 0.9)
        self.declare_parameter("ransac_refine", True)

        self.engine_path = self._resolve_engine_path(
            str(self.get_parameter("engine_path").value)
        )
        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.letterbox = bool(self.get_parameter("letterbox").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.point_stride = max(1, int(self.get_parameter("point_stride").value))
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rgb = bool(self.get_parameter("publish_rgb").value)
        self.sync_slop = float(self.get_parameter("sync_slop").value)
        self.queue_size = int(self.get_parameter("queue_size").value)
        self.points_topic = str(self.get_parameter("points_topic").value)
        self.use_table_roi = bool(self.get_parameter("use_table_roi").value)
        self.roi_yaml_path = self._resolve_roi_yaml_path(
            str(self.get_parameter("roi_yaml_path").value)
        )
        self.model_depth_output = str(
            self.get_parameter("model_depth_output").value
        ).strip()
        self.ransac_max_iterations = int(
            self.get_parameter("ransac_max_iterations").value
        )
        self.ransac_distance_thresh_m = float(
            self.get_parameter("ransac_distance_thresh_m").value
        )
        self.ransac_min_inliers = int(
            self.get_parameter("ransac_min_inliers").value
        )
        self.ransac_stop_ratio = float(
            self.get_parameter("ransac_stop_ratio").value
        )
        self.ransac_refine = bool(self.get_parameter("ransac_refine").value)

        self.trt_engine = None
        self.trt_context = None
        self.trt_stream = None
        self.trt_input_index = None
        self.trt_depth_index = None
        self.trt_input_shape = None
        self.trt_output_shapes: Dict[int, Tuple[int, ...]] = {}
        self.trt_input_dtype = None
        self.trt_output_dtypes: Dict[int, np.dtype] = {}
        self.trt_bindings = None
        self.trt_input_host = None
        self.trt_input_device = None
        self.trt_output_hosts: Dict[int, np.ndarray] = {}
        self.trt_output_devices: Dict[int, object] = {}

        self.grid_shape = None
        self.u_grid = None
        self.v_grid = None

        self.roi_corners_2d = None
        self.roi_mask = None
        self.roi_mask_sampled = None
        self.roi_mask_shape = None
        self.roi_mask_warned = False

        if trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT or pycuda not available; cannot run TensorRT engine."
            )

        if not self._init_trt_engine():
            raise RuntimeError("Failed to initialize TensorRT engine.")

        self.points_pub = self.create_publisher(PointCloud2, self.points_topic, 10)

        color_topic = str(self.get_parameter("color_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.color_sub = Subscriber(
            self, Image, color_topic, qos_profile=qos_profile_sensor_data
        )
        self.info_sub = Subscriber(
            self, CameraInfo, camera_info_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.info_sub],
            queue_size=self.queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self._synced_cb)

        self.get_logger().info("RANSAC MDE PCD node started.")

    def _resolve_engine_path(self, raw_path: str) -> Path:
        if raw_path:
            path = Path(raw_path)
            if path.is_absolute():
                return path
            candidate = Path.cwd() / path
            if candidate.exists():
                return candidate
            try:
                share_dir = Path(get_package_share_directory("table_depth_fusion"))
                share_candidate = share_dir / path
                if share_candidate.exists():
                    return share_candidate
            except Exception:
                pass
            return candidate
        return resolve_default_engine_path()

    def _resolve_roi_yaml_path(self, raw_path: str) -> Path:
        if raw_path.strip():
            path = Path(raw_path)
            if path.is_absolute():
                return path
            return Path.cwd() / path
        return resolve_default_roi_yaml_path()

    def _init_trt_engine(self) -> bool:
        engine_path = self.engine_path
        if not engine_path.exists():
            self.get_logger().error(f"TensorRT engine not found: {engine_path}")
            return False

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            self.get_logger().error("Failed to deserialize TensorRT engine.")
            return False

        context = engine.create_execution_context()
        if context is None:
            self.get_logger().error("Failed to create TensorRT execution context.")
            return False

        input_indices = []
        output_indices = []
        for idx in range(engine.num_bindings):
            if engine.binding_is_input(idx):
                input_indices.append(idx)
            else:
                output_indices.append(idx)

        if not input_indices or not output_indices:
            self.get_logger().error("TensorRT engine bindings are invalid.")
            return False

        input_index = input_indices[0]
        input_shape = engine.get_binding_shape(input_index)
        if engine.has_implicit_batch_dimension:
            input_shape = (engine.max_batch_size,) + tuple(input_shape)

        if any(dim < 0 for dim in input_shape):
            resolved = self._resolve_trt_input_shape(input_shape)
            if resolved is None:
                self.get_logger().error(
                    "TensorRT input shape is dynamic; set input_width/input_height."
                )
                return False
            if not context.set_binding_shape(input_index, resolved):
                self.get_logger().error("Failed to set TensorRT binding shape.")
                return False
            input_shape = resolved

        output_shapes = {}
        output_dtypes = {}
        for idx in output_indices:
            shape = context.get_binding_shape(idx)
            if any(dim < 0 for dim in shape):
                self.get_logger().error("TensorRT output shape could not be resolved.")
                return False
            output_shapes[idx] = tuple(int(dim) for dim in shape)
            output_dtypes[idx] = trt.nptype(engine.get_binding_dtype(idx))

        depth_index = self._select_output_index(
            engine,
            output_indices,
            self.model_depth_output,
            ("predicted_depth", "depth"),
        )
        if depth_index is None:
            self.get_logger().error("Failed to find depth output binding.")
            return False

        self.trt_engine = engine
        self.trt_context = context
        self.trt_stream = cuda.Stream()
        self.trt_input_index = input_index
        self.trt_depth_index = depth_index
        self.trt_input_shape = tuple(int(dim) for dim in input_shape)
        self.trt_output_shapes = output_shapes
        self.trt_input_dtype = trt.nptype(engine.get_binding_dtype(input_index))
        self.trt_output_dtypes = output_dtypes
        self.trt_bindings = [None] * engine.num_bindings

        if not self._allocate_trt_buffers(self.trt_input_shape):
            return False

        self.get_logger().info(f"TensorRT engine loaded: {engine_path}")
        return True

    @staticmethod
    def _select_output_index(
        engine,
        output_indices,
        name_hint: str,
        keywords: Tuple[str, ...],
    ) -> Optional[int]:
        if name_hint:
            for idx in output_indices:
                name = engine.get_binding_name(idx)
                if name == name_hint or name_hint in name:
                    return idx
        for keyword in keywords:
            for idx in output_indices:
                name = engine.get_binding_name(idx)
                if keyword in name:
                    return idx
        return output_indices[0] if output_indices else None

    def _resolve_trt_input_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        if self.input_height <= 0 or self.input_width <= 0:
            return None

        dims = list(input_shape)
        if len(dims) == 5:
            dims[0] = dims[0] if dims[0] > 0 else 1
            dims[1] = dims[1] if dims[1] > 0 else 1
            dims[2] = dims[2] if dims[2] > 0 else 3
            dims[3] = self.input_height
            dims[4] = self.input_width
        elif len(dims) == 4:
            dims[0] = dims[0] if dims[0] > 0 else 1
            dims[1] = dims[1] if dims[1] > 0 else 3
            dims[2] = self.input_height
            dims[3] = self.input_width
        else:
            return None
        return tuple(int(dim) for dim in dims)

    def _allocate_trt_buffers(self, input_shape: Tuple[int, ...]) -> bool:
        if self.trt_engine is None or self.trt_context is None:
            return False

        if input_shape != self.trt_input_shape:
            if not self.trt_context.set_binding_shape(self.trt_input_index, input_shape):
                self.get_logger().error("Failed to update TensorRT binding shape.")
                return False
            self.trt_input_shape = tuple(int(dim) for dim in input_shape)
            self.trt_output_shapes = {
                idx: tuple(
                    int(dim) for dim in self.trt_context.get_binding_shape(idx)
                )
                for idx in self.trt_output_shapes
            }

        input_size = int(trt.volume(self.trt_input_shape))
        self.trt_input_host = cuda.pagelocked_empty(input_size, self.trt_input_dtype)
        self.trt_input_device = cuda.mem_alloc(self.trt_input_host.nbytes)
        self.trt_bindings[self.trt_input_index] = int(self.trt_input_device)

        self.trt_output_hosts = {}
        self.trt_output_devices = {}
        for idx, shape in self.trt_output_shapes.items():
            size = int(trt.volume(shape))
            dtype = self.trt_output_dtypes[idx]
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.trt_output_hosts[idx] = host
            self.trt_output_devices[idx] = device
            self.trt_bindings[idx] = int(device)
        return True

    def _synced_cb(self, color_msg: Image, info_msg: CameraInfo) -> None:
        try:
            input_tensor, meta, color_bgr = self._prepare_model_input(color_msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to prepare model input: {exc}")
            return

        depth_raw = self._infer_trt(input_tensor)
        if depth_raw is None:
            return

        depth = self._postprocess_depth(depth_raw, meta)
        if depth is None:
            return
        depth = depth.astype(np.float32) * self.depth_scale

        camera_k = self._camera_info_to_k(info_msg)
        if camera_k is None:
            self.get_logger().warning("CameraInfo missing intrinsics; skipping.")
            return

        if self.use_table_roi:
            self._ensure_roi_mask(depth.shape[0], depth.shape[1])

        if self.publish_rgb:
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None

        points, colors = self._depth_to_points(depth, camera_k, rgb)
        if points is None or points.shape[0] < 3:
            return

        plane = self._ransac_plane(points)
        if plane is None:
            return

        transform = self._compute_table_transform(plane, camera_k)
        if transform is None:
            return

        points_table = self._transform_points(points, transform)

        header = Header(stamp=color_msg.header.stamp, frame_id=self.frame_id)
        msg = self._create_cloud(header, points_table, colors)
        self.points_pub.publish(msg)

    def _prepare_model_input(
        self, color_msg: Image
    ) -> Tuple[np.ndarray, dict, np.ndarray]:
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        orig_h, orig_w = color.shape[:2]
        input_h, input_w = self._resolve_input_size(orig_h, orig_w)

        if self.letterbox:
            resized, meta = self._letterbox_image(color, input_w, input_h)
        else:
            resized = cv2.resize(color, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            meta = {
                "pad_left": 0,
                "pad_top": 0,
                "new_w": input_w,
                "new_h": input_h,
                "scale_x": input_w / float(orig_w),
                "scale_y": input_h / float(orig_h),
            }

        meta.update(
            {
                "input_w": input_w,
                "input_h": input_h,
                "orig_w": orig_w,
                "orig_h": orig_h,
            }
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        return input_tensor, meta, color

    @staticmethod
    def _letterbox_image(
        image: np.ndarray, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, dict]:
        src_h, src_w = image.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        meta = {
            "pad_left": pad_left,
            "pad_top": pad_top,
            "new_w": new_w,
            "new_h": new_h,
            "scale_x": scale,
            "scale_y": scale,
        }
        return padded, meta

    def _resolve_input_size(self, orig_h: int, orig_w: int) -> Tuple[int, int]:
        if self.input_height > 0 and self.input_width > 0:
            return self.input_height, self.input_width

        if self.trt_input_shape:
            if len(self.trt_input_shape) >= 5:
                return int(self.trt_input_shape[3]), int(self.trt_input_shape[4])
            if len(self.trt_input_shape) >= 4:
                return int(self.trt_input_shape[2]), int(self.trt_input_shape[3])

        return orig_h, orig_w

    def _match_trt_input_rank(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self.trt_input_shape:
            return input_tensor

        expected_rank = len(self.trt_input_shape)
        if expected_rank == input_tensor.ndim:
            return input_tensor
        if expected_rank == 5 and input_tensor.ndim == 4:
            return input_tensor[:, None, ...]
        if expected_rank == 4 and input_tensor.ndim == 5:
            if input_tensor.shape[1] == 1:
                return input_tensor[:, 0, ...]
        return input_tensor

    def _infer_trt(self, input_tensor: np.ndarray) -> Optional[np.ndarray]:
        if self.trt_context is None:
            return None

        input_tensor = self._match_trt_input_rank(input_tensor)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=self.trt_input_dtype)

        if self.trt_input_host is None or input_tensor.size != self.trt_input_host.size:
            if not self._allocate_trt_buffers(input_tensor.shape):
                return None

        np.copyto(self.trt_input_host, input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.trt_input_device, self.trt_input_host, self.trt_stream
        )
        self.trt_context.execute_async_v2(
            bindings=self.trt_bindings, stream_handle=self.trt_stream.handle
        )
        for idx, device in self.trt_output_devices.items():
            host = self.trt_output_hosts[idx]
            cuda.memcpy_dtoh_async(host, device, self.trt_stream)
        self.trt_stream.synchronize()

        if self.trt_depth_index is None:
            return None
        return self._reshape_output(self.trt_depth_index)

    def _reshape_output(self, output_index: int) -> Optional[np.ndarray]:
        host = self.trt_output_hosts.get(output_index)
        shape = self.trt_output_shapes.get(output_index)
        dtype = self.trt_output_dtypes.get(output_index)
        if host is None or shape is None or dtype is None:
            return None
        return np.array(host, dtype=dtype).reshape(shape)

    @staticmethod
    def _squeeze_depth_output(output: np.ndarray) -> Optional[np.ndarray]:
        if output.ndim == 5:
            if output.shape[2] == 1:
                return output[0, 0, 0]
            return output[0, 0]
        if output.ndim == 4:
            if output.shape[1] == 1:
                return output[0, 0]
            return output[0]
        if output.ndim == 3:
            return output[0]
        if output.ndim == 2:
            return output
        return None

    def _postprocess_depth(self, depth: np.ndarray, meta: dict) -> Optional[np.ndarray]:
        depth = self._squeeze_depth_output(depth)
        if depth is None:
            self.get_logger().warning("Unexpected depth output shape.")
            return None

        depth = depth.astype(np.float32)
        input_w = meta["input_w"]
        input_h = meta["input_h"]
        if depth.shape != (input_h, input_w):
            depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

        pad_left = meta.get("pad_left", 0)
        pad_top = meta.get("pad_top", 0)
        new_w = meta.get("new_w", depth.shape[1])
        new_h = meta.get("new_h", depth.shape[0])
        if pad_left or pad_top or new_w != depth.shape[1] or new_h != depth.shape[0]:
            depth = depth[pad_top : pad_top + new_h, pad_left : pad_left + new_w]

        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        return depth

    @staticmethod
    def _camera_info_to_k(info_msg: CameraInfo) -> Optional[np.ndarray]:
        k_vals = np.asarray(info_msg.k, dtype=np.float32)
        if k_vals.size == 9:
            return k_vals.reshape(3, 3)
        p_vals = np.asarray(info_msg.p, dtype=np.float32)
        if p_vals.size >= 9:
            return p_vals[:9].reshape(3, 3)
        return None

    def _get_uv_grid(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        shape = (height, width, self.point_stride)
        if self.grid_shape != shape:
            u = np.arange(0, width, self.point_stride, dtype=np.float32)
            v = np.arange(0, height, self.point_stride, dtype=np.float32)
            self.u_grid, self.v_grid = np.meshgrid(u, v)
            self.grid_shape = shape
        return self.u_grid, self.v_grid

    def _depth_to_points(
        self, depth: np.ndarray, k_mat: np.ndarray, rgb: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        height, width = depth.shape
        sampled = depth[:: self.point_stride, :: self.point_stride]

        valid = np.isfinite(sampled)
        if self.min_depth_m > 0.0:
            valid &= sampled > self.min_depth_m
        if self.max_depth_m > 0.0:
            valid &= sampled < self.max_depth_m

        if self.use_table_roi:
            roi_mask = self._get_roi_mask_sampled(height, width)
            if roi_mask is not None:
                if roi_mask.shape != sampled.shape:
                    self._ensure_roi_mask(height, width)
                    roi_mask = self._get_roi_mask_sampled(height, width)
                if roi_mask is not None and roi_mask.shape == sampled.shape:
                    valid &= roi_mask
            elif not self.roi_mask_warned:
                self.get_logger().warning(
                    "ROI mask unavailable; publishing full-frame point cloud."
                )
                self.roi_mask_warned = True

        if not np.any(valid):
            return None, None

        fx = float(k_mat[0, 0])
        fy = float(k_mat[1, 1])
        cx = float(k_mat[0, 2])
        cy = float(k_mat[1, 2])
        if fx == 0.0 or fy == 0.0:
            return None, None

        u_grid, v_grid = self._get_uv_grid(width, height)
        z = sampled
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        mask = valid.reshape(-1)
        points = points[mask]

        colors = None
        if rgb is not None:
            sampled_rgb = rgb[:: self.point_stride, :: self.point_stride]
            colors = sampled_rgb.reshape(-1, 3)[mask]
        return points.astype(np.float32), colors

    def _ensure_roi_mask(self, height: int, width: int) -> None:
        if not self.use_table_roi:
            return
        if self.roi_mask_shape == (height, width):
            return
        if self.roi_corners_2d is None:
            self.roi_corners_2d = self._load_roi_corners()
        if self.roi_corners_2d is None:
            return
        mask = np.zeros((height, width), dtype=np.uint8)
        corners = np.round(self.roi_corners_2d).astype(np.int32)
        cv2.fillPoly(mask, [corners], 1)
        self.roi_mask = mask.astype(bool)
        self.roi_mask_sampled = self.roi_mask[
            :: self.point_stride, :: self.point_stride
        ]
        self.roi_mask_shape = (height, width)

    def _get_roi_mask_sampled(
        self, height: int, width: int
    ) -> Optional[np.ndarray]:
        if not self.use_table_roi:
            return None
        if self.roi_mask_shape != (height, width) or self.roi_mask_sampled is None:
            self._ensure_roi_mask(height, width)
        return self.roi_mask_sampled

    def _load_roi_corners(self) -> Optional[np.ndarray]:
        if yaml is None:
            if not self.roi_mask_warned:
                self.get_logger().warning(
                    "PyYAML not available; cannot load roi_corners_2d."
                )
                self.roi_mask_warned = True
            return None
        if self.roi_yaml_path is None or not self.roi_yaml_path.exists():
            if not self.roi_mask_warned:
                self.get_logger().warning(
                    f"ROI YAML not found: {self.roi_yaml_path}"
                )
                self.roi_mask_warned = True
            return None
        try:
            with self.roi_yaml_path.open("r") as yaml_file:
                data = yaml.safe_load(yaml_file)
        except Exception as exc:
            if not self.roi_mask_warned:
                self.get_logger().warning(f"Failed to read ROI YAML: {exc}")
                self.roi_mask_warned = True
            return None

        corners = self._extract_roi_corners_from_yaml(data)
        if corners is None and not self.roi_mask_warned:
            self.get_logger().warning(
                f"roi_corners_2d not found in {self.roi_yaml_path}"
            )
            self.roi_mask_warned = True
        return corners

    def _extract_roi_corners_from_yaml(self, data) -> Optional[np.ndarray]:
        if not isinstance(data, dict):
            return None
        node_section = data.get("fusion_depth_calibration_node")
        if isinstance(node_section, dict):
            params = node_section.get("ros__parameters")
            if isinstance(params, dict) and "roi_corners_2d" in params:
                return self._parse_roi_corners(params["roi_corners_2d"])
        for _, node_data in data.items():
            if isinstance(node_data, dict):
                params = node_data.get("ros__parameters")
                if isinstance(params, dict) and "roi_corners_2d" in params:
                    return self._parse_roi_corners(params["roi_corners_2d"])
        if "roi_corners_2d" in data:
            return self._parse_roi_corners(data["roi_corners_2d"])
        return None

    @staticmethod
    def _parse_roi_corners(value) -> Optional[np.ndarray]:
        try:
            arr = np.array(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size != 8:
            return None
        return arr.reshape(4, 2)

    def _ransac_plane(
        self, points: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        if points.shape[0] < 3:
            return None

        num_points = points.shape[0]
        best_inliers = None
        best_normal = None
        best_d = None
        best_count = 0

        for _ in range(max(1, self.ransac_max_iterations)):
            sample_idx = self.rng.choice(num_points, 3, replace=False)
            p1, p2, p3 = points[sample_idx]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, p1))
            distances = np.abs(points @ normal + d)
            inliers = distances < self.ransac_distance_thresh_m
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_normal = normal
                best_d = d
                if self.ransac_stop_ratio > 0.0:
                    if count >= int(self.ransac_stop_ratio * num_points):
                        break

        if best_inliers is None or best_count < self.ransac_min_inliers:
            return None

        normal = best_normal
        d = best_d
        inliers = best_inliers

        if self.ransac_refine:
            refined = self._fit_plane_svd(points[inliers])
            if refined is not None:
                refined_normal, refined_d = refined
                if np.dot(refined_normal, normal) < 0.0:
                    refined_normal = -refined_normal
                    refined_d = -refined_d
                normal = refined_normal
                d = refined_d

        if normal[2] < 0.0:
            normal = -normal
            d = -d

        return normal, d, inliers

    @staticmethod
    def _fit_plane_svd(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        if points.shape[0] < 3:
            return None
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, v_t = np.linalg.svd(centered, full_matrices=False)
        normal = v_t[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        normal = normal / norm
        d = -float(np.dot(normal, centroid))
        return normal, d

    def _compute_table_transform(
        self, plane: Tuple[np.ndarray, float, np.ndarray], k_mat: np.ndarray
    ) -> Optional[np.ndarray]:
        if self.roi_corners_2d is None:
            self.roi_corners_2d = self._load_roi_corners()
        if self.roi_corners_2d is None:
            return None

        normal, d, _ = plane
        fx = float(k_mat[0, 0])
        fy = float(k_mat[1, 1])
        cx = float(k_mat[0, 2])
        cy = float(k_mat[1, 2])
        if fx == 0.0 or fy == 0.0:
            return None

        corners_cam = []
        for corner in self.roi_corners_2d:
            u, v = float(corner[0]), float(corner[1])
            ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
            denom = float(np.dot(normal, ray))
            if abs(denom) < 1e-6:
                return None
            t = -d / denom
            if t <= 0.0:
                return None
            corners_cam.append(ray * t)
        corners_cam = np.array(corners_cam, dtype=np.float32)

        tl, tr, br, bl = corners_cam
        origin = bl
        x_dir = br - bl
        y_dir = tl - bl

        x_dir = x_dir - normal * float(np.dot(normal, x_dir))
        y_dir = y_dir - normal * float(np.dot(normal, y_dir))
        x_axis = self._normalize_vector(x_dir)
        y_axis = self._normalize_vector(y_dir)
        if x_axis is None or y_axis is None:
            return None

        z_axis = np.cross(x_axis, y_axis)
        z_axis = self._normalize_vector(z_axis)
        if z_axis is None:
            return None

        if float(np.dot(z_axis, normal)) < 0.0:
            z_axis = -z_axis

        y_axis = np.cross(z_axis, x_axis)
        y_axis = self._normalize_vector(y_axis)
        if y_axis is None:
            return None

        r_tc = np.stack([x_axis, y_axis, z_axis], axis=1)
        r_ct = r_tc.T
        t_ct = -r_ct @ origin.reshape(3, 1)

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = r_ct.astype(np.float32)
        transform[:3, 3] = t_ct.flatten().astype(np.float32)
        return transform

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return vector / norm

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points.astype(np.float32), ones], axis=1)
        return (transform @ points_h.T).T[:, :3]

    def _create_cloud(
        self, header: Header, points: np.ndarray, colors: Optional[np.ndarray]
    ) -> PointCloud2:
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = int(points.shape[0])
        msg.is_bigendian = False
        msg.is_dense = False

        if colors is None:
            msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 12
            msg.row_step = msg.point_step * msg.width
            msg.data = points.astype(np.float32).tobytes()
            return msg

        colors = colors.astype(np.uint32)
        rgb = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        rgb_packed = rgb.view(np.float32)
        cloud = np.empty((points.shape[0], 4), dtype=np.float32)
        cloud[:, 0:3] = points.astype(np.float32)
        cloud[:, 3] = rgb_packed

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = cloud.tobytes()
        return msg


def main() -> None:
    rclpy.init()
    node = RansacMdePcdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
