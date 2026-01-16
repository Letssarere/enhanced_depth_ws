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


def resolve_default_engine_path() -> Path:
    candidates = []
    try:
        share_dir = Path(get_package_share_directory("da3_pointcloud_compare"))
        candidates.append(
            share_dir / "models" / "da3_small" / "tensorrt" / "model_fp16.engine"
        )
    except Exception:
        pass

    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
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
        / "da3_pointcloud_compare"
        / "models"
        / "da3_small"
        / "tensorrt"
        / "model_fp16.engine"
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

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(
        package_root / "models" / "da3_small" / "tensorrt" / "model_fp16.engine"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


class DA3PointCloudCompareNode(Node):
    def __init__(self) -> None:
        super().__init__("da3_pointcloud_compare_node")
        self.bridge = CvBridge()

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
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("publish_rgb", True)
        self.declare_parameter("sync_slop", 0.1)
        self.declare_parameter("queue_size", 5)
        self.declare_parameter("model_depth_output", "")
        self.declare_parameter("model_intrinsics_output", "")
        self.declare_parameter("model_intrinsics_format", "pixels")
        self.declare_parameter("model_intrinsics_warn_once", True)
        self.declare_parameter("points_model_topic", "/da3/points_model_k")
        self.declare_parameter("points_camera_topic", "/da3/points_camera_k")

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
        self.model_depth_output = str(
            self.get_parameter("model_depth_output").value
        ).strip()
        self.model_intrinsics_output = str(
            self.get_parameter("model_intrinsics_output").value
        ).strip()
        self.model_intrinsics_format = str(
            self.get_parameter("model_intrinsics_format").value
        ).strip().lower()
        self.model_intrinsics_warn_once = bool(
            self.get_parameter("model_intrinsics_warn_once").value
        )

        self.model_topic = str(self.get_parameter("points_model_topic").value)
        self.camera_topic = str(self.get_parameter("points_camera_topic").value)

        self.trt_engine = None
        self.trt_context = None
        self.trt_stream = None
        self.trt_input_index = None
        self.trt_depth_index = None
        self.trt_intrinsics_index = None
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

        self.intrinsics_missing_warned = False
        self.intrinsics_format_warned = False

        if trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT or pycuda not available; cannot run DA3 TensorRT engine."
            )

        if not self._init_trt_engine():
            raise RuntimeError("Failed to initialize TensorRT engine.")

        self.model_pub = self.create_publisher(PointCloud2, self.model_topic, 10)
        self.camera_pub = self.create_publisher(PointCloud2, self.camera_topic, 10)

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

        self.get_logger().info("DA3 point cloud compare node started.")

    def _resolve_engine_path(self, raw_path: str) -> Path:
        if raw_path:
            path = Path(raw_path)
            if path.is_absolute():
                return path
            return Path.cwd() / path
        return resolve_default_engine_path()

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

        if len(input_indices) > 1:
            self.get_logger().warning(
                "TensorRT engine has multiple inputs; using the first input."
            )

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
            required=False,
        )
        if depth_index is None:
            self.get_logger().error("Failed to find depth output binding.")
            return False

        intrinsics_index = self._select_output_index(
            engine,
            output_indices,
            self.model_intrinsics_output,
            ("intrinsics", "intrinsic"),
            required=False,
        )
        if intrinsics_index is None:
            self.get_logger().warning(
                "Model intrinsics output not found; set model_intrinsics_output."
            )

        self.trt_engine = engine
        self.trt_context = context
        self.trt_stream = cuda.Stream()
        self.trt_input_index = input_index
        self.trt_depth_index = depth_index
        self.trt_intrinsics_index = intrinsics_index
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
        required: bool,
    ) -> Optional[int]:
        if name_hint:
            for idx in output_indices:
                name = engine.get_binding_name(idx)
                if name == name_hint or name_hint in name:
                    return idx
            if required:
                return None
        for keyword in keywords:
            for idx in output_indices:
                name = engine.get_binding_name(idx)
                if keyword in name:
                    return idx
        if required:
            return None
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

        depth_raw, intrinsics_raw = self._infer_trt(input_tensor)
        if depth_raw is None:
            return

        depth = self._postprocess_depth(depth_raw, meta)
        if depth is None:
            return
        depth = depth.astype(np.float32) * self.depth_scale

        model_k = self._parse_model_intrinsics(intrinsics_raw, meta)
        if model_k is None and not self.intrinsics_missing_warned:
            if self.model_intrinsics_warn_once:
                self.get_logger().warning(
                    "Model intrinsics unavailable; set model_intrinsics_output or check engine."
                )
                self.intrinsics_missing_warned = True

        camera_k = self._camera_info_to_k(info_msg)
        if camera_k is None:
            self.get_logger().warning("CameraInfo missing intrinsics; skipping.")
            return

        if self.publish_rgb:
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None

        frame_id = self.frame_id or color_msg.header.frame_id
        header = Header(stamp=color_msg.header.stamp, frame_id=frame_id)

        if model_k is not None:
            points, colors = self._depth_to_points(depth, model_k, rgb)
            if points is not None:
                msg = self._create_cloud(header, points, colors)
                self.model_pub.publish(msg)

        points, colors = self._depth_to_points(depth, camera_k, rgb)
        if points is not None:
            msg = self._create_cloud(header, points, colors)
            self.camera_pub.publish(msg)

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

    def _infer_trt(
        self, input_tensor: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.trt_context is None:
            return None, None

        input_tensor = self._match_trt_input_rank(input_tensor)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=self.trt_input_dtype)

        if self.trt_input_host is None or input_tensor.size != self.trt_input_host.size:
            if not self._allocate_trt_buffers(input_tensor.shape):
                return None, None

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

        depth = None
        intrinsics = None
        if self.trt_depth_index is not None:
            depth = self._reshape_output(self.trt_depth_index)
        if self.trt_intrinsics_index is not None:
            intrinsics = self._reshape_output(self.trt_intrinsics_index)
        return depth, intrinsics

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

    def _parse_model_intrinsics(
        self, intrinsics: Optional[np.ndarray], meta: dict
    ) -> Optional[np.ndarray]:
        if intrinsics is None:
            return None

        raw = np.array(intrinsics, dtype=np.float32).squeeze()
        if raw.size == 9:
            k_mat = raw.reshape(3, 3)
        elif raw.size >= 4:
            fx, fy, cx, cy = raw.reshape(-1)[:4]
            k_mat = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        else:
            return None

        if self.model_intrinsics_format == "normalized":
            input_w = float(meta["input_w"])
            input_h = float(meta["input_h"])
            k_mat[0, 0] *= input_w
            k_mat[1, 1] *= input_h
            k_mat[0, 2] *= input_w
            k_mat[1, 2] *= input_h
        elif self.model_intrinsics_format != "pixels":
            if not self.intrinsics_format_warned:
                self.get_logger().warning(
                    f"Unknown model_intrinsics_format '{self.model_intrinsics_format}', using pixels."
                )
                self.intrinsics_format_warned = True

        scale_x = float(meta.get("scale_x", 1.0))
        scale_y = float(meta.get("scale_y", 1.0))
        pad_left = float(meta.get("pad_left", 0.0))
        pad_top = float(meta.get("pad_top", 0.0))

        if scale_x <= 0.0 or scale_y <= 0.0:
            return k_mat

        k_adj = k_mat.copy()
        k_adj[0, 0] /= scale_x
        k_adj[1, 1] /= scale_y
        k_adj[0, 2] = (k_adj[0, 2] - pad_left) / scale_x
        k_adj[1, 2] = (k_adj[1, 2] - pad_top) / scale_y
        return k_adj

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
    node = DA3PointCloudCompareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
