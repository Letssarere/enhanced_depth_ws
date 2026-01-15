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

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - handled at runtime
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:  # pragma: no cover - handled at runtime
    cuda = None


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
            / "depth_anything_v3_small"
            / "onnx"
            / "model.onnx"
        )
    except Exception:
        pass

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(
        package_root
        / "models"
        / "depth_anything_v3_small"
        / "onnx"
        / "model.onnx"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_default_engine_path() -> Path:
    candidates = []
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

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(
        package_root
        / "models"
        / "depth_anything_v3_small"
        / "tensorrt"
        / "model_fp16.engine"
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
        self.declare_parameter("mde_backend", "onnxruntime")
        self.declare_parameter("mde_model_path", "")
        self.declare_parameter("mde_engine_path", "")
        self.declare_parameter("mde_input_width", 0)
        self.declare_parameter("mde_input_height", 0)
        self.declare_parameter("mde_letterbox", True)
        self.declare_parameter("mde_full_image", True)
        self.declare_parameter("publish_full_fused_depth", True)

        self.config_path = Path(self.get_parameter("config_path").value)
        self.depth_unit_scale = float(self.get_parameter("depth_unit_scale").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.point_stride = max(1, int(self.get_parameter("point_stride").value))
        self.safe_zone_min_points = int(self.get_parameter("safe_zone_min_points").value)
        self.use_mde = bool(self.get_parameter("use_mde").value)
        self.mde_backend = str(self.get_parameter("mde_backend").value).lower()
        self.mde_warned = False
        self.mde_input_width = int(self.get_parameter("mde_input_width").value)
        self.mde_input_height = int(self.get_parameter("mde_input_height").value)
        self.mde_letterbox = bool(self.get_parameter("mde_letterbox").value)
        self.mde_full_image = bool(self.get_parameter("mde_full_image").value)
        self.publish_full_fused_depth = bool(
            self.get_parameter("publish_full_fused_depth").value
        )
        self.mde_model_path = self._resolve_model_path(
            self.get_parameter("mde_model_path").value
        )
        self.mde_engine_path = self._resolve_engine_path(
            self.get_parameter("mde_engine_path").value
        )
        self.ort_session = None
        self.ort_input_name = None
        self.ort_output_name = None
        self.ort_input_shape = None
        self.trt_engine = None
        self.trt_context = None
        self.trt_input_index = None
        self.trt_output_index = None
        self.trt_output_indices = []
        self.trt_input_shape = None
        self.trt_output_shape = None
        self.trt_output_shapes = {}
        self.trt_input_dtype = None
        self.trt_output_dtype = None
        self.trt_output_dtypes = {}
        self.trt_input_device = None
        self.trt_output_devices = {}
        self.trt_input_host = None
        self.trt_output_host = None
        self.trt_bindings = None
        self.trt_stream = None

        self._load_config()
        self._prepare_grids()

        self.fused_pub = self.create_publisher(Image, "/table/fused_depth_image", 10)
        self.points_pub = self.create_publisher(PointCloud2, "/table/roi_points", 10)

        depth_topic = self.get_parameter("depth_topic").value
        if self.mde_backend not in ("onnxruntime", "tensorrt"):
            self.get_logger().warning(
                f"Unknown mde_backend '{self.mde_backend}', using onnxruntime."
            )
            self.mde_backend = "onnxruntime"
        if self.use_mde:
            if self.mde_backend == "tensorrt":
                if not self._init_trt_engine():
                    if ort is None:
                        self.get_logger().error(
                            "TensorRT init failed and onnxruntime unavailable; "
                            "disabling MDE inference."
                        )
                        self.use_mde = False
                    else:
                        if self._init_mde_session():
                            self.mde_backend = "onnxruntime"
                            self.get_logger().warning(
                                "Falling back to onnxruntime for MDE."
                            )
                        else:
                            self.use_mde = False
            else:
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

    def _resolve_engine_path(self, raw_path: str) -> Path:
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
        return resolve_default_engine_path()

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
        self.ort_output_name = self._select_ort_output_name(outputs)
        self.get_logger().info(
            f"MDE model loaded: {model_path} (input shape: {self.ort_input_shape})"
        )
        return True

    @staticmethod
    def _select_ort_output_name(outputs) -> str:
        for output in outputs:
            if "predicted_depth" in output.name:
                return output.name
        return outputs[0].name

    def _init_trt_engine(self) -> bool:
        if trt is None or cuda is None:
            self.get_logger().error(
                "TensorRT or pycuda not available; cannot use TensorRT backend."
            )
            return False

        engine_path = self.mde_engine_path
        if not engine_path.exists():
            self.get_logger().error(f"TensorRT engine not found: {engine_path}")
            if self.mde_input_height > 0 and self.mde_input_width > 0:
                self.get_logger().info(
                    "Generate with: trtexec --onnx <model.onnx> "
                    f"--saveEngine {engine_path} "
                    f"--minShapes pixel_values:1x1x3x{self.mde_input_height}x"
                    f"{self.mde_input_width} "
                    f"--optShapes pixel_values:1x1x3x{self.mde_input_height}x"
                    f"{self.mde_input_width} "
                    f"--maxShapes pixel_values:1x1x3x{self.mde_input_height}x"
                    f"{self.mde_input_width} --fp16"
                )
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
        output_index = self._select_trt_output_index(engine, output_indices)

        input_shape = engine.get_binding_shape(input_index)
        if engine.has_implicit_batch_dimension:
            input_shape = (engine.max_batch_size,) + tuple(input_shape)

        if any(dim < 0 for dim in input_shape):
            resolved = self._resolve_trt_input_shape(input_shape)
            if resolved is None:
                self.get_logger().error(
                    "TensorRT input shape is dynamic; set mde_input_width/height."
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

        self.trt_engine = engine
        self.trt_context = context
        self.trt_input_index = input_index
        self.trt_output_index = output_index
        self.trt_output_indices = output_indices
        self.trt_input_shape = tuple(int(dim) for dim in input_shape)
        self.trt_output_shapes = output_shapes
        self.trt_output_shape = output_shapes[output_index]
        self.trt_input_dtype = trt.nptype(engine.get_binding_dtype(input_index))
        self.trt_output_dtype = output_dtypes[output_index]
        self.trt_output_dtypes = output_dtypes
        self.trt_bindings = [None] * engine.num_bindings
        self.trt_stream = cuda.Stream()

        if not self._allocate_trt_buffers(self.trt_input_shape):
            return False

        self.get_logger().info(f"TensorRT engine loaded: {engine_path}")
        return True

    @staticmethod
    def _select_trt_output_index(engine, output_indices) -> int:
        for idx in output_indices:
            name = engine.get_binding_name(idx)
            if "predicted_depth" in name:
                return idx
        return output_indices[0]

    def _allocate_trt_buffers(self, input_shape: Tuple[int, ...]) -> bool:
        if self.trt_engine is None:
            return False

        if self.trt_context is None:
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
                for idx in self.trt_output_indices
            }
            self.trt_output_shape = self.trt_output_shapes[self.trt_output_index]

        output_shape = self.trt_output_shape
        if output_shape is None:
            return False

        input_size = int(trt.volume(self.trt_input_shape))
        self.trt_input_host = cuda.pagelocked_empty(input_size, self.trt_input_dtype)
        self.trt_input_device = cuda.mem_alloc(self.trt_input_host.nbytes)
        self.trt_bindings[self.trt_input_index] = int(self.trt_input_device)

        self.trt_output_devices = {}
        self.trt_output_host = None
        for idx in self.trt_output_indices:
            shape = self.trt_output_shapes.get(idx)
            if shape is None:
                self.get_logger().error("TensorRT output shape missing.")
                return False
            size = int(trt.volume(shape))
            dtype = self.trt_output_dtypes.get(idx, self.trt_output_dtype)
            device = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.trt_output_devices[idx] = device
            self.trt_bindings[idx] = int(device)
            if idx == self.trt_output_index:
                self.trt_output_dtype = dtype
                self.trt_output_shape = shape
                self.trt_output_host = cuda.pagelocked_empty(size, dtype)
        return True

    def _resolve_trt_input_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        if self.mde_input_height <= 0 or self.mde_input_width <= 0:
            return None

        dims = list(input_shape)
        if len(dims) == 5:
            dims[0] = dims[0] if dims[0] > 0 else 1
            dims[1] = dims[1] if dims[1] > 0 else 1
            dims[2] = dims[2] if dims[2] > 0 else 3
            dims[3] = self.mde_input_height
            dims[4] = self.mde_input_width
        elif len(dims) == 4:
            dims[0] = dims[0] if dims[0] > 0 else 1
            dims[1] = dims[1] if dims[1] > 0 else 3
            dims[2] = self.mde_input_height
            dims[3] = self.mde_input_width
        else:
            return None
        return tuple(int(dim) for dim in dims)

    def _prepare_mde_input(
        self, color_msg: Image, x_min: int, y_min: int, crop_w: int, crop_h: int
    ) -> Tuple[np.ndarray, dict]:
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        crop = color[y_min : y_min + crop_h, x_min : x_min + crop_w]

        input_h, input_w = self._resolve_mde_input_size(crop_h, crop_w)
        if self.mde_letterbox:
            resized, meta = self._letterbox_image(crop, input_w, input_h)
        else:
            resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            meta = {
                "pad_left": 0,
                "pad_top": 0,
                "new_w": input_w,
                "new_h": input_h,
            }
        meta["input_w"] = input_w
        meta["input_h"] = input_h

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        return input_tensor, meta

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
        meta = {"pad_left": pad_left, "pad_top": pad_top, "new_w": new_w, "new_h": new_h}
        return padded, meta

    @staticmethod
    def _postprocess_mde_output(
        depth: np.ndarray, meta: dict, crop_w: int, crop_h: int
    ) -> np.ndarray:
        depth = depth.astype(np.float32)
        pad_left = meta.get("pad_left", 0)
        pad_top = meta.get("pad_top", 0)
        new_w = meta.get("new_w", depth.shape[1])
        new_h = meta.get("new_h", depth.shape[0])
        input_w = meta.get("input_w", depth.shape[1])
        input_h = meta.get("input_h", depth.shape[0])
        if depth.shape != (input_h, input_w):
            depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
        if pad_left or pad_top or new_w != depth.shape[1] or new_h != depth.shape[0]:
            depth = depth[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        if depth.shape != (crop_h, crop_w):
            depth = cv2.resize(depth, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        return depth

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

        fused_depth_full = depth
        fused_depth_roi = depth_crop
        if self.use_mde:
            if self.mde_full_image:
                full_h, full_w = depth.shape[:2]
                mde_full = self._infer_mde(color_msg, 0, 0, full_w, full_h)
                if mde_full is not None and mde_full.shape == depth.shape:
                    mde_crop = mde_full[
                        y_min : y_min + crop_h, x_min : x_min + crop_w
                    ]
                    params = self._compute_scale_recovery_params(
                        depth_crop, mde_crop
                    )
                    if params is not None:
                        scale, bias = params
                        fused_depth_full = mde_full * scale + bias
                    fused_depth_roi = fused_depth_full[
                        y_min : y_min + crop_h, x_min : x_min + crop_w
                    ]
                else:
                    if mde_full is not None:
                        self.get_logger().warning(
                            "Full MDE output shape mismatch; using metric depth."
                        )
            else:
                mde_depth = self._infer_mde(color_msg, x_min, y_min, crop_w, crop_h)
                if mde_depth is not None:
                    fused_depth_roi = self._scale_recover(depth_crop, mde_depth)

        fused_to_publish = (
            fused_depth_full if self.publish_full_fused_depth else fused_depth_roi
        )
        self._publish_fused_depth(fused_to_publish, depth_msg)
        self._publish_point_cloud(fused_depth_roi, depth_msg.header.stamp)

    def _depth_only_cb(self, depth_msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32) * self.depth_unit_scale

        x_min, y_min, crop_w, crop_h = self.crop_roi
        depth_crop = depth[y_min : y_min + crop_h, x_min : x_min + crop_w]

        fused_to_publish = depth if self.publish_full_fused_depth else depth_crop
        self._publish_fused_depth(fused_to_publish, depth_msg)
        self._publish_point_cloud(depth_crop, depth_msg.header.stamp)

    def _infer_mde(
        self, color_msg: Image, x_min: int, y_min: int, crop_w: int, crop_h: int
    ) -> Optional[np.ndarray]:
        if self.mde_backend == "tensorrt":
            return self._infer_mde_trt(color_msg, x_min, y_min, crop_w, crop_h)
        return self._infer_mde_onnx(color_msg, x_min, y_min, crop_w, crop_h)

    def _infer_mde_onnx(
        self, color_msg: Image, x_min: int, y_min: int, crop_w: int, crop_h: int
    ) -> Optional[np.ndarray]:
        if self.ort_session is None:
            return None

        input_tensor, meta = self._prepare_mde_input(
            color_msg, x_min, y_min, crop_w, crop_h
        )
        input_tensor = self._match_mde_input_rank(input_tensor)

        try:
            output = self.ort_session.run(
                [self.ort_output_name], {self.ort_input_name: input_tensor}
            )[0]
        except Exception as exc:
            if not self.mde_warned:
                self.get_logger().warning(
                    f"MDE inference failed: {exc}. Using metric depth."
                )
                self.mde_warned = True
            return None
        depth = self._squeeze_depth_output(output)
        if depth is None:
            if not self.mde_warned:
                self.get_logger().warning(
                    "Unexpected MDE output shape; skipping MDE fusion."
                )
                self.mde_warned = True
            return None

        return self._postprocess_mde_output(depth, meta, crop_w, crop_h)

    def _infer_mde_trt(
        self, color_msg: Image, x_min: int, y_min: int, crop_w: int, crop_h: int
    ) -> Optional[np.ndarray]:
        if self.trt_context is None:
            return None

        input_tensor, meta = self._prepare_mde_input(
            color_msg, x_min, y_min, crop_w, crop_h
        )
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
        output_device = self.trt_output_devices.get(self.trt_output_index)
        output_host = self.trt_output_host
        if output_device is None or output_host is None:
            self.get_logger().error("TensorRT output buffer not initialized.")
            return None
        cuda.memcpy_dtoh_async(output_host, output_device, self.trt_stream)
        self.trt_stream.synchronize()

        output = np.array(output_host, dtype=self.trt_output_dtype).reshape(
            self.trt_output_shape
        )
        depth = self._squeeze_depth_output(output)
        if depth is None:
            if not self.mde_warned:
                self.get_logger().warning(
                    "Unexpected MDE output shape from TensorRT; skipping MDE fusion."
                )
                self.mde_warned = True
            return None

        return self._postprocess_mde_output(depth, meta, crop_w, crop_h)

    def _resolve_mde_input_size(self, crop_h: int, crop_w: int) -> Tuple[int, int]:
        if self.mde_input_height > 0 and self.mde_input_width > 0:
            return self.mde_input_height, self.mde_input_width

        if self.mde_backend == "tensorrt" and self.trt_input_shape:
            if len(self.trt_input_shape) >= 5:
                return self.trt_input_shape[3], self.trt_input_shape[4]
            if len(self.trt_input_shape) >= 4:
                return self.trt_input_shape[2], self.trt_input_shape[3]

        if self.ort_input_shape:
            if len(self.ort_input_shape) >= 5:
                h = self.ort_input_shape[3]
                w = self.ort_input_shape[4]
                if isinstance(h, int) and isinstance(w, int):
                    return h, w
            elif len(self.ort_input_shape) >= 4:
                h = self.ort_input_shape[2]
                w = self.ort_input_shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    return h, w

        return crop_h, crop_w

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

    def _match_mde_input_rank(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self.ort_input_shape:
            return input_tensor

        expected_rank = len(self.ort_input_shape)
        if expected_rank == input_tensor.ndim:
            return input_tensor
        if expected_rank == 5 and input_tensor.ndim == 4:
            return input_tensor[:, None, ...]
        if expected_rank == 4 and input_tensor.ndim == 5:
            if input_tensor.shape[1] == 1:
                return input_tensor[:, 0, ...]
        return input_tensor

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

    def _compute_scale_recovery_params(
        self, depth_metric: np.ndarray, depth_relative: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        if depth_metric.shape != depth_relative.shape:
            self.get_logger().warning("MDE output shape mismatch; using metric depth.")
            return None

        safe_mask = self.safe_zone_mask
        if safe_mask.shape != depth_metric.shape:
            self.get_logger().warning("Safe zone mask mismatch; using metric depth.")
            return None

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
                return None
            median_mde = float(np.median(mde_vals))
            if abs(median_mde) < 1e-6:
                return None
            scale = float(np.median(real_vals)) / median_mde
            return scale, 0.0
        design = np.column_stack([mde_vals, np.ones_like(mde_vals)])
        solution, _, _, _ = np.linalg.lstsq(design, real_vals, rcond=None)
        scale, bias = solution
        return float(scale), float(bias)

    def _scale_recover(self, depth_metric: np.ndarray, depth_relative: np.ndarray) -> np.ndarray:
        params = self._compute_scale_recovery_params(depth_metric, depth_relative)
        if params is None:
            return depth_metric
        scale, bias = params
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
