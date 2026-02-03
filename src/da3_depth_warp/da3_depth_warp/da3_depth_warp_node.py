from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

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


class DA3DepthWarpNode(Node):
    def __init__(self) -> None:
        super().__init__("da3_depth_warp_node")
        self.bridge = CvBridge()

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("engine_path", str(resolve_default_engine_path()))
        self.declare_parameter("input_width", 0)
        self.declare_parameter("input_height", 0)
        self.declare_parameter("letterbox", True)
        self.declare_parameter("depth_scale", 1.0)
        self.declare_parameter("model_depth_output", "")
        self.declare_parameter("warp_corners_2d", [])
        self.declare_parameter("adaptive_block_size", 51)
        self.declare_parameter("adaptive_C", 5)
        self.declare_parameter("depth_raw_topic", "/da3/depth_raw")
        self.declare_parameter("depth_warp_topic", "/da3/depth_warp")
        self.declare_parameter("depth_binary_topic", "/da3/depth_binary")

        self.engine_path = self._resolve_engine_path(
            str(self.get_parameter("engine_path").value)
        )
        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.letterbox = bool(self.get_parameter("letterbox").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.model_depth_output = str(
            self.get_parameter("model_depth_output").value
        ).strip()

        self.warp_corners_2d = self._parse_warp_corners(
            self.get_parameter("warp_corners_2d").value
        )
        self.warp_matrix = None
        self.warp_size = None
        self.warp_warned = False
        if self.warp_corners_2d is not None:
            self._update_warp_matrix()

        self.adaptive_block_size = int(
            self.get_parameter("adaptive_block_size").value
        )
        if self.adaptive_block_size < 3:
            self.adaptive_block_size = 3
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1
        self.adaptive_C = float(self.get_parameter("adaptive_C").value)

        self.depth_raw_topic = str(self.get_parameter("depth_raw_topic").value)
        self.depth_warp_topic = str(self.get_parameter("depth_warp_topic").value)
        self.depth_binary_topic = str(self.get_parameter("depth_binary_topic").value)

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

        if trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT or pycuda not available; cannot run TensorRT engine."
            )

        if not self._init_trt_engine():
            raise RuntimeError("Failed to initialize TensorRT engine.")

        self.depth_raw_pub = self.create_publisher(Image, self.depth_raw_topic, 10)
        self.depth_warp_pub = self.create_publisher(Image, self.depth_warp_topic, 10)
        self.depth_binary_pub = self.create_publisher(
            Image, self.depth_binary_topic, 10
        )

        color_topic = str(self.get_parameter("color_topic").value)
        self.color_sub = self.create_subscription(
            Image, color_topic, self._color_cb, qos_profile_sensor_data
        )

        self.get_logger().info("DA3 depth warp node started.")

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

    def _color_cb(self, color_msg: Image) -> None:
        try:
            input_tensor, meta = self._prepare_model_input(color_msg)
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

        header = color_msg.header
        self._publish_image(self.depth_raw_pub, depth, header, "32FC1")

        if self.warp_matrix is None or self.warp_size is None:
            if self.warp_corners_2d is not None and not self.warp_warned:
                self.get_logger().warning("Warp matrix unavailable; skipping warp.")
                self.warp_warned = True
            return

        warped = self._warp_depth(depth)
        if warped is None:
            return
        self._publish_image(self.depth_warp_pub, warped, header, "32FC1")

        depth_u8 = self._normalize_depth(warped)
        binary = cv2.adaptiveThreshold(
            depth_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            self.adaptive_C,
        )
        self._publish_image(self.depth_binary_pub, binary, header, "mono8")

    def _prepare_model_input(self, color_msg: Image) -> Tuple[np.ndarray, dict]:
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

    def _update_warp_matrix(self) -> None:
        if self.warp_corners_2d is None:
            return
        matrix, size = self._compute_warp_matrix(self.warp_corners_2d)
        if matrix is None or size is None:
            self.get_logger().warning("Invalid warp_corners_2d; warp disabled.")
            self.warp_corners_2d = None
            self.warp_matrix = None
            self.warp_size = None
            return
        self.warp_matrix = matrix
        self.warp_size = size

    @staticmethod
    def _compute_warp_matrix(
        corners: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        corners = np.array(corners, dtype=np.float32).reshape(4, 2)
        tl, tr, bl, br = corners

        width_a = float(np.linalg.norm(tr - tl))
        width_b = float(np.linalg.norm(br - bl))
        height_a = float(np.linalg.norm(bl - tl))
        height_b = float(np.linalg.norm(br - tr))

        width = int(round(max(width_a, width_b)))
        height = int(round(max(height_a, height_b)))
        if width < 1 or height < 1:
            return None, None

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return matrix, (width, height)

    @staticmethod
    def _parse_warp_corners(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.array(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.size != 8:
            return None
        return arr.reshape(4, 2)

    def _warp_depth(self, depth: np.ndarray) -> Optional[np.ndarray]:
        if self.warp_matrix is None or self.warp_size is None:
            return None
        width, height = self.warp_size
        return cv2.warpPerspective(
            depth,
            self.warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _normalize_depth(depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            return np.zeros(depth.shape, dtype=np.uint8)
        min_val = float(np.min(depth[valid]))
        max_val = float(np.max(depth[valid]))
        if max_val <= min_val:
            return np.zeros(depth.shape, dtype=np.uint8)
        norm = (depth - min_val) / (max_val - min_val)
        norm = np.clip(norm, 0.0, 1.0)
        norm[~valid] = 0.0
        return (norm * 255.0).astype(np.uint8)

    def _publish_image(self, publisher, image: np.ndarray, header, encoding: str) -> None:
        msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        msg.header = header
        publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = DA3DepthWarpNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
