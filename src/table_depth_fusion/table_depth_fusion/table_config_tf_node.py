import math
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


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


class TableConfigTfNode(Node):
    def __init__(self) -> None:
        super().__init__("table_config_tf_node")
        self.declare_parameter("config_path", "")
        self.declare_parameter("parent_frame", "camera_color_optical_frame")
        self.declare_parameter("child_frame", "table_link")
        self.declare_parameter("translation_scale", 1.0)

        raw_config_path = str(self.get_parameter("config_path").value).strip()
        if raw_config_path:
            self.config_path = Path(raw_config_path)
        else:
            self.config_path = resolve_default_config_path()

        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.translation_scale = float(self.get_parameter("translation_scale").value)

        self.broadcaster = StaticTransformBroadcaster(self)
        if not self._publish_static_tf():
            self.get_logger().error("Failed to publish static TF; shutting down.")
            rclpy.shutdown()

    def _publish_static_tf(self) -> bool:
        if not self.config_path.exists():
            self.get_logger().error(f"Config file not found: {self.config_path}")
            return False

        try:
            data = np.load(self.config_path)
            t_matrix = data["T_matrix"].astype(np.float64)
        except Exception as exc:
            self.get_logger().error(f"Failed to read T_matrix: {exc}")
            return False

        if t_matrix.shape != (4, 4):
            self.get_logger().error(f"Unexpected T_matrix shape: {t_matrix.shape}")
            return False

        rotation = t_matrix[:3, :3]
        translation = t_matrix[:3, 3] * self.translation_scale
        qx, qy, qz, qw = self._quaternion_from_matrix(rotation)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"Published TF {self.parent_frame} -> {self.child_frame} "
            f"from {self.config_path}"
        )
        return True

    @staticmethod
    def _quaternion_from_matrix(matrix: np.ndarray) -> tuple:
        m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

        trace = m00 + m11 + m22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm > 0.0:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm
        return float(qx), float(qy), float(qz), float(qw)


def main() -> None:
    rclpy.init()
    node = TableConfigTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
