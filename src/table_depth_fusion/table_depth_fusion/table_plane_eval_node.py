import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


def resolve_default_config_path() -> Path:
    candidates = []
    workspace_candidate = (
        Path.cwd() / "src" / "table_depth_fusion" / "config" / "table_config.npz"
    )
    candidates.append(workspace_candidate)

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(package_root / "config" / "table_config.npz")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_default_calibration_params_path() -> Path:
    candidates = []
    workspace_candidate = (
        Path.cwd() / "src" / "table_depth_fusion" / "config" / "calibration_params.yaml"
    )
    candidates.append(workspace_candidate)
    try:
        share_dir = Path(get_package_share_directory("table_depth_fusion"))
        candidates.append(share_dir / "config" / "calibration_params.yaml")
    except Exception:
        pass

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(package_root / "config" / "calibration_params.yaml")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_default_output_dir() -> Path:
    candidates = []
    workspace_candidate = Path.cwd() / "src" / "table_depth_fusion" / "logs"
    candidates.append(workspace_candidate)
    candidates.append(Path.cwd() / "logs")

    package_root = Path(__file__).resolve().parents[1]
    candidates.append(package_root / "logs")

    for candidate in candidates:
        if candidate.exists():
            return candidate
        if candidate.parent.exists() and os.access(candidate.parent, os.W_OK):
            return candidate
    return candidates[0]


class TablePlaneEvalNode(Node):
    def __init__(self) -> None:
        super().__init__("table_plane_eval_node")
        self.declare_parameter("points_topic", "/table/roi_points")
        self.declare_parameter("config_path", str(resolve_default_config_path()))
        self.declare_parameter("roi_size_m", [0.0, 0.0])
        self.declare_parameter(
            "roi_yaml_path", str(resolve_default_calibration_params_path())
        )
        self.declare_parameter("warmup_sec", 2.0)
        self.declare_parameter("eval_duration_sec", 10.0)
        self.declare_parameter("min_points", 2000)
        self.declare_parameter("z_tol_m", 0.003)
        self.declare_parameter("xy_margin_m", 0.002)
        self.declare_parameter("save_csv", True)
        self.declare_parameter("save_json", True)
        self.declare_parameter("output_dir", str(resolve_default_output_dir()))
        self.declare_parameter("method_tag", "")
        self.declare_parameter("exit_after_eval", True)

        self.config_path = self._resolve_config_path(
            str(self.get_parameter("config_path").value)
        )
        self.roi_yaml_path = self._resolve_optional_path(
            str(self.get_parameter("roi_yaml_path").value)
        )
        self.roi_size_m = self._parse_roi_size(self.get_parameter("roi_size_m").value)
        if self.roi_size_m is None:
            self.roi_size_m = self._load_roi_size_from_config(self.config_path)
        if self.roi_size_m is None:
            self.roi_size_m = self._load_roi_size_from_yaml(self.roi_yaml_path)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.eval_duration_sec = float(self.get_parameter("eval_duration_sec").value)
        self.min_points = int(self.get_parameter("min_points").value)
        self.z_tol_m = float(self.get_parameter("z_tol_m").value)
        self.xy_margin_m = float(self.get_parameter("xy_margin_m").value)
        self.save_csv = bool(self.get_parameter("save_csv").value)
        self.save_json = bool(self.get_parameter("save_json").value)
        self.method_tag = str(self.get_parameter("method_tag").value).strip()
        self.exit_after_eval = bool(self.get_parameter("exit_after_eval").value)

        raw_output_dir = str(self.get_parameter("output_dir").value).strip()
        self.output_dir = self._resolve_output_dir(raw_output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.get_logger().error(
                f"Failed to create output_dir '{self.output_dir}': {exc}"
            )
            self.save_csv = False
            self.save_json = False

        self.start_time = None
        self.eval_start_time = None
        self.completed = False
        self.records: List[Dict[str, object]] = []

        self.xy_check_enabled = self.roi_size_m is not None
        if not self.xy_check_enabled:
            self.get_logger().warning(
                "ROI size unavailable; XY checks disabled. "
                "Set roi_size_m, update table_config.npz, or provide roi_yaml_path."
            )

        points_topic = str(self.get_parameter("points_topic").value)
        self.create_subscription(PointCloud2, points_topic, self._points_cb, 10)
        self.get_logger().info(
            f"Plane eval started. Warmup={self.warmup_sec:.1f}s, "
            f"Eval={self.eval_duration_sec:.1f}s, Output={self.output_dir}"
        )

    def _resolve_output_dir(self, value: str) -> Path:
        if not value:
            return resolve_default_output_dir()
        path = Path(value)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _resolve_config_path(self, value: str) -> Path:
        raw = value.strip()
        if raw:
            path = Path(raw)
            if path.is_absolute():
                return path
            return Path.cwd() / path
        return resolve_default_config_path()

    def _resolve_optional_path(self, value: str) -> Optional[Path]:
        raw = value.strip()
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _parse_roi_size(self, value) -> Optional[np.ndarray]:
        if isinstance(value, np.ndarray) and value.size == 2:
            size = value.astype(np.float32, copy=False).reshape(2)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            size = np.array(value, dtype=np.float32)
        else:
            return None
        if size[0] > 0.0 and size[1] > 0.0:
            return size
        return None

    def _load_roi_size_from_config(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            self.get_logger().warning(f"Config file not found: {path}")
            return None
        try:
            data = np.load(path)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read config file: {exc}")
            return None
        if "roi_size_m" not in data:
            self.get_logger().warning(f"roi_size_m not found in {path}")
            return None
        return self._parse_roi_size(data["roi_size_m"])

    def _load_roi_size_from_yaml(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        if yaml is None:
            self.get_logger().warning("PyYAML is not available; skipping YAML lookup.")
            return None
        if not path.exists():
            self.get_logger().warning(f"ROI YAML not found: {path}")
            return None
        try:
            with path.open("r") as yaml_file:
                data = yaml.safe_load(yaml_file)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read ROI YAML: {exc}")
            return None

        roi_size = self._extract_roi_size_from_yaml(data)
        if roi_size is None:
            self.get_logger().warning(f"roi_size_m not found in {path}")
        return roi_size

    def _extract_roi_size_from_yaml(self, data) -> Optional[np.ndarray]:
        if not isinstance(data, dict):
            return None
        node_section = data.get("fusion_depth_calibration_node")
        if isinstance(node_section, dict):
            params = node_section.get("ros__parameters")
            if isinstance(params, dict) and "roi_size_m" in params:
                return self._parse_roi_size(params["roi_size_m"])
        for _, node_data in data.items():
            if isinstance(node_data, dict):
                params = node_data.get("ros__parameters")
                if isinstance(params, dict) and "roi_size_m" in params:
                    return self._parse_roi_size(params["roi_size_m"])
        if "roi_size_m" in data:
            return self._parse_roi_size(data["roi_size_m"])
        return None

    def _points_cb(self, msg: PointCloud2) -> None:
        if self.completed:
            return

        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now

        elapsed = (now - self.start_time).nanoseconds * 1e-9
        if elapsed < self.warmup_sec:
            return

        if self.eval_start_time is None:
            self.eval_start_time = now

        eval_elapsed = (now - self.eval_start_time).nanoseconds * 1e-9
        if eval_elapsed > self.eval_duration_sec:
            self._finalize()
            return

        points = self._cloud_to_numpy(msg)
        num_points = int(points.shape[0])
        record: Dict[str, object] = {
            "stamp": self._stamp_to_sec(msg),
            "status": "ok",
            "num_points": num_points,
            "z_median": float("nan"),
            "z_rmse": float("nan"),
            "z_outlier_ratio": float("nan"),
            "xy_outlier_ratio": float("nan"),
            "x_min": float("nan"),
            "x_max": float("nan"),
            "y_min": float("nan"),
            "y_max": float("nan"),
            "tilt_deg": float("nan"),
        }

        if num_points < self.min_points:
            record["status"] = "skip_min_points"
            self.records.append(record)
            return

        metrics = self._compute_metrics(points)
        record.update(metrics)
        self.records.append(record)

    def _cloud_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        points_iter = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        points_list = list(points_iter)
        if not points_list:
            return np.zeros((0, 3), dtype=np.float32)
        points = np.array(points_list)
        if points.dtype.names is not None:
            points = np.stack(
                [points["x"], points["y"], points["z"]], axis=-1
            )
        if points.ndim == 1:
            points = np.array(points_list, dtype=np.float32).reshape(-1, 3)
        return points.astype(np.float32, copy=False)

    def _compute_metrics(self, points: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        z_vals = points[:, 2]
        metrics["z_median"] = float(np.median(z_vals))
        metrics["z_rmse"] = float(np.sqrt(np.mean(z_vals ** 2)))
        metrics["z_outlier_ratio"] = float(np.mean(np.abs(z_vals) > self.z_tol_m))

        x_vals = points[:, 0]
        y_vals = points[:, 1]
        metrics["x_min"] = float(np.min(x_vals))
        metrics["x_max"] = float(np.max(x_vals))
        metrics["y_min"] = float(np.min(y_vals))
        metrics["y_max"] = float(np.max(y_vals))

        if self.xy_check_enabled:
            width_m, height_m = float(self.roi_size_m[0]), float(self.roi_size_m[1])
            x_ok = (x_vals >= -self.xy_margin_m) & (
                x_vals <= width_m + self.xy_margin_m
            )
            y_ok = (y_vals >= -self.xy_margin_m) & (
                y_vals <= height_m + self.xy_margin_m
            )
            metrics["xy_outlier_ratio"] = float(np.mean(~(x_ok & y_ok)))

        metrics["tilt_deg"] = self._compute_tilt_deg(points)
        return metrics

    def _compute_tilt_deg(self, points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return float("nan")

        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, v_t = np.linalg.svd(centered, full_matrices=False)
        normal = v_t[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return float("nan")
        normal = normal / norm
        tilt_rad = np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))
        return float(np.degrees(tilt_rad))

    def _stamp_to_sec(self, msg: PointCloud2) -> float:
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if stamp_sec > 0.0:
            return stamp_sec
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _finalize(self) -> None:
        if self.completed:
            return
        self.completed = True
        if not self.records:
            self.get_logger().warning("No records collected during evaluation.")
            if self.exit_after_eval and rclpy.ok():
                rclpy.shutdown()
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self.method_tag if self.method_tag else "default"
        base_name = f"plane_eval_{tag}_{timestamp}"

        if self.save_csv:
            csv_path = self.output_dir / f"{base_name}.csv"
            self._write_csv(csv_path)
            self.get_logger().info(f"Saved CSV: {csv_path}")
        if self.save_json:
            json_path = self.output_dir / f"{base_name}.json"
            self._write_json(json_path)
            self.get_logger().info(f"Saved JSON: {json_path}")

        self.get_logger().info(
            f"Plane eval complete. Records={len(self.records)} "
            f"CSV={self.save_csv} JSON={self.save_json}"
        )
        if self.exit_after_eval and rclpy.ok():
            rclpy.shutdown()

    def _write_csv(self, path: Path) -> None:
        fieldnames = [
            "stamp",
            "status",
            "num_points",
            "z_median",
            "z_rmse",
            "z_outlier_ratio",
            "xy_outlier_ratio",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "tilt_deg",
        ]
        try:
            with path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.records:
                    writer.writerow(record)
        except Exception as exc:
            self.get_logger().error(f"Failed to write CSV {path}: {exc}")

    def _write_json(self, path: Path) -> None:
        metrics = {
            "z_median": self._summary_stats("z_median"),
            "z_rmse": self._summary_stats("z_rmse"),
            "z_outlier_ratio": self._summary_stats("z_outlier_ratio"),
            "xy_outlier_ratio": self._summary_stats("xy_outlier_ratio"),
            "tilt_deg": self._summary_stats("tilt_deg"),
        }
        summary = {
            "method_tag": self.method_tag or "default",
            "roi_size_m": (
                self.roi_size_m.tolist() if self.roi_size_m is not None else None
            ),
            "warmup_sec": self.warmup_sec,
            "eval_duration_sec": self.eval_duration_sec,
            "min_points": self.min_points,
            "z_tol_m": self.z_tol_m,
            "xy_margin_m": self.xy_margin_m,
            "records_total": len(self.records),
            "records_valid": sum(
                1 for record in self.records if record.get("status") == "ok"
            ),
            "records_skipped": sum(
                1 for record in self.records if record.get("status") != "ok"
            ),
            "metrics": metrics,
        }
        try:
            with path.open("w") as json_file:
                json.dump(summary, json_file, indent=2)
        except Exception as exc:
            self.get_logger().error(f"Failed to write JSON {path}: {exc}")

    def _summary_stats(self, key: str) -> Dict[str, float]:
        values = [
            record.get(key, float("nan"))
            for record in self.records
            if record.get("status") == "ok"
        ]
        data = np.array(values, dtype=np.float64)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return {"count": 0}
        return {
            "count": int(data.size),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "p95": float(np.percentile(data, 95)),
        }


def main() -> None:
    rclpy.init()
    node = TablePlaneEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
