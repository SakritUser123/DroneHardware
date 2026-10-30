"""Configuration loading.

All parameters live in ``config.yaml``.  ``DEFAULTS`` below mirrors that file so the
code still works if a key is missing (e.g. an older config.yaml).  A unit test checks
that both contain the same keys.
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "mock": False,
    "log_level": "INFO",
    "lidar": {
        "model": "LD19", "port": "/dev/ttyUSB0", "baud": 230400,
        "min_range_m": 0.02, "max_range_m": 12.0, "min_intensity": 15,
        "ld06_angle_correction": False, "latency_s": 0.0, "scan_buffer_size": 30,
        "no_data_timeout_s": 3.0, "upside_down": False,
    },
    "camera": {
        "width": 640, "height": 480, "sensor_output_size": [2304, 1296],
        "lens_position": 1.0, "exposure_time_us": 0, "timestamp_offset_s": 0.0,
        "hfov_deg": 66.0, "vfov_deg": 41.0, "jpeg_quality": 95, "save_format": "jpg",
    },
    "model": {
        "path": "models/depth_anything_v2_small.onnx", "input_long_side": 182,
        "threads": 4, "mock_model": False,
    },
    "calibration": {
        "camera_file": "calib/camera.yaml", "extrinsics_file": "calib/extrinsics.yaml",
        "checkerboard_cols": 9, "checkerboard_rows": 6, "square_size_mm": 25.0,
        "min_images": 15,
    },
    "extrinsics_default": {
        "cam_up_m": 0.05, "cam_forward_m": 0.0, "cam_left_m": 0.0,
        "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
    },
    "projection": {"min_z_m": 0.05, "sample_window": 5, "border_px": 3},
    "sync": {"tolerance_s": 0.05, "angle_bin_deg": 0.5},
    "fusion": {
        "min_points": 15, "min_inliers": 10, "ransac_iters": 200, "inlier_rel_err": 0.10,
        "min_d_spread": 0.05, "ema_alpha": 0.5, "max_range_margin": 1.5,
        "min_depth_m": 0.05, "max_fallback_frames": 30, "conf_sigma_px": 60, "seed": 0,
    },
    "display": {"depth_max_m": 6.0, "point_radius": 3, "window_scale": 1.0},
    "record": {"sessions_dir": "sessions", "fps": 5, "duration_s": 0},
    "evaluate": {"gt_type": "range", "near_line_px": 40},
    "obstacle_output": {
        "enabled": False, "sectors": 72, "band_up_m": 0.3, "band_down_m": 0.3,
        "percentile": 0, "min_confidence": 0.0, "min_distance_cm": 20,
        "max_distance_cm": 1200, "mavlink_port": "/dev/serial0", "mavlink_baud": 921600,
        "send_rate_hz": 10,
    },
}

# keys holding file paths that are resolved relative to the config file
_PATH_KEYS = [
    ("model", "path"),
    ("calibration", "camera_file"),
    ("calibration", "extrinsics_file"),
    ("record", "sessions_dir"),
]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text: str) -> Any:
    """Parse a command-line value using YAML rules (numbers, bools, lists)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad --set '{item}'. Use --set section.key=value")
        key, value = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config section '{p}' in --set {item}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key '{key}' in --set {item}")
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: Optional[str] = "config.yaml",
                overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    user: Dict[str, Any] = {}
    base_dir = os.getcwd()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        base_dir = os.path.dirname(os.path.abspath(path))
    elif path:
        log.warning("Config file %s not found — using built-in defaults.", path)
    cfg = deep_merge(DEFAULTS, user)
    cfg = apply_overrides(cfg, overrides)
    for sec, key in _PATH_KEYS:
        p = cfg[sec][key]
        if p and not os.path.isabs(p):
            cfg[sec][key] = os.path.normpath(os.path.join(base_dir, p))
    cfg["_base_dir"] = base_dir
    return cfg
