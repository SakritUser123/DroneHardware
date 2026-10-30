"""Camera intrinsics, undistortion and lidar→camera extrinsics (load / save / defaults)."""
from __future__ import annotations

import datetime as _dt
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- intrinsics
@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list = field(default_factory=lambda: [0.0] * 5)   # k1 k2 p1 p2 k3 (OpenCV order)
    rms_px: float = -1.0
    estimated: bool = False      # True = rough guess from FOV, not a real calibration

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist, float).reshape(-1)

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """Same camera at another resolution (same aspect ratio / same sensor crop)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy,
                          (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                          list(self.dist), self.rms_px, self.estimated)


def estimate_intrinsics(width: int, height: int, hfov_deg: float = 66.0,
                        vfov_deg: float = 41.0) -> Intrinsics:
    """Rough pinhole guess from the datasheet FOV (sensor is 16:9; 4:3 output crops the sides)."""
    sensor_aspect = math.tan(math.radians(hfov_deg / 2)) / math.tan(math.radians(vfov_deg / 2))
    if width / height <= sensor_aspect:    # image narrower than the sensor → height uses full vFOV
        f = (height / 2) / math.tan(math.radians(vfov_deg / 2))
    else:
        f = (width / 2) / math.tan(math.radians(hfov_deg / 2))
    return Intrinsics(width, height, f, f, (width - 1) / 2, (height - 1) / 2, [0.0] * 5,
                      estimated=True)


def save_intrinsics(path: str, intr: Intrinsics, extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = asdict(intr)
    d["dist"] = [float(x) for x in intr.dist]
    d["saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    if extra:
        d.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera intrinsics written by `python -m lidepth calibrate-camera`\n")
        f.write("# fx, fy, cx, cy in pixels; dist = OpenCV [k1, k2, p1, p2, k3]\n")
        yaml.safe_dump(d, f, sort_keys=False)


def load_intrinsics(path: str, width: int, height: int, cam_cfg: Optional[Dict] = None,
                    quiet: bool = False) -> Intrinsics:
    """Load camera.yaml (rescaled to the requested size) or fall back to a FOV estimate."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        intr = Intrinsics(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]),
                          float(d["cx"]), float(d["cy"]), list(d.get("dist", [0] * 5)),
                          float(d.get("rms_px", -1)), bool(d.get("estimated", False)))
        if (intr.width, intr.height) != (width, height):
            if abs(intr.width / intr.height - width / height) > 0.01:
                log.warning("camera.yaml was calibrated at %dx%d but the camera runs at %dx%d "
                            "(different aspect ratio → different crop). Re-run calibrate-camera!",
                            intr.width, intr.height, width, height)
            intr = intr.scaled(width, height)
        return intr
    cam_cfg = cam_cfg or {}
    if not quiet:
        log.warning("No camera calibration at %s — using a rough FOV-based guess. "
                    "Run `python -m lidepth calibrate-camera` for real results.", path)
    return estimate_intrinsics(width, height, float(cam_cfg.get("hfov_deg", 66.0)),
                               float(cam_cfg.get("vfov_deg", 41.0)))


class Undistorter:
    """Removes lens distortion. After ``apply`` use ``K_new`` (no distortion) for projection."""

    def __init__(self, intr: Intrinsics):
        import cv2
        self.intr = intr
        size = (intr.width, intr.height)
        self.identity = not np.any(np.abs(intr.D) > 1e-9)
        if self.identity:
            self.K_new = intr.K.copy()
            self.map1 = self.map2 = None
        else:
            K_new, _roi = cv2.getOptimalNewCameraMatrix(intr.K, intr.D, size, 0.0, size)
            self.K_new = K_new
            self.map1, self.map2 = cv2.initUndistortRectifyMap(intr.K, intr.D, None, K_new, size,
                                                               cv2.CV_16SC2)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if self.identity:
            return img
        import cv2
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- extrinsics
@dataclass
class Extrinsics:
    """Pose of the camera relative to the lidar.

    Lidar frame: x forward (lidar 0°), y left, z up; origin = lidar centre, z=0 = scan plane.
    cam_up/forward/left_m : camera optical centre position in that frame (metres).
    yaw_deg   : lidar angle (clockwise, as the lidar reports it) the camera looks along.
    pitch_deg : camera tilt, positive = looking up (points move DOWN in the image).
    roll_deg  : positive = projected points rotate CLOCKWISE in the image.
    """
    cam_up_m: float = 0.05
    cam_forward_m: float = 0.0
    cam_left_m: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def rotation_translation(self, upside_down: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, t) such that p_cam = R @ p_lidar + t (OpenCV camera: x right, y down, z fwd)."""
        yaw = math.radians(self.yaw_deg) * (-1.0 if upside_down else 1.0)
        pitch, roll = math.radians(self.pitch_deg), math.radians(self.roll_deg)
        cz, sz = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        cp, sp = math.cos(pitch), math.sin(pitch)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        A = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)   # body (fwd,left,up) → optical
        cr, sr = math.cos(roll), math.sin(roll)
        Rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        R = Rroll @ A @ Ry @ Rz
        c = np.array([self.cam_forward_m, self.cam_left_m, self.cam_up_m])
        t = -R @ c
        return R, t


def load_extrinsics(path: str, defaults: Dict, quiet: bool = False) -> Extrinsics:
    d = dict(defaults or {})
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d.update({k: v for k, v in (yaml.safe_load(f) or {}).items()
                      if k in Extrinsics.__dataclass_fields__})
    elif not quiet:
        log.warning("No extrinsics file at %s — using defaults from config.yaml. "
                    "Run `python -m lidepth calibrate-extrinsics`.", path)
    return Extrinsics(**{k: float(v) for k, v in d.items() if k in Extrinsics.__dataclass_fields__})


def save_extrinsics(path: str, ext: Extrinsics) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Lidar→camera extrinsics written by `python -m lidepth calibrate-extrinsics`\n")
        f.write("# cam_*_m: camera position in the lidar frame (x fwd, y left, z up) [m]\n")
        f.write("# yaw_deg: lidar angle (clockwise) the camera looks along; pitch +up; roll +clockwise\n")
        d = {k: round(float(v), 5) for k, v in asdict(ext).items()}
        d["saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        yaml.safe_dump(d, f, sort_keys=False)
