"""One frame in → metric depth out.  Shared by `live`, `process` and the tools."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .calibration import Extrinsics, Intrinsics, Undistorter, load_extrinsics, load_intrinsics
from .depth_model import MockDepthModel
from .fusion import FusionEngine, FusionOutput
from .lidar.buffer import LidarBuffer, Scan
from .projection import Projected, project_scan

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    image: np.ndarray            # undistorted BGR image
    d_model: np.ndarray          # raw model output (relative inverse depth), image size
    proj: Projected              # lidar points that landed in the image
    points: Scan                 # time-synchronised lidar points used
    fusion: FusionOutput
    t_ns: int
    timings: Dict[str, float] = field(default_factory=dict)


class DepthPipeline:
    def __init__(self, cfg: Dict, model, intrinsics: Optional[Intrinsics] = None,
                 extrinsics: Optional[Extrinsics] = None):
        self.cfg = cfg
        c = cfg["camera"]
        self.w, self.h = int(c["width"]), int(c["height"])
        self.intr = intrinsics or load_intrinsics(cfg["calibration"]["camera_file"], self.w, self.h, c)
        self.undist = Undistorter(self.intr)
        self.K = self.undist.K_new
        self.set_extrinsics(extrinsics or load_extrinsics(cfg["calibration"]["extrinsics_file"],
                                                          cfg["extrinsics_default"]))
        self.model = model
        self.fusion = FusionEngine(cfg["fusion"], float(cfg["lidar"]["max_range_m"]),
                                   int(cfg["projection"]["sample_window"]))

    def set_extrinsics(self, ext: Extrinsics) -> None:
        self.ext = ext
        self.R, self.t = ext.rotation_translation(bool(self.cfg["lidar"]["upside_down"]))

    def project(self, pts: Scan) -> Projected:
        p = self.cfg["projection"]
        return project_scan(pts.angle_deg, pts.range_m, self.R, self.t, self.K, self.w, self.h,
                            float(p["min_z_m"]), int(p["border_px"]),
                            bool(self.cfg["lidar"]["upside_down"]))

    def sync_points(self, buffer: LidarBuffer, t_ns: int) -> Scan:
        s = self.cfg["sync"]
        return buffer.query(t_ns, float(s["tolerance_s"]), float(s["angle_bin_deg"]))

    def process(self, image_bgr: np.ndarray, t_ns: int, buffer: LidarBuffer,
                truth_depth: Optional[np.ndarray] = None) -> FrameResult:
        tm: Dict[str, float] = {}
        if image_bgr.shape[1] != self.w or image_bgr.shape[0] != self.h:
            import cv2
            image_bgr = cv2.resize(image_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        img = self.undist.apply(image_bgr)
        tm["undistort"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if isinstance(self.model, MockDepthModel):
            self.model.truth = truth_depth
        d = self.model.predict(img)
        tm["model"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pts = self.sync_points(buffer, t_ns)
        proj = self.project(pts)
        fus = self.fusion.process(d, proj)
        tm["fusion"] = time.perf_counter() - t0
        return FrameResult(img, d, proj, pts, fus, t_ns, tm)
