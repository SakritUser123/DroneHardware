"""Helpers shared by the command-line tools."""
from __future__ import annotations

import logging
import os

from ..calibration import load_extrinsics, load_intrinsics
from ..pipeline import DepthPipeline
from ..session import SessionReader

log = logging.getLogger(__name__)


def session_calibration(cfg, reader: SessionReader, use_session_calib: bool = False):
    """Pick the calibration for processing a recorded session.

    * mock session  → the simulator's exact camera model saved with the session
    * --session-calib → the calibration files copied into the session at record time
    * otherwise     → the CURRENT calib/*.yaml (so you can re-calibrate and re-process)
    """
    c = cfg["camera"]
    w, h = int(c["width"]), int(c["height"])
    img = reader.image(0)
    if (img.shape[1], img.shape[0]) != (w, h):
        log.warning("Session images are %dx%d but config says %dx%d — using the session size.",
                    img.shape[1], img.shape[0], w, h)
        c["width"], c["height"] = img.shape[1], img.shape[0]
        w, h = img.shape[1], img.shape[0]
    cam_file = cfg["calibration"]["camera_file"]
    ext_file = cfg["calibration"]["extrinsics_file"]
    if reader.is_mock and os.path.exists(os.path.join(reader.path, "sim_intrinsics.yaml")):
        cam_file = os.path.join(reader.path, "sim_intrinsics.yaml")
        ext_file = os.path.join(reader.path, "sim_extrinsics.yaml")
        log.info("Mock session: using the simulator's camera model and extrinsics")
    elif use_session_calib:
        cam_file = os.path.join(reader.path, os.path.basename(cam_file))
        ext_file = os.path.join(reader.path, os.path.basename(ext_file))
        log.info("Using calibration stored in the session folder")
    intr = load_intrinsics(cam_file, w, h, c)
    ext = load_extrinsics(ext_file, cfg["extrinsics_default"])
    return intr, ext


def pipeline_for_session(cfg, reader: SessionReader, model=None, use_session_calib: bool = False):
    """Build the processing pipeline (and the depth model, unless one is given)."""
    from ..depth_model import build_model
    if reader.is_mock and not os.path.exists(cfg["model"]["path"]):
        cfg["mock"] = True        # no ONNX file → build_model falls back to the fake model
    if model is None:
        model = build_model(cfg)
    intr, ext = session_calibration(cfg, reader, use_session_calib)
    return DepthPipeline(cfg, model, intr, ext)
