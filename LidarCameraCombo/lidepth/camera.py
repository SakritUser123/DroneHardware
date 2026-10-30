"""Raspberry Pi Camera Module 3 capture (Picamera2) with timestamps on the lidar's clock.

Clock note: libcamera's ``SensorTimestamp`` is the kernel V4L2 buffer time, which uses
CLOCK_MONOTONIC — the same clock as Python's ``time.monotonic_ns()`` on Linux, which the
lidar driver uses.  We add half the exposure time so the timestamp is the middle of the
exposure.  A sanity check falls back to arrival time if the clocks ever disagree.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray            # BGR uint8 (H, W, 3), as captured (still distorted)
    t_ns: int                    # capture time, host monotonic clock (middle of exposure)
    index: int = 0
    meta: Dict = field(default_factory=dict)
    truth_depth: Optional[np.ndarray] = None   # only in mock mode (ground truth Z, metres)


class PiCamera:
    def __init__(self, cam_cfg: Dict):
        self.cfg = cam_cfg
        self.picam2 = None
        self.index = 0
        self._warned_clock = False
        self._flush_ok = True

    def start(self) -> "PiCamera":
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "Picamera2 not found. On the Pi: sudo apt install -y python3-picamera2, and create "
                "your venv with --system-site-packages (see README). On a laptop use --mock or "
                "the `process` command on a recorded session.") from e
        c = self.cfg
        size = (int(c["width"]), int(c["height"]))
        try:
            self.picam2 = Picamera2()
        except (IndexError, RuntimeError) as e:
            raise RuntimeError(
                "No camera detected. Check the ribbon cable (contacts facing the right way, "
                "latch closed), then run `rpicam-hello --list-cameras`.") from e
        main = {"size": size, "format": "RGB888"}   # RGB888 in Picamera2 = BGR byte order (OpenCV)
        sensor = {"output_size": tuple(c["sensor_output_size"])} if c.get("sensor_output_size") else None
        try:
            conf = self.picam2.create_video_configuration(main=main, sensor=sensor, buffer_count=4) \
                if sensor else self.picam2.create_video_configuration(main=main, buffer_count=4)
        except TypeError:
            log.warning("This Picamera2 version does not accept 'sensor=' — using default mode.")
            conf = self.picam2.create_video_configuration(main=main, buffer_count=4)
        self.picam2.configure(conf)
        controls_ = {}
        try:
            from libcamera import controls as lc
            controls_["AfMode"] = lc.AfModeEnum.Manual
            controls_["LensPosition"] = float(c["lens_position"])
        except Exception as e:   # camera without autofocus
            log.warning("Could not set manual focus (%s) — fine for fixed-focus cameras.", e)
        if int(c.get("exposure_time_us", 0)) > 0:
            controls_["ExposureTime"] = int(c["exposure_time_us"])
        if controls_:
            self.picam2.set_controls(controls_)
        self.picam2.start()
        time.sleep(1.0)  # let AE/AWB settle and the lens move
        log.info("Camera started at %dx%d, lens position %.2f dioptres", size[0], size[1],
                 float(c["lens_position"]))
        return self

    def capture(self) -> Frame:
        if self._flush_ok:
            try:
                req = self.picam2.capture_request(flush=True)
            except TypeError:
                self._flush_ok = False
                req = self.picam2.capture_request()
        else:
            req = self.picam2.capture_request()
        t_arrival = time.monotonic_ns()
        try:
            img = req.make_array("main").copy()
            md = req.get_metadata()
        finally:
            req.release()
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3].copy()
        ts = md.get("SensorTimestamp")
        exp_us = md.get("ExposureTime", 0) or 0
        if ts is None:
            t_ns = t_arrival
        else:
            t_ns = int(ts) + int(exp_us * 1000 / 2)
            if abs(t_arrival - t_ns) > 2e9:     # clocks disagree → don't trust it
                if not self._warned_clock:
                    log.warning("Camera SensorTimestamp is %.1f s away from monotonic time — "
                                "using arrival time instead (sync will be less exact).",
                                (t_arrival - t_ns) / 1e9)
                    self._warned_clock = True
                t_ns = t_arrival
        t_ns += int(float(self.cfg.get("timestamp_offset_s", 0.0)) * 1e9)
        self.index += 1
        meta = {"exposure_us": exp_us, "lens_position": md.get("LensPosition"),
                "analogue_gain": md.get("AnalogueGain"), "arrival_ns": t_arrival}
        return Frame(img, t_ns, self.index, meta)

    def stop(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass


def build_camera(cfg, sim=None):
    if cfg.get("mock"):
        from .sim import MockCamera
        return MockCamera(cfg, sim)
    return PiCamera(cfg["camera"])
