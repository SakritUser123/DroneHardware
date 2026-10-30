from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


class FpsCounter:
    def __init__(self, window: int = 20):
        self.times = deque(maxlen=window)

    def tick(self) -> float:
        self.times.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-9)


def has_display() -> bool:
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def build_sensors(cfg, packet_sink=None):
    """Return (camera, lidar, simulator_or_None) for real or mock hardware."""
    if cfg.get("mock"):
        from .sim import MockCamera, MockLidar, Simulator
        sim = Simulator(cfg)
        return MockCamera(cfg, sim), MockLidar(cfg, sim, packet_sink), sim
    from .camera import PiCamera
    from .lidar.driver import LidarDriver
    return PiCamera(cfg["camera"]), LidarDriver(cfg["lidar"], packet_sink), None
