"""Hardware-free simulator: a small 3-D room seen by a virtual camera and a virtual LD19.

The simulated lidar produces REAL protocol bytes (with CRC) that go through the same
parser as the hardware, so mock runs exercise the whole software chain.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration import Extrinsics, estimate_intrinsics, load_extrinsics, load_intrinsics
from .camera import Frame
from .lidar.buffer import LidarBuffer
from .lidar.driver import PacketProcessor
from .lidar.protocol import POINTS_PER_PACK, PACKET_LEN, StreamParser, encode_packet

log = logging.getLogger(__name__)


@dataclass
class Box:
    lo: Tuple[float, float, float]     # (x, y, z) min corner, lidar frame (x fwd, y left, z up)
    hi: Tuple[float, float, float]
    color: Tuple[int, int, int]        # BGR
    name: str = ""


def _ray_box(o: np.ndarray, d: np.ndarray, lo, hi):
    """Slab test. o (3,), d (N,3). Returns (t_near, t_far) arrays (inf if no hit)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (np.asarray(lo) - o) * inv
        t2 = (np.asarray(hi) - o) * inv
    t1 = np.nan_to_num(t1, nan=-np.inf)
    t2 = np.nan_to_num(t2, nan=np.inf)
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    return tmin, tmax


class Scene:
    """Room 8 m × 6 m × 2.6 m with boxes, a pole, a low box below the lidar plane, and one
    box that slides back and forth so consecutive frames differ."""

    def __init__(self):
        self.room = Box((-3.0, -3.0, -0.6), (5.5, 3.0, 2.0), (200, 200, 200), "room")
        self.static = [
            Box((2.0, 0.6, -0.6), (2.6, 1.4, 0.5), (60, 120, 200), "box_left"),
            Box((3.4, -1.8, -0.6), (4.0, -0.9, 0.9), (80, 180, 90), "box_right"),
            Box((1.6, -0.45, -0.6), (1.65, -0.4, 2.0), (40, 40, 40), "pole"),
            Box((1.2, 0.2, -0.6), (1.6, 0.6, -0.25), (150, 70, 170), "low_box"),  # below scan plane
        ]
        self.moving_base = Box((2.8, -0.3, -0.6), (3.2, 0.3, 0.3), (40, 200, 230), "moving")

    def boxes(self, t_s: float) -> List[Box]:
        dx = 0.8 * math.sin(2 * math.pi * t_s / 8.0)
        m = self.moving_base
        mov = Box((m.lo[0] + dx, m.lo[1], m.lo[2]), (m.hi[0] + dx, m.hi[1], m.hi[2]), m.color, m.name)
        return self.static + [mov]

    def cast(self, o: np.ndarray, d: np.ndarray, t_s: float):
        """Nearest hit parameter t along each ray (d not normalised) + object id (0 = room)."""
        _, tfar = _ray_box(o, d, self.room.lo, self.room.hi)
        best = np.where(np.isfinite(tfar) & (tfar > 0), tfar, np.inf)
        oid = np.zeros(len(d), np.int32)
        for i, b in enumerate(self.boxes(t_s), start=1):
            tn, tf = _ray_box(o, d, b.lo, b.hi)
            hit = (tf >= np.maximum(tn, 0)) & (tn > 1e-6) & (tn < best)
            best = np.where(hit, tn, best)
            oid = np.where(hit, i, oid)
        return best, oid


class Simulator:
    """Deterministic in time: render(t) and packets_between(t0, t1) are pure functions."""

    def __init__(self, cfg: Dict, t0_ns: Optional[int] = None):
        self.cfg = cfg
        self.scene = Scene()
        c = cfg["camera"]
        self.w, self.h = int(c["width"]), int(c["height"])
        self.intr = load_intrinsics(cfg["calibration"]["camera_file"], self.w, self.h, c, quiet=True)
        if self.intr.dist and any(abs(x) > 0 for x in self.intr.dist):
            # simulator renders an ideal pinhole image
            self.intr = estimate_intrinsics(self.w, self.h, c["hfov_deg"], c["vfov_deg"])
        self.ext: Extrinsics = load_extrinsics(cfg["calibration"]["extrinsics_file"],
                                               cfg["extrinsics_default"], quiet=True)
        self.R, self.t = self.ext.rotation_translation(bool(cfg["lidar"]["upside_down"]))
        self.t0_ns = time.monotonic_ns() if t0_ns is None else int(t0_ns)
        self.rate_hz = 10.0
        self.samples_per_s = 4500.0
        self.dt_pt = 1.0 / self.samples_per_s
        self.step_deg = 360.0 * self.rate_hz / self.samples_per_s
        self.baud = int(cfg["lidar"]["baud"])
        self.tx_s = PACKET_LEN * 10.0 / self.baud
        self.rng = np.random.default_rng(3)
        uu, vv = np.meshgrid(np.arange(self.w), np.arange(self.h))
        K = self.intr.K
        dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
        self._dirs_l = (dc.reshape(-1, 3) @ self.R).astype(np.float32)   # R^T d (row-vector form)
        self._cam_origin = -self.R.T @ self.t

    # ------------------------------------------------------------------ camera
    def render(self, t_ns: int) -> Tuple[np.ndarray, np.ndarray]:
        """BGR image and true Z (along the optical axis) at time t."""
        ts = (t_ns - self.t0_ns) / 1e9
        o = self._cam_origin
        tpar, oid = self.scene.cast(o, self._dirs_l, ts)
        z = tpar.reshape(self.h, self.w)              # d_c has z=1 → t == Z
        pts = o + self._dirs_l * tpar[:, None]
        boxes = self.scene.boxes(ts)
        colors = np.array([self.scene.room.color] + [b.color for b in boxes], float)
        col = colors[oid]
        # texture: checker on everything (helps the eye; real images have texture too)
        chk = ((np.floor(pts[:, 0] * 4) + np.floor(pts[:, 1] * 4) + np.floor(pts[:, 2] * 4)) % 2)
        col *= (0.85 + 0.15 * chk)[:, None]
        # floor darker, simple distance fog
        floor = (oid == 0) & (pts[:, 2] < self.scene.room.lo[2] + 1e-3)
        col[floor] *= 0.6
        fog = np.clip(1.0 - tpar / 12.0, 0.35, 1.0)
        col *= fog[:, None]
        img = col.reshape(self.h, self.w, 3) + self.rng.normal(0, 3, (self.h, self.w, 3))
        return np.clip(img, 0, 255).astype(np.uint8), z.astype(np.float32)

    # ------------------------------------------------------------------ lidar
    def _packet(self, k: int) -> Tuple[bytes, int]:
        """Packet number k → (raw bytes, host arrival time ns)."""
        first = k * POINTS_PER_PACK
        t_pts_s = (first + np.arange(POINTS_PER_PACK)) * self.dt_pt     # seconds since t0
        ang = np.mod((first + np.arange(POINTS_PER_PACK)) * self.step_deg, 360.0)
        ts_mid = float(t_pts_s.mean())
        a = np.radians(ang)
        sign = 1.0 if self.cfg["lidar"]["upside_down"] else -1.0
        d = np.stack([np.cos(a), sign * np.sin(a), np.zeros_like(a)], -1)
        rng_m, _ = self.scene.cast(np.zeros(3), d, ts_mid)
        rng_m = rng_m + self.rng.normal(0, 0.008, len(rng_m))
        dist_mm = np.clip(np.round(rng_m * 1000), 0, 65535)
        inten = np.full(POINTS_PER_PACK, 200)
        drop = self.rng.random(POINTS_PER_PACK) < 0.02
        dist_mm[drop] = 0
        weak = self.rng.random(POINTS_PER_PACK) < 0.02
        inten[weak] = 5
        speed = int(round(360 * self.rate_hz))
        start_c = int(round(ang[0] * 100)) % 36000
        end_c = int(round(ang[-1] * 100)) % 36000
        t_end_s = t_pts_s[-1]
        ts_ms = int(t_end_s * 1000) % 30000
        raw = encode_packet(speed, start_c, end_c, dist_mm, inten, ts_ms)
        t_arr = self.t0_ns + int((t_end_s + self.tx_s) * 1e9)
        return raw, t_arr

    def packets_between(self, t_from_ns: int, t_to_ns: int) -> List[Tuple[bytes, int]]:
        pkt_dt = POINTS_PER_PACK * self.dt_pt
        lag = (POINTS_PER_PACK - 1) * self.dt_pt + self.tx_s
        k0 = max(0, int(math.floor(((t_from_ns - self.t0_ns) / 1e9 - lag) / pkt_dt)))
        out = []
        k = k0
        while True:
            raw, ta = self._packet(k)
            if ta >= t_to_ns:
                break
            if ta >= t_from_ns:
                out.append((raw, ta))
            k += 1
        return out


class MockLidar:
    """Same interface as LidarDriver, fed by the simulator in (real) time."""

    def __init__(self, cfg: Dict, sim: Simulator, packet_sink=None):
        self.cfg = cfg["lidar"]
        self.sim = sim
        self.buffer = LidarBuffer(int(self.cfg.get("scan_buffer_size", 30)))
        self.processor = PacketProcessor(self.cfg, self.buffer)
        self.parser = StreamParser(int(self.cfg["baud"]))
        self.packet_sink = packet_sink
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="mock-lidar", daemon=True)
        self._thread.start()
        log.info("Mock lidar started (simulated LD19, 10 Hz)")
        return self

    def _run(self):
        last = time.monotonic_ns()
        while not self._stop.is_set():
            time.sleep(0.005)
            now = time.monotonic_ns()
            for raw, ta in self.sim.packets_between(last, now):
                for pkt in self.parser.feed(raw, ta):
                    if self.packet_sink:
                        self.packet_sink(pkt.raw, pkt.t_arrival_ns)
                    self.processor.add(pkt)
            last = now

    def wait_for_data(self, timeout_s=None):
        t0 = time.monotonic()
        while self.buffer.n_scans() < 1:
            if time.monotonic() - t0 > 3:
                raise RuntimeError("Mock lidar produced no scan")
            time.sleep(0.02)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def stats(self):
        return {"packets_ok": self.parser.n_ok, "crc_errors": self.parser.n_crc_err,
                "bytes": self.parser.n_bytes, "scans": self.buffer.n_scans(),
                "scan_hz": self.buffer.last_speed_dps / 360.0}


class MockCamera:
    """Same interface as PiCamera. Renders the simulated scene at the current time."""

    def __init__(self, cfg: Dict, sim: Optional[Simulator] = None, fps: float = 15.0):
        self.sim = sim or Simulator(cfg)
        self.index = 0
        self.period = 1.0 / fps
        self._last = 0.0

    def start(self):
        log.info("Mock camera started (synthetic room, %dx%d)", self.sim.w, self.sim.h)
        return self

    def capture(self) -> Frame:
        wait = self._last + self.period - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        t = time.monotonic_ns()
        img, z = self.sim.render(t)
        self.index += 1
        return Frame(img, t, self.index, {"mock": True}, truth_depth=z)

    def stop(self):
        pass
