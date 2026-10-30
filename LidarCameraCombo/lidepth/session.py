"""Recorded-session format (folder) — written on the Pi, read on the Pi or the laptop.

    sessions/2026-10-29_14-03-11/
        meta.yaml            config snapshot + info
        frames.csv           index, t_ns, file, exposure_us, lens_position
        frames/000001.jpg    camera images (raw, still distorted)
        lidar_packets.bin    every raw 47-byte packet, prefixed by int64 host arrival time (ns)
        truth/000001.npy     (mock sessions only) true depth for automatic evaluation
"""
from __future__ import annotations

import csv
import datetime as _dt
import logging
import os
import struct
import threading
from typing import Dict, List, Optional

import numpy as np
import yaml

from .lidar.buffer import LidarBuffer
from .lidar.driver import PacketProcessor
from .lidar.protocol import PACKET_LEN, parse_packet

log = logging.getLogger(__name__)

REC = struct.Struct("<q%ds" % PACKET_LEN)


def _clean_cfg(cfg: Dict) -> Dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


class SessionWriter:
    def __init__(self, root: str, cfg: Dict, name: Optional[str] = None):
        name = name or _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(root, name)
        os.makedirs(os.path.join(self.path, "frames"), exist_ok=True)
        self.cfg = cfg
        self._lock = threading.Lock()
        self._pk = open(os.path.join(self.path, "lidar_packets.bin"), "wb")
        self._fcsv_f = open(os.path.join(self.path, "frames.csv"), "w", newline="")
        self._fcsv = csv.writer(self._fcsv_f)
        self._fcsv.writerow(["index", "t_ns", "file", "exposure_us", "lens_position"])
        self.n_packets = 0
        self.n_frames = 0
        ext = str(cfg["camera"].get("save_format", "jpg")).lower()
        self.ext = "png" if ext == "png" else "jpg"
        with open(os.path.join(self.path, "meta.yaml"), "w") as f:
            yaml.safe_dump({"created": _dt.datetime.now().isoformat(timespec="seconds"),
                            "mock": bool(cfg.get("mock")), "config": _clean_cfg(cfg)}, f,
                           sort_keys=False)
        for key in ("camera_file", "extrinsics_file"):   # keep a copy of the calibration used
            src = cfg["calibration"][key]
            if os.path.exists(src):
                with open(src) as a, open(os.path.join(self.path, os.path.basename(src)), "w") as b:
                    b.write(a.read())

    def add_packet(self, raw: bytes, t_ns: int) -> None:
        with self._lock:
            self._pk.write(REC.pack(int(t_ns), raw))
            self.n_packets += 1

    def add_frame(self, frame, truth: Optional[np.ndarray] = None) -> str:
        import cv2
        fname = f"{frame.index:06d}.{self.ext}"
        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["camera"]["jpeg_quality"])] if self.ext == "jpg" else []
        cv2.imwrite(os.path.join(self.path, "frames", fname), frame.image, params)
        if truth is not None:
            os.makedirs(os.path.join(self.path, "truth"), exist_ok=True)
            np.save(os.path.join(self.path, "truth", f"{frame.index:06d}.npy"), truth.astype(np.float32))
        with self._lock:
            self._fcsv.writerow([frame.index, frame.t_ns, fname, frame.meta.get("exposure_us", ""),
                                 frame.meta.get("lens_position", "")])
            self._fcsv_f.flush()
            self.n_frames += 1
        return fname

    def close(self) -> None:
        with self._lock:
            self._pk.close()
            self._fcsv_f.close()


class SessionReader:
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Session folder not found: {path}")
        self.path = path
        with open(os.path.join(path, "meta.yaml")) as f:
            self.meta = yaml.safe_load(f) or {}
        self.frames: List[Dict] = []
        with open(os.path.join(path, "frames.csv"), newline="") as f:
            for row in csv.DictReader(f):
                self.frames.append({"index": int(row["index"]), "t_ns": int(row["t_ns"]),
                                    "file": row["file"]})
        if not self.frames:
            raise RuntimeError(f"Session {path} has no frames.")

    @property
    def is_mock(self) -> bool:
        return bool(self.meta.get("mock"))

    def image(self, i: int) -> np.ndarray:
        import cv2
        p = os.path.join(self.path, "frames", self.frames[i]["file"])
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image {p}")
        return img

    def truth(self, i: int) -> Optional[np.ndarray]:
        p = os.path.join(self.path, "truth", f"{self.frames[i]['index']:06d}.npy")
        return np.load(p) if os.path.exists(p) else None

    def iter_packets(self):
        p = os.path.join(self.path, "lidar_packets.bin")
        with open(p, "rb") as f:
            data = f.read()
        n = len(data) // REC.size
        if len(data) % REC.size:
            log.warning("lidar_packets.bin has a truncated last record (ignored)")
        for k in range(n):
            t_ns, raw = REC.unpack_from(data, k * REC.size)
            yield t_ns, raw

    def lidar_buffer(self, lidar_cfg: Dict) -> LidarBuffer:
        """Replay every recorded packet through the same code as the live driver."""
        buf = LidarBuffer(max_scans=10 ** 7)
        proc = PacketProcessor(lidar_cfg, buf)
        n_ok = n_bad = 0
        for t_ns, raw in self.iter_packets():
            pkt = parse_packet(raw)
            if pkt is None:
                n_bad += 1
                continue
            pkt.t_arrival_ns = t_ns
            proc.add(pkt)
            n_ok += 1
        log.info("Session lidar: %d packets (%d bad), %d scans", n_ok, n_bad, buf.n_scans())
        if n_ok == 0:
            raise RuntimeError("No valid lidar packets in this session — was the lidar running?")
        return buf
