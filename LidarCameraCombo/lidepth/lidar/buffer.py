"""Thread-safe storage of lidar points: assembles packets into 360° scans and answers
"which points were measured around time t?" using each point's own timestamp."""
from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class Scan:
    """One full revolution. All arrays have the same length."""
    t_ns: np.ndarray        # int64, per-point capture time (host monotonic clock)
    angle_deg: np.ndarray   # 0..360, clockwise
    range_m: np.ndarray
    intensity: np.ndarray

    @property
    def t_start_ns(self) -> int:
        return int(self.t_ns.min()) if len(self.t_ns) else 0

    @property
    def t_end_ns(self) -> int:
        return int(self.t_ns.max()) if len(self.t_ns) else 0

    def __len__(self) -> int:
        return len(self.t_ns)


def _empty_scan() -> Scan:
    return Scan(np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))


def _concat(parts: List[Scan]) -> Scan:
    if not parts:
        return _empty_scan()
    return Scan(
        np.concatenate([p.t_ns for p in parts]),
        np.concatenate([p.angle_deg for p in parts]),
        np.concatenate([p.range_m for p in parts]),
        np.concatenate([p.intensity for p in parts]),
    )


class LidarBuffer:
    """Ring buffer of the most recent complete scans + the scan being built."""

    def __init__(self, max_scans: int = 30):
        self._lock = threading.Lock()
        self._scans: Deque[Scan] = collections.deque(maxlen=max_scans)
        self._partial: List[Scan] = []
        self._last_start_angle: Optional[float] = None
        self.n_points_total = 0
        self.last_packet_ns = 0
        self.last_speed_dps = 0.0

    def add_packet_points(self, start_angle_deg: float, t_ns, angle_deg, range_m, intensity,
                          packet_time_ns: int = 0, speed_dps: float = 0.0) -> None:
        """Add the (already filtered) points of one packet.

        A new scan starts when the packet start angle wraps around (e.g. 359° → 1°).
        """
        piece = Scan(np.asarray(t_ns, np.int64), np.asarray(angle_deg, float),
                     np.asarray(range_m, float), np.asarray(intensity, float))
        with self._lock:
            if self._last_start_angle is not None and start_angle_deg < self._last_start_angle - 180.0:
                self._scans.append(_concat(self._partial))
                self._partial = []
            self._last_start_angle = start_angle_deg
            if len(piece):
                self._partial.append(piece)
            self.n_points_total += len(piece)
            self.last_packet_ns = packet_time_ns or self.last_packet_ns
            if speed_dps:
                self.last_speed_dps = speed_dps

    def latest_scan(self) -> Optional[Scan]:
        with self._lock:
            return self._scans[-1] if self._scans else None

    def scans(self) -> List[Scan]:
        with self._lock:
            return list(self._scans)

    def n_scans(self) -> int:
        with self._lock:
            return len(self._scans)

    def query(self, t_center_ns: int, tolerance_s: float, angle_bin_deg: float = 0.5) -> Scan:
        """Points whose OWN timestamp is within ±tolerance of ``t_center_ns``.

        If the window is longer than half a revolution the same direction can appear
        twice; for each angle bin we keep only the point closest in time to the frame.
        """
        tol_ns = int(tolerance_s * 1e9)
        lo, hi = t_center_ns - tol_ns, t_center_ns + tol_ns
        with self._lock:
            cands = [s for s in self._scans if len(s) and s.t_end_ns >= lo and s.t_start_ns <= hi]
            cands += [p for p in self._partial if len(p) and p.t_ns.max() >= lo and p.t_ns.min() <= hi]
            allp = _concat(cands)
        if len(allp) == 0:
            return allp
        m = (allp.t_ns >= lo) & (allp.t_ns <= hi)
        sel = Scan(allp.t_ns[m], allp.angle_deg[m], allp.range_m[m], allp.intensity[m])
        return dedupe_by_angle(sel, t_center_ns, angle_bin_deg)


def dedupe_by_angle(s: Scan, t_center_ns: int, angle_bin_deg: float) -> Scan:
    if len(s) == 0 or angle_bin_deg <= 0:
        return s
    bins = np.floor(s.angle_deg / angle_bin_deg).astype(np.int64)
    dt = np.abs(s.t_ns - t_center_ns)
    order = np.lexsort((dt, bins))              # sort by bin, then by |dt|
    b_sorted = bins[order]
    first = np.ones(len(order), bool)
    first[1:] = b_sorted[1:] != b_sorted[:-1]
    keep = np.sort(order[first])
    return Scan(s.t_ns[keep], s.angle_deg[keep], s.range_m[keep], s.intensity[keep])
