"""FUTURE DRONE USE (disabled for the ground test).

Turns the metric depth map into the closest obstacle per horizontal sector and sends it to
ArduPilot as MAVLink OBSTACLE_DISTANCE (#330).  ArduPilot side: PRX1_TYPE = 2 (MAVLink),
SERIALx_PROTOCOL = 2 (MAVLink2) on the port wired to the Pi.

Sector convention (matches MAVLink, frame MAV_FRAME_BODY_FRD): sector i is centred on
angle_offset + i*increment degrees, measured CLOCKWISE from the vehicle's forward axis.
We assume the lidar's 0° = vehicle forward (true when the lidar is mounted facing forward).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

UINT16_MAX = 65535   # MAVLink: "unknown / no data in this sector"


def depth_to_sectors(depth_m: np.ndarray, K: np.ndarray, R_cam_from_lidar: np.ndarray,
                     cam_pos_lidar: np.ndarray, cfg: Dict,
                     confidence: Optional[np.ndarray] = None) -> np.ndarray:
    """Return (sectors,) distances in cm (uint16). Unknown sectors = 65535; sectors seen but
    empty = max_distance + 1 (MAVLink: "no obstacle")."""
    n_sec = int(cfg["sectors"])
    inc = 360.0 / n_sec
    h, w = depth_m.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if confidence is not None:
        valid &= confidence >= float(cfg.get("min_confidence", 0.0))
    x_c = (uu - K[0, 2]) / K[0, 0] * z
    y_c = (vv - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x_c, y_c, z], -1)[valid]                     # (N,3) camera frame
    p_l = pc @ R_cam_from_lidar + cam_pos_lidar                 # R^T (p_c) + camera position
    # vertical band relative to the CAMERA height
    dz = p_l[:, 2] - cam_pos_lidar[2]
    band = (dz <= float(cfg["band_up_m"])) & (dz >= -float(cfg["band_down_m"]))
    p_l = p_l[band]
    out = np.full(n_sec, UINT16_MAX, np.uint16)
    # Which sectors does the camera actually see? (azimuth of the image columns)
    if len(p_l) == 0:
        return out
    az_cw = np.degrees(np.arctan2(-p_l[:, 1], p_l[:, 0]))      # clockwise from forward
    dist = np.hypot(p_l[:, 0], p_l[:, 1])
    sec = np.mod(np.round(az_cw / inc), n_sec).astype(int)
    max_cm = int(cfg["max_distance_cm"])
    min_cm = int(cfg["min_distance_cm"])
    pct = float(cfg.get("percentile", 0))
    seen = np.unique(sec)
    for s in seen:
        dd = dist[sec == s]
        val = float(np.min(dd)) if pct <= 0 else float(np.percentile(dd, pct))
        cm = int(round(val * 100))
        if cm > max_cm:
            cm = max_cm + 1
        out[s] = max(cm, min_cm)
    return out


class MavlinkObstacleSender:
    """Sends OBSTACLE_DISTANCE via pymavlink. Only created when obstacle_output.enabled."""

    def __init__(self, cfg: Dict):
        try:
            from pymavlink import mavutil
        except ImportError as e:
            raise RuntimeError("pymavlink not installed: pip install pymavlink") from e
        self.cfg = cfg
        self.mavutil = mavutil
        port, baud = cfg["mavlink_port"], int(cfg["mavlink_baud"])
        log.info("Opening MAVLink on %s @ %d", port, baud)
        self.conn = mavutil.mavlink_connection(port, baud=baud, source_system=1,
                                               source_component=mavutil.mavlink.MAV_COMP_ID_OBSTACLE_AVOIDANCE)
        self.min_period = 1.0 / float(cfg.get("send_rate_hz", 10))
        self._last = 0.0
        self._t0 = time.monotonic()

    def send(self, sectors_cm: np.ndarray) -> bool:
        now = time.monotonic()
        if now - self._last < self.min_period:
            return False
        self._last = now
        n = len(sectors_cm)
        dist72 = [UINT16_MAX] * 72
        if n == 72:
            dist72 = [int(x) for x in sectors_cm]
            inc = 5
        else:   # MAVLink always carries 72 entries; extra ones stay "unknown"
            dist72[:min(n, 72)] = [int(x) for x in sectors_cm[:72]]
            inc = int(round(360 / n))
        m = self.mavutil.mavlink
        self.conn.mav.obstacle_distance_send(
            int((now - self._t0) * 1e6),          # time_usec
            m.MAV_DISTANCE_SENSOR_UNKNOWN,         # sensor_type (camera+lidar fusion)
            dist72,
            inc,                                   # increment (deg, uint8)
            int(self.cfg["min_distance_cm"]),
            int(self.cfg["max_distance_cm"]),
            float(360.0 / n),                      # increment_f (deg)
            0.0,                                   # angle_offset: sector 0 = straight ahead
            m.MAV_FRAME_BODY_FRD,
        )
        return True
