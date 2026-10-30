"""Lidar points → camera pixels, and robust sampling of the model at those pixels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lidar.protocol import angle_to_xyz


@dataclass
class Projected:
    u: np.ndarray        # pixel column (float)
    v: np.ndarray        # pixel row (float)
    z_cam: np.ndarray    # depth along the optical axis, metres  (the "Z" in 1/Z = a d + b)
    range_m: np.ndarray  # original lidar range
    idx: np.ndarray      # index into the input point arrays

    def __len__(self) -> int:
        return len(self.u)


def lidar_to_camera(points_lidar: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(N,3) lidar-frame points → (N,3) camera optical frame (x right, y down, z forward)."""
    return points_lidar @ R.T + t


def project_points(points_cam: np.ndarray, K: np.ndarray):
    """Pinhole projection of camera-frame points (no distortion: image already undistorted)."""
    z = points_cam[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * points_cam[:, 0] / zs + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / zs + K[1, 2]
    return u, v, z


def project_scan(angle_deg: np.ndarray, range_m: np.ndarray, R: np.ndarray, t: np.ndarray,
                 K: np.ndarray, width: int, height: int, min_z: float = 0.05,
                 border_px: int = 0, upside_down: bool = False) -> Projected:
    """Full chain: (angle, range) → lidar xyz → camera xyz → pixel; keep visible points only."""
    angle_deg = np.asarray(angle_deg, float)
    range_m = np.asarray(range_m, float)
    if len(angle_deg) == 0:
        e = np.zeros(0)
        return Projected(e, e, e, e, np.zeros(0, int))
    p_l = angle_to_xyz(angle_deg, range_m, upside_down)
    p_c = lidar_to_camera(p_l, R, t)
    u, v, z = project_points(p_c, K)
    b = border_px
    keep = (z > min_z) & (u >= b) & (u <= width - 1 - b) & (v >= b) & (v <= height - 1 - b)
    idx = np.nonzero(keep)[0]
    return Projected(u[keep], v[keep], z[keep], range_m[keep], idx)


def sample_median(img: np.ndarray, u: np.ndarray, v: np.ndarray, window: int = 5) -> np.ndarray:
    """Median of ``img`` in a window×window patch around each (u, v). NaNs are ignored."""
    h, w = img.shape[:2]
    r = max(int(window) // 2, 0)
    ui = np.rint(u).astype(int)
    vi = np.rint(v).astype(int)
    if len(ui) == 0:
        return np.zeros(0, np.float64)
    offs = np.arange(-r, r + 1)
    du, dv = np.meshgrid(offs, offs)
    uu = np.clip(ui[:, None] + du.ravel()[None, :], 0, w - 1)
    vv = np.clip(vi[:, None] + dv.ravel()[None, :], 0, h - 1)
    patches = img[vv, uu].astype(np.float64)
    with np.errstate(all="ignore"):
        if np.isnan(patches).any():
            return np.nanmedian(patches, axis=1)
        return np.median(patches, axis=1)


def lidar_line_rows(u: np.ndarray, v: np.ndarray, width: int) -> np.ndarray:
    """Row of the lidar scan line for every image column (linear interpolation between
    projected points, constant extrapolation at the ends). NaN if there are no points."""
    if len(u) == 0:
        return np.full(width, np.nan)
    order = np.argsort(u)
    us, vs = u[order], v[order]
    # median-smooth a little so a single bad point doesn't bend the line
    if len(vs) >= 5:
        k = 2
        pad = np.pad(vs, k, mode="edge")
        vs = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1), axis=1)
    return np.interp(np.arange(width), us, vs)


def pixel_rays(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """(H, W) factor sqrt(x²+y²+1): Euclidean range = Z * factor."""
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    x = (uu - K[0, 2]) / K[0, 0]
    y = (vv - K[1, 2]) / K[1, 1]
    return np.sqrt(x * x + y * y + 1.0)
