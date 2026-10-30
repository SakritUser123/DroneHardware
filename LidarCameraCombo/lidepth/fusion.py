"""Core algorithm: turn the model's RELATIVE inverse depth into METRIC depth using lidar.

Depth Anything V2 outputs d ≈ s·(1/Z) + c with unknown scale s and shift c.  Hence

        1 / Z_metric = a · d_model + b

We fit (a, b) on every frame from ALL lidar–pixel pairs with RANSAC (2-point minimal
samples) followed by a weighted least-squares refit on the inliers, then apply it to
every pixel.  Weights = Z so the residual is (approximately) the relative depth error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .projection import Projected, lidar_line_rows, sample_median

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    ok: bool
    a: float = float("nan")
    b: float = float("nan")
    n_points: int = 0
    n_inliers: int = 0
    rel_rmse: float = float("nan")      # RMS of |Zpred - Z| / Z on inliers
    mae_m: float = float("nan")         # mean |Zpred - Z| on inliers, metres
    reason: str = ""
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def _predict_z(a, b, d):
    den = a * d + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 1.0 / den, np.inf)


def _wls(d: np.ndarray, z: np.ndarray):
    """Minimise Σ (z_i (a d_i + b) - 1)²  ⇔  relative error in depth."""
    A = np.stack([z * d, z], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.ones_like(z), rcond=None)
    return float(sol[0]), float(sol[1])


def fit_inverse_depth(d: np.ndarray, z: np.ndarray, cfg: Dict,
                      rng: Optional[np.random.Generator] = None) -> FitResult:
    """Robustly fit 1/z = a d + b.  ``cfg`` is the ``fusion`` section of the config."""
    d = np.asarray(d, float)
    z = np.asarray(z, float)
    valid = np.isfinite(d) & np.isfinite(z) & (z > 0)
    n_all = len(d)
    idx_valid = np.nonzero(valid)[0]
    d, z = d[valid], z[valid]
    n = len(d)
    res = FitResult(ok=False, n_points=n, inliers=np.zeros(n_all, bool))
    if n < int(cfg["min_points"]):
        res.reason = f"too few points ({n} < {cfg['min_points']})"
        return res
    lo, hi = np.percentile(d, [5, 95])
    scale = max(np.max(np.abs(d)), 1e-9)
    if (hi - lo) / scale < float(cfg["min_d_spread"]):
        res.reason = "model values too uniform (all lidar points at similar depth?)"
        return res

    rng = rng if rng is not None else np.random.default_rng(int(cfg.get("seed", 0)))
    iters = int(cfg["ransac_iters"])
    thr = float(cfg["inlier_rel_err"])
    y = 1.0 / z
    i1 = rng.integers(0, n, iters)
    i2 = rng.integers(0, n, iters)
    dd = d[i1] - d[i2]
    good = (i1 != i2) & (np.abs(dd) > 1e-9 * scale)
    i1, i2, dd = i1[good], i2[good], dd[good]
    if len(i1) == 0:
        res.reason = "RANSAC could not draw distinct samples"
        return res
    a_c = (y[i1] - y[i2]) / dd
    b_c = y[i1] - a_c * d[i1]
    pos = a_c > 0
    a_c, b_c = a_c[pos], b_c[pos]
    if len(a_c) == 0:
        res.reason = "no RANSAC hypothesis with a > 0 (model output inverted? wrong extrinsics?)"
        return res
    zp = _predict_z(a_c[:, None], b_c[:, None], d[None, :])          # (H, n)
    rel = np.abs(zp - z[None, :]) / z[None, :]
    inl = rel < thr
    cnt = inl.sum(axis=1)
    cost = np.where(inl, rel, thr).sum(axis=1)                     # MSAC tie-break
    best = int(np.lexsort((cost, -cnt))[0])
    mask = inl[best]

    a, b = float(a_c[best]), float(b_c[best])
    for _ in range(2):                                             # refit on inliers, re-score
        if mask.sum() < 2:
            break
        a, b = _wls(d[mask], z[mask])
        relf = np.abs(_predict_z(a, b, d) - z) / z
        mask = relf < thr

    n_in = int(mask.sum())
    res.a, res.b, res.n_inliers = a, b, n_in
    full = np.zeros(n_all, bool)
    full[idx_valid[mask]] = True
    res.inliers = full
    if n_in < int(cfg["min_inliers"]):
        res.reason = f"too few inliers ({n_in} < {cfg['min_inliers']})"
        return res
    if not (a > 0):
        res.reason = f"fit has a <= 0 (a={a:.4g})"
        return res
    zp = _predict_z(a, b, d[mask])
    err = zp - z[mask]
    res.rel_rmse = float(np.sqrt(np.mean((err / z[mask]) ** 2)))
    res.mae_m = float(np.mean(np.abs(err)))
    res.ok = True
    return res


def apply_fit(d_map: np.ndarray, a: float, b: float, max_depth_m: float,
              min_depth_m: float = 0.05) -> np.ndarray:
    """Per-pixel metric depth Z (float32, metres); NaN where unknown."""
    den = a * d_map.astype(np.float64) + b
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / den
    bad = ~(den > 0) | ~np.isfinite(z) | (z > max_depth_m) | (z < min_depth_m)
    z[bad] = np.nan
    return z.astype(np.float32)


@dataclass
class FusionOutput:
    depth_m: np.ndarray              # (H, W) float32, NaN = unknown  (Z along optical axis)
    confidence: np.ndarray           # (H, W) float32 0..1
    status: str                      # "fit", "fallback", "none"
    a: float
    b: float
    fit: FitResult                   # this frame's raw fit (may have failed)
    pairs_d: np.ndarray              # model values at lidar pixels
    pairs_z: np.ndarray              # lidar Z at those pixels
    line_rows: np.ndarray            # (W,) lidar line row per column


class FusionEngine:
    """Per-frame fit + temporal EMA smoothing + fallback to the last good fit."""

    def __init__(self, fusion_cfg: Dict, max_range_m: float, sample_window: int = 5):
        self.cfg = fusion_cfg
        self.max_depth = float(max_range_m) * float(fusion_cfg["max_range_margin"])
        self.window = int(sample_window)
        self.rng = np.random.default_rng(int(fusion_cfg.get("seed", 0)))
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.last_quality = 0.0
        self.fail_count = 0

    def reset(self) -> None:
        self.a = self.b = None
        self.fail_count = 0

    def process(self, d_map: np.ndarray, proj: Projected) -> FusionOutput:
        h, w = d_map.shape
        d_samples = sample_median(d_map, proj.u, proj.v, self.window)
        fit = fit_inverse_depth(d_samples, proj.z_cam, self.cfg, self.rng)
        alpha = float(self.cfg["ema_alpha"])
        status = "none"
        if fit.ok:
            if self.a is None:
                self.a, self.b = fit.a, fit.b
            else:
                self.a = alpha * fit.a + (1 - alpha) * self.a
                self.b = alpha * fit.b + (1 - alpha) * self.b
            self.fail_count = 0
            self.last_quality = float(np.clip(1.0 - fit.rel_rmse / 0.3, 0.1, 1.0))
            status = "fit"
        elif self.a is not None and self.fail_count < int(self.cfg["max_fallback_frames"]):
            self.fail_count += 1
            status = "fallback"
            log.debug("fit failed (%s) → using last good fit", fit.reason)
        else:
            if self.a is not None:
                log.warning("No good fit for %d frames — resetting.", self.fail_count)
            self.reset()

        line = lidar_line_rows(proj.u, proj.v, w)
        if status == "none":
            depth = np.full((h, w), np.nan, np.float32)
            conf = np.zeros((h, w), np.float32)
            a = b = float("nan")
        else:
            a, b = float(self.a), float(self.b)
            depth = apply_fit(d_map, a, b, self.max_depth, float(self.cfg["min_depth_m"]))
            conf = self.confidence_map(line, h, w)
            q = self.last_quality * (0.5 if status == "fallback" else 1.0)
            conf *= q
            conf[np.isnan(depth)] = 0.0
        return FusionOutput(depth, conf, status, a, b, fit, d_samples, proj.z_cam, line)

    def confidence_map(self, line_rows: np.ndarray, h: int, w: int) -> np.ndarray:
        sigma = float(self.cfg["conf_sigma_px"])
        if np.all(np.isnan(line_rows)):
            return np.full((h, w), 0.2, np.float32)
        rows = np.arange(h, dtype=np.float32)[:, None]
        dv = rows - line_rows[None, :].astype(np.float32)
        return np.exp(-0.5 * (dv / sigma) ** 2).astype(np.float32)
