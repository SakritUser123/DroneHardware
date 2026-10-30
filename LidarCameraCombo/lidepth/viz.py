"""Display helpers (colours are for humans only — never used for math)."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def colorize_depth(depth_m: np.ndarray, max_m: float, min_m: float = 0.0) -> np.ndarray:
    """Metric depth → BGR (TURBO: red = near, blue = far). Unknown pixels are dark grey."""
    z = np.nan_to_num(depth_m, nan=-1.0)
    t = np.clip((z - min_m) / max(max_m - min_m, 1e-6), 0, 1)
    img = cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(depth_m)] = (40, 40, 40)
    return img


def colorize_relative(d: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(d, [2, 98])
    t = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def distance_color(r: float, max_m: float):
    t = float(np.clip(r / max(max_m, 1e-6), 0, 1))
    c = cv2.applyColorMap(np.array([[int((1 - t) * 255)]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw_points(img: np.ndarray, u, v, dist, max_m: float, radius: int = 3,
                inliers: Optional[np.ndarray] = None) -> np.ndarray:
    out = img.copy()
    for k in range(len(u)):
        col = distance_color(dist[k], max_m)
        p = (int(round(u[k])), int(round(v[k])))
        cv2.circle(out, p, radius, col, -1, cv2.LINE_AA)
        edge = (255, 255, 255) if inliers is None or inliers[k] else (0, 0, 0)
        cv2.circle(out, p, radius, edge, 1, cv2.LINE_AA)
    return out


def legend_bar(height: int, max_m: float, width: int = 70, min_m: float = 0.0) -> np.ndarray:
    bar = np.zeros((height, width, 3), np.uint8)
    grad = np.linspace(min_m, max_m, height - 40)[:, None] * np.ones((1, 20))
    bar[20:height - 20, 6:26] = colorize_depth(grad, max_m, min_m)
    n = 6
    for i in range(n + 1):
        val = min_m + (max_m - min_m) * i / n
        y = int(20 + (height - 41) * i / n)
        cv2.line(bar, (26, y), (32, y), (255, 255, 255), 1)
        cv2.putText(bar, f"{val:.1f}", (34, y + 4), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, "m", (30, 14), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def put_lines(img: np.ndarray, lines: List[str], org=(8, 20), scale=0.5, color=(255, 255, 255)):
    x, y = org
    for ln in lines:
        cv2.putText(img, ln, (x, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += int(22 * scale / 0.5)
    return img


def draw_line_rows(img: np.ndarray, rows: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    if rows is None or np.all(np.isnan(rows)):
        return img
    pts = np.stack([np.arange(len(rows)), np.nan_to_num(rows)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    return img


def compose_view(res, max_m: float, fps: float, radius: int = 3) -> np.ndarray:
    """Left: camera + lidar points. Right: metric depth + legend. Bottom: stats."""
    f = res.fusion
    inl = f.fit.inliers if len(f.fit.inliers) == len(res.proj) else None
    left = draw_points(res.image, res.proj.u, res.proj.v, res.proj.z_cam, max_m, radius, inl)
    right = colorize_depth(res.fusion.depth_m, max_m)
    right = draw_line_rows(right, f.line_rows)
    h = left.shape[0]
    view = np.hstack([left, right, legend_bar(h, max_m)])
    put_lines(view, ["camera + lidar (white ring = fit inlier)"], (8, 18), 0.45)
    put_lines(view, ["metric depth (Z)"], (left.shape[1] + 8, 18), 0.45)
    fit = f.fit
    color = (80, 255, 80) if f.status == "fit" else ((0, 200, 255) if f.status == "fallback" else (80, 80, 255))
    stats = [
        f"FPS {fps:4.1f} | model {res.timings.get('model', 0) * 1000:5.0f} ms | "
        f"lidar pts in view {len(res.proj)}",
        f"status {f.status.upper()}  a={f.a:.4g}  b={f.b:.4g}  inliers {fit.n_inliers}/{fit.n_points}"
        + (f"  rel.err {fit.rel_rmse * 100:.1f}%  MAE {fit.mae_m * 100:.1f} cm" if fit.ok else f"  ({fit.reason})"),
    ]
    panel = np.zeros((52, view.shape[1], 3), np.uint8)
    put_lines(panel, stats[:1], (8, 20), 0.5)
    put_lines(panel, stats[1:], (8, 42), 0.5, color)
    return np.vstack([view, panel])
