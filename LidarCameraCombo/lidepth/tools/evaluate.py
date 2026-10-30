"""`evaluate`: compare metric depth with tape-measured ground truth.

Interactive:  python -m lidepth evaluate SESSION
    click a point on an object → type its tape-measured distance in cm → ENTER
    BACKSPACE edits, U = undo last point, N/B = next/previous frame, Q = finish
From a file:  python -m lidepth evaluate SESSION --points-csv my_points.csv
    (columns: frame,u,v,distance_cm  — frame = the number in the image file name)
Mock session: python -m lidepth evaluate SESSION --auto-gt 40   (uses the simulator's truth)

Distances are measured from the CAMERA LENS straight to the clicked spot
(evaluate.gt_type = range). Set gt_type = z if you measured only the forward distance.
"""
from __future__ import annotations

import csv
import logging
import os

import cv2
import numpy as np

from ..calibration import Undistorter
from ..projection import pixel_rays
from ..session import SessionReader
from ..utils import has_display
from .common import session_calibration

log = logging.getLogger(__name__)


def _load_processed(out_dir: str, frame_index: int):
    stem = f"{frame_index:06d}"
    dp = os.path.join(out_dir, "depth", stem + ".npy")
    cp = os.path.join(out_dir, "confidence", stem + ".npz")
    if not os.path.exists(dp):
        return None, None, None
    depth = np.load(dp)
    conf = line = None
    if os.path.exists(cp):
        z = np.load(cp)
        conf, line = z["confidence"].astype(np.float32), z["line_rows"]
    return depth, conf, line


def _pred_at(depth: np.ndarray, u: int, v: int, r: int = 1) -> float:
    h, w = depth.shape
    patch = depth[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
    with np.errstate(all="ignore"):
        return float(np.nanmedian(patch)) if np.isfinite(patch).any() else float("nan")


def _collect_interactive(reader, undist, out_dir):
    pts = []                 # (frame_index, u, v, cm)
    i = 0
    pending = None
    typed = ""
    state = {"click": None}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "evaluate (click, type cm, ENTER | U undo | N/B frame | Q done)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        fr = reader.frames[i]
        img = undist.apply(reader.image(i))
        depth, _, line = _load_processed(out_dir, fr["index"])
        vis = img.copy()
        if line is not None and not np.all(np.isnan(line)):
            poly = np.stack([np.arange(len(line)), np.nan_to_num(line)], 1).astype(np.int32)
            cv2.polylines(vis, [poly], False, (0, 255, 255), 1)
        for (fi, u, v, cm) in pts:
            if fi == fr["index"]:
                cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)
                cv2.putText(vis, f"{cm:.0f}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if state["click"] is not None:
            pending = state["click"]
            state["click"] = None
            typed = ""
        if pending is not None:
            cv2.drawMarker(vis, pending, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(vis, f"distance cm: {typed}_", (8, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"frame {fr['index']}  ({i + 1}/{len(reader.frames)})  points: {len(pts)}"
                    + ("" if depth is not None else "  [not processed]"), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        if pending is not None and (chr(k).isdigit() or chr(k) == "."):
            typed += chr(k)
        elif pending is not None and k in (8, 127):
            typed = typed[:-1]
        elif pending is not None and k in (13, 10):
            try:
                pts.append((fr["index"], int(pending[0]), int(pending[1]), float(typed)))
                print(f"  frame {fr['index']} pixel {pending} = {float(typed):.1f} cm")
            except ValueError:
                print("  type a number, e.g. 152")
            pending, typed = None, ""
        elif k == 27 and pending is not None:
            pending, typed = None, ""
        elif chr(k).lower() == "u" and pts:
            print(f"  removed {pts.pop()}")
        elif chr(k).lower() == "n":
            i, pending = min(i + 1, len(reader.frames) - 1), None
        elif chr(k).lower() == "b":
            i, pending = max(i - 1, 0), None
        elif chr(k).lower() == "q" or k == 27:
            break
    cv2.destroyAllWindows()
    return pts


def _auto_gt(reader, out_dir, n_per_frame, gt_type, K, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for i, fr in enumerate(reader.frames):
        truth = reader.truth(i)
        depth, _, _ = _load_processed(out_dir, fr["index"])
        if truth is None or depth is None:
            continue
        h, w = truth.shape
        rays = pixel_rays(K, w, h)
        for _ in range(n_per_frame):
            u, v = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
            z = float(truth[v, u])
            if not np.isfinite(z):
                continue
            gt = z * rays[v, u] if gt_type == "range" else z
            pts.append((fr["index"], u, v, gt * 100.0))
    return pts


def run(cfg, args):
    reader = SessionReader(args.session)
    out_dir = args.processed or os.path.join(reader.path, "processed")
    if not os.path.isdir(os.path.join(out_dir, "depth")):
        print("No processed results yet → running `process` first…")
        from types import SimpleNamespace
        from . import process
        process.run(cfg, SimpleNamespace(session=reader.path, out=out_dir, every=1, max_frames=None,
                                         no_viz=False, save_model_output=False,
                                         session_calib=args.session_calib))
    intr, _ = session_calibration(cfg, reader, args.session_calib)
    undist = Undistorter(intr)
    K = undist.K_new
    gt_type = str(cfg["evaluate"]["gt_type"]).lower()
    near_px = float(cfg["evaluate"]["near_line_px"])

    if args.points_csv:
        pts = []
        with open(args.points_csv, newline="") as f:
            for row in csv.DictReader(f):
                pts.append((int(row["frame"]), int(float(row["u"])), int(float(row["v"])),
                            float(row["distance_cm"])))
    elif args.auto_gt:
        if not reader.is_mock:
            raise RuntimeError("--auto-gt only works on mock sessions (they contain true depth).")
        pts = _auto_gt(reader, out_dir, int(args.auto_gt), gt_type, K)
    else:
        if not has_display():
            raise RuntimeError("Interactive evaluate needs a screen. Use --points-csv on a headless Pi, "
                               "or copy the session to the laptop.")
        pts = _collect_interactive(reader, undist, out_dir)
    if not pts:
        print("No ground-truth points — nothing to evaluate.")
        return None

    eval_dir = os.path.join(out_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    rows = []
    rays = None
    for fi, u, v, cm in pts:
        depth, conf, line = _load_processed(out_dir, fi)
        if depth is None:
            log.warning("frame %d not processed — skipped", fi)
            continue
        if rays is None:
            rays = pixel_rays(K, depth.shape[1], depth.shape[0])
        z = _pred_at(depth, u, v)
        pred_m = z * rays[v, u] if gt_type == "range" else z
        gt_m = cm / 100.0
        dv = abs(v - float(line[u])) if line is not None and np.isfinite(line[u]) else float("nan")
        near = bool(np.isfinite(dv) and dv <= near_px)
        err = pred_m - gt_m
        rows.append({"frame": fi, "u": u, "v": v, "gt_m": gt_m, "pred_m": pred_m,
                     "err_m": err, "abs_err_m": abs(err), "pct_err": 100 * abs(err) / gt_m,
                     "rows_from_lidar_line": dv, "group": "near" if near else "far",
                     "confidence": float(conf[v, u]) if conf is not None else float("nan")})
    with open(os.path.join(eval_dir, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(eval_dir, "gt_points.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "u", "v", "distance_cm"])
        wr.writerows(pts)

    def stats(sel):
        valid = [r for r in sel if np.isfinite(r["pred_m"])]
        if not valid:
            return {"n": len(sel), "n_valid": 0}
        ae = np.array([r["abs_err_m"] for r in valid])
        pe = np.array([r["pct_err"] for r in valid])
        return {"n": len(sel), "n_valid": len(valid), "MAE_cm": 100 * ae.mean(),
                "median_AE_cm": 100 * np.median(ae), "mean_pct": pe.mean(), "median_pct": np.median(pe),
                "within_5pct": 100 * np.mean(pe <= 5), "within_10pct": 100 * np.mean(pe <= 10)}

    groups = {"near lidar line": [r for r in rows if r["group"] == "near"],
              "far from line": [r for r in rows if r["group"] == "far"], "all": rows}
    print(f"\n=== Evaluation ({gt_type}, near = within {near_px:.0f} rows of the lidar line) ===")
    print(f"{'group':18s} {'n':>4s} {'valid':>5s} {'MAE cm':>8s} {'medAE cm':>9s} {'mean %':>7s} "
          f"{'med %':>6s} {'<=5%':>6s} {'<=10%':>6s}")
    summary = {}
    for g, sel in groups.items():
        s = stats(sel)
        summary[g] = s
        if s.get("n_valid"):
            print(f"{g:18s} {s['n']:4d} {s['n_valid']:5d} {s['MAE_cm']:8.1f} {s['median_AE_cm']:9.1f} "
                  f"{s['mean_pct']:7.1f} {s['median_pct']:6.1f} {s['within_5pct']:5.0f}% {s['within_10pct']:5.0f}%")
        else:
            print(f"{g:18s} {s['n']:4d}     0   (no valid predictions)")
    with open(os.path.join(eval_dir, "summary.csv"), "w", newline="") as f:
        keys = ["n", "n_valid", "MAE_cm", "median_AE_cm", "mean_pct", "median_pct", "within_5pct", "within_10pct"]
        wr = csv.writer(f)
        wr.writerow(["group"] + keys)
        for g, s in summary.items():
            wr.writerow([g] + [s.get(k, "") for k in keys])
    _plot(rows, os.path.join(eval_dir, "evaluation.png"), near_px)
    print(f"Saved results.csv, summary.csv and evaluation.png in {eval_dir}")
    return summary


def _plot(rows, path, near_px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [r for r in rows if np.isfinite(r["pred_m"])]
    if not valid:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"near": "#1f77b4", "far": "#ff7f0e"}
    for g in ("near", "far"):
        sel = [r for r in valid if r["group"] == g]
        if sel:
            ax[0].scatter([r["gt_m"] for r in sel], [r["pred_m"] for r in sel], s=18, c=colors[g],
                          label=f"{g} line ({len(sel)})", alpha=0.8)
            ax[1].scatter([r["rows_from_lidar_line"] for r in sel], [r["pct_err"] for r in sel], s=18,
                          c=colors[g], alpha=0.8)
    m = max(max(r["gt_m"] for r in valid), max(r["pred_m"] for r in valid)) * 1.05
    ax[0].plot([0, m], [0, m], "k--", lw=1, label="perfect")
    ax[0].set_xlabel("tape-measured distance (m)")
    ax[0].set_ylabel("predicted distance (m)")
    ax[0].set_title("Predicted vs measured")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].axvline(near_px, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("rows above/below the lidar line (px)")
    ax[1].set_ylabel("absolute error (%)")
    ax[1].set_title("Error vs distance from the lidar line")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
