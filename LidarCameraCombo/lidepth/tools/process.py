"""`process`: run model + fusion on a recorded session (Pi or laptop)."""
from __future__ import annotations

import csv
import logging
import os
import time

import cv2
import numpy as np
import yaml

from ..session import SessionReader
from ..utils import FpsCounter
from ..viz import compose_view
from .common import pipeline_for_session

log = logging.getLogger(__name__)

FIT_COLUMNS = ["frame", "t_ns", "status", "a", "b", "a_frame", "b_frame", "n_points",
               "n_inliers", "rel_rmse", "mae_m", "reason", "lidar_pts_synced", "model_ms", "total_ms"]


def run(cfg, args) -> str:
    reader = SessionReader(args.session)
    pipe = pipeline_for_session(cfg, reader, use_session_calib=args.session_calib)
    buf = reader.lidar_buffer(cfg["lidar"])
    out = args.out or os.path.join(reader.path, "processed")
    for sub in ("depth", "confidence", "viz"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    if args.save_model_output:
        os.makedirs(os.path.join(out, "model_raw"), exist_ok=True)
    with open(os.path.join(out, "camera_used.yaml"), "w") as f:
        K = pipe.K
        yaml.safe_dump({"note": "pinhole model of the UNDISTORTED images the depth maps refer to",
                        "width": pipe.w, "height": pipe.h, "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                        "extrinsics": {k: float(v) for k, v in vars(pipe.ext).items()}}, f, sort_keys=False)

    max_m = float(cfg["display"]["depth_max_m"])
    fpsc = FpsCounter(50)
    frames = list(range(0, len(reader.frames), max(1, int(args.every))))
    if args.max_frames:
        frames = frames[:int(args.max_frames)]
    n_ok = 0
    rel_errs = []
    total_times = []
    with open(os.path.join(out, "fit_log.csv"), "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(FIT_COLUMNS)
        for n, i in enumerate(frames):
            fr = reader.frames[i]
            t0 = time.perf_counter()
            res = pipe.process(reader.image(i), fr["t_ns"], buf, truth_depth=reader.truth(i))
            dt = time.perf_counter() - t0
            total_times.append(dt)
            fps = fpsc.tick()
            stem = f"{fr['index']:06d}"
            np.save(os.path.join(out, "depth", stem + ".npy"), res.fusion.depth_m)
            np.savez_compressed(os.path.join(out, "confidence", stem + ".npz"),
                                confidence=res.fusion.confidence.astype(np.float16),
                                line_rows=res.fusion.line_rows.astype(np.float32))
            if args.save_model_output:
                np.save(os.path.join(out, "model_raw", stem + ".npy"), res.d_model)
            if not args.no_viz:
                cv2.imwrite(os.path.join(out, "viz", stem + ".jpg"), compose_view(res, max_m, fps))
            f = res.fusion
            ft = f.fit
            wr.writerow([fr["index"], fr["t_ns"], f.status, f"{f.a:.6g}", f"{f.b:.6g}",
                         f"{ft.a:.6g}", f"{ft.b:.6g}", ft.n_points, ft.n_inliers,
                         f"{ft.rel_rmse:.4f}", f"{ft.mae_m:.4f}", ft.reason, len(res.points),
                         f"{res.timings.get('model', 0) * 1000:.1f}", f"{dt * 1000:.1f}"])
            if ft.ok:
                n_ok += 1
                rel_errs.append(ft.rel_rmse)
            if n % 10 == 0 or n == len(frames) - 1:
                log.info("frame %d/%d  %s  inliers %d/%d  rel.err %s  (%.0f ms)", n + 1, len(frames),
                         f.status, ft.n_inliers, ft.n_points,
                         f"{ft.rel_rmse * 100:.1f}%" if ft.ok else ft.reason, dt * 1000)
    summary = {
        "frames": len(frames), "frames_with_good_fit": n_ok,
        "median_fit_rel_rmse_pct": float(np.median(rel_errs) * 100) if rel_errs else None,
        "mean_processing_ms": float(np.mean(total_times) * 1000) if total_times else None,
        "processing_fps": float(1.0 / np.mean(total_times)) if total_times else None,
    }
    with open(os.path.join(out, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print("\n=== process summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Outputs in {out}\nNext: python -m lidepth evaluate {reader.path}")
    return out
