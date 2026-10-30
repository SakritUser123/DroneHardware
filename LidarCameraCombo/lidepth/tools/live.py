"""`live`: end-to-end on the Pi (or mock), with an on-screen view and FPS."""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

from ..depth_model import build_model
from ..pipeline import DepthPipeline
from ..utils import FpsCounter, build_sensors, has_display
from ..viz import compose_view

log = logging.getLogger(__name__)


def run(cfg, args) -> float:
    show = not args.no_display
    if show and not has_display():
        log.warning("No display found (SSH without X?) → running headless. Use VNC or the Pi's "
                    "desktop to see the window, or --save-dir to save views.")
        show = False
    cam, lidar, sim = build_sensors(cfg)
    model = build_model(cfg)
    if sim is not None:
        pipe = DepthPipeline(cfg, model, sim.intr, sim.ext)
    else:
        pipe = DepthPipeline(cfg, model)

    sender = None
    oc = cfg["obstacle_output"]
    if oc.get("enabled"):
        from ..output.obstacle import MavlinkObstacleSender, depth_to_sectors
        sender = MavlinkObstacleSender(oc)
        cam_pos = np.array([pipe.ext.cam_forward_m, pipe.ext.cam_left_m, pipe.ext.cam_up_m])
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    max_m = float(cfg["display"]["depth_max_m"])
    scale = float(cfg["display"]["window_scale"])
    radius = int(cfg["display"]["point_radius"])
    fpsc = FpsCounter(20)
    frame_times = []
    n = 0
    win = "lidepth live  (q = quit, s = save snapshot)"
    try:
        lidar.start()
        lidar.wait_for_data()
        cam.start()
        last_log = time.monotonic()
        while True:
            t0 = time.perf_counter()
            frame = cam.capture()
            res = pipe.process(frame.image, frame.t_ns, lidar.buffer, truth_depth=frame.truth_depth)
            if sender is not None:
                sectors = depth_to_sectors(res.fusion.depth_m, pipe.K, pipe.R, cam_pos, oc,
                                           res.fusion.confidence)
                sender.send(sectors)
            fps = fpsc.tick()
            frame_times.append(time.perf_counter() - t0)
            n += 1
            view = None
            if show or args.save_dir:
                view = compose_view(res, max_m, fps, radius)
            if args.save_dir and (n % max(1, int(args.save_every)) == 0):
                cv2.imwrite(os.path.join(args.save_dir, f"live_{n:05d}.jpg"), view)
            if show:
                if scale != 1.0:
                    view = cv2.resize(view, None, fx=scale, fy=scale)
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    d = args.save_dir or "."
                    np.save(os.path.join(d, f"depth_{stamp}.npy"), res.fusion.depth_m)
                    cv2.imwrite(os.path.join(d, f"view_{stamp}.jpg"), view)
                    log.info("Snapshot saved (depth_%s.npy)", stamp)
            if time.monotonic() - last_log > 2.0:
                f = res.fusion
                log.info("FPS %.2f | model %.0f ms | %s | inliers %d/%d | rel.err %s", fps,
                         res.timings.get("model", 0) * 1000, f.status, f.fit.n_inliers, f.fit.n_points,
                         f"{f.fit.rel_rmse * 100:.1f}%" if f.fit.ok else f.fit.reason)
                last_log = time.monotonic()
            if lidar.error:
                raise RuntimeError(lidar.error)
            if args.frames and n >= int(args.frames):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        lidar.stop()
        cam.stop()
        if show:
            cv2.destroyAllWindows()
    avg = (len(frame_times) - 1) / max(sum(frame_times[1:]), 1e-9) if len(frame_times) > 1 else 0.0
    print(f"\nMeasured end-to-end FPS: {avg:.2f} over {len(frame_times)} frames "
          f"(model input long side = {cfg['model']['input_long_side']} px)")
    return avg
