"""`calibrate-extrinsics`: line up projected lidar points with real edges, by keyboard.

Keys (click the window first):
    a / d   yaw  −/+          w / s   pitch +/−        z / x   roll −/+
    r / f   camera up +/−     t / g   forward +/−      y / h   left +/−
    [ / ]   smaller / bigger steps                     e       toggle edge view
    n / b   next / previous frame (when using --session)
    0       reset to config defaults     p or ENTER   SAVE     q / ESC   quit (asks to save)
"""
from __future__ import annotations

import copy
import logging
import math

import cv2
import numpy as np

from ..calibration import Extrinsics, load_extrinsics, save_extrinsics
from ..lidar.protocol import angle_to_xyz
from ..utils import build_sensors, has_display
from ..viz import distance_color, draw_points, legend_bar, put_lines

log = logging.getLogger(__name__)

HELP = ["a/d yaw  w/s pitch  z/x roll  r/f up  t/g fwd  y/h left",
        "[ ] step  e edges  n/b frame  0 reset  p save  q quit"]


def topdown(points_ang, points_rng, ext: Extrinsics, hfov_deg: float, max_m: float, size=240,
            upside_down=False):
    """Small bird's-eye view: lidar scan + camera field-of-view wedge (helps find yaw)."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    s = (size / 2 - 5) / max_m
    for rr in range(1, int(max_m) + 1):
        cv2.circle(img, (c, c), int(rr * s), (50, 50, 50), 1)
    if len(points_ang):
        p = angle_to_xyz(points_ang, points_rng, upside_down)
        for (x, y, _), r in zip(p, points_rng):
            px, py = int(c - y * s), int(c - x * s)     # forward = up, left = left
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(img, (px, py), 1, distance_color(r, max_m), -1)
    yaw = math.radians(-ext.yaw_deg if not upside_down else ext.yaw_deg)
    for side in (-1, 1):
        a = yaw + side * math.radians(hfov_deg / 2)
        cv2.line(img, (c, c), (int(c - math.sin(a) * size), int(c - math.cos(a) * size)), (0, 255, 255), 1)
    cv2.circle(img, (c, c), 3, (255, 255, 255), -1)
    put_lines(img, ["top view (fwd = up)"], (4, 14), 0.4)
    return img


def run(cfg, args):
    if not has_display():
        raise RuntimeError("calibrate-extrinsics needs a screen (use the Pi desktop or VNC, or run it on "
                           "the laptop with --session on a recorded session).")
    from ..pipeline import DepthPipeline
    ext_file = cfg["calibration"]["extrinsics_file"]
    reader = buf = cam = lidar = None
    if args.session:
        from ..session import SessionReader
        from .common import session_calibration
        reader = SessionReader(args.session)
        intr, ext = session_calibration(cfg, reader)
        buf = reader.lidar_buffer(cfg["lidar"])
    else:
        cam, lidar, sim = build_sensors(cfg)
        intr = sim.intr if sim else None
        ext = sim.ext if sim else load_extrinsics(ext_file, cfg["extrinsics_default"])
        lidar.start()
        lidar.wait_for_data()
        cam.start()
        buf = lidar.buffer
    pipe = DepthPipeline(cfg, model=None, intrinsics=intr, extrinsics=ext)
    defaults = Extrinsics(**{k: float(v) for k, v in cfg["extrinsics_default"].items()})
    ext = copy.deepcopy(pipe.ext)
    step = 1.0
    edges = False
    idx = 0
    max_m = float(cfg["display"]["depth_max_m"])
    hfov = 2 * math.degrees(math.atan(pipe.w / 2 / pipe.K[0, 0]))
    win = "calibrate-extrinsics"
    saved = True                  # nothing to save until something changes
    try:
        while True:
            if reader is not None:
                fr = reader.frames[idx]
                raw, t_ns = reader.image(idx), fr["t_ns"]
            else:
                f = cam.capture()
                raw, t_ns = f.image, f.t_ns
            img = pipe.undist.apply(raw)
            pipe.set_extrinsics(ext)
            pts = pipe.sync_points(buf, t_ns)
            proj = pipe.project(pts)
            base = img
            if edges:
                e = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 150)
                base = (img * 0.35).astype(np.uint8)
                base[e > 0] = (255, 255, 255)
            vis = draw_points(base, proj.u, proj.v, proj.z_cam, max_m, int(cfg["display"]["point_radius"]))
            put_lines(vis, [f"yaw {ext.yaw_deg:+.2f}  pitch {ext.pitch_deg:+.2f}  roll {ext.roll_deg:+.2f} deg",
                            f"up {ext.cam_up_m * 100:+.1f}  fwd {ext.cam_forward_m * 100:+.1f}  "
                            f"left {ext.cam_left_m * 100:+.1f} cm   step x{step:g}",
                            f"points in view: {len(proj)}" + ("" if saved else "   [unsaved changes]")] + HELP,
                      (8, 20), 0.45)
            side = np.zeros((vis.shape[0], 240, 3), np.uint8)
            td = topdown(pts.angle_deg if len(pts) else np.zeros(0), pts.range_m if len(pts) else np.zeros(0),
                         ext, hfov, max_m, 240, bool(cfg["lidar"]["upside_down"]))
            side[:240] = td
            view = np.hstack([vis, side, legend_bar(vis.shape[0], max_m)])
            cv2.imshow(win, view)
            k = cv2.waitKey(30 if reader is not None else 1) & 0xFF
            if k == 255:
                continue
            ch = chr(k) if k < 128 else ""
            if ch in "adwszxrftgyh0":
                saved = False             # something changed since the last save
            deg, cm = 0.5 * step, 0.005 * step
            if ch == "a": ext.yaw_deg -= deg
            elif ch == "d": ext.yaw_deg += deg
            elif ch == "w": ext.pitch_deg += deg
            elif ch == "s": ext.pitch_deg -= deg
            elif ch == "z": ext.roll_deg -= deg
            elif ch == "x": ext.roll_deg += deg
            elif ch == "r": ext.cam_up_m += cm
            elif ch == "f": ext.cam_up_m -= cm
            elif ch == "t": ext.cam_forward_m += cm
            elif ch == "g": ext.cam_forward_m -= cm
            elif ch == "y": ext.cam_left_m += cm
            elif ch == "h": ext.cam_left_m -= cm
            elif ch == "[": step = max(0.1, step / 2)
            elif ch == "]": step = min(20.0, step * 2)
            elif ch == "e": edges = not edges
            elif ch == "0": ext = copy.deepcopy(defaults)
            elif ch == "n" and reader is not None: idx = min(idx + 1, len(reader.frames) - 1)
            elif ch == "b" and reader is not None: idx = max(idx - 1, 0)
            elif ch == "p" or k == 13:
                save_extrinsics(ext_file, ext)
                saved = True
                print(f"Saved {ext_file}: {ext}")
            elif ch == "q" or k == 27:
                if not saved:
                    try:
                        ans = input("Save before quitting? [y/N] ").strip().lower()
                    except EOFError:
                        ans = "n"
                    if ans == "y":
                        save_extrinsics(ext_file, ext)
                        print(f"Saved {ext_file}")
                break
    finally:
        if lidar is not None:
            lidar.stop()
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()
    return ext
