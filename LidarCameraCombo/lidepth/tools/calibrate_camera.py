"""`calibrate-camera`: OpenCV checkerboard calibration → calib/camera.yaml.

Live (Pi):   python -m lidepth calibrate-camera            (SPACE = grab, C = calibrate, Q = quit)
Headless:    python -m lidepth calibrate-camera --auto     (grabs a view every 2 s when the board is seen)
From files:  python -m lidepth calibrate-camera --images calib/camera_images
"""
from __future__ import annotations

import glob
import logging
import os
import time

import cv2
import numpy as np

from ..calibration import Intrinsics, save_intrinsics
from ..utils import has_display

log = logging.getLogger(__name__)


def find_corners(gray: np.ndarray, pattern):
    """Return refined corners (N,1,2) or None."""
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE |
                                                  cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return corners.astype(np.float32)
    ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH |
                                            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def calibrate(image_points, size, pattern, square_mm: float):
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)
    obj = [objp] * len(image_points)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, image_points, size, None, None)
    per = []
    for o, ip, rv, tv in zip(obj, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, D)
        per.append(float(np.sqrt(np.mean(np.sum((proj - ip) ** 2, axis=2)))))
    return rms, K, D.reshape(-1), per


def _finish(cfg, pts, size, pattern, names=None):
    cc = cfg["calibration"]
    if len(pts) < 5:
        raise RuntimeError(f"Only {len(pts)} usable views — need at least 5 (recommended {cc['min_images']}).")
    rms, K, D, per = calibrate(pts, size, pattern, float(cc["square_size_mm"]))
    intr = Intrinsics(size[0], size[1], float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                      [float(x) for x in D[:5]], float(rms))
    save_intrinsics(cc["camera_file"], intr, {
        "n_images": len(pts), "lens_position": cfg["camera"]["lens_position"],
        "square_size_mm": cc["square_size_mm"], "pattern_inner_corners": list(pattern)})
    print("\n=== Camera calibration ===")
    print(f"  views used      : {len(pts)}")
    print(f"  RMS reproj error: {rms:.3f} px   (good: < 0.5, OK: < 1.0)")
    print(f"  fx, fy          : {K[0, 0]:.1f}, {K[1, 1]:.1f}")
    print(f"  cx, cy          : {K[0, 2]:.1f}, {K[1, 2]:.1f}   (image centre {size[0] / 2:.0f}, {size[1] / 2:.0f})")
    print(f"  distortion      : {np.round(D[:5], 4).tolist()}")
    hfov = 2 * np.degrees(np.arctan(size[0] / 2 / K[0, 0]))
    vfov = 2 * np.degrees(np.arctan(size[1] / 2 / K[1, 1]))
    print(f"  field of view   : {hfov:.1f}° x {vfov:.1f}°")
    worst = np.argsort(per)[::-1][:3]
    print("  worst views     : " + ", ".join(f"{(names[i] if names else i)} ({per[i]:.2f}px)" for i in worst))
    if rms > 1.0:
        print("  WARNING: error is high. Delete blurry/bad views (see 'worst views') and re-run with --images.")
    print(f"Saved to {cc['camera_file']}")
    return intr


def run(cfg, args):
    cc = cfg["calibration"]
    pattern = (int(cc["checkerboard_cols"]), int(cc["checkerboard_rows"]))
    if args.images:
        files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                       glob.glob(os.path.join(args.images, "*.png")))
        if not files:
            raise RuntimeError(f"No .jpg/.png images in {args.images}")
        pts, names, size = [], [], None
        for fpath in files:
            img = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            c = find_corners(gray, pattern)
            if c is None:
                log.warning("No checkerboard found in %s", os.path.basename(fpath))
                continue
            pts.append(c)
            names.append(os.path.basename(fpath))
        c = cfg["camera"]
        if size != (int(c["width"]), int(c["height"])):
            log.warning("Images are %dx%d but camera.width/height are %sx%s — calibrate at the size "
                        "you will run!", size[0], size[1], c["width"], c["height"])
        return _finish(cfg, pts, size, pattern, names)

    from ..camera import PiCamera
    cam = PiCamera(cfg["camera"]).start()
    save_dir = os.path.join(os.path.dirname(cc["camera_file"]), "camera_images")
    os.makedirs(save_dir, exist_ok=True)
    show = has_display() and not args.auto
    pts, names, size = [], [], None
    last_grab = 0.0
    print(f"Checkerboard: {pattern[0]}x{pattern[1]} INNER corners, square {cc['square_size_mm']} mm.")
    print("Move the board around: centre, all 4 corners, near & far, tilted ±30°. Keep it sharp.")
    try:
        while True:
            fr = cam.capture()
            img = fr.image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            corners = find_corners(gray, pattern)
            grab = False
            if show:
                vis = img.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(vis, pattern, corners, True)
                cv2.putText(vis, f"views: {len(pts)}/{cc['min_images']}  SPACE=grab C=calibrate Q=quit",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("calibrate-camera", vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    return None
                if k == ord("c"):
                    break
                grab = k == ord(" ")
                if grab and corners is None:
                    print("  board not detected — not saved")
            else:
                grab = corners is not None and time.monotonic() - last_grab > 2.0
                if len(pts) >= int(cc["min_images"]) + 5:
                    break
            if grab and corners is not None:
                name = f"view_{len(pts):03d}.png"
                cv2.imwrite(os.path.join(save_dir, name), img)
                pts.append(corners)
                names.append(name)
                last_grab = time.monotonic()
                print(f"  grabbed {name} ({len(pts)} views)")
    finally:
        cam.stop()
        if show:
            cv2.destroyAllWindows()
    return _finish(cfg, pts, size, pattern, names)


def make_checkerboard(cfg, out_path: str) -> str:
    """Write a printable checkerboard PDF (A4 landscape) with exact square size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cc = cfg["calibration"]
    cols, rows = int(cc["checkerboard_cols"]) + 1, int(cc["checkerboard_rows"]) + 1   # squares
    sq = float(cc["square_size_mm"])
    page_w, page_h = 297.0, 210.0
    if cols * sq > page_w - 20 or rows * sq > page_h - 25:
        raise ValueError(f"{cols}x{rows} squares of {sq} mm do not fit on A4; reduce square_size_mm")
    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_aspect("equal")
    ax.axis("off")
    x0 = (page_w - cols * sq) / 2
    y0 = (page_h - rows * sq) / 2 + 5
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * sq, y0 + r * sq), sq, sq, color="black", lw=0))
    ax.text(page_w / 2, 8, f"{cols - 1}x{rows - 1} inner corners, squares {sq:.0f} mm — print at 100% "
            f"(no 'fit to page'), then MEASURE a square and put the value in config.yaml",
            ha="center", va="center", fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Checkerboard written to {out_path}. Tape it FLAT onto a stiff board (cardboard/clipboard).")
    return out_path
