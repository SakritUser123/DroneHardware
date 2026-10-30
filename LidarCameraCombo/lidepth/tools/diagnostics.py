"""`lidar-test` (is the lidar wired correctly?) and `benchmark` (how fast is the model?)."""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def lidar_test(cfg, args):
    from ..utils import build_sensors
    _, lidar, _ = build_sensors(cfg)
    lidar.start()
    try:
        lidar.wait_for_data()
        t_end = time.monotonic() + float(args.seconds)
        while time.monotonic() < t_end:
            time.sleep(1.0)
            st = lidar.stats()
            s = lidar.buffer.latest_scan()
            if s is None or len(s) == 0:
                print("no complete scan yet…")
                continue
            front = (s.angle_deg < 15) | (s.angle_deg > 345)
            i = int(np.argmin(s.range_m))
            print(f"scan {st['scan_hz']:4.1f} Hz | {len(s):3d} pts/scan | packets {st['packets_ok']} "
                  f"CRC errors {st['crc_errors']} | straight ahead (±15°): "
                  f"{np.median(s.range_m[front]) if front.any() else float('nan'):.2f} m | "
                  f"closest {s.range_m[i]:.2f} m at {s.angle_deg[i]:.0f}°")
        if args.save:
            import cv2
            from .tune_extrinsics import topdown
            from ..calibration import Extrinsics
            s = lidar.buffer.latest_scan()
            img = topdown(s.angle_deg, s.range_m, Extrinsics(), 0.0, float(cfg["display"]["depth_max_m"]),
                          480, bool(cfg["lidar"]["upside_down"]))
            cv2.imwrite(args.save, img)
            print(f"Top view of the last scan saved to {args.save} (forward = up)")
        print("\nTip: hold a box ~1 m in front of the lidar's FRONT (0° mark) — 'straight ahead' should read ~1.0 m.")
    finally:
        lidar.stop()


def benchmark(cfg, args):
    from ..depth_model import OnnxDepthModel
    from ..fusion import FusionEngine
    from ..projection import Projected
    m = cfg["model"]
    w, h = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])
    img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    sizes = [int(s) for s in (args.sizes or [m["input_long_side"]])]
    print(f"Benchmark on {w}x{h} images, {m['threads']} threads, model {m['path']}")
    print(f"{'long side':>9s} {'net input':>10s} {'model ms':>9s} {'model FPS':>9s}")
    results = {}
    for s in sizes:
        model = OnnxDepthModel(m["path"], s, int(m["threads"]))
        mw, mh = model.input_size(w, h)
        model.predict(img)                        # warm-up
        ts = []
        for _ in range(int(args.iters)):
            t0 = time.perf_counter()
            model.predict(img)
            ts.append(time.perf_counter() - t0)
        ms = 1000 * float(np.median(ts))
        results[s] = ms
        print(f"{s:9d} {mw:4d}x{mh:<5d} {ms:9.0f} {1000 / ms:9.2f}")
    # fusion cost with ~120 synthetic points
    fe = FusionEngine(cfg["fusion"], 12.0)
    rng = np.random.default_rng(1)
    u = rng.uniform(0, w - 1, 120)
    v = np.full(120, h * 0.6)
    z = rng.uniform(0.5, 6, 120)
    d = np.tile((1.0 / z.mean() * 5 + 0.1), (h, w)).astype(np.float32)
    d[np.rint(v).astype(int), np.rint(u).astype(int)] = (5 / z + 0.1)
    proj = Projected(u, v, z, z, np.arange(120))
    t0 = time.perf_counter()
    for _ in range(10):
        fe.process(d, proj)
    print(f"fusion + depth map: {100 * (time.perf_counter() - t0):.0f} ms per frame")
    print("\nEnd-to-end FPS ≈ 1 / (model + fusion + capture/display ~ 30–60 ms). "
          "Lower model.input_long_side for speed (see README).")
    return results
