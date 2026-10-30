"""Command-line interface:  python -m lidepth <command> [options]

Commands (in the order you normally use them):
    lidar-test            check the lidar wiring / port / baud
    make-checkerboard     write a printable checkerboard PDF
    calibrate-camera      intrinsics → calib/camera.yaml
    calibrate-extrinsics  interactive lidar→camera alignment → calib/extrinsics.yaml
    record                save camera frames + raw lidar to sessions/<date>/
    process               offline: model + fusion on a session → metric depth .npy + plots
    evaluate              compare with tape-measured ground truth
    live                  everything in real time on the Pi
    benchmark             model speed at several input sizes
Add --mock to run without hardware.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .utils import setup_logging

log = logging.getLogger("lidepth")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set lidar.port=/dev/serial0 (repeatable)")
    common.add_argument("--mock", action="store_true", help="no hardware: simulated lidar + camera")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    p = argparse.ArgumentParser(prog="python -m lidepth", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("record", parents=[common], help="record frames + lidar to a session folder")
    s.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C; mock: 10)")
    s.add_argument("--fps", type=float, default=None, help="camera frames per second to save")
    s.add_argument("--name", default=None, help="session folder name (default: date_time)")

    s = sub.add_parser("process", parents=[common], help="offline processing of a recorded session")
    s.add_argument("session", help="path to sessions/<name>")
    s.add_argument("--out", default=None, help="output folder (default: <session>/processed)")
    s.add_argument("--every", type=int, default=1, help="process every N-th frame")
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--no-viz", action="store_true", help="don't write visualisation JPEGs")
    s.add_argument("--save-model-output", action="store_true", help="also save raw model output .npy")
    s.add_argument("--session-calib", action="store_true",
                   help="use the calibration copied into the session instead of calib/*.yaml")

    s = sub.add_parser("live", parents=[common], help="real-time view on the Pi")
    s.add_argument("--no-display", action="store_true", help="headless: log FPS only")
    s.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q/Ctrl+C)")
    s.add_argument("--save-dir", default=None, help="save views here")
    s.add_argument("--save-every", type=int, default=10, help="with --save-dir: save every N-th view")

    s = sub.add_parser("calibrate-camera", parents=[common], help="checkerboard intrinsic calibration")
    s.add_argument("--images", default=None, help="calibrate from a folder of images instead of live")
    s.add_argument("--auto", action="store_true", help="headless: grab views automatically every 2 s")

    s = sub.add_parser("calibrate-extrinsics", parents=[common], help="interactive lidar→camera tuning")
    s.add_argument("--session", default=None, help="tune on a recorded session instead of live sensors")

    s = sub.add_parser("evaluate", parents=[common], help="error vs tape-measured ground truth")
    s.add_argument("session")
    s.add_argument("--processed", default=None, help="processed folder (default: <session>/processed)")
    s.add_argument("--points-csv", default=None, help="CSV with frame,u,v,distance_cm (non-interactive)")
    s.add_argument("--auto-gt", type=int, default=0, help="mock sessions: N random truth points per frame")
    s.add_argument("--session-calib", action="store_true")

    s = sub.add_parser("make-checkerboard", parents=[common], help="printable checkerboard PDF")
    s.add_argument("--out", default="calib/checkerboard_A4.pdf")

    s = sub.add_parser("lidar-test", parents=[common], help="print live lidar statistics")
    s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--save", default=None, help="save a top-view PNG of the last scan")

    s = sub.add_parser("benchmark", parents=[common], help="time the depth model")
    s.add_argument("--sizes", nargs="*", type=int, default=None, help="e.g. --sizes 182 252 364 518")
    s.add_argument("--iters", type=int, default=5)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
    except (KeyError, ValueError) as e:
        print(f"Config error: {e}")
        return 2
    if args.mock:
        cfg["mock"] = True
    setup_logging("DEBUG" if args.verbose else cfg.get("log_level", "INFO"))
    try:
        if args.cmd == "record":
            from .tools import record
            record.run(cfg, args)
        elif args.cmd == "process":
            from .tools import process
            process.run(cfg, args)
        elif args.cmd == "live":
            from .tools import live
            live.run(cfg, args)
        elif args.cmd == "calibrate-camera":
            from .tools import calibrate_camera
            calibrate_camera.run(cfg, args)
        elif args.cmd == "calibrate-extrinsics":
            from .tools import tune_extrinsics
            tune_extrinsics.run(cfg, args)
        elif args.cmd == "evaluate":
            from .tools import evaluate
            evaluate.run(cfg, args)
        elif args.cmd == "make-checkerboard":
            from .tools.calibrate_camera import make_checkerboard
            make_checkerboard(cfg, args.out)
        elif args.cmd == "lidar-test":
            from .tools.diagnostics import lidar_test
            lidar_test(cfg, args)
        elif args.cmd == "benchmark":
            from .tools.diagnostics import benchmark
            benchmark(cfg, args)
    except (RuntimeError, FileNotFoundError) as e:
        log.error("%s", e)
        if args.verbose:
            raise
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
