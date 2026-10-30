"""`record`: save synchronised camera frames + every raw lidar packet to a session folder."""
from __future__ import annotations

import logging
import time

from ..session import SessionWriter
from ..utils import build_sensors

log = logging.getLogger(__name__)


def run(cfg, args) -> str:
    rc = cfg["record"]
    fps = float(args.fps or rc["fps"])
    duration = float(args.duration if args.duration is not None else rc["duration_s"])
    writer = SessionWriter(rc["sessions_dir"], cfg, args.name)
    log.info("Recording to %s", writer.path)
    if cfg.get("mock"):
        _record_mock(cfg, writer, fps, duration or 10.0)
        writer.close()
        log.info("Done: %d frames, %d lidar packets → %s", writer.n_frames, writer.n_packets, writer.path)
        return writer.path

    cam, lidar, _ = build_sensors(cfg, packet_sink=writer.add_packet)
    try:
        lidar.start()
        lidar.wait_for_data()
        cam.start()
        period = 1.0 / fps
        t_start = time.monotonic()
        next_t = t_start
        last_print = t_start
        print("Recording… press Ctrl+C to stop.")
        while True:
            now = time.monotonic()
            if duration > 0 and now - t_start >= duration:
                break
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t += period
            frame = cam.capture()
            writer.add_frame(frame)
            if now - last_print > 2.0:
                st = lidar.stats()
                log.info("frames %d | lidar packets %d (CRC errors %d) | %.1f Hz", writer.n_frames,
                         writer.n_packets, st["crc_errors"], st["scan_hz"])
                last_print = now
            if lidar.error:
                raise RuntimeError(lidar.error)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        time.sleep(0.15)          # let the lidar deliver the points around the last frame
        lidar.stop()
        cam.stop()
        writer.close()
    log.info("Saved %d frames and %d lidar packets to %s", writer.n_frames, writer.n_packets, writer.path)
    print(f"\nSession saved: {writer.path}\nNext: python -m lidepth process {writer.path}")
    return writer.path


def _record_mock(cfg, writer, fps: float, duration: float) -> None:
    """Fast, deterministic mock recording using simulated (virtual) time."""
    from ..camera import Frame
    from ..lidar.protocol import StreamParser
    from ..sim import Simulator
    from ..calibration import save_extrinsics, save_intrinsics
    import os
    sim = Simulator(cfg, t0_ns=10_000_000_000)
    # the simulator's exact camera model, so `process` uses the same geometry
    save_intrinsics(os.path.join(writer.path, "sim_intrinsics.yaml"), sim.intr)
    save_extrinsics(os.path.join(writer.path, "sim_extrinsics.yaml"), sim.ext)
    parser = StreamParser(int(cfg["lidar"]["baud"]))
    t = sim.t0_ns + int(0.5e9)
    last = sim.t0_ns
    n = max(1, int(duration * fps))
    for k in range(n):
        for raw, ta in sim.packets_between(last, t + int(0.06e9)):
            for pkt in parser.feed(raw, ta):
                writer.add_packet(pkt.raw, pkt.t_arrival_ns)
        last = t + int(0.06e9)
        img, z = sim.render(t)
        writer.add_frame(Frame(img, t, k + 1, {"mock": True}), truth=z)
        t += int(1e9 / fps)
