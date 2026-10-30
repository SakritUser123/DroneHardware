"""Serial driver for the LD06 / LD19 lidar, running in a background thread."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

from .buffer import LidarBuffer
from .protocol import Packet, StreamParser, filter_points, packet_to_points

log = logging.getLogger(__name__)

PacketSink = Callable[[bytes, int], None]   # (raw 47 bytes, host arrival time ns)


class PacketProcessor:
    """Shared by the live driver and offline processing: packet → filtered points → buffer."""

    def __init__(self, lidar_cfg: Dict, buffer: LidarBuffer):
        self.cfg = lidar_cfg
        self.buffer = buffer
        model = str(lidar_cfg.get("model", "LD19")).upper()
        if model not in ("LD19", "LD06"):
            raise ValueError(f"lidar.model must be LD19 or LD06, got {model}")
        self.ld06_corr = bool(lidar_cfg.get("ld06_angle_correction", False))
        if self.ld06_corr and model != "LD06":
            log.warning("ld06_angle_correction is enabled but lidar.model is %s", model)

    def add(self, pkt: Packet) -> None:
        c = self.cfg
        t, ang, rng, inten = packet_to_points(pkt, latency_s=float(c["latency_s"]),
                                              baud=int(c["baud"]), ld06_correction=self.ld06_corr)
        t, ang, rng, inten = filter_points(t, ang, rng, inten, float(c["min_range_m"]),
                                           float(c["max_range_m"]), float(c["min_intensity"]))
        self.buffer.add_packet_points(pkt.start_angle_cdeg / 100.0, t, ang, rng, inten,
                                      packet_time_ns=pkt.t_arrival_ns, speed_dps=pkt.speed_dps)


class LidarDriver:
    """Opens the serial port, parses packets, fills a LidarBuffer.

    Usage::

        drv = LidarDriver(cfg["lidar"]); drv.start(); drv.wait_for_data()
        pts = drv.buffer.query(t_frame_ns, 0.05)
    """

    def __init__(self, lidar_cfg: Dict, packet_sink: Optional[PacketSink] = None):
        self.cfg = lidar_cfg
        self.buffer = LidarBuffer(int(lidar_cfg.get("scan_buffer_size", 30)))
        self.processor = PacketProcessor(lidar_cfg, self.buffer)
        self.parser = StreamParser(int(lidar_cfg["baud"]))
        self.packet_sink = packet_sink
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser = None
        self.error: Optional[str] = None

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "LidarDriver":
        try:
            import serial  # pyserial
        except ImportError as e:  # pragma: no cover - environment issue
            raise RuntimeError("pyserial is not installed: sudo apt install python3-serial "
                               "(or pip install pyserial)") from e
        port, baud = self.cfg["port"], int(self.cfg["baud"])
        try:
            self._ser = serial.Serial(port, baud, timeout=0.05)
        except serial.SerialException as e:
            hint = ("Check the port name (run: ls /dev/ttyUSB* /dev/ttyACM* /dev/serial*), "
                    "that you are in the 'dialout' group (sudo usermod -aG dialout $USER, then "
                    "log out/in), and for GPIO that the serial port is enabled in raspi-config.")
            raise RuntimeError(f"Cannot open lidar port {port}: {e}. {hint}") from e
        log.info("Lidar port %s opened at %d baud (model %s)", port, baud, self.cfg["model"])
        self._thread = threading.Thread(target=self._run, name="lidar", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- thread ------------------------------------------------------------------
    def _run(self) -> None:
        ser = self._ser
        while not self._stop.is_set():
            try:
                n = ser.in_waiting
                data = ser.read(n if n > 0 else 1)
            except Exception as e:  # unplugged, etc.
                self.error = f"Serial read failed: {e}"
                log.error(self.error)
                return
            if not data:
                continue
            t_read = time.monotonic_ns()
            for pkt in self.parser.feed(data, t_read):
                if self.packet_sink is not None:
                    self.packet_sink(pkt.raw, pkt.t_arrival_ns)
                self.processor.add(pkt)

    # -- helpers -----------------------------------------------------------------
    def wait_for_data(self, timeout_s: Optional[float] = None) -> None:
        """Block until a full scan arrived, else raise a helpful error."""
        timeout_s = float(self.cfg.get("no_data_timeout_s", 3.0)) if timeout_s is None else timeout_s
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.error:
                raise RuntimeError(self.error)
            if self.buffer.n_scans() >= 1:
                log.info("Lidar OK: %d packets, %d CRC errors, %.1f Hz",
                         self.parser.n_ok, self.parser.n_crc_err, self.buffer.last_speed_dps / 360.0)
                return
            time.sleep(0.05)
        port, baud = self.cfg["port"], self.cfg["baud"]
        if self.parser.n_bytes == 0:
            msg = (f"No data on {port} — check wiring/baud. Is the lidar spinning? "
                   f"(USB adapter: is the board's LED on? GPIO: lidar TX → Pi RX = GPIO15 / pin 10, "
                   f"and serial enabled in raspi-config with the login shell DISABLED.)")
        elif self.parser.n_ok == 0:
            msg = (f"Bytes are arriving on {port} but no valid packet was decoded "
                   f"({self.parser.n_bytes} bytes, {self.parser.n_crc_err} CRC errors). "
                   f"Check baud (expected 230400, configured {baud}) and lidar.model.")
        else:
            msg = (f"Packets arrive on {port} but no full 360° scan was assembled within "
                   f"{timeout_s:.0f} s — is the lidar spinning freely?")
        raise RuntimeError(msg)

    def stats(self) -> Dict:
        return {
            "packets_ok": self.parser.n_ok,
            "crc_errors": self.parser.n_crc_err,
            "bytes": self.parser.n_bytes,
            "scans": self.buffer.n_scans(),
            "scan_hz": self.buffer.last_speed_dps / 360.0,
        }
