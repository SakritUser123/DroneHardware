"""LDRobot LD06 / LD19 serial protocol (pure Python + NumPy, no hardware needed).

Packet layout (47 bytes, little endian), from the LD19 Development Manual / LD06 datasheet:

    offset size field
    0      1    header      = 0x54
    1      1    ver_len     = 0x2C   (upper 3 bits: packet type = 1, lower 5 bits: 12 points)
    2      2    speed       uint16, degrees per second
    4      2    start_angle uint16, 0.01 degree
    6      36   12 x (distance uint16 mm, intensity uint8)
    42     2    end_angle   uint16, 0.01 degree
    44     2    timestamp   uint16, milliseconds, wraps at 30000
    46     1    crc8        over bytes 0..45 using the official LDRobot table

The lidar spins CLOCKWISE (seen from above); angles grow clockwise.
LD06 and LD19 share this exact wire format (the official ldlidar_stl SDK uses the same
parser for both); the differences are in the hardware specs, see README.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

HEADER = 0x54
VER_LEN = 0x2C
POINTS_PER_PACK = 12
PACKET_LEN = 47
TIMESTAMP_WRAP_MS = 30000

# Official LDRobot CRC8 table (polynomial 0x4D), copied from the LD19 Development Manual.
CRC_TABLE = bytes([
    0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae, 0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
    0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07, 0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
    0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1, 0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
    0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18, 0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
    0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90, 0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
    0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39, 0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
    0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f, 0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
    0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26, 0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
    0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2, 0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
    0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b, 0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
    0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd, 0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
    0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64, 0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
    0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec, 0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
    0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45, 0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
    0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3, 0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
    0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a, 0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8,
])

_STRUCT = struct.Struct("<BBHH" + "HB" * POINTS_PER_PACK + "HHB")
assert _STRUCT.size == PACKET_LEN


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


@dataclass
class Packet:
    speed_dps: int            # rotation speed, degrees / second
    start_angle_cdeg: int     # 0.01 deg
    end_angle_cdeg: int       # 0.01 deg
    distances_mm: np.ndarray  # (12,) uint16
    intensities: np.ndarray   # (12,) uint8
    timestamp_ms: int         # device clock, wraps at 30000
    t_arrival_ns: int = 0     # host time.monotonic_ns() when the last byte arrived
    raw: bytes = b""          # the original 47 bytes (kept for recording)

    def angles_deg(self) -> np.ndarray:
        return interpolate_angles(self.start_angle_cdeg, self.end_angle_cdeg)


def interpolate_angles(start_cdeg: int, end_cdeg: int, n: int = POINTS_PER_PACK) -> np.ndarray:
    """Angle (deg, 0..360) of each point, linear between start and end, handling 360→0 wrap."""
    start = float(start_cdeg)
    end = float(end_cdeg)
    if end < start:          # packet crosses 0°  (e.g. 355.00° → 3.00°)
        end += 36000.0
    step = (end - start) / (n - 1) if n > 1 else 0.0
    ang = (start + step * np.arange(n)) / 100.0
    return np.mod(ang, 360.0)


def packet_span_deg(pkt: Packet) -> float:
    end = pkt.end_angle_cdeg
    if end < pkt.start_angle_cdeg:
        end += 36000
    return (end - pkt.start_angle_cdeg) / 100.0


def parse_packet(buf: bytes, check_crc: bool = True) -> Optional[Packet]:
    """Parse exactly one 47-byte packet. Returns None if header/length/CRC is wrong."""
    if len(buf) != PACKET_LEN or buf[0] != HEADER or buf[1] != VER_LEN:
        return None
    if check_crc and crc8(buf[:PACKET_LEN - 1]) != buf[PACKET_LEN - 1]:
        return None
    f = _STRUCT.unpack(buf)
    speed, start = f[2], f[3]
    pts = f[4:4 + 2 * POINTS_PER_PACK]
    end, ts = f[4 + 2 * POINTS_PER_PACK], f[5 + 2 * POINTS_PER_PACK]
    return Packet(
        raw=bytes(buf),
        speed_dps=speed,
        start_angle_cdeg=start,
        end_angle_cdeg=end,
        distances_mm=np.array(pts[0::2], dtype=np.uint16),
        intensities=np.array(pts[1::2], dtype=np.uint8),
        timestamp_ms=ts,
    )


def encode_packet(speed_dps: int, start_cdeg: int, end_cdeg: int,
                  distances_mm, intensities, timestamp_ms: int) -> bytes:
    """Build a valid packet (used by tests and by the lidar simulator)."""
    fields = [HEADER, VER_LEN, int(speed_dps) & 0xFFFF, int(start_cdeg) % 36000]
    for d, i in zip(distances_mm, intensities):
        fields += [int(d) & 0xFFFF, int(i) & 0xFF]
    fields += [int(end_cdeg) % 36000, int(timestamp_ms) % TIMESTAMP_WRAP_MS, 0]
    raw = bytearray(_STRUCT.pack(*fields))
    raw[-1] = crc8(bytes(raw[:-1]))
    return bytes(raw)


class StreamParser:
    """Turns an arbitrary byte stream into packets. Re-synchronises after garbage/bad CRC."""

    def __init__(self, baud: int = 230400):
        self.buf = bytearray()
        self.byte_time_ns = 1e9 * 10 / baud        # 8N1 = 10 bits per byte
        self.n_ok = 0
        self.n_crc_err = 0
        self.n_bytes = 0

    def feed(self, data: bytes, t_read_ns: int) -> List[Packet]:
        """Add bytes read at host time ``t_read_ns``; return the complete packets.

        Each packet's ``t_arrival_ns`` is back-dated by the serial transfer time of the
        bytes that came after it in this read chunk.
        """
        self.n_bytes += len(data)
        self.buf.extend(data)
        found: List[Tuple[int, Packet]] = []   # (end offset in buf, packet)
        pos = 0
        n = len(self.buf)
        while pos + PACKET_LEN <= n:
            if self.buf[pos] != HEADER or self.buf[pos + 1] != VER_LEN:
                nxt = self.buf.find(bytes([HEADER]), pos + 1)
                pos = nxt if nxt >= 0 else n
                continue
            pkt = parse_packet(bytes(self.buf[pos:pos + PACKET_LEN]))
            if pkt is None:
                self.n_crc_err += 1
                pos += 1
                continue
            self.n_ok += 1
            found.append((pos + PACKET_LEN, pkt))
            pos += PACKET_LEN
        del self.buf[:pos]
        for end_off, pkt in found:
            bytes_after = n - end_off
            pkt.t_arrival_ns = t_read_ns - int(round(bytes_after * self.byte_time_ns))
        return [p for _, p in found]


def ld06_angle_correction_deg(dist_mm: np.ndarray) -> np.ndarray:
    """Near-range angle correction used by the legacy LD06 SDK (optional, off by default)."""
    d = dist_mm.astype(np.float64)
    out = np.zeros_like(d)
    m = d > 0
    out[m] = np.degrees(np.arctan(21.8 * (155.3 - d[m]) / (155.3 * d[m])))
    return out


def packet_to_points(pkt: Packet, latency_s: float = 0.0, baud: int = 230400,
                     ld06_correction: bool = False):
    """Convert a packet into per-point arrays (t_ns int64, angle_deg, range_m, intensity).

    Per-point capture time: the packet is sent right after its last point is measured,
    so point i was measured (11 - i) * dt earlier, with dt = angular step / rotation speed.
    We also subtract the 47-byte serial transfer time and a configurable fixed latency.
    """
    ang = pkt.angles_deg()
    if ld06_correction:
        ang = np.mod(ang + ld06_angle_correction_deg(pkt.distances_mm), 360.0)
    step_deg = packet_span_deg(pkt) / (POINTS_PER_PACK - 1)
    speed = pkt.speed_dps if pkt.speed_dps > 0 else 3600.0  # 10 Hz nominal if unknown
    dt_s = step_deg / speed
    tx_s = PACKET_LEN * 10.0 / baud
    idx_from_end = (POINTS_PER_PACK - 1) - np.arange(POINTS_PER_PACK)
    t = pkt.t_arrival_ns - int((tx_s + latency_s) * 1e9) - (idx_from_end * dt_s * 1e9).astype(np.int64)
    rng = pkt.distances_mm.astype(np.float64) / 1000.0
    return t.astype(np.int64), ang, rng, pkt.intensities.astype(np.float64)


def filter_points(t, ang, rng, inten, min_range_m: float, max_range_m: float,
                  min_intensity: float):
    """Drop distance 0, out-of-range values and weak returns."""
    keep = (rng > 0) & (rng >= min_range_m) & (rng <= max_range_m) & (inten >= min_intensity)
    return t[keep], ang[keep], rng[keep], inten[keep]


def angle_to_xyz(angle_deg: np.ndarray, range_m: np.ndarray, upside_down: bool = False) -> np.ndarray:
    """Lidar polar (clockwise angle) → right-handed lidar frame (x fwd, y left, z up)."""
    a = np.radians(angle_deg)
    sign = 1.0 if upside_down else -1.0       # clockwise angle = negative math angle
    x = range_m * np.cos(a)
    y = sign * range_m * np.sin(a)
    return np.stack([x, y, np.zeros_like(x)], axis=-1)


def degrees_between(a: float, b: float) -> float:
    """Smallest signed difference b - a in degrees."""
    return (b - a + 180.0) % 360.0 - 180.0


__all__ = [
    "HEADER", "VER_LEN", "PACKET_LEN", "POINTS_PER_PACK", "CRC_TABLE", "crc8", "Packet",
    "parse_packet", "encode_packet", "StreamParser", "interpolate_angles", "packet_to_points",
    "filter_points", "angle_to_xyz", "ld06_angle_correction_deg", "packet_span_deg",
    "degrees_between",
]
