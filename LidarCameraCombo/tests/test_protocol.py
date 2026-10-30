import numpy as np
import pytest

from lidepth.lidar.buffer import LidarBuffer
from lidepth.lidar.protocol import (CRC_TABLE, PACKET_LEN, StreamParser, angle_to_xyz, crc8,
                                    encode_packet, filter_points, interpolate_angles,
                                    packet_to_points, parse_packet)


def crc_table_from_polynomial(poly=0x4D):
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        t.append(c)
    return bytes(t)


def test_crc_table_is_official():
    # Excerpts printed in the LD19 Development Manual's CRC table
    assert list(CRC_TABLE[:8]) == [0x00, 0x4D, 0x9A, 0xD7, 0x79, 0x34, 0xE3, 0xAE]
    assert bytes([0xF5, 0x1F, 0x52, 0x85, 0xC8, 0x66, 0x2B, 0xFC, 0xB1, 0xED, 0xA0, 0x77]) in CRC_TABLE
    assert bytes([0x3F, 0xCA, 0x87, 0x50, 0x1D, 0xB3, 0xFE, 0x29, 0x64, 0x38, 0x75, 0xA2]) in CRC_TABLE
    assert CRC_TABLE == crc_table_from_polynomial(0x4D)


# A hand-written packet: speed 3600 deg/s (10 Hz), 10.00° → 18.80°, ranges 1000..1110 mm.
SAMPLE = bytes.fromhex(
    "542C" "100E" "E803"
    "E803C8" "F203C8" "FC03C8" "0604C8" "1004C8" "1A04C8"
    "2404C8" "2E04C8" "3804C8" "4204C8" "4C04C8" "5604C8"
    "5807" "3412")


def sample_packet():
    return SAMPLE + bytes([crc8(SAMPLE)])


def test_parse_sample_bytes():
    raw = sample_packet()
    assert len(raw) == PACKET_LEN
    p = parse_packet(raw)
    assert p is not None
    assert p.speed_dps == 3600
    assert p.start_angle_cdeg == 1000 and p.end_angle_cdeg == 1880
    assert p.distances_mm.tolist() == list(range(1000, 1120, 10))
    assert np.all(p.intensities == 200)
    assert p.timestamp_ms == 0x1234
    assert np.allclose(p.angles_deg(), 10.0 + 0.8 * np.arange(12))


def test_bad_crc_and_header_rejected():
    raw = bytearray(sample_packet())
    raw[10] ^= 0xFF
    assert parse_packet(bytes(raw)) is None
    bad = bytearray(sample_packet())
    bad[1] = 0x2D
    assert parse_packet(bytes(bad)) is None
    assert parse_packet(sample_packet()[:-1]) is None


def test_encode_roundtrip():
    d = np.arange(12) * 100 + 500
    i = np.arange(12) * 10
    raw = encode_packet(3600, 35500, 400, d, i, 29999)
    p = parse_packet(raw)
    assert p.distances_mm.tolist() == d.tolist()
    assert p.intensities.tolist() == i.tolist()
    assert p.timestamp_ms == 29999


def test_angle_wraparound():
    a = interpolate_angles(35500, 400)        # 355° → 4°, 9° span
    step = 9.0 / 11
    exp = np.mod(355.0 + step * np.arange(12), 360.0)
    assert np.allclose(a, exp)
    assert a[0] == pytest.approx(355.0) and a[-1] == pytest.approx(4.0)
    assert np.all((a >= 0) & (a < 360))
    # crossing exactly at 0
    b = interpolate_angles(35912, 792)        # 359.12° → 7.92°, step 0.80°
    assert b[1] == pytest.approx(359.92)
    assert b[2] == pytest.approx(0.72)
    assert b[-1] == pytest.approx(7.92)


def test_stream_parser_resyncs_and_counts_errors():
    good = sample_packet()
    corrupt = bytearray(good)
    corrupt[20] ^= 0x55
    stream = b"\x00\x54\x13garbage" + good + bytes(corrupt) + good[:20]
    sp = StreamParser(230400)
    pkts = sp.feed(stream, t_read_ns=1_000_000_000)
    assert len(pkts) == 1 and sp.n_crc_err >= 1
    pkts2 = sp.feed(good[20:] + good, t_read_ns=2_000_000_000)
    assert len(pkts2) == 2
    # the earlier packet in the chunk is back-dated by its 47 bytes of transfer time
    byte_ns = 1e9 * 10 / 230400
    assert pkts2[1].t_arrival_ns == 2_000_000_000
    assert pkts2[0].t_arrival_ns == pytest.approx(2_000_000_000 - 47 * byte_ns, abs=2)


def test_per_point_timestamps_and_filter():
    p = parse_packet(sample_packet())
    p.t_arrival_ns = 5_000_000_000
    t, ang, rng, inten = packet_to_points(p, latency_s=0.0, baud=230400)
    dt = 0.8 / 3600                        # angular step / speed
    assert np.all(np.diff(t) > 0)
    assert (t[1] - t[0]) / 1e9 == pytest.approx(dt, rel=1e-3)
    assert (p.t_arrival_ns - t[-1]) / 1e9 == pytest.approx(47 * 10 / 230400, rel=1e-3)
    rng2 = rng.copy()
    rng2[0] = 0.0
    inten2 = inten.copy()
    inten2[1] = 3
    rng2[2] = 20.0
    out = filter_points(t, ang, rng2, inten2, 0.02, 12.0, 15)
    assert len(out[0]) == 9


def test_angle_to_xyz_clockwise():
    p = angle_to_xyz(np.array([0.0, 90.0, 270.0]), np.array([1.0, 1.0, 1.0]))
    assert np.allclose(p[0], [1, 0, 0])
    assert np.allclose(p[1], [0, -1, 0], atol=1e-12)   # 90° clockwise = to the RIGHT
    assert np.allclose(p[2], [0, 1, 0], atol=1e-12)


def test_buffer_scan_assembly_and_time_query():
    buf = LidarBuffer(10)
    t0 = 1_000_000_000
    # 3 revolutions of 30 packets (12° each) at 10 Hz
    for k in range(90):
        start = (k * 12.0) % 360
        ts = t0 + int(k * 0.1 / 30 * 1e9) + np.arange(12) * 100_000
        buf.add_packet_points(start, ts, np.mod(start + np.arange(12), 360), np.ones(12), np.ones(12))
    assert buf.n_scans() == 2           # third one still in progress
    q = buf.query(t0 + int(0.15e9), 0.02, 0.5)
    assert len(q) > 0
    assert np.all(np.abs(q.t_ns - (t0 + 0.15e9)) <= 0.02e9)


def test_buffer_dedupes_repeated_angles():
    buf = LidarBuffer(10)
    # same angle seen at t=0 and t=100 ms; frame at t=90 ms with ±60 ms window → keep the 2nd
    buf.add_packet_points(0.0, np.array([0, 100_000_000]), np.array([5.0, 5.0]), np.array([1.0, 2.0]),
                          np.array([100.0, 100.0]))
    q = buf.query(90_000_000, 0.2, 0.5)
    assert len(q) == 1 and q.range_m[0] == 2.0
