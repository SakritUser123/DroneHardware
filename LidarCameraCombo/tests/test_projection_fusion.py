import numpy as np
import pytest

from lidepth.calibration import Extrinsics, estimate_intrinsics
from lidepth.config import DEFAULTS
from lidepth.fusion import FusionEngine, apply_fit, fit_inverse_depth
from lidepth.projection import Projected, project_scan, sample_median

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
W, H = 640, 480


def test_point_straight_ahead_camera_above():
    ext = Extrinsics(cam_up_m=0.10)
    R, t = ext.rotation_translation()
    p = project_scan([0.0], [2.0], R, t, K, W, H)
    assert len(p) == 1
    assert p.u[0] == pytest.approx(320.0)
    assert p.v[0] == pytest.approx(240.0 + 600.0 * 0.10 / 2.0)   # below centre by f*h/Z
    assert p.z_cam[0] == pytest.approx(2.0)


def test_camera_below_lidar_points_above_centre():
    R, t = Extrinsics(cam_up_m=-0.05).rotation_translation()
    p = project_scan([0.0], [1.0], R, t, K, W, H)
    assert p.v[0] == pytest.approx(240.0 - 30.0)


def test_right_side_point_goes_right_and_yaw_recentres():
    R, t = Extrinsics(cam_up_m=0.0).rotation_translation()
    p = project_scan([20.0], [3.0], R, t, K, W, H)      # 20° clockwise = to the right
    assert p.u[0] == pytest.approx(320.0 + 600.0 * np.tan(np.radians(20)))
    R2, t2 = Extrinsics(cam_up_m=0.0, yaw_deg=20.0).rotation_translation()
    p2 = project_scan([20.0], [3.0], R2, t2, K, W, H)
    assert p2.u[0] == pytest.approx(320.0)
    assert p2.z_cam[0] == pytest.approx(3.0)


def test_points_behind_or_outside_are_dropped():
    R, t = Extrinsics().rotation_translation()
    p = project_scan([180.0, 90.0, 0.0], [2.0, 2.0, 2.0], R, t, K, W, H)
    assert len(p) == 1 and p.idx[0] == 2


def test_forward_offset_and_pitch_sign():
    R, t = Extrinsics(cam_up_m=0.0, cam_forward_m=0.5).rotation_translation()
    p = project_scan([0.0], [2.0], R, t, K, W, H)
    assert p.z_cam[0] == pytest.approx(1.5)
    R, t = Extrinsics(cam_up_m=0.0, pitch_deg=5.0).rotation_translation()   # looking up
    p = project_scan([0.0], [4.0], R, t, K, W, H)
    assert p.v[0] > 240.0                                                   # points move down


def test_left_offset_moves_points_right():
    R, t = Extrinsics(cam_up_m=0.0, cam_left_m=0.2).rotation_translation()
    p = project_scan([0.0], [2.0], R, t, K, W, H)
    assert p.u[0] == pytest.approx(320 + 600 * 0.2 / 2.0)


def test_sample_median_ignores_outlier_and_nan():
    img = np.full((50, 50), 2.0)
    img[20, 20] = 100.0
    img[21, 21] = np.nan
    v = sample_median(img, np.array([20.2]), np.array([19.8]), 5)
    assert v[0] == pytest.approx(2.0)


def test_estimated_intrinsics_reasonable():
    intr = estimate_intrinsics(640, 480, 66, 41)   # 4:3 crop of a 16:9 sensor → vertical FOV binds
    assert intr.fy == pytest.approx(240 / np.tan(np.radians(20.5)))
    intr2 = estimate_intrinsics(640, 360, 66, 41)
    assert intr2.fx == pytest.approx(320 / np.tan(np.radians(33)), rel=0.05)


def synthetic_pairs(a, b, n=150, noise=0.02, outlier_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, 8.0, n)
    d = (1.0 / z - b) / a                         # exact model value
    d = d * (1 + noise * rng.standard_normal(n))  # multiplicative noise
    k = int(outlier_frac * n)
    idx = rng.choice(n, k, replace=False)
    z_meas = z.copy()
    z_meas[idx] = rng.uniform(0.3, 10.0, k)       # gross lidar/model mismatches
    return d, z_meas, z, idx


@pytest.mark.parametrize("a,b", [(0.15, -0.05), (0.02, 0.1), (3.0, 0.4)])
def test_fit_recovers_known_parameters(a, b):
    cfg = dict(DEFAULTS["fusion"])
    d, z_meas, z_true, out_idx = synthetic_pairs(a, b)
    res = fit_inverse_depth(d, z_meas, cfg, np.random.default_rng(0))
    assert res.ok, res.reason
    assert res.a == pytest.approx(a, rel=0.05)
    assert res.b == pytest.approx(b, abs=0.05 * (abs(a) * np.ptp(d)) + 0.01)
    # most outliers rejected, most good points kept
    assert res.inliers[out_idx].mean() < 0.3
    good = np.setdiff1d(np.arange(len(d)), out_idx)
    assert res.inliers[good].mean() > 0.85
    z_pred = 1 / (res.a * d[good] + res.b)
    assert np.median(np.abs(z_pred - z_true[good]) / z_true[good]) < 0.03


def test_fit_rejects_too_few_points_and_negative_a():
    cfg = dict(DEFAULTS["fusion"])
    d, z, _, _ = synthetic_pairs(0.1, 0.05, n=8)
    assert not fit_inverse_depth(d, z, cfg).ok
    d, z, _, _ = synthetic_pairs(0.1, 0.05, n=100, outlier_frac=0)
    res = fit_inverse_depth(-d, z, cfg)                    # inverted model → a would be < 0
    assert not res.ok


def test_fit_rejects_flat_wall():
    cfg = dict(DEFAULTS["fusion"])
    d = np.full(100, 1.0) + 1e-4 * np.random.default_rng(0).standard_normal(100)
    z = np.full(100, 2.0)
    res = fit_inverse_depth(d, z, cfg)
    assert not res.ok and "uniform" in res.reason


def test_apply_fit_marks_invalid():
    d = np.array([[1.0, -10.0, 0.0001, 5.0]])
    z = apply_fit(d, 0.2, 0.0, max_depth_m=18.0)
    assert z[0, 0] == pytest.approx(5.0)
    assert np.isnan(z[0, 1])        # negative denominator
    assert np.isnan(z[0, 2])        # 5000 m > max range
    assert z[0, 3] == pytest.approx(1.0)


def test_engine_ema_and_fallback():
    cfg = dict(DEFAULTS["fusion"])
    cfg["ema_alpha"] = 0.5
    eng = FusionEngine(cfg, 12.0, 1)
    a, b = 0.2, 0.05
    # d map with known relation along row 300
    h, w = 480, 640
    z_row = np.linspace(0.8, 6.0, w)
    d_map = np.tile((1 / z_row - b) / a, (h, 1)).astype(np.float32)
    u = np.arange(10, w - 10, 8).astype(float)
    v = np.full_like(u, 300.0)
    z = z_row[u.astype(int)]
    proj = Projected(u, v, z, z, np.arange(len(u)))
    out = eng.process(d_map, proj)
    assert out.status == "fit" and out.a == pytest.approx(a, rel=1e-3)
    assert np.nanmax(np.abs(out.depth_m[300] - z_row)) < 1e-2
    assert out.confidence[300].mean() > out.confidence[10].mean()     # higher near the line
    # second frame with a different scale → EMA moves half way
    d2 = np.tile((1 / z_row - b) / (2 * a), (h, 1)).astype(np.float32)
    out2 = eng.process(d2, proj)
    assert out2.a == pytest.approx((a + 2 * a) / 2, rel=1e-2)
    # empty frame → fallback to last good fit
    empty = Projected(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int))
    out3 = eng.process(d2, empty)
    assert out3.status == "fallback" and out3.a == pytest.approx(out2.a)


def test_camera_calibration_recovers_known_intrinsics():
    import cv2
    from lidepth.tools.calibrate_camera import calibrate
    K_true = np.array([[640.0, 0, 318.0], [0, 642.0, 243.0], [0, 0, 1]])
    D_true = np.array([0.12, -0.25, 0.001, -0.0005, 0.08])
    cols, rows, sq = 9, 6, 25.0
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq / 1000
    objp -= objp.mean(axis=0)
    rng = np.random.default_rng(0)
    views = []
    for _ in range(20):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08), rng.uniform(0.45, 0.9)])
        ip, _ = cv2.projectPoints(objp, rvec, tvec, K_true, D_true)
        views.append((ip + rng.normal(0, 0.1, ip.shape)).astype(np.float32))
    rms, K, D, per = calibrate(views, (640, 480), (cols, rows), sq)
    assert rms < 0.3
    assert K[0, 0] == pytest.approx(640, rel=0.01) and K[1, 1] == pytest.approx(642, rel=0.01)
    assert K[0, 2] == pytest.approx(318, abs=3) and K[1, 2] == pytest.approx(243, abs=3)
