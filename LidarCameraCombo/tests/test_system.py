"""Config consistency, obstacle sectors, and a full mock record → process → evaluate run."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lidepth.calibration import Extrinsics
from lidepth.config import DEFAULTS, load_config
from lidepth.output.obstacle import UINT16_MAX, depth_to_sectors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _keys(d, prefix=""):
    out = set()
    for k, v in d.items():
        out.add(prefix + k)
        if isinstance(v, dict):
            out |= _keys(v, prefix + k + ".")
    return out


def test_config_yaml_matches_defaults():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        user = yaml.safe_load(f)
    assert _keys(user) == _keys(DEFAULTS)


def test_overrides():
    cfg = load_config(os.path.join(ROOT, "config.yaml"),
                      ["lidar.port=/dev/serial0", "model.input_long_side=182", "obstacle_output.enabled=true"])
    assert cfg["lidar"]["port"] == "/dev/serial0"
    assert cfg["model"]["input_long_side"] == 182
    assert cfg["obstacle_output"]["enabled"] is True
    with pytest.raises(KeyError):
        load_config(os.path.join(ROOT, "config.yaml"), ["lidar.prot=/dev/x"])


def test_obstacle_sectors_from_depth():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    ext = Extrinsics(cam_up_m=0.0)
    R, _ = ext.rotation_translation()
    depth = np.full((480, 640), 4.0, np.float32)
    depth[200:280, 300:340] = 1.5            # obstacle straight ahead
    cfg = dict(DEFAULTS["obstacle_output"])
    s = depth_to_sectors(depth, K, R, np.zeros(3), cfg)
    assert len(s) == 72
    assert s[0] == 150                       # sector 0 = straight ahead, 1.5 m
    # camera sees about ±32° → sectors 1..6 (right) and 66..71 (left) have data
    assert s[3] < UINT16_MAX and s[69] < UINT16_MAX
    assert s[36] == UINT16_MAX              # behind: unknown
    assert 400 <= s[5] <= 500               # flat wall at Z=4 → horizontal distance ≥ 4 m


def test_mock_record_process_evaluate(tmp_path):
    from lidepth.tools import evaluate, process, record
    cfg = load_config(os.path.join(ROOT, "config.yaml"),
                      [f"record.sessions_dir={tmp_path}",
                       f"calibration.camera_file={tmp_path}/none_cam.yaml",
                       f"calibration.extrinsics_file={tmp_path}/none_ext.yaml",
                       f"model.path={tmp_path}/no_model.onnx"])
    cfg["mock"] = True
    path = record.run(cfg, SimpleNamespace(fps=5, duration=1.2, name="t"))
    out = process.run(cfg, SimpleNamespace(session=path, out=None, every=1, max_frames=None, no_viz=False,
                                           save_model_output=False, session_calib=False))
    depth_files = sorted(os.listdir(os.path.join(out, "depth")))
    assert len(depth_files) == 6
    z = np.load(os.path.join(out, "depth", depth_files[0]))
    assert z.shape == (480, 640) and np.isfinite(z).mean() > 0.9
    with open(os.path.join(out, "fit_log.csv")) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 7 and all(",fit," in ln for ln in lines[1:])
    summ = evaluate.run(cfg, SimpleNamespace(session=path, processed=None, points_csv=None, auto_gt=30,
                                             session_calib=False))
    assert summ["near lidar line"]["mean_pct"] < 3.0
    assert summ["all"]["mean_pct"] < 10.0
    assert os.path.exists(os.path.join(out, "evaluation", "evaluation.png"))
