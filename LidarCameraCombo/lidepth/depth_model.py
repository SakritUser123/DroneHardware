"""Depth Anything V2 Small (ONNX Runtime, CPU).

Output = RELATIVE INVERSE DEPTH (bigger = closer), unknown scale AND shift, as a float
array.  All math uses this raw array; colours are produced only in viz.py for display.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

PATCH = 14   # ViT patch size: input sides must be multiples of 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def model_input_size(img_w: int, img_h: int, long_side: int) -> Tuple[int, int]:
    """(W, H) for the network: long side ≈ ``long_side``, both multiples of 14, aspect kept."""
    long_side = max(PATCH * 4, int(round(long_side / PATCH)) * PATCH)
    if img_w >= img_h:
        w = long_side
        h = max(PATCH, int(round(long_side * img_h / img_w / PATCH)) * PATCH)
    else:
        h = long_side
        w = max(PATCH, int(round(long_side * img_w / img_h / PATCH)) * PATCH)
    return w, h


class OnnxDepthModel:
    def __init__(self, model_path: str, input_long_side: int = 252, threads: int = 4):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed: pip install onnxruntime") from e
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Depth model not found at {model_path}. Download it with:\n"
                f"    python scripts/get_model.py\n"
                f"(or run with --mock / model.mock_model=true for a hardware-free test)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.sess.get_outputs()[0].name
        self.long_side = int(input_long_side)
        # Fixed-size exports have integer dims; dynamic ones have strings/None.
        shp = inp.shape
        self.fixed_hw = (shp[2], shp[3]) if isinstance(shp[2], int) and isinstance(shp[3], int) else None
        if self.fixed_hw:
            log.info("Model has fixed input %dx%d (input_long_side ignored)", self.fixed_hw[1], self.fixed_hw[0])
        log.info("Loaded depth model %s (input '%s')", os.path.basename(model_path), self.input_name)
        self.last_infer_s = 0.0

    def input_size(self, img_w: int, img_h: int) -> Tuple[int, int]:
        if self.fixed_hw:
            return int(self.fixed_hw[1]), int(self.fixed_hw[0])
        return model_input_size(img_w, img_h, self.long_side)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        import cv2
        h, w = bgr.shape[:2]
        mw, mh = self.input_size(w, h)
        rgb = cv2.cvtColor(cv2.resize(bgr, (mw, mh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Return relative inverse depth (float32) at the SAME size as the input image."""
        import cv2
        h, w = bgr.shape[:2]
        x = self.preprocess(bgr)
        t0 = time.perf_counter()
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        self.last_infer_s = time.perf_counter() - t0
        d = np.squeeze(np.asarray(out, np.float32))
        if d.ndim != 2:
            raise RuntimeError(f"Unexpected model output shape {out.shape}")
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)


class MockDepthModel:
    """Stand-in for the network in mock mode: returns the TRUE inverse depth passed through an
    unknown affine map plus noise and a smooth bias — so fusion has to do real work."""

    def __init__(self, scale: float = 7.3, shift: float = 0.4, noise: float = 0.02, seed: int = 1):
        self.scale, self.shift, self.noise = scale, shift, noise
        self.rng = np.random.default_rng(seed)
        self.last_infer_s = 0.0
        self.truth: Optional[np.ndarray] = None   # set by the simulator before predict()

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if self.truth is None:
            raise RuntimeError("MockDepthModel needs ground-truth depth (mock camera only)")
        z = self.truth
        h, w = z.shape
        inv = np.where(np.isfinite(z) & (z > 0), 1.0 / np.maximum(z, 1e-3), 0.0)
        yy = np.linspace(-1, 1, h)[:, None]
        bias = 0.12 * yy ** 2                      # error grows away from the image centre row
        d = self.scale * inv + self.shift + bias + self.noise * self.rng.standard_normal((h, w))
        self.last_infer_s = time.perf_counter() - t0
        return d.astype(np.float32)


def build_model(cfg):
    m = cfg["model"]
    if m.get("mock_model") or (cfg.get("mock") and not os.path.exists(m["path"])):
        if not m.get("mock_model"):
            log.warning("Mock mode and no ONNX model found → using the fake depth model.")
        return MockDepthModel()
    return OnnxDepthModel(m["path"], int(m["input_long_side"]), int(m["threads"]))
