#!/usr/bin/env python3
"""Get Depth Anything V2 **Small** (Apache-2.0) as ONNX for ONNX Runtime.

    python scripts/get_model.py                  # download ready-made ONNX (fp32, ~99 MB)  ← easiest
    python scripts/get_model.py --variant int8   # download the int8-quantised ONNX (~27 MB, faster on the Pi)
    python scripts/get_model.py --quantize       # make your own int8 copy of the fp32 file
    python scripts/get_model.py --export         # export yourself from PyTorch (laptop; needs
                                                 #   pip install -r requirements-export.txt)
    python scripts/get_model.py --fix-shape 252x182   # optional: freeze the input size (W x H)

Only the SMALL model is Apache-2.0. Base/Large/Giant are CC-BY-NC-4.0 (non-commercial) — not used here.
The ready-made files come from https://huggingface.co/onnx-community/depth-anything-v2-small (a
conversion of depth-anything/Depth-Anything-V2-Small-hf).  Input: 'pixel_values' (1,3,H,W) normalised
with ImageNet mean/std, H and W multiples of 14.  Output: relative inverse depth (1,H,W).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.normpath(os.path.join(HERE, "..", "models"))
REPO = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/"
FILES = {"fp32": "model.onnx", "int8": "model_int8.onnx", "fp16": "model_fp16.onnx"}
DEFAULT_OUT = os.path.join(MODELS, "depth_anything_v2_small.onnx")


def download(variant: str, out: str) -> str:
    url = REPO + FILES[variant]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    print(f"Downloading {url}\n  → {out}")

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}% of {total / 1e6:.0f} MB")
            sys.stdout.flush()
    try:
        urllib.request.urlretrieve(url, tmp, hook)
    except Exception as e:
        print(f"\nDownload failed: {e}\n"
              "Check the internet connection, or download the file in a browser from\n"
              f"  {url}\nand save it as {out}")
        sys.exit(1)
    print()
    shutil.move(tmp, out)
    return out


def export_from_torch(out: str, opset: int = 17) -> str:
    try:
        import torch
        from transformers import AutoModelForDepthEstimation
    except ImportError:
        print("Export needs PyTorch + transformers: pip install -r requirements-export.txt")
        sys.exit(1)
    print("Loading depth-anything/Depth-Anything-V2-Small-hf from Hugging Face…")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).predicted_depth

    dummy = torch.randn(1, 3, 196, 252)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.onnx.export(Wrap(model), dummy, out, input_names=["pixel_values"],
                      output_names=["predicted_depth"], opset_version=opset,
                      dynamic_axes={"pixel_values": {2: "height", 3: "width"},
                                    "predicted_depth": {1: "height", 2: "width"}})
    print(f"Exported to {out}")
    return out


def quantize(src: str, dst: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"Quantising {src} → {dst} (dynamic int8)…")
    quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)
    return dst


def fix_shape(src: str, dst: str, w: int, h: int) -> str:
    if w % 14 or h % 14:
        print("Width and height must be multiples of 14 (e.g. 252x182, 364x266, 518x392).")
        sys.exit(1)
    try:
        import onnx
        from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed, fix_output_shapes
    except ImportError:
        print("--fix-shape needs the 'onnx' package: pip install onnx")
        sys.exit(1)
    m = onnx.load(src)
    name = m.graph.input[0].name
    make_input_shape_fixed(m.graph, name, [1, 3, h, w])
    fix_output_shapes(m)
    onnx.save(m, dst)
    print(f"Saved fixed-shape model {dst} (input {w}x{h}). Point model.path at it.")
    return dst


def verify(path: str) -> None:
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"Model OK: input '{inp.name}' {inp.shape}, output '{out.name}' {out.shape}")
    h = inp.shape[2] if isinstance(inp.shape[2], int) else 182
    w = inp.shape[3] if isinstance(inp.shape[3], int) else 252
    x = np.random.default_rng(0).standard_normal((1, 3, h, w)).astype(np.float32)
    t0 = time.perf_counter()
    y = sess.run([out.name], {inp.name: x})[0]
    print(f"Test inference {w}x{h}: output {y.shape}, {1000 * (time.perf_counter() - t0):.0f} ms "
          f"(first run is slower)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(FILES), default="fp32")
    ap.add_argument("--out", default=None, help="output path (default models/depth_anything_v2_small[_int8].onnx)")
    ap.add_argument("--export", action="store_true", help="export from PyTorch instead of downloading")
    ap.add_argument("--quantize", action="store_true", help="also write an int8 copy of the model")
    ap.add_argument("--fix-shape", default=None, metavar="WxH", help="also write a fixed-input-size copy")
    a = ap.parse_args()

    suffix = "" if a.variant == "fp32" else f"_{a.variant}"
    out = a.out or os.path.join(MODELS, f"depth_anything_v2_small{suffix}.onnx")
    if a.export:
        export_from_torch(out)
    elif os.path.exists(out):
        print(f"{out} already exists (delete it to download again).")
    else:
        download(a.variant, out)
    verify(out)
    if a.quantize:
        q = out.replace(".onnx", "_int8.onnx")
        quantize(out, q)
        verify(q)
    if a.fix_shape:
        w, h = (int(v) for v in a.fix_shape.lower().split("x"))
        f = out.replace(".onnx", f"_{w}x{h}.onnx")
        fix_shape(out, f, w, h)
        verify(f)
    if out != DEFAULT_OUT:
        print(f"\nTo use it: --set model.path={os.path.relpath(out, os.path.join(HERE, '..'))} "
              f"(or edit model.path in config.yaml)")


if __name__ == "__main__":
    main()
