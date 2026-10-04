"""Benchmark CMNE inference at the paper's scale (5124 sources, k = 80, d = 1280).

Usage: python benchmarks/bench_inference.py [--units 1280] [--steps 50] [--signals 1 20]
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

import cmne


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=int, default=5124)
    p.add_argument("--look-back", type=int, default=80)
    p.add_argument("--units", type=int, default=1280)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--signals", type=int, nargs="+", default=[1, 20])
    a = p.parse_args()

    model = cmne.CMNEModel.create(a.sources, a.look_back, a.units, seed=0)
    n_params = sum(t.numel() for t in model.network.parameters())
    print(
        f"model: {a.sources} sources, k={a.look_back}, d={a.units}, {n_params / 1e6:.1f} M params"
    )

    with tempfile.TemporaryDirectory() as tmp:
        predictors = {"torch": model}
        try:
            onnx_file = cmne.export_onnx(model, Path(tmp) / "m.onnx")
            predictors["onnxruntime"] = cmne.OnnxPredictor(onnx_file)
        except ImportError:
            pass

        rng = np.random.default_rng(0)
        for n_sig in a.signals:
            x = rng.standard_normal((n_sig, a.sources, a.look_back + a.steps)).astype(np.float32)
            for name, pred in predictors.items():
                cmne.apply_cmne(x[..., : a.look_back + 2], pred)  # warm-up
                t = time.perf_counter()
                cmne.apply_cmne(x, pred)
                ms = (time.perf_counter() - t) / a.steps * 1e3
                print(
                    f"{name:>12s}  signals={n_sig:<3d} {ms:7.1f} ms/step  "
                    f"{ms / n_sig:6.2f} ms/step/signal"
                )


if __name__ == "__main__":
    main()
