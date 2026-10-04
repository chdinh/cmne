"""ONNX export and a PyTorch-free ONNX Runtime predictor for deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import CMNEConfig

__all__ = ["export_onnx", "OnnxPredictor"]

_META_KEY = "cmne_config"


def export_onnx(model, fname, opset=18):
    """Export a :class:`~cmne.CMNEModel` to ONNX with a dynamic batch dimension.

    The configuration is stored in the model metadata, so :class:`OnnxPredictor`
    needs only the ``.onnx`` file.
    """
    import copy
    import warnings
    from dataclasses import asdict

    import onnx
    import torch

    cfg = model.config
    net = copy.deepcopy(model.network).to("cpu").eval()
    dummy = torch.zeros(2, cfg.look_back, cfg.n_sources)
    fname = Path(fname)
    with warnings.catch_warnings():
        # torch.export emits internal deprecation/buffer notices for nn.LSTM
        warnings.simplefilter("ignore", (FutureWarning, UserWarning, DeprecationWarning))
        torch.onnx.export(
            net,
            (dummy,),
            str(fname),
            input_names=["windows"],
            output_names=["prediction"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            opset_version=opset,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    proto = onnx.load(str(fname))
    meta = proto.metadata_props.add()
    meta.key, meta.value = _META_KEY, json.dumps(asdict(cfg))
    onnx.save(proto, str(fname))
    return fname


class OnnxPredictor:
    """Run a CMNE network with ONNX Runtime; drop-in predictor for :func:`cmne.apply_cmne`.

    Parameters
    ----------
    fname : path-like
        ``.onnx`` file written by :func:`export_onnx`.
    providers : list of str | None
        ONNX Runtime execution providers; defaults to CUDA if available, else CPU.
    num_threads : int | None
        Intra-op threads for the CPU provider.
    """

    def __init__(self, fname, providers=None, num_threads=None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            opts.intra_op_num_threads = int(num_threads)
        if providers is None:
            # CoreML/DirectML only take parts of the LSTM graph and end up slower than CPU
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()
            ]
        self.session = ort.InferenceSession(str(fname), sess_options=opts, providers=providers)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.config = CMNEConfig(**json.loads(meta[_META_KEY]))
        self._input = self.session.get_inputs()[0].name

    def predict(self, windows):
        x = np.ascontiguousarray(windows, dtype=np.float32)
        return self.session.run(None, {self._input: x})[0]

    __call__ = predict
