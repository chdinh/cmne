"""Contextual Minimum-Norm Estimates (CMNE).

LSTM-based contextual re-weighting of MEG/EEG source estimates
(Dinh et al., Front. Neurosci. 2021, doi:10.3389/fnins.2021.552666).
"""

from ._version import __version__
from .core import CMNEResult, apply_cmne, control_estimate
from .datasets import fetch_assr, simulate_data
from .inverse import inverse_kernel
from .metrics import peak_localization_error, source_snr, spatial_dispersion
from .model import CMNEConfig, CMNEModel, fit, select_device
from .preprocessing import rectified_zscore, sliding_windows, standardize

__all__ = [
    "__version__",
    "CMNEConfig",
    "CMNEModel",
    "CMNEResult",
    "OnnxPredictor",
    "apply_cmne",
    "control_estimate",
    "export_onnx",
    "fetch_assr",
    "fit",
    "inverse_kernel",
    "peak_localization_error",
    "rectified_zscore",
    "select_device",
    "simulate_data",
    "sliding_windows",
    "source_snr",
    "spatial_dispersion",
    "standardize",
]


def __getattr__(name):
    # ONNX support is optional; import lazily so onnxruntime is not required.
    if name in ("export_onnx", "OnnxPredictor"):
        from . import onnx as _onnx

        return getattr(_onnx, name)
    raise AttributeError(f"module 'cmne' has no attribute {name!r}")
