"""Contextual estimate: recursive LSTM re-weighting of source estimates (Eqs. 9-13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import rectified_zscore, standardize

__all__ = ["CMNEResult", "apply_cmne", "control_estimate"]


@dataclass
class CMNEResult:
    """Outputs of :func:`apply_cmne`.

    Arrays (or SourceEstimates) of shape ``(..., n_sources, n_times)``.

    Attributes
    ----------
    sensing : normalised input estimate ``q_t`` (Eq. 9).
    prediction : LSTM prediction of the next estimate (``q_t`` for the first ``k`` samples).
    cmne : contextual estimate ``b_t`` (Eqs. 10-13).
    """

    sensing: object
    prediction: object
    cmne: object


def _prepare(source, rectify):
    stc = None
    if hasattr(source, "data") and hasattr(source, "times"):
        stc, source = source, source.data
    data = np.asarray(source, dtype=np.float32)
    if data.ndim < 2:
        raise ValueError("Expected source data of shape (..., n_sources, n_times).")
    q = rectified_zscore(data) if rectify else standardize(data)
    return stc, np.ascontiguousarray(q, dtype=np.float32)


def _wrap(stc, arr):
    if stc is None:
        return arr
    out = stc.copy()
    out.data = arr.astype(np.float64)
    return out


def apply_cmne(
    source, predictor, look_back=None, rectify=None, normalize_weights=True, progress=False
):
    """Compute the contextual estimate of one or many source estimates.

    For ``t >= k`` the LSTM predicts the next estimate from the previous ``k``
    contextual estimates, and the current estimate is re-weighted by the
    rectified, max-normalised prediction (Eqs. 10-13 of Dinh et al., 2021).
    Several signals are processed as one batch, so applying CMNE to many
    trials costs little more than applying it to one.

    Parameters
    ----------
    source : ndarray | mne.SourceEstimate
        Linear (e.g. dSPM) source estimate(s), shape ``(n_sources, n_times)`` or
        ``(n_signals, n_sources, n_times)``.
    predictor : CMNEModel | OnnxPredictor | callable
        Maps windows ``(batch, k, n_sources)`` to predictions ``(batch, n_sources)``.
    look_back : int | None
        Window length ``k``; taken from the predictor's config when None.
    rectify : bool | None
        Rectify before z-scoring (Eq. 9); taken from the predictor's config when None.
    normalize_weights : bool
        Use ``|pred| / max|pred|`` as weights (Eq. 11). ``False`` multiplies by the raw
        prediction, as in the original 2017 scripts.
    progress : bool
        Print progress.

    Returns
    -------
    result : CMNEResult
        SourceEstimates if ``source`` was a SourceEstimate, arrays otherwise.
    """
    cfg = getattr(predictor, "config", None)
    k = look_back if look_back is not None else getattr(cfg, "look_back", None)
    if k is None:
        raise ValueError("look_back must be given for predictors without a config.")
    rect = rectify if rectify is not None else getattr(cfg, "rectify", True)
    predict = predictor.predict if hasattr(predictor, "predict") else predictor

    stc, q = _prepare(source, rect)
    in_shape = q.shape
    single = q.ndim == 2
    q = q[None] if single else q.reshape(-1, *q.shape[-2:])
    n_sig, n_src, n_times = q.shape
    if n_times <= k:
        raise ValueError(f"Need more than look_back={k} samples, got {n_times}.")

    # time-major working buffers: (n_signals, n_times, n_sources)
    q_t = np.ascontiguousarray(np.swapaxes(q, 1, 2))
    b = q_t.copy()
    pred = q_t.copy()
    eps = np.finfo(np.float32).tiny
    steps = n_times - k
    for i, t in enumerate(range(k, n_times)):
        p = np.asarray(predict(b[:, t - k : t]), dtype=np.float32)
        pred[:, t] = p
        if normalize_weights:
            w = np.abs(p)
            p = w / np.maximum(w.max(axis=1, keepdims=True), eps)
        b[:, t] = p * q_t[:, t]
        if progress and (i + 1) % max(1, steps // 10) == 0:
            print(f"CMNE step {i + 1}/{steps}", flush=True)

    shape = (n_src, n_times) if single else in_shape
    outs = [np.swapaxes(a, 1, 2).reshape(shape) for a in (q_t, pred, b)]
    return CMNEResult(*(_wrap(stc, a) for a in outs))


def control_estimate(source, look_back=80, rectify=True):
    """Control estimate of the paper: ``q_t`` times the mean of the previous ``k`` estimates.

    Mimics CMNE without the LSTM; computed for all ``t`` at once with a cumulative sum.
    """
    stc, q = _prepare(source, rectify)
    out = q.copy()
    csum = np.cumsum(q, axis=-1, dtype=np.float64)
    prev_mean = (
        csum[..., look_back - 1 : -1]
        - np.concatenate([np.zeros_like(csum[..., :1]), csum[..., : -look_back - 1]], axis=-1)
    ) / look_back
    out[..., look_back:] = q[..., look_back:] * prev_mean
    return _wrap(stc, out)
