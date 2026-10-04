"""Normalisation and windowing of source estimates."""

from __future__ import annotations

import numpy as np

__all__ = ["standardize", "rectified_zscore", "sliding_windows"]


def standardize(data, mean=None, std=None, axis=-1):
    """Z-score ``data`` along ``axis``.

    Constant rows (``std == 0``) are centred but not scaled, so the output never
    contains ``inf``/``nan``.

    Parameters
    ----------
    data : array_like
        Input array, e.g. ``(n_sources, n_times)``.
    mean, std : array_like | None
        Pre-computed statistics with ``axis`` removed. Computed from ``data`` when None.
    axis : int
        Axis along which the statistics are taken (time by default).
    """
    data = np.asarray(data)
    if mean is None:
        mean = data.mean(axis=axis)
    if std is None:
        std = data.std(axis=axis)
    mean = np.expand_dims(np.asarray(mean), axis)
    std = np.expand_dims(np.where(np.asarray(std) > 0, std, 1.0), axis)
    return ((data - mean) / std).astype(data.dtype, copy=False)


def rectified_zscore(source_data, axis=-1):
    """Rectify and z-score a source estimate per source, Eq. (9) of Dinh et al. (2021).

    Parameters
    ----------
    source_data : array_like, shape (n_sources, n_times)
        Signed (e.g. dSPM) source estimate.
    """
    return standardize(np.abs(source_data), axis=axis)


def sliding_windows(data, length, axis=0):
    """Return a zero-copy, read-only view of all windows of ``length`` along ``axis``.

    For ``data`` of shape ``(n_times, n_sources)`` and ``axis=0`` the result has
    shape ``(n_times - length + 1, length, n_sources)``.
    """
    view = np.lib.stride_tricks.sliding_window_view(data, length, axis=axis)
    return np.moveaxis(view, -1, axis + 1)
