"""Spatial-fidelity and SNR metrics used in Dinh et al. (2021), Eqs. (22)-(26)."""

from __future__ import annotations

import numpy as np

__all__ = ["peak_localization_error", "spatial_dispersion", "source_snr"]


def _peak(estimate):
    return int(np.argmax(np.abs(estimate)))


def peak_localization_error(estimate, positions, true_index):
    """Distance between the true source and the peak of ``estimate`` (Eqs. 22-23).

    Parameters
    ----------
    estimate : ndarray, shape (n_sources,)
        Source estimate at one time point.
    positions : ndarray, shape (n_sources, 3)
        Source locations (same unit as the result).
    true_index : int
        Index of the active source.
    """
    pos = np.asarray(positions)
    return float(np.linalg.norm(pos[true_index] - pos[_peak(estimate)]))


def spatial_dispersion(estimate, positions):
    """Amplitude-weighted mean distance from the peak (Eqs. 24-25)."""
    pos = np.asarray(positions)
    a = np.abs(np.asarray(estimate, dtype=np.float64))
    total = a.sum()
    if total == 0:
        return 0.0
    d = np.linalg.norm(pos - pos[_peak(a)], axis=1)
    return float(d @ a / total)


def source_snr(source_data, roi, signal_mask):
    """Source-space SNR: variance inside vs. outside the signal period within ``roi`` (Eq. 26).

    Parameters
    ----------
    source_data : ndarray, shape (n_sources, n_times)
    roi : array_like of int | bool
        Sources of the region of interest (e.g. auditory cortex).
    signal_mask : array_like of bool, shape (n_times,)
        True during the signal (e.g. N1/P2) period.
    """
    a = np.abs(np.asarray(source_data))[np.asarray(roi)]
    m = np.asarray(signal_mask, dtype=bool)
    noise = a[:, ~m].std()
    return float(a[:, m].std() ** 2 / noise**2) if noise > 0 else float("inf")
