"""Linear inverse kernels (MNE/dSPM/sLORETA) as plain matrices."""

from __future__ import annotations

import numpy as np

__all__ = ["inverse_kernel"]


def inverse_kernel(
    info,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="dSPM",
    pick_ori="auto",
    nave=1,
    dtype=np.float32,
):
    """Assemble the inverse operator into a single ``(n_sources, n_channels)`` matrix.

    The kernel includes SSP projection, whitening and noise normalisation, so
    ``kernel @ epoch_data`` is identical to :func:`mne.minimum_norm.apply_inverse_epochs`
    but runs as one BLAS call without per-epoch MNE overhead. Columns of channels
    not used by the inverse are zero.

    Parameters
    ----------
    info : mne.Info
        Measurement info of the data the kernel will be applied to.
    inverse_operator : mne.minimum_norm.InverseOperator
        Inverse operator.
    lambda2 : float
        Regularisation, ``1 / SNR**2`` (paper: SNR = 3).
    method : "MNE" | "dSPM" | "sLORETA"
        Inverse method.
    pick_ori : "auto" | "normal" | None
        ``"auto"`` uses ``None`` for fixed-orientation operators and ``"normal"`` otherwise.
        Free orientations combined by vector norm are non-linear and not supported.
    nave : int
        Number of averages assumed for noise normalisation. CMNE z-scores every
        estimate, so this only rescales the output.
    """
    import mne
    from mne.io.constants import FIFF
    from mne.minimum_norm import apply_inverse

    if pick_ori == "auto":
        fixed = inverse_operator["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI
        pick_ori = None if fixed else "normal"

    n_ch = len(info["ch_names"])
    identity = mne.EvokedArray(np.eye(n_ch), info, tmin=0.0, nave=nave, verbose=False)
    stc = apply_inverse(
        identity, inverse_operator, lambda2, method, pick_ori=pick_ori, verbose=False
    )
    if stc.data.ndim != 2:
        raise ValueError("Vector source estimates are not supported; use pick_ori='normal'.")
    return np.ascontiguousarray(stc.data, dtype=dtype)
