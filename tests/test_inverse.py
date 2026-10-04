import numpy as np
import pytest
from mne.minimum_norm import apply_inverse_epochs

import cmne


@pytest.mark.parametrize("method", ["MNE", "dSPM", "sLORETA"])
def test_kernel_matches_mne(sim, method):
    epochs = sim["epochs"][:3]
    k = cmne.inverse_kernel(epochs.info, sim["inv"], method=method, dtype=np.float64)
    stcs = apply_inverse_epochs(epochs, sim["inv"], 1 / 9, method, nave=1, verbose=False)
    for ep, stc in zip(epochs.get_data(copy=False), stcs, strict=True):
        np.testing.assert_allclose(k @ ep, stc.data, rtol=1e-6, atol=1e-8 * np.abs(stc.data).max())


def test_kernel_shape_and_dtype(sim):
    k = sim["kernel"]
    assert k.shape == (sim["inv"]["nsource"], len(sim["epochs"].ch_names))
    assert k.dtype == np.float32 and k.flags.c_contiguous
