import numpy as np
import pytest

import cmne
from cmne import apply_cmne, control_estimate, rectified_zscore


class Echo:
    """Predicts the last sample of each window."""

    def __call__(self, w):
        return w[:, -1]


def _reference_cmne(q, predict, k):
    """Literal, loop-based transcription of Eqs. (10)-(13)."""
    n_src, n_t = q.shape
    b = q.T.copy()
    for t in range(k, n_t):
        p = predict(b[None, t - k : t])[0]
        w = np.abs(p) / np.abs(p).max()
        b[t] = w * q[:, t]
    return b.T


def test_matches_reference(rng):
    x = rng.normal(size=(7, 40))
    lin = rng.normal(size=(7, 7)).astype(np.float32)

    def predict(w):
        return np.tanh(w.mean(axis=1) @ lin)

    res = apply_cmne(x, predict, look_back=5)
    ref = _reference_cmne(rectified_zscore(x).astype(np.float32), predict, 5)
    np.testing.assert_allclose(res.cmne, ref, rtol=1e-4, atol=1e-5)


def test_first_k_samples_are_sensing(rng):
    x = rng.normal(size=(4, 30))
    res = apply_cmne(x, Echo(), look_back=8)
    np.testing.assert_array_equal(res.cmne[:, :8], res.sensing[:, :8])
    np.testing.assert_array_equal(res.prediction[:, :8], res.sensing[:, :8])


def test_weights_bounded(rng):
    x = rng.normal(size=(6, 50))
    res = apply_cmne(x, lambda w: 10 * w[:, -1], look_back=4)
    assert np.all(np.abs(res.cmne) <= np.abs(res.sensing) + 1e-6)


def test_batch_equals_individual(rng):
    xs = rng.normal(size=(3, 5, 25))
    batch = apply_cmne(xs, Echo(), look_back=4).cmne
    assert batch.shape == xs.shape
    for i in range(3):
        np.testing.assert_allclose(batch[i], apply_cmne(xs[i], Echo(), look_back=4).cmne)


def test_raw_prediction_mode(rng):
    x = rng.normal(size=(3, 20))
    res = apply_cmne(x, Echo(), look_back=2, normalize_weights=False)
    t = 10
    np.testing.assert_allclose(res.cmne[:, t], res.prediction[:, t] * res.sensing[:, t])


def test_zero_prediction_is_finite(rng):
    res = apply_cmne(rng.normal(size=(3, 20)), lambda w: np.zeros(w.shape[::2]), look_back=3)
    assert np.isfinite(res.cmne).all()


def test_too_short_raises(rng):
    with pytest.raises(ValueError, match="look_back"):
        apply_cmne(rng.normal(size=(3, 5)), Echo(), look_back=5)


def test_requires_look_back(rng):
    with pytest.raises(ValueError, match="look_back"):
        apply_cmne(rng.normal(size=(3, 20)), Echo())


def test_source_estimate_roundtrip(sim, small_model):
    from mne.minimum_norm import apply_inverse

    evoked = sim["epochs"].average()
    stc = apply_inverse(evoked, sim["inv"], 1 / 9, "dSPM", verbose=False)
    res = apply_cmne(stc, small_model)
    assert type(res.cmne) is type(stc)
    assert res.cmne.data.shape == stc.data.shape
    np.testing.assert_array_equal(res.cmne.times, stc.times)


def test_control_estimate_matches_loop(rng):
    x = rng.normal(size=(4, 30))
    k = 6
    q = rectified_zscore(x)
    ref = q.copy()
    for t in range(k, 30):
        ref[:, t] = q[:, t] * q[:, t - k : t].mean(axis=1)
    np.testing.assert_allclose(control_estimate(x, k), ref, rtol=1e-5, atol=1e-6)


def test_package_exports():
    for name in cmne.__all__:
        assert getattr(cmne, name) is not None
