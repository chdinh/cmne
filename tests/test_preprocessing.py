import numpy as np
import pytest

from cmne import rectified_zscore, sliding_windows, standardize


def test_standardize_zero_mean_unit_std(rng):
    x = rng.normal(3.0, 2.0, size=(5, 200))
    z = standardize(x)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)


def test_standardize_given_stats_and_batch(rng):
    x = rng.normal(size=(3, 4, 50))
    z = standardize(x, mean=np.zeros((3, 4)), std=np.full((3, 4), 2.0))
    np.testing.assert_allclose(z, x / 2)


def test_standardize_constant_row_is_finite():
    x = np.vstack([np.ones(10), np.arange(10.0)])
    z = standardize(x)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[0], 0)


def test_standardize_keeps_float32():
    assert standardize(np.ones((2, 5), np.float32)).dtype == np.float32


def test_rectified_zscore_is_eq9(rng):
    x = rng.normal(size=(4, 30))
    a = np.abs(x)
    np.testing.assert_allclose(
        rectified_zscore(x), (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    )


@pytest.mark.parametrize("length", [1, 3, 10])
def test_sliding_windows(length):
    x = np.arange(40).reshape(10, 4)
    w = sliding_windows(x, length)
    assert w.shape == (10 - length + 1, length, 4)
    for i in range(w.shape[0]):
        np.testing.assert_array_equal(w[i], x[i : i + length])
    assert np.shares_memory(w, x)
