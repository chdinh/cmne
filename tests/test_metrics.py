import numpy as np
import pytest

from cmne import peak_localization_error, source_snr, spatial_dispersion

POS = np.array([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])


def test_peak_error():
    est = np.array([0.1, -5, 0.2, 0.0])
    assert peak_localization_error(est, POS, 1) == 0
    assert peak_localization_error(est, POS, 3) == pytest.approx(np.sqrt(10))


def test_spatial_dispersion():
    assert spatial_dispersion(np.array([0, 1.0, 0, 0]), POS) == 0
    est = np.array([1.0, 1.0, 0, 0])
    assert spatial_dispersion(est, POS) == pytest.approx(0.5)
    assert spatial_dispersion(np.zeros(4), POS) == 0


def test_source_snr(rng):
    data = rng.normal(size=(3, 100)) * 0.1
    mask = np.zeros(100, bool)
    mask[40:60] = True
    data[:, mask] *= 10
    assert source_snr(data, [0, 1, 2], mask) > 50
    assert source_snr(data, [True, False, True], mask) > 50
