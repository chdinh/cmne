import numpy as np
import pytest

import cmne


@pytest.fixture(scope="session")
def sim():
    epochs, inv, truth, pos = cmne.simulate_data(n_epochs=24, n_times=100, n_sources=40, seed=1)
    kernel = cmne.inverse_kernel(epochs.info, inv)
    return dict(epochs=epochs, inv=inv, truth=truth, pos=pos, kernel=kernel)


@pytest.fixture(scope="session")
def small_model(sim):
    data = sim["epochs"].get_data(copy=False)
    return cmne.fit(
        data[:18],
        sim["kernel"],
        look_back=10,
        num_units=16,
        n_steps=40,
        epochs_per_batch=6,
        windows_per_epoch=4,
        learning_rate=3e-3,
        validation=data[18:],
        validate_every=20,
        device="cpu",
        seed=0,
        verbose=False,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)
