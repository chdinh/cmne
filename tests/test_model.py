import numpy as np
import pytest

import cmne
from cmne.model import _WindowSampler


def test_sampler_shapes_and_targets(sim, rng):
    data = sim["epochs"].get_data(copy=False).astype(np.float32)
    s = _WindowSampler(data, sim["kernel"], look_back=7, rectify=True, rng=rng)
    x, y = s.sample(4, 3)
    n_src = sim["kernel"].shape[0]
    assert x.shape == (12, 7, n_src) and y.shape == (12, n_src)
    assert x.dtype == np.float32
    # every (window, target) pair is a contiguous 8-sample slice of a normalised epoch
    src = s.normalised_sources(np.arange(len(data)))
    seq = np.concatenate([x[0], y[:1]])
    found = any(
        np.allclose(src[e, t : t + 8], seq)
        for e in range(len(data))
        for t in range(src.shape[1] - 7)
    )
    assert found


def test_sampler_rejects_short_epochs(sim, rng):
    with pytest.raises(ValueError, match="look_back"):
        _WindowSampler(np.zeros((2, 3, 5), np.float32), np.zeros((4, 3)), 5, True, rng)


def test_training_reduces_loss(small_model):
    loss = small_model.history["loss"]
    assert len(loss) == 40
    assert np.mean(loss[-10:]) < np.mean(loss[:5])
    assert [s for s, _ in small_model.history["val_loss"]] == [20, 40]


def test_fit_is_reproducible(sim):
    data = sim["epochs"].get_data(copy=False)[:6]
    kw = dict(
        look_back=5,
        num_units=8,
        n_steps=3,
        epochs_per_batch=2,
        windows_per_epoch=2,
        device="cpu",
        seed=3,
        verbose=False,
    )
    a = cmne.fit(data, sim["kernel"], **kw).history["loss"]
    b = cmne.fit(data, sim["kernel"], **kw).history["loss"]
    assert a == b


def test_fit_accepts_mne_epochs(sim):
    m = cmne.fit(
        sim["epochs"][:4],
        sim["kernel"],
        look_back=5,
        num_units=8,
        n_steps=2,
        epochs_per_batch=2,
        windows_per_epoch=2,
        device="cpu",
        verbose=False,
    )
    assert m.config.n_sources == sim["kernel"].shape[0]


def test_fit_rejects_wrong_kernel(small_model, sim):
    with pytest.raises(ValueError, match="sources"):
        cmne.fit(
            sim["epochs"].get_data(copy=False)[:2],
            sim["kernel"][:-1],
            model=small_model,
            n_steps=1,
            device="cpu",
            verbose=False,
        )


def test_save_load_roundtrip(small_model, tmp_path, rng):
    f = tmp_path / "m.pt"
    small_model.save(f)
    loaded = cmne.CMNEModel.load(f)
    assert loaded.config == small_model.config
    assert loaded.history == small_model.history
    w = rng.normal(size=(3, 10, small_model.config.n_sources)).astype(np.float32)
    np.testing.assert_allclose(loaded.predict(w), small_model.predict(w), rtol=1e-6)


def test_select_device():
    assert cmne.select_device("cpu").type == "cpu"
    assert cmne.select_device().type in {"cpu", "cuda", "mps"}


def test_cli_demo(capsys):
    from cmne.cli import main

    main(["demo", "--steps", "5"])
    out = capsys.readouterr().out
    assert "CMNE" in out and "dSPM" in out
