import mne
import numpy as np
import pytest
from mne.minimum_norm import write_inverse_operator

from cmne.cli import main


@pytest.fixture(scope="module")
def fif_files(sim, tmp_path_factory):
    d = tmp_path_factory.mktemp("fif")
    epochs = sim["epochs"]
    n_ep, n_ch, n_t = epochs.get_data(copy=False).shape
    gap = 20
    data = np.zeros((n_ch + 1, n_ep * (n_t + gap)))
    events = []
    for i, ep in enumerate(epochs.get_data(copy=False)):
        start = i * (n_t + gap) + gap
        data[:n_ch, start : start + n_t] = ep
        events.append([start, 0, 1])
    info = mne.create_info(
        [*epochs.ch_names, "STI"], epochs.info["sfreq"], [*["eeg"] * n_ch, "stim"]
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(epochs.get_montage())
    raw.save(d / "sim_raw.fif", verbose=False)
    mne.write_events(d / "sim-eve.fif", np.array(events), verbose=False)
    write_inverse_operator(d / "sim-inv.fif", sim["inv"], verbose=False)
    return d


def _data_args(d):
    sfreq = 200.0
    return [
        "--raw",
        str(d / "sim_raw.fif"),
        "--inv",
        str(d / "sim-inv.fif"),
        "--events",
        str(d / "sim-eve.fif"),
        "--tmin",
        "0",
        "--tmax",
        str(99 / sfreq),
        "--no-reject",
        "--device",
        "cpu",
    ]


def test_train_export_apply(fif_files, tmp_path):
    model = tmp_path / "m.pt"
    main(
        [
            "train",
            *_data_args(fif_files),
            "--look-back",
            "10",
            "--num-units",
            "8",
            "--steps",
            "3",
            "--epochs-per-batch",
            "4",
            "--windows-per-epoch",
            "2",
            "-o",
            str(model),
        ]
    )
    assert model.exists() and model.with_suffix(".test-idcs.txt").exists()

    onnx_model = tmp_path / "m.onnx"
    main(["export", str(model), "-o", str(onnx_model)])
    assert onnx_model.exists()

    for m in (model, onnx_model):
        out = tmp_path / f"res_{m.suffix[1:]}"
        main(
            ["apply", *_data_args(fif_files), "--model", str(m), "--n-average", "5", "-o", str(out)]
        )
        for name in ("sensing", "prediction", "cmne", "control"):
            assert list(out.glob(f"{name}*")), name
