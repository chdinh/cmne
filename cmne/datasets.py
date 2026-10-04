"""Download helper for the ASSR data set used in the paper."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

__all__ = ["ASSR_FILES", "fetch_assr", "simulate_data"]

ASSR_URL = "https://osf.io/zhqnp/download"
ASSR_FILES = {
    "raw": "assr_270LP_fs900_raw.fif",
    "inv": "assr_270LP_fs900_raw-ico-4-meg-eeg-inv.fif",
    "eve": "assr_270LP_fs900_raw-eve.fif",
    "test_idcs": "assr_270LP_fs900_raw-test-idcs.txt",
}
_EXTRA = ("assr_270LP_fs900_raw-1.fif", "assr_270LP_fs900_raw-2.fif")


def fetch_assr(path="~/cmne_data", progress=True):
    """Download and unpack the ASSR MEG/EEG data set (~4.3 GB) once.

    Returns
    -------
    files : dict of str -> pathlib.Path
        Paths keyed like :data:`ASSR_FILES`.
    """
    import pooch

    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    wanted = [*ASSR_FILES.values(), *_EXTRA]
    if not all((path / f).exists() for f in wanted):
        archive = pooch.retrieve(
            ASSR_URL, known_hash=None, fname="ASSR.zip", path=path / "tmp", progressbar=progress
        )
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in wanted:
                    with zf.open(member) as src, open(path / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        shutil.rmtree(path / "tmp", ignore_errors=True)
    return {k: path / v for k, v in ASSR_FILES.items()}


def simulate_data(
    n_epochs=40, n_times=160, n_channels=32, n_sources=60, sfreq=200.0, snr=3.0, seed=0
):
    """Small EEG simulation on a sphere model with a source that travels across the cortex.

    Useful for tests, tutorials and checking an installation in seconds.

    Returns
    -------
    epochs : mne.EpochsArray
    inverse_operator : mne.minimum_norm.InverseOperator
        Fixed-orientation inverse operator.
    truth : ndarray, shape (n_epochs, n_sources, n_times)
        Simulated source activity.
    positions : ndarray, shape (n_sources, 3)
        Source positions in metres.
    """
    import mne
    import numpy as np
    from mne.minimum_norm import make_inverse_operator

    rng = np.random.default_rng(seed)
    name = (
        "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"
    )
    montage = mne.channels.make_standard_montage(name)
    keep = np.linspace(0, len(montage.ch_names) - 1, n_channels).round().astype(int)
    info = mne.create_info([montage.ch_names[i] for i in keep], sfreq, "eeg")
    info.set_montage(montage, on_missing="ignore")
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09, verbose=False)

    # Sources on a shell inside the head, normals pointing outwards
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = np.linspace(0.95, 0.05, n_sources)  # upper hemisphere
    r = np.sqrt(1 - z**2)
    nn = np.c_[
        r * np.cos(golden * np.arange(n_sources)), r * np.sin(golden * np.arange(n_sources)), z
    ]
    rr = np.asarray(sphere["r0"]) + 0.06 * nn
    src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=nn), sphere=sphere, verbose=False)
    fwd = mne.make_forward_solution(info, None, src, sphere, verbose=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True, verbose=False)
    n_src = fwd["nsource"]
    gain = fwd["sol"]["data"]

    # A Gabor burst sweeping across five neighbouring sources (cf. Fig. 3 of the paper)
    t = np.arange(n_times) / sfreq
    path = np.argsort(np.linalg.norm(fwd["source_rr"] - fwd["source_rr"][0], axis=1))[:5]
    truth = np.zeros((n_epochs, n_src, n_times))
    onset = t[n_times // 2]
    for j, s in enumerate(path):
        tc = onset + 0.03 * j
        burst = np.exp(-(((t - tc) / 0.02) ** 2)) * np.cos(2 * np.pi * 10 * (t - tc))
        truth[:, s] = 50e-9 * burst
    truth += 10e-9 * rng.standard_normal(truth.shape) / np.sqrt(n_src)

    sens = np.einsum("cs,est->ect", gain, truth)
    noise_std = sens.std() / snr
    sens += noise_std * rng.standard_normal(sens.shape)
    epochs = mne.EpochsArray(sens, info, tmin=0.0, verbose=False)
    epochs.set_eeg_reference(projection=True, verbose=False).apply_proj(verbose=False)

    cov = mne.make_ad_hoc_cov(epochs.info, std=dict(eeg=noise_std), verbose=False)
    inv = make_inverse_operator(
        epochs.info, fwd, cov, loose=0.0, fixed=True, depth=None, verbose=False
    )
    return epochs, inv, truth, fwd["source_rr"]
