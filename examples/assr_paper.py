"""Reproduce the ASSR analysis of Dinh et al. (2021) with the public data set (~4.3 GB).

Paper settings: dSPM (SNR 3), k = 80, d = 1280, 250 minibatches of 30 x 20 windows,
85/15 train/test split, evaluation on the average of 20 held-out epochs.

Run:  python examples/assr_paper.py --data ~/cmne_data [--units 256 --steps 100]  # smaller/faster
"""

import argparse
from pathlib import Path

import mne
import numpy as np
from mne.minimum_norm import apply_inverse, read_inverse_operator

import cmne

p = argparse.ArgumentParser()
p.add_argument("--data", default="~/cmne_data")
p.add_argument("--out", default="cmne_assr_results")
p.add_argument("--units", type=int, default=1280)
p.add_argument("--look-back", type=int, default=80)
p.add_argument("--steps", type=int, default=250)
p.add_argument("--device", default="auto")
args = p.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)

# --- Data (downloaded once) --------------------------------------------------------------
files = cmne.fetch_assr(args.data)
raw = mne.io.read_raw_fif(files["raw"], preload=False)
raw.set_eeg_reference(projection=True)
events = mne.read_events(files["eve"])
picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude="bads")
epochs = mne.Epochs(
    raw,
    events,
    event_id=1,
    tmin=-0.5,
    tmax=1.0,
    baseline=(None, 0),
    picks=picks,
    reject=dict(mag=4e-12, grad=4000e-13),
    preload=True,
    proj=True,
)
inv = read_inverse_operator(files["inv"])

test_idx = np.loadtxt(files["test_idcs"], dtype=int)
test_idx = test_idx[test_idx < len(epochs)]
train_idx = np.setdiff1d(np.arange(len(epochs)), test_idx)
print(f"{len(train_idx)} training / {len(test_idx)} test epochs")

# --- Train -------------------------------------------------------------------------------
kernel = cmne.inverse_kernel(epochs.info, inv, lambda2=1 / 9, method="dSPM")
data = epochs.get_data(copy=False).astype(np.float32)
model = cmne.fit(
    data[train_idx],
    kernel,
    look_back=args.look_back,
    num_units=args.units,
    n_steps=args.steps,
    validation=data[test_idx],
    device=args.device,
    seed=42,
)
model.save(out / "cmne_assr.pt")

# --- Evaluate on an average of 20 held-out epochs ----------------------------------------
rng = np.random.default_rng(42)
evoked = epochs[rng.choice(test_idx, 20, replace=False)].average()
stc = apply_inverse(evoked, inv, 1 / 9, "dSPM", pick_ori="normal")
res = cmne.apply_cmne(stc, model, progress=True)
ctrl = cmne.control_estimate(stc, look_back=args.look_back)

for name, est in [
    ("dSPM", res.sensing),
    ("prediction", res.prediction),
    ("CMNE", res.cmne),
    ("control", ctrl),
]:
    est.save(out / name, overwrite=True)
print(f"Results written to {out}/  (view e.g. with stc.plot(subject=..., subjects_dir=...))")
