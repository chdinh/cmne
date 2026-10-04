"""Command line interface: ``cmne train``, ``cmne apply``, ``cmne export``, ``cmne demo``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_epochs(args):
    import mne

    raw = mne.io.read_raw_fif(args.raw, preload=False, verbose=False)
    if any(ch == "eeg" for ch in raw.get_channel_types()):
        raw.set_eeg_reference(projection=True, verbose=False)
    events = mne.read_events(args.events) if args.events else mne.find_events(raw, verbose=False)
    picks = mne.pick_types(raw.info, meg=True, eeg=not args.meg_only, exclude="bads")
    reject = dict(mag=args.reject_mag, grad=args.reject_grad) if not args.no_reject else None
    epochs = mne.Epochs(
        raw,
        events,
        args.event_id,
        args.tmin,
        args.tmax,
        baseline=(None, 0) if args.tmin < 0 else None,
        picks=picks,
        reject=reject,
        preload=True,
        proj=True,
        verbose=False,
    )
    epochs.drop_bad(verbose=False)
    return epochs


def _split(n, test_idcs_file, test_fraction, seed):
    idx = np.arange(n)
    if test_idcs_file and Path(test_idcs_file).exists():
        test = np.loadtxt(test_idcs_file, dtype=int, ndmin=1)
        test = test[test < n]
    else:
        rng = np.random.default_rng(seed)
        test = np.sort(rng.choice(n, size=max(1, round(n * test_fraction)), replace=False))
    return np.setdiff1d(idx, test), test


def _cmd_train(args):
    from mne.minimum_norm import read_inverse_operator

    import cmne

    epochs = _load_epochs(args)
    inv = read_inverse_operator(args.inv, verbose=False)
    kernel = cmne.inverse_kernel(epochs.info, inv, lambda2=1 / args.snr**2, method=args.method)
    train, test = _split(len(epochs), args.test_idcs, args.test_fraction, args.seed)
    data = epochs.get_data(copy=False).astype(np.float32)
    model = cmne.fit(
        data[train],
        kernel,
        look_back=args.look_back,
        num_units=args.num_units,
        n_steps=args.steps,
        epochs_per_batch=args.epochs_per_batch,
        windows_per_epoch=args.windows_per_epoch,
        learning_rate=args.lr,
        validation=data[test],
        device=args.device,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    np.savetxt(out.with_suffix(".test-idcs.txt"), test, fmt="%d")
    print(f"Saved model to {out}")


def _cmd_export(args):
    import cmne

    print(f"Wrote {cmne.export_onnx(cmne.CMNEModel.load(args.model), args.output)}")


def _cmd_apply(args):
    from mne.io.constants import FIFF
    from mne.minimum_norm import apply_inverse, read_inverse_operator

    import cmne

    epochs = _load_epochs(args)
    inv = read_inverse_operator(args.inv, verbose=False)
    if args.n_average:
        rng = np.random.default_rng(args.seed)
        epochs = epochs[np.sort(rng.choice(len(epochs), args.n_average, replace=False))]
    evoked = epochs.average()
    fixed = inv["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI
    stc = apply_inverse(
        evoked,
        inv,
        1 / args.snr**2,
        args.method,
        pick_ori=None if fixed else "normal",
        verbose=False,
    )
    predictor = (
        cmne.OnnxPredictor(args.model)
        if str(args.model).endswith(".onnx")
        else cmne.CMNEModel.load(args.model, device=args.device)
    )
    res = cmne.apply_cmne(stc, predictor, progress=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sensing", "prediction", "cmne"):
        getattr(res, name).save(out / name, overwrite=True, verbose=False)
    cmne.control_estimate(stc, predictor.config.look_back).save(
        out / "control", overwrite=True, verbose=False
    )
    print(f"Saved source estimates to {out}")


def _cmd_demo(args):
    import cmne

    epochs, inv, truth, pos = cmne.simulate_data(seed=args.seed)
    kernel = cmne.inverse_kernel(epochs.info, inv)
    data = epochs.get_data(copy=False)
    model = cmne.fit(
        data[:30],
        kernel,
        look_back=20,
        num_units=64,
        n_steps=args.steps,
        epochs_per_batch=10,
        windows_per_epoch=8,
        learning_rate=3e-3,
        validation=data[30:],
        device="cpu",
        seed=args.seed,
    )
    src = kernel @ data[30:].mean(0)
    res = cmne.apply_cmne(src, model)
    peak_t = int(np.abs(truth[30:].mean(0)).max(0).argmax())
    true_src = int(np.abs(truth[30:].mean(0)[:, peak_t]).argmax())
    for name, est in (("dSPM", res.sensing), ("CMNE", res.cmne)):
        sd = cmne.spatial_dispersion(est[:, peak_t], pos) * 1e3
        pe = cmne.peak_localization_error(est[:, peak_t], pos, true_src) * 1e3
        print(f"{name:5s} peak error {pe:5.1f} mm   spatial dispersion {sd:5.1f} mm")


def _add_data_args(p):
    p.add_argument("--raw", required=True, help="Raw FIF file")
    p.add_argument("--inv", required=True, help="Inverse operator FIF file")
    p.add_argument("--events", help="Event FIF file (default: find events in raw)")
    p.add_argument("--event-id", type=int, default=1)
    p.add_argument("--tmin", type=float, default=-0.5)
    p.add_argument("--tmax", type=float, default=1.0)
    p.add_argument("--meg-only", action="store_true")
    p.add_argument("--reject-mag", type=float, default=4e-12)
    p.add_argument("--reject-grad", type=float, default=4000e-13)
    p.add_argument("--no-reject", action="store_true")
    p.add_argument("--method", default="dSPM", choices=["MNE", "dSPM", "sLORETA"])
    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cmne", description="Contextual Minimum-Norm Estimates")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="Train the CMNE LSTM on single-trial source estimates")
    _add_data_args(p)
    p.add_argument("--test-idcs", help="File with held-out epoch indices")
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--look-back", type=int, default=80)
    p.add_argument("--num-units", type=int, default=1280)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--epochs-per-batch", type=int, default=30)
    p.add_argument("--windows-per-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("-o", "--output", default="cmne_model.pt")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("apply", help="Apply a trained model to an averaged response")
    _add_data_args(p)
    p.add_argument("--model", required=True, help=".pt or .onnx model")
    p.add_argument("--n-average", type=int, default=20, help="Random epochs to average (0 = all)")
    p.add_argument("-o", "--output", default="cmne_results")
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser("export", help="Export a .pt model to ONNX")
    p.add_argument("model")
    p.add_argument("-o", "--output", default="cmne_model.onnx")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("demo", help="Train and apply CMNE on simulated data (CPU, < 1 min)")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
