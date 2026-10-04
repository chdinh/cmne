"""Quick start: train and apply CMNE on simulated EEG in under a minute on a laptop CPU.

Run:  python examples/quickstart_simulation.py
"""

import matplotlib.pyplot as plt
import numpy as np

import cmne

# 1. Simulated data: a burst travelling across five neighbouring sources
epochs, inv, truth, pos = cmne.simulate_data(n_epochs=60, seed=0)
data = epochs.get_data(copy=False)
train, test = data[:50], data[50:]

# 2. dSPM inverse kernel (one matrix, exact equivalent of apply_inverse_epochs)
kernel = cmne.inverse_kernel(epochs.info, inv, lambda2=1 / 9, method="dSPM")

# 3. Train a small LSTM predictor
model = cmne.fit(
    train,
    kernel,
    look_back=20,
    num_units=64,
    n_steps=300,
    epochs_per_batch=10,
    windows_per_epoch=8,
    learning_rate=3e-3,
    validation=test,
    device="cpu",
    seed=0,
)

# 4. Contextual estimate of the average of the held-out epochs
dspm = kernel @ test.mean(axis=0)
res = cmne.apply_cmne(dspm, model)
ctrl = cmne.control_estimate(dspm, look_back=20)

# 5. Compare spatial fidelity at the peak of the true activity
true_avg = np.abs(truth[50:].mean(axis=0))
t_peak = int(true_avg.max(axis=0).argmax())
src_peak = int(true_avg[:, t_peak].argmax())
for name, est in [("dSPM", res.sensing), ("Control", ctrl), ("CMNE", res.cmne)]:
    pe = cmne.peak_localization_error(est[:, t_peak], pos, src_peak) * 1e3
    sd = cmne.spatial_dispersion(est[:, t_peak], pos) * 1e3
    print(f"{name:8s} peak error {pe:5.1f} mm   spatial dispersion {sd:5.1f} mm")

# 6. Plot
times = epochs.times * 1e3
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True, layout="constrained")
for ax, (name, est) in zip(
    axes,
    [("dSPM", res.sensing), ("LSTM prediction", res.prediction), ("CMNE", res.cmne)],
    strict=True,
):
    est = np.abs(est) / np.abs(est).max()
    ax.plot(times, est.T, color="0.7", lw=0.5)
    ax.plot(times, est[src_peak], color="C3", lw=2, label="true peak source")
    ax.axvspan(times[0], times[20], color="C0", alpha=0.1, label="look-back (no correction)")
    ax.set(title=name, xlabel="time (ms)")
axes[0].set_ylabel("normalised |estimate|")
axes[0].legend(loc="upper left", fontsize=8)
fig.savefig("cmne_quickstart.png", dpi=150)
print("Saved cmne_quickstart.png")
