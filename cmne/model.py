"""LSTM prediction network, training and persistence."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .preprocessing import rectified_zscore, standardize

__all__ = ["CMNEConfig", "CMNEModel", "fit", "select_device"]


@dataclass
class CMNEConfig:
    """Hyper-parameters. Defaults follow Dinh et al. (2021): ``k = 80``, ``d = 1280``."""

    n_sources: int
    look_back: int = 80
    num_units: int = 1280
    rectify: bool = True


def select_device(device="auto"):
    """Return a :class:`torch.device`, picking CUDA, then Apple MPS, then CPU for ``"auto"``."""
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_network(cfg: CMNEConfig):
    import torch
    from torch import nn

    class _Net(nn.Module):
        """``k`` past estimates -> LSTM(d) -> dense -> next estimate (Fig. 2 of the paper)."""

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(cfg.n_sources, cfg.num_units, batch_first=True)
            self.head = nn.Linear(cfg.num_units, cfg.n_sources)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    return _Net()


@dataclass
class CMNEModel:
    """A trained CMNE prediction network plus its configuration and training history."""

    config: CMNEConfig
    network: object
    history: dict = field(default_factory=lambda: {"loss": [], "val_loss": []})

    @classmethod
    def create(cls, n_sources, look_back=80, num_units=1280, rectify=True, seed=None):
        """Initialise an untrained model."""
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        cfg = CMNEConfig(
            n_sources=int(n_sources),
            look_back=int(look_back),
            num_units=int(num_units),
            rectify=bool(rectify),
        )
        return cls(cfg, _build_network(cfg).eval())

    def predict(self, windows):
        """Predict the next estimate for ``windows`` of shape ``(batch, look_back, n_sources)``."""
        import torch

        net = self.network.eval()
        device = next(net.parameters()).device
        with torch.inference_mode():
            x = torch.as_tensor(np.asarray(windows, dtype=np.float32), device=device)
            return net(x).float().cpu().numpy()

    def save(self, fname):
        """Save weights, configuration and history (loadable with ``weights_only=True``)."""
        import torch

        state = {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": state,
                "history": json.dumps(self.history),
            },
            Path(fname),
        )

    @classmethod
    def load(cls, fname, device="cpu"):
        """Load a model written by :meth:`save`."""
        import torch

        ckpt = torch.load(Path(fname), map_location="cpu", weights_only=True)
        cfg = CMNEConfig(**ckpt["config"])
        net = _build_network(cfg)
        net.load_state_dict(ckpt["state_dict"])
        return cls(cfg, net.to(select_device(device)).eval(), json.loads(ckpt["history"]))


def _as_float32_epochs(epochs):
    """Return epochs data as ``float32 (n_epochs, n_channels, n_times)`` without a float64 copy."""
    if isinstance(epochs, np.ndarray):
        return np.asarray(epochs, dtype=np.float32)
    if epochs.preload:
        return epochs.get_data(copy=False).astype(np.float32, copy=False)
    epochs.drop_bad(verbose=False)
    out = np.empty((len(epochs), len(epochs.ch_names), len(epochs.times)), dtype=np.float32)
    for i, ep in enumerate(epochs):
        out[i] = ep
    return out


class _WindowSampler:
    """Draw minibatches of ``(look_back + 1)``-sample windows of normalised source estimates.

    Each minibatch picks ``epochs_per_batch`` epochs, maps them to source space with
    one batched matmul, normalises every epoch per source and draws
    ``windows_per_epoch`` random windows from each. Source estimates for the whole
    data set are never materialised, so memory stays at the size of the sensor data.
    """

    def __init__(self, data, kernel, look_back, rectify, rng):
        self.data = data
        self.kernel = np.ascontiguousarray(kernel, dtype=np.float32)
        self.look_back = look_back
        self.rectify = rectify
        self.rng = rng
        n_times = data.shape[-1]
        if n_times <= look_back:
            raise ValueError(
                f"Epochs have {n_times} samples; need more than look_back={look_back}."
            )

    def normalised_sources(self, idx):
        src = np.matmul(self.kernel, self.data[idx])  # (n, n_sources, n_times)
        src = rectified_zscore(src) if self.rectify else standardize(src)
        return np.swapaxes(src, 1, 2)  # time-major

    def sample(self, epochs_per_batch, windows_per_epoch):
        idx = self.rng.choice(
            len(self.data), size=epochs_per_batch, replace=epochs_per_batch > len(self.data)
        )
        src = self.normalised_sources(idx)
        n_start = src.shape[1] - self.look_back
        starts = self.rng.integers(0, n_start, size=(epochs_per_batch, windows_per_epoch))
        offs = starts[..., None] + np.arange(self.look_back + 1)
        win = src[np.arange(epochs_per_batch)[:, None, None], offs]
        win = win.reshape(-1, self.look_back + 1, src.shape[-1])
        return win[:, :-1], win[:, -1]


def fit(
    epochs,
    kernel,
    *,
    look_back=80,
    num_units=1280,
    n_steps=250,
    epochs_per_batch=30,
    windows_per_epoch=20,
    learning_rate=1e-3,
    rectify=True,
    validation=None,
    validate_every=25,
    model=None,
    device="auto",
    amp=True,
    seed=None,
    verbose=True,
):
    """Train the CMNE prediction network on single-trial source estimates.

    Parameters
    ----------
    epochs : mne.Epochs | ndarray, shape (n_epochs, n_channels, n_times)
        Training epochs. Converted once to ``float32``.
    kernel : ndarray, shape (n_sources, n_channels)
        Inverse kernel from :func:`cmne.inverse_kernel`.
    look_back : int
        Number of past samples ``k`` fed to the LSTM.
    num_units : int
        Hidden units ``d`` of the LSTM. Reduce (e.g. 256) for small GPUs/CPUs.
    n_steps : int
        Number of minibatch updates (paper: 250 for ASSR).
    epochs_per_batch, windows_per_epoch : int
        A minibatch holds ``epochs_per_batch * windows_per_epoch`` windows (paper: 30 x 20).
    rectify : bool
        Train on rectified z-scored estimates (Eq. 9), matching what :func:`apply_cmne` uses.
    validation : mne.Epochs | ndarray | None
        Held-out epochs; the validation loss is logged every ``validate_every`` steps.
    model : CMNEModel | None
        Continue training an existing model (e.g. fine-tuning on a new subject).
    device : str
        ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
    amp : bool
        Use bfloat16 autocast on CUDA for lower memory and faster training.
    seed : int | None
        Seed for reproducible sampling and initialisation.

    Returns
    -------
    model : CMNEModel
    """
    import torch

    rng = np.random.default_rng(seed)
    kernel = np.asarray(kernel)
    if model is None:
        model = CMNEModel.create(kernel.shape[0], look_back, num_units, rectify, seed=seed)
    cfg = model.config
    if cfg.n_sources != kernel.shape[0]:
        raise ValueError(f"Model expects {cfg.n_sources} sources, kernel has {kernel.shape[0]}.")

    sampler = _WindowSampler(_as_float32_epochs(epochs), kernel, cfg.look_back, cfg.rectify, rng)
    val_sampler = None
    if validation is not None:
        val_sampler = _WindowSampler(
            _as_float32_epochs(validation),
            kernel,
            cfg.look_back,
            cfg.rectify,
            np.random.default_rng(0 if seed is None else seed + 1),
        )

    dev = select_device(device)
    net = model.network.to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    use_amp = amp and dev.type == "cuda"
    loss_fn = torch.nn.MSELoss()

    def to_dev(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev, non_blocking=True)

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        x, y = sampler.sample(epochs_per_batch, windows_per_epoch)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = loss_fn(net(to_dev(x)).float(), to_dev(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.history["loss"].append(loss.item())

        log = verbose and (step == 1 or step % max(1, n_steps // 20) == 0 or step == n_steps)
        if val_sampler is not None and (step % validate_every == 0 or step == n_steps):
            net.eval()
            with torch.inference_mode():
                vx, vy = val_sampler.sample(epochs_per_batch, windows_per_epoch)
                val = float(loss_fn(net(to_dev(vx)).float(), to_dev(vy)))
            net.train()
            model.history["val_loss"].append([step, val])
        if log:
            msg = (
                f"step {step:>{len(str(n_steps))}}/{n_steps}  loss {model.history['loss'][-1]:.4f}"
            )
            if model.history["val_loss"] and model.history["val_loss"][-1][0] == step:
                msg += f"  val {model.history['val_loss'][-1][1]:.4f}"
            print(f"{msg}  ({time.perf_counter() - t0:.0f}s, {dev.type})", flush=True)

    if not math.isfinite(model.history["loss"][-1]):
        raise RuntimeError("Training diverged (non-finite loss); lower the learning rate.")
    net.eval()
    return model
