"""Window datasets over the real and simulated trajectories.

At the evaluation resolution the whole corpus fits in RAM: 58 real training
trajectories at 32x64x2 channels are ~1.5 GB, and the 72 simulated ones ~3 GB.
Preloading means no DataLoader workers, which is what we want anyway -- the
submission rules force ``num_workers=0``, so keeping training on the same footing
avoids surprises.

Only u and v are carried. Simulated p is dropped: it is not scored, and giving
the network an output channel that exists during pretraining and vanishes during
finetuning is a transfer problem we do not need. (Using sim p as an *auxiliary*
pretraining target is a separate idea worth trying later.)

Trajectories are stored as (T, C, H, W) so a window slice needs no transpose, and
the per-window normalization scale is assembled from precomputed per-frame sums
rather than re-reducing the field every time -- that alone is most of the
throughput, since a naive version spends more time normalizing than training.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import EPS
from .splits import Traj, partition

T_IN = T_OUT = 20
SUB = 2


def load_traj(path: str, with_p: bool = False, phase: int = 0) -> np.ndarray:
    """(T, C, 32, 64) float32 on the evaluation grid; C = 3 with pressure.

    Real files have no ``p`` dataset -- pressure is not measured -- so the third
    channel is zeros there and the auxiliary loss is masked off for real windows.

    ``phase`` picks which of the SUB*SUB decimation phases of the raw grid to
    keep; 0 is the evaluation grid. The other phases hold almost no extra signal
    (0.2% of energy above the coarse Nyquist) but a substantially independent
    PIV-noise realization of the same band-limited field (measured 2026-08-31:
    residual RMS ~0.006, constant across regimes) -- which is exactly what makes
    them usable as training-time noise-resampling augmentation.
    """
    import h5py
    po, pe = divmod(phase, SUB)
    with h5py.File(path, "r") as h:
        u = h["u"][:, po::SUB, pe::SUB].astype(np.float32)
        v = h["v"][:, po::SUB, pe::SUB].astype(np.float32)
        if not with_p:
            return np.stack([u, v], axis=1)
        pr = (h["p"][:, po::SUB, pe::SUB].astype(np.float32) if "p" in h
              else np.zeros_like(u))
    return np.stack([u, v, pr], axis=1)


# Tail evaluation, added 2026-08-20. Every split in this project holds out whole
# *regimes*; this one holds out the *tail* of every trajectory instead, so training
# sees the early frames of all 81 and evaluation sees the late frames of the same 81.
# Distillation is what forced it: +0.146 on held-out regimes over 9 folds and -0.159
# on the board, and no regime split could reproduce that.
#
# **It was built on a premise that has since been retired.** The reading at the time
# was that the hidden development set is in-grid, so matching regimes exactly was the
# missing property. Cell-counting on 2026-08-20 argued otherwise: the hidden set is
# most likely 19 withheld (Re, AoA) cells, scoring *unseen* combinations, exactly as the
# private set does (`docs/archive/FINDINGS_2026-09-10.md` §0a, `splits.HELDOUT_CELLS`). So this holdout is a
# same-regime, later-frames diagnostic -- useful because a shipping candidate trains
# on every trajectory and has no honest split at all (`tools/honest_pipeline.sh`
# builds on it) -- and not a model of the board. The mimic fold set is that.
#
# The gap is deliberate and load-bearing. Without it the last training window's
# target would overlap the first evaluation window's input, and the shedding
# period is ~9 frames -- a model that had seen frame t-1 would be scored on frame
# t. One full window (T_IN + T_OUT) is dropped between the two.
HELDOUT_TAIL = float(os.environ.get("REALISTIX_HELDOUT_TAIL", "0.0"))


def frame_range(n_frames: int, frames: str) -> tuple[int, int]:
    """(first, last) frame index usable for windows under the tail holdout."""
    if not HELDOUT_TAIL or frames == "all":
        return 0, n_frames
    cut = int(round(n_frames * (1.0 - HELDOUT_TAIL)))
    if frames == "head":
        return 0, cut
    if frames == "tail":
        return min(cut + T_IN + T_OUT, n_frames), n_frames
    raise ValueError(f"frames must be all/head/tail, got {frames!r}")


class WindowDataset(Dataset):
    """(input, target, scale) window triples drawn from preloaded trajectories.

    ``stride`` controls how densely windows are enumerated. Training uses 1;
    anything used for scoring must use >= T_IN + T_OUT so that no window's target
    lands inside another window's input.
    """

    def __init__(self, trajs: list[Traj], split: str, stride: int = 1,
                 normalized: bool = True, with_p: bool = False,
                 frames: str = "all"):
        self.normalized = normalized
        self.with_p = with_p
        self.fields: list[np.ndarray] = []
        self.frame_sumsq: list[np.ndarray] = []
        self.nonzero: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []

        self.trajs = list(trajs)
        for t in trajs:
            f = load_traj(t.path(split), with_p=with_p)
            i = len(self.fields)
            self.fields.append(f)
            # Numerator: invalid cells (airfoil body, out-of-FOV wedge) are stored as
            # exact zeros, so they contribute nothing to a sum of squares whether or
            # not they are masked. Per-frame totals are therefore mask-free and can
            # be precomputed once.
            self.frame_sumsq.append(np.sum(f[:, :2] ** 2, axis=(1, 2, 3)))  # (T,)
            # Denominator: the count of valid cells does depend on the mask, and the
            # mask must be the *window-local* one -- at inference we only ever see
            # one window, so a trajectory-wide mask would train on a scale the
            # submission cannot reproduce (it differs by 1.5-3.5% in practice).
            self.nonzero.append(np.any(np.abs(f[:, :2]) >= EPS, axis=1))    # (T, H, W)
            lo, hi = frame_range(f.shape[0], frames)
            for s in range(lo, hi - (T_IN + T_OUT) + 1, stride):
                self.index.append((i, s))

    def __len__(self) -> int:
        return len(self.index)

    def scale_of(self, fi: int, s: int) -> np.float32:
        """RMS speed over the *input* window's valid cells.

        The target is not visible at inference, so it must not influence the scale.
        """
        tot = self.frame_sumsq[fi][s:s + T_IN].sum()
        n_valid = max(int(self.nonzero[fi][s:s + T_IN].any(axis=0).sum()), 1)
        return np.float32(np.sqrt(tot / (T_IN * n_valid)) + EPS)

    def __getitem__(self, i: int):
        fi, s = self.index[i]
        f = self.fields[fi]
        scale = self.scale_of(fi, s) if self.normalized else np.float32(1.0)
        x = f[s:s + T_IN].copy()
        y = f[s + T_IN:s + T_IN + T_OUT].copy()
        # Velocity scales as U, pressure as rho*U^2, so the two channels need
        # different powers of the window scale to land in comparable units.
        x[:, :2] /= scale
        y[:, :2] /= scale
        if self.with_p:
            x[:, 2] /= scale ** 2
            y[:, 2] /= scale ** 2
        # (T, C, H, W) -> (T*C, H, W): time folded into channels for a 2D U-Net.
        return (torch.from_numpy(x.reshape(-1, *f.shape[2:])),
                torch.from_numpy(y.reshape(-1, *f.shape[2:])),
                torch.tensor(float(scale)))


    @property
    def window_regimes(self):
        """(N, 2) of (Re, AoA) per window, in dataset order.

        Conditioning on the true regime is only meaningful if validation supplies it
        too; inferring it from the window at eval time would measure something else.
        """
        import numpy as _np
        return _np.array([[float(self.trajs[i].re), float(self.trajs[i].aoa)]
                          for i, _ in self.index], dtype=_np.float32)


def build(split: str, which: str, **kw) -> WindowDataset:
    return WindowDataset(partition(split)[which], split, **kw)
