"""Per-window normalization.

The single largest structural gap in this dataset is that simulated velocities
are non-dimensionalized while real ones are in m/s: sim ``mean(u)`` sits at ~0.87
for *every* Reynolds number, while real ``mean(u)`` tracks the free-stream
velocity and so scales linearly with Re, from 0.045 at Re 3750 to 0.27 at Re
26700. A model fed both sees inputs an order of magnitude apart for what is
physically the same flow.

The fix is to normalize each window by a scale computed from that window alone.
Three things fall out of it:

* sim and real land in the same space, so sim pretraining actually transfers;
* the model becomes approximately Re-invariant, which is what the private test
  set's "unseen Reynolds numbers" will probe;
* it costs nothing metrically, because rel-L2, TKE and MVPE are all per-window
  scale-invariant ratios. Only the SPS bounds have to be mapped back to physical
  units, since ``nil`` divides by an absolute ``sigma_global``.

Re and AoA are *not* available at inference (``metadata`` is an empty dict on
scored calls), so the scale has to come from the input window itself. That is
legal -- it is the window we were handed, not another one -- and it is easy here
precisely because velocity magnitude tracks Re so cleanly.

The mask matters: cells inside the airfoil and outside the PIV field of view are
stored as exact zeros, and including them would drag the scale down by however
much of the frame they happen to occupy (6-11% depending on angle of attack).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def valid_mask(window):
    """Cells measured throughout the window: not exactly zero in u and v at every frame.

    Shape (..., T, H, W, C) -> (..., H, W). This is the same rule the scorer uses
    to decide what to score (``target != 0``), applied to the input instead.
    """
    uv = window[..., :2]
    zero = np.all(np.abs(uv) < EPS, axis=-1)      # (..., T, H, W)
    return ~np.all(zero, axis=-3)                  # (..., H, W)


def window_scale(window):
    """One positive velocity scale per window, from the window alone.

    RMS speed over valid cells. RMS rather than the mean of u, because at high
    angle of attack the wake carries enough reversed flow that mean(u) understates
    the free-stream, and because it stays positive by construction.
    """
    mask = valid_mask(window)[..., None, :, :]     # (..., 1, H, W)
    uv = window[..., :2]
    sq = np.sum(uv ** 2, axis=-1)                  # (..., T, H, W)
    num = np.sum(np.where(mask, sq, 0.0), axis=(-3, -2, -1))
    den = np.sum(np.broadcast_to(mask, sq.shape), axis=(-3, -2, -1)) + EPS
    return np.sqrt(num / den) + EPS                # (...,)


def normalize(window, scale=None):
    """Scale a window to unit RMS speed. Returns (normalized, scale)."""
    if scale is None:
        scale = window_scale(window)
    s = np.asarray(scale)[..., None, None, None, None]
    out = window / s
    out[..., 2] = 0.0                              # p is unmeasured; keep it exactly zero
    return out, scale

