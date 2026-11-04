"""Per-element interval widths, trained directly against the SPS reward.

Why not simply predict the error and set the width from it: SPS pays
``exp(-(upper-lower)/sigma)`` on covered elements and exactly zero on uncovered
ones. Doubling a width costs ``1 - exp(-width/sigma)`` of its reward; at the widths
this head actually settles on -- 0.22 sigma inside the PIV shadow and 0.35 sigma
outside it, which reproduce the measured rewards 0.77 and 0.62 against coverages
0.94 and 0.84 -- that is **20-30%**, while missing coverage costs all of it. (An
earlier version of this note said 16%, computed at an assumed width of 0.010 in
absolute units, i.e. 0.18 sigma, rather than at the measured ones.) The objective is
therefore steeply asymmetric, and a width that is merely proportional to a
predicted error is mis-specified: measured on held-out data, per-cell mean error
(32.2), per-cell quantiles (28.0-35.6) and input variability (11.6-26.6) all score
*below* a well-tuned constant band (36.0), even though the input-variability proxy
correlates 0.54 with the true error.

So the head outputs the width and is trained on the reward itself, with the
coverage indicator relaxed to a sigmoid:

    reward_i = exp(-(below_i + above_i) / sigma) * covered_i

which is the *per-element interval term* of the scored quantity as tau -> 0, and
differentiable for tau > 0.

**This is a surrogate, not the metric.** Two things are dropped, and the docstring
claimed otherwise until 2026-09-01:

  * The ``(1 - pm)`` prefactor is constant with respect to the width of a *single
    element* considered alone, which is why it looked droppable. It is not constant
    across samples, and one network shares parameters across all of them, so
    omitting it silently reweights the training distribution. Measured on the
    187-window val_aoa artefact, the combined prefactor spans 0.6921 to 0.8058.
  * The official score is a 0.5/0.3/0.2 combination of three branches (DM, TKE,
    MVPE), each with its own error and therefore its own prefactor; this reward
    averages raw element coverage and width once.

The default retains this historical objective. ``train_final_head.py
--score-prefactors`` (2026-09-09) includes the official factors through
``sps_reward(sample_weights=...)``. It still relaxes coverage to a sigmoid; see
``research/audit_2026-09-09`` for its controlled evaluation.

Widths are absolute, because sigma is: the network sees scale-normalized fields
plus log(scale) as a channel, and emits log-width in physical units.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Block

SIGMA_GLOBAL = 0.0563870259


class UncertaintyHead(nn.Module):
    """(input window, prediction, log scale) -> per-element interval half-width."""

    def __init__(self, t: int = 20, c: int = 2, width: int = 48, mults=(1, 2, 4),
                 extra_channels: int = 0, offset: bool = False):
        super().__init__()
        self.t, self.c = t, c
        # ``extra_channels`` carries ensemble statistics when a residual flow is
        # available. The head is information-limited without them: it cannot infer
        # instance-specific error from the input window, which is why every
        # configuration converged to the same ~39 against a 60 ceiling.
        self.extra_channels = extra_channels
        cin = t * c * 2 + 1 + extra_channels      # window, prediction, log-scale, stats
        chans = [width * m for m in mults]
        self.stem = nn.Conv2d(cin, chans[0], 3, padding=1)
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        prev = chans[0]
        for i, ch in enumerate(chans):
            self.downs.append(Block(prev, ch))
            prev = ch
            self.pools.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                              if i < len(chans) - 1 else nn.Identity())
        self.mid = Block(prev, prev)
        self.ups, self.upsamples = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1)
                                  if i < len(chans) - 1 else nn.Identity())
            prev = ch if i < len(chans) - 1 else prev
            self.ups.append(Block(prev + ch, ch))
            prev = ch
        self.out_norm = nn.GroupNorm(8, prev)
        # Two outputs per element, not one: the interval need not be centred on
        # the prediction, and SPS charges for total width, so an interval reaching
        # from prediction to target costs half what a symmetric one does. Knowing
        # the sign of the error is worth +7.3 SPS at the ceiling (60.3 vs 53.0),
        # and 34% of the error here is systematic per-cell bias, so part of that
        # sign is genuinely predictable.
        self.out = nn.Conv2d(prev, 2 * t * c, 3, padding=1)
        # Start symmetric at the tuned constant band: total width 0.18*sigma.
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(0.09 * SIGMA_GLOBAL))
        # A signed centre offset, in addition to the two half-widths. The scorer
        # requires only lower <= upper -- it never requires the interval to contain
        # the prediction (scoring.py:203 never reads `pred`) -- and 34% of the error
        # is systematic per-cell bias, so part of the error's *sign* is predictable.
        # The shipped parameterisation cannot use that: below and above are both
        # positive by construction, and it visibly binds -- 9.4% of elements sit at
        # |hi-lo|/(hi+lo) > 0.5, the hardest lean the form allows. The offset shifts
        # the centre while `below + above` -- the width SPS charges for -- is
        # unchanged: coverage bought where the bias is, for free. Zero-initialised,
        # so the head starts exactly at the shipped behaviour.
        self.offset = offset
        if offset:
            self.out_off = nn.Conv2d(prev, t * c, 3, padding=1)
            nn.init.zeros_(self.out_off.weight); nn.init.zeros_(self.out_off.bias)

    def features(self, window, pred, log_scale, extra=None):
        """Return the feature map immediately before the interval output layer.

        Keeping this as an explicit interface makes it possible to test alternate
        *training signals* on a frozen shipped trunk without hooks or a duplicate
        implementation of the U-Net.  ``forward`` below still performs exactly the
        same operations in exactly the same order.
        """
        b, _, h, w = window.shape
        plane = log_scale.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        parts = [window, pred, plane]
        if self.extra_channels:
            parts.append(extra)
        x = torch.cat(parts, dim=1)
        z = self.stem(x)
        skips = []
        for blk, pool in zip(self.downs, self.pools):
            z = blk(z)
            skips.append(z)
            z = pool(z)
        z = self.mid(z)
        for blk, up, skip in zip(self.ups, self.upsamples, reversed(skips)):
            z = up(z)
            z = blk(torch.cat([z, skip], dim=1))
        return F.silu(self.out_norm(z))

    def widths_from_features(self, features):
        """Apply the shipped output parameterisation to a cached feature map."""
        log_d = self.out(features)
        # Clamped for numerical sanity: e^-9 ~ 1e-4 is far below any useful width,
        # e^-2 ~ 0.135 is ~2.4 sigma and already scores near zero.
        d = torch.exp(log_d.clamp(-9.0, -2.0))
        n = self.t * self.c
        below, above = d[:, :n], d[:, n:]              # distances from the prediction
        if self.offset:
            # [pred - below + o, pred + above + o]: same width, shifted centre. A
            # negative "below" is a lower bound above the prediction, which the
            # scorer accepts; sps_reward's coverage term handles the sign as-is.
            o = self.out_off(features)
            o = o.clamp(-0.5 * SIGMA_GLOBAL, 0.5 * SIGMA_GLOBAL)
            below, above = below - o, above + o
        return below, above

    def forward_with_features(self, window, pred, log_scale, extra=None):
        """Return ``(below, above, features)`` without running the trunk twice."""
        features = self.features(window, pred, log_scale, extra)
        below, above = self.widths_from_features(features)
        return below, above, features

    def forward(self, window, pred, log_scale, extra=None):
        below, above, _ = self.forward_with_features(window, pred, log_scale, extra)
        return below, above


def sps_reward(below, above, signed_err, scored, sigma: float = SIGMA_GLOBAL,
               tau_frac: float = 0.15, sample_weights=None):
    """Differentiable surrogate for the per-element SPS reward.

    The interval is ``[pred - below, pred + above]``; it is covered when
    ``-below <= (target - pred) <= above``. ``tau`` is set relative to the width
    rather than fixed, so the relaxation stays equally sharp across the two orders
    of magnitude the widths span between quiescent and wake cells.
    """
    width = below + above
    tau = (tau_frac * width).clamp_min(1e-6)
    covered = (torch.sigmoid((signed_err + below) / tau)
               * torch.sigmoid((above - signed_err) / tau))
    reward = torch.exp(-width / sigma) * covered
    if sample_weights is not None:
        reward = reward * sample_weights.reshape(-1, *([1] * (reward.ndim - 1)))
    return (reward * scored).sum() / scored.sum().clamp_min(1.0)
