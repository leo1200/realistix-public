"""A learned replacement for the plain mean over ensemble members.

The three forecasters are trained independently and then *averaged*, but the
average is what the scorer sees -- and averaging is not a neutral operation on the
quantity TKE measures. Each member is trained to minimise its own loss, so each
approximates the conditional mean of a stochastic target; their average is that
same conditional mean with the member-to-member disagreement removed. Measured on
the shipped ensemble, the members carry **7.6% more TKE than their mean at the
median cell and 164% more at p95** (`docs/archive/HANDOFF_2026-09-10.md` §7), and `tke_score` at 78.03 is
by a wide margin our weakest accuracy subscore -- its error is five times
`rel_l2`'s, and the elasticity table prices a 10% relative improvement in it at
+0.305 of final, 83% of what the same relative gain in `rel_l2` is worth.

Restoring that energy *after* training has been tried and is dose-dependently
negative: `tools/tke_restore.py` rescales the mean's fluctuation to the members'
own per-cell TKE and lands at +0.002/+0.006 at the matched dose, then -0.014 and
-0.042 as the dose rises. `tke` rises monotonically the whole way, so the mechanism
is real; what sinks it is that `rel_l2` and SPS coverage give back more at every
dose. That is a post-hoc correction imposing a *fixed* trade-off with no way to
learn where it is worth making.

This is the same mechanism with the trade-off learned instead of imposed. The
combiner reads all member outputs, their mean and their per-element spread, and
the input window, and emits a correction to the mean under the shipped composite
loss -- the objective that already prices `rel_l2` against `tke` the way the
scorer does. Where restoring fluctuation costs more accuracy than it returns, the
optimum is to leave the mean alone, and the network can represent that exactly:
the output convolution is zero-initialised, so training *starts* at the plain
ensemble mean and every departure has to pay for itself. The same device carries
the forecaster's own delta head, and it is why a run begins at persistence rather
than at noise.

Cost matters here because `time_score`'s weight is 0.099. This is deliberately a
few full-resolution convolutions rather than another U-Net: the members have
already done the multi-scale work, and what is left is a per-cell decision about
how much of their disagreement to put back.
"""

from __future__ import annotations

import torch
import torch.nn as nn

T_IN = T_OUT = 20
C = 2


class Combiner(nn.Module):
    """Members -> refined prediction. Starts exactly at their mean.

    ``spread`` and ``window`` are separately switchable because they answer
    different objections. The spread is where the averaging cancellation *is* --
    a cell where the members agree has no energy to restore -- and it is the input
    the interval head already reads for the analogous job. The window is what
    every other component here conditions on, and it carries the local flow state
    that says whether a disagreement is wake or quiescence.
    """

    def __init__(self, n_members: int = 3, width: int = 64, depth: int = 3,
                 spread: bool = True, window: bool = True, groups: int = 8):
        super().__init__()
        self.n_members, self.spread, self.window = n_members, spread, window
        cin = n_members * T_OUT * C + T_OUT * C
        if spread:
            cin += T_OUT * C
        if window:
            cin += T_IN * C
        layers: list[nn.Module] = [nn.Conv2d(cin, width, 3, padding=1)]
        for _ in range(max(0, depth - 1)):
            layers += [nn.GroupNorm(min(groups, width), width), nn.SiLU(),
                       nn.Conv2d(width, width, 3, padding=1)]
        layers += [nn.GroupNorm(min(groups, width), width), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, T_OUT * C, 3, padding=1)
        # Zero-initialised: the combiner is the identity on the ensemble mean at
        # step 0, so a run that learns nothing ships the current model rather than
        # a worse one, and every measured delta is attributable to the combiner.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, members: torch.Tensor, window: torch.Tensor | None = None):
        """``members`` is (M, B, T_OUT*C, H, W); returns (B, T_OUT*C, H, W)."""
        mean = members.mean(0)
        parts = [members.permute(1, 0, 2, 3, 4).reshape(
            members.shape[1], -1, *members.shape[-2:]), mean]
        if self.spread:
            # Unbiased, matching how the interval head reads the same quantity, so
            # a three-member spread is not systematically under-reported.
            parts.append(members.std(0, unbiased=True))
        if self.window:
            if window is None:
                raise ValueError("combiner was built with window=True")
            parts.append(window)
        return mean + self.out(self.body(torch.cat(parts, dim=1)))
