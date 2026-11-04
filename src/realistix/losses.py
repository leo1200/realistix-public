"""Training losses, including differentiable surrogates for the scored metrics.

Plain MSE is a poor proxy for this competition's scoring. It is dominated by the
mean flow, which every model gets nearly right, while the subscore we are stuck
on -- TKE -- depends entirely on the fluctuation about that mean. Three
independently trained deterministic models all converged to a TKE error of
0.64-0.67 under MSE, which is what regression to the conditional mean looks like.

Three of the five subscores are cheap and differentiable, so we can optimize them
directly rather than hoping MSE correlates:

* ``rel_l2`` is the scored relative-L2 metric verbatim, including its per-window
  aggregation -- it carries the largest weight in the identified combination and
  is what the objective was missing until 2026-08-12;
* ``tke_rel_l2`` is the scored TKE metric verbatim;
* ``mvpe_rel_l2`` is the scored MVPE metric verbatim, on the same probe rake.

``spectral`` and ``time_mean`` are not scored. ``spectral`` targets the same
failure as TKE from another side -- it penalizes a prediction whose energy is
missing at high wavenumber, which is exactly what a smoothed forecast looks like.
``time_mean`` is MVPE applied everywhere instead of at 36 probes.

Everything here operates on (B, T*C, H, W) in normalized units, the layout the
U-Net trains in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def _unfold(x, t: int = 20, c: int = 2):
    """(B, T*C, H, W) -> (B, T, C, H, W)."""
    b, _, h, w = x.shape
    return x.reshape(b, t, c, h, w)


def cooperative_mean_gradient(ensemble, member, ensemble_size, mean_scale=1.0):
    """Keep ensemble values; scale only its temporal-mean derivative to member.

    The ordinary average contributes I/M. Add (scale-1)*P_mean/M using a
    zero-valued straight-through term. Fluctuation derivatives stay I/M.
    This output-space preconditioner makes no parameter-space conflict guarantee.
    """
    if mean_scale == 1.0:
        return ensemble
    difference = member.float() - member.float().detach()
    temporal = _unfold(difference)
    mean = temporal.mean(1, keepdim=True).expand_as(temporal).reshape_as(member)
    return ensemble + ((mean_scale - 1) / ensemble_size) * mean


def cooperative_tke_loss(ensemble, member, target, own_fraction, ensemble_scale=1.0):
    """Route TKE between own/ensemble predictions; other losses are separate."""
    return ((1 - own_fraction) * ensemble_scale * tke_rel_l2(ensemble, target)
            + own_fraction * tke_rel_l2(member, target))


def rel_l2(pred, target):
    """Per-window relative L2, averaged over the batch -- the scored aggregation.

    The scorer averages ratios, not norms, so a low-energy window counts as much
    as a high-energy one. Training on plain MSE instead weights windows by their
    energy, which over-serves the high-Reynolds end of the range.

    No ``t``/``c``: this one flattens the whole window, exactly as
    ``scoring.rel_l2_per_sample`` does, so the frame layout never enters. It used
    to take them and ignore them, which reads as if it were shape-aware.
    """
    b = pred.shape[0]
    p, y = pred.reshape(b, -1), target.reshape(b, -1)
    return (torch.linalg.vector_norm(p - y, dim=1)
            / torch.linalg.vector_norm(y, dim=1).clamp_min(EPS)).mean()


def tke(x, t: int = 20, c: int = 2):
    """0.5 * mean_t of squared fluctuation, per the scorer's definition."""
    f = _unfold(x, t, c)
    fluct = f - f.mean(dim=1, keepdim=True)
    return 0.5 * (fluct ** 2).mean(dim=1).sum(dim=1)      # (B, H, W)


def tke_rel_l2(pred, target, t: int = 20, c: int = 2):
    p, y = tke(pred, t, c), tke(target, t, c)
    b = p.shape[0]
    return (torch.linalg.vector_norm((p - y).reshape(b, -1), dim=1)
            / torch.linalg.vector_norm(y.reshape(b, -1), dim=1).clamp_min(EPS)).mean()


def spectral(pred, target, t: int = 20, c: int = 2, log: bool = True):
    """Match the 2D spatial power spectrum, frame by frame.

    A prediction that has regressed toward the mean loses energy at high
    wavenumber while keeping the low-wavenumber content intact. In log space
    every decade of the spectrum counts equally, so the high-wavenumber tail --
    which carries almost no L2 energy and is therefore invisible to MSE -- is
    weighted enough to matter.
    """
    p, y = _unfold(pred, t, c), _unfold(target, t, c)
    pp = torch.fft.rfft2(p, norm="ortho").abs() ** 2
    yy = torch.fft.rfft2(y, norm="ortho").abs() ** 2
    if log:
        return F.l1_loss(torch.log(pp + 1e-6), torch.log(yy + 1e-6))
    return F.l1_loss(pp, yy)


# Probe rake from the scorer's mvpe_rel_l2_per_sample, with sub_s_real=2:
# x = int(((i+1)*16 + 10)/2) for i in 0..3, y = 16 + 2j for j in -4..4.
_PROBE_X = (13, 21, 29, 37)
_PROBE_Y = tuple(range(8, 26, 2))


def mvpe_rel_l2(pred, target, t: int = 20, c: int = 2):
    p, y = _unfold(pred, t, c), _unfold(target, t, c)
    errs = []
    for px in _PROBE_X:
        pp = p[:, :, :, _PROBE_Y, px].mean(dim=1).reshape(p.shape[0], -1)
        yy = y[:, :, :, _PROBE_Y, px].mean(dim=1).reshape(y.shape[0], -1)
        errs.append(torch.linalg.vector_norm(pp - yy, dim=1)
                    / torch.linalg.vector_norm(yy, dim=1).clamp_min(EPS))
    return torch.stack(errs).mean()


def lowfreq_rel(pred, target, t: int = 20, c: int = 2,
                w_lo: float = 0.5, w_hi: float = 2.5, k_hi: float = 8.0):
    """Relative L2 restricted to the slow, large-scale space-time band.

    Measured 2026-08-22 on held-out cells: the temporal-frequency band
    omega in [1, 2] cycles/window -- the shedding/meandering timescale --
    carries 56% of the squared prediction error (35% at k in [2, 8] alone) at
    err/tgt of 0.38-0.55, i.e. half-predicted. The flat mse/rel terms weight
    those modes like any others and `tmean` covers only omega = 0, so this is
    the one part of (k, omega) space that holds most of the error and has
    never carried extra gradient. FFTs run in float32: half-precision FFT is
    where autocast goes to die.
    """
    p = _unfold(pred, t, c).float()
    y = _unfold(target, t, c).float()
    P = torch.fft.fftn(p, dim=(1, 3, 4))
    Y = torch.fft.fftn(y, dim=(1, 3, 4))
    dev = pred.device
    om = (torch.fft.fftfreq(t, device=dev).abs() * t)[None, :, None, None, None]
    ky = (torch.fft.fftfreq(p.shape[-2], device=dev).abs()
          * p.shape[-2])[None, None, None, :, None]
    kx = (torch.fft.fftfreq(p.shape[-1], device=dev).abs()
          * p.shape[-1])[None, None, None, None, :]
    k = torch.sqrt(kx ** 2 + ky ** 2)
    m = ((om >= w_lo) & (om <= w_hi) & (k < k_hi)).float()
    d2 = (((P - Y).abs() ** 2) * m).sum(dim=(1, 2, 3, 4))
    y2 = ((Y.abs() ** 2) * m).sum(dim=(1, 2, 3, 4))
    return (d2.sqrt() / y2.sqrt().clamp_min(EPS)).mean()


def lead_energy(pred, target, t: int = 20, c: int = 2):
    """Relative L2 between the per-lead fluctuation-energy profiles (2026-09-07).

    **DEAD -- kept for the measurement, do not re-run.** Screened 2026-09-07 on the four
    mimic folds x seeds 0/1: **-0.098 at dose 1.0, -0.038 at 0.3, 0/4 folds both times.**

    What the term was built on, and which half of it was wrong. Measured on the shipped
    members (research/optimizer_2026-09-07/claude/lead_profile.py, fold mim1, two seeds):
    the prediction's fluctuation energy about its own window mean is 1.37x the target's at
    lead 0 and 0.45x at lead 19. That decline is real, and the window-TKE term genuinely
    cannot see it -- it constrains the variance over the 20 frames, which a front-loaded
    profile satisfies as well as an even one. **But the inference from stationarity was
    wrong.** The target's own per-lead profile is not flat: removing each window's finite
    20-frame mean leaves a U (1.28 at both ends, 0.90 in the middle), and a stationary flow
    does not imply a flat energy profile for its *conditional point forecast* -- optimal
    forecasts decay toward the mean as coherence falls (0.84 at lead 0, 0.41 at lead 19).
    Codex's target-assisted oracle settles it: imposing the target's own relative lead
    profile at fixed per-cell mean and TKE costs **-0.030 on 4/4 folds**, so the allocation
    this term pushes toward is worse than the one the model finds. Restoring the *total*
    energy instead just reopens the closed amplitude/TKE frontier.

    Per window, E_k = mean over (c, h, w) of the squared fluctuation at lead k; the loss is
    ||E_pred - E_tgt|| / ||E_tgt|| over the 20 leads. It is neutral to the scorer's TKE
    (which sees only the sum) and to MVPE (the mean is removed).
    """
    p = _unfold(pred, t, c).float()
    y = _unfold(target, t, c).float()
    ep = ((p - p.mean(dim=1, keepdim=True)) ** 2).mean(dim=(2, 3, 4))   # (B, T)
    ey = ((y - y.mean(dim=1, keepdim=True)) ** 2).mean(dim=(2, 3, 4))
    return (torch.linalg.vector_norm(ep - ey, dim=1)
            / torch.linalg.vector_norm(ey, dim=1).clamp_min(EPS)).mean()


def diff_energy(pred, target, t: int = 20, c: int = 2):
    """Relative L2 between the per-lead FRAME-DIFFERENCE energy profiles (2026-09-07).

    **DEAD -- kept for the measurement, do not re-run.** Screened 2026-09-07 on the four
    mimic folds x seeds 0/1: **-0.055 at dose 1.0, -0.027 at 0.3, 0/4 folds both times.**

    The defect it targets is real. Measured on the shipped members (lead_anomaly.py): the
    target's |y_k - y_{k-1}|^2 is flat across the horizon (1.00 +- 0.03 of its mean) while
    the prediction's runs 1.4 -> 0.25 over leads 1-17 and then spikes to 3.8x at lead 18 --
    a discontinuity at one of the last leads, at a fold-dependent lead, concentrated on the
    extrapolation cells, where that lead's error roughly doubles (26700_0 on mim1: 0.137
    against a window mean of 0.060). It reproduces in raw non-EMA weights and survives
    window-start, mask and precision changes (Codex round 6), and the shipped `out.weight`
    rows for the last leads carry the largest norms. wd 0.03 halves it; polar-first doubles
    it.

    **Why fixing it is not worth a term.** The single-member price (~1.5% of rel_l2 on the
    affected fold) does not survive the ensemble: the three members put their spikes at
    *different* leads, so the average is already smooth, and a neighbour-average oracle over
    the three-member ensembles reads -0.023 / -0.000 / -0.030 / -0.002 on the four mimic
    folds -- removing the spike costs accuracy, because the spike carries fluctuation energy
    that TKE is paid for. A per-member defect that the shipped object averages away is not a
    scoring defect.

    Per window, D_k = mean over (c, h, w) of (p_k - p_{k-1})^2 for k = 1..19; the loss is
    ||D_pred - D_tgt|| / ||D_tgt||. It is a temporal-derivative statistic, so it is blind to
    the time-mean (MVPE) and nearly blind to the window variance (TKE).
    """
    p = _unfold(pred, t, c).float()
    y = _unfold(target, t, c).float()
    dp = ((p[:, 1:] - p[:, :-1]) ** 2).mean(dim=(2, 3, 4))   # (B, T-1)
    dy = ((y[:, 1:] - y[:, :-1]) ** 2).mean(dim=(2, 3, 4))
    return (torch.linalg.vector_norm(dp - dy, dim=1)
            / torch.linalg.vector_norm(dy, dim=1).clamp_min(EPS)).mean()


def time_mean(pred, target, t: int = 20, c: int = 2):
    """Relative L2 of the 20-frame time-mean field.

    MVPE measures exactly this at 36 probe points; applied everywhere it is the
    un-gameable version -- a model cannot satisfy it by learning probe positions.
    """
    p, y = _unfold(pred, t, c).mean(1), _unfold(target, t, c).mean(1)
    b = p.shape[0]
    return (torch.linalg.vector_norm((p - y).reshape(b, -1), dim=1)
            / torch.linalg.vector_norm(y.reshape(b, -1), dim=1).clamp_min(EPS)).mean()


#: The scorer's global velocity scale, frozen for the season. `starting_kit/scoring.py`
#: computes every interval width against this and not against a per-window quantity.
SIGMA_GLOBAL = 0.0563870259


def sps_bulk(pred, target, wscale, sigma: float = SIGMA_GLOBAL, width: float = 1.0,
             sample_weights=None, keep_denominator: bool = False):
    """1 - mean exp(-|e| * width / sigma) over scored elements: the SPS reward at
    oracle width, and the only term here that is not an L2 norm.

    Why this exists. Re-deriving the loss from the scoring function was this
    project's largest single win (887158, +0.48), and it was done for four of the
    five subscores. `sps_score` was left out because it looked like the interval
    head's business. `tools/sps_decompose.py` shows it is not: SPS factors as
    `100 * W_bar * R_bar`, and while `R_bar` is the head's, the *ceiling* on `R_bar`
    is set entirely by the forecaster --

        R_oracle = mean exp(-|e| / sigma)

    is what a head that knew the realised error would collect, and no head can beat
    it. So the forecaster has an SPS objective after all, and it is this.

    **It is the opposite shape to every other term.** SPS's per-element reward decays
    *exponentially* in the error and is bounded below by zero, so an element already
    wrong by more than about half a sigma contributes nothing and cannot be made to
    contribute; L2 weights that same element quadratically and spends most of its
    gradient there. The two objectives disagree maximally on the tail, and the
    measured consequence is on the record: 898763's Muon members improved the *bulk*
    of the error distribution (best-constant-band `R` up 0.0046 on 4/4 folds) while
    `rel_l2` went slightly **negative** on the board. Nothing in this loss has ever
    pointed at the bulk.

    ``width`` is the total interval width in units of |e|: 1.0 for the asymmetric
    bounds the shipped head emits (an interval running from prediction to target),
    2.0 for a symmetric one. Elements with an exactly-zero target are outside the PIV
    field of view or inside the airfoil and are not scored, here or by the scorer.
    """
    m = target != 0
    e = (pred - target).abs() * wscale.reshape(-1, *([1] * (pred.dim() - 1)))
    r = torch.exp(-e * (width / sigma))
    if sample_weights is not None:
        w = sample_weights.reshape(-1, *([1] * (pred.dim() - 1)))
        if keep_denominator:
            # 2026-09-10 real-only screen: drop a modality's contribution from the
            # numerator while the scored-element denominator stays the whole batch,
            # so the surviving rows keep exactly the per-element weight they had.
            return 1.0 - (r * m * w).sum() / m.sum().clamp_min(1)
        m = m * w
    return 1.0 - (r * m).sum() / m.sum().clamp_min(1)


def sps_prefactors(pred, target, wscale=None):
    """Official per-window SPS factors, including physical-unit epsilon floors.

    Pass wscale for normalized windows. Ratios are scale invariant only while
    their denominator floors are inactive; nearly steady targets can activate
    the TKE floor, so normalization must be undone before applying it.
    """
    if wscale is not None:
        scale = wscale.reshape(-1, *([1] * (pred.ndim - 1)))
        pred, target = pred * scale, target * scale
    b = pred.shape[0]
    def relative(p, y):
        return (torch.linalg.vector_norm((p - y).reshape(b, -1), dim=1)
                / torch.linalg.vector_norm(y.reshape(b, -1), dim=1).clamp_min(EPS))
    dm = relative(pred, target)
    ke = relative(tke(pred), tke(target))
    p, y = _unfold(pred).mean(1), _unfold(target).mean(1)
    mv = torch.stack([relative(p[:, :, _PROBE_Y, px], y[:, :, _PROBE_Y, px])
                      for px in _PROBE_X]).mean(0)
    return 0.5 / (1 + 2 * dm) + 0.3 / (1 + 2 * ke) + 0.2 / (1 + 2 * mv)


def sps_oracle(pred, target, wscale, sigma: float = SIGMA_GLOBAL, width: float = 1.0,
               sample_weights=None):
    """Negative differentiable reward for target-assisted minimum-width intervals.

    Includes the scorer's three per-window accuracy prefactors, unlike sps_bulk.
    At width=1, 1-loss equals aggregate_sps for bounds min(pred,target), max(...).
    This is an upper bound using the training target, NOT an achievable head score.
    Both the bulk reward and the prefactors receive gradients. No hidden data or
    future statistic is needed at inference; this is a training objective only.
    """
    pref = sps_prefactors(pred, target, wscale)
    shape = (-1,) + (1,) * (pred.ndim - 1)
    reward = torch.exp(-(pred - target).abs() * wscale.reshape(shape) * (width / sigma))
    mask = target != 0
    if sample_weights is not None:
        mask = mask * sample_weights.reshape(shape)
    return 1 - (reward * pref.reshape(shape) * mask).sum() / mask.sum().clamp_min(1)


def sps_bulk_mean(pred, target, wscale, sigma: float = SIGMA_GLOBAL,
                  width: float = 1.0):
    """`sps_bulk` applied to the per-pixel TIME-MEAN error only.

    Why this variant exists. The full-field `sps_bulk` dose ladder (2026-08-25) is
    a clean frontier traversal: rel_l2 and the bulk improve, TKE pays, monotone in
    dose. But the scorer's `kinetic_energy` is the temporal variance about each
    signal's OWN time-mean, per pixel -- so the predicted time-mean is invisible to
    TKE **by construction**. A bulk term that reads only the time-mean error has no
    gradient path to the predicted fluctuation at all: `d loss / d pred[t]` is the
    same for every t at a given pixel, and adding a constant to a time series does
    not change its variance. Whatever this term does to the forecast, it cannot pay
    for it in TKE directly; any residual TKE move is indirect, through shared trunk
    weights, and the fold screen prices that.

    Whether the *gain* survives the restriction is the empirical question --
    `tools/error_anatomy.py` measures which component (mean or fluctuation) the
    full term's bulk improvement actually lives in. The time-mean error carries 55%
    of squared rel-L2 error (the meandering finding, FINDINGS section 6), and the
    existing `w_tmean` term covers it only in the tail-weighted relative-L2 shape;
    this is the same address under the bulk-weighted exp shape the scorer pays.

    Masking: body/FOV cells are exactly zero at every frame, so their time-mean is
    exactly zero; measured pixels' means are not. `ym != 0` therefore reproduces
    the scorer's element mask at the pixel level.
    """
    p = _unfold(pred).mean(1)
    y = _unfold(target).mean(1)                       # (B, C, H, W)
    m = y != 0
    e = (p - y).abs() * wscale.reshape(-1, 1, 1, 1)
    r = torch.exp(-e * (width / sigma))
    return 1.0 - (r * m).sum() / m.sum().clamp_min(1)


class CompositeLoss(torch.nn.Module):
    """Weighted sum of the terms above. Weight 0 skips a term entirely."""

    def __init__(self, mse=1.0, rel=0.0, tke=0.0, spec=0.0, mvpe=0.0, tmean=0.0,
                 t=20, c=2):
        super().__init__()
        self.w = {"mse": mse, "rel": rel, "tke": tke, "spec": spec,
                  "mvpe": mvpe, "tmean": tmean}
        self.t, self.c = t, c

    def forward(self, pred, target):
        parts = {}
        if self.w["mse"]:
            parts["mse"] = F.mse_loss(pred, target)
        if self.w["rel"]:
            parts["rel"] = rel_l2(pred, target)
        if self.w["tke"]:
            parts["tke"] = tke_rel_l2(pred, target, self.t, self.c)
        if self.w["spec"]:
            parts["spec"] = spectral(pred, target, self.t, self.c)
        if self.w["mvpe"]:
            parts["mvpe"] = mvpe_rel_l2(pred, target, self.t, self.c)
        if self.w["tmean"]:
            parts["tmean"] = time_mean(pred, target, self.t, self.c)
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
