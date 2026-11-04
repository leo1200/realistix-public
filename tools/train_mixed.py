#!/usr/bin/env python3
"""Does simulated data help where it covers regimes the real split does not?

Two different questions get conflated when people say "does sim help":

  A. Does sim help *in general*, as extra data?  The learning curve already
     answers this pessimistically -- returns are strongly diminishing (+0.15 for
     the last quarter of real trajectories), and sim is biased 4-11x in
     fluctuation energy, the statistic driving our worst subscore.

  B. Does sim help *at regimes real does not cover*?  This is a different claim,
     and it is the one the Decision Phase actually tests: the private set uses
     unseen angles of attack and Reynolds numbers, and the sim split covers two
     Reynolds numbers (15225, 27975) plus 18 (Re, AoA) pairs that real does not.

The learning curve cannot separate them, because subsampling trajectories drops
whole regimes as well as samples. This script can:

  real_only     real train split                          (the arm D baseline)
  plus_sim      + sim at the SAME regimes as real train   (isolates A)
  plus_sim_all  + sim including the held-out regimes      (isolates B, vs plus_sim)

``plus_sim_all`` sees simulated data at Re 13950 and 21600 and at AoA 20 -- the
regimes `val_re` and `val_aoa` are built from. The real trajectories there stay
held out, so this is not leakage; it is exactly the question "if simulation covers
a regime, does that transfer to real measurements in that regime".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realistix.data import HELDOUT_TAIL, T_IN, WindowDataset, build  # noqa: E402
from realistix.gpudata import MixedGPUWindows  # noqa: E402
from realistix.evaluate import amp_dtype_for, fmt, predict_all, subscores  # noqa: E402
from realistix import losses as rlosses  # noqa: E402
from realistix.losses import (CompositeLoss, diff_energy, lead_energy, lowfreq_rel, sps_bulk, sps_oracle,  # noqa: E402
                              sps_bulk_mean)
from realistix.models import build_model, count_params  # noqa: E402
from realistix.splits import partition  # noqa: E402
from realistix.normalize import EPS  # noqa: E402
from train_sim2real import RegimePairs, match_loss, stats  # noqa: E402

MODES = ("real_only", "plus_sim", "plus_sim_all", "sim_only")


class CooperativeForecast(torch.nn.Module):
    """Evaluation view of a trainable member and its fixed companions.

    Checkpoints store the member separately; their recorded companion paths define
    the ensemble whose metrics appear in the history. Never load these histories
    as single-member measurements.
    """
    def __init__(self, member, companions):
        super().__init__()
        self.member = member
        self.companions = torch.nn.ModuleList(companions)

    def forward(self, x, modality=None, log_scale=None, regime=None):
        out = [self.member(x, modality, log_scale, regime).float()]
        out.extend(m(x, modality, log_scale, regime).float() for m in self.companions)
        return torch.stack(out).mean(0)


class Tagged(torch.utils.data.Dataset):
    """Wrap a dataset so each item carries its modality id (0 = real, 1 = sim)."""

    def __init__(self, ds, tag):
        self.ds, self.tag = ds, tag

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, s = self.ds[i]
        return x, y, s, self.tag


def sample_mimo(gpu_ds, batch: int, m: int, rho: float = 0.0):
    """M same-modality windows per row, concatenated in channels, group-major.

    Each group must share its row's modality token (one FiLM lookup per row), so
    the real and simulated parts are drawn separately from their own samplers, M
    draws each, and concatenated -- the same real/sim split MixedGPUWindows.sample
    makes for one draw. Scales are per GROUP (each window normalised by its own
    RMS), returned as (B, M) so the loss can fold groups into the batch axis and
    apply every existing per-sample term unchanged.

    ``rho`` is MIMO's input-repetition knob: with that probability a row feeds one
    window to all M groups, which the literature uses when the trunk is
    capacity-tight.
    """
    n_sim = 0
    if gpu_ds.sim is not None and gpu_ds.real is not None:
        n_sim = int(round(batch * len(gpu_ds.sim)
                          / (len(gpu_ds.real) + len(gpu_ds.sim))))
    draws = []
    for ds, n in ((gpu_ds.real, batch - n_sim), (gpu_ds.sim, n_sim)):
        if ds is None or n == 0:
            continue
        gx, gy, gs = [], [], []
        for _ in range(m):
            x, y, sc, tag, reg = ds.sample(n)
            gx.append(x); gy.append(y); gs.append(sc)
        x = torch.cat(gx, dim=1)                      # (n, M*T*C, H, W)
        y = torch.cat(gy, dim=1)
        sc = torch.stack(gs, dim=1)                   # (n, M)
        if rho > 0:
            rep = torch.rand(n, device=x.device) < rho
            if bool(rep.any()):
                tc = gx[0].shape[1]
                for g in range(1, m):
                    x[rep, g * tc:(g + 1) * tc] = x[rep, :tc]
                    y[rep, g * tc:(g + 1) * tc] = y[rep, :tc]
                    sc[rep, g] = sc[rep, 0]
        draws.append((x, y, sc, tag, reg))
    return tuple(torch.cat([d[i] for d in draws]) for i in range(5))


SIM_SCRAMBLES = ("none", "time", "phase", "mean")


def scramble_sim(x, y, tag, how: str, t: int = 20, c: int = 2):
    """Degrade the simulated half of a batch while keeping stated statistics exact.

    The mechanism test for the +0.71 that joint simulated training pays. The record
    says simulation acts as a *continuous regulariser* rather than as regime
    information (`docs/archive/FINDINGS_2026-09-10.md` section 3): seven routes for using it as information
    are negative, a sequential curriculum recovers 62% of the gain on held-out
    angles and none at all at held-out Reynolds numbers, and correcting the
    simulated levels with a transport map changes accuracy by nothing. Every one of
    those measurements removes or reshapes simulation. None of them asks the
    complementary question, which is the one that identifies the mechanism: how
    much of simulation's *content* does the gain actually need?

      time    the 20 frames of each simulated window are independently permuted,
              in the input and in the target. Dynamics, transport and the shedding
              phase are destroyed; every per-frame field, the window's spatial
              statistics and its power spectrum are untouched. If the gain survives
              this, simulation is not teaching the flow's evolution.
      phase   the Fourier phases of each simulated frame are randomised, per
              channel, with Hermitian symmetry preserved so the field stays real.
              The power spectrum is preserved *exactly* and all spatial structure
              is destroyed. If the gain survives this, simulation is worth no more
              than coloured noise with the right spectrum -- which is free, and
              available in unlimited quantity.
      mean    each simulated window is replaced by its own time-mean, repeated. Sim
              then supplies mean-flow structure and no fluctuation at all, which
              separates the two components the sim/real gap is measured in.

    Real windows are never touched: `tag == 0` rows pass through unchanged, so the
    arm is a strict ablation of the simulated contribution and the real supervision
    is bit-identical to the reference run's.
    """
    if how == "none":
        return x, y
    m = tag == 1
    if not bool(m.any()):
        return x, y
    b, _, h, w = x.shape
    xs = x[m].reshape(-1, t, c, h, w)
    ys = y[m].reshape(-1, t, c, h, w)
    if how == "time":
        # One independent permutation per window and per field, so no residual
        # ordering survives and the input's last frame -- the anchor -- is a
        # uniformly drawn frame of the window rather than its true final state.
        # argsort of uniform noise is a batch of independent permutations, which
        # keeps this off the Python level: it runs every training step.
        for f in (xs, ys):
            perm = torch.rand(f.shape[0], t, device=f.device).argsort(dim=1)
            f.copy_(torch.gather(
                f, 1, perm[:, :, None, None, None].expand_as(f)))
    elif how == "phase":
        for f in (xs, ys):
            F = torch.fft.rfft2(f)
            # The phases of a real Gaussian field's own transform, which satisfy
            # the Hermitian constraints (DC and Nyquist real) by construction. So
            # |F| survives *exactly* rather than up to a symmetry projection: an
            # explicit uniform phase does not, and leaves the spectrum ~1% off.
            G = torch.fft.rfft2(torch.randn_like(f))
            u = G / G.abs().clamp_min(1e-20)
            # k = 0 is kept as it was. Randomising it would flip the sign of the
            # mean flow at random and the arm would then be about flow reversal
            # rather than about spatial structure.
            F2 = F.abs() * u
            F2[..., 0, 0] = F[..., 0, 0]
            f.copy_(torch.fft.irfft2(F2, s=(h, w)))
    elif how == "mean":
        xs.copy_(xs.mean(dim=1, keepdim=True).expand_as(xs))
        ys.copy_(ys.mean(dim=1, keepdim=True).expand_as(ys))
    else:
        raise ValueError(how)
    x = x.clone()
    y = y.clone()
    x[m] = xs.reshape(-1, t * c, h, w)
    y[m] = ys.reshape(-1, t * c, h, w)
    return x, y


def augment_sim(x, y, tag, noise: float, masks, regime, t: int = 20, c: int = 2):
    """Make simulated windows look measured: PIV-like noise, and the optical shadow.

    Both are the organisers' own recommendations -- RealPDEBench (arXiv:2601.01829)
    reports "noise injection" and a "mask-training strategy" as the techniques that
    bridge sim to real on their side -- and neither has ever been tried here. The
    two modality gaps they address are measured: real windows carry frame-to-frame
    measurement noise that simulation lacks (the omega >= 7 band), and real windows
    carry a 6-11% optical shadow of exact zeros that the CFD does not have at all,
    so a network trained on both sees shadowed and unshadowed versions of the same
    physics and the sim windows teach it statistics for cells the scorer will never
    ask about.

    ``noise``  iid Gaussian on the simulated rows, in normalised units, drawn
               independently for input and target the way measurement noise is.
               Real turbulence intensity is ~0.086 in these units, so doses of
               0.01-0.03 are "a fraction of the fluctuation", not of the flow.
    ``masks``  a per-AoA bank of real valid-cell masks; each simulated row gets a
               random real mask *from its own angle of attack* applied to input
               and target both, reproducing how real windows store unmeasured
               cells (exact zeros in x and y). Built from the run's own training
               real trajectories only, so no held-out geometry leaks in.

    One known crack, accepted for a screen: sim rows were normalised before the
    shadow lands, so their scale is computed over the full field where a real
    window's is computed over valid cells -- a few-percent scale inconsistency.
    If the arm pays, move the masking into the sampler before normalisation.

    Real rows pass through untouched.
    """
    m = tag == 1
    if not bool(m.any()) or (noise <= 0 and masks is None):
        return x, y
    x = x.clone(); y = y.clone()
    if noise > 0:
        x[m] = x[m] + noise * torch.randn_like(x[m])
        y[m] = y[m] + noise * torch.randn_like(y[m])
    if masks is not None:
        aoa = regime[m][:, 1].round().long()
        keep = []
        for a in aoa.tolist():
            # plus_sim_all can present a simulated angle whose real trajectories are
            # all held out of this run; the nearest trained angle's shadow is the
            # honest fallback (the shadow geometry moves smoothly with the mount).
            key = (int(a) if int(a) in masks
                   else min(masks, key=lambda k: abs(k - int(a))))
            bank = masks[key]
            keep.append(bank[torch.randint(len(bank), (1,)).item()])
        k = torch.stack(keep).to(x.device)[:, None]      # (n, 1, H, W)
        b, _, h, w = x.shape
        x[m] = (x[m].reshape(-1, t * c, h, w) * k).reshape(-1, t * c, h, w)
        y[m] = (y[m].reshape(-1, t * c, h, w) * k).reshape(-1, t * c, h, w)
    return x, y


# How much of the real release a run is allowed to see. Everything measured so far
# used `train` (57 of the 81 usable trajectories; 58 of 82 before `7575_0` was
# withdrawn), because the rest are what val_re, val_aoa and test_aoa are built from.
#
#   train         58   the split every quoted number was measured on
#   train+val_re  66   gives up Reynolds-interpolation validation, keeps val_aoa
#                      honest -- val_re holds no AoA=20 trajectory, so angle
#                      extrapolation is still a genuine held-out test
#   train+test_aoa 65  the diagnostic for `all`: AoA=20 is the one regime the
#                      58-trajectory split has never seen, and val_aoa is AoA=20 at
#                      a disjoint set of Reynolds numbers, so this measures exactly
#                      what the extra 24 trajectories buy at the angle they cover
#   all           81   nothing held out; only for the shipped model, where the
#                      step count is fixed in advance rather than selected
TRAIN_SETS = ("train", "train+val_re", "train+test_aoa", "all")


def trajs_for(split: str, which: str):
    p = partition(split)
    if which == "train":
        return p["train"]
    if which == "train+val_re":
        return p["train"] + p["val_re"]
    if which == "train+test_aoa":
        return p["train"] + p["test_aoa"]
    return p["train"] + p["val_re"] + p["val_aoa"] + p["test_aoa"]


def build_train(mode: str, sim_stride: int, with_p: bool = False):
    real = Tagged(build("real", "train", stride=1, with_p=with_p), 0)
    if mode == "real_only":
        return real, len(real), 0
    p = partition("sim")
    trajs = (p["train"] if mode == "plus_sim"
             else p["train"] + p["val_re"] + p["val_aoa"] + p["test_aoa"])
    sim = Tagged(WindowDataset(trajs, "sim", stride=sim_stride, with_p=with_p), 1)
    return ConcatDataset([real, sim]), len(real), len(sim)


def build_parser() -> argparse.ArgumentParser:
    """The trainer's own flags, exposed so tools/check_recipe.py can
    parse tools/train_final.sh through the same definitions the trainer
    uses -- a recipe guard that reimplemented them would drift too.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--sim-stride", type=int, default=1)
    ap.add_argument("--sim-batch-frac", type=float, default=None,
                    help="simulation fraction of each GPU batch, rounded to whole rows; "
                         "default uses window counts. Keeps all simulation trajectories")
    ap.add_argument("--sampling", choices=("window", "trajectory"), default="window",
                    help="sample windows uniformly, or sample trajectories then windows")
    ap.add_argument("--sim-loss-frac", type=float, default=None,
                    help="desired simulation weight in the FULL composite+SPS objective, "
                         "independent of sampled batch fraction; plain GPU loss path only")
    # Written into the command rather than left to an inherited environment
    # variable, and echoed below, for the reason kfold.py writes the fold into the
    # command: a run that silently trained on the wrong simulated set would be
    # near-undetectable afterwards.
    ap.add_argument("--sim-dir", default=None,
                    help="alternative train_sim directory (augmented set)")
    # Distillation. Capacity pays on held-out regimes -- width 128 at 12000 steps is
    # +0.104 on 9/9 Re folds -- but the container charges 1.46x per member for it,
    # which eats 56% of the gain, and width 160 loses money outright. A student at
    # the shipped width costs nothing extra at inference, so the question is how
    # much of the teacher's advantage it can absorb.
    #
    # The teacher must be trained on *this run's* training split. A teacher that saw
    # the fold's held-out regimes would leak them into the student through the
    # distillation target, and the fold would report a number that no shipped model
    # could reproduce. `tools/kfold.py` writes one teacher per fold for exactly this.
    ap.add_argument("--distill-from", default=None,
                    help="teacher checkpoint(s), comma-separated; each must be "
                         "trained on the same split as this run. Several teachers "
                         "are averaged, which is the point: the scored object is an "
                         "ensemble mean, so a single teacher hands the student a "
                         "target the shipped model never produces")
    ap.add_argument("--w-distill", type=float, default=1.0,
                    help="weight on the composite loss against the teacher's output")
    ap.add_argument("--cooperate-with", default=None,
                    help="comma-separated frozen, same-fold companion checkpoints; "
                         "train against the ensemble's forecast error, not teacher targets")
    ap.add_argument("--cooperate-own", type=float, default=0.0,
                    help="with --cooperate-with, blend this fraction of ordinary member loss "
                         "into the ensemble objective; evaluation always scores the ensemble")
    ap.add_argument("--cooperate-grad-scale", type=float, default=1.0,
                    help="multiply cooperative gradients before optimizer/clipping; "
                         "member averaging otherwise reduces them by the ensemble size")
    ap.add_argument("--cooperate-start", type=int, default=0,
                    help="train only the individual objective through this step, then "
                         "transition to the requested cooperative blend")
    ap.add_argument("--cooperate-ramp", type=int, default=0,
                    help="steps to linearly transition from individual to cooperative loss")
    ap.add_argument("--cooperate-route", choices=("all", "tke"), default="all",
                    help="route all losses to the ensemble, or only TKE while retaining "
                         "individual field/spectrum/bulk objectives")
    ap.add_argument("--cooperate-tke-scale", type=float, default=1.0,
                    help="ensemble TKE multiplier for --cooperate-route tke")
    ap.add_argument("--cooperate-mean-grad-scale", type=float, default=1.0,
                    help="multiply only the temporal-mean output gradient of the ensemble "
                         "objective; forward values and inference stay unchanged")
    ap.add_argument("--cooperate-drop-prob", type=float, default=0.0,
                    help="probability of omitting one companion from the training mean; "
                         "evaluation always uses every recorded companion")
    ap.add_argument("--distill-only", action="store_true",
                    help="drop the real-target term and match the teacher alone")
    ap.add_argument("--temporal-rank", type=int, default=0,
                    help="T1 (2026-09-17): decode the 20 output frames from this many fixed DCT temporal modes "
                         "(coefficient fields from the output conv). 0 = the shipped direct 20-frame output (inert).")
    ap.add_argument("--distill-gain", type=float, default=0.0,
                    help="DS3 (2026-09-17): rescale each cell's temporal fluctuation of the TEACHER MEAN toward the "
                         "teachers' average temporal variance at this dose before it becomes the target (the shipped "
                         "post-averaging gain, applied to the distillation target). 0 = plain teacher mean (inert).")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from another run's best.pt. With --mode sim_only "
                         "first, this is the sequential curriculum: simulation supplies "
                         "generic structure from all 100 regimes, then real-only "
                         "fine-tuning overwrites the regime-specific simulated bias "
                         "wherever real data exists and leaves the prior only where it "
                         "does not. Joint training cannot do that -- its simulated "
                         "gradients act on the final weights everywhere.")
    ap.add_argument("--aux-s2r", type=float, default=0.0,
                    help="weight on the sim->real auxiliary objective. The trunk is "
                         "asked to correct a simulated window so its statistics match "
                         "real measurements at the SAME regime, using the 82 paired "
                         "regimes. No synthetic sample is created; the training set "
                         "stays the release. Tests whether learning the mapping as a "
                         "side task helps where only simulation covers a regime.")
    ap.add_argument("--regime-bag", type=float, default=1.0,
                    help="train this fraction of REGIMES, drawn at random per seed. "
                         "Bagging over regimes rather than seeds is the only ensemble "
                         "diversity aligned with the shift the private set applies: "
                         "each member has extrapolated across different gaps in the "
                         "(Re, AoA) grid. Seed, loss-weight, schedule and architecture "
                         "diversity all measured flat, because none of them vary along "
                         "the axis that fails.")
    ap.add_argument("--real-frac", type=float, default=1.0,
                    help="fraction of real trajectories to keep, evenly spaced over "
                         "the (Re, AoA) grid. Thinning real coverage is how you find "
                         "out whether sim2real transfer works at all: with 82 of 100 "
                         "grid points measured, any target regime is bracketed by real "
                         "neighbours 1275 apart in Re, and interpolating between them "
                         "beats converting simulated knowledge at the target. Transfer "
                         "should only pay once that bracket is gone.")
    # Real-coverage dose (2026-11-02, research/rb_2026-11-02): --real-frac thins by
    # position in a string-sorted list, which is neither nested across doses nor a
    # distance on either axis. Dropping whole angles is both: the nearest real angle
    # to a held-out one recedes 5 -> 10 -> 15 deg while simulation stays put.
    ap.add_argument("--real-drop-aoa", default="",
                    help="angles whose real trajectories are removed from training (not "
                         "evaluated, simulation untouched), e.g. '10,15'")
    # Clock-corrected simulation (2026-11-03, research/rb_2026-11-02 §3e): resampling the
    # CFD from convective to tunnel time (tools/clock_sim.py) leaves high-Re trajectories
    # ~6x shorter than low-Re ones, and the uniform window draw would then reweight the
    # simulation toward low Re.
    ap.add_argument("--sim-equal-traj", action="store_true",
                    help="give every simulated trajectory the same sampling mass whatever its "
                         "length (the release's equal-length sims are unaffected)")
    ap.add_argument("--sim-paired", action="store_true",
                    help="keep only simulated trajectories that have a real partner. "
                         "Simulation at a regime with no real measurement is the only "
                         "signal the model has there, and it pulls the prediction "
                         "toward simulation: on val_re -- real data at Reynolds "
                         "numbers held out of training -- the unpaired trajectories "
                         "cost 0.031 of rel_l2 (6 SE) and 0.117 of SPS (3.9 SE). On "
                         "held-out angles, where real data at neighbouring angles "
                         "lets the model interpolate, they are worth +0.02 (noise). "
                         "The Decision Phase tests the first case.")
    ap.add_argument("--sim-scramble", choices=SIM_SCRAMBLES, default="none",
                    help="degrade the simulated half of each batch while keeping "
                         "stated statistics exact -- the mechanism test for the "
                         "+0.71 (see scramble_sim). Real windows are untouched")
    ap.add_argument("--phase-aug", type=int, default=1, choices=(1, 4),
                    help="decimation phases of the raw real grid to train on; 4 "
                         "resamples the PIV noise per window (eval stays phase 0)")
    ap.add_argument("--sim-noise", type=float, default=0.0,
                    help="iid Gaussian noise added to simulated windows (input and "
                         "target independently), in normalised units -- RealPDEBench's "
                         "noise-injection recommendation. Real TI is ~0.086 in these "
                         "units; try 0.01-0.03")
    ap.add_argument("--sim-shadow", action="store_true",
                    help="apply a random real valid-cell mask (matched on AoA, drawn "
                         "from this run's training real trajectories) to each simulated "
                         "window -- RealPDEBench's mask-training recommendation. The "
                         "CFD has no optical shadow; real windows are 6-11%% exact zeros")
    ap.add_argument("--sim-frac", type=float, default=1.0,
                    help="fraction of simulated trajectories to keep, evenly spaced "
                         "over the (Re, AoA) grid so coverage thins rather than "
                         "collapsing to a corner of it")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-film", action="store_true",
                    help="override --film off (kfold's BASE hardcodes --film, and a "
                         "store_true cannot be un-set by repetition). For the "
                         "conditioning ablation under the current recipe")
    ap.add_argument("--film", action="store_true",
                    help="condition on modality via FiLM instead of blending regimes")
    ap.add_argument("--aux-p", type=float, default=0.0,
                    help="weight on the simulated-pressure auxiliary loss")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--clip-norm", type=float, default=1.0,
                    help="gradient norm limit; 0 disables clipping")
    ap.add_argument("--clip-mode", choices=("global", "separate"), default="global",
                    help="clip all parameters together, or each optimizer separately")
    # The default predates the loss correction of 2026-08-12 and is kept only so
    # archived runs replay: the shipped recipe passes --w-mvpe 0.0. Removing the term
    # was worth +0.48 of final, and it stays removed -- the dose ladder was re-run at
    # the identified mvpe weight of 0.093 (not the 0.032 it was dropped on) and is
    # flat to negative at every dose (`docs/archive/HANDOFF_2026-09-10.md` section 0a).
    ap.add_argument("--w-mvpe", type=float, default=0.2)
    ap.add_argument("--w-mse", type=float, default=1.0,
                    help="plain MSE. The scorer never computes it -- rel_l2 is a "
                         "per-window *relative* L2 and carries weight 0.474 -- so the "
                         "endpoint of raising --w-rel is removing this term entirely. "
                         "Measured, that endpoint loses: the peak is --w-rel 8 with "
                         "MSE kept at 1.0.")
    ap.add_argument("--w-rel", type=float, default=0.5)
    ap.add_argument("--w-tke", type=float, default=0.5)
    ap.add_argument("--w-spec", type=float, default=0.1)
    ap.add_argument("--w-tmean", type=float, default=0.0)
    ap.add_argument("--w-sim-loss", type=float, default=1.0,
                    help="scale on the simulated windows' loss contribution; 1.0 "
                         "is the shipped single-loss path")
    ap.add_argument("--sim-terms", choices=("all", "nofluct", "mse"), default="all",
                    help="'nofluct' drops tke and spec for simulated windows, so "
                         "sim teaches structure/transport but not fluctuation "
                         "statistics")
    ap.add_argument("--w-mcorr", type=float, default=0.0,
                    help="with --decompose: L2 tax on the mean head's correction, "
                         "pushing the out-of-distribution mean onto the anchor")
    ap.add_argument("--grad", choices=("sum", "config"), default="sum",
                    help="'config' combines the shrinkage-group gradient "
                         "(mse+rel+tmean) and the amplitude-group gradient "
                         "(tke+spec) conflict-free (ConFIG, arXiv:2408.11104, "
                         "two-gradient closed form) instead of summing them. "
                         "Measured 2026-08-23: the two groups' gradients have "
                         "negative cosine on 58%% of steps, so the summed update "
                         "destroys one group's progress to buy the other's.")
    ap.add_argument("--opt", choices=("adamw", "muon", "nsgd", "soap", "soapmuon"),
                    default="adamw",
                    help="update rule for the matrix-shaped parameters. `nsgd` is "
                         "realistix.muon.ScaledMomentum: Muon's per-layer step SIZE "
                         "without its orthogonalisation, which is the control that "
                         "separates the two mechanisms (docs/archive/model_analysis.md 3.5). "
                         "It needs its own lr -- about 0.006, matched on realised "
                         "relative step -- because raw momentum here is nearly rank "
                         "one and rescaling it to an orthogonalised matrix's norm "
                         "puts the whole norm in one direction")
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="Newton-Schulz iterations. Fixed at 5 since the module was "
                         "written and never swept; it sets how flat the update's "
                         "singular-value spectrum is, which is the quantity the rank "
                         "measurement points at (update stable rank 84.3 under Muon "
                         "against 6.6 under AdamW)")
    ap.add_argument("--muon-hidden-only", action="store_true",
                    help="follow the upstream Muon guidance verbatim: 'use Muon to "
                         "optimize all the convolutional filters except the first "
                         "one, and AdamW to optimize everything else' -- i.e. move "
                         "the stem AND the zero-init output convolution to AdamW. "
                         "The shipped recipe orthogonalises both, and the stem is "
                         "the one level whose weight stable rank FELL under Muon")
    ap.add_argument("--muon-1d", action="store_true",
                    help="put the 0/1-D parameters (biases, GroupNorm scales, the "
                         "per-domain affines) on Muon as well instead of AdamW. A 1-D "
                         "tensor reshapes to (n, 1), so the iteration returns its unit "
                         "vector -- normalised SGD on that tensor")
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--muon-mean-lr-scale", type=float, default=1.0,
                    help="with --decompose: scale only the mean output matrix's Muon lr; "
                         "its bias stays on the ordinary AdamW schedule")
    ap.add_argument("--muon-no-nesterov", action="store_true")
    ap.add_argument("--sps-objective", choices=("bulk", "oracle"), default="bulk",
                    help="bulk reward alone, or reward times the scorer's accuracy prefactors")
    ap.add_argument("--norm-groups", type=int, default=8,
                    help="GroupNorm groups in the U-Net; must divide every normalized width")
    ap.add_argument("--w-lowf", type=float, default=0.0,
                    help="weight on the slow/large-scale space-time band loss "
                         "(losses.lowfreq_rel): omega in [1,2] cycles/window, "
                         "k < 8 -- the band holding 56%% of held-out error that "
                         "no term has ever weighted above flat")
    ap.add_argument("--w-elead", type=float, default=0.0,
                    help="weight on losses.lead_energy: relative L2 between the per-lead "
                         "fluctuation-energy profiles of prediction and target. 0 = off. "
                         "**DEAD** (2026-09-07): -0.098 at dose 1.0 and -0.038 at 0.3, 0/4 folds "
                         "both times, and Codex's target-assisted oracle -- impose the target's "
                         "own lead profile at fixed local mean and TKE -- reads -0.030 on 4/4, so "
                         "the allocation the term pushes toward is worse, not better. Do not re-run")
    ap.add_argument("--w-dlead", type=float, default=0.0,
                    help="weight on losses.diff_energy: relative L2 between the per-lead "
                         "frame-difference energy profiles (targets the late-lead "
                         "discontinuity and the over-smooth mid-horizon). 0 = off. "
                         "**DEAD** (2026-09-07): -0.055 at dose 1.0 and -0.027 at 0.3, 0/4 folds. "
                         "The defect it targets is real per member but averages out across an "
                         "ensemble -- see losses.diff_energy. Do not re-run")
    ap.add_argument("--w-sps", type=float, default=0.0,
                    help="weight on losses.sps_bulk, the SPS reward at oracle width "
                         "(1 - mean exp(-|e|/sigma) over scored elements). The one "
                         "subscore the 887158 loss re-derivation left out, because "
                         "sps looked like the head's business; tools/sps_decompose.py "
                         "shows the forecaster sets the ceiling on it. It is the only "
                         "term here that is not an L2 norm -- SPS decays exponentially "
                         "in the error where L2 grows quadratically, so they disagree "
                         "maximally on the tail (2026-08-24)")
    ap.add_argument("--w-sps-mean", type=float, default=0.0,
                    help="weight on losses.sps_bulk_mean: the bulk term restricted "
                         "to the per-pixel TIME-MEAN error. The scorer's TKE is the "
                         "temporal variance about each signal's own time-mean, so "
                         "this term has PROVABLY zero direct gradient on predicted "
                         "TKE (the gradient is constant across t at every pixel). "
                         "Built 2026-08-25 after tools/error_anatomy.py showed the "
                         "full-field term's TKE cost is amplitude damping at k<8 "
                         "that only appears at dose >= 2")
    ap.add_argument("--w-sps-until", type=int, default=0,
                    help="apply --w-sps only until this step, then 0 -- the "
                         "schedule form of the dose-stop one-knob finding: bulk "
                         "pressure pays early and damps late (2026-08-27)")
    ap.add_argument("--sps-width", type=float, default=1.0,
                    help="total interval width in units of |e| for --w-sps: 1.0 for "
                         "the asymmetric bounds the shipped head emits, 2.0 symmetric")
    ap.add_argument("--w-unroll", type=float, default=0.0,
                    help="weight on a second, autoregressive application of the "
                         "model: its own (masked, renormalized) prediction becomes "
                         "the next input window and the composite loss is applied "
                         "to frames t+40..t+59. Gradients flow through both "
                         "applications, so the first window's output is trained to "
                         "carry the state -- including the meandering phase -- that "
                         "the second application needs. Inference is unchanged.")
    ap.add_argument("--mults", default="1,2,4")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--cond", choices=("enc", "all", "norm", "norm_re", "film_re"),
                    default="enc",
                    help="where the modality token is allowed to act; see UNet")
    ap.add_argument("--out-gain", action="store_true",
                    help="per-modality scale on the predicted delta")
    ap.add_argument("--arch", choices=("unet", "fno"), default="unet",
                    help="a second architecture exists to make the ensemble "
                         "structurally diverse, not because it is better alone")
    ap.add_argument("--fno-layers", type=int, default=4)
    ap.add_argument("--fno-modes1", type=int, default=16)
    ap.add_argument("--fno-modes2", type=int, default=17)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay on the weights; 0 disables")
    ap.add_argument("--attn", action="store_true")
    # Topology inside the U-Net family. The *operator* families are closed on
    # external evidence (RealPDEBench ranks a plain U-Net first of ten on this
    # flow), but which U-Net has never been swept -- only its width, depth, block
    # count and level count. Screen these on the Re fold set, not the angle one:
    # linknet and transunet change capacity, and the angle folds train on ~30% less
    # data than the shipped model, which is exactly where a capacity change
    # flatters itself.
    ap.add_argument("--xscan", choices=("", "deep", "all"), default="",
                    help="directional streamwise scan operator (models.unet.XScan): "
                         "first-order recurrences along +x and -x with learned "
                         "per-channel decay -- forward propagation of information "
                         "with spatial correlation decay, stated as an operator "
                         "rather than left to stacked 3x3 hops. 'deep' = last "
                         "encoder level + bottleneck, 'all' = every encoder level. "
                         "Identity at init (2026-08-25)")
    ap.add_argument("--variant", choices=("unet", "linknet", "nested", "transunet"),
                    default="unet")
    ap.add_argument("--tf-layers", type=int, default=4,
                    help="transformer layers at the bottleneck, --variant transunet")
    ap.add_argument("--mimo", type=int, default=0,
                    help="M output groups hosted in one trunk (MIMO): training feeds "
                         "M independently sampled same-modality windows concatenated "
                         "in channels and each group learns its own window's future; "
                         "inference tiles the single window and the model returns the "
                         "group mean, so eval, the head and packing are unchanged. "
                         "The M-1 extra members ride in the width capacity that is "
                         "measured dead for a single model, and the prize is the "
                         "ensemble's 2 x 0.85 ms of container time")
    ap.add_argument("--mimo-rho", type=float, default=0.0,
                    help="probability a training row feeds the SAME window to all "
                         "groups (input repetition, the standard MIMO knob for "
                         "capacity-tight trunks)")
    ap.add_argument("--in-stats", action="store_true",
                    help="hand the network its input window's per-cell time-mean and "
                         "temporal std as 4 extra stem channels -- the two summary "
                         "fields the anatomy says it uses implicitly (the mean drives "
                         "half the delta) or computes poorly (std needs squaring). "
                         "Computed inside UNet.forward, so eval and packing are "
                         "unchanged")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear lr warmup steps before the cosine, on both "
                         "optimizers. Never used anywhere in this project; Muon "
                         "starts at lr 0.02 momentum 0.95 from step 0")
    ap.add_argument("--anchor", choices=("last", "wmean", "blend"), default="last",
                    help="what the predicted delta is added to: the last observed "
                         "frame (shipped), the input window's time-mean, or a "
                         "per-lead-time learned mix of the two")
    ap.add_argument("--out-cumsum", action="store_true",
                    help="the output conv emits per-lead increments; the forecast is their "
                         "cumulative sum over the lead axis on top of the anchor")
    ap.add_argument("--warp", action="store_true",
                    help="predict a per-output-frame displacement field and advect "
                         "the anchor through it before adding the residual")
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--snapshots", action="store_true",
                    help="write step<n>.pt at every eval, for weight averaging")
    # --- 2026-09-05 broad-search screens (docs/IDEAS_BROAD_2026-09-05.md §3). All inert at defaults.
    ap.add_argument("--sam-rho", type=float, default=0.0,
                    help="scale-aware SAM radius: recompute the same batch's gradient at "
                         "theta + rho * D^2 g / ||D g||, D_i = |theta_i| + 0.01 RMS(tensor), then "
                         "take one ordinary step with that gradient (rank-3 screen); 0 = off")
    ap.add_argument("--sam-random", action="store_true",
                    help="control for --sam-rho: a seeded random direction replaces g at the same "
                         "metric radius (identical cost and machinery)")
    ap.add_argument("--raw-avg", default="",
                    help="comma-separated START-END step windows; the uniform average of the RAW "
                         "(non-EMA) weights over each window is saved as rawavg_START_END.pt "
                         "(rank-2 screen, dense iterate averaging)")
    # --- 2026-09-07 optimizer deep dive (research/optimizer_2026-09-07). All inert at defaults.
    ap.add_argument("--muon-order", choices=("momentum_first", "polar_first"),
                    default="momentum_first",
                    help="'polar_first' orthogonalises each step's gradient and takes the "
                         "momentum over the polar factors (noise-aware polarisation); "
                         "'momentum_first' is the shipped msign(EMA(g)), bit-identical. "
                         "**DEAD** (2026-09-07): -0.055/-0.094 at lr 0.02 (mim1), -0.061 on 0/4 "
                         "folds at lr 0.04, and -0.074 against the --zero-rest baseline with both "
                         "groups cleared, so the clip-memory confound is not the cause. It wins a "
                         "200-step training-loss probe and loses every held-out read")
    ap.add_argument("--muon-embed-adamw", action="store_true",
                    help="move every nn.Embedding weight (the 28 per-modality FiLM affines, "
                         "(2, 2C) each) from Muon to the AdamW group -- Muon's own embedding "
                         "rule -- and zero their gradients every step. Off = shipped, "
                         "bit-identical. **NULL** (2026-09-07): -0.003, 1/4 folds, every run "
                         "inside +/-0.02; the shipped placement costs nothing")
    ap.add_argument("--muon-convt-fix", action="store_true",
                    help="flatten ConvTranspose2d weights as (C_out, C_in*k*k) for the polar "
                         "update instead of the shipped (C_in, C_out*k*k); two tensors. This is "
                         "a different metric, not a bug fix -- either unfolding is a choice, and "
                         "output-first also lowers those tensors' update norm by sqrt(2). "
                         "**NULL** (2026-09-07): -0.011, 1/4 folds")
    ap.add_argument("--muon-stem-cov", type=float, default=0.0,
                    help="right-precondition the stem's Nesterov momentum by "
                         "(C + d*tr(C)/40*I)^-1 before the polar step, C = EMA(0.99) of the "
                         "input window's 40x40 second moment, refreshed every 25 steps; the "
                         "update is rescaled to the plain polar norm (a restricted Newton-Muon: "
                         "the 40 input channels are 20 frames of the same field and their second "
                         "moment has condition number ~3e6). d = this value; 0 = off. "
                         "**TOO SMALL** (2026-09-07): +0.006 at d=0.1, consistent in sign on 4/4 "
                         "folds but inside the rig's 0.02-0.03 resolution")
    ap.add_argument("--wd-muon", type=float, default=None,
                    help="weight decay for the Muon group only (default: --wd for both). "
                         "**SHIPPED at 0.03** (918211, 2026-09-08; WD_MUON in train_final.sh). "
                         "Dose: 0.01 +0.006 (3/4), 0.03 +0.018 (12/12 runs, 4/4 folds), 0.1 "
                         "-0.008 (over-regularised). It shrinks every Muon-group matrix's norm "
                         "40-60%% by step 6000 at unchanged stable rank, and its largest read is "
                         "the angle suite (+0.057, 4/4, rel_l2 -1.9..-2.7%%). Read the board "
                         "caveat in train_final.sh: the accuracy channel came in NEGATIVE there "
                         "because tke paid -0.264")
    ap.add_argument("--switch-ema", type=int, default=0,
                    help="every this many steps copy the EMA weights back into the online "
                         "model (Switch-EMA, Li et al. 2024); optimizer state is kept. 0 = off. "
                         "**NULL** (2026-09-07): +0.002 at k=500, 8 runs, nothing beyond +/-0.021 "
                         "-- changing the training path this way does nothing the EMA estimator "
                         "was not already doing")
    ap.add_argument("--muon-adavar", action="store_true",
                    help="variance adaptation AFTER the orthogonalisation: U = O / sqrt(V), "
                         "V = 0.95 V + 0.05 O^2, rescaled to ||O||_F per tensor (rank-6 screen)")
    ap.add_argument("--meta-alpha", type=float, default=0.0,
                    help="first-order cross-angle meta-update (rank-7 screen): per step draw a "
                         "support batch from all training angles but one and a query batch from "
                         "that angle B; virtual step theta' = theta - a g_A with |a g_A| = "
                         "meta_alpha * |theta| (relative displacement), outer gradient "
                         "(g_A + g_B(theta'))/2 to the ordinary optimizer; 0 = off")
    ap.add_argument("--meta-random", action="store_true",
                    help="control for --meta-alpha: identical machinery, but both batches are "
                         "drawn from all angles (random partition instead of by angle)")
    ap.add_argument("--morph-prob", type=float, default=0.0,
                    help="CFD-guided angle morphing (bold round, 2026-09-06): with this probability a "
                         "real window (input AND target) is shifted by f*[mean_sim(Re,a_nb) - mean_sim(Re,a)] "
                         "in normalised units on its valid cells, i.e. its mean flow is moved toward a "
                         "neighbouring angle a_nb = a +- 5 using the paired simulation's mean-flow difference; "
                         "0 = off (bit-identical)")
    ap.add_argument("--morph-angles", default="2.5,7.5,12.5,17.5",
                    help="target angles the morph may synthesise (f = |a'-a|/5 toward the neighbour on that "
                         "side); e.g. '10' morphs the 5 and 15 deg windows all the way to 10 for a fold test")
    ap.add_argument("--morph-extrap", action="store_true",
                    help="with --morph-prob: when the neighbour angle's simulation is absent (a truly "
                         "novel angle), extrapolate the mean-flow trend instead: delta = f*[mean_sim(Re,a) "
                         "- mean_sim(Re,a_opposite)] with a_opposite on the far side")
    ap.add_argument("--foil-frame", choices=("none", "foil", "reverse"), default="none",
                    help="bold round (Codex, models/foil_frame.py): forecast the residual in the foil-aligned "
                         "frame inferred from the observed body mask and blend back; 'reverse' negates the "
                         "inferred angle as the control; 'none' = the plain UNet, bit-identical")
    ap.add_argument("--geo-in", action="store_true",
                    help="bold round: append two geometry channels computed inside forward from the window's "
                         "own exact-zero mask -- the invalid mask and a quantised distance-to-invalid field -- "
                         "so the foil/shadow geometry is an explicit smooth input; off = bit-identical")
    ap.add_argument("--zero-rest", action="store_true",
                    help="also zero the AdamW (1-D parameter) group's gradients every step. The "
                         "shipped recipe zeroes only the Muon group, so biases, GroupNorm affines "
                         "and FiLM embeddings have trained on ACCUMULATED gradients in every Muon "
                         "run (found 2026-09-05); off = the shipped behaviour, bit-identical")
    # ---- 2026-09-11 frontier-weighting screen (F2; inert at its default) ----
    ap.add_argument("--frontier-weight", type=float, default=0.0,
                    help="multiply the REAL-window sampling probability by this factor on the "
                         "training cells axis-adjacent (Re +-1275, AoA +-5) to simulated-grid cells "
                         "that have no real training data in this run; 0 or 1 = the shipped uniform "
                         "sampler, bit for bit; simulated sampling and the real:sim mix are unchanged")
    # ---- 2026-09-10 improvement screens (all inert at their defaults) ----
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="DropPath probability on Block residual branches at the widest level, "
                         "ramping linearly from 0 at the first level; training only; private RNG")
    ap.add_argument("--chan-drop", type=float, default=0.0,
                    help="channel dropout after the second SiLU in widest-level Blocks; training only")
    ap.add_argument("--frame-mask", type=float, default=0.0,
                    help="probability of zeroing each normalised input frame 1..T_IN-1 (the anchor "
                         "frame is never masked; targets and the window scale are unchanged)")
    ap.add_argument("--sps-domain", choices=("all", "real"), default="all",
                    help="'real' drops the simulated rows' contribution from the sps_bulk numerator "
                         "while keeping the whole batch's scored-element denominator")
    ap.add_argument("--sim-sps-scale", type=float, default=1.0,
                    help="multiply simulated rows' wscale inside sps_bulk by this factor (simulation "
                         "is non-dimensional, real is m/s; ~0.18 puts sim errors in real-like units)")
    ap.add_argument("--time-dilate", type=float, default=0.0,
                    help="probability that a sampled window is re-read at a different clock "
                         "(see gpudata.GPUWindows); 0 = shipped sampler")
    ap.add_argument("--dilate-factors", default="2",
                    help="comma list of clock factors for --time-dilate: 2 (every second raw frame) "
                         "and/or 0.5 (linear midpoints)")
    ap.add_argument("--dilate-domain", choices=("real", "all"), default="real",
                    help="dilate real windows only, or real and simulated")
    ap.add_argument("--mix-prob", type=float, default=0.0,
                    help="continuous sim/real fraction (owner's idea, 2026-09-10): with this probability a real "
                         "window's input AND target are shifted by a*(mean_sim(cell) - mean_real(cell)) on valid "
                         "cells, a ~ U(0, --mix-alpha), and the modality FiLM sees a instead of 0; 0 = shipped")
    ap.add_argument("--mix-alpha", type=float, default=0.5, help="upper bound of the mixing fraction a")
    ap.add_argument("--mix-cond", choices=("alpha", "real"), default="alpha",
                    help="'alpha': FiLM interpolates real/sim rows at a; 'real': the network is told nothing "
                         "(a = 0 conditioning) and must infer the shifted mean flow")
    ap.add_argument("--level-prob", type=float, default=0.0,
                    help="LEVEL screen (Codex round 9): with this probability a real window's input AND target are "
                         "shifted by beta * (input temporal-mean field), beta ~ U(-b, b), on cells observed in every "
                         "frame of the trajectory -- a scalar level change of the mean flow with no simulation "
                         "involved; tags/wscale unchanged; 0 = shipped")
    ap.add_argument("--level-beta", type=float, default=0.10, help="half-range b of the level factor")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop training after this step's evaluation, leaving the --steps schedule "
                         "untouched (0 = run to --steps)")
    ap.add_argument("--stem", choices=("dense", "temporal", "untied"), default="dense",
                    help="input stem: the shipped dense time-as-channels conv, a shared "
                         "(2+1)D temporal stem, or its untied per-frame control (rank-4 screen)")
    ap.add_argument("--fixed-step", type=int, default=0,
                    help="with --real-trajs all, the step to keep (0 = the last)")
    ap.add_argument("--real-trajs", choices=TRAIN_SETS, default="train",
                    help="how much of the real release to train on (see TRAIN_SETS)")
    ap.add_argument("--cpu-data", action="store_true",
                    help="use the old DataLoader path (CPU-bound; kept for reference)")
    ap.add_argument("--distill-unsafe", action="store_true",
                    help="accept a teacher that does not record every holdout axis; "
                         "its result is not held out and must not be quoted as such")
    ap.add_argument("--out", required=True)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if args.cooperate_with:
        if (args.distill_from or args.distill_only or args.distill_unsafe or args.cpu_data
                or args.aux_p or args.aux_s2r or args.mimo or args.grad != "sum"
                or args.meta_alpha or args.sam_rho or args.w_unroll or args.w_lowf
                or args.w_elead or args.w_dlead or args.w_sps_mean or args.w_mcorr
                or args.w_sim_loss != 1 or args.sim_terms != "all" or args.sim_loss_frac is not None):
            ap.error("--cooperate-with supports the plain GPU composite+SPS path only, "
                     "with verified companions and no distillation")
        if not 0 <= args.cooperate_own <= 1:
            ap.error("--cooperate-own must be in [0, 1]")
        if not 0 < args.cooperate_grad_scale < float("inf"):
            ap.error("--cooperate-grad-scale must be finite and positive")
        if args.cooperate_start < 0 or args.cooperate_ramp < 0:
            ap.error("cooperative start/ramp must be nonnegative")
        if not 0 <= args.cooperate_drop_prob <= 1:
            ap.error("--cooperate-drop-prob must be in [0,1]")
        if not 0 < args.cooperate_tke_scale < float('inf') or not 0 < args.cooperate_mean_grad_scale < float('inf'):
            ap.error("cooperative term/mean-gradient scales must be finite and positive")
        if args.cooperate_route != 'tke' and args.cooperate_tke_scale != 1:
            ap.error("--cooperate-tke-scale requires --cooperate-route tke")
        if args.cooperate_route == 'tke' and args.cooperate_mean_grad_scale != 1:
            ap.error("mean-gradient scaling is not meaningful for the mean-invariant TKE route")
    elif (args.cooperate_own or args.cooperate_grad_scale != 1 or args.cooperate_start or args.cooperate_ramp
          or args.cooperate_route != 'all' or args.cooperate_tke_scale != 1
          or args.cooperate_mean_grad_scale != 1 or args.cooperate_drop_prob):
        ap.error("cooperative loss controls require --cooperate-with")
    cooperate_rng = random.Random(args.seed + 20903)
    sps_fn = sps_oracle if args.sps_objective == "oracle" else sps_bulk
    if not (0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
        ap.error("Adam betas must be in [0, 1)")
    if not 0 < args.adam_eps < float("inf") or not 0 <= args.clip_norm < float("inf"):
        ap.error("--adam-eps must be finite and positive; --clip-norm finite and nonnegative")
    if args.norm_groups != 8 and args.arch != "unet":
        ap.error("--norm-groups requires --arch unet")
    if args.muon_no_nesterov and args.opt != "muon":
        ap.error("--muon-no-nesterov requires --opt muon")
    if not 0 < args.muon_mean_lr_scale < float("inf"):
        ap.error("--muon-mean-lr-scale must be finite and positive")
    if args.muon_mean_lr_scale != 1 and (not args.decompose or args.opt != "muon"
                                        or args.muon_1d or args.muon_hidden_only):
        ap.error("--muon-mean-lr-scale requires --decompose, --opt muon and ordinary matrix grouping")
    if args.sim_stride != 1 and not (args.cpu_data or args.aux_p):
        ap.error("--sim-stride is only implemented by --cpu-data; use --sim-batch-frac "
                 "to change simulation exposure with the GPU sampler")
    if args.sim_batch_frac is not None:
        if not 0 <= args.sim_batch_frac <= 1:
            ap.error("--sim-batch-frac must be in [0, 1]")
        if args.mimo or args.cpu_data or args.aux_p or args.mode not in ("plus_sim", "plus_sim_all"):
            ap.error("--sim-batch-frac requires a mixed, non-MIMO GPU run")
    if args.sampling != "window" and (args.cpu_data or args.aux_p):
        ap.error("--sampling requires the GPU sampler")
    if args.sim_loss_frac is not None:
        if not 0 < args.sim_loss_frac < 1:
            ap.error("--sim-loss-frac must be strictly between 0 and 1")
        if (args.mode not in ("plus_sim", "plus_sim_all") or args.cpu_data or args.aux_p
                or args.mimo or args.grad != "sum" or args.meta_alpha or args.sam_rho
                or args.distill_from or args.aux_s2r or args.w_unroll or args.w_lowf
                or args.w_elead or args.w_dlead or args.w_sps_mean or args.w_mcorr):
            ap.error("--sim-loss-frac supports the plain mixed GPU composite+SPS path only")
        if args.sim_batch_frac is not None and not 0 < round(args.batch * args.sim_batch_frac) < args.batch:
            ap.error("--sim-loss-frac requires at least one row from each modality")
    if args.no_film:
        args.film = False
    if args.switch_ema and not args.ema:
        raise SystemExit("--switch-ema needs --ema: there is no average to switch to")
    if args.opt != "muon" and (args.muon_order != "momentum_first"
            or args.muon_embed_adamw or args.muon_convt_fix
            or args.muon_stem_cov != 0 or args.wd_muon is not None):
        raise SystemExit("the new --muon-* / --wd-muon flags require --opt muon")
    if args.muon_stem_cov != 0 and not 0 < args.muon_stem_cov < float("inf"):
        raise SystemExit("--muon-stem-cov must be finite and positive, or 0 to disable")
    if args.muon_stem_cov > 0:
        if (args.muon_order != "momentum_first" or args.muon_hidden_only
                or args.meta_alpha > 0):
            raise SystemExit("--muon-stem-cov requires momentum_first, the stem on "
                             "Muon, and the ordinary training step (no --meta-alpha)")
        if (args.arch != "unet" or args.stem != "dense" or args.in_stats
                or args.geo_in or args.mimo or args.aux_p > 0
                or args.foil_frame != "none"):
            raise SystemExit("--muon-stem-cov requires the untransformed 40-channel "
                             "dense UNet stem input")

    # A run that quietly loses its GPU is worse than one that dies. torch falls back
    # to CPU on a CUDA init failure -- seen once on node5, 2026-08-19, where the
    # only symptom was `amp: float32` in a log nobody reads until the study is over.
    # Such a run trains in fp32 instead of bf16, completes normally, and is then
    # scored against a bf16 baseline as if the comparison were paired. Refuse it.
    if not torch.cuda.is_available() and not os.environ.get("REALISTIX_ALLOW_CPU"):
        raise SystemExit("CUDA is not available; refusing to train on CPU because a "
                         "silent fp32 fallback would be paired against bf16 runs. "
                         "Set REALISTIX_ALLOW_CPU=1 if that is really what you want.")

    # Every holdout axis, not just the two regime ones: a distillation student can
    # only prove its teacher never saw the evaluation data if the teacher recorded
    # what it withheld along the same axis. Absent entries are stored as "" so that
    # "trained with no cell holdout" is distinguishable from "predates this record".
    for _k in ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE",
               "REALISTIX_HELDOUT_CELLS", "REALISTIX_HELDOUT_TAIL",
               # Part of the holdout specification: it decides whether the SIMULATED
               # corpus also drops the held-out cells. Added 2026-09-01 with the
               # option itself, after the first pair of runs using it came out
               # byte-identical in their recorded args and distinguishable only by
               # directory name -- the same provenance gap this file's other guards
               # exist to close.
               "REALISTIX_SIM_KEEPS_CELLS"):
        vars(args)[_k] = os.environ.get(_k) or ""
    if args.sim_dir:
        from realistix import splits as _splits
        _splits.SIM_DIR = args.sim_dir
        print(f"SIMDIR {args.sim_dir}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A fixed step is only ever taken inside the `step % eval_every == 0` branch, so
    # a step the schedule never evaluates at silently produces no best.pt and the run
    # still exits reporting success (best=nan). Catch it before eight hours of
    # training rather than after.
    if args.fixed_step:
        if args.fixed_step > args.steps:
            raise SystemExit(f"--fixed-step {args.fixed_step} is past --steps {args.steps}")
        if args.fixed_step % args.eval_every:
            raise SystemExit(f"--fixed-step {args.fixed_step} is not a multiple of "
                             f"--eval-every {args.eval_every}, so it is never evaluated "
                             f"and no best.pt would be written")

    gpu_ds = None
    if args.cpu_data or args.aux_p > 0:
        # This path builds the fixed "train" partition and ignores every sampler
        # control below, so a run asking for one would silently get 57 trajectories
        # instead of 81 (or an unbagged, unthinned, tail-including corpus) and
        # report itself as the requested recipe. Refuse instead.
        ignored = [n for n, v in (("--real-trajs", args.real_trajs != "train"),
                                  ("--regime-bag", args.regime_bag < 1.0),
                                  ("--real-frac", args.real_frac < 1.0),
                                  ("--sim-frac", args.sim_frac < 1.0),
                                  ("--sim-paired", args.sim_paired),
                                  ("--phase-aug", args.phase_aug > 1),
                                  ("REALISTIX_HELDOUT_TAIL", bool(HELDOUT_TAIL)))
                   if v]
        if ignored:
            raise SystemExit(
                "the CPU data path does not implement " + ", ".join(ignored)
                + " -- it always builds the fixed 'train' partition over all frames. "
                  "Drop the flag or drop --cpu-data/--aux-p.")
        ds, n_real, n_sim = build_train(args.mode, args.sim_stride,
                                        with_p=args.aux_p > 0)
        train_dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                              drop_last=True, pin_memory=True)
    else:
        # Whole corpus resident on the GPU; sampling is ~4 ms/batch against ~60 ms
        # through the DataLoader, and it no longer competes for host CPU.
        real_trajs = trajs_for("real", args.real_trajs)
        if args.regime_bag < 1.0:
            g = np.random.default_rng(1000 + args.seed)
            keep = max(1, int(round(len(real_trajs) * args.regime_bag)))
            idx = sorted(g.choice(len(real_trajs), keep, replace=False).tolist())
            real_trajs = [real_trajs[i] for i in idx]
            print(f"regime bag: {keep} of {len(idx) and len(trajs_for('real', args.real_trajs))}"
                  f" regimes for seed {args.seed}", flush=True)
        if args.real_frac < 1.0:
            keep = max(1, int(round(len(real_trajs) * args.real_frac)))
            idx = np.linspace(0, len(real_trajs) - 1, keep).round().astype(int)
            real_trajs = [real_trajs[i] for i in sorted(set(idx.tolist()))]
        if args.real_drop_aoa:
            drop = {int(v) for v in args.real_drop_aoa.split(",") if v}
            real_trajs = [t for t in real_trajs if t.aoa not in drop]
            print(f"real coverage: AoA {sorted(drop)} dropped from training, "
                  f"{len(real_trajs)} real trajectories at AoA "
                  f"{sorted({t.aoa for t in real_trajs})}", flush=True)
        sim_which = ("all" if args.mode in ("plus_sim_all", "sim_only")
                     else args.real_trajs)
        sim_trajs = (trajs_for("sim", sim_which) if args.mode != "real_only" else [])
        if args.mode == "sim_only":
            real_trajs = []          # stage 1 of a sequential curriculum
        if args.sim_paired and sim_trajs:
            real_stems = {t.stem for t in real_trajs}
            sim_trajs = [t for t in sim_trajs if t.stem in real_stems]
        if args.sim_frac < 1.0 and sim_trajs:
            keep = max(1, int(round(len(sim_trajs) * args.sim_frac)))
            idx = np.linspace(0, len(sim_trajs) - 1, keep).round().astype(int)
            sim_trajs = [sim_trajs[i] for i in sorted(set(idx.tolist()))]
        if args.time_dilate > 0 and args.w_unroll > 0:
            raise SystemExit("--time-dilate does not compose with --w-unroll")
        if args.time_dilate > 0 and (args.mimo or args.anchor != "last"):
            raise SystemExit("--time-dilate is validated for the plain last-frame anchor only")
        if not 0 <= args.time_dilate <= 1:
            raise SystemExit("--time-dilate must be a probability")
        gpu_ds = MixedGPUWindows(real_trajs, sim_trajs, device,
                                 frames="head" if HELDOUT_TAIL else "all",
                                 t_extra=20 if args.w_unroll > 0 else 0,
                                 phases=args.phase_aug, sim_batch_frac=args.sim_batch_frac,
                                 sampling=args.sampling,
                                 dilate_prob=args.time_dilate,
                                 dilate_factors=tuple(float(v) for v in args.dilate_factors.split(",") if v),
                                 dilate_domain=args.dilate_domain, dilate_seed=args.seed)
        if args.time_dilate > 0:
            print(f"time dilation: prob {args.time_dilate}, factors {args.dilate_factors}, "
                  f"domain {args.dilate_domain}", flush=True)
        if args.frontier_weight not in (0.0, 1.0):
            # F2 (2026-09-11): the frontier is the set of REAL training cells that touch a
            # simulated-grid cell with no real training data along one axis. Recomputed from
            # this run's real training list, so a fold's withheld cells count as gaps.
            if args.frontier_weight < 0:
                raise SystemExit("--frontier-weight must be >= 0")
            if args.sampling != "window" or args.meta_alpha > 0 or gpu_ds.real is None or gpu_ds.sim is None:
                raise SystemExit("--frontier-weight needs the plain window sampler with real and simulated data")
            from realistix import splits as _fsplits
            have = {(t.re, t.aoa) for t in real_trajs}
            # Gaps are measured against the FULL simulated grid (100 cells), so a fold's
            # withheld cells count as gaps even when their simulation is dropped too.
            grid = {(t.re, t.aoa) for t in _fsplits.available("sim")}
            gaps = grid - have
            frontier = {c for g in gaps for c in ((g[0] + 1275, g[1]), (g[0] - 1275, g[1]),
                                                  (g[0], g[1] + 5), (g[0], g[1] - 5)) if c in have}
            weights = [args.frontier_weight if (t.re, t.aoa) in frontier else 1.0 for t in real_trajs]
            gpu_ds.real.set_traj_weights(weights)
            ww = gpu_ds.real.window_weights
            mass = float(ww[torch.as_tensor([ (t.re, t.aoa) in frontier for t in real_trajs], device=ww.device)[gpu_ds.real.owner]].sum() / ww.sum())
            n_front = sum(1 for t in real_trajs if (t.re, t.aoa) in frontier)
            uni = float(torch.as_tensor([(t.re, t.aoa) in frontier for t in real_trajs], device=ww.device)[gpu_ds.real.owner].double().mean())
            print(f"frontier weight {args.frontier_weight}: {len(gaps)} gap cells, {n_front}/{len(real_trajs)} "
                  f"frontier trajectories, real sampling mass on the frontier {uni:.3f} -> {mass:.3f}; "
                  f"cells {sorted(frontier)}", flush=True)
        if args.sim_equal_traj:
            if args.sampling != "window" or args.meta_alpha > 0 or gpu_ds.sim is None:
                raise SystemExit("--sim-equal-traj needs the plain window sampler with simulated data")
            gpu_ds.sim.set_traj_weights([1.0 / float(c) for c in gpu_ds.sim.counts.tolist()])
            print(f"sim sampling: equal mass per trajectory over {len(gpu_ds.sim.fields)} trajectories "
                  f"({int(gpu_ds.sim.counts.min())}-{int(gpu_ds.sim.counts.max())} windows each)", flush=True)
        n_real = len(gpu_ds.real) if gpu_ds.real else 0
        n_sim = len(gpu_ds.sim) if gpu_ds.sim else 0
        train_dl = None
    # A fold may hold out only one axis. The Re-only fold set exists so that a
    # capacity change is measured on a training set within 12% of the shipped 81
    # trajectories rather than 30% smaller -- the one known blind spot of the
    # angle folds, and the reason `--mults 1,2` read +0.062 on 7/7 folds and
    # -0.013 on the board. Splits that fold does not populate are empty, and an
    # empty split must be absent rather than scored as zero.
    val = {}
    for _n in ("val_re", "val_aoa", "test_aoa"):
        _ds = build("real", _n, stride=40)
        if len(_ds):
            val[_n] = DataLoader(_ds, batch_size=64)
    if HELDOUT_TAIL:
        # The in-grid split: every regime is in training, and evaluation is on the
        # withheld tail of each trajectory. This is the question the leaderboard
        # asks -- its hidden development set is drawn from the same 5-angle grid --
        # and it is the one the regime folds cannot answer.
        from realistix.splits import available as _avail
        _ds = WindowDataset(_avail("real"), "real", stride=40, frames="tail")
        val["val_ingrid"] = DataLoader(_ds, batch_size=64)
        print(f"in-grid holdout: last {HELDOUT_TAIL:.0%} of every trajectory, "
              f"{len(_ds)} eval windows", flush=True)
    if not val:
        raise SystemExit("every validation split is empty; check "
                         "REALISTIX_HELDOUT_AOA / REALISTIX_HELDOUT_RE")
    # Bound once here rather than inside the eval loop, so the end-of-run summary
    # cannot depend on the loop having executed at least once.
    sel = ("val_ingrid" if "val_ingrid" in val
           else "val_aoa" if "val_aoa" in val else "val_re")
    print(f"mode={args.mode}: {n_real} real + {n_sim} sim windows", flush=True)
    if gpu_ds is not None and gpu_ds.real is not None and gpu_ds.sim is not None:
        frac = (gpu_ds.sim.n_mix / (gpu_ds.real.n_mix + gpu_ds.sim.n_mix)
                if args.sim_batch_frac is None else args.sim_batch_frac)
        if args.sim_loss_frac is not None and not 0 < round(args.batch * frac) < args.batch:
            ap.error("--sim-loss-frac requires sampled rows from both modalities")
        print(f"batch mix: {args.batch - round(args.batch * frac)} real + "
              f"{round(args.batch * frac)} sim; sampling={args.sampling}", flush=True)

    model = build_model(vars(args)).to(device)
    if getattr(args, "foil_frame", "none") != "none":
        raise SystemExit("the foil-frame variant was removed in the 2026-09-11 consolidation "
                         "(git checkpoint 36a08c1 has them)")
    if args.stop_at and not (0 < args.stop_at <= args.steps and args.stop_at % args.eval_every == 0):
        raise SystemExit("--stop-at must be in (0, --steps] and a multiple of --eval-every")
    if not 0 <= args.frame_mask < 1 or not 0 <= args.mix_prob <= 1:
        raise SystemExit("--frame-mask in [0, 1), --mix-prob in [0, 1]")
    if args.frame_mask > 0 and (args.cpu_data or args.mimo or args.anchor != "last"):
        raise SystemExit("--frame-mask is validated for the GPU sampler with the plain last-frame anchor only")
    if (args.sps_domain == "real" or args.sim_sps_scale != 1.0) and args.w_sps <= 0:
        raise SystemExit("--sps-domain / --sim-sps-scale need --w-sps > 0")
    if args.drop_path > 0 or args.chan_drop > 0:
        from realistix.models.unet import set_noise_seed
        set_noise_seed(args.seed + 1000003, device)
        print(f"stochastic regularisation: drop_path {args.drop_path} (ramped), "
              f"chan_drop {args.chan_drop} (widest level)", flush=True)
    frame_mask_gen = None
    if args.frame_mask > 0:
        frame_mask_gen = torch.Generator(device=device)
        frame_mask_gen.manual_seed(args.seed + 2000003)
    if args.sps_domain == "real" or args.sim_sps_scale != 1.0:
        if args.mode != "plus_sim" or not real_trajs or not sim_trajs:
            raise SystemExit("--sps-domain real / --sim-sps-scale need both modalities")
        if args.sim_loss_frac is not None:
            raise SystemExit("--sps-domain / --sim-sps-scale do not compose with --sim-loss-frac")
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        # A warm start carries everything the source checkpoint learned, so it is a
        # holdout bypass on exactly the same terms as distillation and was entirely
        # unguarded until 2026-09-01 (including recursively: a teacher warm-started
        # from an all-data checkpoint passed the distillation guard on its own args).
        for _k in ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE",
                   "REALISTIX_HELDOUT_CELLS", "REALISTIX_HELDOUT_TAIL"):
            sv, mv = ck["args"].get(_k), os.environ.get(_k) or ""
            if sv is None:
                print(f"warning: --init-from source records no {_k}; its holdout "
                      f"cannot be verified here", flush=True)
            elif sv != mv:
                raise SystemExit(f"--init-from source trained with {_k}={sv!r}, this "
                                 f"run has {mv!r} -- that would leak the holdout")
        # The holdout env vars are not the whole story: a source trained with
        # --real-trajs all under the same (empty) holdouts has still seen every
        # validation trajectory. Same subset rule as the distillation guard.
        _SUB = {"train": {"train"},
                "train+val_re": {"train", "train+val_re"},
                "train+test_aoa": {"train", "train+test_aoa"},
                "all": set(TRAIN_SETS)}
        sset = ("train" if ck["args"].get("mode") == "sim_only"
                else ck["args"].get("real_trajs", "train"))
        if sset not in _SUB.get(args.real_trajs, set()):
            raise SystemExit(
                f"--init-from source trained on --real-trajs {sset!r}, this run uses "
                f"{args.real_trajs!r}; its weights encode trajectories this run holds out")
        # ...and neither is the source's own metadata, if the source was itself
        # warm-started or distilled from something dirtier. Refuse an ancestry we
        # cannot see rather than trusting a clean-looking leaf.
        for _anc in ("init_from", "distill_from", "cooperate_with"):
            if ck["args"].get(_anc) and not args.distill_unsafe:
                raise SystemExit(
                    f"--init-from source was itself built with --{_anc.replace('_','-')}"
                    f"={ck['args'][_anc]!r}; its ancestry is not verified here. Check that "
                    f"chain by hand and pass --distill-unsafe, or start from a clean run.")
        model.load_state_dict(ck["model"])
        print(f"warm-started from {args.init_from}", flush=True)
    ema = None
    if args.ema:
        import copy
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    print(f"model: {count_params(model)/1e6:.2f}M params", flush=True)
    # arm D's loss, the current best configuration.
    crit = CompositeLoss(mse=args.w_mse, rel=args.w_rel,
                         tke=0 if args.cooperate_route == 'tke' else args.w_tke, spec=args.w_spec,
                         mvpe=args.w_mvpe, tmean=args.w_tmean)
    # A separate loss for simulated windows (2026-08-21). Simulation is a
    # regulariser worth +0.71 as-is, but it carries 1.4-2.1x the real fluctuation
    # energy -- so the hypothesis behind `--sim-terms nofluct` is that sim should
    # teach structure and transport (mse/rel/tmean) and NOT fluctuation statistics
    # (tke/spec), which only the real data measures correctly. `--w-sim-loss`
    # scales the whole simulated contribution. Defaults reproduce the shipped path
    # exactly (crit_sim is None and the single mixed-batch loss below is used).
    if args.mimo and (args.aux_p or args.w_unroll or args.distill_from
                      or args.aux_s2r or args.grad == "config" or args.in_stats
                      or args.sim_scramble != "none" or args.sim_noise > 0
                      or args.sim_shadow or args.w_sim_loss != 1.0
                      or args.sim_terms != "all" or args.cpu_data):
        raise SystemExit("--mimo composes only with the plain GPU loss path; "
                         "measure it alone first")
    if args.aux_p > 0 and (args.w_sps or args.w_sps_mean or args.w_lowf
                           or args.w_unroll or args.w_mcorr
                           or args.sim_scramble != "none" or args.sim_noise > 0
                           or args.sim_shadow):
        raise SystemExit("--aux-p takes the CPU-data branch, whose loss is built "
                         "separately and does NOT include the extra terms -- the "
                         "run would silently train without --w-sps and report "
                         "success. Screen aux-p against the plain recipe, or move "
                         "the extra terms into that branch first (2026-08-26)")
    if args.grad == "config" and (args.distill_from or args.aux_p or args.aux_s2r
                                  or args.w_unroll or args.w_lowf or args.w_mcorr or args.w_sps
                                  or args.w_sps_mean
                                  or args.w_sim_loss != 1.0 or args.sim_terms != "all"):
        raise SystemExit("--grad config composes only with the plain loss path; "
                         "combine it with other loss features after each is "
                         "measured alone")
    crit_sim = None
    if args.w_sim_loss != 1.0 or args.sim_terms != "all" or args.sim_loss_frac is not None:
        nf = args.sim_terms in ("nofluct", "mse")
        if args.sim_terms == "mse":
            # The user-proposed strong form (2026-08-27): simulation contributes
            # structure through plain MSE only -- no relative, amplitude, spectral
            # or mean terms to confuse (sim carries several times real's fluctuation
            # energy, so its tke/spec targets are the wrong physics anyway).
            crit_sim = CompositeLoss(mse=args.w_mse)
        else:
            crit_sim = CompositeLoss(mse=args.w_mse, rel=args.w_rel,
                                     tke=0.0 if nf else args.w_tke,
                                     spec=0.0 if nf else args.w_spec,
                                     mvpe=args.w_mvpe, tmean=args.w_tmean)
    if args.opt in ("muon", "nsgd", "soap", "soapmuon"):
        # Matrix-shaped parameters get the orthogonalized (or, under `nsgd`, the
        # magnitude-matched) update; everything 0/1-D (biases, norms, FiLM
        # embeddings) stays on AdamW at the recipe lr unless --muon-1d moves it.
        # One scheduler over a single optimizer keeps the cosine, the scaler and
        # the clip unchanged, so the comparison isolates the update rule.
        from realistix.muon import Muon, ScaledMomentum
        cut = 1 if args.muon_1d else 2
        _io = ("stem.", "out.")                 # the non-hidden layers, by name
        # H2: embedding weights by module type, not by shape -- a (2, 2C) FiLM table
        # is 2-D and would otherwise be orthogonalised like a conv kernel.
        _emb = {id(m.weight) for m in model.modules()
                if isinstance(m, torch.nn.Embedding)} if args.muon_embed_adamw else set()
        # H3: transposed convolutions store (C_in, C_out, k, k).
        _tr = {id(m.weight) for m in model.modules()
               if isinstance(m, torch.nn.ConvTranspose2d)} if args.muon_convt_fix else set()
        _stem = ({id(model.stem.weight)} if args.muon_stem_cov > 0
                 and isinstance(getattr(model, "stem", None), torch.nn.Conv2d) else set())
        if args.muon_stem_cov > 0 and not _stem:
            raise SystemExit("--muon-stem-cov needs the dense conv stem")
        mats = [q for n, q in model.named_parameters()
                if q.ndim >= cut and id(q) not in _emb and id(q) not in _tr
                and id(q) not in _stem
                and not (args.muon_hidden_only and n.startswith(_io))]
        mats_tr = [q for n, q in model.named_parameters() if id(q) in _tr]
        mats_stem = [q for n, q in model.named_parameters() if id(q) in _stem]
        emb_params = [q for n, q in model.named_parameters() if id(q) in _emb]
        rest = [q for n, q in model.named_parameters()
                if q.ndim < cut or id(q) in _emb
                or (args.muon_hidden_only and n.startswith(_io))]
        wd_muon = args.wd if args.wd_muon is None else args.wd_muon
        if (_emb or _tr or _stem or args.wd_muon is not None
                or args.muon_order != "momentum_first"):
            print(f"muon groups: {len(mats)} matrices"
                  + (f" + {len(mats_tr)} transposed" if mats_tr else "")
                  + (f" + {len(mats_stem)} stem-cov" if mats_stem else "")
                  + f"; {len(rest)} on AdamW"
                  + (f" ({len(_emb)} embeddings moved)" if _emb else "")
                  + f"; wd_muon {wd_muon}; order {args.muon_order}", flush=True)
        if args.opt == "muon":
            if args.muon_mean_lr_scale != 1:
                mean_weight = model.out_mean.weight
                assert any(q is mean_weight for q in mats)
                groups = [dict(params=[q for q in mats if q is not mean_weight]),
                          dict(params=[mean_weight], lr=args.lr_muon * args.muon_mean_lr_scale)]
            else:
                groups = [dict(params=mats)]
            if mats_tr:
                groups.append(dict(params=mats_tr, transpose=True))
            if mats_stem:
                groups.append(dict(params=mats_stem, P=None))
            opt = Muon(groups, lr=args.lr_muon, momentum=args.muon_momentum,
                       ns_steps=args.muon_ns_steps, weight_decay=wd_muon,
                       adavar=args.muon_adavar, order=args.muon_order,
                       nesterov=not args.muon_no_nesterov)
        elif args.opt in ("soap", "soapmuon"):
            # --lr-muon doubles as the matrix-optimizer lr for all non-Adam rules.
            from realistix.soap import SOAP
            opt = SOAP(mats, lr=args.lr_muon, weight_decay=args.wd,
                       muon=(args.opt == "soapmuon"),
                       ns_steps=args.muon_ns_steps)
        else:
            opt = ScaledMomentum(mats, lr=args.lr_muon,
                                 momentum=args.muon_momentum, weight_decay=args.wd)
        opt_rest = (torch.optim.AdamW(rest, lr=args.lr, weight_decay=args.wd,
                                     betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
                    if rest else None)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
        opt_rest = None
        emb_params, mats_stem = [], []
    stem_cov = None
    if args.muon_stem_cov > 0:
        stem_cov = dict(C=torch.zeros(40, 40, device=device), n=0)
    def _sched(o):
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(o, args.steps - args.warmup)
        if not args.warmup:
            return cos
        warm = torch.optim.lr_scheduler.LinearLR(
            o, start_factor=1.0 / args.warmup, total_iters=args.warmup)
        return torch.optim.lr_scheduler.SequentialLR(
            o, [warm, cos], milestones=[args.warmup])
    sched = _sched(opt)
    sched_rest = _sched(opt_rest) if opt_rest is not None else None

    # Precision follows the card rather than being hard-coded. This used to say
    # `torch.bfloat16` unconditionally, which raises on Turing and earlier -- so
    # training could only run on a100/h100/h200, the three pools that are usually
    # full, while a dozen rtx2080ti and titanxp cards sat idle. `evaluate.py`
    # already made inference adaptive after the same crash; this is the training
    # half of that fix. fp16 needs a loss scaler, bf16 does not.
    amp = amp_dtype_for(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp == torch.float16))
    print(f"amp: {str(amp).split('.')[-1]}"
          f"{' + GradScaler' if scaler.is_enabled() else ''}", flush=True)

    teachers = []
    if args.distill_from or args.cooperate_with:
        if args.cooperate_with:
            paths = [os.path.realpath(p) for p in args.cooperate_with.split(",")]
            if len(set(paths)) != len(paths):
                raise SystemExit("cooperative companions must be distinct checkpoints")
            if args.cooperate_drop_prob and len(paths) < 2:
                raise SystemExit("companion dropout requires at least two companions")
            args.cooperate_sha256 = {}
        for tp in (args.distill_from or args.cooperate_with).split(","):
            if args.cooperate_with:
                digest = hashlib.sha256()
                with open(tp, "rb") as source:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        digest.update(chunk)
                args.cooperate_sha256[os.path.realpath(tp)] = digest.hexdigest()
            tck = torch.load(tp, map_location=device)
            t = build_model(tck["args"]).to(device).eval()
            t.load_state_dict(tck["model"])
            for q in t.parameters():
                q.requires_grad_(False)
            teachers.append(t)
            ta = tck["args"]
            print(f"teacher: {tp} width={ta.get('width')} steps={ta.get('steps')} "
                  f"({count_params(t)/1e6:.2f}M)", flush=True)
            # Same ancestry rule as --init-from: a teacher built on top of another
            # checkpoint carries whatever that one saw.
            for _anc in ("init_from", "distill_from", "cooperate_with"):
                if ta.get(_anc) and not args.distill_unsafe:
                    raise SystemExit(
                        f"teacher {tp} was itself built with "
                        f"--{_anc.replace('_','-')}={ta[_anc]!r}; its ancestry is not "
                        f"verified here. Check that chain and pass --distill-unsafe.")
            # Guard against the leak described above: the teacher records the holdout
            # it trained under, and it has to match this run's. Distillation targets
            # carry the teacher's knowledge of whatever it saw, so *every* holdout
            # axis has to agree, not just the two regime ones -- checking AoA/Re only
            # let a teacher trained on all cells, or on the evaluation tail, supply
            # targets in a mimic-cell or tail-holdout run and call the result
            # held out.
            #
            # A teacher that predates the record WARNS rather than fails. An archive
            # scan on 2026-09-01 found 100 distillation students and zero real
            # leaks: every legacy teacher flagged for "no recorded holdout" turned
            # out to be the matching per-fold teacher (its run directory names the
            # fold). Failing on unverifiable provenance would therefore have broken
            # every archived distillation study to prevent nothing.
            for k in ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE",
                      "REALISTIX_HELDOUT_CELLS", "REALISTIX_HELDOUT_TAIL"):
                tv, mv = ta.get(k), os.environ.get(k)
                if tv is None:
                    if args.cooperate_with:
                        raise SystemExit(f"companion {tp} records no {k}; cannot verify holdout")
                    print(f"warning: teacher {tp} records no {k}; its holdout cannot "
                          f"be verified here", flush=True)
                    continue
                if (tv or "") != (mv or ""):
                    raise SystemExit(f"teacher trained with {k}={tv!r}, this run has "
                                     f"{mv!r} -- that would leak the holdout")
            # ...and the training set has to be a subset, not merely "not all".
            # Checking only for "all" let a `train+val_re` teacher supply targets to
            # a `train` student whose val_re is then reported as held out.
            _SUBSET = {"train": {"train"},
                       "train+val_re": {"train", "train+val_re"},
                       "train+test_aoa": {"train", "train+test_aoa"},
                       "all": set(TRAIN_SETS)}
            # `--mode sim_only` empties real_trajs regardless of the flag, so a
            # sim-pretraining teacher records real_trajs="all" while having seen no
            # real data. Treating that as "saw everything" would reject the entire
            # sequential-curriculum family (seq_s0..s2) for a leak that cannot exist.
            tset = "train" if ta.get("mode") == "sim_only" else ta.get("real_trajs", "train")
            if tset not in _SUBSET.get(args.real_trajs, set()):
                raise SystemExit(
                    f"teacher {tp} trained on --real-trajs {tset!r}, this run uses "
                    f"{args.real_trajs!r}; its targets encode trajectories this run "
                    f"holds out")
        if args.cooperate_with:
            print(f"{len(teachers)} frozen companion(s), own loss fraction={args.cooperate_own}, "
                  f"gradient scale={args.cooperate_grad_scale}", flush=True)
        else:
            print(f"{len(teachers)} teacher(s), w_distill={args.w_distill}"
                  + (" [teacher only]" if args.distill_only else ""), flush=True)

    pairs = None
    if args.aux_s2r > 0:
        # Only regimes whose real trajectory is in this run's training set: pairing
        # against a held-out real trajectory would be leakage. `trajs_for` is the
        # unfiltered list, so this used to reach past --regime-bag and --real-frac
        # and, through RegimePairs' full frame range, past the tail holdout as well.
        # Use the trajectories the sampler actually kept, and the same frame range.
        if gpu_ds is None:
            raise SystemExit("--aux-s2r needs the GPU sampler (it is skipped by "
                             "--cpu-data/--aux-p, which would silently drop the "
                             "trajectory filtering this pairing depends on)")
        pairs = RegimePairs(list(real_trajs), device,
                            frames="head" if HELDOUT_TAIL else "all")
        print(f"aux sim->real on {len(pairs.keys)} paired regimes", flush=True)

    shadow_bank = None
    if args.sim_shadow:
        # One valid-cell mask per training real trajectory, keyed by AoA, so a
        # simulated window is shadowed with geometry from its own angle. Built from
        # the run's own training set, never the holdout.
        #
        # "The run's own training set" means the list the sampler actually kept
        # (2026-09-01). `trajs_for(...)` is the unfiltered one, so this reached past
        # --regime-bag and --real-frac, and imported real geometry even under
        # --mode sim_only, which is supposed to see no real data at all.
        if gpu_ds is None:
            raise SystemExit("--sim-shadow needs the GPU sampler; the CPU path does "
                             "not expose the filtered trajectory list")
        from realistix.data import load_traj as _lt
        from realistix.normalize import valid_mask as _vm
        shadow_bank = {}
        for tr in real_trajs:
            f = _lt(tr.path("real"))[:20]
            mk = torch.from_numpy(_vm(f.transpose(0, 2, 3, 1))).float()
            shadow_bank.setdefault(int(tr.aoa), []).append(mk)
        if not shadow_bank:
            raise SystemExit("--sim-shadow has no real trajectories to build masks "
                             "from (--mode sim_only, or the filters removed them all)")
        shadow_bank = {a: torch.stack(v).to(device) for a, v in shadow_bank.items()}
        print("sim-shadow: mask bank " +
              ", ".join(f"AoA {a}: {len(v)}" for a, v in sorted(shadow_bank.items())),
              flush=True)

    # The run's own schedule, next to history.json and written before training so it
    # survives a kill. `kfold.score_of` needs it to tell "finished at 8000" from
    # "killed at 8000 of 16000"; without it the reporter assumed 8000 for everyone
    # and could read a truncated long arm as a complete result.
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"),
              indent=1, default=str)

    best, hist = -1.0, []
    it = iter(train_dl) if train_dl is not None else None
    # Dense raw-iterate averaging windows (rank-2 screen). Accumulated in fp32 from the
    # RAW weights after each optimizer step -- not from the EMA, which is what every
    # earlier "weight averaging" measurement in this repo actually averaged.
    level = None
    if args.level_prob > 0:
        if gpu_ds is None or gpu_ds.real is None:
            raise SystemExit("--level-prob needs the GPU sampler with real data")
        if args.mimo or args.anchor != "last" or HELDOUT_TAIL or args.phase_aug != 1 or args.w_unroll > 0:
            raise SystemExit("--level-prob is validated for the plain last-frame anchor without tail/phase/unroll")
        if not 0 < args.level_prob <= 1 or not 0 < args.level_beta <= 1:
            raise SystemExit("--level-prob in (0, 1], --level-beta in (0, 1]")
        with torch.no_grad():
            stable_real = {}
            for i, f in enumerate(gpu_ds.real.fields):
                key = tuple(int(round(v)) for v in gpu_ds.real.regime[i].tolist())
                stable_real[key] = (f != 0).all(dim=1).all(dim=0).to(torch.float32)
        level = dict(stable=stable_real, n=0, n_real=0, gen=torch.Generator(device=device))
        level["gen"].manual_seed(args.seed + 4000003)
        print(f"level: prob {args.level_prob}, beta U(-{args.level_beta},{args.level_beta}), "
              f"{len(stable_real)} real cells", flush=True)
    mix = None
    if args.mix_prob > 0:
        if gpu_ds is None or gpu_ds.sim is None or gpu_ds.real is None:
            raise SystemExit("--mix-prob needs the GPU sampler with both real and simulated data")
        if not (args.film and args.cond == "norm") or args.mimo or args.anchor != "last":
            raise SystemExit("--mix-prob is validated for --film --cond norm with the plain last-frame anchor")
        if not 0 < args.mix_prob <= 1 or not 0 < args.mix_alpha <= 1:
            raise SystemExit("--mix-prob in (0, 1], --mix-alpha in (0, 1]")
        if HELDOUT_TAIL or args.phase_aug != 1 or args.w_unroll > 0 or args.out_gain:
            raise SystemExit("--mix-prob is not validated with a tail holdout, --phase-aug, --w-unroll or --out-gain")
        # Per-cell time-mean fields of both modalities in trajectory-normalised units (the
        # same convention the morph uses); the shift applied is their difference, scaled.
        # `stable` marks cells observed (non-zero, both components) in EVERY frame of the
        # trajectory: the shift is confined to cells stable in both modalities, so it can
        # never turn a missing observation (an exact zero the scorer excludes) into a
        # fabricated target (Codex round-8 audit, 2026-09-10).
        def _cell_means(ds_):
            out, stable = {}, {}
            with torch.no_grad():
                for i, f in enumerate(ds_.fields):
                    key = tuple(int(round(v)) for v in ds_.regime[i].tolist())
                    valid = (f.abs() >= EPS).any(dim=1).any(dim=0)
                    sc = (f.pow(2).sum() / (f.shape[0] * valid.sum().clamp_min(1))).sqrt() + EPS
                    out[key] = (f / sc).mean(dim=0)
                    stable[key] = (f != 0).all(dim=1).all(dim=0)
            return out, stable
        (sim_m, sim_st), (real_m, real_st) = _cell_means(gpu_ds.sim), _cell_means(gpu_ds.real)
        mix = dict(delta={k: (sim_m[k] - real_m[k]) * (sim_st[k] & real_st[k]).to(sim_m[k].dtype)
                          for k in real_m if k in sim_m}, n=0, n_real=0,
                   gen=torch.Generator(device=device))
        _stab = [float((sim_st[k] & real_st[k]).float().mean()) for k in mix["delta"]]
        print(f"mix: stable-support fraction per cell median {sorted(_stab)[len(_stab)//2]:.3f}, "
              f"min {min(_stab):.3f}", flush=True)
        mix["gen"].manual_seed(args.seed + 3000003)
        print(f"mix: {len(mix['delta'])} paired cells, prob {args.mix_prob}, alpha U(0,{args.mix_alpha}), "
              f"cond {args.mix_cond}", flush=True)
    morph = None
    if args.morph_prob > 0:
        if gpu_ds is None or gpu_ds.sim is None or gpu_ds.real is None:
            raise SystemExit("--morph-prob needs the GPU sampler with both real and simulated data")
        # Per-cell time-mean of every simulated trajectory, in the same normalised units the
        # sampler hands the network (RMS speed over valid cells), keyed by (Re, AoA).
        sim_mean = {}
        with torch.no_grad():
            for i, f in enumerate(gpu_ds.sim.fields):
                re_, ao_ = (int(round(v)) for v in gpu_ds.sim.regime[i].tolist())
                valid = (f.abs() >= EPS).any(dim=1).any(dim=0)               # (H, W)
                sc = (f.pow(2).sum() / (f.shape[0] * valid.sum().clamp_min(1))).sqrt() + EPS
                sim_mean[(re_, ao_)] = (f / sc).mean(dim=0)                    # (2, H, W)
        targets = sorted(float(v) for v in args.morph_angles.split(",") if v)
        morph = dict(mean=sim_mean, targets=targets, n=0, n_real=0)
        print(f"morph: {len(sim_mean)} sim mean fields, prob {args.morph_prob}, targets {targets}", flush=True)
    raw_windows = [tuple(int(v) for v in w.split("-")) for w in args.raw_avg.split(",") if w]
    raw_acc = {w: [None, 0] for w in raw_windows}
    sam_bad = 0
    for step in range(1, args.steps + 1):
        if gpu_ds is not None:
            if args.mimo:
                x, y, wscale, tag, regime = sample_mimo(
                    gpu_ds, args.batch, args.mimo, args.mimo_rho)
            else:
                x, y, wscale, tag, regime = gpu_ds.sample(args.batch)
            if frame_mask_gen is not None:
                # Zero whole normalised input frames (both velocity channels) except the
                # anchor frame the delta is added to; the scale was computed before.
                b_, hh_, ww_ = x.shape[0], x.shape[-2], x.shape[-1]
                keep = (torch.rand(b_, T_IN - 1, 1, 1, 1, device=x.device,
                                   generator=frame_mask_gen) >= args.frame_mask)
                keep = torch.cat([keep, torch.ones(b_, 1, 1, 1, 1, device=x.device,
                                                   dtype=torch.bool)], dim=1)
                x = (x.reshape(b_, T_IN, -1, hh_, ww_) * keep).reshape(x.shape)
        else:
            try:
                x, y, wscale, tag = next(it)
            except StopIteration:
                it = iter(train_dl); x, y, wscale, tag = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # The GPU sampler returns a regime descriptor per window; the DataLoader
            # path has none, and the loss below reads both `wscale` and `regime`
            # unconditionally. Discarding the scale and leaving `regime` undefined
            # made --cpu-data raise UnboundLocalError on its first batch. Zeros are
            # the same descriptor `film_re` falls back to at submission time.
            wscale = wscale.to(device, non_blocking=True)
            regime = torch.zeros(x.shape[0], 2, device=device)
        if morph is not None:
            with torch.no_grad():
                is_real = (tag.to(device) == 0)
                pick = is_real & (torch.rand(x.shape[0], device=device) < args.morph_prob)
                morph["n_real"] += int(is_real.sum())
                for b_ in torch.nonzero(pick).flatten().tolist():
                    re_, a_ = (int(round(v)) for v in regime[b_].tolist())
                    cands = [t_ for t_ in morph["targets"] if 0 < abs(t_ - a_) <= 5]
                    if not cands:
                        continue
                    t_ = cands[int(torch.randint(len(cands), (1,)))]
                    nb = a_ + (5 if t_ > a_ else -5)
                    if (re_, a_) not in morph["mean"]:
                        continue
                    frac = abs(t_ - a_) / 5.0
                    if (re_, nb) in morph["mean"]:
                        delta = frac * (morph["mean"][(re_, nb)] - morph["mean"][(re_, a_)])   # (2, H, W)
                    elif args.morph_extrap and (re_, 2 * a_ - nb) in morph["mean"]:
                        # neighbour's simulation absent: continue the trend from the far side
                        delta = frac * (morph["mean"][(re_, a_)] - morph["mean"][(re_, 2 * a_ - nb)])
                        morph["n_extrap"] = morph.get("n_extrap", 0) + 1
                    else:
                        continue
                    xb = x[b_].reshape(-1, 2, *x.shape[-2:])                               # (T, 2, H, W) view
                    yb = y[b_].reshape(-1, 2, *y.shape[-2:])
                    valid = ((xb.abs() >= EPS).any(dim=1).any(dim=0)).to(x.dtype)           # (H, W)
                    xb.add_(delta[None] * valid)
                    yb.add_(delta[None] * valid)
                    morph["n"] += 1
                if step % args.eval_every == 0:
                    print(f"morph: {morph['n']} of {morph['n_real']} real windows morphed so far "
                          f"({morph.get('n_extrap', 0)} by extrapolation)", flush=True)
        if level is not None:
            with torch.no_grad():
                is_real = (tag.to(device) == 0)
                u = torch.rand(x.shape[0], device=device, generator=level["gen"])
                beta = (torch.rand(x.shape[0], device=device, generator=level["gen"]) * 2 - 1) * args.level_beta
                pick = is_real & (u < args.level_prob)
                level["n_real"] += int(is_real.sum())
                for b_ in torch.nonzero(pick).flatten().tolist():
                    key = tuple(int(round(v)) for v in regime[b_].tolist())
                    st = level["stable"].get(key)
                    if st is None:
                        continue
                    xb = x[b_].reshape(-1, 2, *x.shape[-2:])
                    yb = y[b_].reshape(-1, 2, *y.shape[-2:])
                    d = (float(beta[b_]) * xb.mean(dim=0))[None] * st   # beta * input temporal-mean field, stable cells
                    xb.add_(d); yb.add_(d)
                    level["n"] += 1
                if step % args.eval_every == 0:
                    print(f"level: {level['n']} of {level['n_real']} real windows level-shifted so far", flush=True)
        mix_alpha = None
        if mix is not None:
            with torch.no_grad():
                is_real = (tag.to(device) == 0)
                u = torch.rand(x.shape[0], device=device, generator=mix["gen"])
                a_all = torch.rand(x.shape[0], device=device, generator=mix["gen"]) * args.mix_alpha
                pick = is_real & (u < args.mix_prob)
                mix["n_real"] += int(is_real.sum())
                mix_alpha = tag.to(device).float()
                for b_ in torch.nonzero(pick).flatten().tolist():
                    key = tuple(int(round(v)) for v in regime[b_].tolist())
                    if key not in mix["delta"]:
                        continue
                    a_ = float(a_all[b_])
                    xb = x[b_].reshape(-1, 2, *x.shape[-2:])
                    yb = y[b_].reshape(-1, 2, *y.shape[-2:])
                    valid = ((xb.abs() >= EPS).any(dim=1).any(dim=0)).to(x.dtype)
                    d = (a_ * mix["delta"][key])[None] * valid
                    xb.add_(d); yb.add_(d)
                    if args.mix_cond == "alpha":
                        mix_alpha[b_] = a_
                    mix["n"] += 1
                if step % args.eval_every == 0:
                    print(f"mix: {mix['n']} of {mix['n_real']} real windows mixed so far", flush=True)
        if args.sim_scramble != "none":
            x, y = scramble_sim(x, y, tag.to(device), args.sim_scramble)
        if args.sim_noise > 0 or shadow_bank is not None:
            x, y = augment_sim(x, y, tag.to(device), args.sim_noise,
                               shadow_bank, regime)
        tag_d = tag.to(device, non_blocking=True) if args.film else None
        if mix_alpha is not None and args.film:
            tag_d = mix_alpha            # float: FiLM interpolates the modality rows
        cfg_groups = None
        if args.meta_alpha > 0:
            # First-order cross-angle meta-update (MLDG-style, rank-7 screen). Supported on
            # the shipped bf16 loss path only, so both losses below are exactly the recipe's.
            if (args.aux_p > 0 or args.mimo or args.w_unroll > 0 or teachers
                    or crit_sim is not None or args.w_sps_mean > 0 or args.w_mcorr
                    or pairs is not None or args.grad == "config" or args.sam_rho > 0
                    or gpu_ds is None or scaler.is_enabled()):
                raise SystemExit("--meta-alpha supports the shipped bf16 GPU-sampler loss path only")
            def _std_loss(xb, yb, wsb, tagb, regb):
                with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                    pb = model(xb, tagb, torch.log(wsb), regb)
                    lb, _ = crit(pb, yb)
                    if args.w_lowf > 0:
                        lb = lb + args.w_lowf * lowfreq_rel(pb, yb)
                    if args.w_sps > 0 and not (args.w_sps_until and step > args.w_sps_until):
                        lb = lb + args.w_sps * sps_fn(pb, yb, wsb, width=args.sps_width)
                return lb
            if step == 1:
                meta_angles = sorted({int(a) for a in gpu_ds.real.regime[:, 1].tolist()})
                if len(meta_angles) < 2:
                    raise SystemExit("--meta-alpha needs at least two training angles")
                print(f"meta: training angles {meta_angles}, "
                      f"{'random partition' if args.meta_random else 'query angle held out per step'}",
                      flush=True)
            half = max(args.batch // 2, 1)
            if args.meta_random:
                xa, ya, wa, ta, ra = gpu_ds.sample(half)
                xb, yb, wb, tb, rb = gpu_ds.sample(half)
            else:
                qb = meta_angles[int(torch.randint(len(meta_angles), (1,)))]
                xa, ya, wa, ta, ra = gpu_ds.sample(half, angles=[a for a in meta_angles if a != qb])
                xb, yb, wb, tb, rb = gpu_ds.sample(half, angles=[qb])
            ta_d = ta if args.film else None
            tb_d = tb if args.film else None
            opt.zero_grad(set_to_none=True)
            if opt_rest is not None:
                opt_rest.zero_grad(set_to_none=True)
            loss_a = _std_loss(xa, ya, wa, ta_d, ra)
            loss_a.backward()
            params = [q for q in model.parameters() if q.grad is not None]
            with torch.no_grad():
                g_a = [q.grad.detach().clone() for q in params]
                gn = torch.sqrt(sum((g.float() ** 2).sum() for g in g_a))
                pn = torch.sqrt(sum((q.float() ** 2).sum() for q in params))
                alpha = args.meta_alpha * pn / (gn + 1e-12)     # |alpha g_A| = meta_alpha |theta|
                for q, g in zip(params, g_a):
                    q.sub_(alpha * g)
            opt.zero_grad(set_to_none=True)
            if opt_rest is not None:
                opt_rest.zero_grad(set_to_none=True)
            loss_b = _std_loss(xb, yb, wb, tb_d, rb)
            loss_b.backward()
            with torch.no_grad():
                for q, g in zip(params, g_a):
                    q.add_(alpha * g)                      # restore theta exactly
                    if q.grad is None:
                        q.grad = torch.zeros_like(q)
                    q.grad.add_(g).mul_(0.5)               # (g_A + g_B(theta'))/2
            loss = 0.5 * (loss_a.detach() + loss_b.detach())
            if step == 1 or step % args.eval_every == 0:
                print(f"meta step {step}: loss_A {float(loss_a):.5f} loss_B(theta') {float(loss_b):.5f} "
                      f"alpha {float(alpha):.3e}", flush=True)
        else:
            if args.aux_p > 0:
                # x/y carry (u, v, p); the network only ever sees velocities.
                xv = x.reshape(x.shape[0], 20, 3, 32, 64)[:, :, :2].reshape(x.shape[0], 40, 32, 64)
                yv = y.reshape(y.shape[0], 20, 3, 32, 64)[:, :, :2].reshape(y.shape[0], 40, 32, 64)
                yp = y.reshape(y.shape[0], 20, 3, 32, 64)[:, :, 2]
                with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                    pred, p_pred = model(xv.contiguous(), tag_d)
                    loss, _ = crit(pred, yv.contiguous())
                    # Supervised on simulated windows only: real pressure does not exist,
                    # and its zeros are not a target to regress toward.
                    m = (tag.to(device) == 1).float()[:, None, None, None]
                    if m.sum() > 0:
                        loss = loss + args.aux_p * (((p_pred - yp) ** 2) * m).sum() / (
                            m.sum() * yp.shape[1] * yp.shape[2] * yp.shape[3])
            else:
                with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                    pred = model(x, tag_d,
                                 torch.log(wscale[:, 0] if args.mimo else wscale),
                                 regime)
                    own_pred = pred
                    if args.cooperate_with:
                        chosen = teachers
                        if args.cooperate_drop_prob and cooperate_rng.random() < args.cooperate_drop_prob:
                            omit = cooperate_rng.randrange(len(teachers))
                            chosen = [t for i, t in enumerate(teachers) if i != omit]
                        with torch.no_grad():
                            companions = [t(x, tag_d, torch.log(wscale), regime).float()
                                          for t in chosen]
                        ensemble_pred = torch.stack([pred.float(), *companions]).mean(0)
                        ensemble_pred = rlosses.cooperative_mean_gradient(
                            ensemble_pred, own_pred, len(chosen) + 1, args.cooperate_mean_grad_scale)
                        pred = own_pred if args.cooperate_route == 'tke' else ensemble_pred
                    if args.mimo:
                        # Fold groups into the batch axis: every loss term is a
                        # per-sample mean, so (B, M*T*C) -> (B*M, T*C) with per-group
                        # scales reproduces "M members, each on its own window".
                        b_, _, hh_, ww_ = pred.shape
                        tc_ = pred.shape[1] // args.mimo
                        pred = pred.reshape(b_ * args.mimo, tc_, hh_, ww_)
                        y = y.reshape(b_ * args.mimo, tc_, hh_, ww_)
                        wscale = wscale.reshape(-1)
                    if args.grad == "config":
                        # Group losses for the conflict-free combine at the backward
                        # site. Shrinkage terms in one group, amplitude terms in the
                        # other -- the axis whose gradients oppose on 58% of steps.
                        import torch.nn.functional as F
                        p1 = args.w_mse * F.mse_loss(pred, y) \
                            + args.w_rel * rlosses.rel_l2(pred, y) \
                            + (args.w_tmean * rlosses.time_mean(pred, y, 20, 2)
                               if args.w_tmean else 0.0) \
                            + (args.w_mvpe * rlosses.mvpe_rel_l2(pred, y, 20, 2)
                               if args.w_mvpe else 0.0)
                        p2 = (args.w_tke * rlosses.tke_rel_l2(pred, y, 20, 2)
                              if args.w_tke else 0.0) \
                            + (args.w_spec * rlosses.spectral(pred, y, 20, 2)
                               if args.w_spec else 0.0)
                        cfg_groups = (p1, p2)
                        loss = p1 + p2
                    elif crit_sim is not None:
                        # Every term in CompositeLoss is a per-sample mean, so splitting
                        # the batch by modality and recombining by fraction reproduces
                        # the mixed loss exactly at w_sim_loss=1/sim_terms=all.
                        is_sim = tag.to(device) == 1
                        if bool(is_sim.all()):
                            ls, _ = crit_sim(pred, y)
                            loss = args.w_sim_loss * ls
                        elif not bool(is_sim.any()):
                            loss, _ = crit(pred, y)
                        else:
                            lr, _ = crit(pred[~is_sim], y[~is_sim])
                            ls, _ = crit_sim(pred[is_sim], y[is_sim])
                            fr = is_sim.float().mean()
                            objective_fr = fr if args.sim_loss_frac is None else args.sim_loss_frac
                            loss = (1 - objective_fr) * lr + objective_fr * args.w_sim_loss * ls
                    else:
                        loss, _ = crit(pred, y)
                    if args.w_mcorr and getattr(model, "mcorr", None) is not None:
                        loss = loss + args.w_mcorr * model.mcorr.float().pow(2).mean()
                    if args.w_lowf > 0:
                        loss = loss + args.w_lowf * lowfreq_rel(pred, y)
                    if args.w_elead > 0:
                        loss = loss + args.w_elead * lead_energy(pred, y)
                    if args.w_dlead > 0:
                        loss = loss + args.w_dlead * diff_energy(pred, y)
                    if args.w_sps > 0 and not (args.w_sps_until and step > args.w_sps_until):
                        # wscale de-normalizes to the physical units SIGMA_GLOBAL is in;
                        # the same tensor the model already reads as log(wscale).
                        sample_weights = None
                        if args.sim_loss_frac is not None:
                            # Importance weights preserve the desired domain mixture
                            # in SPS's ELEMENT average, including different support masks.
                            sim = tag.to(device) == 1
                            fr = sim.float().mean()
                            sample_weights = torch.where(sim, args.sim_loss_frac / fr,
                                                         (1 - args.sim_loss_frac) / (1 - fr))
                        sps_kwargs = {}
                        wscale_sps = wscale
                        if args.sps_domain == "real" or args.sim_sps_scale != 1.0:
                            if sps_fn is not rlosses.sps_bulk:
                                raise SystemExit("--sps-domain / --sim-sps-scale need the bulk objective")
                            sim = tag.to(device) == 1
                            if args.sim_sps_scale != 1.0:
                                wscale_sps = torch.where(sim, wscale * args.sim_sps_scale, wscale)
                            if args.sps_domain == "real":
                                sample_weights = (~sim).float()
                                sps_kwargs["keep_denominator"] = True
                        loss = loss + args.w_sps * sps_fn(pred, y, wscale_sps,
                                                        width=args.sps_width,
                                                        sample_weights=sample_weights,
                                                        **sps_kwargs)
                    if args.w_sps_mean > 0:
                        loss = loss + args.w_sps_mean * sps_bulk_mean(
                            pred, y, wscale, width=args.sps_width)
                    if args.w_unroll > 0 and gpu_ds is not None:
                        y2 = gpu_ds.last_y2()
                        # Invalid cells (airfoil body, FOV wedge) are exact zeros in
                        # the data but not in the prediction; mask them before the
                        # prediction becomes an input, exactly as a stored window
                        # would arrive. The second window is renormalized by its own
                        # RMS over valid cells, mirroring inference, and its target
                        # is brought into the same units.
                        bsz = x.shape[0]
                        valid = (x.reshape(bsz, 20, 2, *x.shape[-2:]).abs()
                                 .amax(dim=(1, 2), keepdim=False) > 0)[:, None]
                        p2in = pred * valid
                        tot = p2in.pow(2).sum(dim=(1, 2, 3))
                        nvalid = valid.sum(dim=(1, 2, 3)).clamp_min(1)
                        s2 = (tot / (20 * nvalid)).sqrt() + 1e-8
                        s2 = s2[:, None, None, None]
                        pred2 = model(p2in / s2, tag_d,
                                      torch.log(wscale * s2.reshape(-1)), regime)
                        l2, _ = crit(pred2, y2 / s2)
                        loss = loss + args.w_unroll * l2
                    own_fraction = args.cooperate_own
                    if args.cooperate_with:
                        progress = (min(max((step - args.cooperate_start) / args.cooperate_ramp, 0), 1)
                                    if args.cooperate_ramp else float(step > args.cooperate_start))
                        own_fraction = 1 - progress * (1 - args.cooperate_own)
                    if args.cooperate_with and args.cooperate_route == 'tke':
                        loss = loss + args.w_tke * rlosses.cooperative_tke_loss(
                            ensemble_pred, own_pred, y, own_fraction, args.cooperate_tke_scale)
                    elif args.cooperate_with and own_fraction:
                        own_loss, _ = crit(own_pred, y)
                        if args.w_sps and not (args.w_sps_until and step > args.w_sps_until):
                            own_loss = own_loss + args.w_sps * sps_fn(
                                own_pred, y, wscale, width=args.sps_width)
                        loss = (1 - own_fraction) * loss + own_fraction * own_loss
                    if teachers and args.distill_from:
                        with torch.no_grad():
                            outs = []
                            for t in teachers:
                                o = t(x, tag_d, torch.log(wscale), regime)
                                outs.append(o[0] if isinstance(o, tuple) else o)
                            t_out = torch.stack(outs).mean(0)
                            if args.distill_gain > 0:
                                # Same maths as submissions/k4g `_matched_gain`: per (window, channel, cell) the
                                # mean's fluctuation about its time-mean is scaled by 1 + dose*(min(3, sqrt(V_members
                                # / v_mean)) - 1). The student then learns a full-amplitude target instead of the
                                # damped mean (CANDIDATE_distil1.md: the plain-mean student inherited the damping).
                                _o = torch.stack(outs).float(); _k, _b, _tc, _h, _w = _o.shape
                                _o = _o.reshape(_k, _b, 20, _tc // 20, _h, _w); _ym = t_out.float().reshape(_b, 20, _tc // 20, _h, _w)
                                _mu = _ym.mean(dim=1, keepdim=True); _vmean = ((_ym - _mu) ** 2).mean(dim=1, keepdim=True)
                                _vmem = ((_o - _o.mean(dim=2, keepdim=True)) ** 2).mean(dim=2, keepdim=True).mean(dim=0)
                                _g = 1.0 + args.distill_gain * (torch.sqrt(_vmem / _vmean.clamp_min(1e-12)).clamp(max=3.0) - 1.0)
                                t_out = (_mu + _g * (_ym - _mu)).reshape(_b, _tc, _h, _w).to(t_out.dtype)
                        # The same composite loss, so the student is pulled toward the
                        # teacher under the objective the scorer actually uses rather
                        # than under a plain MSE that would weight the wrong things.
                        d_loss, _ = crit(pred, t_out.detach())
                        loss = args.w_distill * d_loss + (0.0 if args.distill_only else loss)
            if pairs is not None:
                st = pairs.keys[int(torch.randint(len(pairs.keys), (1,)))]
                sw, rw = pairs.batch(st, 16)
                with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                    sim_tag = torch.ones(sw.shape[0], dtype=torch.long, device=device)
                    tr = model.translate(sw, sim_tag if args.film else None)
                loss = loss + args.aux_s2r * match_loss(stats(tr.float()), stats(rw.float()))
            opt.zero_grad(set_to_none=True)
            if args.zero_rest and opt_rest is not None:
                opt_rest.zero_grad(set_to_none=True)
            for q in emb_params:
                # Embeddings moved off Muon must still see per-step gradients, as they
                # did on Muon; the rest of the AdamW group keeps the shipped behaviour.
                q.grad = None
            if stem_cov is not None:
                with torch.no_grad():
                    z = x.detach().float().permute(1, 0, 2, 3).reshape(x.shape[1], -1)
                    stem_cov["C"].mul_(0.99).add_(z @ z.T, alpha=0.01 / z.shape[1])
                    stem_cov["n"] += 1
                    if stem_cov["n"] == 1 or stem_cov["n"] % 25 == 0:
                        C = stem_cov["C"] / (1 - 0.99 ** stem_cov["n"])
                        d = args.muon_stem_cov * torch.trace(C) / C.shape[0]
                        P = torch.linalg.inv(C + d * torch.eye(C.shape[0], device=device))
                        for grp in opt.param_groups:
                            if "P" in grp:
                                grp["P"] = P / (torch.trace(P) / P.shape[0])
            # The scaler has to be *used*, not merely constructed. It was created and
            # announced in the log for several hours while this block still called
            # loss.backward() directly, so every fp16 run trained with no loss scaling
            # at all -- which is precisely the bug `docs/archive/FINDINGS_2026-09-10.md` §9 records on the
            # inference side, where the state dict was cast to fp16 and load_state_dict
            # discarded it while every comment claimed half precision. Same shape of
            # error, other end of the pipeline.
            #
            # unscale_ before clipping is required, not optional: clip_grad_norm_ on
            # scaled gradients clips the wrong quantity. With bf16 the scaler is
            # disabled and all three calls pass straight through, so this is exactly
            # the original code on every card that has bfloat16.
            if cfg_groups is not None:
                # ConFIG, two-gradient closed form (arXiv:2408.11104). The update
                # direction is the normalized sum of the two groups' unit gradients
                # (equal positive inner product with both by construction), scaled by
                # the sum of the raw gradients' projections onto it. Both backwards
                # go through the scaler, so any fp16 scale factor cancels in the unit
                # vectors and passes linearly through the magnitude, and unscale_
                # below removes it exactly as in the summed path.
                params = [p for p in model.parameters() if p.requires_grad]
                scaler.scale(cfg_groups[0]).backward(retain_graph=True)
                g1 = [p.grad.detach().clone() if p.grad is not None else None
                      for p in params]
                for p in params:
                    if p.grad is not None:
                        p.grad = None
                scaler.scale(cfg_groups[1]).backward()
                dot = n1 = n2 = 0.0
                for a_, p in zip(g1, params):
                    if a_ is None or p.grad is None:
                        continue
                    b_ = p.grad
                    dot += float((a_ * b_).sum()); n1 += float((a_ * a_).sum())
                    n2 += float((b_ * b_).sum())
                n1, n2 = n1 ** 0.5 + 1e-12, n2 ** 0.5 + 1e-12
                c = dot / (n1 * n2)
                denom = max(2.0 + 2.0 * c, 1e-4) ** 0.5
                m = (n1 + n2 + dot / n2 + dot / n1) / denom
                for a_, p in zip(g1, params):
                    if p.grad is None:
                        if a_ is not None:
                            p.grad = a_ * (m / (n1 * denom))
                        continue
                    u = (a_ / n1 if a_ is not None else 0.0) + p.grad / n2
                    p.grad = u * (m / denom)
            else:
                backward_loss = loss * args.cooperate_grad_scale if args.cooperate_with else loss
                scaler.scale(backward_loss).backward()
            if args.sam_rho > 0:
                # Scale-aware SAM (ASAM-style metric), rank-3 screen. Supported on the shipped
                # loss path only, so the perturbed loss below is exactly the loss above.
                if (cfg_groups is not None or args.aux_p > 0 or args.mimo or args.w_unroll > 0
                        or teachers or crit_sim is not None or args.w_sps_mean > 0
                        or args.w_mcorr or pairs is not None or scaler.is_enabled()):
                    raise SystemExit("--sam-rho supports the shipped bf16 loss path only")
                params = [q for q in model.parameters() if q.grad is not None]
                with torch.no_grad():
                    if args.sam_random:
                        gen = torch.Generator(device=device)
                        gen.manual_seed(args.seed * 1_000_003 + step)
                        dirs = [torch.randn(q.shape, generator=gen, device=device,
                                            dtype=q.dtype) for q in params]
                    else:
                        dirs = [q.grad for q in params]
                    Ds = [q.abs() + 0.01 * q.float().pow(2).mean().sqrt().to(q.dtype) + 1e-8
                          for q in params]
                    nrm = torch.sqrt(sum(((D * d).float() ** 2).sum() for D, d in zip(Ds, dirs)))
                    eps_ = [(args.sam_rho * D * D * d / (nrm + 1e-12)).to(q.dtype)
                            for q, D, d in zip(params, Ds, dirs)]
                    for q, e_ in zip(params, eps_):
                        q.add_(e_)
                opt.zero_grad(set_to_none=True)
                if opt_rest is not None:
                    opt_rest.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                    pred_s = model(x, tag_d, torch.log(wscale), regime)
                    loss_s, _ = crit(pred_s, y)
                    if args.w_lowf > 0:
                        loss_s = loss_s + args.w_lowf * lowfreq_rel(pred_s, y)
                    if args.w_sps > 0 and not (args.w_sps_until and step > args.w_sps_until):
                        loss_s = loss_s + args.w_sps * sps_fn(pred_s, y, wscale,
                                                                width=args.sps_width)
                loss_s.backward()
                with torch.no_grad():
                    for q, e_ in zip(params, eps_):
                        q.sub_(e_)
                ls_, l0_ = float(loss_s), float(loss)
                if not (ls_ == ls_ and abs(ls_) < float("inf")):
                    raise SystemExit(f"--sam-rho: non-finite perturbed loss at step {step}")
                sam_bad = sam_bad + 1 if ls_ > 1.2 * l0_ else 0
                if sam_bad >= 200:
                    raise SystemExit(f"--sam-rho: perturbed loss > 1.2x for 200 consecutive steps "
                                     f"(last {ls_:.4f} vs {l0_:.4f}); not shrinking the radius silently")
                if step == 1 or step % args.eval_every == 0:
                    print(f"sam[{'random' if args.sam_random else 'adv'}] step {step}: loss "
                          f"{l0_:.5f} -> perturbed {ls_:.5f} (|eps| {float(nrm):.3e})", flush=True)
        scaler.unscale_(opt)
        if opt_rest is not None:
            scaler.unscale_(opt_rest)
        if args.clip_norm:
            if args.clip_mode == "separate" and opt_rest is not None:
                for optimizer in (opt, opt_rest):
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], args.clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        scaler.step(opt)
        if opt_rest is not None:
            scaler.step(opt_rest)
        scaler.update()
        sched.step()
        if sched_rest is not None:
            sched_rest.step()
        if ema is not None:
            with torch.no_grad():
                for a, b_ in zip(ema.parameters(), model.parameters()):
                    a.mul_(args.ema).add_(b_.detach(), alpha=1 - args.ema)
                for a, b_ in zip(ema.buffers(), model.buffers()):
                    a.copy_(b_)
                if args.switch_ema and step % args.switch_ema == 0:
                    # H7 Switch-EMA: the online weights restart from the average;
                    # momentum buffers and the schedule are untouched.
                    for a, b_ in zip(ema.parameters(), model.parameters()):
                        b_.copy_(a)
        for (w0, w1), acc in raw_acc.items():
            if w0 <= step <= w1:
                with torch.no_grad():
                    sd = {k: v.detach().float() for k, v in model.state_dict().items()
                          if v.dtype.is_floating_point}
                    if acc[0] is None:
                        acc[0] = {k: v.clone() for k, v in sd.items()}
                    else:
                        for k, v in sd.items():
                            acc[0][k].add_(v)
                    acc[1] += 1
                if step == w1:
                    avg = {k: (v / acc[1]) for k, v in acc[0].items()}
                    full = model.state_dict()
                    out_sd = {k: (avg[k].to(full[k].dtype) if k in avg else full[k])
                              for k in full}
                    torch.save({"model": out_sd, "args": vars(args), "step": step,
                                "raw_avg": [w0, w1], "n_avg": acc[1]},
                               os.path.join(args.out, f"rawavg_{w0}_{w1}.pt"))
                    print(f"raw-avg window {w0}-{w1}: {acc[1]} iterates saved", flush=True)
                    acc[0] = None

        if step % args.eval_every == 0:
            sc = {}
            scored_model = ema if ema is not None else model
            evaluated_model = (CooperativeForecast(scored_model, teachers)
                               if args.cooperate_with else scored_model)
            for n, dl in val.items():
                p, t = predict_all(evaluated_model, dl, device)
                sc[n] = subscores(p, t)
            # Selection here uses val_aoa only (val_re on an Re-only fold): it is the
            # split whose subscores matched the leaderboard to within a point.
            # This checkpoint rule never selects on test_aoa -- but that is not the
            # same as test_aoa being untouched, and the stronger claim used to be
            # made here. `kfold.fold_score` pools test_aoa with val_aoa into the
            # fold score it takes a max over, and `train_final_head.py` scans every
            # width-inflation factor on it. Treat it as a second validation split.
            acc = np.mean([sc[sel][k] for k in ("rel_l2_score", "tke_score",
                                                "mvpe_score", "sps_score")])
            print(f"step {step:>6} loss {float(loss):.5f}", flush=True)
            for n in sc:
                print(f"    {n:<8} {fmt(sc[n])}", flush=True)
            row = {"step": step, "mode": args.mode,
                   "metric_object": "cooperative_ensemble" if args.cooperate_with else "member",
                   **{f"{n}.{k}": v for n, s in sc.items() for k, v in s.items()}}
            if args.time_dilate > 0 and gpu_ds is not None:
                for dom in ("real", "sim"):
                    ds_ = getattr(gpu_ds, dom, None)
                    if ds_ is not None and ds_.dilate_prob > 0:
                        row.update({f"dilate.{dom}.{k}": v for k, v in ds_.dilate_stats.items()})
                        if ds_.dilate_traj_counts is not None:
                            row[f"dilate.{dom}.traj_counts"] = ds_.dilate_traj_counts.tolist()
                        print(f"    dilation[{dom}] {ds_.dilate_stats}", flush=True)
            hist.append(row)
            json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
            # With --real-trajs all there is nothing left to select on -- every
            # validation trajectory is in the training set, so `acc` above is an
            # in-sample number and picking the argmax of it would be selecting on
            # the training set. The step is fixed in advance instead, from runs
            # that did hold data out: val_aoa peaks at 4000 of the 8000-step
            # cosine for every seed, and the learning curve shows that peak barely
            # moves between 25% and 100% of the trajectories.
            take = (step == (args.fixed_step or args.steps)
                    if args.real_trajs == "all" else acc > best)
            ck = {"model": scored_model.state_dict(), "args": vars(args), "step": step}
            if take:
                best = acc
                torch.save(ck, os.path.join(args.out, "best.pt"))
            if args.real_trajs == "all" or args.snapshots:
                # Keep every snapshot as well. The fixed step is a guess transferred
                # from a run that could validate, and that guess has already moved
                # once: with AoA=20 in the training set the held-out optimum shifts
                # from 4000 to ~6000. Snapshots are cheap next to re-training.
                torch.save(ck, os.path.join(args.out, f"step{step}.pt"))
        if args.stop_at and step >= args.stop_at:
            print(f"stop-at {args.stop_at}: schedule of {args.steps} left intact, training ends here",
                  flush=True)
            break
    # The summary reports whichever split this fold actually selected on. It used to
    # index "val_aoa.mvpe_score" unconditionally, which is fine for every angle fold
    # and raises on an Re-only one -- *after* the training loop, so all eight evals,
    # history.json and every checkpoint were already on disk and the results were
    # intact. It still exited non-zero on ~60 completed runs, which is
    # indistinguishable at a glance from a study that died.
    key = f"{sel}.mvpe_score"
    bm = max((r[key] for r in hist if key in r), default=float("nan"))
    print(f"done: mode={args.mode} best={best:.3f} best_{sel}_mvpe={bm:.3f}", flush=True)


if __name__ == "__main__":
    main()
