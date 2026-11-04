#!/usr/bin/env python3
"""Train the SPS interval head on top of the frozen forecaster ensemble.

The head must be fitted to the ensemble it will ship with, not transplanted from
an earlier model: it learns the error distribution it is drawing intervals around,
and averaging seeds changes that distribution.

Reports the ``test_aoa`` split alongside and never selects the checkpoint on it --
but it is not untouched, and this line said so until 2026-09-01: the width-inflation
scan at the end of this file reports every lambda on it, `kfold.fold_score` pools it
with val_aoa, and the shipped lambda was read off it. Treat it as a second
validation split.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starting_kit"))
import scoring as S  # noqa: E402
from realistix.data import HELDOUT_TAIL, WindowDataset, build  # noqa: E402
from realistix.evaluate import amp_dtype_for, subscores, to_ntchw  # noqa: E402
from realistix.gpudata import GPUWindows  # noqa: E402
from realistix.splits import partition  # noqa: E402
from realistix.models import build_model, count_params  # noqa: E402
from realistix.models.uncertainty import UncertaintyHead, sps_reward  # noqa: E402
from realistix.losses import sps_prefactors  # noqa: E402


class Ensemble(torch.nn.Module):
    """Mean of independently seeded forecasters, always on the real modality token.

    ``extra_paths`` members are held out of the mean and contribute only to the
    spread the interval head reads. That split exists because the two jobs want
    different things from a member: the mean wants accuracy, and the spread wants
    *decorrelated* error. A structurally different model can be too weak to average
    in and still be the best disagreement signal available, since three U-Nets
    trained the same way agree even where they are all wrong.
    """

    def __init__(self, paths, device, extra_paths=(), member_precision='auto'):
        super().__init__()
        self.member_precision=member_precision
        self.models = torch.nn.ModuleList()
        for p in list(paths) + list(extra_paths):
            ck = torch.load(p, map_location=device)
            m = build_model(ck["args"])
            m.load_state_dict(ck["model"])
            self.models.append(m.to(device).eval())
        self.n_pred = len(list(paths))
        self.combiner = None
        for q in self.parameters():
            q.requires_grad_(False)

    def members(self, x):
        if self.member_precision=='float32':
            with torch.autocast(x.device.type,enabled=False):
                return self._members(x.float())
        return self._members(x)

    def _members(self, x):
        outs = []
        for m in self.models:
            o = m(x) if not m.n_modalities else m(
                x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            outs.append(o[0] if isinstance(o, tuple) else o)
        return torch.stack(outs)

    def forward(self, x):
        return self.mean_std(x)[0]

    def mean_std(self, x):
        o = self.members(x).float()
        # Unbiased over a handful of members, so the spread of a 3-4 member
        # ensemble is not systematically under-reported.
        pred = (o[:self.n_pred].mean(0) if self.combiner is None
                else self.combiner(o[:self.n_pred], x).float())
        return pred, o.std(0, unbiased=True)

    def feat_spread(self, x):
        """Per-element std across members of the decoder feature map.

        (B, width, H, W). Fed as log-magnitude: the raw scale is arbitrary and
        spans orders of magnitude between quiescent cells and the wake, exactly as
        the output spread does.
        """
        fs = []
        for m in self.models[:self.n_pred]:
            mod = (torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                   if m.n_modalities else None)
            fs.append(m(x, mod, want=2) if m.n_modalities else m(x, want=2))
        f = torch.stack(fs).float()
        return torch.log(f.std(0, unbiased=True).clamp_min(1e-6))

    def mean_stats(self, x):
        """(pred, std, min-dev, max-dev): the asymmetric disagreement statistics.

        The head emits *asymmetric* bounds but has only ever read the symmetric
        std. The signed extremes of the members' deviation from the mean are the
        natural asymmetric inputs (2026-08-23 head-enrichment arm).
        """
        o = self.members(x).float()
        pred = (o[:self.n_pred].mean(0) if self.combiner is None
                else self.combiner(o[:self.n_pred], x).float())
        mu = o.mean(0)
        return (pred, o.std(0, unbiased=True),
                (o.min(0).values - mu), (o.max(0).values - mu))

    def attach_combiner(self, comb):
        """Predict through a learned combiner instead of the plain mean.

        The head has to be fitted to the ensemble it ships with -- it learns the
        error distribution it draws intervals around, and the combiner changes that
        distribution by design, raising fluctuation energy where the mean cancelled
        it. Transplanting the mean's head onto the combiner would be the same
        mistake as transplanting a head across ensembles, which this file exists to
        avoid. The spread the head reads stays the *members'* disagreement, which
        the combiner does not alter.
        """
        self.combiner = comb
        for q in comb.parameters():
            q.requires_grad_(False)


def spread_channels(std_norm, scale):
    """Ensemble spread as head input, in the units the widths are paid in.

    The head emits absolute half-widths against ``SIGMA_GLOBAL``; the spread has
    to arrive on the same footing, so it is de-normalized to physical units and
    divided by sigma. Both the ratio and its log go in: the ratio spans two orders
    of magnitude between quiescent and wake cells, and the head has to be able to
    read the small end as easily as the large one.
    """
    r = std_norm * scale[:, None, None, None] / S.SIGMA_GLOBAL
    return torch.cat([r, torch.log(r.clamp_min(1e-4))], dim=1)


def spread_channels_mm(std_norm, dmin, dmax, scale):
    """Spread plus the signed member extremes, for asymmetric bounds."""
    sc = scale[:, None, None, None] / S.SIGMA_GLOBAL
    r = std_norm * sc
    return torch.cat([r, torch.log(r.clamp_min(1e-4)), dmin * sc, dmax * sc], dim=1)


@torch.no_grad()
def full_scores(ens, head, dl, device, const_frac=None, lam=1.0,
                feat_spread=False):
    """``lam`` inflates both half-widths.

    The head is fitted on the forecaster's *own training* windows, so the errors it
    calibrates against are smaller than the ones it will meet. That shows up in the
    numbers: val_aoa SPS 42.8 locally against 39.5 on the leaderboard, with local
    coverage already the binding constraint. One multiplicative factor, chosen on
    held-out data, is the cheapest correction for that gap.

    It is a correction, not a fix (noted 2026-09-01). A single scalar rescales the
    whole width distribution; it cannot repair a *conditional* width function learned
    from in-sample residuals. The failure mode it leaves open: members that memorized
    the calibration trajectories disagree little on them, so the head learns "low
    spread implies low error" and stays overconfident exactly where that inference
    breaks -- unseen regimes. The principled version is to fit the head on
    out-of-fold predictions (train forecasters without each calibration fold, predict
    that fold, pool the residuals) and keep a separate split for choosing lam.
    Untested as of this date; the shipped head still uses the in-sample fit.
    """
    preds, tgts, los, his = [], [], [], []
    for x, y, scale in dl:
        x, y = x.to(device), y.to(device)
        sc = scale.to(device)[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device),
                            enabled=device == "cuda"):
            p, sd = ens.mean_std(x)
        p = p.float()
        if const_frac is not None:
            lo_d = hi_d = torch.full_like(p, const_frac * S.SIGMA_GLOBAL)
        else:
            if feat_spread:
                extra = torch.cat([spread_channels(sd, scale.to(device)),
                                   ens.feat_spread(x)], dim=1)
            elif head.extra_channels > 80:
                _, sd2, dmin, dmax = ens.mean_stats(x)
                extra = spread_channels_mm(sd2, dmin, dmax, scale.to(device))
            else:
                extra = (spread_channels(sd, scale.to(device))
                         if head.extra_channels else None)
            lo_d, hi_d = head(x, p, torch.log(scale.to(device)), extra)
            lo_d, hi_d = lo_d * lam, hi_d * lam
        pp = p * sc
        preds.append(to_ntchw(pp).cpu().numpy())
        tgts.append(to_ntchw(y * sc).cpu().numpy())
        los.append(to_ntchw(pp - lo_d).cpu().numpy())
        his.append(to_ntchw(pp + hi_d).cpu().numpy())
    p = np.concatenate(preds).astype(np.float32)
    t = np.concatenate(tgts).astype(np.float32)
    lo = np.concatenate(los).astype(np.float32)
    hi = np.concatenate(his).astype(np.float32)
    out = subscores(p, t)
    c = S.measured_channels(t)
    v, cov = S.aggregate_sps(p, t, c, lo[..., :c], hi[..., :c])
    out["sps_score"] = S.score_sps(v)
    out["_coverage"] = cov
    return out


def build_parser() -> argparse.ArgumentParser:
    """The trainer's own flags, exposed so tools/check_recipe.py can
    parse tools/train_final.sh through the same definitions the trainer
    uses -- a recipe guard that reimplemented them would drift too.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/final_s*")
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--spread-runs", default=None,
                    help="glob of members used only for the spread, not the mean")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--opt", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--tau-frac", type=float, default=0.15,
                    help="sharpness of the differentiable coverage relaxation in "
                         "sps_reward. Set once in 2026-08-early and never swept -- "
                         "the same class of constant as the head step that paid "
                         "+0.414 board sps when finally re-selected (896716)")
    ap.add_argument("--score-prefactors", action="store_true",
                    help="weight the relaxed interval reward by all three official "
                         "per-window accuracy factors (frozen-member training statistics)")
    ap.add_argument("--oof-members", default=None,
                    help="comma-separated ordinary member checkpoints excluding both "
                         "calibration cells and the outer evaluation trajectories")
    ap.add_argument("--oof-cells", default=None,
                    help="comma-separated training Re_AoA cells whose prediction/spread "
                         "are replaced by the nested out-of-fold ensemble during head fitting")
    ap.add_argument("--oof-spread", choices=("inner", "outer"), default="inner",
                    help="use nested or deployed-ensemble spread on OOF calibration windows; "
                         "outer retains the ordinary feature while residuals stay held out")
    ap.add_argument("--member-precision",choices=("auto","float32"),default="auto",
                    help="precision of frozen member predictions during head fitting and evaluation")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--combiner", default=None,
                    help="best.pt from train_combiner.py; the head then draws its "
                         "intervals around the combined prediction")
    ap.add_argument("--spread", action="store_true",
                    help="feed the ensemble's own per-element disagreement to the head")
    ap.add_argument("--no-spread", action="store_true",
                    help="turn --spread back off; the last word for single-member "
                         "builds whose base flags carry --spread")
    ap.add_argument("--feat-spread", action="store_true",
                    help="additionally feed the members' DECODER-FEATURE "
                         "disagreement (width channels, log scale). Free at "
                         "inference -- the features are computed by passes we "
                         "already pay for -- and the only candidate signal that is "
                         "not a function of (window, prediction, output-spread), "
                         "which the K ladder showed the head already saturates")
    ap.add_argument("--spread-stats", choices=("std", "minmax"), default="std",
                    help="'minmax' adds the signed member extremes to the spread "
                         "channels -- the natural asymmetric inputs for a head "
                         "that emits asymmetric bounds")
    ap.add_argument("--offset", action="store_true",
                    help="let the head emit a signed centre offset as well: the "
                         "interval need not contain the prediction (the scorer "
                         "never checks it), the width charge is unchanged, and 34%% "
                         "of the error is predictable per-cell bias (2026-08-25)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-split", default="train",
                    help="which real split the head is fitted on")
    ap.add_argument("--fixed-step", type=int, default=0,
                    help="keep this step instead of the val_aoa argmax. Required when "
                         "the forecasters trained on every trajectory: val_aoa is "
                         "then in-sample, the SPS curve keeps rising as the widths "
                         "narrow past what held-out data would pay for, and selecting "
                         "on it would ship intervals calibrated to errors the model "
                         "will not make. Held-out runs put the optimum at ~1000.")
    ap.add_argument("--out", required=True)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if bool(args.oof_members) != bool(args.oof_cells):
        ap.error('--oof-members and --oof-cells must be supplied together')
    if args.oof_spread != 'inner' and not args.oof_members:
        ap.error('--oof-spread requires --oof-members')
    if args.member_precision=='float32' and args.feat_spread:
        ap.error('float32 member precision currently supports ordinary output spread only')
    if args.oof_members and (not args.spread or args.no_spread or args.feat_spread
                            or args.spread_stats != 'std' or args.combiner or args.spread_runs):
        ap.error('OOF calibration currently requires ordinary mean/std spread heads')
    if args.no_spread:
        args.spread = False
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.runs, "best.pt")))[:args.n_models]
    if not paths:
        raise SystemExit(f"no checkpoints under {args.runs}")
    # Every path must actually load. A glob happily matches a dangling symlink, and
    # an ensemble assembled from run directories is usually symlinks -- so a member
    # whose training died leaves a link that globs fine and trains a head on fewer
    # models than asked for, reporting success.
    for q in paths:
        if not os.path.exists(os.path.realpath(q)):
            raise SystemExit(f"{q} does not resolve to a file (member training died?)")
    if len(paths) < args.n_models:
        raise SystemExit(f"asked for {args.n_models} members, found {len(paths)}")
    # A single member has nothing to disagree with, so `o.std(0, unbiased=True)` over
    # one sample is NaN, not zero. Without --spread the head never reads it and the
    # run is fine (torch only warns); with --spread every input channel would be NaN
    # and the head would train on garbage while reporting a number. Refuse instead --
    # the same class of failure as the fp16 cast that load_state_dict discarded and
    # the GradScaler that nothing called. Found while building the 893273 back-test.
    if args.spread and len(paths) + len(
            glob.glob(os.path.join(args.spread_runs, "best.pt")) if args.spread_runs
            else []) < 2:
        raise SystemExit("--spread needs at least two members: the spread over one "
                         "member is NaN, not zero, and the head would train on it")
    extra_paths = (sorted(glob.glob(os.path.join(args.spread_runs, "best.pt")))
                   if args.spread_runs else [])
    # Pin the members this head is fitted on by content, not path: build_ckpt hashes
    # the members at packing time, and a checkpoint replaced between fitting and
    # packing keeps its path. Recorded in best.pt; build_ckpt and check_recipe compare.
    import hashlib

    def _sha256(q):
        h = hashlib.sha256()
        with open(os.path.realpath(q), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    member_sha256 = [_sha256(q) for q in paths]
    ens = Ensemble(paths, device, extra_paths, args.member_precision)
    if args.combiner:
        from realistix.models.combiner import Combiner
        cck = torch.load(args.combiner, map_location=device)
        ca = cck["args"]
        comb = Combiner(n_members=ca["n_models"], width=ca["width"],
                        depth=ca["depth"], spread=not ca["no_spread"],
                        window=not ca["no_window"]).to(device).eval()
        comb.load_state_dict(cck["combiner"])
        ens.attach_combiner(comb)
        print(f"combiner from {args.combiner} (step {cck['step']})", flush=True)

    def names(ps):
        return ", ".join(os.path.basename(os.path.dirname(os.path.realpath(q)))
                         for q in ps)
    print(f"ensemble of {len(paths)}: {names(paths)}"
          + (f"  | spread-only: {names(extra_paths)}" if extra_paths else ""), flush=True)
    print(f"{count_params(ens)/1e6:.2f}M params total "
          f"({count_params(ens)*4/1e6:.0f} MB fp32)", flush=True)

    # Same GPU-resident sampler the forecaster uses. Through the CPU DataLoader this
    # script pinned ~23 cores and starved every concurrent run; the windows it draws
    # are identical (train split, stride 1, per-window RMS normalization).
    # The members this postprocessor sits on top of have to have honoured the same
    # holdout it is about to be evaluated under. Sampling correctly here means
    # nothing if the forecasters underneath already saw the evaluation data, and
    # both scripts accepted arbitrary checkpoints until 2026-09-01.
    from check_head_frames import members_match_holdout
    _bad, _warn = members_match_holdout(
        paths, strict=bool(os.environ.get("REALISTIX_STRICT_PROVENANCE")))
    for _w in _warn:
        print(f"warning: {_w}", flush=True)
    if _bad and not os.environ.get("REALISTIX_ALLOW_MIXED_HOLDOUT"):
        raise SystemExit("member holdouts disagree with this run:\n  "
                         + "\n  ".join(_bad)
                         + "\nSet REALISTIX_ALLOW_MIXED_HOLDOUT=1 if the result will "
                           "not be quoted as held out.")

    p = partition("real")
    head_trajs = (p["train"] + p["val_re"] + p["val_aoa"] + p["test_aoa"]
                  if args.train_split == "all" else p[args.train_split])
    oof_ens = None
    oof_seen = oof_total = 0
    if args.oof_members:
        from realistix.calibration import validate_oof_members, calibration_mask
        from realistix.splits import available
        oof_paths = [os.path.realpath(q) for q in args.oof_members.split(',')]
        oof_stems = args.oof_cells.split(',')
        args.oof_sha256 = validate_oof_members(oof_paths, oof_stems,
            [t.stem for t in head_trajs], [t.stem for t in available('real')])
        oof_cells = torch.tensor([[int(v) for v in c.split('_')] for c in oof_stems],
                                 dtype=torch.float32, device=device)
        # Loading extra initialized modules must not alter the paired head's
        # initial parameters or subsequent GPU window draws.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
            oof_ens = Ensemble(oof_paths, device, member_precision=args.member_precision)
        print(f'OOF calibration: {len(oof_stems)}/{len(head_trajs)} training cells, '
              f'{len(oof_paths)} nested members; inference/evaluation uses the outer ensemble', flush=True)
    # Under the tail holdout the head has to obey it too. This defaulted to
    # frames="all" until 2026-09-01, so every "honest" pipeline trained the head on
    # the same tail frames `val_ingrid` then scored it on: the forecasters held the
    # tail out, the head did not, and the composite gate that reads sps_score was
    # therefore measured partly in-sample. `train_mixed.py` has always passed this;
    # the head was the one sampler that did not.
    head_frames = "head" if HELDOUT_TAIL else "all"
    train_ds = GPUWindows(head_trajs, "real", device, frames=head_frames)
    # Stamp the sampling policy into the checkpoint. Consumers need to tell a head
    # that honoured the tail holdout from one that did not, and the first attempt at
    # that (comparing the file's mtime against this script's git timestamp) was both
    # bypassable by `touch` and wrong for any head built between the fix and the
    # next unrelated edit to this file. A recorded fact beats a filesystem guess.
    vars(args)["head_frames"] = head_frames
    vars(args)["REALISTIX_HELDOUT_TAIL"] = os.environ.get("REALISTIX_HELDOUT_TAIL") or ""
    val = {}
    for n in ("val_aoa", "test_aoa", "val_re"):
        ds = build("real", n, stride=40)
        if len(ds):
            val[n] = DataLoader(ds, batch_size=64)
    if HELDOUT_TAIL:
        # The honest in-grid split: all regimes in training, evaluation on the
        # withheld tail of every trajectory. Under it the regime splits above are
        # all empty, and an empty split has to be absent rather than scored as
        # nothing -- `full_scores` concatenates its batches and raises on zero.
        from realistix.splits import available as _avail
        val["val_ingrid"] = DataLoader(
            WindowDataset(_avail("real"), "real", stride=40, frames="tail"),
            batch_size=64)
    if not val:
        raise SystemExit("no non-empty validation split; check the holdout env vars")
    sel = ("val_ingrid" if "val_ingrid" in val
           else "val_aoa" if "val_aoa" in val else "val_re")

    n_extra = 0
    if args.spread:
        n_extra = 160 if args.spread_stats == "minmax" else 80
    if args.feat_spread:
        n_extra += ens.models[0].out.in_channels
    head = UncertaintyHead(width=args.width, extra_channels=n_extra,
                           offset=args.offset).to(device)
    print("\n--- constant band 0.18*sigma (the shipped baseline) ---", flush=True)
    for n, dl in val.items():
        s = full_scores(ens, head, dl, device, const_frac=0.18)
        print(f"  {n:<9} sps={s['sps_score']:.2f}  mean4="
              f"{np.mean([s[k] for k in ('rel_l2_score','tke_score','mvpe_score','sps_score')]):.2f}",
              flush=True)

    if args.opt == "muon":
        # Same split as the forecaster's Muon path (2026-08-23): matrix-shaped
        # parameters on orthogonalized momentum, the rest on AdamW at --lr.
        from realistix.muon import Muon
        mats = [p for p in head.parameters() if p.ndim >= 2]
        rest = [p for p in head.parameters() if p.ndim < 2]
        opt = Muon(mats, lr=args.lr_muon, momentum=0.95)
        opt_rest = torch.optim.AdamW(rest, lr=args.lr, weight_decay=1e-5)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
        opt_rest = None
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    sch_rest = (torch.optim.lr_scheduler.CosineAnnealingLR(opt_rest, args.steps)
                if opt_rest is not None else None)
    best, hist, t0 = -1.0, [], time.time()
    for step in range(1, args.steps + 1):
        x, y, scale, _, sampled_regime = train_ds.sample(args.batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype_for(device),
                                             enabled=device == "cuda"):
            if head.extra_channels > 80:
                p, sd, dmin, dmax = ens.mean_stats(x)
            else:
                p, sd = ens.mean_std(x)
            if oof_ens is not None:
                mask = calibration_mask(sampled_regime, oof_cells)
                if mask.any():
                    op, osd = oof_ens.mean_std(x[mask])
                    p[mask] = op
                    if args.oof_spread == 'inner':
                        sd[mask] = osd
                oof_seen += int(mask.sum())
                oof_total += len(mask)
        p = p.float()
        sc = scale[:, None, None, None]
        err = (y - p) * sc
        scored = (y.abs() * sc > 0).float()
        if args.feat_spread:
            extra = torch.cat([spread_channels(sd, scale), ens.feat_spread(x)], dim=1)
        elif head.extra_channels > 80:
            extra = spread_channels_mm(sd, dmin, dmax, scale)
        else:
            extra = spread_channels(sd, scale) if head.extra_channels else None
        lo_d, hi_d = head(x, p, torch.log(scale), extra)
        pref = sps_prefactors(p, y, scale) if args.score_prefactors else None
        loss = -sps_reward(lo_d, hi_d, err, scored, tau_frac=args.tau_frac,
                           sample_weights=pref)
        opt.zero_grad(set_to_none=True)
        if opt_rest is not None:
            opt_rest.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sch.step()
        if opt_rest is not None:
            opt_rest.step(); sch_rest.step()

        if step % args.eval_every == 0:
            sv = full_scores(ens, head, val[sel], device,
                                 feat_spread=args.feat_spread)
            row = {"step": step, **{f"{sel}.{k}": v for k, v in sv.items()}}
            if oof_ens is not None:
                row['oof_window_fraction'] = oof_seen / oof_total
            print(f"[head] {step:>5} {sel} sps={sv['sps_score']:.2f} "
                  f"cov={sv['_coverage']:.3f} [{time.time()-t0:.0f}s]", flush=True)
            hist.append(row)
            json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
            take = (step == args.fixed_step if args.fixed_step
                    else sv["sps_score"] > best)          # selection on val_aoa only
            if take:
                best = sv["sps_score"]
                if oof_ens is not None:
                    args.oof_window_fraction = oof_seen / oof_total
                torch.save({"head": head.state_dict(), "args": vars(args),
                            "paths": paths, "extra_paths": extra_paths, "step": step,
                            "member_sha256": member_sha256},
                           os.path.join(args.out, "best.pt"))

    head.load_state_dict(torch.load(os.path.join(args.out, "best.pt"))["head"])
    print("\n--- learned asymmetric intervals, final ---", flush=True)
    for n, dl in val.items():
        s = full_scores(ens, head, dl, device, feat_spread=args.feat_spread)
        m4 = np.mean([s[k] for k in ("rel_l2_score", "tke_score", "mvpe_score", "sps_score")])
        # Not "UNTOUCHED": the lambda scan below reports every inflation factor on
        # this split too, `kfold.fold_score` pools it with val_aoa, and the shipped
        # lambda was read off it. It is a second validation split that no automatic
        # rule selects on -- which is a weaker guarantee than the label implied.
        tag = "  <- not selected on automatically (but scanned below)" if n == "test_aoa" else ""
        print(f"  {n:<9} rel_l2={s['rel_l2_score']:.2f} tke={s['tke_score']:.2f} "
              f"mvpe={s['mvpe_score']:.2f} sps={s['sps_score']:.2f} | mean4={m4:.2f}{tag}",
              flush=True)

    print("\n--- width inflation ---", flush=True)
    print(f"  {'lam':>5} " + " ".join(f"{n:>18}" for n in val), flush=True)
    scan = {}
    for lam in (0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.9):
        cells = []
        for n, dl in val.items():
            s = full_scores(ens, head, dl, device, lam=lam,
                             feat_spread=args.feat_spread)
            scan.setdefault(n, {})[lam] = s["sps_score"]
            cells.append(f"sps={s['sps_score']:5.2f} c={s['_coverage']:.3f}")
        print(f"  {lam:>5.2f} " + " ".join(f"{c:>18}" for c in cells), flush=True)
    json.dump(scan, open(os.path.join(args.out, "lam_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
