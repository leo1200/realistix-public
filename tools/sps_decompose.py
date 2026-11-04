#!/usr/bin/env python3
"""Split `sps_score` into the part accuracy controls and the part the head controls.

The scorer builds SPS out of three branches that share one interval and differ only
in a per-*window* accuracy prefactor (`starting_kit/scoring.py:aggregate_sps`):

    sps = (1/n) * sum_e  W_s(e) * exp(-width_e / SIGMA_GLOBAL) * 1{covered_e}
    W_s = 0.5 (1 - pm_dm) + 0.3 (1 - pm_tke) + 0.2 (1 - pm_mvpe),  pm = m / (0.5 + m)

So SPS is a product of an **accuracy factor** `W` -- fixed once the forecaster has
run, computable without any interval at all -- and a **width factor** `R = exp(-nil)
1{inside}`, which is the only thing the interval head can move. Two consequences,
and this file measures both rather than asserting them.

**1. Every accuracy change is worth more than the accuracy weights say.** A local rig
scores folds through `weights.ACC_NO_SPS`, on the stated grounds that a fold's
`sps_score` is `evaluate.subscores`' constant 0.15-sigma band and not the learned
head. That is right about `R` and wrong about `W`: the accuracy factor needs no head,
and dropping `sps` entirely threw it away with the proxy. Priced at the board's own
operating point the omission is large -- `weights.acc_effective` puts it at 1.5x on
rel_l2, 1.7x on tke and **1.9x on mvpe** -- and it is not a uniform rescaling, so it
re-ranks. MVPE rises from 0.195 of rel_l2 to 0.255.

**2. Whether the head is mis-trained is an empirical question, not an obvious one.**
`models.uncertainty.sps_reward` optimises the unweighted mean of `R`, and its
docstring justifies dropping `W` because it "is constant in h and drops out". Per
element that is true, and for an unconstrained head the objective is separable across
windows, so `W > 0` cannot move the optimum. What it can move is where a *capacity-
limited* head spends its parameters, and that shows up as covariance between `W_s`
and the reward the head actually buys in window `s`. This script measures that
covariance. If it is already positive the head has found the alignment on its own.

    python tools/sps_decompose.py --head runs/hd_muf/best.pt
    python tools/sps_decompose.py --head runs/hsw_joint3_mim1/best.pt --split val_aoa
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "starting_kit"))
sys.path.insert(0, HERE)

import scoring as S  # noqa: E402
from realistix.data import build  # noqa: E402
from realistix.evaluate import amp_dtype_for, to_ntchw  # noqa: E402
from realistix.models.uncertainty import UncertaintyHead  # noqa: E402
from realistix.weights import WEIGHTS, acc_effective  # noqa: E402
from train_final_head import Ensemble, spread_channels, spread_channels_mm  # noqa: E402

BRANCH = {"rel_l2": 0.5, "tke": 0.3, "mvpe": 0.2}


def window_weights(pred: np.ndarray, tgt: np.ndarray, c: int) -> dict[str, np.ndarray]:
    """The scorer's own per-window metrics, and the `W_s` they imply."""
    m = {"rel_l2": S.rel_l2_per_sample(pred, tgt, c),
         "tke": S.tke_rel_l2_per_sample(pred, tgt, c),
         "mvpe": S.mvpe_rel_l2_per_sample(pred, tgt)}
    m["W"] = sum(b * (1.0 - m[k] / (0.5 + m[k])) for k, b in BRANCH.items())
    return m


@torch.no_grad()
def collect(ens, head, dl, device, feat_spread=False):
    """(prediction, target, lower, upper) for a whole split, in scorer layout."""
    out = [[], [], [], []]
    for x, y, scale in dl:
        x, y = x.to(device), y.to(device)
        sc = scale.to(device)[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device), enabled=device == "cuda"):
            p, sd = ens.mean_std(x)
        p = p.float()
        if feat_spread:
            extra = torch.cat([spread_channels(sd, scale.to(device)),
                               ens.feat_spread(x)], dim=1)
        elif head.extra_channels > 80:
            _, sd2, dmin, dmax = ens.mean_stats(x)
            extra = spread_channels_mm(sd2, dmin, dmax, scale.to(device))
        else:
            extra = spread_channels(sd, scale.to(device)) if head.extra_channels else None
        lo_d, hi_d = head(x, p, torch.log(scale.to(device)), extra)
        pp = p * sc
        for buf, arr in zip(out, (pp, y * sc, pp - lo_d, pp + hi_d)):
            buf.append(to_ntchw(arr).cpu().numpy())
    return [np.concatenate(b).astype(np.float32) for b in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--split", default="val_aoa")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--member-precision",choices=("checkpoint","auto","float32"),default="checkpoint",
                    help="override member precision for a matched evaluation of differently trained heads")
    ap.add_argument("--json", default=None,
                    help="also write {sps, W_bar, R_bar, corr, coverage} here, so a "
                         "sweep can tabulate the split without re-parsing this text")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=device, weights_only=False)
    ha = ck["args"]
    n_extra = 0
    if ha.get("spread"):
        n_extra = 160 if ha.get("spread_stats") == "minmax" else 80
    if ha.get("feat_spread"):
        # Heads trained with --feat-spread carry the members' decoder-feature
        # disagreement as extra input; rebuilding without it fails on stem.weight.
        n_extra += 96
    head = UncertaintyHead(width=ha["width"], extra_channels=n_extra,
                           offset=bool(ha.get("offset"))).to(device)
    head.load_state_dict(ck["head"])
    head.eval()
    ens = Ensemble(ck["paths"], device,
                   [q for q in ck.get("extra_paths", []) if isinstance(q, str)],
                   member_precision=(ha.get('member_precision','auto') if args.member_precision=='checkpoint'
                                     else args.member_precision))

    ds = build("real", args.split, stride=args.stride)
    if not len(ds):
        raise SystemExit(f"split {args.split} is empty under the current holdout env")
    pred, tgt, lo, hi = collect(ens, head, DataLoader(ds, batch_size=64), device,
                                bool(ha.get("feat_spread")))
    c = S.measured_channels(tgt)
    sps, cov = S.aggregate_sps(pred, tgt, c, lo[..., :c], hi[..., :c])
    sps *= 100.0

    m = window_weights(pred, tgt, c)
    W = m["W"]

    # The element-level reward the head controls, and the per-window mean of it.
    t, p = tgt[..., :c], pred[..., :c]
    scored = t != 0.0
    inside = (t >= lo[..., :c]) & (t <= hi[..., :c])
    R = np.exp(-(hi[..., :c] - lo[..., :c]) / S.SIGMA_GLOBAL) * inside
    ax = tuple(range(1, R.ndim))
    n_s = scored.sum(axis=ax).astype(np.float64)          # scored elements per window
    R_win = np.where(n_s > 0, (R * scored).sum(axis=ax) / np.maximum(n_s, 1), 0.0)

    print(f"head    {args.head}")
    print(f"members {', '.join(os.path.basename(os.path.dirname(os.path.realpath(q))) for q in ck['paths'])}")
    print(f"split   {args.split}  ({len(W)} windows, {int(scored.sum())} scored elements)")
    print(f"\nsps_score {sps:.3f}   coverage {cov:.4f}")

    # --- the product decomposition -----------------------------------------------
    # sps = mean_e[W R] exactly; W_bar and R_bar are the element-weighted marginals,
    # so the residual below is the covariance term and nothing else.
    wbar = float((n_s * W).sum() / n_s.sum())
    rbar = float((n_s * R_win).sum() / n_s.sum())
    mean_width = float(np.sum((hi[..., :c] - lo[..., :c]) * scored) / n_s.sum())
    mae = float(np.sum(np.abs(t - p) * scored) / n_s.sum())
    # Target-assisted diagnostic only: the narrowest covering interval anchored
    # at the prediction has width |error|. It cannot be deployed as a head.
    oracle_win = (np.exp(-np.abs(t - p) / S.SIGMA_GLOBAL) * scored).sum(axis=ax) / np.maximum(n_s, 1)
    oracle_sps = float(100 * np.sum(n_s * W * oracle_win) / n_s.sum())
    print("\naccuracy factor  W_bar = %.4f      (what the forecaster fixes)" % wbar)
    print("width factor     R_bar = %.4f      (all the head can move)" % rbar)
    print("product        W*R*100 = %.3f" % (100 * wbar * rbar))
    print("covariance term        = %+.3f     (%.1f%% of sps)"
          % (sps - 100 * wbar * rbar, 100 * (sps - 100 * wbar * rbar) / sps))

    q = np.percentile(W, [5, 25, 50, 75, 95])
    print("\nW_s spread   p5 %.3f  p25 %.3f  p50 %.3f  p75 %.3f  p95 %.3f   (%.2fx)"
          % (*q, q[-1] / q[0]))
    r = np.corrcoef(W, R_win)[0, 1]
    print("corr(W_s, per-window reward) = %+.3f" % r)
    print("  positive: the head already spends width where the metric pays for it,\n"
          "  so re-weighting sps_reward by W_s has little left to find."
          if r > 0.1 else
          "  not positive: the head's width allocation is uncorrelated with what the\n"
          "  metric pays per window, which is what a W-weighted reward would fix.")

    # --- what the omission costs a local screen ----------------------------------
    sub = {k: S.score_error(float(m[k].mean())) for k in BRANCH}
    eff = acc_effective(sub, K=rbar)
    print("\neffective accuracy weights at THIS operating point (K = R_bar):")
    print(f"  {'quantity':8} {'direct':>8} {'via sps':>9} {'total':>8} {'inflation':>10}")
    for k in BRANCH:
        d = WEIGHTS[k + "_score"]
        print(f"  {k:8} {d:8.4f} {eff[k+'_score']-d:9.4f} {eff[k+'_score']:8.4f} "
              f"{eff[k+'_score']/d:9.2f}x")

    if args.json:
        import json
        per_cell = {}
        for re, aoa in np.unique(ds.window_regimes, axis=0):
            use = (ds.window_regimes == (re, aoa)).all(axis=1)
            weight = n_s[use]
            per_cell[f'{int(re)}_{int(aoa)}'] = {
                'windows': int(use.sum()),
                'sps': float(100 * np.sum(weight * W[use] * R_win[use]) / weight.sum()),
                'oracle_sps': float(100 * np.sum(weight * W[use] * oracle_win[use]) / weight.sum()),
                'W_bar': float(np.sum(weight * W[use]) / weight.sum()),
                'R_bar': float(np.sum(weight * R_win[use]) / weight.sum()),
            }
        json.dump({"head": args.head, "split": args.split, "windows": int(len(W)),
                   "member_precision": ens.member_precision,
                   "sps": sps, "coverage": float(cov), "W_bar": wbar, "R_bar": rbar,
                   "mean_width": mean_width, "mae": mae, "oracle_sps": oracle_sps,
                   "gap_to_oracle": oracle_sps - sps, "per_cell": per_cell,
                   "corr_W_reward": float(r),
                   **{k + "_score": sub[k] for k in BRANCH}},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
