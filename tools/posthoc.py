#!/usr/bin/env python3
"""Post-hoc calibration of a finished ensemble, scored with the kit's own metric.

Two scalars, both applied after the forward pass, both free at training time.

**alpha** inflates the prediction's temporal fluctuation about its own time-mean::

    pred' = mean_t(pred) + alpha * (pred - mean_t(pred))

A deterministic model trained on a squared error predicts the conditional mean,
whose fluctuation energy is suppressed -- that is the whole reason TKE sits at
0.56 error while rel-L2 sits at 0.12. TKE scales as alpha^2, so alpha > 1 attacks
the worst subscore directly. What it costs is rel-L2, and what it costs nothing at
all is MVPE: MVPE compares *time-averaged* velocities, and this transformation
leaves every time-mean exactly where it was.

**lam** inflates both interval half-widths. The head is fitted on windows the
forecaster trained on, so the errors it calibrates against are smaller than the
ones it meets; the shipped model scored SPS 42.8 locally and 39.5 on the board.

Both are scanned against ``scoring.py``, on splits the forecaster never saw.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starting_kit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scoring as S  # noqa: E402
from realistix.data import build  # noqa: E402
from realistix.evaluate import amp_dtype_for, to_ntchw  # noqa: E402
from realistix.models.uncertainty import UncertaintyHead  # noqa: E402
from train_final_head import Ensemble, spread_channels  # noqa: E402


@torch.no_grad()
def collect(ens, head, dl, device):
    """One pass over a split: predictions, targets and raw half-widths."""
    out = {k: [] for k in ("pred", "tgt", "below", "above")}
    for x, y, scale in dl:
        x, y, sc_v = x.to(device), y.to(device), scale.to(device)
        sc = sc_v[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device),
                            enabled=device == "cuda"):
            p, sd = ens.mean_std(x)
        p = p.float()
        if head is None:
            below = above = torch.full_like(p, 0.09 * S.SIGMA_GLOBAL)
        else:
            extra = spread_channels(sd, sc_v) if head.extra_channels else None
            below, above = head(x, p, torch.log(sc_v), extra)
        out["pred"].append(to_ntchw(p * sc).cpu().numpy())
        out["tgt"].append(to_ntchw(y * sc).cpu().numpy())
        out["below"].append(to_ntchw(below).cpu().numpy())
        out["above"].append(to_ntchw(above).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def score(d, alpha=1.0, lam=1.0):
    p, t = d["pred"], d["tgt"]
    if alpha != 1.0:
        m = p.mean(axis=1, keepdims=True)
        p = m + alpha * (p - m)
    c = S.measured_channels(t)
    rel = float(np.mean(S.rel_l2_per_sample(p, t, c)))
    tke = float(np.mean(S.tke_rel_l2_per_sample(p, t, c)))
    mvpe = S.mvpe_rel_l2(p, t)
    lo = p[..., :c] - lam * d["below"][..., :c]
    hi = p[..., :c] + lam * d["above"][..., :c]
    sps, cov = S.aggregate_sps(p, t, c, lo, hi)
    r = {"rel_l2_score": S.score_error(rel), "tke_score": S.score_error(tke),
         "mvpe_score": S.score_error(mvpe), "sps_score": S.score_sps(sps),
         "_rel_l2": rel, "_tke": tke, "_mvpe": mvpe, "_coverage": cov}
    r["mean4"] = float(np.mean([r[k] for k in ("rel_l2_score", "tke_score",
                                               "mvpe_score", "sps_score")]))
    return r


def row(name, r):
    return (f"  {name:<10} rel_l2={r['rel_l2_score']:6.2f} tke={r['tke_score']:6.2f} "
            f"mvpe={r['mvpe_score']:6.2f} sps={r['sps_score']:6.2f} "
            f"cov={r['_coverage']:.3f} | mean4={r['mean4']:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None, help="best.pt from train_final_head")
    ap.add_argument("--runs", default=None, help="ensemble glob (else the head's own paths)")
    ap.add_argument("--n-models", type=int, default=3)
    ap.add_argument("--spread-runs", default=None,
                    help="glob of members used only for the spread, not the mean")
    ap.add_argument("--splits", default="val_aoa,test_aoa")
    ap.add_argument("--alphas", default="1.0,1.05,1.1,1.15,1.2,1.3,1.4,1.5")
    ap.add_argument("--lams", default="1.0,1.1,1.25,1.4,1.6")
    ap.add_argument("--out", default=None, help="write the full scan as json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, extra = None, []
    if args.head:
        ck = torch.load(args.head, map_location=device)
        ha = ck["args"]
        head = UncertaintyHead(width=ha["width"],
                               extra_channels=80 if ha.get("spread") else 0).to(device)
        head.load_state_dict(ck["head"])
        head.eval()
        paths = ck["paths"]
        extra = [q for q in ck.get("extra_paths", []) if isinstance(q, str)]
    if args.runs:
        paths = sorted(glob.glob(os.path.join(args.runs, "best.pt")))[:args.n_models]
    if args.spread_runs:
        extra = sorted(glob.glob(os.path.join(args.spread_runs, "best.pt")))
    ens = Ensemble(paths, device, extra)
    print("ensemble: " + ", ".join(os.path.basename(os.path.dirname(p)) for p in paths))
    print(f"head: {args.head or 'constant 0.09 sigma band'}\n")

    splits = args.splits.split(",")
    data = {n: collect(ens, head, DataLoader(build("real", n, stride=40), batch_size=64),
                       device)
            for n in splits}

    scan = {}
    for n in splits:
        base = score(data[n])
        print(f"[{n}] baseline")
        print(row("alpha=1", base))
        print(f"[{n}] fluctuation inflation")
        for a in (float(v) for v in args.alphas.split(",")):
            r = score(data[n], alpha=a)
            scan.setdefault(n, {}).setdefault("alpha", {})[a] = r
            print(row(f"alpha={a}", r))
        print(f"[{n}] width inflation")
        for l in (float(v) for v in args.lams.split(",")):
            r = score(data[n], lam=l)
            scan.setdefault(n, {}).setdefault("lam", {})[l] = r
            print(row(f"lam={l}", r))
        print()

    # The two interact through coverage: widening the fluctuation moves the
    # prediction away from the target on exactly the cells the intervals must cover.
    best = {}
    for n in splits:
        grid = {}
        for a in (float(v) for v in args.alphas.split(",")):
            for l in (float(v) for v in args.lams.split(",")):
                grid[f"{a},{l}"] = score(data[n], alpha=a, lam=l)
        k = max(grid, key=lambda k: grid[k]["mean4"])
        best[n] = (k, grid[k])
        scan.setdefault(n, {})["grid"] = {k: v["mean4"] for k, v in grid.items()}
        print(f"[{n}] best (alpha,lam) = {k}")
        print(row("joint", grid[k]))
    if args.out:
        json.dump(scan, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
