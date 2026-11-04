#!/usr/bin/env python3
"""Measure the meandering correlation -- the one quantity `docs/archive/FINDINGS_2026-09-10.md` §6 names as
reachable headroom, and the one this repository has no tool for.

The chain of reasoning that leads here. The oracle ladder (`tools/info_limit.py`) put
**a third of the residual rel-L2 error in the forecast window's 20-frame time-mean**,
worth about +1.0 of final if it were exact. That reads as enormous headroom until the
time-mean is split into each trajectory's own climatology -- which is trivially
predictable -- plus a deviation, which is the wake meandering and is the part that has
to be forecast. Against that deviation the model reaches a pooled correlation of
**rho = 0.775** on `val_aoa`, against **0.055** for "persist the input window's mean".
So it is doing real work, and an L2-optimal predictor at correlation rho leaves
`1 - rho^2` behind no matter what. Most of the apparent +1.0 is that irreducible term.

What is *not* settled is rho itself. Two attempts at a model-independent upper bound
both failed (the wake autocorrelation gave 0.253, below what the model already does;
the analogue method gave 0.46-0.53, also below), and §6 records the sensitivity:

    rho 0.775 -> 0.80   +0.10 of final
             -> 0.85   +0.33
             -> 0.90   +0.57

Against a 0.487 gap to first place, that is the whole distance. So rho is the target,
and nothing could be aimed at it because nothing measured it -- the 0.775 was computed
once, by hand, and never turned into a rig. A change that raised rho by 0.05 and a
change that did nothing were indistinguishable.

The climatology here is computed from the *targets* of the split being evaluated,
which makes this a diagnostic and not something a submission may do. It never touches
the model: the model sees only its input window, and rho is measured afterwards.

    python tools/meander.py --head runs/px2_head/best.pt --splits val_aoa,val_re
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realistix.data import build  # noqa: E402
from realistix.evaluate import predict_all  # noqa: E402
from realistix.models import build_model  # noqa: E402

RUNS = os.environ.get("REALISTIX_RUNS", "runs")


class Members:
    """The ensemble mean, which is what the shipped model predicts with."""

    def __init__(self, paths, device):
        self.ms = []
        for p in paths:
            ck = torch.load(p, map_location=device)
            m = build_model(ck["args"]).to(device)
            m.load_state_dict(ck["model"]); m.eval()
            for q in m.parameters():
                q.requires_grad_(False)      # predict_all does not wrap in no_grad
            self.ms.append(m)

    def __call__(self, x, *a, **k):
        return torch.stack([m(x, *a, **k) for m in self.ms]).mean(0)

    def eval(self):
        return self

    def parameters(self):
        return self.ms[0].parameters()


def stats(pred, tgt, traj_of):
    """rho and friends for the window time-mean deviation.

    ``pred``/``tgt`` are (N, T, H, W, C); ``traj_of`` maps window -> trajectory id.
    The deviation is the window's 20-frame mean minus the mean of *that trajectory's*
    windows, so what is correlated is the part a climatology cannot supply.
    """
    pm = pred[..., :2].mean(axis=1)                       # (N, H, W, 2)
    tm = tgt[..., :2].mean(axis=1)
    dp, dt = np.empty_like(pm), np.empty_like(tm)
    for t in np.unique(traj_of):
        s = traj_of == t
        dp[s] = pm[s] - pm[s].mean(axis=0, keepdims=True)
        dt[s] = tm[s] - tm[s].mean(axis=0, keepdims=True)
    a, b = dp.ravel(), dt.ravel()
    rho = float(np.corrcoef(a, b)[0, 1])
    # R^2 of the prediction as an estimator of the target deviation, not of a fitted
    # line through it -- the model does not get to rescale afterwards.
    r2 = float(1.0 - ((a - b) ** 2).sum() / (b ** 2).sum())
    return dict(rho=rho, r2=r2, std_ratio=float(a.std() / b.std()),
                n_windows=int(len(pm)),
                # what "persist the input window's mean" would score, for scale
                mean_abs_dev=float(np.abs(b).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=f"{RUNS}/px2_head/best.pt",
                    help="a head checkpoint, used only for its member paths")
    ap.add_argument("--runs", default=None, help="glob of member run dirs instead")
    ap.add_argument("--splits", default="val_aoa,val_re")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.runs:
        import glob
        paths = sorted(f"{d}/best.pt" for d in glob.glob(a.runs) if os.path.isdir(d))
    else:
        paths = torch.load(a.head, map_location="cpu")["paths"]
    print(f"{len(paths)} members: {[os.path.basename(os.path.dirname(p)) for p in paths]}\n")
    ens = Members(paths, dev)

    print(f"{'split':>8} {'rho':>7} {'R2':>7} {'std ratio':>10} {'windows':>8}")
    for sp in a.splits.split(","):
        ds = build("real", sp, stride=40)
        traj_of = np.array([i for i, _ in ds.index])
        p, t = predict_all(ens, DataLoader(ds, batch_size=64), dev)
        s = stats(p, t, traj_of)
        print(f"{sp:>8} {s['rho']:7.3f} {s['r2']:7.3f} {s['std_ratio']:10.3f} "
              f"{s['n_windows']:8d}")
    print("\nFINDINGS.md §6 recorded rho 0.775 on val_aoa and 0.675 on val_re for the")
    print("ensemble of the day; treat a reproduction of those as the rig working.")


if __name__ == "__main__":
    main()
