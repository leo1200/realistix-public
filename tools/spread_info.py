#!/usr/bin/env python3
"""How much SPS is reachable from the ensemble spread alone, before any head exists.

`tools/sps_decompose.py` splits `sps_score` into an accuracy factor the forecaster
fixes and a width factor `R_bar` the interval head owns. The board history says
`R_bar` is the only thing that has moved since 2026-08-13, and 898763 moved it
without changing the head recipe at all -- the members changed, and the same head
recipe got better intervals out of them. The obvious mechanism is that the new
members' per-element disagreement is a better error predictor. This measures that
directly, and it needs no head, so it separates "the spread carries more information"
from "this particular head happened to train better".

The construction is non-parametric and is an upper bound on any spread-only head
with symmetric intervals. Bin every scored element by its spread, and inside each
bin choose the single half-width `h` maximising the realised reward

    mean over the bin of  exp(-2h / SIGMA_GLOBAL) * 1{|error| <= h}

which is what a head with unlimited capacity, reading nothing but the spread, would
converge to. Two reference rows bracket it: one bin (the best constant band, no
spread information at all) and the per-element oracle (`h = |error|`, unreachable).
The bound sits between them, and where it sits *is* the information content of the
spread in the metric's own units.

Asymmetric bounds are worth more -- the shipped head emits them and an interval
running from prediction to target costs half a symmetric one -- so this understates
the reachable level. It is used as a **paired** comparison between two member sets,
where that offset cancels.

    python tools/spread_info.py --runs 'runs/kf_mimbase_mim1_s*' --split val_aoa
"""

from __future__ import annotations

import argparse
import glob
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
from realistix.evaluate import amp_dtype_for  # noqa: E402
from realistix.weights import WEIGHTS  # noqa: E402
from train_final_head import Ensemble  # noqa: E402


@torch.no_grad()
def gather(ens, dl, device):
    """Flat arrays of (|error|, spread, scored) in physical units, plus per-window W."""
    err, spr, msk = [], [], []
    for x, y, scale in dl:
        x, y = x.to(device), y.to(device)
        sc = scale.to(device)[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device), enabled=device == "cuda"):
            p, sd = ens.mean_std(x)
        err.append(((y - p.float()) * sc).abs().cpu().numpy())
        spr.append((sd.float() * sc).cpu().numpy())
        msk.append(((y * sc).abs() > 0).cpu().numpy())
    cat = lambda v: np.concatenate(v).ravel()
    return cat(err), cat(spr), cat(msk)


def best_h(e: np.ndarray, sigma: float, grid: np.ndarray) -> tuple[float, float]:
    """(reward, half-width) of the best single symmetric band for one bin."""
    # exp(-2h/sigma) * P(|e| <= h), evaluated on a fixed grid so bins are comparable.
    cov = (e[None, :] <= grid[:, None]).mean(axis=1)
    r = np.exp(-2.0 * grid / sigma) * cov
    i = int(np.argmax(r))
    return float(r[i]), float(grid[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="glob of member run directories")
    ap.add_argument("--extra-runs", default=None,
                    help="glob of runs whose members join the SPREAD but not the "
                         "mean (Ensemble extra_paths) -- the screen for a "
                         "heterogeneous disagreement-only member")
    ap.add_argument("--split", default="val_aoa")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--grid", type=int, default=400)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(args.runs, "best.pt")))
    extra = (sorted(glob.glob(os.path.join(args.extra_runs, "best.pt")))
             if args.extra_runs else [])
    if len(paths) < 2:
        raise SystemExit(f"need at least two members for a spread; found {len(paths)}")
    ens = Ensemble(paths, device, extra_paths=extra)
    ds = build("real", args.split, stride=args.stride)
    if not len(ds):
        raise SystemExit(f"split {args.split} is empty under the current holdout env")
    e, s, m = gather(ens, DataLoader(ds, batch_size=64), device)
    e, s = e[m], s[m]
    sigma = S.SIGMA_GLOBAL
    grid = np.linspace(0.0, 1.5 * sigma, args.grid)

    print(f"members  {', '.join(os.path.basename(os.path.dirname(os.path.realpath(q))) for q in paths)}")
    print(f"split    {args.split}   {e.size} scored elements")
    print(f"corr(spread, |error|)          {np.corrcoef(s, e)[0, 1]:+.4f}")
    print(f"spearman-ish (rank, 200k sub)  "
          f"{np.corrcoef(np.argsort(np.argsort(s[:200000])), np.argsort(np.argsort(e[:200000])))[0, 1]:+.4f}")

    r_const, h_const = best_h(e, sigma, grid)
    print(f"\nbest constant band             R = {r_const:.4f}  (h = {h_const/sigma:.3f} sigma)")

    # Equal-count bins in spread, so every bin carries the same weight in the mean.
    order = np.argsort(s, kind="stable")
    parts = np.array_split(order, args.bins)
    tot, hs = 0.0, []
    for idx in parts:
        r, h = best_h(e[idx], sigma, grid)
        tot += r * idx.size
        hs.append(h)
    r_bin = tot / e.size
    print(f"best width per spread bin      R = {r_bin:.4f}  "
          f"(h spans {min(hs)/sigma:.3f}-{max(hs)/sigma:.3f} sigma over {args.bins} bins)")

    r_oracle = float(np.exp(-2.0 * e / sigma).mean())
    print(f"per-element oracle (h = |e|)   R = {r_oracle:.4f}")
    print(f"\nspread is worth {100*(r_bin - r_const):+.3f} of R_bar over a constant band, "
          f"{100*(r_bin - r_const)/max(r_oracle - r_const, 1e-9):.1f}% of the way to the oracle")
    print("multiply by W_bar (~0.71 at the board, ~0.76 on a mimic fold) and by the "
          f"sps weight {WEIGHTS['sps_score']} for units of final score.")
    print("\nComparing two member sets, read the two columns separately -- they are "
          "different mechanisms:\n"
          "  R_const        error MAGNITUDE near the band width, i.e. the bulk of |e|.\n"
          "                 Note this is not rel_l2, which is an L2 norm and is carried\n"
          "                 by the tail; a change can move one and not the other.\n"
          "  R_bin-R_const  spread INFORMATION -- how well the disagreement ranks the\n"
          "                 error, which is the only part a better ensemble buys the head.")


if __name__ == "__main__":
    main()
