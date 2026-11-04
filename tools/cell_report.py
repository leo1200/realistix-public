#!/usr/bin/env python3
"""Decompose mimic-fold error per held-out CELL, with an in-sample control.

The question this answers (2026-08-21): the hidden 19 cells are not all the same
kind of unseen. Most are bracketed by real neighbours; the 27975 column is past the
highest measured Reynolds number, i.e. one-sided extrapolation, and no local
instrument has ever separated that case from ordinary in-grid interpolation. If the
extrapolation penalty at the range edge is qualitatively larger, then a quarter of
the hidden score behaves differently from what the pooled fold numbers describe --
and MVPE, the subscore no local instrument tracks, is the natural place for a
regime-mean drift to hide.

Design. For each mimic fold F and each cell c it holds out, score tag T's seed
ensemble on c's windows alone. Then score the ensembles of every fold G != F whose
holdout does NOT contain c on the same windows: those models trained on c, so their
error is the in-sample level and the difference is the generalisation penalty for
that cell, with intrinsic difficulty controlled. Cells are then pooled by kind:

  edge-hi    a held-out high-Re column with nothing above it in training
             (mim1's 26700: the only true stand-in for the hidden 27975 column)
  edge-lo    the same at the bottom of the range (mim2's 3750)
  hi-col     held-out high-Re cells with real data still above them in training
             (mim3's 25425, mim4's 24150 -- interpolation, despite the altitude)
  interior   a held-out interior Reynolds column (13950 / 11400 / 17775 / 8850)
  aoa5       the AoA=5 singletons at mid Re (interpolation on both axes)
  single     everything else

Windows use the scoring stride (40). Sample sizes per cell are ~20 windows, so
read the pooled categories, not a single cell.

    python tools/cell_report.py --tags mimbase,mimdist --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "starting_kit"))

from kfold import FOLD_SETS, RUNS  # noqa: E402
from fold_ensemble import reached_end  # noqa: E402
from realistix.data import WindowDataset  # noqa: E402
from realistix.evaluate import amp_dtype_for, subscores, to_ntchw  # noqa: E402
from realistix.models import build_model  # noqa: E402
from realistix.splits import Traj  # noqa: E402

MIMIC = FOLD_SETS["mimic"]
CATS = ("far-1", "far-2", "far-3",
        "edge-hi", "edge-lo", "hi-col", "interior", "aoa5", "single")


def held_cells(fold: dict) -> list[str]:
    return [c for c in fold["env"]["REALISTIX_HELDOUT_CELLS"].split(",") if c]


def categorize(cell: str, fold_name: str) -> str:
    re_, aoa = (int(v) for v in cell.split("_"))
    if fold_name == "far1":
        # Extrapolation distance in grid steps past the fold's maximum trained
        # Reynolds number (22875): the dose axis this fold exists to vary.
        return {24150: "far-1", 25425: "far-2", 26700: "far-3"}[re_]
    if fold_name == "mim1" and re_ == 26700:
        return "edge-hi"
    if fold_name == "mim2" and re_ == 3750:
        return "edge-lo"
    if (fold_name, re_) in {("mim3", 25425), ("mim4", 24150)}:
        return "hi-col"
    if re_ in {13950, 11400, 17775, 8850}:
        return "interior"
    if aoa == 5:
        return "aoa5"
    return "single"


def load_ensemble(tag: str, fold_name: str, seeds: list[int], device: str):
    models = []
    for s in seeds:
        d = os.path.join(RUNS, f"kf_{tag}_{fold_name}_s{s}")
        p = os.path.join(d, "best.pt")
        if not os.path.exists(p):
            return None
        ck = torch.load(p, map_location=device, weights_only=False)
        if not reached_end(d, ck):
            return None
        m = build_model(ck["args"])
        m.load_state_dict(ck["model"])
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def cell_scores(models, dl, device):
    ens, tgt = [], []
    for x, y, scale in dl:
        x, y = x.to(device), y.to(device)
        sc = scale.to(device)[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device), enabled=device == "cuda"):
            outs = torch.stack([m(x) for m in models]).float()
        ens.append(to_ntchw(outs.mean(0) * sc).cpu().numpy())
        tgt.append(to_ntchw(y * sc).cpu().numpy())
    t = np.concatenate(tgt).astype(np.float32)
    s = subscores(np.concatenate(ens).astype(np.float32), t)
    return {k: s[k] for k in ("_rel_l2", "_tke", "_mvpe")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="mimbase")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fold-set", default="mimic", choices=("mimic", "far"))
    ap.add_argument("--control-tag", default="mimbase",
                    help="fold-set far only: whose mimic-fold ensembles supply the "
                         "in-sample level (no far-fold model ever trains on a far "
                         "cell, so the control has to come from the mimic folds)")
    ap.add_argument("--out", default="", help="also dump the raw table as JSON")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    walk = FOLD_SETS[a.fold_set]

    # cell -> mimic fold(s) that hold it out; those folds never serve as a cell's
    # in-sample control, whichever fold set is being scored.
    holds = {}
    for f in MIMIC:
        for c in held_cells(f):
            holds.setdefault(c, []).append(f["name"])

    control_ens = None
    if a.fold_set == "far":
        control_ens = {f["name"]: load_ensemble(a.control_tag, f["name"], seeds, device)
                       for f in MIMIC}

    rows = []
    for tag in a.tags.split(","):
        ens = {f["name"]: load_ensemble(tag, f["name"], seeds, device) for f in walk}
        missing = [k for k, v in ens.items() if v is None]
        if missing:
            print(f"{tag}: missing/unfinished folds {missing}, skipped")
            continue
        for f in walk:
            for cell in held_cells(f):
                re_, aoa = (int(v) for v in cell.split("_"))
                traj = Traj(re_, aoa)
                if not os.path.exists(traj.path("real")):
                    continue  # 17775_0 etc: withheld from the release itself
                dl = DataLoader(WindowDataset([traj], "real", stride=40), batch_size=64)
                out = cell_scores(ens[f["name"]], dl, device)
                pool = control_ens if control_ens is not None else ens
                ins = [cell_scores(pool[g["name"]], dl, device)
                       for g in MIMIC if g["name"] not in holds.get(cell, [])
                       and pool.get(g["name"]) is not None]
                row = dict(tag=tag, cell=cell, fold=f["name"],
                           cat=categorize(cell, f["name"]),
                           n=len(dl.dataset), held=out,
                           insample={k: float(np.mean([i[k] for i in ins])) for k in out})
                rows.append(row)
                print(f"{tag} {f['name']} {cell:>9} [{row['cat']:>8}] "
                      + " ".join(f"{k[1:]}: {out[k]:.4f} (in {row['insample'][k]:.4f})"
                                 for k in out), flush=True)

    for tag in sorted({r["tag"] for r in rows}):
        print(f"\n== {tag}: penalty (held-out minus in-sample), pooled by category ==")
        print(f"{'category':>9}{'cells':>7}{'rel_l2':>18}{'tke':>18}{'mvpe':>18}")
        for cat in CATS:
            sel = [r for r in rows if r["tag"] == tag and r["cat"] == cat]
            if not sel:
                continue
            cols = []
            for k in ("_rel_l2", "_tke", "_mvpe"):
                pen = [r["held"][k] - r["insample"][k] for r in sel]
                lvl = [r["held"][k] for r in sel]
                cols.append(f"{np.mean(pen):+.4f} @{np.mean(lvl):.4f}")
            print(f"{cat:>9}{len(sel):>7}" + "".join(f"{c:>18}" for c in cols))
        print("  (each entry: mean penalty @ mean held-out level, raw errors, lower=better)")

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
