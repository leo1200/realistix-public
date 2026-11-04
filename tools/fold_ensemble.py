#!/usr/bin/env python3
"""Score a fold's seeds as an ENSEMBLE, not as an average of single models.

The lead this exists to test. `kfold.py --report` averages the per-seed scores of a
fold, so every number this project has quoted from the fold rig is a *per-member*
effect. The board scores a three-member ensemble. If ensembling compresses a gain --
and §7 records exactly that for distillation, where +0.078 per member became +0.036
for the ensemble -- then the folds must systematically over-predict, and the size of
the compression is the missing calibration.

That is not a hypothesis about the data, it is a property of the instrument, and it
is measurable with runs that already exist plus two extra seeds per fold.

The 2026-08-21 evidence it was built for: reversing the loss correction reads -0.131
on the mimic folds against -0.069 on the board, a transfer of 0.52. If the ensemble
number lands near -0.069, the fold rig's disagreement with the board is an
instrument artefact and is fixed by scoring ensembles. If it lands near -0.131, the
disagreement is real and lives somewhere else.

    python tools/fold_ensemble.py mimbase mimoldloss --fold-set mimic --seeds 0,1,2

Needs a GPU and the fold runs on disk. Reports, per fold and pooled: each tag's mean
member, its ensemble, what ensembling buys, and the paired candidate-minus-base
delta computed both ways.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "starting_kit"))

from kfold import FINAL_STEP, FOLD_SETS, RUNS  # noqa: E402
from realistix import splits  # noqa: E402
from realistix.data import WindowDataset  # noqa: E402
from realistix.evaluate import amp_dtype_for, subscores, to_ntchw  # noqa: E402
from realistix.models import build_model  # noqa: E402
from realistix.weights import ACC, ACC_NO_SPS  # noqa: E402


def reached_end(run_dir: str, ck: dict) -> bool:
    """True only once the run reached the end of *its own* schedule.

    `best.pt` exists from the first eval, so a run at step 7000 of 8000 loads and
    scores exactly like a finished one -- and scores worse, because it is
    under-trained. The first version of this tool had no such check and read two
    still-training seeds as real members, which biased the arm they were in by
    0.05. Same failure as `kfold.score_of` and `fold_subscores` guard against, and
    the same one that turned a +0.028 screen into a confident "-0.340" on
    2026-08-18. An instrument that silently substitutes what it cannot do reports a
    biased answer, not a missing one.

    The schedule comes from the checkpoint's own args, not from `kfold.FINAL_STEP`.
    A distilled arm trains 16000 steps, so a fixed 8000 floor would wave through a
    run that is barely half done -- which is the same bug wearing the next costume.
    """
    import json
    h = os.path.join(run_dir, "history.json")
    if not os.path.exists(h):
        return False
    rows = json.load(open(h))
    if not rows:
        return False
    want = int(ck.get("args", {}).get("steps") or FINAL_STEP)
    return max(r["step"] for r in rows) >= want


def fold_trajs(fold: dict) -> list:
    """The trajectories this fold holds out.

    `splits.partition` reads its holdouts from module globals, which are set from
    the environment at import. Rebinding them here is what lets one process walk
    every fold; the alternative is a subprocess per fold, which is slower and hides
    the split being used.
    """
    env = fold["env"]
    splits.HELDOUT_AOA = tuple(int(v) for v in env.get("REALISTIX_HELDOUT_AOA", "").split(",") if v)
    splits.HELDOUT_RE = tuple(int(v) for v in env.get("REALISTIX_HELDOUT_RE", "").split(",") if v)
    splits.HELDOUT_CELLS = frozenset(
        v.strip() for v in env.get("REALISTIX_HELDOUT_CELLS", "").split(",") if v.strip())
    p = splits.partition("real")
    held = p["val_aoa"] + p["test_aoa"] + p["val_re"]
    if not held:
        raise SystemExit(f"fold {fold['name']} holds out nothing")
    return held


@torch.no_grad()
def score(run_dirs: list[str], dl, device):
    """Subscores per member and for their ensemble mean, or None if any is unfinished."""
    models = []
    for d in run_dirs:
        p = os.path.join(d, "best.pt")
        if not os.path.exists(p):
            return None
        ck = torch.load(p, map_location=device)
        if not reached_end(d, ck):
            return None
        m = build_model(ck["args"])
        m.load_state_dict(ck["model"])
        models.append(m.to(device).eval())

    per, ens, tgt, spread = [[] for _ in models], [], [], []
    for x, y, scale in dl:
        x, y = x.to(device), y.to(device)
        sc = scale.to(device)[:, None, None, None]
        with torch.autocast("cuda", dtype=amp_dtype_for(device), enabled=device == "cuda"):
            outs = torch.stack([m(x) for m in models]).float()
        for i, o in enumerate(outs):
            per[i].append(to_ntchw(o * sc).cpu().numpy())
        ens.append(to_ntchw(outs.mean(0) * sc).cpu().numpy())
        tgt.append(to_ntchw(y * sc).cpu().numpy())
        # The per-element member disagreement the SPS head reads. Distillation from
        # a shared teacher halved it (0.00136 -> 0.00072, HANDOFF §7); whether a
        # distinct-teacher pool restores it is the 2026-08-21 question.
        spread.append(float((outs * sc).std(0).mean()))
    t = np.concatenate(tgt).astype(np.float32)
    return ([subscores(np.concatenate(p).astype(np.float32), t) for p in per],
            subscores(np.concatenate(ens).astype(np.float32), t),
            float(np.mean(spread)))


def wsum(s: dict, w: dict) -> float:
    return sum(v * s[k] for k, v in w.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="reference first")
    ap.add_argument("--fold-set", default="mimic", choices=sorted(FOLD_SETS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", default="", help="subset of the fold set, by name")
    ap.add_argument("--with-sps", action="store_true",
                    help="include the constant-band sps proxy; off by default because "
                         "it is not the head that ships and it over-credits accuracy")
    a = ap.parse_args()

    W = ACC if a.with_sps else ACC_NO_SPS
    seeds = [int(s) for s in a.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = a.tags[0]
    out = {t: {"member": [], "ens": []} for t in a.tags}

    folds = [f for f in FOLD_SETS[a.fold_set]
             if not a.folds or f["name"] in a.folds.split(",")]
    if not folds:
        ap.error(f"no folds named {a.folds!r} in fold set {a.fold_set!r}")
    for fold in folds:
        dl = DataLoader(WindowDataset(fold_trajs(fold), "real", stride=40), batch_size=64)
        line = f"{fold['name']:>16} ({len(dl.dataset):3d} windows) "
        for t in a.tags:
            dirs = [os.path.join(RUNS, f"kf_{t}_{fold['name']}_s{s}") for s in seeds]
            got = score(dirs, dl, device)
            if got is None:
                line += f" {t}: a seed is missing or still training "
                out[t]["member"].append(None); out[t]["ens"].append(None)
                continue
            singles, ensemble, spread = got
            m, e = float(np.mean([wsum(s, W) for s in singles])), wsum(ensemble, W)
            out[t]["member"].append(m); out[t]["ens"].append(e)
            out[t].setdefault("spread", []).append(spread)
            line += f" {t}: member {m:.3f} ens {e:.3f} ({e - m:+.3f}) spr {spread:.5f} "
        print(line, flush=True)

    print("\npaired against '%s', over the folds both sides completed:" % base)
    print(f"  {'tag':>14}{'per-member Δ':>15}{'ensemble Δ':>13}{'compression':>13}")
    for t in a.tags[1:]:
        d_m = [c - b for c, b in zip(out[t]["member"], out[base]["member"])
               if c is not None and b is not None]
        d_e = [c - b for c, b in zip(out[t]["ens"], out[base]["ens"])
               if c is not None and b is not None]
        if not d_m:
            continue
        mm, me = float(np.mean(d_m)), float(np.mean(d_e))
        ratio = me / mm if abs(mm) > 1e-9 else float("nan")
        print(f"  {t:>14}{mm:+15.3f}{me:+13.3f}{ratio:>13.2f}")
    print("\n  'compression' is ensemble Δ / per-member Δ. If it is well below 1 the")
    print("  fold rig over-states what the board will pay, purely because the board")
    print("  scores an ensemble and `kfold.py --report` averages single models.")


if __name__ == "__main__":
    main()
