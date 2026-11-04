#!/usr/bin/env python3
"""Rank runs by the identified scoring weights instead of the unweighted mean.

The weights live in `realistix.weights` -- one copy of the fit, imported here.
Fourteen submissions, no intercept, leave-one-out RMS 0.0115: rel_l2 0.4738, sps
0.2529, tke 0.1057, mvpe 0.0926. `time` is excluded because it is a property of
the packed submission, not of a training run. mean4 gave the same ordering on
everything archived, but it weights mvpe over-heavily relative to rel_l2.

Two cautions before quoting a number out of this.

* Runs ranked before 2026-08-21 used the eight-point fit, which put mvpe at 0.0319
  rather than 0.0926. Re-running this today will not reproduce those figures
  exactly; use `weights.EIGHT_POINT` if a number has to match the documents.
* `sps_score` here is `evaluate.subscores`' constant 0.15-sigma band, not the
  learned asymmetric head that ships, so it over-credits any change that improves
  centring. `kfold.py --report` prints an accuracy-only delta alongside for that
  reason; this module has no such split, so treat its ordering as indicative.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from realistix.weights import ACC as W  # noqa: E402


def summarise(pat, split="val_aoa"):
    """Weighted score per run matching ``pat`` (a glob of run directories)."""
    out = []
    for d in sorted(x for x in glob.glob(pat) if os.path.isdir(x)):
        # A run still in flight has a directory but no history yet, which is the
        # normal state halfway through a sweep rather than an error.
        if not os.path.exists(f"{d}/history.json"):
            continue
        h = json.load(open(f"{d}/history.json"))
        b = max(h, key=lambda r: sum(W[k] * r[f"{split}.{k}"] for k in W))
        out.append((sum(W[k] * b[f"{split}.{k}"] for k in W),
                    b[f"{split}.rel_l2_score"], b[f"{split}.sps_score"]))
    return np.array(out) if out else None


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val_aoa"
    base = None
    print(f"[{split}]  weighted score, then its two biggest components")
    print(f"{'run':28s} {'n':>2s} {'weighted':>9s} {'rel_l2':>8s} {'sps':>7s} {'vs base':>9s}")
    for pat in sys.argv[2:] or ["condnorm_s*", "nomvpe_s*", "relup_s*"]:
        v = summarise(pat, split)
        if v is None:
            continue
        if base is None:
            base = v[:, 0].mean()
        print(f"{pat:28s} {len(v):2d} {v[:,0].mean():9.3f} {v[:,1].mean():8.3f} "
              f"{v[:,2].mean():7.2f} {v[:,0].mean()-base:+9.3f}")
