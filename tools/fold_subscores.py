#!/usr/bin/env python3
"""Fold deltas per subscore, and with the constant-band SPS proxy removed.

`kfold.py` scores with sps_score at its identified weight, and that sps comes from
`evaluate.subscores`, which draws a fixed symmetric 0.15-sigma band -- not the
learned asymmetric head the submission ships. The proxy rewards better-centred
predictions directly, so it over-credits any change that improves accuracy; on
distillation it contributed +0.113 of a +0.163 local delta while the board paid
+0.001. Every headline fold number in this project carries that term.

This re-reads the archived histories and reports both, per subscore, so the
accuracy-only column can be compared against the board's accuracy subscores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from kfold import FOLD_SETS, RUNS  # noqa: E402
from realistix.weights import ACC  # noqa: E402

K = ("rel_l2_score", "tke_score", "mvpe_score", "sps_score")
# The fourteen-point refit; mvpe is 0.053-0.093 depending on one influential point,
# so the accuracy total is quoted with the conservative end as well.
W = ACC


def _last_step(run_dir):
    p = os.path.join(run_dir, "history.json")
    if not os.path.exists(p):
        return None
    h = json.load(open(p))
    return max((r["step"] for r in h), default=None)


def fold_subscores(run_dir, splits, final_step=None):
    """Subscores at the run's own accuracy optimum, or None if it is not finished.

    The completion check is not optional. `history.json` is written at every eval,
    so a run at step 2000 of 8000 looks exactly like a finished one to anything that
    reads the last row -- and an under-trained run scores far worse than a bad
    configuration does. That mistake turned a +0.028 screen into a confident
    "-0.340" on 2026-08-18 (`docs/archive/HANDOFF_2026-09-10.md` section 0b), and it recurred here on
    2026-08-20: two of four `mimm12` folds were still training and the arm read
    -0.604 against a board result of -0.009.
    """
    p = os.path.join(run_dir, "history.json")
    if not os.path.exists(p):
        return None
    h = json.load(open(p))
    if any(r.get("metric_object", "member") != "member" for r in h):
        raise ValueError(f"{run_dir}: cooperative ensemble metrics cannot be compared "
                         "to member-only fold histories")
    if not h:
        return None
    want = final_step or max(r["step"] for r in h)
    if max(r["step"] for r in h) < want:
        return None

    def row(r):
        out = {}
        for k in K:
            vals = [r[f"{s}.{k}"] for s in splits if f"{s}.{k}" in r]
            if not vals:
                return None
            out[k] = float(np.mean(vals))
        return out

    # Each run is read at its own optimum, because the schedules being compared
    # differ in length: cap_w128_12k peaks at 8000 of 12000 and declines after, so
    # reading it at the last eval would score it past its own best and make a
    # 12000-step arm look worse than an 8000-step one for the wrong reason.
    #
    # The step is chosen on *accuracy only*. Selecting on the full weighted score
    # would let the constant-band SPS proxy -- the term this script exists to
    # remove -- choose the checkpoint as well as inflate it.
    rows = [r for r in (row(x) for x in h) if r]
    if not rows:
        return None
    acc = lambda r: sum(W[k] * r[k] for k in K if k != "sps_score")
    return max(rows, key=acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="reference first")
    ap.add_argument("--fold-set", default="re", choices=sorted(FOLD_SETS))
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    folds = FOLD_SETS[a.fold_set]
    splits = (["val_re"] if a.fold_set == "re" else ["val_aoa", "test_aoa"])
    seeds = [s for s in a.seeds.split(",") if s]
    base = a.tags[0]

    print(f"{a.fold_set} folds, splits {splits}, paired against '{base}'\n")
    print(f"{'tag':>16}" + "".join(f"{k.split('_score')[0]:>9}" for k in K)
          + f"{'weighted':>10}{'acc only':>10}")
    for t in a.tags[1:]:
        ds, unfinished = [], 0
        for f in folds:
            for s in seeds:
                bd = os.path.join(RUNS, f"kf_{base}_{f['name']}_s{s}")
                cd = os.path.join(RUNS, f"kf_{t}_{f['name']}_s{s}")
                # The reference sets the expected schedule; a candidate that has
                # not reached it is unfinished, not bad.
                b = fold_subscores(bd, splits)
                steps = _last_step(bd)
                c = fold_subscores(cd, splits, final_step=steps)
                if b and c:
                    ds.append({k: c[k] - b[k] for k in K})
                elif b and _last_step(cd):
                    unfinished += 1
        if not ds:
            print(f"{t:>16}  no paired folds"); continue
        m = {k: float(np.mean([d[k] for d in ds])) for k in K}
        tot = sum(W[k] * m[k] for k in K)
        acc = sum(W[k] * m[k] for k in K if k != "sps_score")
        note = f"   (n={len(ds)}" + (f", {unfinished} UNFINISHED skipped)" if unfinished else ")")
        print(f"{t:>16}" + "".join(f"{m[k]:+9.3f}" for k in K)
              + f"{tot:+10.3f}{acc:+10.3f}{note}")


if __name__ == "__main__":
    main()
