#!/usr/bin/env python3
"""Estimate a candidate's *private test* score -- the one that decides placement.

**Superseded 2026-08-20 by `tools/predict_score.py`; see docs/archive/developer_guide.md.**
Kept for the archived fold comparisons and for the training-fraction extrapolation
below, which nothing else implements. Its framing was wrong and the correction is
the point:

> This file used to open "the leaderboard and the Decision Phase ask different
> questions". They do not. The simulated release covers a complete 20 x 5 grid of
> (Reynolds, angle) and the real release covers 81 of those 100 cells; the 19 cells
> with simulation and no real measurement are structured, and withheld-per-angle
> 4/6/2/3/4 reproduces the released counts 16/14/18/17/16 exactly. Under
> RealPDEBench's n / (N-n)/2 / (N-n)/2 protocol the hidden development set and the
> private test set are two halves of those same 19 cells. Both score unseen
> parameter *combinations*, and one instrument serves both. `docs/archive/FINDINGS_2026-09-10.md` 0a.

So the regime folds are still the right family of instrument, but not because the
board measures something else -- the mimic fold set (`kfold.py --fold-set mimic`)
holds out cells shaped like the withheld 19 and is the closer match.

Three corrections have to be applied to a raw fold number, and each one was learned
the hard way:

1. **Drop the sps term.** `evaluate.subscores` draws a constant 0.15-sigma band, not
   the learned asymmetric head the model ships. It over-credits any change that
   improves accuracy, and supplied 60-75% of every fold headline quoted before
   2026-08-20 -- distillation's "+0.146" is +0.034 without it.
2. **Read each run at its own optimum.** Schedules differ, and a 12000-step arm read
   at its last eval is past its peak.
3. **Extrapolate to full data.** The angle folds train on 70.5% of the release and
   the Re folds on 88.9%; the Decision Phase re-trains on 100%. Effects do move with
   that: `--mults 1,2` reads +0.062 / +0.017 / -0.013 at 70.5 / 88.9 / 100%.

The extrapolation is the weak step and is reported, never hidden: it rests on one
change measured at a handful of points. Where both fold sets exist the slope is
taken from them; where only one does, the number is quoted as-is with a warning.

**Two reasons not to quote its precision.** The three numbers above are not in the
same units -- the two fold values include the constant-band sps proxy that
`docs/archive/FINDINGS_2026-09-10.md` 7 and correction 1 above both say to drop, while the board value is a
final-score delta. In accuracy-only units the Re-fold figure is +0.009, not +0.017
(`fold_subscores.py`). And a fourth point exists: the mimic folds train on 87.7%
and read +0.004 accuracy-only, where a line through the pre-existing angle-fold and
board numbers predicts +0.020. The direction survives -- a capacity cut is worth
less the more data you train on, and the zero crossing is somewhere in the low 90s
of a percent, so `--mults 1,2` should still not ship at 100%. The "predicted the
middle to 0.002" claim does not survive, and this module does not make it.

    python tools/final_model.py rebase cap_w160_12k dist_tens16k
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)
ANGLE_FRAC, RE_FRAC = 0.705, 0.889


def acc_delta(ref, tag, fold_set, seeds):
    """Accuracy-only weighted delta from tools/fold_subscores.py."""
    out = subprocess.run(
        [PY, os.path.join(HERE, "fold_subscores.py"), ref, tag,
         "--fold-set", fold_set, "--seeds", seeds],
        capture_output=True, text=True)
    for line in (out.stdout or "").splitlines():
        p = line.split()
        # "<tag> no paired folds" is the miss case; a result line ends with (n=N).
        if p and p[0] == tag and p[-1].startswith("(n="):
            return float(p[-2]), int(p[-1].strip("(n=)"))
    return None, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="reference first")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    ref = a.tags[0]

    print("Decision-Phase estimate: regime folds, accuracy-only, extrapolated to "
          "100% of the release\n")
    print(f"{'tag':>16}{'angle 70.5%':>13}{'Re 88.9%':>11}{'slope':>9}"
          f"{'-> 100%':>10}   basis")
    for t in a.tags[1:]:
        da, na = acc_delta(ref, t, "angle", a.seeds)
        dr, nr = acc_delta(ref, t, "re", a.seeds)
        if da is None and dr is None:
            print(f"{t:>16}   no paired folds"); continue
        if da is not None and dr is not None:
            slope = (dr - da) / (RE_FRAC - ANGLE_FRAC)
            est = dr + slope * (1.0 - RE_FRAC)
            basis = f"both fold sets (n={na},{nr})"
        else:
            d = dr if dr is not None else da
            slope, est = float("nan"), d
            basis = ("Re folds only -- NOT extrapolated" if dr is not None
                     else "angle folds only -- NOT extrapolated")
        print(f"{t:>16}{da if da is not None else float('nan'):+13.3f}"
              f"{dr if dr is not None else float('nan'):+11.3f}"
              f"{slope:+9.2f}{est:+10.3f}   {basis}")
    print("\nCaveats that travel with every number above:")
    print("  * the extrapolation rests on one change measured at three data")
    print("    fractions; treat the sign as better supported than the magnitude.")
    print("  * fold-to-fold seed SD is ~0.025, so paired deltas resolve ~0.02-0.03.")
    print("  * this predicts the private test set, NOT the leaderboard. For the")
    print("    board use tools/board_model.py, which is uncalibrated and says so.")


if __name__ == "__main__":
    main()
