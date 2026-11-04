#!/usr/bin/env python3
"""One evaluation pipeline for both the leaderboard and the private test.

This replaces `tools/board_model.py` and `tools/final_model.py`, which existed
because the two targets looked like different questions. They are not.

**Why one pipeline -- and the hypothesis it rests on.** The simulated release covers
a complete 20 x 5 grid of (Reynolds, angle); the real release covers 81 of those 100
cells. The 19 cells with simulation and no real measurement are structured -- two
whole Reynolds columns (15225 interior, 27975 past our maximum), an AoA=5 block at
high Re, and five singletons -- and the withheld-per-angle counts 4/6/2/3/4
reproduce the released 16/14/18/17/16 exactly. RealPDEBench's protocol is n for
training and (N-n)/2 each for validation and test; with N=100 and n=81 that is 9.5.
*If* Codabench follows it, the hidden development set and the private test set are
two halves of the same 19 withheld cells, and an instrument that predicts one
predicts the other.

That last step is a **hypothesis, not a finding** (corrected 2026-09-01). It was
stated here as fact for ten days. The official materials say only that the split is
unpublished and that private evaluation uses unseen regimes; 9.5 is not an
allocation; and `splits.py` separately expects novel intermediate angles that the
19-cell reconstruction does not contain. Everything downstream -- the mimic folds,
this tool's screening, the extrapolation-distance arguments -- is conditional on it.
If the real geometry differs, tuning to this pattern is tuning to an artefact.

That is why `tools/kfold.py --fold-set mimic` exists: four folds, each withholding
ten cells shaped like the hypothesised set, training on 71 of 81. It is the only
fold set built from an explicit model of what might be held out, and on the
`--mults 1,2` back-test it is the closest of four instruments (mimic +0.004 against
the board's -0.006, where the angle folds said +0.062 and the tail holdout -0.111).

**Read the accuracy column, not the weighted one.** `evaluate.subscores` draws a
constant 0.15-sigma band, not the learned asymmetric head the model ships; it
over-credits accuracy gains and supplied 60-75% of every fold headline quoted before
2026-08-20.

**Status.** Four back-tests exist (`BACKTEST` below) and the tool prints them on
every run. One is at noise, one is confounded with the combiner, one is the point the
transfer coefficient was fitted on, and one -- the 2026-08-12 loss correction, run in
reverse -- is genuinely held out. Their accuracy totals transfer at 1.66 / 0.52 /
2.19 / 1.74, so there is no coefficient: this screens candidates on sign and
ordering, and it does not project a score. (The third of those totals was 3.29 until
2026-09-01, when the head that produced its local arm was found to have trained on
its own evaluation tail and the pair was remeasured -- `docs/archive/FINDINGS_2026-09-10.md` section 0d.)

    python tools/predict_score.py mimbase mimm12 mimdist
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)

sys.path.insert(0, os.path.join(HERE, "..", "src"))
from realistix import board  # noqa: E402

# Changes measured both ways. The board side is **computed** from the recorded
# subscores by `realistix.board.acc_delta`, never transcribed -- the three tables
# that used to quote these numbers had drifted into three different conventions, one
# of which (sps included on the board side, excluded on the local side) manufactured
# the recorded verdict "w_rel 16: 2.4x too small". Matched, that ratio is 1.70.
#
# `acc_delta` excludes sps, which is the convention this tool's local column uses,
# because a fold's sps is the constant 0.15-sigma band and not the learned head.
# (submission, reference, note). The direction must match the local arm: a tag that
# is the *old* setting relative to `mimbase` needs the board pair the same way round.
BACKTEST = {
    "mimm12":  (891698, 887158, "--mults 1,2 -- both sides inside noise, untestable"),
    "mimdist": (894861, 887158, "distillation -- FAILS, wrong sign. CONFOUNDED with "
                                "the combiner, which shipped in the same submission"),
    "mimrw16": (895690, 887158, "w_rel 16 -- right sign; the transfer coefficient is "
                                "fitted here, so this is not a held-out test"),
    # The 2026-08-12 loss correction, run backwards: `mimoldloss` carries w_rel 0.5 /
    # w_mvpe 0.2 against the base's 8.0 / 0.0, so the board pair is 884969 (last
    # submission on the old loss, same 3 members and w96 head) against 887158.
    # 887158 also dropped 7575_0, which the fold baseline excludes anyway -- so the
    # local arm isolates the loss and the board arm carries one extra trajectory's
    # worth of confound out of 82.
    "mimoldloss": (884969, 887158, "the loss correction run in reverse -- the largest "
                                   "isolated accuracy gain in the history"),
}
BOARD = {t: board.acc_delta(s, r) for t, (s, r, _) in BACKTEST.items()}

# Per-subscore transfer, from the one back-test with a board effect large enough to
# resolve (895690, w_rel 16 against 8):
#
#   rel_l2   mimic +0.079  board +0.074   ratio 0.94   <- the highest-weight subscore,
#                                                         predicted to 7%
#   tke      mimic -0.957  board -1.350   ratio 1.41
#   mvpe     mimic -0.005  board +0.013   both at noise
#   sps      mimic +0.130  board -0.180   WRONG SIGN -- the constant 0.15-sigma band,
#                                         not the learned head; do not read it
#
# ...and that was a calibration from ONE change. Two more landed on 2026-08-21:
#
#                          rel_l2   tke   mvpe   sps   accuracy total
#   w_rel 16   (mimic)       0.94  1.41      .     -        1.66
#   loss corr  (mimic)       0.68  0.91      .     -        0.52   <- held out
#   1 member   (honest)      1.14 -0.40   2.67  1.91        2.19   <- REMEASURED
#   2m + w48   (honest)         .     .      .     .           .    <- REMEASURED
#                                       ( . = local move below the 0.05 floor )
#
# REMEASURED 2026-09-01, and both rows moved. `train_final_head.py` ignored the tail
# holdout when building its training sampler until that date, so these heads trained
# on the tail `val_ingrid` then scored them on -- inflating the local sps that sits in
# each ratio's denominator. Rebuilt with the fix (members untouched, accuracy
# bit-identical), the inflation was +0.155 sps at 1 member, +0.083 at 2 and +0.048 at
# 3: largest exactly where the head had least legitimate spread signal to read.
#
# So the 1-member sps transfer is **1.91**, not 3.50, and its weighted total 2.19,
# not 3.29. The 2m+w48 row's local sps move grows past the floor in the same
# direction but its board pair is confounded with a head-width change, so it is no
# longer quoted as a calibration at all. `docs/archive/FINDINGS_2026-09-10.md` section 0d has the table.
#
# The totals still span 3-4x and tke changes SIGN. There is no single coefficient.
# What survives: **rel_l2 transfers at 0.68-1.14 wherever it moved enough to
# measure**, on both instruments, and it carries 0.474 of the objective. sps has
# exactly ONE clean observation after the 2026-09-01 remeasurement -- the 1-member
# change, at **1.91** -- so treat it as "member-count changes amplify sps about
# twofold" and not as a fitted constant. Predict rel_l2, read the sign of the rest,
# and never sum-and-scale.
#
# And these are all PER-MEMBER numbers: this rig averages per-seed scores while the
# board scores three members. Measured 2026-08-21 with tools/fold_ensemble.py, over
# the four mimic folds at three seeds:
#
#                  per-member   ensemble   board
#   --mults 1,2        +0.005     -0.053  -0.006   <- sign only right as an ensemble
#   distillation       +0.035     +0.020  -0.146   <- halved, still wrong sign
#   w_rel 16           -0.066     -0.086  -0.106
#   loss correction    -0.128     -0.175  -0.069
#
# For a capacity change, per-member scoring gets the SIGN wrong: the baseline gains
# from ensembling on 4/4 folds and --mults 1,2 loses on 4/4. Score the ensemble for
# anything that touches capacity or member diversity. It is not a calibration --
# the transfer spread widens from 3x to 11x -- and it does not rescue distillation.
#
# The open lead: the board number is a three-member ensemble and the mimic number is
# a single model -- `kfold.py --report` averages per-seed scores rather than scoring
# their ensemble, so this rig has never measured one. HANDOFF section 7 records
# ensembling halving a per-member distillation gain, which is the right direction to
# explain the over-prediction. Scoring the folds as ensembles is the cheapest thing
# left to try.
# 894861 shipped distillation *and* a learned combiner together, and every fold
# measurement of distillation omits the combiner. Four honest instruments -- angle
# folds, Re folds, tail holdout, mimic folds -- put distillation at +0.03 to +0.16;
# the board put the bundle at -0.146 on accuracy under the fourteen-point weights,
# of which 48% is MVPE. The combiner interaction was measured on 2026-08-20 at
# -0.014, a tenth of the gap, so it is not the explanation; distillation is anomalous
# in a way no local split reproduces.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tags", nargs="+", help="reference first")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--fold-set", default="mimic",
                    help="'mimic' is the one that models the withheld cells; 're' and "
                         "'angle' are kept for the archived comparisons only")
    a = ap.parse_args()

    out = subprocess.run(
        [PY, os.path.join(HERE, "fold_subscores.py"), *a.tags,
         "--fold-set", a.fold_set, "--seeds", a.seeds],
        capture_output=True, text=True)
    print(out.stdout.rstrip())
    if out.returncode:
        # Exit non-zero. Returning here made a crashed fold_subscores.py look like a
        # successful prediction run with missing arms, which an automated study
        # records as a result.
        print(out.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(f"fold_subscores.py exited {out.returncode}")

    rows = {}
    malformed = []
    for line in out.stdout.splitlines():
        p = line.split()
        if len(p) > 6 and p[0] in a.tags[1:] and p[-1].startswith(("(n=", "UNFINISHED")):
            try:
                rows[p[0]] = float(p[-2] if p[-1].startswith("(n=") else p[-4])
            except ValueError:
                # Silently dropping these turned a format change in fold_subscores.py
                # into a table of "[not run here]" rows that looked like missing runs.
                malformed.append(line)
    if malformed:
        print("\nwarning: could not parse these result lines -- the fold_subscores "
              "output format may have changed:", file=sys.stderr)
        for line in malformed:
            print(f"  {line}", file=sys.stderr)

    print("\ntrack record on this fold set (board side computed by "
          "realistix.board.acc_delta,\nsps excluded on BOTH sides so the two columns "
          "are comparable):")
    print(f"  {'change':>12}{'local':>9}{'board':>9}{'transfer':>10}")
    for t, (_sub, _ref, what) in BACKTEST.items():
        v = rows.get(t)
        if v is None:
            print(f"  {t:>12}{'-':>9}{BOARD[t]:+9.3f}{'-':>10}   {what}  [not run here]")
            continue
        # Never divide by a local move that sits at the instrument's own noise --
        # that is the "range, not slope" error this project has made four times, and
        # it is how a 2m+w48 back-test produced an apparent rel_l2 transfer of 7.8
        # off a local move of 0.015. Below the floor, print the delta, not a ratio.
        FLOOR = 0.05
        r = (f"{BOARD[t] / v:>10.2f}" if abs(v) >= FLOOR else f"{'below floor':>10}")
        print(f"  {t:>12}{v:+9.3f}{BOARD[t]:+9.3f}{r}   {what}")
    print("\n  The transfer column is the whole calibration: 1.66, 0.52, 3.29 and")
    print("  1.74 over the four paired observations -- the last two carry an sps")
    print("  arm from a head that trained on its own evaluation tail, retracted")
    print("  2026-09-01. No coefficient to apply, and fewer clean points than four.")
    print("  Screen on the rel_l2 column, which transfers at 0.68-1.14 wherever it")
    print("  moved enough to measure; read the sign of the rest, never sum-and-scale.")
    print(f"\n  The board reproduced itself to {board.NOISE_ACC:.3f} of accuracy "
          f"(892587, same recipe\n  retrained) -- ONE retrain, so treat it as the "
          "smallest difference worth\n  arguing about, not as a measured floor "
          "that nothing can beat.")
    print("\n  Read the accuracy column, not the weighted one: the latter carries a")
    print("  constant-band SPS proxy that got the SIGN wrong on the one change big")
    print("  enough to check (mimic +0.130, board -0.180).")
    print("  Per-subscore transfer over the clean points: rel_l2 0.94/0.68/1.14 --")
    print("  stable, and 0.474 of the objective. tke 1.41/0.91/-0.40 -- sign unstable.")
    print("  Resolution is ~0.02-0.03; treat anything under +0.05 as not a result.")
    print("  IF the hidden set is one fixed draw of ~10 cells (a hypothesis -- see the")
    print("  module docstring) and fold-to-fold delta SD is ~0.03, ~0.05 on the board")
    print("  is a floor no calibration can beat.")
    print("  MVPE is the blind spot: it has moved 0.033 at most here against 0.761 on")
    print("  the board, and its weight tripled on 2026-08-20. A change that touches")
    print("  MVPE is unpredicted -- which is most of what distillation was.")


if __name__ == "__main__":
    main()
