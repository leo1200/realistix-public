#!/usr/bin/env python3
"""What of the submission history can be turned into a (local, board) pair, and how.

The bottleneck on predicting a board score is not the method, it is that only
**one** paired observation has ever been usable for a magnitude. Fifteen submissions
have been scored; three have local counterparts; of those three one is at noise, one
is confounded, and one is the point the transfer coefficient was fitted on.

Paired observations are cheap in the way that matters: they cost GPU time and **no
submissions**. This module is the inventory of which ones are still available, which
instrument can express each, and which cannot be expressed at all -- because that
last category is larger than it looks and is why "re-screen the archive" yields
about five points and not fifteen.

    python tools/rescreen_archive.py              # the inventory
    python tools/rescreen_archive.py --launch     # submit what is runnable

**Why most transitions are not fold-able.** `kfold.py` trains one model per fold on
a training set the fold defines. So it can express any change to `train_mixed.py`'s
arguments -- losses, schedules, conditioning, capacity -- and it cannot express a
change in ensemble size, in interval-head configuration, or in how much data the
model sees, because those are respectively post-training, post-training, and fixed
by the fold. `honest_pipeline.sh` builds and scores a packed artefact, so it can
express all three; it is slower and it has its own uncalibrated status.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
PY = os.environ.get("PY", sys.executable)

from realistix import board  # noqa: E402

# (sub, ref, what changed, instrument, tag, kfold --args, status)
#   instrument: "mimic"  -- kfold.py --fold-set mimic
#               "honest" -- honest_pipeline.sh / backtest_pack.sh
#               None     -- no local instrument can express it; the reason is the tag
PLAN = [
    (891698, 887158, "--mults 1,2", "mimic", "mimm12", "--mults 1,2", "done"),
    (895690, 887158, "w_rel 16", "mimic", "mimrw16", "--w-rel 16.0", "done"),
    (894861, 887158, "distillation + combiner", "mimic", "mimdist", None,
     "done, and confounded: two changes in one submission"),
    (884969, 887158, "the loss correction, reversed", "mimic", "mimoldloss",
     "--w-rel 0.5 --w-mvpe 0.2",
     "DONE 2026-08-21 -- local -0.131 against board -0.069, transfer 0.52"),
    (893273, 887158, "1 member instead of 3", "honest", "hp_b1",
     None, "REMEASURED 2026-09-01 -- local -0.081 against board -0.177, transfer "
           "2.19; sps 1.91 (was 3.29/3.50 off a head that had trained on its own "
           "evaluation tail). FINDINGS 0d"),
    # The whole effect of this one is in SPS (-0.976 of it), which the mimic folds
    # cannot see -- their sps is the constant band and got the sign wrong on the one
    # change big enough to check. The honest pipeline scores a packed artefact
    # through the learned head, so it is the only instrument that could calibrate
    # SPS at all, and this is the largest SPS move in the history.
    (886080, 884969, "2 members + w48 head (an SPS point)", "honest", "hp_b2w48",
     None, "REMEASURED 2026-09-01 -- the tail-head leak inflated its sps by 0.083. "
           "Its board pair also mixes a member-count and a head-width change, so "
           "it is not quoted as an sps calibration. FINDINGS 0d"),
    (892587, 887158, "retrain, same recipe", None, None, None,
     "the null. No local counterpart needed -- it is the closest thing to a noise "
     "reference, though one retrain cannot estimate a floor"),
    (891095, 887158, "inference code only", None, None, None,
     "accuracy null; the change is not expressible in training at all"),
    (882969, 882102, "all-82 real + spread-fed head", None, None, None,
     "training-set size is fixed by the fold; the head is post-training"),
    (881057, 879743, "3-member ensemble + learned intervals", None, None, None,
     "ensemble size and head are both post-training"),
    (883878, 882969, "domain-specific norm, unfolded", None, None, None,
     "board accuracy delta is +0.004 -- too small to calibrate against"),
    (879743, 879338, "FNO -> U-Net", None, None, None,
     "a different model family; reproducing it measures today's code, not 08-05"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="submit the runnable screens")
    ap.add_argument("--pq-type", default="a100")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()

    usable = 0
    for group, title in (("mimic", "mimic folds -- kfold.py --fold-set mimic"),
                         ("honest", "honest pipeline -- packed artefact, tail holdout"),
                         (None, "NO local instrument can express these")):
        rows = [r for r in PLAN if r[3] == group]
        if not rows:
            continue
        print(f"\n=== {title} ===")
        for sub, ref, what, _, tag, args, status in rows:
            # Match the convention to the instrument. A fold's sps is the constant
            # 0.15-sigma band, so it is excluded; the honest pipeline scores a packed
            # artefact through the learned head, so its sps is comparable and stays.
            sps = group == "honest"
            d = board.acc_delta(sub, ref, with_sps=sps)
            flag = "  " if abs(d) > board.NOISE_ACC * 2 else " ~"   # ~ = inside noise
            print(f"{flag}{sub} vs {ref}  {what:34.34s} board {d:+.4f}"
                  f"{'  (sps included)' if sps else ''}")
            print(f"    {status}")
            if group is not None and abs(d) > board.NOISE_ACC * 2:
                usable += 1
            if group == "mimic" and args and status.startswith("runnable"):
                if a.launch:
                    cmd = [PY, os.path.join(HERE, "kfold.py"), "--fold-set", "mimic",
                           "--tag", tag, "--seeds", a.seeds, f"--args={args}",
                           "--pq", "--pq-type", a.pq_type]
                    print("    launching: " + " ".join(cmd))
                    subprocess.run(cmd, check=False)

    print(f"\n{usable} paired observations with a board effect above twice the "
          f"{board.NOISE_ACC:.3f} noise floor. Before 2026-08-21 there was ONE.")
    print("\nTransfer on the four paired points: 1.66, 0.52, 3.29, 1.74 -- no")
    print("(the last two were remeasured 2026-09-01; see the row notes)")
    print("coefficient. rel_l2 alone transfers at 0.68-1.14; screen on that, and")
    print("never take a ratio whose local move is under 0.05.")
    print("\nThe honest ceiling on this exercise: the history bundles most of its")
    print("changes, so re-screening buys about five usable points and not fifteen.")
    print("Beyond that, paired observations only arrive one per submission per day --")
    print("which is the argument for spending each one on a single isolated variable")
    print("with its prediction recorded in advance.")


if __name__ == "__main__":
    main()
