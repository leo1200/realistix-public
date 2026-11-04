#!/usr/bin/env python3
"""Cross-validate a configuration over held-out *regimes*, not over windows.

The reason this exists. Every number in the project is measured on one split:
`val_aoa` is 175 windows from 9 trajectories, `val_re` is 159 from 8. Windows
inside a trajectory are strongly correlated, so the effective sample size is the
number of held-out *regimes* -- 8 or 9 -- and not the window count. Error bars
quoted across seeds at a fixed split hold the largest variance term constant and
read tighter than they are.

What this rig resolves, corrected 2026-08-17. A per-lead-time width calibration --
a null effect -- scored +0.043 where it was fitted and -0.054 on the other split,
and that 0.10 swing was read here as "the resolution of the apparatus". It is not:
it is what a *fitted* post-hoc parameter does to a single 175-window split, which
is a third failure mode and not this one. Measured instead across 7 folds at 2
seeds, the within-fold seed SD of a run is **0.025**, so a paired A/B on fixed
folds resolves about **0.02-0.03**. Precision was never the problem. Generalisation
is: the fold means span 73.40 to 75.85, so whether a difference found on one
holdout survives another is the question this rig exists to answer, and the reason
to read the spread of the *differences* rather than any single fold.

So rotate the holdout. `splits.py` reads `REALISTIX_HELDOUT_AOA` and
`REALISTIX_HELDOUT_RE` from the environment, which is all the machinery needed:
each fold trains on everything except one regime slice and scores on it. A
candidate is then compared to the baseline **fold by fold, paired on the fold**,
and the uncertainty comes from the spread across folds -- the term that actually
moves. Paired differencing is what makes this affordable: the fold-to-fold spread
of the *scores* is large (an interior angle scores ~5 points above 20 degrees), while
the spread of the *difference* is small.

    # once, to establish the reference
    python tools/kfold.py --tag base --seeds 0,1
    # then per candidate, same folds and seeds
    python tools/kfold.py --tag relup --seeds 0,1 --args "--w-rel 16.0"
    python tools/kfold.py --report base relup

Costs, at ~12 min per 8000-step run on an idle H100: 7 folds x 2 seeds = 14 runs,
so about 3 GPU-hours per configuration, or 45 minutes on the four cards. That is
the price of being able to tell +0.1 from 0, and nothing else on the list is worth
running until it is paid.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from realistix.weights import ACC, ACC_NO_SPS  # noqa: E402

RUNS = os.environ.get("REALISTIX_RUNS", "runs")
PY = os.environ.get("PY", "python")
HERE = os.path.dirname(os.path.abspath(__file__))

# The identified weights, from `realistix.weights` so there is one copy of the fit.
# `time` is a property of the packed submission, not of a training run, so it is out.
#
# Two weightings, and the report prints a delta under each. `W` keeps `sps_score`,
# which is what every archived number in docs/archive/HANDOFF_2026-09-10.md and FINDINGS.md was computed
# with, so the levels below stay comparable to the record. But that sps comes from
# `evaluate.subscores`' constant 0.15-sigma band rather than the learned asymmetric
# head the submission ships, and the proxy rewards better-centred predictions
# directly -- it supplied 60-75% of every fold headline quoted before 2026-08-20.
# `W_ACC` drops it. **Read the accuracy-only delta** unless the change is known not
# to touch accuracy.
W = ACC
W_ACC = ACC_NO_SPS

# Seven folds, each holding out one whole angle *and* one disjoint pair of Reynolds
# numbers -- which is what the private set does ("novel angles of attack or Reynolds
# numbers"), and also what keeps every split non-empty: `partition` always fills
# val_aoa/test_aoa from the held-out angle and val_re from the held-out Re, so a fold
# that dropped either axis would evaluate on nothing.
#
# Angles 0 and 20 sit at the ends of the grid and are extrapolation; 5/10/15 are
# interpolation, the likelier reading of "novel angles" on a 5-degree grid. The Re
# pairs are disjoint across folds, and angles 10 and 20 appear twice against
# different pairs so there is some replication at the fold level.
FOLDS = [
    dict(name="a00_re1", env=dict(REALISTIX_HELDOUT_AOA="0", REALISTIX_HELDOUT_RE="13950,21600")),
    dict(name="a05_re2", env=dict(REALISTIX_HELDOUT_AOA="5", REALISTIX_HELDOUT_RE="8850,17775")),
    dict(name="a10_re3", env=dict(REALISTIX_HELDOUT_AOA="10", REALISTIX_HELDOUT_RE="6300,19050")),
    dict(name="a15_re4", env=dict(REALISTIX_HELDOUT_AOA="15", REALISTIX_HELDOUT_RE="11400,20325")),
    dict(name="a20_re5", env=dict(REALISTIX_HELDOUT_AOA="20", REALISTIX_HELDOUT_RE="5025,24150")),
    dict(name="a10_re6", env=dict(REALISTIX_HELDOUT_AOA="10", REALISTIX_HELDOUT_RE="10125,22875")),
    dict(name="a20_re7", env=dict(REALISTIX_HELDOUT_AOA="20", REALISTIX_HELDOUT_RE="12675,16500")),
]

# The Re-only fold set, added 2026-08-19. The seven folds above each remove a whole
# angle *and* a Reynolds pair -- about 29% of the 81 real trajectories -- so a change
# that trades capacity for regularisation flatters itself there. That is not a
# hypothetical: `--mults 1,2` read +0.062 on 7/7 folds and -0.013 on the board, and
# the oracle-step explanation for the disagreement was ruled out on 2026-08-18 (the
# fixed-step scorer gives +0.068, 13/14). The remaining explanation is the training
# set size, and the only way to test it is a fold that removes less.
#
# Holding out a Reynolds pair alone removes 8-10 trajectories, i.e. 10-12%, so these
# folds train on 88-90% of what the shipped model sees against the angle folds' 71%.
# "Novel Reynolds numbers" is half of what the private set varies, so this is a
# legitimate proxy in its own right and not merely a capacity control.
#
# The 18 Reynolds numbers are paired low-with-high (i with i+9) so each fold spans
# the range rather than removing a contiguous block, and the nine pairs are disjoint
# and exhaustive. Directory names carry the pair, because a run directory that does
# not name its own study has already caused one study to read another's runs.
_RE_PAIRS = [(3750, 16500), (5025, 17775), (6300, 19050), (7575, 20325),
             (8850, 21600), (10125, 22875), (11400, 24150), (12675, 25425),
             (13950, 26700)]
RE_FOLDS = [dict(name=f"rp{i+1}_{lo}_{hi}",
                 env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE=f"{lo},{hi}"))
            for i, (lo, hi) in enumerate(_RE_PAIRS)]

# The "mimic" fold set, added 2026-08-20. It holds out *cells* chosen to have the
# same shape as the 19 the organisers withheld: one interior Reynolds column, a
# range-edge column, a small AoA block at high Re, and a singleton. Ten cells per
# fold, so training keeps 71 of 81 -- 88% of the release. That is close to the Re
# folds' 89% and well short of the 100% a shipped model trains at, which matters
# for any capacity change (see the linear-in-training-fraction result in
# FINDINGS.md section 7); it is *not* comparable to the 81/100 of the grid the real
# release covers, which is a fraction of a different thing.
#
# This is the only fold set whose *structure* matches what the board is *hypothesised*
# to score (`docs/archive/FINDINGS_2026-09-10.md` §0a -- inferred from the release, not published, so read
# this paragraph as conditional throughout).
# The angle folds hold out a whole angle (the hypothesised hidden set never does), the Re
# folds hold out a whole Reynolds pair (it does that for two columns out of
# nineteen cells), and the in-grid tails hold out later frames of trajectories the
# model trained on (it never does that at all). See splits.HELDOUT_CELLS.
def _cells(*groups):
    return ",".join(c for g in groups for c in g)


def _col(re_, aoas=(0, 5, 10, 15, 20)):
    return [f"{re_}_{a}" for a in aoas]


MIMIC_FOLDS = [
    dict(name="mim1", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_cells(_col(13950), _col(26700, (0, 10, 15)),
                                        ["21600_5", "20325_5"]))),
    dict(name="mim2", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_cells(_col(11400), _col(3750, (0, 10, 20)),
                                        ["19050_5", "16500_5"]))),
    dict(name="mim3", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_cells(_col(17775, (5, 10, 15, 20)), _col(25425, (0, 10, 15)),
                                        ["12675_5", "11400_20", "6300_0"]))),
    dict(name="mim4", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_cells(_col(8850), _col(24150, (0, 10, 15)),
                                        ["10125_5", "22875_10"]))),
]

# The far-edge fold, added 2026-08-21 after the per-cell decomposition
# (`tools/cell_report.py`). The mimic folds test extrapolation one grid step past a
# reduced maximum; the hidden set's edge sits harder than that -- the 27975 column is
# past ALL real data, and the AoA=5 block puts its highest cell four steps beyond the
# last real AoA=5 measurement. Per-cell penalties are 3-5x larger at range edges and
# concentrate in MVPE, the subscore the board moved 23x further than any local
# instrument -- so severity-of-extrapolation is the one axis the local rigs have
# never varied. This fold removes the top three Reynolds columns entirely (max
# trained Re drops to 22875) and therefore scores 24150 at one step past the edge,
# 25425 at two and 26700 at three: an extrapolation-distance dose-response in a
# single fold. Read it per cell via cell_report.py, never pooled -- pooling is
# exactly what hid the edge behaviour in the first place.
FAR_FOLDS = [
    dict(name="far1", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_cells(_col(24150, (0, 10, 15)),
                                        _col(25425, (0, 10, 15, 20)),
                                        _col(26700, (0, 10, 15))))),
]

# Novel-angle extrapolation at 20 deg (2026-11-02, research/rb_2026-11-02): every real
# trajectory at 20 deg is scored; every Re is listed so that simulation without a real
# partner at 20 deg is removed too. `a20` trains with no simulation at the target angle,
# `a20k` keeps all of it (REALISTIX_SIM_KEEPS_CELLS): their difference is what
# simulation *at the target* is worth. Combine with --real-drop-aoa for the coverage dose.
_A20 = _cells([f"{3750 + 1275 * i}_20" for i in range(20)])
THIN_FOLDS = [
    dict(name="a20", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_A20, REALISTIX_SIM_KEEPS_CELLS="")),
    dict(name="a20k", env=dict(REALISTIX_HELDOUT_AOA="", REALISTIX_HELDOUT_RE="",
         REALISTIX_HELDOUT_CELLS=_A20, REALISTIX_SIM_KEEPS_CELLS="1")),
]

FOLD_SETS = {"angle": FOLDS, "re": RE_FOLDS, "mimic": MIMIC_FOLDS, "far": FAR_FOLDS,
             "thin": THIN_FOLDS}


# Archived reference recipe. Keep it stable so historical experiment commands replay.
BASE = ("--mode plus_sim --film --cond norm --width 96 --ema 0.999 --w-tmean 0.5 "
        "--w-rel 8.0 --w-mvpe 0.0 --steps 8000 --eval-every 1000")
CURRENT_BASE = (BASE + " --opt muon --lr-muon 0.02 --muon-momentum 0.95"
                " --ema 0.998 --w-sps 1.0 --w-tke 1.0 --w-mvpe 0.3 --wd-muon 0.03")


def fold_score(row: dict, w: dict = None) -> float:
    """One number per eval: the two shift types weighted equally.

    The held-out angle supplies val_aoa and test_aoa (split by Reynolds number, so
    both are out of sample for the fold, so pooling them adds evidence -- at the
    cost of test_aoa being selected on, since this score is maximised over
    snapshots; see `score_of` below); the held-out
    Reynolds pair supplies val_re. The angle side carries ~4x the trajectories, so a
    flat mean over the three splits would quietly make this an angle benchmark. The
    private set varies both, so they are weighted 50/50.

    An Re-only fold populates val_re alone, so the 50/50 weighting has nothing to
    weight and the score is that split. Written as "use the splits that exist"
    rather than branching on the fold set, so a fold cannot silently be scored on
    a split it never held out.
    """
    wt = W if w is None else w
    sc = lambda s: sum(v * row[f"{s}.{k}"] for k, v in wt.items())
    have = lambda s: f"{s}.rel_l2_score" in row
    angle = [sc(s) for s in ("val_aoa", "test_aoa") if have(s)]
    re_side = [sc("val_re")] if have("val_re") else []
    if angle and re_side:
        return 0.5 * float(np.mean(angle)) + 0.5 * float(np.mean(re_side))
    both = angle + re_side
    if not both:
        raise KeyError("no scored split in this row")
    return float(np.mean(both))


# A run is only readable once it has reached the end of its schedule. Without this
# the report happily scores a run that is still training: `kf_blend_a10_re3_s0` was
# at step 2000 of 8000 when it was read as 72.921 against a baseline's 74.657, and
# that single in-flight fold turned a +0.028 screen into a confident "-0.340 +/-
# 0.349, inside noise". Nothing in the output said the run was unfinished -- an
# incomplete run and a bad configuration are indistinguishable when the apparatus
# takes a max over whatever happens to be on disk.
FINAL_STEP = 8000


def score_of(run_dir: str, fold: dict, w: dict = None) -> float | None:
    """Best fold score over the run's evals, or None if the run has not finished.

    Selecting the step on the same quantity that is then reported is optimistic in
    *level* -- but every configuration gets the identical treatment and the report
    below is a paired difference, so the bias cancels. It is the level that must not
    be quoted, not the difference. (Measured 2026-08-18 on 14 archived runs: the max
    sits +0.020 above a fixed 5000/6000/7000 window, and the two agree to 0.006 on
    every archived comparison, so the bias is real and has never changed a
    conclusion. `optuna_search.py` scores at the fixed window regardless.)

    Two things that argument does not cover (added 2026-09-01):

    * `fold_score` pools `test_aoa` with `val_aoa`, so this max is taken over a
      quantity that includes the split described elsewhere as untouched. On 112
      archived three-split histories, including `test_aoa` changed the selected
      checkpoint in 13 of them (75 against validation-only selection). There is no
      untouched confirmation split behind these folds; treat `test_aoa` as a second
      validation split.
    * The bias cancels within one paired comparison. It does not cancel across the
      hundreds of screening decisions these same four folds have carried -- 111 TPE
      trials and 41 joint loss configurations among them. Effects at or below the
      fold resolution (~0.02-0.03) that were *found* by searching here are selected
      quantities, and the folds cannot then confirm them. Only a fresh instrument or
      the board can.
    """
    p = os.path.join(run_dir, "history.json")
    if not os.path.exists(p):
        return None
    h = json.load(open(p))
    if any(r.get("metric_object", "member") != "member" for r in h):
        raise ValueError(f"{run_dir}: history scores a cooperative ensemble, not a member; "
                         "compare it against the same fixed companions explicitly")
    # The bar is the run's *own* schedule where it recorded one, not the 8000-step
    # default: `_finished` already derives it from --steps, and hardcoding FINAL_STEP
    # here let a 12k or 16k arm be reported from a truncated 8k history as if it had
    # finished. Fall back to FINAL_STEP for checkpoints that predate recorded args.
    want = FINAL_STEP
    args_p = os.path.join(run_dir, "args.json")
    if os.path.exists(args_p):
        try:
            want = int(json.load(open(args_p)).get("steps", FINAL_STEP))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if not h or max(r["step"] for r in h) < want:
        return None                       # still training; not a result yet
    try:
        return max(fold_score(row, w) for row in h)
    except KeyError:                      # a split was empty for this fold
        return None


def _finished(out: str, args: str, env: dict | None = None) -> bool:
    """Has this run trained to completion, as opposed to merely started?

    `history.json` is rewritten at every eval, so a run killed at step 3000 leaves
    one behind and the old existence check treated it as done -- silently returning
    a truncated model, and a `best.pt` selected over a third of the schedule. That is
    the worst failure this rig has, because it looks exactly like a result. Two
    drivers were killed and restarted on 2026-08-24 alone; the check is on the last
    recorded step now, against whatever `--steps` the arm actually asked for.
    """
    p = os.path.join(out, "history.json")
    if not os.path.exists(p) or not os.path.exists(os.path.join(out, "best.pt")):
        return False
    try:
        with open(p) as stream:
            h = json.load(stream)
    except (OSError, ValueError):
        return False                                   # unreadable: retrain it
    steps = re.findall(r"--steps(?:\s+|=)(\d+)", f"{BASE} {args}")
    want = int(steps[-1])
    if not (bool(h) and max(r["step"] for r in h) >= want):
        return False

    # Reaching the step count is not the same as being *this* run. The check used to
    # stop here, so requesting a tag that already exists with different flags skipped
    # training and later reported the OLD configuration's numbers under the NEW name
    # -- demonstrated by creating an 8000-step run at w_rel 8 and asking for the same
    # tag at w_rel 16, which this function happily called finished. Compare the recipe
    # the run recorded against the one being asked for.
    ap = os.path.join(out, "args.json")
    if not os.path.exists(ap):
        print(f"  warning: {out} has no args.json (trained before 2026-09-01); "
              f"cannot confirm it was built from the requested recipe", flush=True)
        return True
    try:
        with open(ap) as stream:
            rec = json.load(stream)
    except (OSError, ValueError):
        return False
    want_flags, mismatched = shlex.split(f"{BASE} {args}"), []
    requested = {}
    i = 0
    while i < len(want_flags):
        tok = want_flags[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].split("=")[0].replace("-", "_")
        if "=" in tok:
            val = tok.split("=", 1)[1]
        elif i + 1 < len(want_flags) and not want_flags[i + 1].startswith("--"):
            val, i = want_flags[i + 1], i + 1
        else:
            val = True                                   # a store_true flag
        # Match argparse: the LAST occurrence wins. Comparing BASE's --ema .999
        # before a candidate's --ema .998 rejected valid resumptions of that run.
        requested[key] = val
        i += 1
    if requested.get("no_film"):
        requested["film"] = False      # train_mixed's post-parse override
    for key, val in requested.items():
        got = rec.get(key, "<absent>")
        if val is True:
            if not got:
                mismatched.append(f"--{key.replace('_','-')}: requested, run has {got!r}")
        else:
            try:
                ok = abs(float(got) - float(val)) <= 1e-9 * max(1.0, abs(float(val)))
            except (TypeError, ValueError):
                ok = str(got) == str(val)
            if not ok:
                mismatched.append(f"--{key.replace('_','-')}: requested {val!r}, "
                                  f"run has {got!r}")
    if env is not None:
        for key in ("REALISTIX_HELDOUT_AOA", "REALISTIX_HELDOUT_RE", "REALISTIX_HELDOUT_CELLS"):
            want = env.get(key, "")
            if key in rec and rec[key] != want:
                mismatched.append(f"{key}: requested {want!r}, run has {rec[key]!r}")
    if mismatched:
        print(f"  {out} exists but was built from a different recipe:", flush=True)
        for msg in mismatched:
            print(f"    {msg}", flush=True)
        print("    pick a new --tag, or delete it to rebuild", flush=True)
        raise SystemExit(1)
    return True


def main() -> None:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="name for this configuration")
    ap.add_argument("--recipe", choices=("archive", "current"), default="archive",
                    help="archive preserves old commands; current uses the 2026-09-08 "
                         "member recipe before applying --args")
    ap.add_argument("--args", default="",
                    help="extra train_mixed.py flags; '{fold}' expands to the fold "
                         "name, which is what a per-fold teacher path needs")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--fold-set", default="angle", choices=sorted(FOLD_SETS),
                    help="'angle' holds out a whole angle plus an Re pair (~29%% of "
                         "the trajectories); 're' holds out an Re pair alone (~12%%), "
                         "which is the set to use for any capacity change")
    ap.add_argument("--folds", default="", help="subset of the fold set, by name")
    ap.add_argument("--report", nargs="*", help="tags to compare instead of training")
    ap.add_argument("--pq", action="store_true",
                    help="submit through the cluster scheduler instead of running "
                         "locally; pq gives one user per GPU, local cards do not")
    ap.add_argument("--pq-type", default="h100")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.recipe == "current":
        BASE = CURRENT_BASE

    all_folds = FOLD_SETS[a.fold_set]
    folds = ([f for f in all_folds if f["name"] in a.folds.split(",")]
             if a.folds else list(all_folds))
    if not folds:
        ap.error(f"no folds named {a.folds!r} in fold set {a.fold_set!r}")
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.report:
        base = a.report[0]
        print(f"paired against '{base}', by fold; the spread across folds is the "
              f"uncertainty that matters\n")
        head = f"{'fold':>7}" + "".join(f"{t:>12}" for t in a.report)
        print(head + "\n" + "-" * len(head))
        cols = {t: [] for t in a.report}
        cols_acc = {t: [] for t in a.report}
        missing = {t: [] for t in a.report}
        for f in folds:
            row = f"{f['name']:>7}"
            for t in a.report:
                runs = [os.path.join(RUNS, f"kf_{t}_{f['name']}_s{s}") for s in seeds]
                for store, w in ((cols, None), (cols_acc, W_ACC)):
                    per_seed = [score_of(r, f, w) for r in runs]
                    # All requested seeds or none. Averaging "whatever finished" made
                    # a baseline over seeds 0,1 comparable-looking against a candidate
                    # with only seed 0, and the completeness report below only ever
                    # noticed a fold with *zero* seeds. `fold_ensemble.py` already
                    # refuses this; so does this now.
                    vals = [v for v in per_seed if v is not None]
                    if len(vals) != len(runs):
                        missing[t].append(
                            f"{f['name']}({len(vals)}/{len(runs)} seeds)")
                        store[t].append(None)
                    else:
                        store[t].append(float(np.mean(vals)))
                v = cols[t][-1]
                row += f"{v:>12.3f}" if v is not None else f"{'-':>12}"
            print(row)
        print("-" * len(head))
        gaps = {t: sum(v is None for v in cols[t]) for t in a.report}
        if any(gaps.values()):
            print("\nincomplete: " + ", ".join(f"{t} missing {n} of {len(folds)} folds"
                                               for t, n in gaps.items() if n)
                  + "  -- deltas below are over the folds both sides have finished")
        if any(missing.values()):
            print("dropped for partial seeds: "
                  + "; ".join(f"{t}: {', '.join(v)}" for t, v in missing.items() if v)
                  + "\n  (a fold counts only when every requested seed has finished, so "
                    "the two\n   sides of a paired difference rest on the same runs)")

        def delta(store, t):
            d = [c - b for c, b in zip(store[t], store[base])
                 if c is not None and b is not None]
            if not d:
                return None
            d = np.array(d)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            return d.mean(), se, int((d > 0).sum()), len(d)

        print()
        for t in a.report[1:]:
            full, acc = delta(cols, t), delta(cols_acc, t)
            if acc is None:
                continue
            m, se, win, n = acc
            print(f"{t:>12}  accuracy-only delta {m:+.3f} +/- {se:.3f} over {n} folds "
                  f"({win}/{n} positive)"
                  + ("  <- inside noise" if abs(m) < 2 * se else "  <- real"))
            if full is not None:
                print(f"{'':>12}  with the sps proxy   {full[0]:+.3f} +/- {full[1]:.3f} "
                      f"({full[2]}/{full[3]} positive)  -- comparable to the archive, "
                      f"not to the board")
        # Said every time, because reading the wrong one of these two lines is the
        # single most expensive mistake this rig has produced: the sps proxy is a
        # constant band, it over-credits anything that improves centring, and it
        # inflated every headline in the documents by 3-4x before 2026-08-20.
        print("\nRead the accuracy-only line. `evaluate.subscores` draws a constant "
              "0.15-sigma\nband, not the learned head that ships, so the sps term is "
              "a proxy that rewards\naccuracy a second time.")
        return

    if not a.tag:
        ap.error("--tag is required unless --report is given")
    print(f"recipe={a.recipe}: {BASE}", flush=True)
    jobs = []
    for f, s in itertools.product(folds, seeds):
        out = os.path.join(RUNS, f"kf_{a.tag}_{f['name']}_s{s}")
        args = a.args.replace("{fold}", f["name"])
        if _finished(out, args, f["env"]):
            continue                                   # resumable: skip what finished
        # `{fold}` in --args expands to this fold's name. Distillation needs it: the
        # teacher must be trained on the *same* split as the student, so its path is
        # per-fold, and without substitution the only way to run a distilled arm was
        # four separate invocations with --folds. That is easy to get wrong in the
        # direction that leaks the holdout, which is the one mistake this rig cannot
        # afford. Everything else ignores it.
        cmd = (f"{PY} {HERE}/train_mixed.py {BASE} {args} --seed {s} --out {out}")
        jobs.append((f, cmd, out))
    print(f"{len(jobs)} runs to do for '{a.tag}' "
          f"({len(folds)} folds x {len(seeds)} seeds)", flush=True)
    if a.dry_run:
        for f, cmd, _ in jobs:
            print(f"  [{f['name']}] {cmd}")
        return

    if a.pq:
        # One user per GPU, and the whole cluster instead of this node's four cards,
        # which other users occupy directly and pq cannot see. The fold is written
        # into the command rather than left to the inherited environment: pq does
        # capture it, but a fold that silently trained on the wrong holdout would be
        # near-undetectable afterwards, so the run announces its own holdout into
        # its log and `--pq` verifies that line before reporting.
        for f, cmd, out in jobs:
            name = os.path.basename(out)
            # A fold that sets REALISTIX_SIM_KEEPS_CELLS pins it in the command too; folds
            # that do not leave it to the environment, as before.
            keep = ("" if "REALISTIX_SIM_KEEPS_CELLS" not in f["env"] else
                    f" REALISTIX_SIM_KEEPS_CELLS='{f['env']['REALISTIX_SIM_KEEPS_CELLS']}'")
            inner = (f"export REALISTIX_HELDOUT_AOA='{f['env']['REALISTIX_HELDOUT_AOA']}' "
                     f"REALISTIX_HELDOUT_RE='{f['env']['REALISTIX_HELDOUT_RE']}' "
                     f"REALISTIX_HELDOUT_CELLS='{f['env'].get('REALISTIX_HELDOUT_CELLS', '')}'"
                     f"{keep}; "
                     f"echo \"FOLD {f['name']} aoa=$REALISTIX_HELDOUT_AOA "
                     f"re=$REALISTIX_HELDOUT_RE cells=$REALISTIX_HELDOUT_CELLS "
                     f"simkeep=$REALISTIX_SIM_KEEPS_CELLS\"; "
                     f"exec {cmd}")
            r = subprocess.run(["pq", "sub", "-t", a.pq_type, "-n", "1",
                                "--name", name, "--", "bash", "-c", inner],
                               capture_output=True, text=True)
            print(f"  sub {name}: {(r.stdout or r.stderr).strip().splitlines()[-1]}",
                  flush=True)
        print(f"\nsubmitted {len(jobs)} jobs. Watch with `pq stat`; then\n"
              f"  python tools/kfold.py --report base {a.tag}")
        return

    gpus = [g.strip() for g in a.gpus.split(",")]
    running: list = []
    while jobs or running:
        while jobs and len(running) < len(gpus):
            f, cmd, out = jobs.pop(0)
            g = [x for x in gpus if x not in {r[1] for r in running}][0]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, **f["env"])
            log = open(out + ".log", "w")
            p = subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=log)
            running.append((p, g, out))
            print(f"  start {os.path.basename(out)} on gpu {g}", flush=True)
        done = [r for r in running if r[0].poll() is not None]
        for r in done:
            running.remove(r)
            print(f"  done  {os.path.basename(r[2])} rc={r[0].returncode}", flush=True)
        if not done:
            import time
            time.sleep(20)
    print(f"\n'{a.tag}' complete. Compare with:\n"
          f"  python tools/kfold.py --report base {a.tag}")


if __name__ == "__main__":
    main()
