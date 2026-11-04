"""The leaderboard's score function, in one place.

**Since 2026-09-17 the function is known exactly** (`POWER_MEAN`, `final_exact`,
`marginals`): `final_score` is a power mean of the five subscores,

    F = (0.5 rel^p + 0.2 sps^p + 0.1 tke^p + 0.1 time^p + 0.1 mvpe^p)^(1/p),  p = 0.6854,

which reproduces all 41 scored cards to 9e-11 (Codex, `research/audit_2026-09-17/
codex/power_mean_audit.json`; verified independently). Everything below it in this
module is the *linear* history: `final_score` was believed to be a weighted sum and
the weights were fitted on the scored submissions. Those fits were secants of the
power mean at our operating point -- within 0.01 of the exact marginals -- and they
drifted with the card population, which is why "the weights moved" kept recurring.
They stay here because every archived number was computed under them; new pricing
should use `marginals()` (a delta) or `final_exact()` (a level). This module is the
only copy of that fit. It used to be pasted into ten tools, and when the fit was redone
on fourteen submissions only one of them was updated -- so the rig that ranks every
experiment went on weighting MVPE at a third of its identified value.

**The fit.** Least squares on the fourteen submissions scored by 2026-08-20, no
intercept (`docs/archive/HANDOFF_2026-09-10.md` section 4):

    rel_l2 0.4738   sps 0.2529   tke 0.1057   time 0.0947   mvpe 0.0926

Sum 1.020, in-sample RMS 0.0052, leave-one-out RMS 0.0116.

**What changed, and why the archive does not match.** The eight-point fit that
preceded it (`EIGHT_POINT` below) put `mvpe` at 0.0319. Across the first eight
submissions MVPE spanned 92.8 to 94.0 and moved by hundredths between neighbours,
so its column was nearly collinear with the intercept-free fit and its weight was
pinned by almost nothing; 894861 is the first submission to move it hard (-0.761)
and it identifies the coefficient. Four of the five weights barely moved.

Every fold number quoted in `docs/archive/HANDOFF_2026-09-10.md` and `docs/archive/FINDINGS_2026-09-10.md` before 2026-08-21 was
computed under `EIGHT_POINT`. Re-reporting an archived comparison today will not
reproduce the recorded figure exactly. The difference is small for anything that
did not move MVPE and it never reverses an ordering we have checked, but a number
that has to match the documents must be recomputed with `EIGHT_POINT` explicitly
rather than assumed.

**Levels or deltas.** `WEIGHTS` is fitted on *levels* with no intercept, which is
how this project has always quoted it. Every actual use is a *delta* -- what will
this change be worth -- and fitting the same model on differences gives a measurably
better answer, because the level fit spends its freedom reproducing a constant that
every submission shares. Both rows below were fitted on the fifteen submissions
available on 2026-08-21, which is one more than `WEIGHTS`:

    level  rel_l2 0.4717  sps 0.2531  tke 0.1047  time 0.0950  mvpe 0.0952
    DELTA  rel_l2 0.4039  sps 0.2603  tke 0.0954  time 0.0999  mvpe 0.1030

Across those fifteen submissions' 105 pairwise differences, `DELTA` predicts the
observed change in `final_score` at RMS **0.0042** against the level weights' 0.0075.
(105 differences of 15 scorecards are 105 *algebraic combinations* of 15
observations, not 105 independent data points -- the effective sample size is 15,
and this RMS is correspondingly optimistic. It is a fit quality, not a validation.)
Both are already below the 0.015 the board reproduced itself to on one retrain --
a reference scale rather than a measured floor -- so this is a refinement and not a
correction -- but `DELTA` is the right one for converting a
subscore delta into a score delta, and `realistix.board.final_delta` uses it. Note
that the level fit's weights sum to 1.02 and the delta fit's to 0.96: neither is a
convex combination, both are local linearisations valid over the 77.8-81.2 range
observed, and neither should be extrapolated to a perfect scorecard.

**Both fits are held at the point counts above rather than refreshed per
submission**, so that every archived comparison stays reproducible. Refitting the
levels on all twenty-two scored submissions (2026-08-28) gives rel_l2 0.4681, sps
0.2515, tke 0.1039, time 0.0915, mvpe 0.1035 at leave-one-out RMS 0.0106 -- the
largest coefficient move is `mvpe`, by 0.011, and no ordering changes. Refresh them
only alongside a re-report of the numbers that were computed under them.

**Which set to use.** `ACC` drops `time`, which is a property of the packed
submission rather than of a training run, so it is the one a fold comparison wants.
`ACC_NO_SPS` drops `sps` as well, because the `sps_score` a local rig reports comes
from `evaluate.subscores`' constant 0.15-sigma band and not from the learned
asymmetric head that ships -- the proxy rewards better-centred predictions directly
and supplied 60-75% of every fold headline quoted before 2026-08-20. Read a fold
delta through `ACC_NO_SPS` unless the change is known not to touch accuracy.
"""

from __future__ import annotations

#: The exact score function (identified 2026-09-17): outer power p and inner weights.
POWER_MEAN = dict(p=0.6854, w=dict(rel_l2_score=0.5, sps_score=0.2, tke_score=0.1,
                                   time_score=0.1, mvpe_score=0.1))


def final_exact(sub: dict, prefix: str = "") -> float:
    """`final_score` from the five subscores, exactly (all 41 cards to 9e-11)."""
    p, w = POWER_MEAN["p"], POWER_MEAN["w"]
    return sum(v * float(sub[f"{prefix}{k}"]) ** p for k, v in w.items()) ** (1.0 / p)


def marginals(sub: dict, prefix: str = "") -> dict:
    """dF/ds_j at the given scorecard: w_j (F/s_j)^(1-p). At 929146 (2026-09-17):
    rel 0.477, sps 0.243, tke 0.101, time 0.0965, mvpe 0.0957. Because p < 1 the
    weakest subscore carries the largest marginal; re-evaluate at a new operating point
    rather than extrapolating."""
    p, w = POWER_MEAN["p"], POWER_MEAN["w"]
    F = final_exact(sub, prefix)
    return {k: v * (F / float(sub[f"{prefix}{k}"])) ** (1.0 - p) for k, v in w.items()}


def final_delta_exact(base: dict, new: dict, prefix: str = "") -> float:
    """Exact change of `final_score` between two scorecards (levels, not a linearisation)."""
    return final_exact(new, prefix) - final_exact(base, prefix)


#: The fourteen-point fit (2026-08-20). All five subscores; this is `final_score`.
WEIGHTS = dict(rel_l2_score=0.4738, sps_score=0.2529, tke_score=0.1057,
               time_score=0.0947, mvpe_score=0.0926)

#: What a training run can move: everything except container wall-clock.
ACC = {k: v for k, v in WEIGHTS.items() if k != "time_score"}

#: What the board actually pays for, on a local rig whose SPS is a proxy.
ACC_NO_SPS = {k: v for k, v in ACC.items() if k != "sps_score"}

#: Fitted on all 105 pairwise *differences* rather than on levels. Use this to turn
#: a subscore delta into a final-score delta; see the module docstring.
DELTA = dict(rel_l2_score=0.4039, sps_score=0.2603, tke_score=0.0954,
             time_score=0.0999, mvpe_score=0.1030)

#: `time_score`'s weight, for pricing inference cost in units of final score.
W_TIME = WEIGHTS["time_score"]

#: The superseded eight-point fit (leave-one-out RMS 0.0122). Kept because every
#: number recorded before 2026-08-21 was computed with it; see the module docstring.
EIGHT_POINT = dict(rel_l2_score=0.5240, sps_score=0.2563, tke_score=0.1111,
                   time_score=0.0992, mvpe_score=0.0319)


def weighted(row: dict, w: dict = ACC, prefix: str = "") -> float:
    """Weighted sum of ``row``'s subscores under ``w``.

    ``prefix`` selects a split out of a flat history row, e.g. ``"val_aoa."``.
    """
    return sum(v * row[f"{prefix}{k}"] for k, v in w.items())


#: Branch weights inside `starting_kit.scoring.aggregate_sps`. SPS is one interval
#: scored three times, differing only in which accuracy metric supplies the
#: per-window prefactor.
SPS_BRANCH = {"rel_l2": 0.5, "tke": 0.3, "mvpe": 0.2}

#: The width/coverage factor `R_bar` at the shipped operating point, fitted across
#: the submission history as `sps / W_bar` (see `tools/sps_decompose.py`). It is a
#: property of the interval head and is stable to ~1% while the head recipe is
#: fixed: 0.586 through the 887158 era, 0.602 after the head step re-selection
#: (896716), 0.606 after the lr/tau sweep (897732), 0.611 under the Muon members
#: (898763). Every one of those steps *raised* it, which is what "the head got
#: better" means in this decomposition.
K_SHIPPED = 0.611


def acc_effective(subscores: dict, K: float = K_SHIPPED, base: dict = WEIGHTS) -> dict:
    """Accuracy weights including the SPS branch prefactor, at a given operating point.

    Why the shipped weights understate accuracy. `final_score` is linear in the five
    subscores, so `WEIGHTS` is the whole story for a *subscore* delta -- but a
    forecaster change does not deliver a subscore delta, it delivers an *error*
    delta, and that error reaches the objective twice. Once through its own subscore,
    `score_error(e) = 100 / (1 + e/2)`. And once through `sps_score`, because
    `aggregate_sps` multiplies every element's reward by a per-window
    `(1 - pm) = 0.5 / (0.5 + e)` built from the same three errors. Chain-ruling the
    second path back into subscore units:

        w_eff(x) = w(x) + w_sps * 100 * K * (d pref / d e_x) / (d S_x / d e_x)

    At the board's 898763 operating point that is 1.49x on rel_l2, 1.65x on tke and
    1.94x on mvpe -- and because the three inflations differ, it re-ranks rather than
    rescales: MVPE goes from 0.195 of rel_l2 under `ACC_NO_SPS` to 0.255.

    ``subscores`` is ``{"rel_l2": 94.8, ...}`` in *score* units; only the three
    accuracy entries are read. ``K`` is the width/coverage factor of the head the
    change would ship against -- `K_SHIPPED` unless you measured your own.

    Two caveats, both stated rather than hidden. The prefactor is evaluated at the
    mean error instead of averaged over windows, and `1/(0.5+e)` is convex, so this
    slightly understates `W_bar`; the bias is stable across submissions and cancels
    in a delta. And a large move invalidates the linearisation -- re-evaluate at the
    new point rather than extrapolating, exactly as for `DELTA` itself.
    """
    w_sps = base["sps_score"]
    out = dict(base)
    for q, b in SPS_BRANCH.items():
        e = 2.0 * (100.0 / subscores[q] - 1.0)          # invert score_error
        dS_de = -50.0 / (1.0 + 0.5 * e) ** 2
        dpref_de = -b * 0.5 / (0.5 + e) ** 2
        out[q + "_score"] = base[q + "_score"] + w_sps * 100.0 * K * dpref_de / dS_de
    return out
