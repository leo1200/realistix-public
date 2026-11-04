"""The scored submissions, and one definition of "the board delta".

Why this exists. Three instruments quote board deltas -- `predict_score.py`'s
back-test table, `docs/archive/developer_guide.md`, and `docs/archive/FINDINGS_2026-09-10.md` -- and until
2026-08-21 they used three different conventions without saying so:

    --mults 1,2   -0.009   four accuracy subscores, sps excluded
    distillation  -0.106   four accuracy subscores, sps excluded
    w_rel 16      -0.152   sps INCLUDED

That is not a bookkeeping nit. The local side of a mimic back-test always excludes
sps, because there it is `evaluate.subscores`' constant 0.15-sigma band rather than
the learned head. Comparing the sps-excluded local number against the sps-included
board number is what produced the recorded verdict "`w_rel 16`: right sign, **2.4x
too small**". Matched, that ratio is **1.70** -- which is the same ~1.7x the same
documents quote as the accuracy transfer coefficient. The one back-test that
appeared to contradict the calibration and the calibration itself were the same
observation computed two ways.

So the table is computed here, from the recorded subscores, and never transcribed.

**Conventions.** `acc_delta` defaults to the sps-excluded form, because that is the
one a fold-based local measurement can be compared against. Pass `with_sps=True`
only when the local side is a *packed* artefact scored through `submission.py`
(`tools/honest_pipeline.sh`), whose sps comes from the learned head and is therefore
comparable. `final_delta` uses all five and reproduces the leaderboard total.
"""

from __future__ import annotations

from .weights import ACC, ACC_NO_SPS, DELTA

#: Every scored submission: id -> (final, rel_l2, tke, mvpe, time, sps).
#: `final` is the leaderboard's own figure, at the precision it reported.
SUBMISSIONS: dict[int, tuple[float, ...]] = {
    879338: (77.780,   94.1680, 74.0260, 92.8360, 91.4840, 31.9390),
    879743: (78.414,   93.9140, 77.7450, 93.0780, 91.5510, 33.2650),
    881057: (79.964,   94.3000, 77.4600, 93.3100, 90.0000, 39.2130),
    882102: (80.095,   94.2740, 78.0060, 93.5440, 89.9780, 39.4890),
    882969: (80.792,   94.4910, 78.9390, 93.8790, 89.4770, 41.5650),
    883878: (80.746,   94.5090, 78.8650, 93.9680, 88.8210, 41.5980),
    884969: (80.889,   94.5090, 78.8640, 93.9680, 90.3070, 41.5970),
    886080: (80.677,   94.3920, 79.4990, 93.9280, 90.5620, 40.6210),
    887158: (81.15749, 94.8340, 78.0300, 93.9990, 90.3790, 42.3620),
    891095: (81.137,   94.8340, 78.0290, 94.0000, 90.1770, 42.3580),
    891698: (81.144,   94.8740, 77.7420, 94.0570, 90.3980, 42.3180),
    892587: (81.143,   94.8440, 78.0080, 94.0460, 90.2660, 42.3210),
    893273: (81.075,   94.7520, 77.9580, 93.8200, 91.3310, 41.9110),
    894861: (80.99865, 94.7170, 77.8440, 93.2380, 90.2480, 42.3660),
    895690: (81.01684, 94.9084, 76.6799, 94.0119, 90.4178, 42.1825),
    896716: (81.24771, 94.83401, 78.02959, 93.99944, 90.25143, 42.78022),
    897732: (81.31545, 94.83401, 78.02959, 93.99944, 90.14276, 43.10031),
    898763: (81.38978, 94.79646, 78.35820, 93.64737, 90.40185, 43.37715),
    901812: (81.44630, 94.83350, 78.27681, 93.75968, 90.31537, 43.56071),
    902375: (81.51651, 94.85970, 78.30605, 93.91327, 90.55086, 43.63168),
    903484: (81.48739, 94.89701, 77.77870, 94.05661, 90.43671, 43.64767),
    904950: (81.59694, 94.82381, 79.05878, 93.91028, 90.50468, 43.73919),
    907243: (81.58884, 94.85549, 79.06772, 93.98471, 90.02907, 43.80020),
    908202: (81.58194, 94.85549, 79.06772, 93.98471, 89.95789, 43.80020),
    909666: (81.57239, 94.82253, 79.10584, 93.89632, 90.36411, 43.68270),
    910411: (81.55030, 94.73654, 79.14772, 93.89995, 91.22892, 43.39844),
    911563: (81.55306, 94.82381, 79.05878, 93.91028, 90.04014, 43.74379),
    912717: (81.67313, 94.82381, 79.05878, 93.91028, 91.28116, 43.74379),
    914270: (81.73109, 94.82381, 79.05878, 93.91028, 91.88201, 43.74379),
    915299: (81.75365, 94.82385, 79.05839, 93.91058, 92.11282, 43.74508),
}

#: What each submission changed, and against what. `None` means the pair is bundled
#: or the reference is not the immediately preceding submission.
REFERENCE = {
    891095: (887158, "887158's weights; fp16 module cast, bucketed shapes, warm import"),
    891698: (887158, "--mults 1,2: deepest resolution level dropped"),
    892587: (887158, "clean-room retrain of the same recipe on other hardware"),
    893273: (887158, "1 ensemble member instead of 3, matched head without spread"),
    894861: (887158, "distillation AND a learned combiner -- confounded"),
    895690: (887158, "w_rel 16 against the shipped 8"),
    887158: (886080, "loss re-derived from the weights; also drops 7575_0"),
    896716: (887158, "identical members, head step re-selected 1000->4000 on the "
                     "mimic folds -- the sole change is the interval head"),
    897732: (896716, "same members, head retrained at lr 8e-4 + tau_frac 0.05 -- "
                     "the joint sweep winner; sps transfer 0.86 vs screen"),
    898763: (897732, "three all-81 members retrained under Muon; head recipe "
                     "unchanged -- the first members-level change since 887158"),
    901812: (898763, "members retrained with --w-sps 1.0 (the SPS-bulk loss term); "
                     "the first submission selected by the interval-channel screen"),
    902375: (901812, "identical runs taken at their step-7000 snapshots (held-out "
                     "argmax under w_sps, 8/8 folds); head retrained, recipe "
                     "unchanged -- a bookkeeping find, zero training cost"),
    903484: (902375, "w_sps 2.0 members at their own argmax step 6000 -- the "
                     "dose-stop interaction; the calibrated era's first miss"),
    904950: (902375, "w_tke 1.0 + EMA 0.998 at step 6000 -- tke restoration off "
                     "the w_sps-shifted operating point; predicted +0.05, paid +0.080"),
    907243: (904950, "3x width-128 at mults 1,1.25,2 -- the FLOP-matched width "
             "profile, recipe otherwise 904950. Predicted +0.04 [-0.02, +0.10], "
             "committed before the score (df5940e). Band HIT at -0.008; every "
             "accuracy subscore positive and inside its channel (acc-3 +0.038, "
             "mvpe transfer ~6.7x confirming section 0b), and the whole final "
             "miss is time_score -0.476. CORRECTED READ (same day): the recent "
             "same-code-path era has time_score sd 0.171 with +/-0.26 swings "
             "between byte-identical artefacts, and 904950 sat at the era MAX "
             "(90.505) -- so this is ~1.8 sigma against the era mean, i.e. "
             "somewhere between an unlucky draw and a real ~0.1-0.3 penalty. "
             "The bytes-at-full-res mechanism is plausible but NOT established "
             "(891698: a far smaller model timed SLOWER). First calibration "
             "point on the capacity channel: fold acc +0.011 -> board acc-3 "
             "+0.038"),
    908202: (907243, "byte-identical resubmit of 907243 -- the time-redraw probe "
             "(first attempt died in a platform storage SSL failure; failed "
             "submissions do not consume quota). Predicted 81.615 [81.586, "
             "81.640]; landed 81.582, band MISSED by 0.004 -- and the miss is "
             "the measurement: all four accuracy subscores reproduced "
             "bit-identically, and time drew 89.958 against 907243's 90.029. "
             "Two same-artefact draws 0.07 apart, both ~0.35 below the era "
             "mean: the wprofF time penalty is REAL at n=2 (~-0.033 of final), "
             "same-artefact redraw noise is ~+/-0.04, and the era's 0.171 "
             "time spread was therefore mostly real per-artefact differences. "
             "The bytes-at-full-res hypothesis is back, now with support"),
    909666: (908202, "3x width-96 at mults 1,1.5,2.5 (deep-trim profile) -- the "
             "time-model probe, three hypotheses recorded in advance. Accuracy "
             "dead flat as predicted (rel -0.001, tke +0.047, mvpe -0.014, sps "
             "-0.056 vs 904950); time drew 90.364 = 8.29 ms. Hypothesis A (clock "
             "tracks total bytes) REFUTED -- 12.5% fewer bytes, slower. C (full "
             "channel-count penalty) refuted. Closest to B with a ~+0.27 ms "
             "residual: the container taxes the FULL-RES level hardest and adds "
             "a small penalty for non-standard channel counts weighted by grid "
             "size. Final 81.572; board keeps 81.597. Third vertex of the "
             "time-model triangle, and the design rule it yields: keep the "
             "trunk shape-identical to the shipped one when buying time"),
    910411: (904950, "one width-96 member instead of three, lean loss (mse/tmean/"
             "spec dropped: jointly +0.010, 4/4 folds), head without the spread "
             "input. The single-member reopening: under w_tke 1.0 the members are "
             "amplitude-calibrated and ensemble averaging over-damps (tke sign "
             "flipped, +0.5-0.6 for one member at every seed). Composed honest "
             "gate 85.149 vs hp_cur 84.886 = weighted acc+sps dead neutral, plus "
             "the board-measured one-pass time credit. Prediction committed "
             "pre-score (CANDIDATE_onelean.md): +0.08, band [0.00, +0.15] vs "
             "904950's 81.597. Falsifier: below 81.60 closes the axis."),
    911563: (904950, "904950's three members, bit-identical; only the interval "
             "head changed, retrained without the ensemble-spread input. "
             "Prediction committed pre-score (CANDIDATE_nospreadhead.md): +0.02, "
             "band [0.00, +0.06]. Read 81.553, so the falsifier tripped -- but "
             "the decomposition says the change did nothing rather than harm: "
             "rel_l2, tke and mvpe are identical to the last digit (the members "
             "really were untouched, the cleanest artefact verification here yet) "
             "and sps moved +0.005. The whole -0.044 is time_score, on a trunk "
             "that is shape-identical and a head with FEWER inputs (120->80): the "
             "container ran 8.024 ms/sample on 08-28 and 8.919 on 09-02, an 11% "
             "platform slowdown, and section 9a allows time comparisons only "
             "day-adjacent. The calibration point is the sps one: the tail arm "
             "read +0.181 and the head-config coefficient of 1.22 +- 0.32 "
             "predicted board sps about +0.22 against +0.005 observed -- a ~40x "
             "over-prediction on the coefficient this project trusted most."),
    912717: (911563, "911563's model.pth byte-identical (md5 137325e2...); only submission.py "
             "changed: _load() plus a three-shape cuDNN warm-up run at module import "
             "instead of lazily inside predict(). Prediction committed pre-score "
             "(CANDIDATE_initmove.md): +0.10, band [0.00, +0.30]. Read 81.673 = +0.120, "
             "band HIT, and the cleanest single-variable read on the board: all four "
             "accuracy subscores identical to the last digit, time_score +1.241 = "
             "8.919 -> 6.651 ms/sample. So 2.27 ms/sample of one-off setup WAS being "
             "charged inside the timed region, which refutes FINDINGS section 9's "
             "reading of 891095 and retires the +0.147 'hard ceiling' (section 9c). "
             "New best; rank 3, 0.088 behind first."),
    914270: (912717, "912717's model.pth byte-identical; submission.py captures each bucketed "
             "batch shape as a CUDA graph at import (members on three streams to 8 "
             "windows, sequential above) and replays it. Prediction committed pre-score "
             "(CANDIDATE_cugraph.md): +0.25, band [0.00, +0.40]. Read 81.731 = +0.058, "
             "band HIT but far below central: time 91.882 = 5.69 ms/sample against 6.65, "
             "a graph/eager ratio of 0.86 where 2-3 windows per call predicted 0.3-0.5. "
             "Accuracy identical to the last digit. The reading that fits: ONE window per "
             "call on a fast, unloaded CPU (eager launch ~10 us/kernel -> 6.6 ms; graph "
             "replay is then the GPU-side latency floor of ~640 serial kernels, ~4.4 ms "
             "on A100 plus host work). New best; rank 3, 0.030 behind first."),
    915299: (914270, "914270's weights; members and head cast to fp16 in channels_last on the "
             "graph path (removes the ~60 layout-conversion kernels per member). "
             "Prediction committed pre-score (CANDIDATE_cugraph16.md): +0.05, band "
             "[0.00, +0.10]. Read 81.754 = +0.023, band HIT below central; time 92.113 = "
             "5.35 ms/sample against 5.69, ratio 0.94 where the GPU chain shrank 0.85-0.87 "
             "locally -- so ~3.4 ms of the container's per-call time is fixed overhead "
             "outside our kernels and ~2 ms is the chain. Accuracy within 0.002 on every "
             "channel: fp16 arithmetic is free. New best; RANK 2, 0.008 behind first."),
}

_COL = {"rel_l2": 1, "tke": 2, "mvpe": 3, "time": 4, "sps": 5}


def subscores(sub: int) -> dict[str, float]:
    row = SUBMISSIONS[sub]
    return {k: row[i] for k, i in _COL.items()}


def final(sub: int) -> float:
    return SUBMISSIONS[sub][0]


def acc_delta(sub: int, ref: int, *, with_sps: bool = False, weights=None) -> float:
    """Board accuracy delta of ``sub`` against ``ref``, in units of final score.

    Excludes `time_score`, which a training change does not move and container
    wall-clock does. Excludes `sps` unless ``with_sps`` -- see the module docstring.
    """
    w = weights or (ACC if with_sps else ACC_NO_SPS)
    a, b = subscores(sub), subscores(ref)
    return sum(v * (a[k.replace("_score", "")] - b[k.replace("_score", "")])
               for k, v in w.items() if k != "time_score")


def final_delta(sub: int, ref: int, weights=DELTA) -> float:
    """Predicted change in `final_score`, from the subscores alone."""
    a, b = subscores(sub), subscores(ref)
    return sum(v * (a[k.replace("_score", "")] - b[k.replace("_score", "")])
               for k, v in weights.items())


def decompose(sub: int, ref: int, weights=ACC) -> dict[str, tuple[float, float]]:
    """Per-subscore ``{name: (delta, contribution to final)}``."""
    a, b = subscores(sub), subscores(ref)
    out = {}
    for k, v in weights.items():
        n = k.replace("_score", "")
        d = a[n] - b[n]
        out[n] = (d, v * d)
    return out


def table(pairs=None, with_sps: bool = False) -> str:
    """The board-delta table, computed rather than transcribed."""
    pairs = pairs or [(s, r) for s, (r, _) in sorted(REFERENCE.items())]
    lines = [f"{'change':46s} {'accuracy':>9s} {'final':>9s} {'actual':>9s}"]
    lines.append("-" * len(lines[0]))
    for sub, ref in pairs:
        note = REFERENCE.get(sub, (ref, ""))[1]
        lines.append(f"{str(sub) + ' ' + note:46.46s} "
                     f"{acc_delta(sub, ref, with_sps=with_sps):>+9.4f} "
                     f"{final_delta(sub, ref):>+9.4f} "
                     f"{final(sub) - final(ref):>+9.4f}")
    return "\n".join(lines)


# The board's own reproducibility, measured rather than assumed.
#
#   891095 carries 887158's exact weights and differs only in inference code: every
#   accuracy subscore reproduces to 0.004, so the scoring itself is deterministic and
#   the hidden set is fixed.
#   892587 is the same recipe retrained from scratch on other hardware: per-subscore
#   moves of +0.010 / -0.022 / +0.047 / -0.041, which is +0.007 of final in the
#   sps-excluded convention `acc_delta` uses, and -0.004 with sps included. Its
#   `time_score` moved 0.113 points = 0.216 ms/sample = 0.011 of final.
#
# So the floor on "what will this recipe score" is ~0.015 of final, and three
# quarters of it is container wall-clock rather than anything about the model. A
# back-test disagreeing by less than it is a pass.
#
# Caveat (2026-09-01): this is ONE retrain. A single realization cannot estimate a
# standard deviation, so 0.015 is an order-of-magnitude reference point, not a
# measured floor, and the earlier claim that "no predictor can beat that" does not
# follow from it. Treat it as the smallest difference worth arguing about, and do
# not use it to certify a prediction as being at the limit of what is knowable.
NOISE_ACC, NOISE_TIME, NOISE_FINAL = 0.007, 0.011, 0.015

if __name__ == "__main__":
    print(table())
    print(f"\nboard reproducibility: accuracy {NOISE_ACC}, time {NOISE_TIME}, "
          f"final {NOISE_FINAL} (892587 retrain)")
