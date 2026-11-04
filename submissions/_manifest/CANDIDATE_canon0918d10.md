# Candidate canon0918d10 — the retained best at gain dose 1.0 (pre-registered 2026-09-20, armed for the 09-21 slot)

**What.** `submissions/canon0918d10/`: `model.pth` byte-identical to the retained best 932643/934147
(`submissions/canon0918/model.pth`); `submission.py` differs in **one character**, `GAIN_DOSE = 0.5 → 1.0`. The
post-averaging gain then sets each cell's temporal variance *to* the members' average variance instead of half-way to
it. Time-mean invariant by construction (MVPE untouched); the interval student still brackets the plain mean.
Zip `build/canon0918d10.zip`, sha256 `f1a0e8c56d2667b65d82ca8d6c42b41a594638b95f8f16cf261542b766dde881`
(207,362,725 B extracted). Standalone: 9/9 graphs, finite, ordered, 3.55 ms, temporal-variance ratio 1.0153 vs dose 0.5,
time-mean max abs 4.9e-5 (MVPE invariant as required); `check_pack --eval` **MATCH**.

**Why now — the axis was closed on a misattribution.** Dose 1.0 was built on 09-15 (`CANDIDATE_k4g10.md`) and never
uploaded: the dose-0.5 board read (924146) and its exact repeat (925203) were **timing-only** failures and the axis was
declared "closed at dose 0.5". The 09-18 control (923184's own bytes, 5.27 → 6.11 ms) and the 09-19/20 reads (4.78,
4.65 ms) established that the clock is a **wide platform draw, not a code cost** — so the closure rested on a
misattribution, and the accuracy channel was positive throughout (dose 0.5: exact board N **+0.0094**).

**New measurement (2026-09-20, `research/flow2_2026-09-20/sweep_sps_<fold>.json`).** Held-out fold cells, the fold A2
students' official SPS with the shipped containment guard, exact marginals, cell-balanced (far_drop Re-row):

| dose | mim1 | far_drop | angle05 | mean exact ΔF |
|---|---|---|---|---|
| 0.75 | +0.0098 | +0.0124 | +0.0295 | +0.0172 |
| 0.9 | +0.0130 | +0.0161 | +0.0398 | +0.0230 |
| **1.0** | **+0.0140** | **+0.0166** | **+0.0428** | **+0.0245** |
| 1.1 | +0.0140 | +0.0154 | +0.0422 | +0.0239 |
| 1.25 | +0.0121 | +0.0100 | +0.0339 | +0.0187 |

Concave with a flat optimum at dose ≈ 1.0, **all three folds positive at every dose ≥ 0.75**. Channels at dose 1.0
(mim1): rel 0.0746 → 0.0750, TKE 0.4851 → 0.4801, MVPE unchanged, SPS 52.797 → 52.827. TKE-funded, as the gain always
was; the k4g rule (deterministic post-averaging transform: N ≥ +0.010 under every fit, MVPE invariant, channel floors)
is met with margin.

**Prediction (exact power mean, vs the retained best 934147 = 81.90253).** rel **94.8418** [94.838, 94.846] ·
TKE **79.3346** [79.27, 79.40] · MVPE **94.0909748382** (invariant, ±1e-5) · SPS **43.9880** [43.96, 44.02] ·
**exact accuracy-only N = +0.0151 [+0.005, +0.025]** (the fold mean +0.0245 discounted by the 0.63× board:fold transfer
the dose-0.5 read measured on this channel). Clock: a draw from the current distribution (4.65–6.11 ms in the last four
reads); at 5.0 ms → **81.905**, at 4.65 ms → **81.918**, at 6.0 ms → **81.845**.

**Decision rule.** **Promote** if the final improves AND N ≥ +0.005 under the exact score AND |ΔMVPE| ≤ 1e-4.
**Kill the dose axis for good** if N ≤ 0 — the transform will then have been measured three times on the board.
**Quarantine** (build fault) if MVPE moves more than 1e-4 or rel moves more than 0.02 from the prediction.
A timing-only failure (N ≥ +0.005, final below) promotes the *method* for the package and earns one repeat.

## Armed (2026-09-20)
`research/audit_2026-09-17/claude/arm_canon0918d10.sh` — waits for 2026-09-21T00:00:30Z, verifies the sha256 before
sending, retries to 23:45Z; log `logs/submit_canon0918d10.log`.

## Board result — submission (09-21T00:00Z, scoring rpw7): final **81.80097**, clock 5.956 ms — **KILL**
| | rel | tke | mvpe | sps | clock | final |
|---|---|---|---|---|---|---|
| 934147 (dose 0.5, retained best) | 94.8462055 | 79.2328683 | 94.0909748 | 43.9750022 | 4.649 ms | 81.90253 |
| **this (dose 1.0)** | 94.8201792 | 79.2409123 | 94.0909583 | 43.9595441 | 5.956 ms | 81.80097 |
| Δ | **−0.0260** | **+0.0080** | −0.00002 | **−0.0155** | | |
**Exact accuracy-only N = −0.0154** against a registered prediction of **+0.0151 [+0.005, +0.025]** — outside the band,
opposite sign. The kill rule (N ≤ 0) fires: **the gain dose closes at 0.5 for good**, now with three board reads behind it.

**Where the fold instrument misled.** MVPE was invariant to 2e-5 as the algebra requires, so the artefact is sound and
the channels are readable. The fold suite predicted rel −0.0004 and TKE +0.102 in score units; the board delivered
rel **−0.026** (65× worse than predicted) and TKE **+0.008** (13× smaller). Against the dose-0.5 board read — which
transferred *with* the predicted sign on every channel (tke +0.156, rel −0.023, MVPE invariant) — the difference is the
dose: at dose 0.5 the gain moves the fluctuation half-way to the members' variance and the rel cost is small; at dose
1.0 the same transform over-amplifies a **pattern that is wrong at exactly the cells where the members disagree most**,
so rel pays 4× what the folds said while TKE gains almost nothing extra. The fold ensembles are three-member,
fold-trained models on ten held-out cells; the hidden set is a different, wider geometry, and this is the largest
fold-to-board disagreement on record for a deterministic transform (the 09-13 lesson "price a gate with TKE zeroed"
would have caught it: TKE-zeroed, the dose-1.0 fold gain is ≈ +0.004, not +0.0245).

**Consequences.** (1) The dose axis is closed. (2) **The flow-blend numbers in `research/flow2_2026-09-20/` rest on the
same instrument and the same dose-1.0 setting, so their +0.040 fold mean must be read as ≈ +0.015 at best and possibly
negative** — the blend is not a candidate without a board-calibrated amplitude story. (3) The retained best is unchanged
(934147); the frozen method keeps dose 0.5.
