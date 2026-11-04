# Candidate k4g10 — the member-variance-matched fluctuation gain at dose 1.0 (pre-registered 2026-09-15, joint Claude + Codex; NOT yet uploaded)

**What.** 923184's four members and head, unchanged, and `submissions/k4g10/submission.py` — byte-identical to the
twice-scored `k4g` file except `GAIN_DOSE = 0.5` → `1.0` (one character). After averaging, per window / channel / cell
the mean's temporal fluctuation about its own time-mean is rescaled by `1 + λ(g − 1)`, `g = min(3, sqrt(V_members/v_mean))`,
now at **λ = 1.0**: the mean's temporal variance is matched to the members' average, the mechanism's fixed point rather
than a point on a ramp. Time-mean algebraically invariant (MVPE untouched); the head still brackets the PLAIN mean.

**Why now.** `REOPEN_SUMMARY.md` §2 registered the escalation path explicitly — "a higher dose is a new pre-registration
after the repeat" — and the one permitted exact repeat happened on 09-14 (925203). The dose-0.5 board read is the only
mechanism this month that transferred with the predicted sign on every channel (non-time **+0.0104**, tke +0.156,
MVPE invariant to seven digits); the fold family is concave with its direct-D optimum at dose ≥ 1, and 0.5 was chosen
for conservatism, not because the read said so. This is interpolation along a curve measured before either board read,
not selection on a board result.

**Standalone check, idle A100 (`research/slot_2026-09-15/k4g10_standalone_a100.json`).** `build/k4g10.zip`
(207,362,725 bytes extracted, of the 268,435,456 cap) imports in 9.7 s with no repo modules, four members, **9/9 graphs**,
outputs finite and ordered; vs k4 on 17 windows: time-mean max abs 4.9e-5 (fp16 — **MVPE invariant**), temporal-variance
ratio 1.0302 (k4g read 1.0146), bounds max abs 1.7e-2 (the `lo ≤ pred ≤ hi` guard bites harder at full dose — priced
below, in-sample sps still rises). **4.0 ms** per one-window call (k4 3.94, k4g 4.04): free on the clock.
`check_pack --eval` **MATCH** (2.83e-4 packed-vs-runs, intervals matching).

**Prediction (board subscore deltas vs 923184; judgement bands, not confidence intervals).** Built from the per-channel
board:in-sample transfer ratios *measured at dose 0.5* (rel 2.18×, tke 1.23×, sps 0.63×, mvpe 0×) applied to the dose-1.0
in-sample deltas (rel −0.025, tke +0.249, mvpe 0, sps +0.061 — `NOTES.md` val_aoa table), cross-checked against the
fold-D ratio dose1.0/dose0.5 = 1.42:

| | rel | tke | mvpe | sps | non-time N | time |
|---|---|---|---|---|---|---|
| predicted Δ | −0.055 [−0.090, −0.025] | +0.306 [+0.150, +0.450] | 0 [−0.001, +0.001] | +0.038 [−0.020, +0.090] | **+0.015 [+0.004, +0.026]** | the day's clock |

N = +0.0160 (locked fit) / +0.0154 (34-card fit) / +0.0148 (fold scaling) — **+0.015 central under all three**.
**final +0.015 + 0.095·(time_score − 92.164)**; centre ≈ 81.821 on 923184's own clock (5.27 ms), ≈ 81.808 at the
eight-read clock mean (5.46 ms, sd 0.24). **New best (> 81.80564) iff time_score > 92.006, i.e. t < 5.50 ms — P ≈ 0.57.**
This cannot reach #1 (81.85874) on any clock draw; the reachable band is rank 2–3.

**Decision rule (registered before the score).** Promote if the final improves AND N (non-time, actual SPS once) ≥ +0.010
under both fitted weight sets, with rel ≥ −0.100, sps ≥ −0.060, |Δmvpe| ≤ 0.005. Reject if N ≤ 0 or a channel floor
fails — **and in that case the gain axis closes at dose 0.5 for good; no dose 0.75, no third dose**. A timing-only
failure (N ≥ +0.010, final below) closes the axis too: the mechanism will have been measured three times and the
remaining variance is the platform's, not ours. **No exact repeat of this artefact.** The kill number is N ≤ 0.

**What it does not do.** All of the profit is TKE-funded (exact packed N₀ with TKE and its prefactor removed is negative
at both doses). The board is the only instrument that has ever priced this against the true weights, and it put dose 0.5
at the margin; dose 1.0 is the same bet at twice the stake. A board pass does not prove private-regime improvement —
the Decision-Phase retrain re-derives the gain from the organisers' own members.

## Armed
(awaiting the owner's go — the 2026-09-15 UTC slot is unused and expires at 2026-09-16T00:00Z)
