# Candidate k4g — member-variance-matched fluctuation gain on the 923184 ensemble mean (pre-registered 2026-09-12, Codex round 7; NOT uploaded)

**What.** 923184's four members and head, unchanged. After averaging, per window / channel / cell the mean's temporal
fluctuation about its own time-mean is rescaled by 1 + λ (g − 1), g = min(3, sqrt(V_members / v_mean)) where V_members is
the members' average temporal variance and v_mean the mean's, **λ = 0.5**. The time-mean is invariant (MVPE untouched
algebraically); the interval head reads and brackets the PLAIN mean, so widths and coverage are k4's (the lo ≤ pred ≤ hi
guard widens 0.019 % of elements, SPS interval residual −0.00007). No target, no cross-window information; window-local.
Inference source `submissions/k4g/submission.py` (`_matched_gain`, `GAIN_DOSE = 0.5`, `GAIN_CAP = 3.0`).

**Why.** Averaging K members cancels the fluctuation they disagree on; the scorer's TKE pays for that damping while rel
and MVPE are mean-type functionals for which the mean is right. Held-out fold ensembles (step 6000, cell-balanced):
| | mim1 K=4 | far_drop K=4 | angle05 K=3 |
|---|---|---|---|
| dose 0.5: D / A | +0.019 / +0.034 | +0.024 / +0.040 | +0.062 / +0.105 |
| dose 1.0: D / A | +0.027 / +0.048 | +0.034 / +0.060 | +0.094 / +0.161 |
(D = direct weighted ΔQ, A = with the SPS accuracy prefactor; TKE −1.7…−7 %, rel +0.5…+2.6 %, MVPE 0.) Median, trimmed
mean, fixed scalar gains, coherence/reliability shrinkage, lead ramps and energy interpolation were priced too: none
beats the verified candidate by more than noise; the plain mean is not optimal on the composite objective. **Closed
neighbour:** `tke_restore` (archive: +0.002/+0.006 at β = 1, old recipe, K = 3, head bracketing the TRANSFORMED
prediction). Codex reproduced both formulas: identical TKE map at full dose; the interval policy explains ≈ 0.010 of the
gap in-sample, K and recipe the rest. In-sample on the packed weights (same windows): dose 0.5 rel −0.012, tke +0.126,
sps +0.032, N = +0.016; dose 1.0 N = +0.030. A100 N = 1 latency +0.05 ms. All profit depends on TKE: exact packed N₀
(TKE and its prefactor removed) is −0.009 / −0.019 at half / full dose.

**Prediction vs 923184 (board subscore deltas; judgement bands, not confidence intervals):** rel −0.025 [−0.060, −0.005];
tke +0.200 [−0.100, +0.500]; mvpe 0 [−0.001, +0.001]; sps +0.032 [−0.060, +0.120]; time −0.010 [−0.080, +0.080];
**final +0.017 [−0.060, +0.100]** (centre ≈ 81.822).

**Decision rule (replaces the blanket TKE veto for this candidate only).** Promote if the final improves AND
N (non-time, actual SPS once) ≥ +0.010 under both fitted weight sets, with rel ≥ −0.060, sps ≥ −0.060, |Δmvpe| ≤ 0.005.
Reject if N ≤ 0 or a channel floor fails; **do not respond by increasing the dose**. A smaller positive N or a
timing-only failure leaves 923184 in place and permits at most one exact repeat. A board pass does not prove
private-regime improvement; the Decision-Phase retrain re-derives the gain from the organisers' members.
Transfer risk, stated: the transform is applied after averaging so no later averaging cancels it (unlike the trained TKE
arms), the temporal mean and the interval estimator are preserved (coop25's failure modes), but member disagreement
measures training variability rather than missing physical variance, and wd03's board TKE reversal came from a fold
read that also said "unchanged". Slot proposed by Codex: **09-13**, ahead of the P0 repeat (09-14); incumbent repeat to
reserve. Owner decides.

**Standalone check on an idle A100 (pq, 2026-09-12, `readout/k4g_standalone_a100.json`):** `build/k4g.zip` (191,640,125
bytes; 207,362,725 extracted; sha256 c99727b364a332ef…) imports in 6.1 s with no repo modules, captures 9/9 graphs, outputs
finite and ordered; vs k4 on 17 windows: time-mean max abs diff 5.5e-5 (fp16), bounds max abs diff one fp16 ulp (guard
elements), temporal-variance ratio 1.0146; 4.04 ms per one-window call (k4 3.94 on the same card class). Ready to arm.

## Armed (owner's go, 2026-09-12 06:15 UTC)
`tools/submit_at.sh 2026-09-13T00:00:30Z build/k4g.zip` — log `research/audit_2026-09-11/claude/submit_k4g.log`; zip source
verified byte-equal to `submissions/k4g/submission.py`. Prediction and decision rule as above, recorded before the score.
Slot order after this: 09-14 P0 repeat · 09-15 meanfix (three-member calibration) · 09-16 `final` (on 923184 weights) ·
09-17 cc11 · 09-18 cc11h · 09-19 reserve (exact incumbent repeat, or k4g repeat if it passes narrowly).

## Board result — submission 924146, scored 2026-09-13 00:05 UTC: final 81.79118 (−0.015 vs 923184); **non-time +0.010, time −0.024**
| | rel | tke | mvpe | sps | time | final |
|---|---|---|---|---|---|---|
| 923184 (k4) | 94.853 | 78.977 | 94.1013087 | 43.841 | 92.164 (5.27 ms) | 81.806 |
| **924146 (k4g)** | 94.829 | 79.132 | 94.1013127 | 43.861 | 91.916 (5.64 ms) | 81.791 |
| Δ | −0.023 | **+0.156** | **+0.000004** | +0.020 | −0.248 | −0.015 |
Against the pre-registration: rel −0.025 [−0.060, −0.005] → −0.023 ✓; tke +0.200 [−0.100, +0.500] → +0.156 ✓; mvpe 0 → 0.000004
(**MVPE invariance holds on the board to seven digits**); sps +0.032 [−0.060, +0.120] → +0.020 ✓; time −0.010 [−0.080, +0.080]
→ −0.248 ✗ (the day's clock: 5.27 → 5.64 ms per sample, the same-artefact time noise the record prices at ±0.4 of the
time subscore); final +0.017 [−0.060, +0.100] → −0.015 ✓ inside the band.
**Rule:** N = +0.0104 (locked) / +0.0106 (DELTA) / +0.0101 (34-card) — at the +0.010 threshold under all three; channel
floors met (rel ≥ −0.060, sps ≥ −0.060, |Δmvpe| ≤ 0.005); **final did not improve, and the whole shortfall is the time
channel → "timing-only failure": 923184 stays the incumbent, no dose change, one exact repeat permitted.** The accuracy
channels of an identical zip are deterministic, so a repeat re-rolls only the clock: it beats 81.8056 whenever the day's
time_score exceeds 92.05 (t < 5.32 ms; 923184 drew 5.27, P0 5.34) — roughly even odds — and it delivers the owed
same-artefact time-noise calibration for free. Rel paid about twice the in-sample estimate (−0.023 vs −0.012), as the
held-out folds predicted; TKE arrived above the in-sample read (+0.156 vs +0.126). The transform transfers as a
mechanism; whether it pays depends on the true weights, which put it at the margin.

## Exact repeat armed (owner's go, 2026-09-13 19:00 UTC) — the one repeat the rule permits
`tools/submit_at.sh 2026-09-14T00:00:30Z build/k4g.zip` (sha256 c99727b364a332ef… = the archived 924146 zip, byte for byte);
log `research/audit_2026-09-11/claude/submit_k4g_repeat.log`. **Prediction:** rel 94.829, tke 79.132, mvpe 94.1013, sps
43.861 to the digit (deterministic); time_score = the day's clock, same-artefact band 91.7–92.6 (5.0–5.8 ms). New best
(> 81.8056) iff time_score > 92.05, i.e. t < 5.32 ms. **Reading:** the accuracy digits confirm determinism; the time
re-roll is the owed same-artefact time-noise calibration; a final above 81.8056 promotes k4g (non-time +0.010 met all
fits) and makes the gain part of the frozen recipe; a lower final leaves 923184 and closes the dose question at 0.5.

## Exact repeat — submission 925203, scored 2026-09-14 00:05 UTC: final 81.7707; accuracy identical to the digit, clock slower again
| | rel | tke | mvpe | sps | time (ms) | final |
|---|---|---|---|---|---|---|
| 924146 | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.8607297 | 91.916 (5.64) | 81.7912 |
| **925203** | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.8607354 | **91.704 (5.97)** | **81.7707** |
Determinism confirmed (sps differs at the seventh digit only). **Not a new best → per the rule k4g is not promoted;
923184 stays; the dose question closes at 0.5; no third slot for this artefact.** The per-sample clock reads
918211 5.25 · P0 5.34 · k4 5.27 (09-12) · k4g 5.64 (09-13) · k4g 5.97 (09-14): both k4g days are slower than every
non-P1 read, and the second is slower still. Two readings are possible — a platform drifting slower over these days,
or k4g running a slower path in the container (graph capture failing there and predict() falling back to eager; the
local A100 pair showed only +0.05 ms with 9/9 graphs). **Settle it before spending anything else on k4g: an exact
923184 repeat on 09-15** (same chain minus the gain; if it also reads ≈ 5.9 ms the platform drifted and k4g's non-time
+0.010 stands as an open option for the Decision-Phase package; if it reads ≈ 5.3 ms the gain path is slower in the
container and needs a capture/fallback fix). P0 repeat moves to 09-16.

## The clock question, settled without a slot (2026-09-14)
**Mechanism ruled out.** Paired idle-A100 measurement of the packed artefacts (`readout/eager_pair.json`): graphed N = 1
k4 4.188 ms vs k4g 4.173 ms (**the gain costs −0.015 ms, i.e. nothing**), both capturing 9/9 buckets; the eager path costs
**26–27 ms**, six times the graphed path. So a capture failure in the container would read ≈ 78 time_score, not 91.7:
**graphs are working there and there is no mechanism by which k4g costs +0.4…+0.7 ms.**
**The clock is the platform.** Scoring hosts and per-sample times of our ordinary shipped-chain submissions:
| id | host | ms | | id | host | ms |
|---|---|---|---|---|---|---|
| 915299 | rpw1 | 5.34 | | 921738 P0 (no GPU work) | rpw0 | 5.34 |
| 917205 | rpw7 | 5.49 | | 923184 k4 | rpw7 | 5.27 |
| 918211 | rpw3 | 5.25 | | 924146 k4g | rpw3 | 5.64 |
| 920181 | rpw1 | 5.35 | | 925203 k4g | rpw0 | 5.97 |
Mean 5.46 ms, sd 0.24 ms (≈ 0.15 of time_score, **0.014 of final**), range 5.25–5.97 — and a call with almost no GPU work
(P0, 5.34) sits mid-range, so the variation is harness/host, not our kernels. k4g's two draws are +0.29 and +0.62 ms above
the non-k4g mean: one ordinary, one a two-sigma day. **No third read of this artefact** (the registered rule permits one
repeat; a third would be selection on noise). The decision about shipping the gain rests on its non-time +0.010, not the clock.

## Build-gate fix (2026-09-14)
`tools/check_pack.py` previously read the transformed prediction as a 1.1e-3 MISMATCH, which would have failed
`train_final.sh`'s own release gate had the gain shipped. It now applies the same transform to its fp32 reference when
the submission module defines `GAIN_DOSE`, with the head still reading the plain mean. Verified on an idle A100:
k4 MATCH 2.81e-4, **k4g MATCH 2.83e-4**, final MATCH 2.81e-4, intervals matching in all three.
