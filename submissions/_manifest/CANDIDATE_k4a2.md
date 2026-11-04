# Candidate k4a2 — the six-head interval ensemble distilled into ONE head (pre-registered 2026-09-16, joint Claude + Codex; NOT uploaded)

**What.** 923184's four members and the board-tested dose-0.5 gain, unchanged; `submission.py` is **k4g's byte for
byte** (md5 `188f1de3…`), i.e. the single-head inference path. The only change is the head's weights: six teacher heads
(seeds 0, 11, 21, 31, 41, 51, shipped head recipe, all-data) are averaged, and a single student head is fitted to that
average — teacher-0 initialisation, 4000 steps, log-space regression on `below`/`above` with under-width penalised 3×,
seed 7. **The teachers are training artefacts; inference runs one head.**

**Why.** 926634 put the two-head output average on the board (+0.0807 SPS, new best) but the second head costs ~1.1 ms
and the clock took back 90 % of the gain. `FINDINGS.md` §2b established the interval *policy* is already optimal, so
the averaging gain is pure estimator-variance reduction — a fixed, smooth target a single student can carry.
Held-out folds (fresh outer-training-only heads on the frozen K4 fold ensembles), ΔSPS vs the single seed-0 head:

| estimator | mim1 | far_drop | angle05 | mean | inference |
|---|---|---|---|---|---|
| shipped pair (926634) | +0.0742 | +0.1251 | +0.1003 | +0.0999 | two heads |
| avg6 (the teacher) | +0.1186 | +0.1737 | +0.1541 | +0.1488 | six heads |
| **A2 student** | **+0.0900** | **+0.1704** | **+0.1066** | **+0.1223** | **one head** |

A2 captures **82 %** of avg6 and beats the shipped pair on **every fold** (+0.0158 / +0.0453 / +0.0063), at half the
pair's head cost. Registered gates PRIMARY (≥ +0.0998) and CAPTURE (≥ 0.70) both pass; `research/distil_2026-09-16/PREREG.md`.
A first attempt at 4000 steps with a fresh init captured only 40 % and failed; that was a training-budget artefact, and
the "approximation floor" diagnosis in round 1 was wrong.

**Build gates, all passed** (`research/distil_2026-09-16/build_a2.sh` log, idle A100):
* `build_ckpt` on the shipped `ens4` members → 207,362,725 bytes extracted, **identical to k4g's footprint**, cap 268,435,456.
* Standalone, fresh process, no repo modules: 9/9 graphs, finite, ordered, **time-mean max abs diff vs k4g = 0.0 and
  temporal-variance ratio exactly 1.0** — the point forecast is bit-identical, so rel/tke/mvpe are algebraically fixed
  to 926634's values and only SPS and time can move. Bounds differ by 2.26e-2 (the new head).
* `check_pack --eval` **MATCH** (packed-vs-runs 2.33e-4; intervals vs `student_A2` relative width error 1.78e-3).
* **Latency, ABBA-balanced** (three cycles of k4g/k4a2/k4a2/k4g, fresh process, 300 N=1 calls, quiet A100):
  k4g **4.009**, k4a2 **3.991** — delta **−0.018 ms**, identical as the architecture requires, against the two-head
  incumbent's 5.157. A first fixed-order A,B,C measurement showed k4a2 +0.245 ms in 4/4 cycles; that was drift
  confounded with position, not an artefact of the pack, and the balanced design removes it. The packs are structurally
  identical (head_width 96, head_spread True, 80 head tensors, 26,247,904 head bytes, fp16, four members) so there is
  no mechanism for a latency difference.

**Prediction vs 926634 (the current board holder).** rel **94.8293790855** / tke **79.1320014641** / mvpe
**94.1013126942** — all three algebraically fixed, as 926634 itself demonstrated to ten digits.
SPS **43.9596** (+0.0182 = the +0.02247 incremental fold gain at the 0.81 transfer 926634 measured), 90 % band
[43.9096, 44.0096]. Time: one head, so the k4g-class clock; at the eight-read ordinary centre 5.456 ms → 92.037.
**final ≈ 81.827**, band [81.794, 81.862]. Non-time increment **+0.0046** under both fits.
Priced against the incumbent at each clock slope: **+0.0046 (slope 0) / +0.0154 (0.2) / +0.0208 (0.3) / +0.0240 (0.36)**
— **positive at every slope**, which is the point: it is strictly better on SPS *and* strictly cheaper, so unlike the
two-head pair it does not depend on resolving the slope. Expected rank 2–3; it does not close the 0.0519 gap to first.

**Decision rule (registered before the score).** **Kill if board SPS ≤ 43.9414123258** (926634's value) regardless of a
favourable clock — a better final on a lucky clock does not validate the head. Promote if the final improves AND
SPS > 43.9414. **Quarantine** (suspect a build fault, do not interpret) if rel / tke / mvpe depart from 926634 by more
than 0.005. A timing-only failure earns no automatic repeat. No automatic search over K, steps, seed or under-width
weight follows from this registration.

**Stated risks.** A2 loses to the pair on **7 of 34 overlapping fold-cell readings**, worst **−0.142 SPS at 13950_5**
(angle05); every fold's equal-cell mean is positive. That supports a submission, not a claim of universal improvement,
and the private set is what pays. The transfer factor 0.81 rests on **one** board point (926634). The Decision-Phase
retrain must reproduce the whole contract — four forecasters → six fresh SPS teachers → one distilled student → teachers
discarded — which the current `train_final.sh` default does not do.

## Armed (owner's go, 2026-09-16 19:48 UTC)
`research/distil_2026-09-16/arm_k4a2.sh` — waits for 2026-09-17T00:00:30Z, then submits
`build/k4a2.zip`, sha256 **verified before sending** against the audited
`ec556824be90b6eef8be157621f18c6aecf7e2191227b21ee37e9759add25127` (CRC clean; 207,362,725 bytes extracted,
61,072,731 below the cap; `submission.py` md5 `188f1de3…` = k4g's). Log
`logs/submit_k4a2.log`.
Hardened against the 09-15 outage that cost a slot: log off the home quota, `timeout` on every network call, quota
re-checked before every attempt so a submission that succeeds but reports badly cannot be sent twice, and retries to
23:45Z. Prediction, kill number and quarantine rule as registered above, recorded before the score.

## Independent audit (Codex, 2026-09-16 14:17 UTC; `research/distil_2026-09-16/codex_audit/`)

**Verified.** An independent all-data training **replay reproduced all 80 fp32 student tensors byte for byte**, and an
independently retrained teacher (seed 21) reproduced exactly. The six teacher digests, seeds, step 4000,
`train_split=all` and four member hashes all check out against the actual files; the replay used 81 ordered release
trajectories, all frames, genuinely unset holdout variables. The stem is [96, 161, 3, 3] = 40 window + 40 forecast +
1 log-scale + 80 spread, so `head_width=96` / `head_spread=True` are correct. Every member tensor reproduces the
folding/casting exactly; every head tensor equals the student's fp16 cast. Archive sha256
`ec556824be90b6eef8be157621f18c6aecf7e2191227b21ee37e9759add25127`, 207,362,725 bytes, 61,072,731 below the cap.
Nine graph buckets captured; N = 1…129 finite and ordered; injected capture/runtime failures fall back correctly.
All 187 release fixture windows give point predictions identical across k4g, A2 and the pair. Independent ABBA timing:
k4g 3.975014, A2 3.975237, **paired difference +0.00036 ms**. Independent packed rescoring reproduces the fold gain
(+0.01804 / +0.04350 / +0.00472, mean **+0.02209** vs the registered +0.02247).

**The gates are weaker than the result.** Mutation tests: a zeroed or substituted head in the pack, a flipped member
bit, a removed head key and wrong spread metadata are all **rejected** — but a **source student replaced by seed-0
weights with copied metadata**, a **genuine step-2000 snapshot**, **wrong-member weights with copied provenance**, and
**bad or duplicated teacher digests** all **pass both `build_ckpt` and `check_pack`**. Packing fidelity is not training
provenance. A2 is credible because of the replay, not because the chain would have caught these.
Also found and fixed: `build_a2.sh` (and the four other round scripts) lacked `set -e`, so a failed check could still
be followed by `BUILD_A2_DONE`. Still open: the old standalone script prints statistics without asserting;
`check_pack` guards its reference intervals against the **plain** mean while shipped inference guards against the
**gained** mean; historical fold students lack teacher-digest stamps; concurrent fold scorers share
`measure_<fold>.json` before renaming (a race hazard, no corruption demonstrated).

**Corrections to this file's own claims.**
* "Positive at every slope" is right about the *clock* (A2 is strictly cheaper) but **is not unconditional once
  transfer uncertainty is added**. The 0.806 transfer factor is **one board intervention**, not a law, and there is no
  defensible confidence interval for it. Under the harsher reading — discounting A2's whole gain above h0 rather than
  only its increment above the observed pair — A2 crosses the pair's SPS at transfer **0.658**, and at transfer 0.5 it
  gives SPS 43.922, i.e. **−0.0049 of final before the latency credit**. The ~1.1 ms saving still covers that at slope
  0.3, but not at slope 0. The honest statement is: positive under central assumptions, and in the worst joint corner
  (transfer 0.5, slope 0) about −0.005.
* The 0.0189-per-local-ms exchange rate is **conditional on slope 0.3**, not independently established.
* The negative-cell count is worse than stated: **10 of 34 in packed evaluation**, not 7 of 34 in source. At the worst
  cell (13950_5) **avg6 itself loses 0.046 to the pair** and A2 loses a further 0.096 to avg6 — better distillation
  cannot remove the teacher-target component. The fold suite has 34 readings over only **28 distinct cells**, shares
  training data, and does not cover fractional or beyond-range angles or a new joint Re/AoA corner.

**Tunables, re-examined (none change inference cost).** Teacher initialisation is **load-bearing** at the 4k budget
(fresh symmetric 4k 0.0589 → teacher-init symmetric 4k 0.1160). Seed is not lucky (seed 17 gives 0.1232 vs seed 7's
0.1226). Under-width 3 adds ≈ +0.0066 fold; 3 is not an identified optimum. K by fixed-prefix averaging:
0.10011 / 0.12486 / 0.13522 / 0.14905 at K = 2/3/4/6; K = 12 would remove only another 1/12 of variance.
Naive teacher **weight** averaging is catastrophic (−4.50 mean) — independently initialised features are not aligned.
Distilling the **gained** bounds is not a free improvement and must not be bundled in.
**4000 steps is adequate but not optimal:** the A1+A2 combination (teacher-init, asymmetric ×3, **12 000 steps**) reads
**+0.1395** fold mean, +0.0169 over A2 (≈ +0.0035 final), captures 93.6 % of avg6 and improves 32/34 cells — but its
worst angle cell drops to −0.155 vs the pair. It has no all-data packed artefact or retrain certificate, so it is
**retained for a pre-registered fresh-geometry comparison, not for a slot**.

**Audit's recommendation, accepted:** arm the frozen, checksum-pinned A2 for 09-17, then prioritise the private
retraining contract. Registered kill rule unchanged (`board SPS ≤ 43.9414123258` kills the *replacement of the pair*);
below h0's 43.8607297194 kills this student configuration outright. Between the two, the student keeps an interval
benefit and may still be an economic latency trade. A lucky-clock final is not evidence of better intervals.

## Board result — submission 929146, uploaded 2026-09-17T00:00:49Z, scored 00:05 UTC: final **81.78851** (not a new best)
Ingestion rpw2, scoring rpw1.

| | rel | tke | mvpe | sps | time (ms) | final |
|---|---|---|---|---|---|---|
| 924146 (k4g, h0) | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.8607297194 | 91.916 (5.639) | 81.79118 |
| 926634 (k4h2g05, pair) | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.9414123258 | 91.875 (5.702) | 81.80680 |
| **929146 (k4a2, student)** | 94.8293790855 | 79.1320014641 | 94.1013126942 | **43.9453064676** | **91.675 (6.011)** | **81.78851** |
| Δ vs 926634 | 0 | 0 | 0 | **+0.0038941** | −0.1994 | −0.01829 |

**Against the registration.** rel / TKE / MVPE reproduce 926634 to ten digits: the quarantine rule did not fire, the
point forecast is confirmed bit-identical, and SPS and the clock are the only channels that moved. **SPS 43.9453 >
43.9414: the kill rule did not fire** — the single student is at least the two-head pair on the hidden set. But the
increment over the pair is **+0.0039** against a predicted +0.0182 (band [−0.032, +0.068]: inside, low side). The
student-over-h0 gain is **+0.0846** (fold +0.1223 → transfer 0.69×); the pair's was +0.0807 (fold +0.0999 → 0.81×).
So the pair's gain transferred and the *extra* K = 6 averaging the student carries did not (increment transfer
≈ 0.17×). N vs 926634 = +0.0010 under both fits.

**Verdict: NOT PROMOTED (timing-only failure), head VALIDATED.** The final fell 0.018 below 926634 because the clock
drew **6.011 ms**, the slowest read since the ungraphed eager path (09-04), for an artefact whose local latency is
identical to k4g's (ABBA paired difference +0.0004 ms, Codex) and 1.1 ms *below* the two-head pair it replaces.
Per the registration a timing-only failure earns no automatic repeat. What the read establishes:
* The distilled single head is a valid replacement for the pair at one head's cost: same or better SPS, strictly
  less GPU work. It should be the head in the Decision-Phase contract (`RETRAIN_CONTRACT.md`) regardless of the
  public rank; the incumbent on the board stays 926634 only because of the clock draw.
* **The clock is not a slope on local GPU-ms.** Four reads on the k4g inference code (k4g 5.639 / 5.966, pair 5.702,
  A2 6.011) span 0.37 ms with the two-head artefact in the *middle*; the one-head A2 read is the slowest. Whatever the
  container charges, it does not follow the second head's +1.05 local ms. See `research/audit_2026-09-17/claude/AUDIT.md`
  for the level shift between the pre-gain and post-gain code chains.
