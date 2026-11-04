# Calibration canon0918 — the frozen method rebuilt from scratch on an A100 (pre-registered 2026-09-18 04:20 UTC, before the build finished; the 09-19 drift-branch slot)

**What.** `bash tools/train_final.sh` with every default (`NAME=canon0918 SUB=canon0918`): four members seeds 0–3
(8000-step cosine read at 6000), six SPS teachers seeds 0/11/21/31/41/51, the distilled student (seed 7, teacher-0
init, under-width ×3, 4000 steps), packed with `k4a2`'s inference code (gain 0.5, one head). Built on a pq
**A100-PCIE-40GB** (node9) — the evaluation card's silicon class — from an empty run root; gates in the log:
`check_pack --eval` MATCH, `check_recipe --replay-student` OK. This is the artefact the organisers' Decision-Phase
retrain of our method would produce on their hardware (same class ⇒ bitwise-reproducible members and heads).

**Why this slot.** The 09-18 control read 6.110 ms on 923184's own bytes: the clock drifted, the gain code is
exonerated, and no time-channel artefact is worth a slot. The registered drift-branch options were an exact A2 repeat
(measures only the clock) or a retrain calibration — this measures the one thing the record does not have: **how the
frozen method's fresh A100 build scores on the hidden set**, i.e. the public-board value of the Decision-Phase artefact.

**Prediction (exact power mean; judgement bands).** Members and heads are a different model from the H100-built
929146 (same seeds, no tensor equal — `REPRODUCE.md`); in-sample, A100 builds have read 0.01–0.03 below the H100
incumbent, almost all in SPS. rel 94.83 [94.78, 94.88] · tke 79.13 [79.00, 79.30] · mvpe 94.10 [94.05, 94.15] ·
sps 43.90 [43.75, 44.00]; **exact N vs 929146 = −0.010 [−0.040, +0.020]**. Clock at the new level ≈ 6.0 ms [5.6, 6.3]
→ **final ≈ 81.775 [81.72, 81.83]**. Not a promotion candidate: it can beat 926634 only on a clock draw below ≈ 5.6 ms.

**Decision rule.** **Quarantine** (build or transport fault; investigate, do not interpret) if |N| > 0.04 or any single
subscore departs from 929146 by more than 0.10. Otherwise the read is the retrain-variance calibration for the
Decision-Phase argument: record it in `REPRODUCE.md` and `STATE.md`. No seed selection follows from a good or bad read;
the frozen recipe is what ships, whatever this draw scores.

## Armed (conditional)
`research/audit_2026-09-17/claude/arm_canon0918.sh` waits for the build log's `CANON_DONE`, requires `MATCH` and
`OK --` in the log and the zip at `build/canon0918.zip`, records its sha256 here, then
waits for 2026-09-19T00:00:30Z. Log `logs/submit_canon0918.log`.

zip sha256 `8d8daa69cea758597b24676825e24686e1eaf7a728bb2dd28b05ee001005a832` (191628659 bytes), armed 2026-09-18T06:51:11Z

**Build receipt (2026-09-18 04:09 → 06:50 UTC, A100-PCIE-40GB node9, 2 h 40 min end to end).** Pack 207.3 MB;
`check_pack --eval` MATCH (packed vs runs 2.06e-4; interval width error 1.9e-3); `check_recipe` OK; `check_recipe
--replay-student` OK. Standalone fresh-process smoke of the zip: finite, ordered, N = 1 median 3.15 ms; vs k4a2 the
time-mean differs by 6.2e-3 and the temporal-variance ratio is 1.003 — a different model, as a different card class
must give. **In-sample read vs 929146 (val_aoa fixture, 187 windows):** rel +0.0033, tke +0.0019, mvpe −0.0009,
sps −0.0151 → **exact N −0.0018** at a fixed clock — inside the registered band [−0.040, +0.020] and consistent with
the record's "A100 builds read slightly lower, mostly in SPS". Board prediction unchanged: N −0.010 [−0.040, +0.020],
final ≈ 81.78 at ≈ 6.0 ms.

## Board result — submission 932643, uploaded 2026-09-19T00:00Z, scored 00:07 UTC (ingestion rpw7, scoring rpw6): **81.89314 — new best, +0.086, rank 4**
| | rel | tke | mvpe | sps | time (ms) | final |
|---|---|---|---|---|---|---|
| 929146 (H100-built A2, 09-17) | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.9453064676 | 91.675 (6.011) | 81.78851 |
| 926634 (retained best until today) | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.9414123258 | 91.875 (5.702) | 81.80680 |
| **932643 (A100 canonical rebuild)** | **94.8462054793** | **79.2328627255** | 94.0909748382 | **43.9749840516** | **92.507 (4.782)** | **81.89314** |
| Δ vs 929146 | +0.0168 | +0.1009 | −0.0103 | +0.0297 | +0.832 | +0.1046 |

**Against the registration.** Exact accuracy-only N vs 929146 = **+0.0245** (registered −0.010 [−0.040, +0.020]: inside
the quarantine bound, just above the band's upper edge — the fresh A100 draw is *better* than the H100 incumbent on
rel, TKE and SPS, worse on MVPE; the in-sample read had said −0.002, so in-sample under-priced this draw by ≈ 0.03,
the size the record gives retrain variance). Clock **4.782 ms**, below the registered drift band [5.6, 6.3] — the fastest
read of any artefact ever (923184's own bytes drew 5.27 on 09-12 and 6.11 on 09-18). At the 09-18 level this accuracy
would have scored 81.807; at the old 5.34 level 81.856. **The clock is a wide draw (4.8–6.1 ms across a week for
equivalent artefacts, sd ≈ 0.5), not a level**: every "band" this project priced from a level is retired.

**Verdict.** New retained best; `verify_best` pins 932643 (archived read-only on both copies). More important than the
rank: **this is the Decision-Phase artefact** — the frozen method rebuilt from scratch on the evaluation card's silicon
class through `tools/train_final.sh` with every default — and its hidden-set value is now measured: N +0.025 over the
H100 build, 81.893 on today's clock. `check_recipe` now defaults to this pack. Board 2026-09-19 05:00 UTC: roysegal 81.961
· modu-lemon 81.915 (09-19 03:13, +0.39 in three days) · np-user 81.898 · **us 81.893** · skabob 81.824; gap to first 0.068.
