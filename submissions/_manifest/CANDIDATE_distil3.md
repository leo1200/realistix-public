# Screen DS3 — distil the GAINED four-member mean into one forecaster (pre-registered 2026-09-17 evening, before the reads)

**Why.** DS1/DS2 (`CANDIDATE_distil1.md`) showed a same-width student of the plain mean inherits the mean's rel_l2 on
mim1 / far_drop (ratios 1.000 / 0.999) but also its damped fluctuation amplitude (TKE +1.1–1.7 % vs the mean), and as
one model it has no member variance for the post-averaging gain to restore — so it scored like a single member. DS3
makes the distillation *target* the gained mean: the teacher mean's per-cell fluctuation is rescaled toward the
teachers' average variance at dose 0.5 (the shipped gain's maths, `--distill-gain 0.5` in `tools/train_mixed.py`; new
flag, inert at 0, identity-smoked against the pre-patch tree: history identical, 164/164 tensors bitwise). The student
learns a full-amplitude target with the mean's rel. Mechanism: the ensemble effect (rel from averaging, TKE from the
gain) compressed into one model — the owner's "distil the ensemble effect" question.

**Arm.** Incumbent recipe + `--distill-from <fold B members s0–s3 @ 6000> --distill-gain 0.5` (student loss =
composite(target) + composite(gained teacher mean)), folds mim1 / angle05 / far_drop, seed 0; runs `bt_w1_DS3_<fold>_s0`;
3 × 0.45 A100-h on pq.

**Instrument and gate (fixed now).** `research/distil1_2026-09-17/read_ds.sh` extended to DS3: single-student Q vs
Q(B mean + gain 0.5), cell-balanced, 09-13 weights and exact marginals; MVPE ratio vs the B mean. Pass: student ≥ gained
mean − 0.005 on every fold and on the mean, MVPE ratio ≤ 1.005 every fold, rel ratio ≤ 1.00 mean. Kill: mean < gained
mean − 0.02 or MVPE > 1.01 on any fold. Prior: Q ≈ gained mean − 0.015 [−0.04, +0.005]; P(pass) ≈ 0.2. If it passes:
SPS read with fresh single-model fold heads vs the A2 student on the mean; the value is −2.7 ms local and a one-model
contract, accuracy-neutral by construction.

## Read (2026-09-17 ~15:00 UTC; `research/distil1_2026-09-17/donors_<fold>.json`, reader `read_donors.py`)
| fold | ΔQ student vs gained B ensemble | ΔQ vs a single B member | rel ratio vs B mean | TKE ratio | MVPE ratio | ρ(student, B mean) |
|---|---|---|---|---|---|---|
| mim1 | −0.016 | +0.017 | 1.009 | 0.988 | 1.009 | 0.989 |
| angle05 | −0.027 | +0.002 | 1.018 | 0.974 | 1.010 | 0.985 |
| far_drop | −0.018 | +0.025 | 1.008 | 0.985 | 1.010 | 0.988 |
| **mean** | **−0.020** | **+0.015** | | | | |

**Verdict: KILL** at the −0.02 line (−0.0201), no fold within the −0.005 pass margin; MVPE ratios 1.009–1.010 exceed
the 1.005 floor on every fold. The mechanism half-worked: the gained target fixes the amplitude (the student's TKE is
now 1.2–2.6 % *better* than the plain mean's) and the student beats a single member by +0.015 on average — it carries
≈ 40 % of the gained ensemble's advantage — but its rel_l2 sits 0.8–1.8 % above the mean's (DS1's plain-mean student
matched the mean's rel on two folds), so the two halves of the ensemble effect were not captured together. A single
model is also ≈ 0.99-correlated with the mean: as a donor it adds nothing (+0.001). Closed for this phase. The
remaining single-model route (student of the plain mean + a learned amplitude target) is a follow-up project.
