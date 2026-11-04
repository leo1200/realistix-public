# Screen DS1 / DS2 — distil the four-member mean into ONE forecaster (pre-registered 2026-09-17, before the reads)

**Why reopen.** The ensemble decomposition (`FINDINGS.md` §2, `research/lean_2026-09-17/k_ladder_*.json`): members' error
fields correlate at ρ = 0.88–0.94; the four-member mean removes 4–9 % of a member's squared error; K = 1 → 4 is worth
rel −2…−4 %, MVPE −2…−3 %, TKE +2…+4 % (damping). A single model that recovers most of that variance-reduced target
would match the mean's point forecast at a quarter of the compute and with a simpler retrain contract. The archived
distillation (894861, 08-20) was *three* w96 students from w128+w160 teachers **plus a learned combiner**, under the
pre-Muon recipe; it lost −0.146 on the board with an unexplained MVPE collapse (−0.76). A single student of the mean,
same width, current recipe, was never measured. (`FINDINGS.md` §7 lists "distillation and combiners" from that era.)

**Arms** (recipe drift-checked by `launch_wave.py`; only the distillation flags move; runs `bt_w1_DS{1,2}_<fold>_s0`):
DS1 = incumbent recipe + `--distill-from <fold B members s0–s3 @ step 6000>` (student loss = composite(target) +
1.0 × composite(teacher mean); trainer semantics `tools/train_mixed.py`); DS2 = `--distill-only` (teacher term alone).
Teachers are the fold's own B members, trained on the same split (the trainer's leak guard). Folds mim1 / angle05 /
far_drop, seed 0; 6 runs ≈ 3 A100-h on pq.

**Instrument.** `research/lean_2026-09-17/k_ladder.py`-style Q (cell-balanced, 09-13 weights) of the single student vs
(a) the four-member B mean plain, (b) the B mean with the gain at dose 0.5, (c) a single B member; exact marginals
alongside. MVPE ratio vs the B mean is the anomaly watch.

**Gate (fixed now).** Primary: student Q ≥ Q(B mean + gain 0.5) − 0.005 on every fold and ≥ on the fold mean, with
MVPE ratio ≤ 1.005 on every fold (the 894861 failure mode) and rel ratio ≤ 1.00 vs the B mean on the fold mean. A pass
means the point forecast of one model matches the gained ensemble; then the SPS side is read with fresh fold heads
(no spread input) on the student vs the fold A2 student on the mean, and the clock value priced. Kill: mean Q below the
gained mean by > 0.02, or MVPE ratio > 1.01 on any fold (the anomaly reproduced). Between: closed as sub-resolution.
Prior: DS1 recovers 50–80 % of the K = 4 rel gain and keeps single-model TKE amplitude → Q ≈ gained-mean − 0.01
[−0.03, +0.01]; P(pass) ≈ 0.3. Value if it passes: −2.7 ms local (0 to +0.06 of final depending on the unresolved clock
slope) and a one-model retrain contract; accuracy neutral by construction of the gate.

## Queued 2026-09-17 (pq a100; `research/distil1_2026-09-17/arms_<fold>.json`)
Job ids 1789647615050…9128; all six reached step 6000 by 2026-09-17T13:15Z.

## Read (2026-09-17 ~13:20 UTC; `research/distil1_2026-09-17/read_ds.json`)
| fold | arm | Q student | Q B mean | Q B mean + gain | Q single B member | Δ vs gained mean | rel ratio vs mean | TKE ratio | MVPE ratio |
|---|---|---|---|---|---|---|---|---|---|
| mim1 | DS1 | 63.131 | 63.153 | 63.163 | 63.131 | **−0.032** | 1.000 | 1.013 | 1.003 |
| mim1 | DS2 | 63.126 | 63.153 | 63.163 | 63.131 | −0.037 | 1.000 | 1.016 | 0.996 |
| angle05 | DS1 | 63.783 | 63.767 | 63.829 | 63.814 | **−0.046** | 1.007 | 0.986 | 1.003 |
| angle05 | DS2 | 63.768 | 63.767 | 63.829 | 63.814 | −0.061 | 1.010 | 0.992 | 1.013 |
| far_drop | DS1 | 62.127 | 62.154 | 62.168 | 62.122 | **−0.041** | 0.999 | 1.011 | 1.025 |
| far_drop | DS2 | 62.131 | 62.154 | 62.168 | 62.122 | −0.037 | 0.997 | 1.017 | 0.999 |

**Verdict: KILL** (mean −0.040 / −0.045 vs the gained ensemble; kill line −0.02). The honest mechanism read: on mim1 and
far_drop the student *does* inherit the mean's rel_l2 (ratio 1.000 / 0.999 — the 2–3 % the ensemble buys), but it also
inherits the mean's damped fluctuation amplitude (TKE 1.1–1.7 % worse than the plain mean) and, being one model, has no
member variance for the gain to restore it; angle05 loses rel too (+0.7–1.0 %). Net, a distilled single model scores
exactly like a single B member (63.131 vs 63.131 on mim1). A student plus a learned amplitude target could in principle
close the TKE half — a two-step research path, not a freeze candidate. The 894861 MVPE collapse did not reproduce
(ratios ≤ 1.025). Closed for this phase; four members stay.
