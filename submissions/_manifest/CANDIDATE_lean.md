# Screen LEAN — the loss without its dead terms, read as three-member ensembles with the gain (pre-registered 2026-09-17, owner's go; runs queued before this file was closed)

**Arm.** `--w-mse 0.0 --w-tmean 0.0 --w-spec 0.0` on the incumbent recipe (`kfold.CURRENT_BASE`, `--stop-at 6000`),
seeds 0/1/2 on mim1, angle05 (target-angle sim dropped) and far_drop; run dirs `bt_w1_LEAN_<fold>_s<seed>`; recipe
drift-checked by `launch_wave.py` (only `w_mse`, `w_tmean`, `w_spec` move). Controls: the existing `bt_w1_B_*` seeds
0/1/2 (the 09-13 re-read's B). Cost 9 runs ≈ 4 A100-h on pq (`research/lean_2026-09-17/`).

**Why it is not closed.** `lean` read **+0.010 on 4/4 mimic folds** per member on 2026-08-31 (`docs/MODEL.md`), under
the recipe before wd 0.03 and w_mvpe 0.3, and was never re-read at ensemble level with the gain: it is not one of the
nineteen arms of the 2026-09-13 systematic re-read. It is the only forecaster-side change in the record with a
measured positive sign that was never shipped or killed. Mechanism: the per-window relative-L2, TKE and SPS-bulk terms
are the scorer; MSE, time-mean and spectral terms pull the optimum elsewhere by a small amount.

**Instrument.** `research/audit_2026-09-11/claude/ens_combo.py <fold> --runs <3 seeds> --tag LEAN`: the three-member
mean at step 6000, plain and with the member-variance-matched gain at dose 0.5, cell-balanced stride-40 scoring on the
fold's held-out cells, Q = .4738 ΔS_rel + .1057 ΔS_tke + .0926 ΔS_mvpe vs B (the 09-13 instrument, so the read is
comparable to the nineteen-arm table); exact power-mean N reported alongside.

**Gate (fixed before the read).** Primary: **mean over the three folds of ΔQ with the gain (dose 0.5) ≥ +0.010**, every
fold ≥ −0.005, MVPE ratio ≤ 1.005 on every fold, rel ratio ≤ 1.000 on the mean, and — because this is a trained arm —
**TKE-zeroed ΔQ ≥ 0 on the mean** (the k4g rule permits TKE funding only for deterministic post-averaging transforms).
Kill: mean ΔQ_gain < +0.005, or any fold < −0.020, or MVPE worse than 1.005 on two folds. Between: closed as
sub-resolution, no further dose or variant. **Prior:** +0.005 [−0.010, +0.020] ΔQ; P(pass) ≈ 0.25.

**If it passes.** Retrain the four all-data members lean (`LEAN=1 bash tools/train_final.sh`, ≈ 1.3 h H100 + teachers
and student ≈ 30 min), build with the k4a2 inference code, check_recipe/check_pack/replay, and pre-register a board
read — predicted +0.005 [−0.010, +0.015] of final at a fixed clock — as the 09-20 freeze artefact only if every gate
passes by 09-19 evening; otherwise the incumbent recipe freezes and LEAN is recorded as a positive-but-unshipped screen.

## Queued 2026-09-17 (pq a100; job ids in `research/improve_2026-09-10/claude/wave_jobs.jsonl`)
Job ids 1789639356249…6965 (nine pq a100 jobs; all nine reached step 6000 by 2026-09-17T11:54Z — the queue drained faster
than expected).

## Read (2026-09-17 ~12:00 UTC; `research/lean_2026-09-17/read_lean.json`, combos in `research/audit_2026-09-11/claude/cells/combo_*_LEAN.json`)
| fold | ΔQ plain | ΔQ gain 0.5 | ΔQ gain, TKE zeroed | ΔN exact marginals | rel ratio | TKE ratio | MVPE ratio |
|---|---|---|---|---|---|---|---|
| mim1 | −0.0526 | **−0.0111** | +0.0173 | −0.0098 | 0.988 | 1.017 | **1.007** |
| angle05 | +0.0425 | +0.0197 | +0.0040 | +0.0190 | 0.997 | 0.991 | 0.999 |
| far_drop | +0.0080 | +0.0046 | −0.0034 | +0.0041 | 1.000 | 0.996 | **1.011** |
| **mean** | −0.0007 | **+0.0044** | +0.0060 | +0.0044 | | | |

**Verdict: KILL** (mean ΔQ with the gain +0.0044 < +0.005; mim1 −0.011 breaches the −0.005 floor; MVPE ratio > 1.005 on
two folds). The rel_l2 gain the 08-31 per-member read saw is real on every fold (ratios 0.988–1.000), but the
ensemble pays it back in fluctuation amplitude on mim1 (TKE 1.017 after the gain) and in MVPE drift on mim1 and
far_drop — the compression-at-ensemble-level signature of every trained forecaster lever this month. Closed at
ensemble level with the gain-aware read it never had; no dose or variant follows. The incumbent loss freezes.
