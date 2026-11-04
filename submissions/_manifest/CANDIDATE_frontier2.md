# Candidate frontier2 (F2) — frontier-cell upweighting (pre-registered 2026-09-11, joint Claude + Codex round 5; screen not yet run)

**Mechanism.** Multiply the existing *real-window* sampling probabilities by 2 on the union of axis-adjacent training
cells (offsets ±1275 in Re or ±5° in AoA) around sim-grid cells that have no real training data (7575_0 excluded);
recomputed from each fold's training availability, including withheld cells even when their simulation is dropped.
No diagonal, distance tuning, repeated multiplication or difficulty weighting; sim sampling and modality counts
fixed; head sampling unchanged; all-ones weights must reproduce B's sampling path bit for bit. Full-data stencil:
29 cells, real sampling mass 35.0 % → 51.9 %. Closest completed controls: uniform trajectory sampling −0.004 direct
(2 runs, 2026-09-09); an earlier fixed-2× edge-importance proposal never run.

**Screen (≤ 15 A100-h, no slot).** B vs F2 on five folds — mim1, angle05 (sim dropped), far_drop, I12675 (sim
dropped), joint5/keep (withhold 17775/19050/20325/21600 at 5°, sim kept) — seed 0 then 1, matching B runs reused,
missing controls added; survivors add seed 2 and are scored as three-member means at checkpoint 6000.
**Gate** (cell-balanced stride-40 ΔQ): five-fold mean ≥ +0.025, far ≥ +0.015, angle and joint ≥ 0, interior ≥ −0.01,
every fold ≥ −0.02, positive with TKE zeroed; positive seed-1 mean before the ensemble extension. Any miss kills F2.
A survivor needs the separately funded thirteen-fold confirmation (≈ 57 h) before a slot.

**Prediction versus 918211** (Codex round 5 text): public total Δ 0.00 [−0.03, +0.03]; private-retrain planning
Δ +0.01 [−0.04, +0.05]. The mechanism predicts far-Re improvement without angle/interior regression. Board non-time
Δ ≤ 0 kills further deployment investment; public improvement alone does not establish private benefit.

## Screen read (2026-09-11, seeds 0 / 1, cell-aggregated ΔQ vs B at step 6000, `research/audit_2026-09-11/claude/readout/screens_read.log`)
| fold | seed 0 | seed 1 |
|---|---|---|
| mim1 | −0.021 | +0.009 |
| angle05 (drop) | +0.020 | **−0.042** |
| far_drop | +0.013 | +0.023 |
| I12675 (drop) | +0.004 | +0.003 |
| joint5 (keep) | −0.000 | −0.001 |
| **five-fold mean** | **+0.003** | **−0.002** |
Gate: mean ≥ +0.025 — **miss** on both seeds; far ≥ +0.015 passes (+0.013 is a near miss on seed 0, +0.023 on seed 1);
angle05 seed 1 breaks the −0.02 floor; MVPE error rises +0.5…+2.5 % on 8 of 10 reads (the frontier emphasis biases the
mean flow elsewhere). **F2 is killed under the pre-registered rule.** No ensemble extension, no slot. What survives as
a fact: far-Re gains at the +0.01–0.02 level from frontier emphasis, again paid at the novel angle — the same shape as
the mean-flow mixing result (`FINDINGS.md` §3). Cost: 14 A100 runs ≈ 3 A100-h.
