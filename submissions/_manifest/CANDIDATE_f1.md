# Screen F1 — width-64 FNO as a low-weight donor to the four-member U-Net ensemble (owner-approved 2026-09-17; registration `research/audit_2026-09-17/codex/round5/PREREG_FUTURE.md`; read `codex/round6/F1_READ.md`)

**Arm.** Historical four-block FNO (modes 16×17) restored from commit 36a08c1 at width 64, complex weights as explicit
real/imaginary pairs (Muon-safe), the incumbent recipe otherwise (normalisation, anchor, loss, Muon, EMA, 8000-step
cosine read at 6000); U-Net path proven byte-identical (identity smoke 164/164 tensors). Seed 0 on mim1 / angle05 /
far_drop; 0.11 A100-h per fold. Read: fixed 0.10 donor weight into the B mean with the weighted member-variance gain
(dose .5, cap 3) vs B + gain; ρ and RMS ratio vs B; cell-balanced Q under the 09-13 weights and the exact marginals.
Claude's independent reader (`research/distil1_2026-09-17/read_donors.py`) reproduces every number.

| fold | RMS ratio vs B | ρ (Pearson) | donor w = 0.10 ΔQ (09-13 / exact) | fixed 0.20 | MSE-optimal w* / ΔQ | rel ratio | MVPE ratio |
|---|---|---|---|---|---|---|---|
| mim1 | 1.053 | 0.874 | −0.0017 / −0.0011 | −0.0089 | 0.30 / −0.0190 | 0.995 | 0.994 |
| angle05 | 1.074 | **0.796** | +0.0044 / +0.0047 | +0.0031 | 0.33 / −0.0046 | 0.992 | 0.996 |
| far_drop | 1.054 | 0.902 | +0.0075 / +0.0078 | +0.0103 | 0.24 / +0.0102 | 0.997 | 0.990 |
| **mean** | 1.06 | 0.86 | **+0.0034 / +0.0038** | +0.0015 | | 0.995 | |

**Verdict: KILL** (mean ΔQ ≤ +0.01; one fold negative; mean rel ratio 0.995 > the 0.99 floor). What it measured that the
record did not have: under the *current* recipe the FNO is only 5–7 % worse than the U-Net (the August FNO was 35 %
worse) and 0.80–0.90 correlated with it — the least correlated accurate family we have — so its optimal blends are
positive on two folds, but the direct TKE channel pays −0.0065 and the total stays at a third of the gate. Rel improves
0.5 % on the mean (the requirement was 1 %). Every registered planning band held except the RMS ratio (better than
predicted). The F1 code (fno.py, `arch=fno`) stays in the tree as an inert research family; no seed-1 replication, no
CNO / mode / weight sweep. A heterogeneous donor is now measured under the current recipe: closed at this budget.
