# Screen head144 — interval-head width 144 / 192 at the current lr / τ, and a re-selection grid on 144 (owner's go 2026-09-13; gate fixed BEFORE the read)

**Why.** The head-width sweep (`hsw_w144/w192`, 2026-08) ran at lr 1e-4 / τ 0.15 and read +0.046 / +0.064 SPS over
width 96 on four mimic folds; it was rejected for putting 2–4× the head FLOPs into the timed region. P0/P1 show the
clock barely charges kernel work, and width was never re-selected at the shipped lr 8e-4 / τ 0.05, under which width 96
gained +0.30 SPS over its old tuning. Closed neighbour: time-priced, old tuning.

**Screen.** Held-out fold heads on the current-recipe B fold ensembles (seeds 0/1/2, step 6000) of mim1, angle05 (drop),
far_drop; `--train-split train --fixed-step` as the earlier sweeps (`hsw_*`); spread input; seed 0. Configs:
w96 (control, shipped constants) · w144 · w192 at lr 8e-4 / τ 0.05 / 4000 steps; on 144: lr 4e-4, lr 1.6e-3, τ 0.10,
τ 0.025, 6000 steps. 24 fits ≈ 3 A100-h (`research/audit_2026-09-11/claude/head_fit.sh`, runs `cleanclone/heads/h_<fold>_<cfg>`).
**Read:** each head's held-out `val_aoa.sps_score` (and coverage) at its fixed step, minus the w96 control on the same fold.
**Gate for building a candidate:** mean held-out SPS gain over the three folds ≥ +0.10 (≈ +0.025 final through the
sps weight), no fold < −0.05, coverage within ±0.02 of the control; a re-selected constant counts only if it also beats
plain w144. A passer is then refitted on the 923184 members (K = 4, all data, seed 0 exported, seed 11 nuisance), packed
with the incumbent inference (size: head 144 ≈ 59 MB vs 26 MB → pack ≈ 240 MB, under the cap), checked
(check_pack incl. intervals, extraction, paired A100 latency), and pre-registered for a slot with the head-config
transfer history stated (1.2× once, 0.02× once). Prior: +0.005 [−0.02, +0.02] final.

## Read (2026-09-13 06:07 UTC, 24 held-out fits on pq A100; `research/audit_2026-09-11/claude/readout` via `head_table.py`) — FAIL
Held-out val_aoa SPS at the fixed step, Δ vs the width-96 control (mim1 / angle05 / far_drop; mean):
| config | Δ sps | mean |
|---|---|---|
| w144 (lr 8e-4, τ .05) | −0.579 / +0.055 / +0.085 | **−0.146** |
| w192 | −0.842 / −0.042 / −0.871 | −0.585 |
| w144, lr 4e-4 | −0.004 / −0.001 / +0.064 | +0.020 |
| w144, lr 1.6e-3 | −0.747 / −0.273 / −0.787 | −0.602 |
| w144, τ 0.10 | −0.015 / +0.124 / −0.100 | +0.003 |
| w144, τ 0.025 | −0.061 / +0.050 / +0.048 | +0.013 |
| w144, 6000 steps | −0.252 / −0.172 / +0.033 | −0.130 |
Coverage moves ≤ 0.009 everywhere. Gate (mean ≥ +0.10, no fold < −0.05): nothing passes; the best re-selected variant
(w144 at half the learning rate) ties the control (+0.020, inside the fold noise of ≈ ±0.05 sps). Wider heads at the
shipped learning rate lose sharply on mim1 (−0.6 / −0.8): the extra capacity fits the training cells' residuals and
the held-out cells pay. **Closed:** head capacity is not the interval channel's limit at the current members; the
shipped width-96 / lr 8e-4 / τ 0.05 / 4000-step head stays. The archive's +0.06 at old tuning was the old learning
rate's under-fit, since recovered by lr 8e-4 at width 96. Cost ≈ 3 A100-h. No candidate built.
