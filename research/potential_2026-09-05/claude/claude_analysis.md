# Claude's independent pass (2026-09-05) — measurements

## A. Is the slow (meandering) component data-limited or information-limited?
Instrument: /tmp/bt/meander_gap.py — each mimic fold's own 3-member `kf_tke1em8` ensemble,
scored on (i) the fold's TRAINING trajectories at stride 40 (windows it trained on, 1462-1471
windows / 71 traj) and (ii) its held-out cells (201-210 windows / 10 traj). rho is the
window-time-mean deviation correlation of FINDINGS §6 (climatology removed per trajectory).

| fold | rho train | rho held-out | rho_fluct train / held-out | rel_l2 train / held-out | mean share of sq. error |
|---|---|---|---|---|---|
| mim1 | 0.758 | 0.659 | 0.709 / 0.614 | 0.0710 / 0.0749 | 0.28 / 0.28 |
| mim2 | 0.741 | 0.853 | 0.695 / 0.706 | 0.0733 / 0.0638 | 0.27 / 0.30 |
| mim3 | 0.778 | 0.687 | 0.715 / 0.662 | 0.0693 / 0.0767 | 0.26 / 0.28 |
| mim4 | 0.753 | 0.727 | 0.705 / 0.653 | 0.0709 / 0.0728 | 0.27 / 0.28 |

Controls: width-128, 12k-step single model on a10_re3 (`kf_cap_w128_12k_a10_re3_s0`):
rho train 0.692 / val_aoa 0.702 / val_re 0.748. EMA-free arm on mim1 (`kf_ema00_mim1_s[01]`,
2 members): rho train **0.806** / held-out **0.663**; rel_l2 train 0.0688 (better) / held-out
0.0773 (worse than the EMA arm's 0.0749).

Reading. A 68 M-parameter ensemble that has seen every one of these training windows ~3x
reaches rho 0.74-0.78 ON ITS OWN TRAINING WINDOWS. Memorisation of slow-mode realisations
(my H1) is refuted: the in-sample ceiling is nearly as low as the out-of-sample one. Removing
EMA buys +0.05 rho in-sample and loses it out-of-sample -- the regulariser is already at the
overfitting margin on the slow mode. Fold-to-fold the held-out rho ranges 0.66-0.85 with the
in-sample value inside that range, so the dominant limit is the predictability of the
meandering from a 20-frame planar window, with a modest regime-transfer gap (~0.1 rho on the
harder folds) on top. Physically consistent: the foil is tapered (100 -> 20 mm) so the wake is
strongly three-dimensional, the leading edge sits on the upstream boundary, and PIV samples one
plane -- the forcing of the meandering is largely out of frame. That is aleatoric for ANY
single-window predictor, legal or not.

Data side (/tmp/bt/slow_dof.py, pure data): the 20-frame running-mean deviation decorrelates
with integral timescale 18.6 frames (0.37 s) median; effective independent slow-mode samples
~22 per trajectory, ~2200 in the release. Not a tiny sample -- the "few hundred realisations"
premise of H1 was wrong by 4x, and the input/output window-mean correlation from data alone is
0.055-0.16 (persistence is worthless; the model's 0.75 comes from reading the instantaneous
structure, exactly as §6 says).

Consequence for the budget: the §6 sensitivity table (rho 0.85 -> +0.33) describes headroom
that no 20-frame predictor has been shown to reach even in-sample. Mean-component share of
squared error is ~0.27-0.30 here (not 55 %; §6's figure was a different split/vintage), so
even the illegal ceiling is smaller than the table suggests.

## B. What this says about the accuracy channels overall
rho_fluct (the TKE-bearing component) shows the same in-sample/out-of-sample parity
(0.70-0.72 train vs 0.61-0.71 held-out). Per-band coherence declines 0.85 -> 0.32 with k
(model_analysis §2.2; Codex reproduces 0.73 -> 0.31 on mim4). The forecaster sits at a
predictability floor of its input on both components; the frontier tke/rel_l2 has been
measured at zero five ways. There is no accuracy "vein" left that a better map over the same
input could plausibly open by more than the ~0.1-rho regime-transfer gap on hard folds --
and the hidden set's cells are mostly interior or one step out (FINDINGS §0c).

## C. Where the remaining potential actually sits (my ranking)
1. time_score: the only channel with a mechanical ceiling. 5.34 ms/sample now (92.11). Local
   per-call cost of the shipped graph path is ~2.2 ms on H100 / ~3.3 ms on A100 with h2d+d2h
   0.16 ms and host work 0.008 ms (Codex's astra_runtime). The container's extra ~2 ms is
   UNEXPLAINED: harness overhead (not ours, floor) or slower kernel chain (ours, reducible).
   Decisive, cheap, legal test: one "floor probe" submission whose predict() returns
   persistence + default band from numpy in ~0.05 ms. Its time_score reads the harness floor
   directly; the board keeps the best score so rank is unaffected; cost = one daily slot.
   If floor <= 1.5 ms: a fused/persistent-kernel inference at ~1.5 ms total is worth up to
   +0.3 final (time 95.7) -- the largest priced lever anywhere. If floor ~3.4 ms: the channel
   is exhausted at +0.1 and the project should stop spending on kernels.
2. Decision Phase (where prizes are decided): unseen regimes + judged novelty/cleanliness.
   No current-recipe number exists for INTERIOR-angle interpolation (folds a05_re2/a10_re3 were
   last run under old recipes). Worth one 2-fold x 2-seed measurement to put in the report.
3. Accuracy: no breakthrough available from the same input; the honest contribution is the
   evidence above (predictability floor measured in-sample) -- a publishable negative that
   strengthens the report's "design for unseen regimes" story.

## D. Whole-angle holdout: the largest measured generalisation penalty in the project, unmeasured under the current recipe
In-sample rel_l2 by angle (mim1 ensemble on its training trajectories, /tmp/bt/per_angle.py):
AoA 0: 0.0435, 5: 0.0450, 10: 0.0718, 15: 0.0898, 20: 0.0967 (difficulty rises 2.2x with angle).
Archived whole-angle holdouts (old recipe, `kf_base_*`, 2 seeds): AoA=5 held out 0.052 (+16 % vs
in-sample), AoA=10 held out 0.087-0.090 (+22 %), AoA=20 held out 0.115-0.122 (+20 %).
Compare the mimic-cell penalty (fold-level, held-out minus in-sample): -13 % to +11 %.
So a NOVEL ANGLE costs ~20 % of relative error even when interpolated between measured angles,
i.e. rel_l2_score ~ -0.7 and ~ -0.35 final before the sps prefactor -- an order of magnitude above
every board-level lever left, and exactly the private set's stated geometry ("novel angles of
attack"). Current-recipe runs on a05_re2 and a10_re3 (seed 0) are training now
(`bt_cur_a05_re2_s0`, `bt_cur_a10_re3_s0`) to give the first current-recipe number.

### D2. Current-recipe whole-angle holdouts (seed 0, run 2026-09-05, `bt_cur_a05_re2_s0`, `bt_cur_a10_re3_s0`)
rel_l2 raw error, held-out angle (val_aoa/test_aoa) | held-out Re (val_re):
a05_re2: 4k 0.0515/0.0512 | 0.0800; 6k 0.0522/0.0520 | 0.0799; 8k 0.0534/0.0534 | 0.0808
a10_re3: 4k 0.0849/0.0836 | 0.0741; 6k 0.0863/0.0846 | 0.0743; 8k 0.0888/0.0871 | 0.0752
a10_re3 mvpe (held-out angle): 0.0811 (4k) -> 0.0863 (6k) -> 0.0930 (8k); val_re mvpe flat ~0.063.
Penalty vs in-sample per-angle reference: +16 % (5 deg), +18-20 % (10 deg) -- identical to the old recipe
(kf_base 0.052 / 0.087-0.090). Recipe-invariant. Held-out-ANGLE error rises monotonically after step
4000-5000 while held-out-Re does not: late-schedule specialisation to the seen angles.
