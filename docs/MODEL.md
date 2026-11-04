# The shipped model, component by component

What ships, what each component buys, and how sure we are — one table per layer of
the system, every number tagged with the recipe it was measured under. Where a
headline number predated the current recipe it has been **re-measured on
2026-08-31** (four mimic folds × two seeds, paired against the shipped
configuration, the same rig every recent decision used); rows still carrying an
old vintage say so.

The architecture drawing is [`model.png`](model.png) (source `model.tex`); the
mechanism-level anatomy is [`docs/archive/model_analysis.md`](archive/model_analysis.md).

**Units.** "Fold Δ" is the mimic-fold accuracy-only delta (`ACC_NO_SPS` weighting),
resolution ~0.02–0.03. "Board" numbers are final-score deltas from paired
submissions. The two scales are *not* interchangeable: accuracy transfers at 0.5–3× with the
sign occasionally flipping, and the interval-channel coefficient once quoted here
(1.22 ± 0.32) over-predicted 911563 by ~40× (`docs/archive/FINDINGS_2026-09-10.md` §0b).

## The system in one paragraph (rewritten 2026-09-17)

Four width-96 U-Nets (mults 1,2,4, ~22.65 M each; seeds 0–3), trained jointly on all 81
measured and 100 simulated trajectories with a per-modality affine at every
normalisation site, per-window RMS-speed normalisation, a loss assembled from the
scorer's own subscores, Muon on the matrix parameters, EMA 0.998, an 8000-step
cosine taken at step 6000. At inference the four forecasts are averaged and each
cell's temporal fluctuation about its time-mean is rescaled half-way toward the
members' own average temporal variance (the "gain", dose 0.5: target-free, MVPE-
invariant, exact board value +0.0094). The interval estimator is one width-96 head
that carries the *average* of six independently fitted SPS-reward heads (teachers,
seeds 0/11/21/31/41/51; distilled with teacher-0 initialisation and an asymmetric
log-width loss, `tools/distil_head.py`); it reads the window, the plain ensemble mean
and the members' per-element spread. Packed in fp16: 207 MB of 256; 5.3–6.0 ms per
sample in the container (the two levels of the clock are an open question, STATE.md).
Board (929146, 2026-09-17): rel 94.829 / tke 79.132 / mvpe 94.101 / sps 43.945.
The board best (926634) is the same forecast with the *two-head output average*
instead of the student: sps 43.941 at +1 ms of GPU work.

## Data, and the conditioning that was built to integrate it

These are one coupled decision, not two components — the per-modality affine was
introduced *as the mechanism* for training jointly on two distributions whose
fluctuation energy differs 2.6×. The 08-31 arms measured the 2×2 under the
current recipe (the no-sim column collapses: without a second modality the sim
affine rows never train and the real rows fold into the GroupNorms):

| fold Δ vs shipped | with conditioning | without |
|---|---|---|
| **sim + real** | reference | −0.009 ± 0.006 (noise) |
| **real only** | −0.053 ± 0.003 | ≈ same, by construction |

So under the current recipe: **simulation is worth +0.053 with the affine and
≈ +0.044 without it — the data integrates itself, and the conditioning's marginal
is noise.** Under the old, MSE-dominated recipe the same affine measured +0.17;
the rel-8 per-window-relative loss removed most of the averaging conflict it
existed to absorb, and the scrambles (§3a) showed sim's value never required
fitting sim's targets well in the first place. The affine's *mechanism* is still
real (with film on, the sim token requests 1.31× the fluctuation amplitude —
`docs/archive/model_analysis.md` §1.3); its *necessity* is gone. It stays shipped: removal is
not a gain, it folds away at export, and the verified build carries it.

| component | value | vintage / evidence |
|---|---|---|
| **Joint simulated training** (`--mode plus_sim`) | **+0.053 ± 0.003 fold** (0/4 for removal); "~0.10 of final" is a one-point ratio transfer off Muon and is an order of magnitude, not a measurement — see `docs/archive/model_analysis.md` §0.5 | **current** (08-29). Supersedes the +0.71 of the old record — `docs/archive/FINDINGS_2026-09-10.md` §3a. Behaves as extra biased data; >half the value survives destroying sim's structure |
| **Per-modality affines** (`--film --cond norm`) | marginal ≈ 0 given the current loss (table above) | current (08-31, `nofilm2`/`condenc`) |
| **Per-window normalisation** (RMS speed over valid cells) | not re-ablated; structural | sim is non-dimensional, real is m/s spanning 6× — without it the modalities are an order of magnitude apart and Re-generalisation inherits the velocity scale. What it aligns and what it leaves: `docs/archive/model_analysis.md` §4 |
| **Sim "improvements"** (noise injection, shadow masks, level correction) | −0.012 / −0.001 / ±0.000 | current (08-29). The organisers' own toolbox buys nothing under regime-held-out evaluation |
| **Regime conditioning** (window-inferred or true labels) | **negative, monotone in specialisation** (−0.09 to −0.17 at val_re) | 08-13, four designs; the project's most robust negative — `docs/archive/FINDINGS_2026-09-10.md` §4 |

## Architecture

| component | value | vintage / evidence |
|---|---|---|
| **U-Net over operator families** | U-Net 0.0159 rel-L2 vs FNO 0.0228, CNO 0.0253, Transolver 0.0370 at comparable size | RealPDEBench's own evaluation (`docs/archive/FINDINGS_2026-09-10.md` §6b) + our FNO losing locally |
| **Width 96** (vs 64) | **+0.015 ± 0.001 fold** (0/4 for removal) — a twentieth of the old +0.29 | **current (08-31, `w64cur`)**; 128/160/192 remain within seed noise of 96. The profile axis (+0.011, full-res widening only) is container-taxed — `docs/archive/FINDINGS_2026-09-10.md` §9a |
| **mults 1,2,4 / blocks 2 / persistence anchor / zero-init out** | structural; alternatives measured | `--mults 1,2` lost on the board (−0.013, tke collapse); anchor `blend` refuted (−0.004, 7 folds); decompose flat; every addition that survived started as the exact identity |
| **3-member ensemble** (vs 1) | accuracy+sps worth ~**0.12 board under the current recipe** (910411: single member −0.047 net *including* +0.068 of time credit; 893273 under the old recipe: −0.068 net) | **board-measured twice, two recipes** — the one-pass axis is closed four ways (893273, taps, MIMO ρ-curve, 910411). The 2026-08-31 tail gates read the single member *ahead* (+0.26): that was +0.57 of tke funding, and board tke paid +0.09 while sps amplified −0.07 → −0.34. The tail rig over-prices amplitude calibration on late-window targets; the premium is real on hidden regimes |

## Loss (all terms verified against `scoring.py`)

| term | weight | value | vintage / evidence |
|---|---|---|---|
| relative L2 (`rel`) | 8.0 | the largest single win in the project's history (+0.48 board with the 887158 re-derivation); dose optimum 8, re-confirmed under Muon | 08-13 derivation; dose current |
| plain MSE | 1.0 | **+0.004 for dropping it — dead under the current recipe** (the old "much worse without" is stale) | current (08-31, `mse00`). Joint removal of mse+tmean+spec (`lean`) reads **+0.010, 4/4** — the loss is honestly `rel + tke + sps`, and slightly better that way; `minrec` (lean + no-film) gating |
| TKE (`tke`) | 1.0 | **+0.080 board** for 0.5 → 1.0 (with EMA 0.998, shipped 904950) | current (08-28 re-selection) |
| SPS bulk (`sps_bulk`) | 1.0 | **+0.057 board** (901812, predicted +0.05); the only non-L2 term — pays the bulk of \|e\| where L2 pays the tail | current (08-25) |
| time-mean (`tmean`) | 0.5 | **−0.002 for dropping it — dead** | current (08-31, `tm000`) |
| spectral (`spec`) | 0.1 | **+0.003 for dropping it** — i.e. worth ≈ nothing under the current recipe | current (08-29, `sp000`) |
| MVPE | **0.3** | shipped 917205 (+0.009 final, mvpe +0.047); fold d_mvpe +0.030 (8/8) | current (09-07); the 08-29 null at 0.1/0.3 was on the pre-wd recipe |

## Optimisation

| component | value | vintage / evidence |
|---|---|---|
| **Muon on matrix params** (lr 0.02, mom 0.95) | **+0.070 ± 0.007 fold** (0/4 for removal) — the gain *doubled* vs the +0.037 measured under the previous recipe (board then: +0.074). Baseline fairness confirmed: AdamW at 1.5e-4/3e-4/5e-4 reads −0.120/−0.070/−0.079 | **current (08-31/09-01)**. The axis is fully mapped: **AdamW −0.070 < SOAP −0.025 < SOAP-Muon ≈ Muon** (−0.004/+0.003 at two doses) — the orthogonalisation is the entire active ingredient (matched-step control −0.102 agrees from the other side; update stable rank 84 vs 6.6); Shampoo-style preconditioning recovers two-thirds alone and adds nothing on top. Every knob at its optimum incl. upstream's own placement rule (−0.014 here) |
| **EMA 0.998** | **+0.047 ± 0.013 fold** (0/4 for removal) — a seventh of the ancient +0.31 | current (08-31, `ema00`); the 0.999→0.998 dose was part of +0.080 board |
| **8000-step cosine taken at 6000** | the constant that moved four times, tracking the objective each time; current argmax re-read after every recipe change | current (fold histories) |
| **Muon-group weight decay 0.03** (`--wd-muon`) | **+0.018 ± 0.006 fold** (12/12 runs, 4/4 folds), **+0.057 on the angle suite** (4/4, rel_l2 −1.9…−2.7 %); dose 0.01 +0.006, 0.1 −0.008. Shrinks every Muon-group matrix 40–60 % in norm at unchanged stable rank. **Board (918211): +0.005 of final but −0.008 of accuracy** — rel +0.009 and mvpe +0.134 as predicted, tke **−0.264**, headline carried by a +0.157 time drift | current (09-07/08, `docs/log/2026-09-optimizer.md`) |
| batch 32, lr 3e-4 and wd 1e-4 (AdamW group) | all re-confirmed under Muon (batch 64 −0.164; the AdamW group's wd is *not* the row above and stays 1e-4) | current |

## Uncertainty head

| component | value | vintage / evidence |
|---|---|---|
| **Trained on the SPS reward itself** (not an error proxy) | error-proxy widths all score *below* a tuned constant band (32.2 / 28.0–35.6 vs 36.0) | 08-1x, structural — the objective is steeply asymmetric |
| **Ensemble spread as input** | **≈ 0 under current members** (removing it reads +0.042 sps, sd 0.046, 3/4) — the old +0.44 does not survive | **current (08-31, `hk4` A/B)**. Consistent with the K-axis closure and the 9%-capture result: as the members improved, their errors became predictable from (window, prediction) alone. Kept shipped pending the single-member gate; note this re-prices the one-pass axis (see FINDINGS §8a). The shipped `submissions/*/submission.py` docstrings still quote the old **+0.4 SPS** — those files are frozen records of what was uploaded and are deliberately *not* corrected in place; this table is the current attribution |
| **Asymmetric bounds** (both edges emitted) | sps 38.9 → 42.8 era measurement; uncentred intervals refuted separately (0/4) | 08-1x |
| **Output-averaging of independently fitted heads** (K = 2 shipped as 926634) | **+0.0807 board SPS** from the head pair alone (fold +0.0999, transfer 0.81×); K-ladder fold gains +0.100/+0.125/+0.135/+0.149 at K = 2/3/4/6 → G∞ ≈ 0.18; Shapley ≈ 2/3 centring, 1/3 width. The policy of one head is optimal (c* = 1.000, stratified refits negative); the gain is *fitting-noise* removal | **current (09-15/16)**, `FINDINGS.md` §2b |
| **Distillation of the six-head average into one head** (shipped method, 929146) | fold +0.1223 vs one head (82 % of avg6, beats the pair on every fold); **board SPS 43.9453 vs the pair's 43.9414**, accuracy identical, one head's latency. Teacher-0 init and under-width ×3 are load-bearing at 4000 steps (fresh init: 40 % capture). **12000 steps shipped 2026-09-23** (registered R4): on the fresh 15-cell joint geometry 12k − 4k = **+0.0336 SPS**, 15/15 cells positive, and the 12k student scores **+0.0203 above the six-head average it imitates** where the 4k student sits 0.0133 below it — past some horizon distillation stops being compression and becomes the better estimator. Board 940198: **+0.0262 SPS**, rel/MVPE bit-identical, N = +0.00637; fold→board transfer 0.78×, matching the head pair's 0.81× | **current (09-16/17)**, `CANDIDATE_k4a2.md`, `research/distil_2026-09-16/` |
| **Post-averaging fluctuation gain** (dose 0.5, forecaster side) | board: tke +0.155, rel −0.024, MVPE invariant to 7 digits, exact N **+0.0094**; dose 1.0 built, unread (`CANDIDATE_k4g10.md`); its clock cost is the open question of STATE.md | **current (09-13/14)**, twice measured |
| lr 8e-4, τ 0.05, width 96, 4000/4000 | re-selected twice; second re-selection (against current members) confirmed unchanged, all four arms flat-to-negative | current (08-29, `hr` study) |
| **fp16 weights and output** | 3e-6 rel-L2 cost; output copy was 69% of inference wall time | 08-13 |

## What the container pays (closed 2026-09-11)

Two probe submissions with our exact payload and one window per call — persistence (0.35 ms of GPU work) and the
same plus 3.2 ms of GPU work — read 5.34 and 6.49 ms per sample against the incumbent's 5.25 ms: the container's
per-call cost is ≈ 5.2 ms independent of our kernels and our chain contributes ≈ 0.1 ms visible. Setup relocation
(912717, +0.120) was real; kernel work is worth ≤ +0.03 and is closed (`FINDINGS.md` §5). Width and member count
are probably only weakly time-constrained on the board, but the timing model is not identified (the incumbent does
not sit on the P0/P1 line): measure any wider member's packed latency before assuming it is free.

## 2026-09-10 screens (all sub-bar; `FINDINGS.md` §3, §7)

zero-rest + DropPath 0.08 on the residual branches +0.011 pooled vs the unchanged recipe (two seeds, 5/6 pairs);
mean-flow mixing along the sim/real direction far-Re +0.011 / +0.020 / +0.015 (corrected runs, one seed) but
−0.013…−0.024 at a novel angle; real-only or unit-corrected sim SPS −0.023…−0.030
at a novel angle; time dilation −0.003; input-frame masking −0.05; DropPath × 12k horizon +0.004. None shipped.

## Confidence ledger

* **Measured under the current recipe (08-29/31):** simulated data (+0.053),
  conditioning (≈ 0), placement (≈ 0), Muon (+0.070), EMA (+0.047), width 96
  (+0.015), all six loss-term marginals, all Muon knobs, head constants,
  ensemble-vs-profile capacity, the time model, the head spread input (≈ 0).
  The joint dead-term removal (`lean`) reads **+0.010 (4/4)**, and `minrec` (lean +
  no-film) is additive-neutral, so lean-with-film is the best recipe *measured here*
  (film still folds away at export).

  **On the sub-resolution entries in this list** (added 2026-09-01): +0.010 for
  `lean`, +0.015 for width 96, and the smaller loss-term marginals all sit at or
  below the ~0.02–0.03 fold resolution stated at the top of this file, and below the
  +0.05 "not a result" floor `tools/predict_score.py` prints on every run. A
  consistent sign on four folds that were reused for hundreds of screening decisions
  is weak evidence of a real effect and no evidence of an out-of-sample optimum.
  Read these as "no reason to change it", not as "score-optimal" — the earlier
  wording claimed the latter.
  The single-member finding replicated at three seeds on the tail gate
  (85.078 ± 0.020 vs the three-member 84.886) and shipped as 910411 —
  **which read −0.047 and tripped its pre-registered falsifier**: the tail
  advantage was tke-funded (+0.57 tail → +0.09 board) and sps amplified
  (−0.07 → −0.34). The axis is closed on the board; the gate rule that
  survives it: **a gate cleared by tke funding is not cleared** (price gates
  with tke zeroed). Full post-mortem in CANDIDATE_onelean.md.
* **Structural (justified by construction, not ablation):** per-window
  normalisation, persistence anchor + zero-init, reward-trained head, fp16.
* **Superseded numbers that must not be quoted:** sim +0.71 (→ §3a), conditioning
  +0.17, EMA +0.31, width +0.29 (→ the 08-31 re-measurement above), the additive
  time model (→ §9a).
* **The meta-finding (FINDINGS §8a):** component attributions do not transfer
  between recipes any better than constants do. As the recipe improved, its value
  *consolidated* — the loss re-derivation, Muon and the sps/tke terms absorbed
  most of what conditioning, EMA-size and width used to carry. Every attribution
  in this file therefore carries its vintage, and a stale vintage is a re-measure
  order, not a citation.
