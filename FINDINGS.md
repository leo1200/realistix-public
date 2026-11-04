# Findings that matter — consolidated 2026-09-11

The results a fresh session needs, deduplicated from the 2026-08/09 record (`docs/archive/FINDINGS_2026-09-10.md`
and neighbours hold the full chronology, every retraction and the long-form arguments). Fold deltas are
accuracy-only direct member scores (`.4738 S(rel) + .1057 S(tke) + .0926 S(mvpe)`, S = 100/(1+e/2)) unless stated.

## 1. The objective is the scorer

Deriving the loss from `scoring.py` was the largest gain in the project: per-window relative L2 at weight 8
against MSE 1 (+0.48 board), TKE restored to 1.0 (+0.080), an SPS-bulk term `1 − mean exp(−|e|/σ)` (+0.057),
MVPE at 0.3 (+0.009), Muon (+0.074), EMA 0.998, step 6000 of an 8000-step cosine. Every weight's optimum is a
property of the rest of the loss: constants re-selected under one recipe did not survive the next, and component
attributions do not transfer between recipes either (`docs/MODEL.md` carries each with its vintage).

Two structural facts about the terms. The per-modality affine's marginal value is ≈ 0 under the current loss;
simulation's whole value is +0.053 fold (a regulariser: more than half survives destroying its structure). The
`sps_bulk` term prices simulation in non-dimensional units against the real-data sigma and simulation carries
87.5 % of its gradient — yet removing or unit-correcting that contribution *loses* ≈ 0.025 at a novel angle
(2026-09-10), so it stays as it is.

## 2. Every tested lever on the same inputs has failed (an information limit is *not* established)

Slow-mode skill (window-mean deviation correlation ρ ≈ 0.75) is the same on the model's own training windows as on
held-out cells; the window-mean carries ~30 % of squared rel_l2 error; coherence falls from 0.85 to 0.31 with
wavenumber; PIV noise is ~10 % of the error norm. None of that proves a ceiling — the archived record retracted that
reading — but no architecture, loss, optimiser or data lever measured since 2026-08-28 moved skill on the same inputs: width saturates at 96, depth/blocks/attention/operators are flat or
negative, 12k horizons and true 6k cosines lose, distillation lost −0.146 on the board, conditioning on the regime
is the project's most robust negative (−0.09 to −0.17), the single member loses ~0.12 net (a fourth member paid +0.039 exact, 923184). TKE error (0.53) is 89–94 % pattern; every intervention that raised TKE paid rel/mvpe/sps (coop25:
board tke +0.137, final −0.023). Post-hoc corrections (mean, fluctuation, width, per-lead, per-scale) are all
≈ 0 or negative. Every post-hoc lever on the interval head reads ≈ 0, and since 2026-09-15 that is a measurement of the
head's *policy* rather than an estimate (§2b) — but averaging two independently fitted heads' *outputs* pays
+0.0998 SPS held out; the head's spread input itself measures ≈ 0 under current members, so the ensemble's value is the
accuracy and sps premium measured on the board (single member −0.047 to −0.068 net), not a mechanism we can name.

**What the member ensemble is, measured (2026-09-17, `research/lean_2026-09-17/k_ladder_*.json`; B members seeds 0–3 at
step 6000, held-out cells of mim1 / angle05 / far_drop).** The members' error fields are correlated at **ρ = 0.94 / 0.88 /
0.94**; the four-member mean removes only **4.2 / 9.0 / 4.5 %** of a single member's squared error, exactly what
independent-noise averaging predicts (ρ + (1−ρ)/4). On the accuracy channel the K = 1 → 4 gain is **+0.015 / −0.021 /
+0.019** plain: rel error −2.2 / −3.9 / −2.5 %, MVPE −1.5 / −2.6 / −1.7 %, but TKE error **+1.7 / +4.4 / +2.4 %** (the
amplitude the mean loses — what the gain restores). So the forecaster ensemble is worth ≈ 3 % of rel and MVPE, costs
TKE until the gain is applied, and its remaining board value (single member −0.12 non-time) sits in SPS through the
accuracy prefactor and the head's calibration. Consequences: (i) ~90 % of the error is *shared* by all seeds — recipe
bias no averaging removes and no "ensemble-level training" can decorrelate (coop25 −0.023 board); (ii) a single model
only 3 % better on rel would match the mean's point forecast, which is why distilling the mean into one model is a
legitimate, unmeasured-under-this-recipe route to the same accuracy at a quarter of the compute.

## 2b. The interval head's policy is optimal; its *fitting noise* is not (2026-09-15, joint Claude + Codex)

SPS is the lowest subscore (43.841) at the second-largest weight (0.253), and the archived "90–97 % of its reachable
gap" was flagged as an estimate. It is now measured on both sides.

* **Policy: closed.** On the shipped bounds (two local splits, 187 and 175 windows) W̄ = 0.674 / 0.673, coverage 0.884,
  corr(width, |err|) 0.652. Rescaling the head's bounds about the prediction has its optimum at **c\* = 1.000 exactly**;
  a per-stratum rescale on 400 strata (lead × width-decile × channel) buys **+0.0000** on the held-out half, and on
  81,920 strata (y × x × lead × channel) it buys +0.0057 on the fit half and **−0.0050** on the read half — pure
  overfitting. Codex reproduced this on three held-out folds with fresh outer-training-only heads and a 240-bin
  empirical interval solver: global / lead / lead×channel×wake / width-bin scale+centre / Gaussian / Laplace /
  empirical corrections are **all negative on the fold mean**. The realised-|error| oracle is 0.842 and the
  perfect-centring-at-current-width oracle 0.753 / 0.693 / 0.824 by fold — both target-assisted, both unreachable
  without a better forecaster. Standardised residual kurtosis is 19.4 / 18.8 / 8.5, so the conditional is heavy-tailed
  under any coarse feature set. What this does **not** establish is a full-information Bayes ceiling.
* **Fitting noise: open, and it pays.** Two independently fitted width-96 heads (seeds 0 and 11, identical constants,
  both reading the same plain mean and spread) whose **output bounds** are averaged 50:50 read, on a gate registered
  before the fits (mean ≥ +0.08 SPS, every fold ≥ 0): mim1 +0.0739, far_drop +0.1252, angle05 +0.1003, **mean +0.0998**,
  all 34 fold-cell readings positive. Shapley attribution: ≈ 2/3 interval centring, 1/3 width. The point forecast is
  bit-identical, so nothing but SPS moves. Head *weight* soups, wider heads (144/192), head inputs/constants and
  nested/OOF heads are closed; the head's *output* had never been ensembled. Cost: **+1.05 ms** locally for the second
  head (k4 4.175 / k4g 4.082 / k4g10 4.111 / two-head 5.157, three interleaved cycles on one idle A100).
  **Shipped and confirmed on the board 2026-09-16 (926634):** sps 43.8607297 → **43.9414123**, i.e. **+0.0807 from the
  head pair alone** — the held-out fold mean of +0.0998 transferred at **0.81×**, and rel/tke/mvpe reproduced 924146 to
  ten digits, so the channel is clean. N = +0.0306 / +0.0302; new best by +0.00116.
  `submissions/_manifest/CANDIDATE_k4h2g05.md`, `research/slot_2026-09-15/`.
* **The clock (rewritten 2026-09-17; the full story is §5).** No board pair identifies a slope on local GPU-ms: the
  two-head artefact (+1.05 local ms) read 5.702, *between* the single-head k4g reads (5.639 / 5.966) and below the
  single-head A2 (6.011). What the record does show is a **level shift** between the pre-gain chain (5.34 ± 0.08) and
  the gain chain (5.83 ± 0.19), confounded with date; the 09-18 byte-identical 923184 repeat separates them.
* **Arithmetic corrections.** The SPS prefactor is `1/(1+2e)`, not the displayed `1/(1+e/2)`: R̄ = 0.716, W̄ = 0.612.
  `CANDIDATE_k4g.md`'s half-dose break-even "5.32 ms" is wrong — the locked weights give **5.431 ms**, and the
  identical-zip repeat identifies the time weight as ≈ **0.09643**, giving N_half ≈ 0.00943 and break-even 5.414 ms.
* **Killed the same day:** release-only covariance/convective augmentation — pseudo-extrapolating the highest held-out
  Re at four angles reduces TKE-pattern mismatch at 1 of 4 angles (−10.2 %, −28.3 %, −7.2 %, **+10.8 %**). The
  ~18 A100-h ensemble arm is not funded (`research/slot_2026-09-15/codex/covariance_gate.json`).

* **The distilled student on the board (929146, 2026-09-17):** SPS 43.9453 (+0.0846 over h0, +0.0039 over the pair —
  the K = 6 increment of +0.0225 fold transferred at ≈ 0.17×), rel/tke/mvpe identical to ten digits, clock 6.011 ms →
  81.7885, not promoted. One head now carries the pair's whole gain. Head-noise headroom above the student ≈ +0.06 fold
  SPS (G∞ ≈ 0.18 from the K-ladder; A1+A2 12k student +0.017 measured) ≈ +0.004 of final — the axis is nearly harvested.

## 3. What generalisation costs, and what it is sensitive to

* Whole-angle holdout: +16–20 % relative rel_l2 at a novel angle, of which 6–9.5 % is real-coverage-specific
  (size-matched controls), growing after step 4000 (late specialisation to seen angles; 16/16 runs).
* Reynolds extrapolation: penalties grow with distance (rel +0.006/+0.011/+0.015; **mvpe** +0.005/+0.013/+0.025
  at 1/2/3 steps); mvpe is the channel that drifts first and the one the board pays hardest.
* Levers that read positive per member at novel regimes — early stopping (+0.07), CFD-guided angle morphing
  (+0.05–0.07), geometry channels, temporal-shared stem, wd 0.03 (+0.057), mean-flow mixing with the sim/real
  difference (corrected runs: far-Re +0.011 / +0.020 / +0.015, one seed, the MVPE part ~90 % one cell) — all compress or reverse at ensemble level, mostly through fluctuation
  amplitude at 5° (better members average below the truth's TKE) or through a cost at the other geometry.
  A target-informed scalar amplitude fix recovers 0.0001: it is a pattern problem.
* Re-selecting the loss constants on the three-member mean (2026-09-12, w_tke 0.7/1.5/2.0 and w_sps 0.5, seeds 0/1/2 on
  mim1/angle05/far_drop): per member a higher TKE weight cuts TKE error 0.005–0.013 every time; the ensemble mean does
  not inherit it (TKE ratios 0.97–1.04, sign inconsistent across folds) while the rel cost survives averaging. The TKE
  weight moves amplitude that averaging removes — the mechanism behind every compressed TKE lever, measured directly.
  Diversity-favouring member settings read as four-member means (EMA 0.995 / 0.99, Muon wd 0.01; 2026-09-12) are flat
  or negative too (−0.003 / −0.006 / −0.037): ensemble-level re-selection of the recipe is closed on both axes tried.
* Regularisation is the one direction with a repeatable weak signal: wd 0.03 shipped (fold +0.018 → board
  accuracy −0.008); zero-rest + DropPath 0.08 reads +0.011 pooled over two seeds vs the unchanged recipe (5/6
  pairs; the DropPath-only effect is unmeasured because every noise arm inherited gradient clearing) — below the
  +0.015 gate.

## 4. Calibration of local instruments against the board

**The final score is identified exactly (2026-09-17, `research/audit_2026-09-17/codex/power_mean_audit.json`, verified
independently):** `F = (0.5 rel^p + 0.2 sps^p + 0.1 tke^p + 0.1 time^p + 0.1 mvpe^p)^(1/p)` with `p = 0.6854`, RMS
3.8e-11 over all 41 scored cards. The linear fits below were secants of this at our operating point (which is why they
drifted with the card population); the marginals at 929146 are rel 0.477 / sps 0.243 / tke 0.101 / time 0.0965 /
mvpe 0.0957. Exact re-reads: k4g's N vs 923184 = **+0.0094** (below the +0.010 rule it was registered under); the pair
vs k4g +0.0196; A2 vs pair +0.0009; wd03mv vs mvpe03b **−0.0100** (a time-funded win); K4 vs K3 +0.0389.

Locked weights (14-card fit, `realistix.weights`): rel 0.474, sps 0.253, tke 0.106, time 0.095, mvpe 0.093 (LOO
0.012); a 34-card refit gives 0.462 / 0.252 / 0.102 / 0.092 / 0.111 (LOO 0.010) — report sensitivity under both,
never the fourth decimal. Per-channel board-over-local ratios are sparse, intervention-specific anecdotes, not
laws: rel ≈ 1× on several points; mvpe under-priced locally (≈ 5× on a sub-floor local move); tke sign-unstable
(and a regulariser's in-sample tke drop predicted a board tke loss); the interval-head config coefficient read
1.2× once and 0.02× once (911563). Ensemble-level scoring is qualitatively necessary for capacity/diversity changes
(`--mults 1,2`: +0.005 per member, −0.053 as an ensemble, −0.006 board). Fourteen pre-registered submissions:
nine inside their band. Same-artefact time noise ±0.04 of final; platform drift by week. Retrain noise across
hardware: one old-recipe pair (892587 vs 887158) read −0.004 non-time incl. sps and −0.014 observed final (time
−0.011). Current recipe (2026-09-11, in-sample): same card class reproduces bit for bit, members and head; a
different card class with the same seeds gives a different model (no tensor equal) reading +0.001 (H100) to −0.033
(A100) weighted vs the incumbent, the A100 gap almost all sps (`REPRODUCE.md`).

## 5. Time: the container's cost is mostly fixed; the model is not fully identified (2026-09-10/11)

**2026-09-22 — the clock fell about a millisecond, our best is a lucky draw inside the new level, and the repeat
lottery is small.** Three reads of the canon accuracy chain: 4.782 (09-19) / 4.649 (09-20) / **5.131 (09-22, 939933,
the same bytes as 934147)** → mean 4.854, sd 0.249, against the 09-13→09-18 chain's 5.83 ± 0.19. Accuracy reproduced
exactly on the repeat (rel and MVPE to every digit; tke and sps landed on *932643's* fp16-path digits, Δ ≤ 1.8e-5 —
the scorer's own non-determinism, now measured twice). Consequences:
* **The registered fp32-payload rule is resolved against fp32.** The rule was "keep fp16 unless two fp32 reads both
  sit below every fp16 read of the chain"; the two fp32 reads are 4.649 and 5.131, and 5.131 > the 4.782 fp16
  minimum. **fp16 stays in the frozen method** — closed, not deferred.
* **Repeating the frozen bytes is worth ≈ +0.008 over five slots, with P(reaching third place) = 0.00 %**
  (200k draws from N(4.854, 0.249); `research/push_2026-09-22/`). Our retained best already *contains* a −0.8 sd
  draw, so a further draw is a conditional lottery, not the unconditional +0.03 per slot this project had been
  assuming. `d(final)/dt = −0.070865` per board-ms at 4.649 ms.
* **Third place is not reachable through the clock**: it needs **3.771 ms**, below anything ever observed.

**2026-09-22 — two unshipped candidates are rejected on arithmetic, not on new measurements.**
* **The learned variance target for the gain** reads **N₀ = −0.0069** (folds −0.0074 / −0.0105 / −0.0029, every one
  negative) once its TKE funding is removed the way the 09-13 rule requires — removing the TKE *score* move is not
  enough, because TKE also funds SPS through the per-window prefactor. Its headline **+0.012 was the wrong ranking
  quantity.** Rejected, not deferred.
* **`k4a2portable`** is **−0.0245** of final against the retained best *before* any timing effect, because it carries
  929146's older weights; even granting a full −0.14 board-ms it is −0.0145. Any portable variant must be rebuilt on
  the canonical weights and re-certified for parity. The registered manifest's "byte-identical" claim covers
  `model.pth`, not the outputs, which it reports at up to 2.44e-4.

**2026-09-22 — the exact price of third place, for any future round.** Moving one displayed subscore, the others
fixed, to gain +0.0659: rel 94.846 → 94.984 (**rel error −2.82 %**, or −1.92 % if the SPS prefactor follows) ·
sps 43.975 → 44.246 (**W̄ +0.616 %**) · tke 79.233 → 79.886 (−3.93 %) · time 92.605 → 93.290 (4.649 → 3.771 ms) ·
mvpe 94.091 → 94.780 (−12.3 %). **W̄ +1 % is +0.1068, not the +0.111 this file carried** — the old figure kept the
retired 0.253 SPS coefficient instead of the exact marginal 0.243221.


**2026-09-17 — two clock levels.** Graphed fp16 reads 09-05→09-12 (cugraph16, mvpe03b, wd03mv, coop25, P0, k4): 5.344
5.485 5.254 5.354 5.338 5.270 → **5.341 ± 0.082 ms**. Every read on the k4g inference code 09-13→09-17 (k4g ×2, the
two-head pair, A2): 5.639 5.966 5.702 6.011 → **5.830 ± 0.186**. The "sd 0.24" quoted since 09-15 pooled the two. The
two-head artefact sits between the single-head ones, so the clock is not a slope on local GPU-ms. Code and date are
confounded on the board (same scoring hosts in both clusters, but the API exposes neither ingestion host nor durations).
Locally (Codex, paired same-process H100, `research/audit_2026-09-17/codex/paired_runtime.json`) the in-graph gain costs
+0.04–0.05 ms quiet and **+0.37 / +0.61 ms with one busy process on the launching CPU** (+0.42 ms at N = 16 even quiet):
a captured graph is not free of CPU cost. The gain's exact accuracy value is +0.0094 ≈ 0.15 board-ms, so if the shift is
code-caused the gain as implemented is net negative. Control: byte-identical 923184 on 09-18 (≤ 5.4 → code; ≥ 5.6 → drift). **A flow-matched donor is decorrelated and equally accurate — and still not enough (2026-09-20/21).** The FM1 sample
mean reads ρ 0.81–0.87 against our ensemble's error at RMS ratio 0.97–1.00; distilled into one network it is ρ 0.84–0.89
at 1.78 ms/window with 87 % of the teacher's value. Blended component-wise (time-mean directly, fluctuation pattern at
renormalised variance, so TKE is invariant by construction) it reads **+0.0136 fold ΔQ, every fold positive** — below the
+0.015 gate, and +0.0089 once the weight is fitted honestly. `CANDIDATE_fm3.md`. Naive blending loses on amplitude, and
the dose-1.0 repair of that loss failed on the board (next entry). **Conditioning the donor on our own forecast closes
the axis (FM4, 2026-09-22):** it reaches RMS parity with our ensemble (1.14 → 1.03) but inherits our errors (ρ 0.81–0.89
→ 0.94–0.98) and the blend falls to +0.0089. Donor accuracy and donor decorrelation are bought against each other; the
product that the blend pays for does not improve. Four registered flow screens (intervals, amplitude, component-wise
blend, conditioned donor) and no candidate.

**The gain dose is closed at 0.5 (2026-09-21, three board reads).** Dose 1.0 read exact N −0.0154 against a +0.0151
fold-based prediction: rel −0.026 where the folds said −0.0004. A deterministic post-averaging transform can transfer on
one dose and reverse on another — the fold suite prices the *amplitude* it restores but not the *pattern* it amplifies.
Price every such gate with TKE zeroed (dose 1.0 was ≈ +0.004 that way).

**Read 2026-09-18: 6.110 ms → DRIFT.** Same bytes,
same accuracy to the digit (SPS 9e-6 apart — scorer non-determinism; a 1e-6 quarantine is too tight), 0.84 ms slower than
on 09-12. The platform level rose ≈ 0.8 ms in six days; the gain code is exonerated; predictions re-centre at ≈ 6.0 ms. Round 2: the post graph has 228 nodes (k4g) vs
208 (k4) but per-node launch cost is not the mechanism; a Triton-fused gain + assembly (190 nodes) removes the whole
local penalty at parity (`research/audit_2026-09-17/codex/round2/`) — but round 3 found the evaluation image has **no C
compiler** and Triton's cache key includes the `libtriton.so` hash (pip ≠ conda image: cugraph16t's archived launcher
keys match pip 4/4 and the image 0/4), so a shipped Triton kernel falls back on the container; that is how cugraph16t
lost its fusion. **Torch-only fix:** computing the per-window preprocessing once and sharing it across the member
graphs and the head (`k4a2portable`, `CANDIDATE_k4a2portable.md`) removes the whole k4g penalty on a pq A100 and beats
k4 (N=1 quiet 5.014 vs k4 5.100 vs k4g 5.155 ms; contended 6.971 / 6.989 / 7.071). Other measured levers: pinned
non-blocking D2H copies (+0.08–0.36 ms, new host buffer semantics), a one-buffer assembly (+0.04–0.14 ms, rounding
changes); skipping the fp32 upcast or reordering spread arithmetic changes intervals for ≤ 0.03 ms and is rejected.

Persistence with our exact payload and one window per call: **5.34 ms per sample**; the same plus 3.2 ms of GPU
work: **6.49 ms**; the incumbent (3.9 ms of GPU work on an A100): 5.25 ms. The two probes give slope 0.36 board-ms
per local GPU-ms and intercept ≈ 5.2 ms — but that line predicts 6.6 ms for the incumbent, which reads 5.25, so
the three points are not on one line (graph replay may overlap the harness cost; platform drift between days is
another candidate). What is established: a call with almost no GPU work costs the same as the incumbent, so kernel
work on the incumbent's chain has at most a few hundredths of a point to give; the pre-registered repeat of P0 on
a third slot is still owed before the slope is quoted. Setup relocation (+0.120) was real.
`submissions/_manifest/CANDIDATE_timing_probes.md`.

## 6. Data facts

Real u,v in m/s (scale ×6 across Re), sim non-dimensional; per-window RMS normalisation collapses both. Sim has
a median 3.8× the fluctuation energy of real (1.1–28×), mean flows correlate 0.85, no time alignment (corr 0.02 at
every lag). Re and AoA are recoverable from a window (AoA 98.8 % from the zero mask, Re to 3.6 %) but must not be
conditioned on. 79/82 real files have 868 frames; 7575_0 is a duplicate and excluded. Regime coverage beats window
count: seven AoA-20 trajectories were worth +1.25 where eight interior ones were worth +0.10. Native lag-1
fluctuation correlations 0.80–0.99; the release's ~2200 effective slow-mode samples are not the binding limit.

## 6b. The 2026-09-13 systematic re-read (`research/audit_2026-09-11/claude/REOPEN_SUMMARY.md`)

Every closed arm on the current recipe, re-read as a three-member ensemble on mim1 / angle05 / far_drop, plain and with
the member-variance-matched gain: no arm beats the incumbent members by more than the read's noise, and with the gain
applied the arm differences collapse (best +0.009 at full dose, promotion gate +0.025). The arms that looked good plain
were buying fluctuation amplitude at the novel angle, which the gain now supplies to the incumbent for free; the arms
that looked bad were losing it. What is left is pattern, and nothing moves it. Time-priced closures were the one class
worth reopening (the fourth member paid +0.038); the post-averaging gain transferred (+0.010 non-time, MVPE invariant).

## 7. Closed axes (do not reopen without a new mechanism)

Loss weights (twice per member, and w_tke / w_sps re-selected on the three-member mean, 2026-09-12); Muon knobs and update-rule variants; SOAP/Shampoo; schedules (length, anneal, peak lr,
warm-up, stopping step at ensemble level, snapshot soups, raw averaging, SAM, meta-updates); width/profile/depth;
attention/XScan/FNO/CNO; decomposition, cumsum, warp, anchors; autoregressive/unrolled; distillation and combiners;
regime conditioning of any form; sim transfer mechanisms (7), sim augmentation, sim dose/share, sim SPS removal
or unit correction; noise/phase augmentation, input-frame masking, time dilation (both variants), mean-flow
mixing (sim direction and scalar level); DropPath doses 0.03/0.08/0.15 and the 12k interaction; head inputs,
head constants, post-hoc widths, nested/OOF heads, and (2026-09-15) every rescaling and re-centring of the head's
own interval policy, stratified as finely as 81,920 cells; ensemble size both ways; TKE post-hoc; mean-side corrections;
time-channel kernel engineering (fusion, Triton, single graph — bounded at a few hundredths by the probes); frontier-cell upweighting; ensemble-level re-selection of loss weights and member-variance settings; the SPS-prefactor objective; a fifth member; every 09-10 screen arm re-read as a gain-aware ensemble (2026-09-13); the lean loss (dead MSE / time-mean / spectral terms dropped) at ensemble level with the gain (2026-09-17: mean +0.004, mim1 −0.011, MVPE drift — `CANDIDATE_lean.md`); distilling the four-member mean into one same-width model under the current recipe (2026-09-17: −0.040 vs the gained mean, the student inherits the mean's damping — `CANDIDATE_distil1.md`); heterogeneous low-weight donors from every surviving family (FNO ρ 0.6 but 35 % worse → blends lose; temporal-stem / frame / MIMO / XScan donors ≈ 0 or negative — `research/audit_2026-09-17/codex/round5/`); a **current-recipe width-64 FNO donor** (2026-09-17 evening, `CANDIDATE_f1.md`: 5–7 % worse, ρ 0.80–0.90, 0.10-blend +0.003 vs a +0.025 gate); distilling the **gained** mean into one model (`CANDIDATE_distil3.md`: −0.020, amplitude fixed but rel slips 1–2 %); a **12-mode DCT temporal decoder** (`CANDIDATE_t1.md`: donor −0.003, replacement −0.016, MVPE +2–6 %); the open round of
2026-09-18 (`research/leap2_2026-09-18/RESULTS.md`): a **likelihood-trained mixture-density interval head** (−0.35 SPS, 0/34
cells), a **full-field conditional-flow interval model** (−1.75 SPS, all cells), and **input-only test-time adaptation**
(−0.19…−0.32 Q, MVPE +5–42 %, the auxiliary task itself −0.04) — `CANDIDATE_{d1,fm1,tta}.md`. Evidence and numbers: `docs/archive/`, `research/audit_2026-09-11/`.

## 8. Method lessons

Re-check stale numbers after a recipe change; pre-register the prediction and the kill number; screen on rel_l2
and read the sign of the rest; score ensembles, not members, for anything touching capacity or diversity; price
every gate with tke zeroed; a closure is a swept range on a geometry, not a refuted mechanism; two independent
passes (Claude + Codex) with a cross-examination round catch implementation confounds that one pass does not.
