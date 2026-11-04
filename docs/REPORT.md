# Forecasting a measured turbulent wake: Track 1 (Sim2Real) technical report

*Written 2026-09-24/25 for the Decision Phase package (supersedes the 2026-09-03 draft; audit finding M4). Team: Leonard Storcks (Codabench account `iapetos1918`). Every number carries its provenance; "fold"
numbers are local held-out measurements, "board" numbers are paired public-leaderboard reads. *

## 1. Summary

**Task.** Forecast 20 frames (0.4 s) of a particle-image-velocimetry (PIV) measurement of the 2-D velocity field
(u, v) around a NACA 4418 hydrofoil from the preceding 20 frames, on a 32 × 64 grid. Scored on hidden measured data
by five subscores (relative L2, TKE, mean-velocity-profile error, interval score SPS, time) combined into one final
score (§9). The top 10 of the development board are retrained from scratch by the organisers and ranked on a private
set with unseen regimes.

**Method.** Four width-96 U-Nets (seeds 0–3, 22.65 M parameters each in training) trained jointly on all 81 measured
and 100 simulated trajectories of the release, with per-window RMS-speed normalisation, a per-modality affine at every
normalisation site (folded away at export), a last-frame anchor with a zero-initialised output, and a loss assembled
from the scorer's own subscores; Muon on matrix parameters, AdamW on the rest, EMA 0.998, an 8000-step cosine read at
step 6000. At inference the four forecasts are averaged and each cell's temporal fluctuation about its time-mean is
rescaled half-way toward the members' own average temporal variance (the *gain*, dose 0.5: target-free,
time-mean-preserving). Per-element interval bounds come from one width-96 head that is the 12000-step distillation of
the *average* of six independently fitted heads, each trained on the SPS reward itself; it reads the window, the plain
ensemble mean and the members' per-element spread. Inference runs fp16 channels_last CUDA graphs captured at import.
`bash tools/train_final.sh` rebuilds all of it from the release; `tools/check_recipe.py` proves the result bitwise.

**Standing.** Submission 947208 (the deliverable's trained model with the fused-kernel inference of §6.3, plus cuDNN
benchmark-mode algorithm selection): **81.99600** — rel 94.846 / tke 79.233 / mvpe 94.091 / sps 44.002 / 3.511 ms per
sample — fifth of 50 when it was scored (2026-09-27 00:06 UTC) and ninth of 50 on the final development board, 0.222 behind first and
0.111 above eleventh. The same model read 81.90992 with native inference (942498) and 81.94269 with the first fused
kernels (943854); accuracy is unchanged to 3e-4 across all three (`research/last_2026-09-24/NOTES.md`).

**What is new.** We do not claim a new architecture (a plain U-Net wins RealPDEBench's own comparison on this flow,
§8.4). The contributions are elsewhere: (i) a factorisation of SPS into a forecaster-owned accuracy factor and a
head-owned width factor, which yields a forecaster loss term for the *bulk* of the error distribution (§3.3);
(ii) a post-averaging fluctuation gain that restores the TKE an ensemble mean destroys while leaving the time-mean,
hence MVPE, exactly unchanged (§4); (iii) a measurement that the reward-trained interval head's *policy* is at its
optimum and its residual is *fitting noise*, removed by output-averaging six heads and distilling the average into one
student that, on the hardest held-out geometry we have, scores above the average it imitates (§5); (iv) fused
GroupNorm+SiLU CUDA kernels deployed into an evaluation image that has no C compiler, which roughly halved our timed
call on the evaluation host (§6.3);
(v) a retrain contract whose checker is mutation-tested and whose output a pure `git archive` reproduces bitwise (§7).
Two negative results we expect to carry to the private set: conditioning on the regime is monotonically harmful, and
simulation helps as extra biased data, not as transferred physics (§8).

## 2. Data and preprocessing

### 2.1 The release
A trajectory is one nominal Reynolds number (18 measured values from 3750 to 26700 in steps of 1275; simulation
continues to 27975) at one angle of attack (0°, 5°, 10°, 15°, 20°). Fields are sampled every 0.02 s on 64 × 128; the
evaluation grid is decimation phase 0 of that (32 × 64), and we train on exactly that grid (the other three phases
added nothing, −0.006). 79 of 82 measured files have 868 frames; simulated ones have 1000. Measured velocities are in
m/s and scale ×6 across Re; simulated ones are non-dimensional. After normalisation the paired mean flows correlate at
0.85, while simulation carries a median 3.8× the fluctuation energy of measurement (1.1–28×) and has no time alignment
with it (correlation 0.02 at every lag) (`FINDINGS.md` §6).

**81 measured trajectories, not 82.** `7575_0` was withdrawn by the organisers on 2026-08-12 because it reuses
`6300_0`'s raw measurements (we verified u, v bit-identical). `realistix.splits.EXCLUDED` drops it; its simulated
trajectory stays in. `train_final.sh` refuses to build unless the release resolves to exactly 81 real + 100 simulated
trajectories, by count and by a hashed, ordered manifest (`tools/release_manifest.json`).

### 2.2 Preprocessing
* **Per-window normalisation.** Each input window is divided by its RMS speed over the window's valid cells (a cell is
  valid if it is non-zero in any frame of that window — a window-local mask, since one window is all `predict` sees);
  the forecast is multiplied back. This puts m/s measurements and non-dimensional simulations on one scale and divides
  the free-stream speed out of the input, so the network cannot specialise on it. Not re-ablated: it is structural
  (without it the two modalities differ by an order of magnitude).
* **No metadata, no regime input.** `metadata` is empty on scored calls and we do not use it. Re and angle are
  recoverable from a window (angle 98.8 % from the zero-cell geometry, Re to 3.6 %), and §8.3 explains why the model is
  deliberately not given them.
* **Masks.** About 10 % of measured cells are exactly zero (foil body, laser shadow, outside the field of view). The
  scorer excludes exactly-zero targets from SPS only; our SPS-derived loss term and the head's reward use the same
  element mask, the L2-type terms follow the scorer (unmasked). No hard output mask is applied (it would touch 0.1 % of
  squared error).
* **Sampling.** Windows of 20 input + 20 target frames, drawn uniformly over (trajectory, start) pairs; a batch of 32
  holds 13 measured and 19 simulated windows (the pools' proportions). No augmentation of any kind. The interval heads
  train on measured windows only.

### 2.3 What is scored (details and the final-score formula in §9)
Only u, v are scored. **rel**: per-window ‖pred − target‖ / ‖target‖, averaged; **tke**: relative L2 of the per-cell
temporal variance over the 20 output frames (a constant forecast scores exactly 66.67); **mvpe**: relative L2 of the
20-frame time-mean at 36 wake probes; each mapped by 100/(1 + 0.5 e). **sps**: per element exp(−(upper − lower)/σ) if
the target is covered, else 0, σ = 0.0563870, weighted by per-window accuracy prefactors. **time**:
100/(1 + √(t / 728.96 ms)), t the container's time per sample.

## 3. The forecaster

### 3.1 Architecture
Input: the 20 normalised frames as 40 channels (u, v × 20). A dense 3 × 3 stem (40 → 96); three levels of widths
96 / 192 / 384, each with two pre-activation residual blocks (GroupNorm(8) → SiLU → 3 × 3 conv, twice; 1 × 1 skip when
widths differ); stride-2 3 × 3 convolutions down; two bottleneck blocks at 384; 4 × 4 stride-2 transposed convolutions
up with concatenated skips; output GroupNorm → SiLU → 3 × 3 conv (96 → 40), **zero-initialised** and added to the last
observed frame repeated over the 20 leads. At initialisation the network is exactly the persistence forecast. 22.65 M
parameters in training, 22.62 M after export. No dropout, no attention, no stochastic depth.

**Per-modality affine.** During training each GroupNorm's scale and shift is looked up per data source (measured /
simulated), so the two sources are normalised to different statistics while sharing every convolution. At export the
measured row is folded exactly into the GroupNorm parameters; the shipped member is a plain U-Net with no conditioning
input. Under the current loss its marginal is noise (−0.009 ± 0.006 fold); its mechanism is real (the simulated row
requests 1.31× the fluctuation amplitude) but its necessity is gone. It stays because removal is not a gain and the
verified build carries it (`docs/MODEL.md`).

### 3.2 Loss (every term verified against `starting_kit/scoring.py`)

    L = 8.0·rel + 1.0·tke + 1.0·sps_bulk + 0.3·mvpe + 1.0·mse + 0.5·tmean + 0.1·spec

| term | evidence (vintage in `docs/MODEL.md`) |
|---|---|
| `rel`, the scorer's per-window relative L2, weight 8 | the largest single gain in the project: +0.48 board (887158); dose optimum 8, re-confirmed under Muon |
| `tke`, relative L2 of the TKE field, weight 1.0 | +0.080 board for 0.5 → 1.0 together with EMA 0.998 (904950) |
| `sps_bulk` (§3.3), weight 1.0 | +0.057 board (901812, predicted +0.05) |
| `mvpe`, probe time-mean relative L2, weight 0.3 | +0.009 board (917205), MVPE subscore +0.047 |
| `mse` 1.0 / `tmean` 0.5 / `spec` 0.1 | each individually dead under the current recipe (+0.004 / −0.002 / +0.003 fold for dropping) |

The three dead terms are kept. Dropping them jointly reads +0.010 per member (4/4 folds, below resolution), but as a
gained four-member ensemble it reads +0.004 with mim1 −0.011 and an MVPE drift (`CANDIDATE_lean.md`), and the verified,
bitwise-reproduced build carries them. The loss is honestly `rel + tke + sps_bulk + mvpe`.

### 3.3 Why `sps_bulk` exists
SPS factors per window into an accuracy prefactor the forecaster owns and a width factor the head owns:
`sps ≈ 100 · R̄ · W̄`, with R̄ the mean of `0.5/(0.5+e)` over the rel/tke/mvpe errors weighted 0.5/0.3/0.2 (i.e.
`1/(1+2e)` per branch) and W̄ the mean over elements of `exp(−width/σ)·1{covered}`. At our operating point R̄ = 0.716,
W̄ = 0.612 (`FINDINGS.md` §2b). The best any head can do on an element is an interval running exactly from prediction
to target, worth `exp(−|e|/σ)`; so the forecaster sets the ceiling of W̄ and can be trained on it:

    sps_bulk = 1 − mean over scored elements of exp(−|e|/σ),   e in physical units

It is bounded and saturating: it pays for making many small errors smaller, where L2 spends its gradient on the tail.
At dose 1.0 fluctuation amplitude is intact; higher doses damp it (2.0 lost 0.029 on the board), which is why `tke` was
re-weighted with it. One quirk is disclosed: simulated windows enter this term in non-dimensional units against the
measured σ and carry 87.5 % of its gradient; removing or unit-correcting that contribution *loses* ≈ 0.025 at a novel
angle, so it stays as is (`FINDINGS.md` §1).

### 3.4 Optimisation
* **Muon** (Nesterov momentum 0.95, five Newton–Schulz iterations, lr 0.02, weight decay 0.03) on every parameter
  with two or more dimensions; **AdamW** (lr 3e-4, weight decay 1e-4) on biases and GroupNorm parameters; global
  gradient clip 1.0; bf16 autocast with fp32 losses. Muon is worth +0.070 ± 0.007 fold (0/4 folds favour removal) and
  paid +0.074 on the board (898763); AdamW at 1.5e-4 / 3e-4 / 5e-4 reads −0.120 / −0.070 / −0.079. SOAP recovers two
  thirds of the gain and adds nothing on top of Muon: the orthogonalisation is the active ingredient.
* **Muon weight decay 0.03**: +0.018 ± 0.006 fold (12/12 runs), +0.057 on the angle suite; on the board (918211)
  +0.005 of final but −0.008 of accuracy (rel and MVPE up as predicted, tke −0.264). Shipped; we record it as a
  rel/MVPE-for-TKE trade, not a clean win.
* **EMA 0.998**, EMA weights shipped: +0.047 ± 0.013 fold.
* **Schedule**: 8000-step cosine without warm-up, read at step 6000 (lr at 14.6 % of peak). With every trajectory in
  training nothing is left to validate on, so the step is transferred from held-out fold histories; it moved four
  times (between 6000 and 8000) as the objective changed and is re-read after every recipe change.
* **Width 96** vs 64: +0.015 ± 0.001 fold; 128 / 160 / 192 are within seed noise of 96.
* A disclosed recipe quirk: the AdamW group's gradients are never zeroed between steps (they accumulate). Clearing
  them read +0.000 on the mimic folds and +0.020 on the angle suite (below our gate); "fixing" it is a recipe change,
  so the contract keeps the behaviour that was measured and reproduced.

## 4. Ensemble and the post-averaging fluctuation gain

**What four members buy.** On three held-out geometries (mim1 / angle05 / far_drop, seeds 0–3 at step 6000) the members'
error fields correlate at ρ = 0.94 / 0.88 / 0.94, and the four-member mean removes 4.2 / 9.0 / 4.5 % of a single
member's squared error — what independent-noise averaging predicts. rel error falls 2.2 / 3.9 / 2.5 %, MVPE
1.5 / 2.6 / 1.7 %, but **TKE error rises 1.7 / 4.4 / 2.4 %**: averaging damps each cell's temporal fluctuation by the
members' disagreement (`FINDINGS.md` §2). On the board a single member lost 0.047 net (910411) and the fourth member
paid +0.0389 exactly (923184); a fifth member is closed.

**The gain.** Per window, channel and cell, with μ the mean forecast's time-mean, v_mean its temporal variance and
v_mem the members' own temporal variance averaged over members:

    y' = μ + g·(y − μ),   g = 1 + 0.5·(min(3, √(v_mem / v_mean)) − 1)

* *TKE-motivated*: the scorer's TKE is exactly the per-cell temporal variance, which the mean under-states.
* *MVPE-invariant by construction*: the time-mean of every cell is unchanged, and MVPE reads only time-means.
* *Target-free*: v_mem is supplied by the forward passes we already make. The cap is inactive in practice.
* The interval head reads and brackets the *plain* mean; the containment guard `lower ≤ prediction ≤ upper` is applied
  against the gained forecast.

Board (924146 vs 923184): tke +0.155, rel −0.024, MVPE identical to seven digits, exact N **+0.0094**. On the fresh
15-cell joint geometry (§8.2): tke 77.971 → 78.406, rel 95.703 → 95.676, MVPE 96.385 unchanged. **Dose 1.0 failed on
the board** (N −0.0154 against a fold-based +0.0151; rel −0.026 where the folds said −0.0004): the fold suite prices the
amplitude the gain restores, not the pattern it amplifies. The dose is closed at 0.5; a learned variance target was
rejected once its TKE funding was removed (N₀ −0.0069, every fold negative) (`FINDINGS.md` §5).

## 5. The interval head

### 5.1 What it is
A width-96 U-Net (widths 96/192/384, one residual block per level plus a bottleneck block, 13.12 M parameters). Input:
the normalised window (40 channels), the plain ensemble mean (40), a plane of log window-scale (1), and the members'
per-element standard deviation in units of σ with its log (80). Output: two non-negative fields, the extent below and
above the plain mean, as exp of a clamped log-width, in physical units. It is trained on the frozen four members, on
measured windows, directly on the relaxed element reward
`exp(−(below+above)/σ) · sigmoid((e+below)/τ) · sigmoid((above−e)/τ)` with τ = 0.05 × width (AdamW lr 8e-4, weight
decay 1e-5, batch 32, 4000 steps). Re-selecting its step, lr and τ on hidden-shaped folds paid +0.414 and +0.320 SPS
on the board (896716, 897732).

**Why reward-trained and asymmetric.** For an element with residual density f, the expected reward of [a, b] is
`e^{−(b−a)/σ}[F(b) − F(a)]`; its stationarity conditions give f(a) = f(b) = M/σ with M = F(b) − F(a): the optimal
interval is a highest-density interval whose coverage is set by the density, not chosen. Quantile, calibrated or
error-proxy widths are therefore mis-specified; measured, error-proxy widths all score below a tuned constant band
(32.2 / 28.0–35.6 vs 36.0) (`docs/MODEL.md`).

### 5.2 The policy is at its optimum; the residual is fitting noise (`FINDINGS.md` §2b)
Rescaling the shipped bounds about the prediction has its optimum at c* = 1.000; a per-stratum rescale on 400 strata
buys +0.0000 held out, on 81,920 strata +0.0057 on the fit half and −0.0050 on the read half. An independent pass
(Codex) on three held-out folds with outer-training-only heads found every global, per-lead, per-channel, width-bin,
Gaussian, Laplace and empirical correction negative on the fold mean. But two independently fitted heads whose *output
bounds* are averaged read +0.0998 SPS on a pre-registered gate (all 34 fold-cells positive; ≈ 2/3 of it centring,
1/3 width) and paid **+0.0807 SPS on the board** (926634, transfer 0.81×). The fold gain rises with the number of
heads, +0.100 / +0.125 / +0.135 / +0.149 at K = 2 / 3 / 4 / 6. So the head's error is estimator variance, and
averaging removes it.

### 5.3 Six teachers, one student
Six heads (seeds 0/11/21/31/41/51) are fitted with the recipe above; their bounds are averaged in linear units, and one
head of the same architecture is regressed onto that average in log space (`tools/distil_head.py`): initialised from
teacher 0, under-width penalised 3× (too narrow loses coverage outright, too wide only pays exp(−Δ/σ)), lr 8e-4 cosine,
**12000 steps**, seed 7. Teachers are training artefacts; only the student ships, at one head's inference cost.
* 4000-step student (929146): board SPS 43.9453 vs the pair's 43.9414, accuracy identical to ten digits. Teacher-0
  initialisation is load-bearing (a fresh init captured 40 % of the teacher average, this recipe 82 %).
* Fresh 15-cell joint geometry (R3, §8.2; members, teachers and students all refitted there): student − pair
  **+0.0827 SPS, 15/15 cells positive**; 12000 − 4000 steps **+0.0336, 15/15**; and the 12000-step student scores
  **+0.0203 above the six-head average it imitates**, where the 4000-step one sits 0.0133 below it. Past some horizon
  distillation stops being compression and becomes the better estimator (one smooth network fitted to a noisy average).
  The K = 6 increment over the pair is 4× larger there than on the old folds: head fitting noise grows with difficulty,
  so the mechanism should be worth more on a harder private set, not less (`research/push_2026-09-22/R3_RESULT.md`,
  `R4_RESULT.md`).
* Board 940198 (12000 steps): **+0.0262 SPS**, rel and MVPE bit-identical, fold → board transfer 0.78×.

**Ruled out on the head.** The spread input measures ≈ 0 under current members (kept: removal is not a gain); wider
heads (144/192), weight soups, nested / out-of-fold heads (−0.02…−0.06); a likelihood-trained mixture-density head
(−0.35 SPS, 0/34 cells) and a full-field conditional-flow interval model (−1.75 SPS); twelve teachers (+0.0067, below
the floor); 24000-step students (diverge at lr 8e-4). The head axis is harvested.

## 6. Inference engineering

### 6.1 Lifecycle and what the clock measures
The evaluation host is an A800-SXM4-80GB; the dev run is 17 fresh processes, each making one untimed warm-up call and
one timed call of N = 5 (942498's ingestion log). Everything that is not per-window arithmetic runs at import: loading,
moving weights to fp16 channels_last, and capturing every batch bucket as a CUDA graph. Moving setup out of the timed
region paid +0.120 on the board (912717). On the board our timed call took 8.4–8.9 ms, i.e. 1.7 ms of the 4.6 ms per
sample the harness reports: about 37 % of the timed region is our code. The timed call runs at whatever clock the GPU
is in; import-time capture leaves it boosted (a 2 s idle gap or first-call capture cost ≈ +5 ms per call on an A100 in
the 2026-09-24 audit), and at matched clocks the A800 is ≈ 0.94× an A100-PCIe.

### 6.2 The contract inference path (`submissions/canon0918f32/submission.py` = the deliverable's)
* **fp16 channels_last** on the GPU removes the layout conversions cuDNN inserts to reach its NHWC kernels (60–66 of
  ~300 kernels per member); outputs move by one to two fp16 ulps.
* **CUDA graphs per batch bucket**, captured at import: up to eight windows the four members replay concurrently on
  four streams, then one post graph (mean, gain, head, assembly); 16 / 32 / 64 windows use one sequential graph;
  a call is padded (by repeating its last window) to the next bucket, or chunked above eight. Every bucket is
  self-checked against the eager path; any failure falls back to eager unchanged. Arrays are returned as float32
  (fp16 values widened exactly).
* The `dec12kfast` lineage (same model; not uploaded on its own) adds exact 5- and 7-window buckets (11 in total), builds
  one fp32 output buffer inside the graph and copies it once into one of four page-locked buffers pre-faulted at
  import and never reused. At matched GPU clocks in a harness-like fresh process on an A100: 9.14 → 8.28 ms per timed
  call (≈ +0.012 final) (audit M2).

### 6.3 Fused GroupNorm+SiLU kernels (board 943854, 2026-09-25)
**Why.** A kernel-level profile showed torch 2.2's channels_last fp16 GroupNorm at 66–80 µs per call at 5 × 96 × 32 × 64
on an A100, against 17 µs for the 3 × 3 convolution beside it; GroupNorm+SiLU was 1.17 of a member's 2.26 ms. Torch-op
compositions were slower than native, and Triton is not usable: the image has no C compiler, and Triton's cache key
includes the `libtriton.so` hash, so a shipped Triton kernel silently falls back (916264, cugraph16t, lost its fusion
that way).

**Route.** The CUDA source is embedded in `submission.py`; the archive ships cubins that NVRTC 12.1 (torch 2.2.2's own
`nvidia-cuda-nvrtc`, deterministic) compiled from exactly that source, named by its hash (`fastgn_<sha12>_sm80/sm90`;
`tools/build_cubins.py` regenerates and verifies them). At import the cubin for the device is loaded through the CUDA
driver API, else the source is compiled with NVRTC, else native graphs are captured. Kernels are launched with
`cuLaunchKernel` on torch's current stream, so they are captured into the graphs like torch's own; they run only inside
captured graphs (the eager path stays native). Every bucket's fused graph is checked against native eager on synthetic
data at import (tolerance 2e-3; measured one fp16 ulp, 2.4e-4…3.7e-4); a failure disables fusion for that and later
buckets (= the native path).

**Kernel.** One block per (window, group); each thread owns a fixed 4-channel slot (8-byte loads); fp32 shifted
moments; fixed-order warp/block reductions with no atomics (bitwise repeatable); scale, shift and SiLU in one pass with
one fp16 rounding. Every GroupNorm in both networks is followed by a SiLU (29 sites per member, 15 in the head).
The v3 kernels also absorb the convolution biases (conv 1's into the next GroupNorm, conv 2's and the skip's into one
residual-add kernel, the output conv's into the last-frame anchor or the head's bias+clamp+exp): ≈ 33 fewer kernels per
member, ≈ 23 in the head. The "lean" setup uses one CPU thread when a GPU is present, loads the checkpoint on the CPU
and drops redundant eager warm-ups, while still capturing all 11 buckets at import; stderr diagnostics (~280 B per
process, FAQ-permitted) log clocks, page faults and epoch stamps, i.e. the per-process wall.

| build (A100, fresh process, timed call 2, N = 5) | kernel | call 2 | per-process wall |
|---|---|---|---|
| dec12kfast | native | 7.58 [7.50–7.83] ms | ~17 s |
| dec12kgnv2 | one fused GN kernel | 5.29 [4.83–5.67] ms | ~17 s |
| dec12kgnv3 | + conv biases folded | 4.82 [4.47–4.92] ms | ~17 s |
| **dec12kgnv3lean(b)** (shipped as 943854) | v3 + lean setup + diagnostics | **4.50 [4.45–4.74] ms** | **~12 s** (10.1 vs 16.6 s paired with v2, −39 %) |

Held-out parity (292 windows of the joint geometry in N = 5 calls, official scorer) vs the native path: rel +1.0e-5,
tke −7.7e-5, mvpe −1.3e-5, sps +5.8e-5; max |Δ| 4.9e-4 on the prediction, 9.8e-4 on the bounds. Validation: N = 0…1000
(shape, dtype, finite, ordered, pressure zero, repeat-equal), NaN/Inf/zero/mask/scale/permutation/batch-independence
edges, CPU-only, and fallback mutants (no cubin and no NVRTC → bitwise the native path; corrupt cubin → NVRTC path,
bitwise equal to the cubin path). The fused kernels repeat bitwise (300–600 relaunches per shape); the native transposed convolution (cuDNN dgrad) does
not always: at every bucket, in every variant including the native path, replays occasionally differ by one fp16 ulp on
≤ 2 % of elements, most often under GPU contention (a cuDNN property; deterministic cuDNN costs +0.4 ms and was not
adopted; cuDNN benchmark mode would save a further ≈ 0.4 ms but can pick different algorithms in different processes,
so it was not adopted either).

**On the board (943854; registered before upload in `submissions/_manifest/CANDIDATE_dec12kgn.md`).** The ingestion log
shows the kernels loaded from the shipped cubin in all 17 fresh processes (`fused=True/cubin fb=11`) and our timed call
at a median **4.60 ms** (registered 4.5 [4.0, 5.2]) against 8.4–8.9 ms for the native path the day before; accuracy
unchanged to 3e-4 on every subscore. Final 81.94269 (+0.0328 vs 942498): the harness's own share of the clock drew
3.24 ms per sample against 2.92 the day before (about one clock-draw standard deviation), so the time subscore gained
0.34 rather than the ≈ 0.70 our share alone would give at a fixed platform. The contract's inference source is
`submissions/dec12kfinal5b/submission.py`: the same computation (bitwise on finite inputs up to the glue kernels below),
with the comments rewritten and non-finite input velocities read as zero (a corrupted cell no longer zeroes its whole
window), plus two more kernels in the post graph: one writes the members' mean, their spread and the head's
161-channel input in one pass, the other does the gain, scaling, bounds, guards and the float32 output layout in one
pass (post-graph device time 1056 → 800 µs on an A100, timed call −0.45 ms; outputs within one fp16 ulp, held-out
scores identical to six decimals; `CANDIDATE_dec12kfinal4.md`), and the prediction written before the head so that its
copy to the host overlaps the head (timed call −0.26 to −0.35 ms on an A100, outputs unchanged;
`CANDIDATE_dec12kfinal5.md`). `train_final.sh`
regenerates the cubins; `check_recipe` requires them byte-identical to a fresh NVRTC compile (mutation-tested, 18/18).
The in-situ log also measured the per-process wall: 4.5 s import, 1.8 s from the timed call to the next process's
import, 6.7 s per process, 107 s for the 17-process dev run. The last two slots carried the glue kernels with cuDNN
benchmark mode (945471: our call 4.00 ms, but the harness share drew 3.39 ms per sample, 81.94157) and then the
overlapped copy as well (947208: our call **3.80 ms** median, harness share 2.74, **81.99600**); benchmark mode was
board-only until the Decision-Phase package had to reproduce 947208 (import +1.3 s in situ, algorithm choice can
differ between processes); since 2026-10-01 the contract ships 947208's own file, `dec12kfinal5b`. At this operating point the time subscore moves F by ≈ 0.0015 per 0.1 ms of our call:
the harness's own share of t, not our code, now dominates it.

### 6.4 fp16 packing, stated plainly
The organisers discourage fp16 packing and ask that the 256 MB budget be met by design (`competition_info/submission.md`).
We did not meet it by design. Four fp32 members plus the head are ≈ 414 MB against the 268,435,456-byte cap (two
members would fit); the fp16 pack is 207,325,674 bytes. We chose four members over fp32 storage because the fourth
member paid +0.038 on the board and a single member lost 0.047. On the GPU the weights stay fp16 and all arithmetic is
fp16, so the stored tensors are exactly what executes; the only lossy step is the one-time cast when packing
(`tools/build_ckpt.py`), which costs ≈ 3e-6 in relative L2 and moves no subscore by 0.001. Off CUDA the weights are
widened to fp32.

## 7. Reproducibility

**One command.** `bash tools/train_final.sh` reads only `train_real` and `train_sim` and runs: preflight (interpreter,
a CUDA card with bf16 under the granted mask, no duplicate seeds, every research holdout variable cleared, data
manifest, fresh output directories, the contract's `submission.py` copied in, with its kernel cubins regenerated if it
carries any) → four members (`train_mixed.py`, seeds 0–3) → six teachers (`train_final_head.py`) → the student
(`distil_head.py`) → pack (`build_ckpt.py`, `pack_submission.py`) → pack parity (`check_pack.py`: the packed `predict`
against the members and student rebuilt in fp32, on a parity set cut from the release) → recipe and pack integrity
(`check_recipe.py`). `PRINT_RECIPE=1` prints the resolved argument lists. Measured 2 h 40 min on one A100-PCIE-40GB with
the 4000-step student (`REPRODUCE.md`); the 12000-step student adds an estimated 15–45 min.

**What the checker proves.** `tools/check_recipe.py` reads the pack's recorded provenance (member, student and teacher
paths with SHA-256 and seeds), compares every recorded argument list with the script's *resolved* literals, checks the
steps (members 6000, teachers 4000, student 12000), the teacher count and seeds, the member-hash chain through every
teacher, re-derives every packed tensor (folded, fp16) bitwise, checks the pack schema and data manifest, requires
`submission.py` to be the contract's inference source byte for byte and any kernel cubins beside it to be a fresh NVRTC
compile of its embedded source; `--replay-student` re-runs the distillation and requires the student bitwise. The
checker itself is mutation-tested: `tools/tests/mutate_recipe.py` builds mutants (zeroed head, missing head tensor,
flipped member tensor, a teacher substituted for the student, a step-2000 student, perturbed student weights with copied
metadata, bad or duplicated teacher digests, five teachers, a student recording other members, a plain head where a
student is required, a changed inference gain, and stray / corrupted / missing / stale kernel cubins) plus a positive
control; all behave as required (16/16 on the current contract, 18/18 with a fused inference source, 2026-09-24). The
lesson behind it: packing fidelity and metadata are not training provenance; only a replay authenticates a training run
(`docs/LESSONS.md`).

**Bitwise rebuild from a git archive (2026-09-24).** `git archive d490ad1` into an empty directory, with no repository
on the path and fresh run, build and evaluation roots, built on node14 (A100-PCIE-40GB, driver 580.126.09): exit 0,
data manifest OK, `check_pack` MATCH, `check_recipe` OK, and against the deliverable built on node10 (driver
595.71.05) **members 4 × 136/136, teachers 6 × 80/80 and student 80/80 tensors bitwise identical**. The deliverable's
members are also bitwise the 2026-09-19 canonical members (544/544), and its student replays 80/80 on a third node.

**How we got there (disclosed).** On 2026-09-23 the then-retained best (934147) failed its own student replay (0/80):
a stale, empty `.git/index.lock` had made every commit fail silently for eight days, so the code that built it was
never committed. Score impact was nil (the rebuilt student matches to three decimals on every channel), but the
deliverable was rebuilt from a pinned commit into an empty root (`dec12k`), which is what 942498 carries. A 2026-09-24
audit (Claude and Codex independently) then found that a git-only export could not retrain (the data-manifest pair and
the mutation suite were untracked) and that the checker did not bind member seeds or `submission.py`; all were fixed in
3afa023 before the archive build above.

**Across hardware.** The same card class reproduces bit for bit, end to end. A different card class with the same
seeds gives a *different* model (no tensor equal; members as far apart as two seeds): under bf16 autocast the seed pins
the initialisation and the card class picks the trajectory. In-sample the H100 build sat within 0.001 weighted of the
reference and A100 builds 0.01–0.03 below, almost all in SPS; on the board the A100 canonical rebuild (932643) read
exact accuracy-only N +0.025 *above* the H100-built 929146, and two independently built 12000-step students (H100 and
A100) landed 0.0004 SPS apart (940198 vs 942498). The organisers' retrain is therefore a fresh draw (≈ ± 0.02 of final
by the audit's estimate); what lowers its variance — four members, a distilled head, fixed seeds and steps — is the
reproducibility story, not bitwise identity across hardware.

**Environment.** Python 3.10.20, torch 2.2.2+cu121, numpy 1.26.4, h5py 3.16.0, mirroring
`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`. The submission imports torch and numpy only (plus the standard
library's ctypes for the fused variant). The extracted archive is `submission.py` + `model.pth` (+ two cubins for the
fused variant).

## 8. Generalisation: evidence and limits

### 8.1 Instruments
We never validate on random windows. Folds withhold whole (Re, angle) cells, train on the rest and score the withheld
cells, paired against the reference on the same fold. The **mimic folds** (four ten-cell holdouts shaped like the 18
cells that have simulation but no released measurement — our hypothesis for the hidden sets, unconfirmed) are precise
(seed sd 0.009 per run, resolution ≈ 0.02–0.03) but not calibrated: rel transferred ≈ 1× on the few paired points,
MVPE was under-priced locally, TKE changed sign. Every attribution in this report therefore carries its instrument.

### 8.2 Held-out results
* **Whole-angle holdout**: +16–20 % relative rel-L2 at a novel angle, of which 6–9.5 % is specific to missing measured
  coverage (size-matched controls) and grows after step 4000 (16/16 runs) (`FINDINGS.md` §3).
* **Reynolds extrapolation** (the `far` fold, 2026-08-21, an earlier recipe): error penalties at one / two / three grid
  steps past the training edge — rel +0.006 / +0.011 / +0.015, tke +0.040 / +0.094 / +0.082, **mvpe +0.005 / +0.013 /
  +0.025**. MVPE drifts first; it is the channel the board paid hardest.
* **Edge sensitivity under the current recipe** (2026-09-24 audit; each cell scored once held out and once in training,
  two seeds per fold, gained two-member mean): held-out / in-training error ratio at the low-Re edge (Re 3750, three
  cells) rel ×1.10, tke ×1.20, mvpe ×1.15; at the high-Re edge (26700, three cells) rel ×1.08, tke ×1.08, mvpe ×1.25;
  in two interior columns (11400, 13950; ten cells) rel ×1.04–1.05, tke ×1.06–1.09, mvpe ×1.06–1.08. The gain stays
  positive at both edges.
* **Joint geometry R3** (registered 2026-09-18): 15 cells withheld at once — `17775_5, 19050_5, 20325_5, 21600_5,
  24150_{0,10,15,20}, 25425_{0,10,15,20}, 26700_{0,10,15}`, their simulation kept, 66 measured trajectories left. It is
  the only instrument that withholds high Re and the 5° frontier together, which is how the rules describe the private
  set. Fresh members, teachers and students, nothing selected after a read, official scorer, 292 windows: rel 95.676 /
  tke 78.406 / mvpe 96.385 / sps 46.965 (gain 0.5, 12000-step student). These absolute values are not
  board-calibrated; the head comparisons on this geometry are in §5.3.
* **Robustness** (2026-09-24 audit, 54 in-sample windows from nine cells, the deliverable on the CPU path, ΔF at a
  fixed clock): Gaussian noise at 1 / 3 / 10 % of the window's RMS speed −0.40 / −3.27 / −11.5; 5 % of cell-frames
  zeroed −12.2; one mid-window frame zeroed −8.8; **the last input frame zeroed −37.0** (the anchor). The forecaster is
  exactly scale-equivariant (0.6–2×) and its fp16 activations peak ≈ 23× below the fp16 limit.

### 8.3 Design choices made for unseen regimes
* **No regime conditioning.** Four designs that condition on window-inferred or true (Re, angle) all lose, −0.09 to
  −0.17 at held-out Re, monotone in how much specialisation they permit. Conditioning on a parameter one must
  extrapolate over turns a regime-general map into an unsupervised lookup at the scored regimes.
* **Simulation as data.** Joint training on simulation is worth +0.053 ± 0.003 fold (0/4 folds favour removal); more
  than half of that survives destroying the simulation's structure, and seven explicit transfer mechanisms
  (simulation-to-measurement translation as an auxiliary objective, pretrain-then-finetune curricula, regime bagging,
  regime-conditioned normalisation, …) were all negative. The organisers' own toolbox for the gap (noise injection,
  shadow masks, level correction) reads −0.012 / −0.001 / ±0.000. We ship joint training because it is the form in
  which that value is captured, and claim no transfer mechanism.
* **Coverage over count.** With 58 trajectories in training, seven 20° trajectories were worth +1.25 on a 20° split
  where eight interior ones were worth +0.10 (early-August measurement); so every measured regime is in training and
  the stopping step is transferred.

### 8.4 What did not work (closed on a measured range; `FINDINGS.md` §7)
Architecture: FNO / CNO / Transolver (RealPDEBench on this flow: U-Net 0.0159 rel-L2 vs 0.0228 / 0.0253 / 0.0370),
attention, directional scans, dropping the deepest level (+0.062 on 7/7 folds, −0.013 on the board), anchors other
than the last frame (`blend` −0.004 ± 0.011 over 7 folds), width beyond 96. Training: loss-weight re-selection
(including on the ensemble mean), every Muon knob, SOAP/Shampoo, schedules, soups, SAM, longer or true-6000 cosines,
autoregressive and unrolled training, noise/phase augmentation, input-frame masking, time dilation, mean-flow mixing,
DropPath (+0.011 pooled, below gate). Distillation of the forecaster: from wider teachers −0.146 on the board; of the
four-member mean into one model −0.040, of the gained mean −0.020. Donors: FNO, temporal-stem, DCT-decoder and
flow-matched donors (the best, a flow-matched component-wise blend, read +0.0136 fold against a +0.015 gate and was
killed on its registered gate). Post-hoc: mean-side, fluctuation, width, per-lead and per-scale corrections, TKE
restoration, input-only test-time adaptation (−0.19…−0.32, MVPE +5–42 %). TKE is at its climatology floor: a forecast
carrying the exact expected per-cell variance scores 0.558 TKE error on the release against the board's 0.527.

## 9. The score formula and how decisions were priced

**The final score is an exact power mean**, identified from our own 41 scored cards (RMS 3.8e-11;
`research/audit_2026-09-17/codex/power_mean_audit.json`; the organisers do not publish it):

    F = (0.5·rel^p + 0.2·sps^p + 0.1·tke^p + 0.1·time^p + 0.1·mvpe^p)^(1/p),   p = 0.6854

It reproduces 942498 to 81.90992. Because p < 1 the mean is concave: a point on the weakest subscore is worth most.
Marginals `dF/ds_j = w_j (F/s_j)^(1−p)` at our operating point are rel 0.477, sps 0.243, tke 0.101, time 0.0965,
mvpe 0.0957; dF/dt = −0.071 per ms per sample. Through the SPS prefactor accuracy counts twice: rel error −10 % is worth
+0.35 of final, tke −10 % +0.30, mvpe −10 % +0.10, and +1 % relative on the width factor W̄ +0.107.

**How decisions were priced.**
* Every submission since 2026-08-22 was registered — predicted delta, band and kill rule — in
  `submissions/_manifest/CANDIDATE_*.md` before upload, and read the same day. Misses calibrated the transfer ratios
  above (head pair 0.81×, 12000-step student 0.78×, the K = 6 increment 0.17×, gain dose 1.0 with the sign reversed).
* Screens read rel-L2 and the sign of the rest; anything touching capacity or diversity is scored as an ensemble;
  every gate is priced with TKE zeroed (a gate cleared by TKE funding is not cleared — the single-member arm tripped
  exactly that falsifier on the board); a candidate needs a pre-registered accuracy delta ≥ +0.06 before a slot.
* The board clock is a draw, not a level (sd ≈ 0.25 ms per sample ≈ ± 0.018 of final; identical bytes read 5.27 and
  6.11 ms six days apart), so accuracy is priced at a fixed clock and byte-repeat slots are clock calibration.
* The prize is decided on a private set after retraining, so we optimised the contract (variance-reducing devices,
  bitwise reproducibility, a mutation-tested checker) over public thousandths.

## Known limitations

* **Unconfirmed test geometry.** The hidden sets being the 18 simulation-only cells is our inference from the release,
  not a published fact; the folds are sensitivity analyses, not knowledge of the holdout.
* **Local-to-board transfer is anecdotal** per channel (TKE sign-unstable, MVPE under-priced locally); only board
  reads are quoted as magnitudes.
* **The head is fitted in-sample** on members trained on the same trajectories (nested / out-of-fold heads were worse).
* **Recipe baggage.** Three loss terms and the per-modality affine are dead or neutral under the current loss, the
  AdamW group accumulates gradients, and simulated windows enter `sps_bulk` in non-dimensional units; each is kept
  because the reproduced build carries it and removal measured no gain.
* **fp16 packing** is used against the organisers' stated preference (§6.4).
* **Retrain variance.** A retrain on different hardware is a different model of the same method (≈ ± 0.02 of final).
* **Input faults.** A NaN or Inf in a window's u or v zeroes that window's outputs (they stay finite; the release has no
  NaN). A zeroed last input frame costs 37 points of final. `predict` is not thread-safe.
* **Wall-clock budget.** Each scored batch runs in a fresh process (6.7 s per process in situ, 4.5 s of it import); the
  published prediction-stage limits disagree (180 s vs 5 min). Under a 180 s wall about 26 such processes fit, i.e. a
  private set about 1.5 times the dev set's 17 batches; under 5 minutes about 44.
* **Fused kernels** depend on libcuda and a loadable cubin or NVRTC in the evaluation image; the native fallback is
  bitwise the unfused path, so the failure mode is lost speed, not wrong output.
* **Environment specification** is a three-package pin plus an off-tree receipt; no lockfile is tracked yet.
* **No information limit was measured.** What is established is that no lever we tried on the same inputs moved skill
  after 2026-08-28; how far the forecast is from optimal is not known.
