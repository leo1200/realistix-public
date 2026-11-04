# iapetos1918: Track 1 method

Codabench user `iapetos1918`; this package reproduces submission 947208 (final 81.99600: rel 94.846, tke 79.233,
mvpe 94.091, sps 44.002, time 93.510 = 3.511 ms per sample; rank 9 of 50 on the final development board).

Effect sizes below are of three kinds. **Board**: the change in final score between two paired development-board
submissions (it includes the day's draw of the platform clock, sd about ±0.018); **board, accuracy only**: the same
pair through the final-score formula, as we reconstructed it from our own scored submissions (it reproduces each of
them to within 1e-10), with the time subscore held fixed. **Board SPS**: the change in the SPS
subscore. **Fold**: an accuracy-only change (rel, tke, mvpe in final-score units) on folds of the training release that
withhold whole (Re, angle) cells; fold resolution is about 0.02.

## 1. Summary
Four width-96 U-Nets are trained on mixed batches of measured and simulated windows with a loss built from the
scorer's own subscores; their forecasts are averaged, and each cell's temporal fluctuation is scaled half-way (in amplitude) toward
the members' own level. The bounds come from one interval head distilled from six heads trained directly on a
relaxed SPS reward. The scorer-derived loss mattered most (board +0.27 over the previous best), then the reward-trained head and the inference
speed work (sections 7–8).

## 2. Model
- Architecture and number of parameters:
  - **Forecaster (×4)**: a U-Net on 40 input channels (u, v × 20 frames). A 3×3 stem to 96 channels; three levels of
    96/192/384 channels with two pre-activation residual blocks each (GroupNorm(8) → SiLU → 3×3 conv, twice; 1×1 skip
    where widths differ); stride-2 3×3 convolutions down, two bottleneck blocks, 4×4 stride-2 transposed convolutions
    up with concatenated skips; output GroupNorm → SiLU → zero-initialised 3×3 conv (96 → 40), added to the last input
    frame repeated over the 20 output frames, so training starts from persistence. In training, every block GroupNorm
    is followed by a per-modality scale and shift (one row for measured, one for simulated data): FiLM-style
    conditioning on the modality only, never on Re or angle (`--film --cond norm`). At packing the measured row is
    folded into the GroupNorm and the simulated row dropped, so the shipped member is a plain U-Net with no conditioning
    input. Under the final loss its marginal is within noise (removing it read −0.009 ± 0.006 fold; it was worth +0.17
    under our earlier MSE loss); it is kept because it folds away at no inference cost.
    22,645,864 parameters as trained, 22,616,680 as packed.
  - **Interval head** (student; teachers have the same shape): a U-Net with widths 96/192/384, one residual block per
    level and one bottleneck block; 13,123,952 parameters. Input, 161 channels: the normalised window (40), the plain
    four-member mean (40), log of the window scale (1), the members' per-element spread r = std·s/σ (40) and
    log max(r, 1e-4) (40), with σ = 0.0563870 m/s the scorer's `SIGMA_GLOBAL`. Output, 80 channels: the extents below
    and above the plain mean in physical units, each exp of a log-width clamped to [−9, −2].
  - **Pack**: four members plus the student, 103,590,672 parameters stored in fp16 (`model.pth`, 207 MB, under the
    256 MiB cap). Four fp32 members would not fit; we chose four members over fp32 storage (the fourth member paid
    +0.039, accuracy only). All network arithmetic on the GPU is fp16, so the stored tensors are what executes; the cast costs about 3e-6
    relative L2.
- Inputs and outputs, including preprocessing and normalization: input float32 (N, 20, 32, 64, 3) in physical units;
  only u and v are used, non-finite values read as 0. Each window is divided by its RMS speed
  s = sqrt(Σ(u²+v²) / (20·n_valid)) + 1e-8 over the cells that are non-zero in any frame, and the forecast is multiplied
  back by s; training normalises the same way, which puts measured (m/s) and simulated (non-dimensional) data on one
  scale. Output: float32 `prediction`, `lower`, `upper` of the input's shape, pressure channel 0.
- Starting point: random initialisation (PyTorch defaults; the forecaster's output convolution and modality tables
  start at zero, the head's output layer at a constant band). The student starts from teacher 0. No official checkpoint.

## 3. Data
- Training data used and how the two are combined: all 81 measured and all 100 simulated trajectories, cut into
  windows of 20 input + 20 target frames on the scored 32×64 grid (65,926 measured and 96,100 simulated windows). Each
  member batch of 32 holds 13 measured and 19 simulated windows, drawn uniformly over (trajectory, start frame). Both
  sources share every convolution and the same loss; the modality index only selects the GroupNorm scale-and-shift row.
  Teachers and the student use measured windows only.
- Validation split carved out of the training data, if any: none; every trajectory is in training. Metrics printed
  under split names (`val_re`, `val_aoa`, `test_aoa`) are in-sample and drive no choice.
- Augmentation: none.
- Data left out, and why: the measured `7575_0` (withdrawn by the organisers as a duplicate of `6300_0`; its simulation
  is used); simulated pressure (not measured, not scored); the other three 2×2 decimation phases of the raw 64×128
  fields (fold −0.006); `mean_std_real.pt` (per-window normalisation replaces it).

## 4. Training
One row per stage, in the order `train()` runs them (`tools/train_final.sh`).

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. Members ×4 (seeds 0–3, one after another) | all 181 trajectories, mixed batches | 8·rel + 1·tke + 1·sps_bulk + 0.3·mvpe + 1·mse + 0.5·tmean + 0.1·spec | Muon (lr 0.02, Nesterov momentum 0.95, 5 Newton–Schulz steps, decoupled wd 0.03) on matrix parameters; AdamW (lr 3e-4, wd 1e-4) on the rest; cosine over 8000 steps, no warm-up; clip 1.0; bf16 autocast; weight EMA 0.998 | 8000 run, EMA weights at step 6000 kept | 32 (13 measured + 19 simulated) |
| 2. Teachers ×6 (seeds 0, 11, 21, 31, 41, 51) | measured windows; frozen members give mean and spread | −relaxed SPS reward | AdamW lr 8e-4, wd 1e-5, cosine, clip 1.0 | 4000 | 32 |
| 3. Student (seed 7) | measured windows; frozen members and teachers | asymmetric log-width regression onto the teachers' averaged bounds | AdamW lr 8e-4, wd 1e-5, cosine, clip 1.0; starts from teacher 0 | 12000 | 32 |
| 4. Packing and checks | – | – | fold the modality rows, cast to fp16, compile the kernel cubins; parity and recipe checks | – | – |

**Member loss.** Every term mirrors a subscore of `scoring.py`: `rel` per-window relative L2; `tke` relative L2 of
the per-cell temporal variance over the 20 output frames; `mvpe` relative L2 of the 20-frame time-mean at the scorer's
36 probes; `tmean` the same on the whole field; `mse`; `spec` L1 between log 2-D power spectra. `sps_bulk` rests on a
factorisation of SPS: per window, SPS ≈ 100·R̄·W̄, where the accuracy factor R̄ depends only on the forecast and the
width factor W̄ = mean of exp(−width/σ)·1{covered} belongs to the head. An interval that contains the prediction and
covers the target is at least |e| wide, so the forecaster sets W̄'s ceiling exp(−|e|/σ). `sps_bulk` = 1 − mean of
exp(−|e|/σ) over non-zero-target elements (e de-normalised by s): it pays for making the many small errors smaller,
where L2 spends its gradient on the tail (board +0.057). Simulated windows enter it in their non-dimensional units
against the same σ; correcting that lost about 0.025 at a held-out angle, so it stays.

**Teacher reward.** Per non-zero-target element, exp(−(b+a)/σ)·sigmoid((e+b)/τ)·sigmoid((a−e)/τ), with b, a the
extents below and above the plain mean, e = target − plain mean and τ = 0.05·(b+a): a smooth version of the scorer's
covered-and-narrow reward. **Student loss.** For each extent, the mean over elements of w·(log d − log d̄)², with d̄ the
linear average of the six teachers' bounds and w = 3 where the student is narrower than d̄ (too narrow loses coverage
outright, too wide only pays exp(−Δ/σ)). The student keeps the six-head average at one head's inference cost.

**Disclosed quirk.** Only Muon's gradients are zeroed between steps; the AdamW group's (biases, GroupNorm parameters)
accumulate. Zeroing them read +0.000 and +0.020 on two fold families, both within resolution; the recipe was tuned and
reproduced with this behaviour, so it stays.

- How runs or checkpoints are selected, if at all: nothing is selected inside `train()`. Seeds and steps are fixed
  and every member and teacher is used. The constants (member stopping step, member and teacher counts, loss weights,
  optimizer and EMA settings, head lr, τ and steps, student steps, gain dose) were chosen during development on
  cell-holdout folds of the training release and on development-board reads.
- Seeds, and the run-to-run spread you have seen: seed sd of one member run 0.009 (fold); head seeds 0.02–0.03 SPS on folds,
  and the six teachers of the 947208 run read in-sample SPS 50.81–51.04.
  Same card class: bit-identical. `train()` with this image's Python stack (run on the host, see the README), on an
  A100-PCIE-40GB, reproduced the 947208
  model tensor for tensor (all 624 packed tensors and every intermediate checkpoint). Different architecture: a
  different model of the same method (an H200 rehearsal: in-sample rel 95.484 vs 95.483, sps 51.055 vs 51.050); our
  estimate across card classes is about ±0.02 of final (e.g. an A100 rebuild read +0.025 accuracy-only against an H100
  build; two 12000-step students built on H100 and A100 over identical members read 0.0004 SPS apart).

## 5. Uncertainty bounds
The student reads the window, the plain four-member mean ȳ and the members' spread, and returns absolute widths
`below` and `above`. lower = ȳ·s − below, upper = ȳ·s + above (the interval brackets the plain mean); then
lower = min(lower, prediction) and upper = max(upper, prediction) with the gained prediction, so lower ≤ prediction ≤
upper always holds; non-finite values become 0; pressure is 0. On a 15-cell held-out geometry (members, teachers and
student all refitted without those cells) the 12000-step student scored +0.020 SPS above the six-teacher average it is
fitted to.

## 6. Inference
- Track 1: what `predict` does per call, numeric precision, any compilation or caching:
  - `load_predictor` imports the inference file (`submission.py`, 947208's file) once per process. The import loads
    `model.pth` to the GPU as fp16 channels_last weights, sets one CPU thread and cuDNN benchmark mode (restricted to
    deterministic algorithms; the algorithm choice is timed once at setup), loads the fused kernels, and captures one
    CUDA graph per batch size 1–8, 16, 32 and 64. Up to 8 windows the four members replay on four streams, then the
    mean, gain and prediction, then the head and bounds; 16/32/64 are one sequential graph each. Every graph is
    checked on synthetic data against the uncaptured PyTorch path (tolerance 2e-3) and recaptured with native ops if
    the check fails. Import takes about 6 s on an A800.
  - Per call: windows are split exactly into captured sizes (no padding; e.g. 63 = 32 + 16 + 8 + 7; the package's wrapper
    also chunks calls above 1024 windows), normalised, run through the members, averaged, gained, run through the head,
    bounded, and returned as fp16 values widened exactly to float32.
  - **Fluctuation gain.** Per window, channel and cell: y' = μ + g·(y − μ), g = 1 + 0.5·(min(3, √(v_mem/v_mean)) − 1),
    with μ and v_mean the time-mean and temporal variance of the mean forecast and v_mem the members' average temporal
    variance. Averaging four members damps the temporal fluctuations the tke subscore scores; the gain restores part of
    them toward the members' own level, uses no target, and leaves time-means (and so mvpe) unchanged.
  - **Fused kernels.** GroupNorm+SiLU with the neighbouring convolution biases folded in, a head-input kernel and an
    output-assembly kernel, written as CUDA source embedded in `submission.py`, compiled at training time by NVRTC into
    sm80/sm90 cubins and launched through the CUDA driver API inside the graphs. Without a loadable cubin the source is
    compiled at import; failing that, torch's native ops run. No torch.compile, Triton or C compiler.
  - Precision: fp16 weights and activations; ensemble statistics and the gain in fp32. Outputs can differ between
    processes by fp16 rounding (benchmark mode times algorithms per process); the fused kernels are bitwise repeatable.
- Track 2: not applicable.

## 7. Speed
| Change | Evidence | Effect |
|---|---|---|
| Checkpoint load and cuDNN warm-up moved from the first call to import | board | +0.120 (8.92 → 6.65 ms per sample) |
| CUDA graphs per batch size, captured at import; members on parallel streams | board | +0.058 (6.65 → 5.69 ms) |
| fp16 weights and activations, channels_last | board | +0.023 (5.69 → 5.34 ms) |
| Fused GroupNorm+SiLU kernel | A100, 5-window call | 7.58 → 5.29 ms |
| + convolution biases folded into the kernels; lean setup | A100 | 5.29 → 4.50 ms |
| Head-input and output-assembly kernels | A100, paired | −0.45 ms |
| Prediction's copy to the host overlapped with the head | A100, paired | −0.26 to −0.35 ms |
| cuDNN benchmark mode (deterministic algorithms) | A100, paired | −0.18 to −0.40 ms |

On the board the same `model.pth` went from a 5-window call of 8.4–8.9 ms (submission 942498, native kernels, final
81.90992) to 3.80 ms median (947208, 81.99600), accuracy unchanged to 3e-4; about +0.015 of that gain was a low draw of
the harness's own share of the time (2.74 ms per sample that day, 2.92–3.39 on the three before).

## 8. What mattered
| Change | Measure | Effect |
|---|---|---|
| Loss re-derived from the scorer (the same submission also dropped `7575_0`) | board | +0.268 over the previous best (+0.480 over its immediate parent, which had regressed) |
| Setup moved out of the timed call | board | +0.120 |
| Head trained 4000 instead of 1000 steps | board | +0.090 (SPS +0.418) |
| Fused kernels, overlapped output copy, cuDNN benchmark mode (942498 → 947208, same `model.pth`; section 7) | board | +0.086 |
| tke weight 1.0, EMA 0.998 and member step 6000, together | board | +0.080 |
| Muon on matrix parameters | board; fold | +0.074; +0.070 |
| Head lr 8e-4 and τ = 0.05 | board | +0.068 (SPS +0.320) |
| `sps_bulk` term | board | +0.057 |
| Joint training on simulation | fold | +0.053 |
| Weight EMA | fold | +0.047 |
| Fourth member | board, accuracy only | +0.039 |
| Averaging two heads' bounds (now one distilled student) | board SPS | +0.081 |
| Student trained 12000 instead of 4000 steps | board SPS | +0.026 |
| Fluctuation gain, dose 0.5 | board, accuracy only | +0.009 (tke +0.155) |

Negative results: four designs that condition on the flow regime lost 0.09–0.17 at held-out Reynolds numbers; gain
dose 1.0 read −0.015 against a fold prediction of +0.015.

## 9. Compute
One NVIDIA A100-PCIE-40GB (driver 595.71.05) on a node with 2 × AMD EPYC 7452 and 1007 GiB RAM; python 3.10.20, pip
torch 2.2.2+cu121, numpy 1.26.4, h5py 3.16.0; `OMP_NUM_THREADS=4`. Training time 2 h 15 min 30 s (members 1 h 04 min,
teachers 48 min, student 23 min, packing and checks under 1 min); rehearsals of this package took 2 h 22 min and 3 h 33 min
on two shared nodes of the same card class and 1 h 36 min on an H200 (an A800 was not available to us). Peak GPU memory 7.8 GiB (whole device,
in the parity check after packing; 5.7 GiB while training members), measured in the bit-identical rehearsal. The run used `tools/train_final.sh` through a driver script with the run name `dec12k`
(`logs/947208_training_run/`).

## 10. References
- Ronneberger et al., U-Net, MICCAI 2015; Wu and He, Group Normalization, ECCV 2018.
- Perez et al., FiLM, AAAI 2018; Chang et al., Domain-Specific Batch Normalization, CVPR 2019 (per-modality affine).
- Jordan et al., Muon, 2024, https://github.com/KellerJordan/Muon (MIT; adapted in `src/realistix/muon.py`).
- Loshchilov and Hutter, AdamW, ICLR 2019, and SGDR, ICLR 2017 (cosine schedule).
- Polyak and Juditsky, 1992 (iterate averaging; we use an exponential moving average of the weights).
- Lakshminarayanan et al., Deep Ensembles, NeurIPS 2017; Hinton et al., Distilling the Knowledge in a Neural Network, 2015.
- RealPDEBench, arXiv:2601.01829; the organisers' `starting_kit/scoring.py`, shipped unmodified.
- PyTorch 2.2.2 (CUDA graphs, cuDNN), NumPy, h5py, NVIDIA NVRTC 12.1 and the CUDA driver API.
- Development tools: AI coding assistants (Claude Code and OpenAI Codex), used under the author's direction to write and
  review code and to run and analyse experiments (see the README).
