# Competition Overview — NeurIPS 2026 RealPDE, Track 1 (Sim2Real)

Working notes on what the competition asks for, what the data actually looks like on
disk, and what constrains a submission. Facts marked **[verified]** were checked
against the files in `data/`; scoring behaviour is read off `starting_kit/scoring.py`,
which is the leaderboard's own implementation; everything else comes from
`competition_info/`, a copy of the Codabench pages.

Status of this document: sections 1-7 were written 2026-08-05, the day the Development
Phase restarted, and are **reference material** -- the task, the timeline, the data on
disk, the scoring code and the rules. They are still current; the [verified] facts in
section 3 have held up against everything measured since, including the grid
orientation that section 3.3 pins down.

Section 8 was a snapshot of where we stood that day and is **superseded**: read
`FINDINGS.md` for the results and `HANDOFF.md` for the state. It is kept, trimmed, as
a record of what the first day's evidence looked like -- several of its conclusions
were later overturned, and by how much is itself worth knowing.

- Competition: <https://realpdecompetition.github.io/>
- Track 1 on Codabench: <https://www.codabench.org/competitions/17363/> (Track 2 is 17385)
- Our account: `iapetos1918`. Twelve scored submissions in the current phase; best 81.157 (see `HANDOFF.md` §2).

## 1. The task

Predict the future of a real (PIV-measured) 2D velocity field around a NACA4418
airfoil, given a short window of its recent past.

```
input   (N, T_in=20,  H=32, W=64, C=3)   ->   output  (N, T_out=20, H=32, W=64, C=3)
```

- `C = [u, v, p]`. On real data `p` is not measured; it is zero in the inputs and is
  ignored by the scorer. Return zeros for it.
- The evaluation grid is the raw 64x128 PIV grid subsampled 2x in both directions.
- 20 frames at dt = 0.02 s, so the model sees 0.4 s and predicts the next 0.4 s. **[verified]**

The "sim-to-real" part is that the training release pairs each real trajectory with a
matched CFD trajectory that additionally carries the pressure field. The intended
recipe is pretrain on sim, finetune on real; nothing forces that, but the released
baseline checkpoints come in exactly those two flavours.

## 2. Timeline

| Phase | Codabench id | Window (UTC) | Max/day |
| --- | --- | --- | --- |
| Warm-up | 28954 | 2026-07-05 .. 07-19 | 3 |
| Main Development (archived) | 28955 | 2026-07-20 .. 08-04 | 1 |
| **Main Development (current)** | **29663** | **2026-08-05 .. 09-27** | **1** |
| Final Decision | 28956 | 2026-09-28 .. 10-25 | 1 |

Other dates: team registration closes **2026-08-20** (a form, separate from joining on
Codabench — a Codabench join request without a form is denied after that date).
Final results 2026-11-10, code due 2026-11-25, NeurIPS presentation 2026-12-06.

Also: 100 submissions per phase cap. Deadlines are strict UTC, not anywhere-on-earth.

### The 2026-08-05 restart

The Development Phase was rebuilt from scratch today and the leaderboard emptied. The
organizers cut the old evaluation windows at a stride shorter than the prediction
horizon, so one window's target sat inside another window's input — a model reading
across the windows it was handed could find answers it had already been given. Three
things changed with the restart and they matter for us:

1. `predict` may now be called **more than once**, each call in a fresh isolated
   subprocess, and `metadata` is an **empty dict** on scored calls. Module-level
   state does not survive between calls.
2. `sps_score` now maps its aggregate to 0-100 linearly instead of through a logistic,
   and targets outside the PIV field of view or inside the airfoil body are excluded
   from SPS.
3. The execution time limit went from 3 to 5 minutes.

Old scores are not comparable: the archived board topped out near 94, the new one
near 80 on the first day.

## 3. Data

### 3.1 What is on disk

`data/` is a symlink to `data/` (home has ~12 GB free,
so nothing big may live in `$HOME`).

```
data/RealPDE-Competition-Data/
  training_data/train_real/   82 files, 7.0 GB      real PIV trajectories
  training_data/train_sim/   100 files, 8.7 GB      matched CFD trajectories
  example_data/3750_0.h5                            one real trajectory (dup of train_real)
  baseline_checkpoints/sim_pretrain/                CNO 32 MB, FNO 403 MB (+fp16 201 MB), Transolver 50 MB
  baseline_checkpoints/sim_real_ft/                 the same four, finetuned on real
  train_real.tar.gz, train_sim.tar.gz               the original archives
data/_extract_test/                                 stray partial extraction, safe to delete
```

Source: <https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data>
(mirrored on Google Drive; prefer HF, the Drive copy has a download quota).

Starting kit v9 is extracted to `starting_kit/` (the zip is `realpde_t1_starting_kit_v9.zip`):
`scoring.py` (the five subscores, verbatim), `submission_template.py` (persistence),
`submission_example_fno.py` (**the organizers' actual leaderboard submission**),
`load_baseline.py`, `pack_ckpt_fp16.py`, `smoke_test_kit.py`, vendored `einops`, and
the vendored `rpde_baselines/` model code (CNO / FNO / Transolver).

`smoke_test_kit.py` passes end to end in our env: all 7 checkpoints load `strict=True`,
all three architectures forward at both 32x64 and 64x128, the fp16 round-trip
reproduces the fp32 forward to 4.3e-4 relative, and every baseline builds and runs
with `h5py` / `matplotlib` / `pandas` / `scipy` and the system `einops` blocked — i.e.
under the evaluation image's package set. Checkpoints have to be flat in one
directory for it; `ckpts_flat/` holds symlinks.

Baseline `best_val_loss` from the checkpoint metadata: sim-only CNO/FNO/Transolver are
0.38-0.39, the real-finetuned ones are 0.0086 / 0.0089 / 0.0122. Those are not
comparable across splits — the sim/real unit mismatch in 3.4 accounts for most of the
factor of 40.

### 3.2 File layout **[verified]**

Every `.h5` is `<Re>_<AoA>.h5` and holds flat top-level datasets — *not* the
`measured_data/` nesting the Hugging Face README mentions:

| key | real | sim |
| --- | --- | --- |
| `u`, `v` | `(T, 64, 128)` float64 | `(T, 64, 128)` float32 |
| `p` | absent | `(T, 64, 128)` float32 |
| `t` | `(T,)` float32 | `(T,)` float64 |
| `x`, `y` | `(64, 128)` float64 | same |
| `re`, `aoa` | scalar int32 | scalar int32 |

- Real: `T = 868` for 79 of 82 files; three are short (`10125_0` has 607, and two
  others have 492 and 282). Sim: `T = 1000` for all 100.
- Real `dt = 0.0200` (std 4e-7, so PIV timing jitters slightly); sim `dt = 0.02002`
  exactly. Both start at `t = 0.08`. Real spans 17.34 s, sim 20.0 s.
- The `re` dataset inside a real file is the *measured* Reynolds number and differs
  slightly from the filename stem (e.g. `10125_10.h5` has `re = 10142`). Sim files
  match their stem exactly. Use the stem to pair sim with real, the dataset value as
  the physical conditioning input.

### 3.3 Geometry **[verified]**

> **The NACA4418 label stands, but check the geometry here, not in RealPDEBench.** That
> benchmark documents a Foil scenario that resembles this one as a tapered NACA0025, and
> it is a *different dataset*: 82/100 trajectories against 99/99, 64x128 against 128x256,
> dt 0.02 s against 2.5e-3, 12-17 s recordings against 9.97 s, measured Re to 26761 against
> 17031. Against this release's own CFD solid cells, NACA4418 is the only section that
> contains every one at every angle. `src/realistix/viz.py` has the full argument and the
> measured chord (60 mm), which the pages below do not state.

Real and sim live on the **same physical grid** — max discrepancy 2e-5 m, i.e. 1% of a
cell. No spatial registration is needed.

- Domain `x` 0.0017 .. 0.2190 m, `y` 0.0821 .. 0.1899 m; uniform `dx = dy = 1.711e-3` m.
- Row 0 is the *top* (`y` decreases with row index), column 0 is upstream.
- The airfoil sits in the left third; the wake fills the rest of the frame downstream.
- In chords (**60 mm**: the mid-span chord of a 100 -> 20 mm tapered foil, measurement
  plane z = 50 mm, cross-checked against the CFD's solid cells): the domain is
  **3.7 c x 1.8 c** and the leading edge sits **on** the upstream boundary, one cell in.
  There is no approach flow in frame at all.
  So this is framed as a *wake* measurement -- roughly 80% of the frame is downstream of
  the trailing edge, and there is essentially no approach flow. That, plus the shadow
  below, is why the field looks unlike a textbook aerofoil rendering.
- Inflow speed scales linearly with the nominal Reynolds number (0.058, 0.200, 0.379 m/s
  at Re 3750, 13950, 26700), so the working fluid is water. But **do not invert
  `U*c/nu` to get the chord**: with the in-frame free stream and water at 20 C it gives
  20.1 cells against the true 14.6, because the organisers' reference velocity and the
  water temperature are both unpublished. Reproducing 50 mm needs nu = 7.7e-7 (water near
  35 C) or a reference velocity 1.33x the visible free stream. This note has now been
  wrong in both directions; the geometric route via the CFD's frozen cells is the one to
  trust.
- Evaluation subsampling is `[::2, ::2]` on this grid — confirmed by the kit's
  `load_baseline.make_example_input`, which slices `[t, ::sub_s, ::sub_s]` with
  `sub_s = 2`. Plain decimation, no averaging or anti-aliasing.
- **And that decimation is free** (`tools/norm_audit.py --alias`, 2026-08-28): only
  **0.2%** of the raw 64x128 frames' energy sits above the 32x64 Nyquist
  [p10 0.2%, p90 0.2%], so there is nothing appreciable to alias. The PIV
  interrogation window has already low-passed the field well below its own grid
  Nyquist. No prefilter is needed and there is **no aliasing floor on high-`k`
  predictability** — closing that idea before it costs anything.
- The measured **fluctuation** spectrum on the 32x64 grid, real against matched sim,
  as an amplitude ratio per radial band: **1.23** at k<2, 1.37 at 2–4, 1.73 at 4–6,
  **1.95** at 6–8, **2.01** at 8–12, 1.47 above 12. So the "median 3.8x energy" sim/real
  gap quoted elsewhere is a band-average of a ratio that varies by 1.6x across the
  resolved scales, and 72% of real fluctuation energy sits below k=4 where the gap is
  only 1.5–1.9x. Any single-number correction is fitting the average of a curve.
- Fluctuation is **0.7%** of a real window's RMS-speed energy (turbulence intensity
  0.086, [0.041, 0.127] over windows) against 2.0% for simulation (0.143,
  [0.121, 0.165]). So the per-window scale in `realistix.normalize` is 99.3% a
  measurement of the *mean flow*.

Masking differs between the two modalities, which matters because SPS excludes
"outside the PIV field of view or inside the airfoil body":

- **Sim** zeroes only ~21 cells (the airfoil, thinly resolved).
- **Real** zeroes ~480 cells that persist across all frames: the airfoil body plus a
  large triangular laser-shadow / out-of-FOV wedge in the lower-left corner. The mask
  depends on angle of attack, so it must be derived per (Re, AoA) rather than assumed.

No mask is handed to us at evaluation time; it has to be inferred from the input
window (exact zeros in both `u` and `v` across all 20 input frames is a reliable
detector on training data).

### 3.4 The one big sim-real gap **[verified]**

Simulated velocities are **non-dimensionalized**, real velocities are in **m/s**:

| Re (stem) | real `mean(u)` | sim `mean(u)` |
| --- | --- | --- |
| 3750 | 0.0448 | 0.8520 |
| 10125 | 0.1152 | 0.8796 |
| 19050 | 0.2134 | 0.8914 |
| 26700 | 0.2703 | 0.8753 |

Sim `mean(u) ~ 0.87` and `std(u) ~ 0.43` for *every* Reynolds number, while real
`mean(u)` grows almost exactly linearly with Re (`u ~ 1.0e-5 * Re`, i.e. the free-stream
velocity `U_inf = nu*Re/L`). Sim `p` is likewise non-dimensional (mean -0.60, std 0.26).

So the transfer is not just "sim is smoother than real": the two splits are in
different unit systems. Any sim pretraining has to normalize real inputs by a
Re-dependent velocity scale (or learn it), otherwise the pretrained weights see inputs
an order of magnitude off. This is probably the single highest-leverage preprocessing
decision in the whole track.

### 3.5 Normalization statistics

The kit hands us the organizers' own numbers, and they matter twice over: the
GaussianNormalizer stats in `submission_example_fno.py` are what the released
checkpoints were trained with, and `sigma_global` in `scoring.py` is built from the
same pair.

| | organizers (`train_real` 32x64 windows) | our pooled recomputation |
| --- | --- | --- |
| `mean(u)` | 0.154961 | 0.153015 |
| `std(u)` | 0.096806 | 0.114261 |
| `mean(v)` | -0.000514 | -0.000560 |
| `std(v)` | 0.015961 | 0.015499 |

and `SIGMA_GLOBAL = (0.09681041 + 0.01596364) / 2 = 0.0563870259`, straight from the
`scoring.py` comment.

Three of the four agree with an independent pass over all 82 real files to within 3%.
`std(u)` does not: ours is 18% higher. Means and `std(v)` matching while `std(u)` alone
is low points at how the windows were pooled — `u` is the channel whose magnitude
tracks Re (3.4), so it is the one sensitive to which trajectories or windows entered
the fit. Unresolved, and not worth resolving: **use the organizers' numbers**, because
consistency with the pretrained checkpoints is what they buy. The p channel's stored
std is 0 and is replaced by 1 at runtime.

The `submission_example_fno.py` docstring is blunt that skipping this normalization "is
the single most common reason a correctly-loaded baseline still scores badly".

### 3.6 Coverage: Re x AoA **[verified]**

AoA is always one of 0, 5, 10, 15, 20 degrees. Re stems run 3750 .. 27975 in steps of
1275 (20 values). Sim has all 100 combinations; real has 82.

```
          0     5    10    15    20        RS = real+sim, .S = sim only
 3750    RS    RS    RS    .S    RS
 5025    RS    RS    RS    RS    RS
 6300    RS    RS    RS    RS    RS
 7575    RS    RS    RS    RS    RS
 8850    RS    RS    RS    RS    RS
10125    RS    RS    RS    RS    RS
11400    RS    RS    RS    RS    RS
12675    RS    RS    RS    RS    RS
13950    RS    RS    RS    RS    RS
15225    .S    .S    .S    .S    .S
16500    RS    RS    RS    RS    RS
17775    .S    RS    RS    RS    RS
19050    RS    RS    RS    RS    RS
20325    RS    RS    RS    RS    RS
21600    RS    RS    RS    RS    RS
22875    RS    .S    RS    RS    .S
24150    RS    .S    RS    RS    RS
25425    RS    .S    RS    RS    RS
26700    RS    .S    RS    RS    .S
27975    .S    .S    .S    .S    .S
```

Every real trajectory has a sim partner; 18 sim trajectories have no real partner. Two
whole Reynolds numbers (15225 and 27975) are sim-only — natural held-out folds for
testing generalization to unseen Re, which is exactly what the private test set does
("unseen parameter regimes, e.g. novel angles of attack or Reynolds numbers").

### 3.7 Codabench-side layout

Inside the evaluation container each phase is

```
data/<phase>/input_data/samples.npz        `input` (N, 20, 32, 64, 3), optional `sample_id`
data/<phase>/reference_data/targets.npz    `target` (N, 20, 32, 64, 3)   (held out)
```

Only `input` is exposed to a submission. Reading anything else off the container
filesystem is grounds for disqualification.

## 4. Scoring

The leaderboard shows one number, `final_score` in 0-100, higher better. It is combined
from five 0-100 subscores **by an unpublished formula**. Our own submission detail page
shows the five subscores; other teams' are hidden.

| subscore | what it measures |
| --- | --- |
| `rel_l2_score` | relative L2 on u, v |
| `tke_score` | error in turbulent kinetic energy field |
| `mvpe_score` | mean velocity profile error at probes behind the airfoil |
| `time_score` | inference wall time per sample |
| `sps_score` | Safe Prediction Score — accuracy plus calibrated intervals |

All of the below is read off `starting_kit/scoring.py`, which is the leaderboard's own
implementation.

**How many channels are scored.** `measured_channels()` counts channels that are not
`allclose` to zero in the *target*. On real data `p` is identically zero, so `c = 2`
and only u, v are ever touched. Returning garbage in the p channel is harmless;
returning NaN there is not (the finiteness check covers the whole array).

**Rel-L2** is computed **per window and then averaged over N** — a mean of ratios, not
a ratio of norms. One badly-predicted window cannot be diluted by well-predicted ones
in proportion to its energy.

**TKE** = `0.5 * (mean_t((u - mean_t u)^2) + mean_t((v - mean_t v)^2))`, then relative
L2 between the predicted and target TKE fields, again per window. This is a
*fluctuation* statistic, and it is the sharpest constraint in the whole scoring rule:
**any temporally constant prediction has zero fluctuation, so its TKE error is exactly
1.0 and its `tke_score` is exactly 66.67, no matter how good it is otherwise.**
Persistence measures precisely that (7.2). Rel-L2 rewards smoothing toward the mean;
TKE punishes it. That tension is the interesting part of the task.

**MVPE**, now fully pinned down. With `sub_s_real = 2`, `d = 16`, `center_x = 10`,
`center_y = 32`, `n_probe = 9`: probes sit at **x in {13, 21, 29, 37}** and **y in
{8, 10, 12, ..., 24}** on the 32x64 grid, i.e. four vertical rakes in the wake, nine
points each, all in the upper-middle band. Velocities are averaged over the 20 output
frames, then compared as a relative L2 per rake and averaged over the four. So MVPE
sees only 36 of the 2048 spatial cells, and only their time-mean. It is a long-horizon
*bias* check, blind to everything outside the wake band.

**Error-to-score map** for those three: `score = 100 / (1 + 0.5 * error)`. So error 0
gives 100, error 0.1 gives 95.2, error 0.5 gives 80, error 1.0 gives 66.7. Note how
flat this is — cutting Rel-L2 from 0.20 to 0.15 buys 2.3 points of that subscore.

**Time**: `100 / (1 + sqrt(r))` with `r = t_neural / 0.72896 s`. Checkpoint loading is
*not* timed, only inference. A 10 ms/sample model scores 89.6, 100 ms scores 72.7,
728 ms scores 50. This looked like the cheapest source of points in the competition
and it is not: measured against the container, a network costing literally nothing is
worth **+0.147 of final and no more**, four submissions have been spent finding that
out, and the accessible total was one export-time fold, already banked
(`HANDOFF.md` §0c). **Retracted 2026-09-03**: 912717 collected +0.120 by moving setup out
of the timed region, and the container turns out to call `predict()` on two or three
windows at a time, where the clock is kernel-launch latency — `FINDINGS.md` §9c.

**SPS**: optionally return per-element `lower` / `upper`. If omitted, the scorer uses a
default band of `pred +/- 0.05*|pred|`. With `sigma_global = 0.0563870`, per element
`inside = (lower <= target <= upper)` and `nil = (upper - lower) / sigma_global`;
each branch error `e` in {Rel-L2, TKE, MVPE} is squashed to `pm = e / (0.5 + e)` and
elements are rewarded `(1 - pm) * exp(-nil)` where inside, else 0. Then
`weighted = 0.5*dm + 0.3*tke + 0.2*mvpe` and `sps_score = 100 * weighted`.

Details that only the source makes clear:

- **The exclusion mask is literally `target != 0`.** That is how "outside the PIV field
  of view or inside the airfoil body" is implemented — no geometry, no mask file, just
  exact zeros. On our local validation set 10.5% of measured elements are excluded.
  Those elements leave both the numerator and the denominator (`n_scored`).
- `pm` is a **per-window scalar** broadcast over every element of that window, and it
  is computed from the *unmasked* whole-field errors. So the three branches differ
  only in a scalar prefactor `(1 - pm)`; the entire per-element structure of SPS is
  `exp(-nil)` gated by `inside`. Maximizing SPS is therefore, elementwise, exactly
  "maximize `E[exp(-width/sigma) * 1{covered}]`" — a clean, self-contained calibration
  problem that is almost decoupled from the forecasting problem.
- The default band collapses where `pred ~ 0`: width `0.1*|pred|` goes to zero, so
  `exp(-nil) -> 1` but `inside` is essentially never true. That is why the default
  band scores so badly (7.2).

The `exp(-nil)` factor is the lever: a band one `sigma_global` wide costs a factor
`e^-1 = 0.37`, so intervals must be genuinely tight, not merely safe. Narrow intervals
around bad predictions cannot buy score because uncovered elements contribute 0.
Returning input-derived intervals is explicitly encouraged and is *not* considered
cross-window leakage.

**Time** comes from a `mean_t_neural_s` array written into `predictions.npz` by the
ingestion program, not from anything we control directly.

**Bounds are all-or-nothing across the run.** `predict` is called several times; if
some calls return bounds and others do not, every bound is discarded.

**Zero-score conditions** (all five subscores go to 0): non-finite predictions,
wrong prediction shape, or bounds that are non-finite or reversed. The run *fails
outright* (no score at all) if bounds have a shape mismatching the prediction, or if
the sample count does not match the input.

## 5. Submission mechanics

A zip with `submission.py` at the root, optional weights alongside.

```python
def predict(input_array, metadata=None):
    """input_array: (N, 20, 32, 64, 3) numpy -> (N, 20, 32, 64, 3)"""
```

or a `SubmissionModel` class with `__init__(submission_dir=None, device="cpu")`,
`predict(...)`, and optionally `load_checkpoint(path, device)` (called when `model.pth`
exists). To return intervals, return `{"prediction": ..., "lower": ..., "upper": ...}`.

Hard constraints:

- **256 MB extracted**, weights included. The organizers explicitly discourage fp16
  packing as a workaround and point at safetensors plus designing the model to fit.
  This rules out the FNO baseline at fp32 (403 MB).
- **5 minutes wall clock** for the whole container run including model loading.
  The Transolver baseline runs in ~1 minute.
- **Fully offline**, no network, `requirements.txt` ignored.
- Image `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, **Python 3.10, torch 2.2.2,
  numpy 1.26**. Available: torch/torchvision/torchaudio, numpy, pillow, PyYAML,
  requests, tqdm, sympy, networkx. **Absent: scipy, pandas, h5py, einops,
  scikit-learn, matplotlib, opencv.** Pure-Python packages may be vendored next to
  `submission.py`; compiled ones cannot.
- `torch.utils.data.DataLoader` must use `num_workers=0`.
- `predict` may be called multiple times in fresh subprocesses; `metadata` is `{}`.
  Never index into it, and never rely on module-level caches surviving.

## 6. Rules that shape the approach

- **Training data must come from the official release.** No self-generated or
  augmented-into-extra-training data, no pretrained models trained on anything else.
  Checkpoints trained from the release, including the official baselines, are fine.
  This was confirmed to us directly by the organizers (`competition_info/email.md`) in
  reply to our question about generating extra CFD — the answer was no, and the rule
  was added to the announcement because of it. It is enforced by re-training the
  shortlisted teams from scratch.
- **No cross-window information.** Producing a prediction using any evaluation window
  other than the one being answered is disqualification, as is reading evaluation
  files off disk or encoding evaluation data into outputs. Input-derived uncertainty
  intervals are fine.
- Teams of at most 3; one account per team.
- **The development leaderboard does not decide the prizes.** Top 10 are shortlisted,
  re-trained from scratch by the organizers on their cluster, and ranked on a private
  test set with *unseen* Re / AoA regimes. So (a) the training pipeline must be clean,
  scripted and reproducible from day one, and (b) generalization outside the training
  grid matters more than squeezing the public board. Sim-only Re 15225 and 27975 are
  the obvious local proxies for that.
- Winners must open-source before the NeurIPS presentation.
- The exact final package format is announced ~5 days before the Development Phase
  closes (i.e. around 2026-09-22).

Prizes: 6000 / 3000 / 1500 USD per track; top 3 present, top 5 co-author the results
paper and may each name an advisor.

## 7. Measured baselines

### 7.1 The local validation set

Built in `/tmp/rpde_val` (scripts there, not yet promoted into the repo): 8 held-out
real trajectories spanning Re 5025-26700 and all five angles of attack, cut into
**168 windows at stride 40 = T_in + T_out**, so no window's target lies in another
window's input — the bug that forced the restart. Subsampled `[::2, ::2]` to 32x64,
p channel zeroed. 10.5% of measured target elements are exact zeros (the SPS mask).

This is optimistic relative to the hidden set: the trajectories are held out but their
(Re, AoA) regimes are not. A stricter split is the next step.

### 7.2 Subscores of the two reference submissions

Both run through the kit's own `scoring.py`, on the set above:

| | persistence | official FNO (`sim_real_fno_fp16`) |
| --- | --- | --- |
| `rel_l2_score` | 93.18 (err 0.146) | **96.62** (err 0.070) |
| `tke_score` | 66.67 (err 1.000) | **78.73** (err 0.540) |
| `mvpe_score` | 93.62 (err 0.136) | **97.20** (err 0.058) |
| `time_score` | 98.06 (0.29 ms/sample) | 93.65 (3.36 ms/sample) |
| `sps_score` | 15.95 | 21.87 |
| unweighted mean | 73.49 | 77.61 |

Persistence lands at exactly 66.67 on TKE, confirming the 1.0 floor above.

**The combination rule is probably the plain mean.** Our FNO run averages 77.61; the
organizers' `realpde` account, running what `submission_example_fno.py` says is this
same submission, sits at 78.63 on the live board. Different evaluation set, so this is
suggestive rather than proof — but no weighting we can construct gets closer, and it
is worth assuming provisionally because of what it implies next.

### 7.3 Where the points actually are

(The "if" below was resolved later: the scored submissions identify the weights, and
the final score is *not* an equal average. The fit lives in one place,
`src/realistix/weights.py`, and is `rel_l2` 0.474, `sps` 0.253, `tke` 0.106, `time`
0.095, `mvpe` 0.093 at leave-one-out RMS 0.012 on fourteen submissions. The
eight-submission fit that first identified them put `mvpe` at 0.032 — its one
badly-determined column — so a number quoted from before 2026-08-21 is on those
weights. The conclusion below survives either way, because SPS came second in every
fit; but do not price anything off an equal average, and read the weights from
`realistix.weights` rather than copying them into a tool -- ten tools once carried
their own copy and nine of them went stale at the refit.)

If the final score is anywhere near an equal average, `sps_score` at ~22 is not a
rounding detail — it is **the** deficit. Rel-L2 and MVPE are already at 96-97 and the
error map is flat enough that halving their errors buys ~1.5 points each.

Sweeping a single constant half-width on the FNO predictions, changing nothing about
the model:

| band half-width | `sps_score` | coverage |
| --- | --- | --- |
| default `0.05*abs(pred)` | 21.87 | 0.388 |
| 0.02 * sigma_global | 19.66 | 0.266 |
| 0.05 * sigma_global | 36.25 | 0.522 |
| **0.10 * sigma_global** | **45.29** | 0.721 |
| 0.20 * sigma_global | 44.56 | 0.866 |
| 0.40 * sigma_global | 32.65 | 0.947 |

A single hard-coded constant more than doubles SPS, +23 points on that subscore, for
about four lines of code. The optimum is broad and sits near 0.1 sigma_global
(~0.0056 in physical units) because the FNO's actual residual is small — median
0.0024, mean 0.0055, p90 0.0131 — so the band only has to be a fraction of
`sigma_global` to cover most elements while `exp(-nil)` stays near 1.

Per-channel widths scaled to each channel's own spread scored *worse* (38.8 for
`(0.2, 0.03) * sigma`), because the residual is not proportional to channel std;
narrowing `v` below its error destroys coverage faster than the width term repays.
The gain from here is heteroscedastic, per-element widths driven by an actual
uncertainty estimate, not a better global constant.

Nobody on the board seems to have done this yet: rank 1 is 80.11 against the
organizers' own baseline at 78.63, a 1.5-point spread. A working SPS band alone is
worth several times that.

## 8. Where things stood on 2026-08-05 — superseded, kept for the deltas

A snapshot of the first day's evidence. Four of its five conclusions were later
overturned, which is the reason to keep it: the same measurements, repeated after a
configuration change, did not come back the same way.

| first-day claim | what it turned into |
| --- | --- |
| Top of the board ~80.1, tenth ~78.6, 33 entries. "Differentiating inside the top 10 will come from `time_score` and `sps_score` as much as from raw accuracy." | Half right. SPS was indeed decisive (+1.83 from the loss fix alone). `time_score` was not: three submissions went there and the accessible total was one export-time fold, `FINDINGS.md` §9. |
| **The TKE ceiling is empirical, not theoretical** — three deterministic models all stall at TKE error 0.64-0.67; "breaking this needs a model that samples rather than averages." | The ceiling moved to 0.43-0.47 with no sampler, purely from the corrected loss and the extra regimes. And sampling is now *priced out*: TKE is phase-blind, so a generative model buys nothing a per-cell amplitude rescale does, and that rescale is dose-dependently negative (`ideas.md`). |
| **Sim pretraining did not pay for itself** — 77.05 against 77.01, inside noise. | Right conclusion, wrong reason. Sequential pretraining fails by catastrophic forgetting; simulation *joint* with real is worth **+0.71**, the largest single term in the recipe. It is a continuous regulariser, not an initialisation (`FINDINGS.md` §3). |
| `val_aoa` "degrades after ~3000 steps" | The checkpoint step is a property of the training set and the metric, not a constant: 4000 at 58 trajectories, 6000-8000 once AoA=20 is in, and it moved again under the corrected loss (`FINDINGS.md` §8). |
| **Timing is not currently measurable at this resolution** | True locally and it stayed true, but for a reason not guessed here: the container's ~8.2 ms/sample is almost all fixed overhead. Removing 76% of the parameters moved the scored runtime by 1.7% of the local saving. |

The one first-day observation that held completely is §8.1, **the comparison trap**:
the released baseline checkpoints were trained on all 82 real trajectories, so every
local split is in-sample for them and their numbers are not comparable with ours. That
is still true and still worth restating whenever a baseline number is quoted.

## 9. Environment

`micromamba activate realistix`. The env **mirrors the evaluation image** — Python
3.10, torch 2.2.2+cu121, numpy 1.26.4 — so anything that runs locally runs on
Codabench, which matters when a failed submission costs a day. On top of that it has
dev-only tools that must never be imported from `submission.py`: h5py, scipy, pandas,
matplotlib, einops, safetensors, jupyterlab, tensorboard, tqdm.

`setuptools` is pinned `<81`: the CNO stack imports `pkg_resources`, which setuptools
83 no longer ships, and the evaluation image does ship it. Without the pin the kit's
smoke test fails on CNO only.

The env lives at `env_store/realistix` (4.1 GB) and is
symlinked into `<python-env>`, because `$HOME` is at 89%.
Package caches are redirected too: set `CONDA_PKGS_DIRS` and `PIP_CACHE_DIR` under
`` before installing anything new.

Verify the whole stack — env, kit and checkpoints together — with:

```bash
cd /tmp && CKPT_DIR=ckpts_flat \
  EXAMPLE_H5=data/RealPDE-Competition-Data/example_data/3750_0.h5 \
  PYTHONPATH=~/realistix/starting_kit python ~/realistix/starting_kit/smoke_test_kit.py
```
