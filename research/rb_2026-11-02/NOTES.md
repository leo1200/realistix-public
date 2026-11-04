# RealPDEBench foil study, 2026-11-02 — interpolation vs bridging the gap in sim2real

Post-competition. **External data: nothing here may enter a submission or the 2026-11-25 code release.**

**Code, as of 2026-11-04.** The study ran from a snapshot of the tree (`thin_tree/`, referred to below). Its
non-external additions now live in the main tree, off by default: `--real-drop-aoa` and `--sim-equal-traj`
(`tools/train_mixed.py`), the `thin` fold set (`tools/kfold.py`), `tools/clock_sim.py`. Verified on an A100
(`runs/_smoke_2026-11-04/smoke.sh`): at defaults the tree reproduces the pre-study commit bit for bit (history and all 164
tensors of best.pt), and with the new flags it reproduces the snapshot bit for bit; the snapshot was then removed. The
RealPDEBench streams (§1–2, external data) are kept out of the training code as `rb_streams.patch` (applies to commit
d18f428), with the fetch and conversion scripts in `realpdebench/`.

## 0. The question
At an unseen (Re, AoA) cell the forecaster can only be right in one of two ways: it **interpolates** between our own
real measurements at neighbouring cells, or it **bridges** from data that is not a real measurement of this foil at
this cell (simulation, or measurements of something else). The record already leans hard one way:

* own simulation, jointly trained: +0.053 ± 0.003 fold, and more than half survives destroying its structure
  (`docs/LESSONS.md`, "a regulariser, not a teacher");
* own simulation *at the held-out cell itself* (`REALISTIX_SIM_KEEPS_CELLS=1`, far fold, 3 seed pairs): +0.011 ± 0.016,
  inside noise (`docs/archive/FINDINGS_2026-09-10.md` §0c);
* seven explicit sim→real transfer mechanisms: all negative.

RealPDEBench's foil (a tapered NACA0025, a different campaign; 98 real / 99 sim, Re 2968–17812 at step 781, AoA
0–20) adds the one thing our release cannot: **more real measurements**, and more simulation, of a *different*
geometry. Paired against our recipe on the mimic folds that tells apart:

| reading | real-RB | sim-RB | means |
|---|---|---|---|
| A | + | ≈ 0 | measurement statistics are what is scarce; real data bridges even across geometry |
| B | + | + (≈ real) | generic regularisation by more dynamics, as with our own sim |
| C | ≤ 0 | ≤ 0 | nothing transfers across geometry; generalisation is interpolation on this foil's grid |

and, per cell (`tools/cell_report.py`), whether any gain sits at the extrapolation cells (edge-hi 26700, edge-lo 3750)
or the interpolation cells (interior columns, AoA=5 singletons, hi-col).

## 1. Data
`data/realpdebench/` (fetch.sh, convert.py): 128×256 at dt 0.005 → every 4th frame, 2×2 block mean → 64×128 at
dt 0.02 (our raw layout; the loader's [::2, ::2] gives 32×64). 998 frames per trajectory, no masked cells (the foil
is outside the field of view). Leakage check: RB real 3750_{0,5,10,20} vs our 3750_* — fluctuation correlation
+0.016 / −0.013 / −0.050 / −0.003, mean u 0.28 vs 0.045: different measurements.

What the data looks like, per trajectory (time-mean over valid cells, 32×64):

| set | mean u | temporal fluctuation | fluct / mean |
|---|---|---|---|
| ours, real | 0.06–0.28 (∝ Re) | 0.007–0.039 | 5–15 % |
| ours, sim | ≈ 0.9 (non-dimensional) | ≈ 0.20 | ≈ 21 % |
| RB real | 0.28–0.31 (flat in Re) | 0.003–0.006 | ≈ 2 % |
| RB sim | 0.10–0.52 (∝ Re) | 0.012–0.036 | ≈ 7 % |

RB real is a thin wake on a uniform stream. Under the recipe's RMS-speed scale its normalised dynamics are ~2 %
against RB sim's ~7 %, so a real-vs-sim comparison at that scale would mostly compare gradient weights. The main
arms therefore use `--rb-norm center` (window mean removed, scale = fluctuation RMS), which puts both RB streams on
unit fluctuation; one arm repeats "both" at the recipe's scale. The scale enters only the SPS loss (`--cond norm`
ignores it).

## 2. Arms (pre-registered 2026-11-02, before any result)
`kfold.py --recipe current --fold-set mimic --seeds 0,1 --pq --pq-type a100`, RB at `--rb-batch-frac 0.25` (the
release's real:sim ratio inside the other 75 % is unchanged). RB real carries the real FiLM tag (0), RB sim the sim
tag (1); there is no geometry flag.

| tag | args |
|---|---|
| `rbbase` | none: a fresh baseline on today's tree (archived baselines predate later code) |
| `rbreal` | `--rb-real-dir conv/real` |
| `rbsim` | `--rb-sim-dir conv/sim` |
| `rbboth` | both (the 25 % split between them by window count, ≈ half each) |
| `rbbothr` | both, `--rb-norm rms` |

**Fixed compute.** Steps stay at 8000, so every RB arm sees 25 % fewer of our own windows. That is the practical
question ("is a quarter of the batch better spent on RB than on our data?"), not "does RB help at equal exposure".

**Read rule.** `kfold.py --report rbbase <tag>`, accuracy-only line; a delta counts only if |Δ| > 2 SE over the four
folds *and* ≥ 3/4 folds agree in sign. Fold resolution is ~0.02–0.03. Then `cell_report.py --tags ... --seeds 0,1`,
read by category, never per cell.

**Predictions** (written before the runs): reading C overall — `rbreal` and `rbsim` within ±0.03 of base, `rbboth`
likewise, `rbbothr` ≈ 0 (RB real contributes little gradient at the recipe's scale). If anything is positive it is
`rbsim`, as a regulariser, mirroring our own sim. No category of `cell_report` moves by more than its noise.

**Neither RB scale matches our windows** (256 sampled windows from 12 trajectories each, normalised units; residual =
target minus the anchor frame, which is what the forecaster predicts):

| stream | scale | residual RMS | frame-to-frame std |
|---|---|---|---|
| ours real | recipe | 0.099 | 0.036 |
| ours sim | recipe | 0.177 | 0.060 |
| RB real | recipe / centred | 0.021 / 0.662 | 0.007 / 0.238 |
| RB sim | recipe / centred | 0.079 / 0.327 | 0.028 / 0.117 |

RB real is 5× too quiet at the recipe's scale and 7× too loud centred; RB sim at the recipe's scale is the closest
match of anything. So `rbbothr` is not only a normalisation check: it is also the arm whose sim stream looks most like
ours. Keep this table beside any reading of `rbreal`.

## 3. Smoke (runs/_smoke_2026-10-28/smoke.sh) — passed 2026-11-02
The first attempt (2026-10-28) died at import: `git archive HEAD src tools` leaves out `starting_kit/`, which
`evaluate.py` finds by relative path. Fixed by linking the kit into both trees. pq `1793614880502-smoke_rb2`
(node9): history and all 164 tensors of best.pt identical between the pre-change tree and today's at defaults;
the RB run trains (98 real trajectories / 93 982 windows, 99 sim / 94 941); `SMOKE_RB_PASS`.

## 3b. Would more simulation help? The gap against the neighbour spacing (`gap.py`, `gap2.py`)
Statistics only (time-mean and fluctuation-RMS maps on 32×64, each trajectory normalised by its own RMS speed over the
shared mask), not forecast skill: a necessary condition. Median relative error against the real target cell:

| estimate of real at cell c | mean flow | fluct RMS |
|---|---|---|
| real, mean of the two Re neighbours (±1275) | **0.027** | **0.136** |
| real, mean of the two AoA neighbours (±5°) | 0.100 | 0.313 |
| real, nearest AoA neighbour, one-sided | 0.182 | 0.419 |
| release sim (3D CFD) at c itself | 0.263 | 1.275 |
| … Gaussian-filtered, σ = 2 cells | 0.234 | 0.796 (fluct energy 2.02× → 1.40× real) |
| Re neighbours + ρ·(sim trend), best ρ | ρ* = 0 | ρ* = 0 |

The sim at the target cell is ~10× further from the real target than real data one Re step away, and 2.6× further than
real data 5° away; filtering removes part of the fluctuation excess but barely touches the mean-flow gap, so the gap
is in the flow, not the resolution. Its regime trend is no better: any fraction of it added to real interpolation
makes it worse (98–100 % of bracketed cells at ρ = 1, monotone in ρ). This is the static reason behind §0's training
nulls. A new simulation campaign would need to get the mean flow within ~3 % (Re) / ~10 % (AoA) of the measurement at
the same cell to compete as a teacher; gate any such campaign on a handful of cells that have real data, with this
script, before generating a corpus. (Competition rules forbid generated data, so post-competition only.)

**What the gap is made of** (`gap3.py`, `gap4.py`; median relative error, mean flow / fluct RMS):

| | mean flow | fluct RMS |
|---|---|---|
| noise floor, real: split-half difference of a trajectory's own statistics | 0.021 | 0.136 |
| noise floor, sim (stationary: quarters 1v2 0.071, 3v4 0.075 — no start-up transient) | 0.055 | 0.186 |
| sim at c, raw | 0.267 | 1.314 |
| + mean gap of all other cells (one global correction) | 0.197 | 0.540 |
| + mean gap of other cells at the **same angle** | **0.092** | 0.403 |
| + mean gap of other cells at the same Re | 0.240 | 0.640 |
| + mean gap at the **adjacent angles only** (the honest novel-angle case) | 0.180 | 0.613 |

Three parts. (i) Most of the mean-flow gap is a systematic offset that depends on the angle and hardly on Re: calibrated
on real data at the same angle, the sim reaches 0.092, just under real AoA interpolation (0.100); raw error grows at
20° (0.39). (ii) But that offset changes across 5° as fast as the flow does, so where calibration would matter — an
angle with no real data — it is only as good as copying the nearest measured angle (0.180 vs 0.182) and worse in
fluctuations. (iii) The sim's statistics are 2.5× noisier than the measurement's at similar length, stationary, so not
a transient: its fluctuations are too energetic and too slow (2× the energy), and that noise alone (~0.03 per cell)
explains most of why the sim's Re trend is unusable (§3b, ρ* = 0). Real Re interpolation sits close to the
measurement's own noise floor; no simulation can beat it there.

**Can recalibrating the operating point close it?** (`gap5.py`; the 100 sims bilinearly interpolated in (log Re, AoA)
stand in for a simulator run at any operating point; effective Re = k·Re, k ∈ [0.6, 1.6]; effective AoA = AoA + d,
d ∈ [−5°, +5°]; statistics cached in `moments.npz`.) No. The best remap *per cell*, an oracle, reaches only
0.212 / 1.167 (from 0.267 / 1.314). The per-angle fits run to the edges of the box and disagree with each other
(k 0.6 at 0–10°, 1.4–1.6 at 15–20°; d +5° at 0°, −5° at 10–20°), and interpolating them to a held-out angle gives
0.213 / 1.198, worse than the additive correction (0.180). So the measured flow is not in the release CFD's own family
of flows at any nearby operating point: the misfit is in the model of the experiment (walls, inflow, span, transition,
measurement), not in its nominal parameters. A calibrated simulator would need parameters of that kind.

## 3d. What the simulation gets wrong that matters to a forecaster (2026-11-03; `trend.py`, `freq2.py`, `freq3.py`)
**The regime trend.** Change of the flow between neighbouring cells, sim against real (time-mean and fluctuation maps,
median over pairs): correlation +0.09 / +0.06 for 5° steps, +0.17 / +0.10 for 10°, +0.24 / +0.15 for 15°, with relative
error 1.1–1.6 — the CFD's predicted change is further from the measured change than "no change" is. (Re steps read
≈ 0, but there the real change is near the sampling noise, so only the angle rows say anything.) This is why simulation
at an unmeasured regime adds nothing: it does not know how the flow differs there.

**The clock.** Spectral centroid of v at nine wake probes, cycles per frame: real time scales with Re (log-log slope
+0.75; probe speed +0.88 — the tunnel speed sets Re and dt is 0.02 s), the release sim's does not (+0.10; speed +0.01
— convective units, U = 1 at every Re, dt = 0.02 c/U). Sim/real ratio 1.19 below Re 7000, 0.66, 0.47, 0.41 above
21000; dominant-peak ratio 0.40. So a simulated window at a given regime evolves at 0.4–1.2× the real pace per frame,
and the forecaster, which learns per-frame dynamics with amplitude normalised away but not time, is taught a different
clock by each modality. The record measured the real clock as convective (St tight across Re,
`docs/archive/ideas_2026-09-10.md` §3) but never compared the two modalities' clocks.
The transported set matches the real spectral centroid (1.04×) but not through the dynamics: its dominant peak is
still 0.60× real, its high-frequency energy (≥ 0.25 cycles/frame) 5× real (0.011 vs 0.002), and its frame-to-frame
correlation 0.897 against real 0.934 and sim 0.973 — statically realistic, dynamically slow and flickery (window
stitching, `train_sim_t/PROVENANCE.txt`). That is a candidate reason it is a worse regulariser than raw simulation.

**Did the clock cost us the benefit? Cross-cell check on existing runs** (`clock_cells.py`, `clock_read.py`; the 20°
dose arms' 3-seed ensembles scored per cell, rel_l2, CPU; clock ratio per 20° cell 1.80 at Re 3750, ≈ 1 at 5025–8850,
0.39 at 25425):
* Simulation as regulariser (real − raw sim, none at 20°) fades with Re at d = 5 and 10: +6 % at low Re to +0.4–1.8 %
  at the top, where the clock is 0.4×. Consistent with the clock — but it tracks Re better than clock mismatch
  (Spearman −0.94/−0.91 vs −0.74/−0.84), and the one cell that separates them, Re 3750 with the clock 1.8× off on the
  fast side, gains +6.2 %, as much as the well-clocked cells. At d = 15 the pattern disappears (+5–10 % throughout).
* Simulation at the target (raw incl. − none at 20°) is within ±1 % at every cell and level — **including the cells
  where the clock happens to match** (5025–8850: +0.0 to +1.0 %). The clock is not what blocks transfer there.
* Transported vs raw is negative on average (−2.7 / −2.0 / −0.9 %) and least negative where raw's clock is furthest
  off (Spearman +0.56 with mismatch, +0.48 with Re).
Reading: the clock may cap the regulariser at high Re (cannot be separated from Re with these cells) and is not the
reason simulation at the target fails. Only a clock-corrected set decides the first; the regime trend above is the
better candidate for the second.

## 3e. Re-clocked simulation (pre-registered 2026-11-03, before any run)
**The set.** `data/train_sim_clk` (`tools/clock_sim.py`): every release trajectory resampled in time by k = Re / 4900 sim
frames per real frame (linear interpolation; Re 3750 → 1306 frames, Re 27975 → 175). Re0 from two independent fits on
cells away from 20° (`clock_fit.py`): wake frequencies 4941 (free slope 0.88), freestream speed with water's viscosity
4790 (chord 6.9 cm; U ∝ Re^0.97). Check (`clock_check.py`): dominant-peak ratio to real 0.35 → 1.01 and flat in Re
(slope −0.94 → +0.01; bands 0.93–1.06); at 20°, outside the fit, 1.27 — the CFD sheds faster than the tunnel there,
which is physics, not units. The spectral centroid overshoots after re-clocking (1.33): the CFD's spectrum has less
low-frequency meandering relative to its shedding than the measurement.
**Trainer.** `thin_tree` gains `--sim-equal-traj` (equal sampling mass per simulated trajectory; the release's
equal-length sims are unaffected), so re-clocking does not also reweight the simulation toward low Re (CPU check:
draw share 0.067–0.076 for 14 trajectories of 145–1267 windows). Sim share pinned at 19/32, the baseline's own mix on
every mimic fold (`batch mix: 13 real + 19 sim`).

| tag | fold | against | args |
|---|---|---|---|
| `rbclk` | mimic, seeds 0, 1 | `rbbase` | `--sim-dir data/train_sim_clk --sim-batch-frac 0.594 --sim-equal-traj` |
| `th05clkx`, `th15clkx` | a20, seeds 0–2 | `th05rawx`, `th15rawx` | same, `--real-drop-aoa` '' / '10,15' |
| `th05clkk`, `th15clkk` | a20k, seeds 0–2 | `th05rawk`, `th15rawk` | same |

**Predictions** (from §3d's cross-cell read): `rbclk` − `rbbase` within ±0.03, if anything positive; on the 20° fold the
re-clocked regulariser (clkx − real) at most slightly above the raw one, any gain at the high-Re cells where the raw
clock was 0.4×; simulation at the target (clkk − clkx) ≈ 0 at d = 5 and d = 15, because the regime-trend error is
untouched (and 20° itself is still 1.27× off). Read rule as before; per-cell read with `clock_cells.py`.

## 3c. When does simulation start to pay? Real-coverage dose at a novel angle (pre-registered 2026-11-02)
The owner's question: weaken interpolation by removing real data and find where transfer begins. The record has one
point: unpaired sim with real thinned to 20 % was −0.229 on val_re (`docs/archive/FINDINGS_2026-09-10.md` §3, August
recipe, raw sim, `--real-frac`, which thins by position in a string-sorted list). The Aug-25 flow-matching transport
map (`data/train_sim_t`, `runs/fs2r_chain`; mean-flow statistics fixed on 24 unseen regimes) read −0.032 ± 0.006 as a
replacement for raw sim on the mimic folds (0/4; `research/audit_2026-09-10/codex/paired_screens_classified.json`).
That map trained on 57 regimes and **never saw real data at 20°**, which makes 20° the one angle where the
transported set can sit at the target without leakage.

**Static prediction** (`gap6.py`; target 20°, real kept only ≥ d away; median rel err, mean flow / fluct RMS):

| d | nearest real angle | raw sim at 20° | transported sim at 20° | raw sim + gap at nearest real angle |
|---|---|---|---|---|
| 5° | 0.205 / 0.381 | 0.360 / 0.980 | 0.220 / 0.362 | 0.234 / 0.468 |
| 10° | 0.307 / 0.558 | 0.360 / 0.975 | 0.220 / 0.378 | 0.363 / 0.629 |
| 15° | 0.373 / 0.773 | 0.359 / 0.942 | 0.215 / 0.353 | 0.372 / 0.850 |

(At 0°, where the map saw the target, transported reads 0.064 — in-sample, not usable.) In statistics the transported
set ties extrapolation at today's coverage and wins from d = 10°; raw sim reaches the mean-flow crossover only at
d ≈ 15° and never on fluctuations.

**Design.** `thin_tree/` = the live tree plus `--real-drop-aoa` (default off) and a `thin` fold set in `kfold.py`
(`a20`: all 20 Re at 20° held out from real *and* sim; `a20k`: the same real holdout, simulation at 20° kept). Every
arm scores the identical 16 real cells at 20°. Real training: 65 trajectories (0–15°, d = 5), 48 (0–10°, d = 10),
30 (0–5°, d = 15), verified on CPU (`thin_check.py`). Simulation share of the batch pinned at 19/32
(`--sim-batch-frac 0.594`, the recipe's natural mix) so the sim dose does not grow as real recedes.

| arm | fold | args |
|---|---|---|
| `thXXreal` | a20 | `--mode real_only` |
| `thXXrawx` | a20 | raw sim, none at 20° |
| `thXXrawk` | a20k | raw sim, all 100 incl. 20° |
| `thXXtrk` | a20k | transported sim (`--sim-dir data/train_sim_t`), all 100 incl. 20° |

XX ∈ {05, 10, 15} = d; `--real-drop-aoa` '', '15', '10,15'. Seeds 0, 1, 2: 36 runs. Contrasts: **transfer** = rawk −
rawx (sim at the target), **realism** = trk − rawk, **regulariser** = rawx − real. Read on the accuracy-only fold score;
a contrast counts if |Δ| > 2 SE over the three seeds and 3/3 seeds agree.

**Caveat on trk at d ≥ 10.** The map trained on real 10° and 15°, which those levels remove from the real channel, so
its transported windows carry information the real channel no longer has: optimistic, an upper bound. Only d = 5 is
clean. If d ≥ 10 reads positive, retrain the map per level (tools/translate_sim.py at 951df90) before believing it.

**Predictions.** d = 5: transfer ≈ 0 (as the far-fold +0.011), realism small or negative (as the mimic −0.032).
Real-only and rawx degrade steeply with d. Transfer turns positive by d = 15 if the forecaster follows the static
crossover; trk > rawk at d = 10 and 15. If transfer stays ≈ 0 even at d = 15 while the statics say sim should win,
then it is not the bracket that blocks transfer but the forecasting task itself (FINDINGS_2026-09-10 §3, "the mapping
and the forecasting task are in tension").

## 4. Runs
40 jobs submitted 2026-11-02 ~11:30 CET (seed 0 of every arm first, then seed 1; `submit.log`). They execute the
live tree at `.`, so **src/ and tools/ must not change until they finish**.
Two jobs (`rbboth` mim3 s0, `rbbothr` mim1 s0) died at start with CUDA OOM: another user's process held 36 GB of the
card pq assigned (pq cannot see direct users). Both resubmitted unchanged.

### 4a. RB result (2026-11-02 18:40 CET; `report_mimic.txt`) — reading C, as predicted
Accuracy-only, paired against `rbbase` by fold, seeds 0+1:

| arm | Δ | folds positive | read rule |
|---|---|---|---|
| `rbreal` (centred) | −0.014 ± 0.003 | 0/4 | counts, negative |
| `rbsim` (centred) | −0.007 ± 0.003 | 1/4 | counts (barely), negative |
| `rbboth` (centred) | −0.011 ± 0.003 | 0/4 | counts, negative |
| `rbbothr` (recipe scale) | −0.004 ± 0.004 | 2/4 | inside noise |

Everything is inside the pre-registered ±0.03 and below the fold resolution: RealPDEBench's foil, real or simulated,
does not transfer to ours, and a quarter of the batch is better spent on our own data. The ordering follows the scale
mismatch of §2 — the stream furthest from our windows (centred RB real, 7× too loud) costs most, the recipe-scale
arm least — but at these sizes that is a tendency, not a finding. The cost may be little more than the 25 % dilution
of our own data; no arm isolates that (a null-data control at the same batch share would).

**Per cell** (`cells_rb.txt`, `cells_rb.json`; `cell_report.py --seeds 0,1`, CPU; `rbbothr` not decomposed — its fold
read is zero).
Held-out error level against `rbbase`, paired by cell, rel_l2 / mvpe:

| category | cells | rbreal | rbsim | rbboth |
|---|---|---|---|---|
| edge-hi (26700, extrapolation) | 3 | −1.6 % / −0.7 % | −0.2 % / −1.0 % | +1.3 % / −1.8 % |
| edge-lo (3750, extrapolation) | 3 | +0.1 % / +1.4 % | +1.4 % / +2.0 % | +1.0 % / +1.3 % |
| hi-col | 6 | +1.4 % / +0.1 % | −1.0 % / −0.1 % | −0.8 % / +0.8 % |
| interior | 20 | +0.1 % / +0.7 % | +0.1 % / +0.6 % | +0.3 % / +0.7 % |
| aoa5 | 6 | −0.6 % / +0.3 % | +1.2 % / +0.6 % | +1.6 % / +0.5 % |

No category moves by more than 2 %, and signs disagree across arms and channels: the external data neither helps the
extrapolation cells nor hurts the interpolation ones. The baseline's own pattern is the familiar one — the edge
penalty (held-out minus in-sample) is 2–3× the interior one in rel_l2 (+0.008 vs +0.003) and 4–5× in MVPE (+0.018 vs
+0.004) — and RB data leaves it where it is. The RB arms' smaller *penalties* in some rows come from a slightly worse
in-sample fit, not a better held-out one, so read levels, not penalties.

### 4b. Coverage dose (§3c), provisional 2026-11-02 18:40 CET (27/36 runs; `thin_report.py`)
Accuracy-only score on the 16 real cells at 20°, mean over finished seeds:

| d | real only | raw sim, none at 20° | raw sim incl. 20° | transported incl. 20° |
|---|---|---|---|---|
| 5° | 61.95 | 62.06 | 62.07 | 62.00 |
| 10° | 61.41 | 61.57 | 61.60 | 61.61 |
| 15° | 60.53 | 60.86 | 60.85 | 61.00 |

Paired by seed: sim away from the target is worth +0.115 (3/3) → +0.168 (2/2) → +0.324 (2/2) as real recedes; sim **at**
the target adds +0.008 (2/3) → +0.027 (2/2) → −0.003 (1/2); transported vs raw −0.073 (0/2) → +0.014 → +0.145 (2/2,
the leaky upper bound). So simulation's value grows steeply as interpolation weakens, but not through the target
angle. Two readings of "sim away from the target": it fills the removed angles between data and target (bridging),
or regularisation is simply worth more with fewer real trajectories.

**Added 2026-11-02 18:45 CET, pre-registered: `thXXrawp`** (fold a20, `--sim-paired`: simulation only where real data
remains, so none at the removed angles; 9 runs). rawx − rawp = what simulation at the removed angles is worth; rawp −
real = regularisation at matched regimes. Prediction: rawx − rawp ≈ 0 at d = 5 (nothing removed) and grows with d if
bridging is real; if it stays ≈ 0, the whole dose is regularisation.

### 4c. Coverage dose, final 2026-11-02 21:10 CET (45/45 runs, 3 seeds each; `report_thin.txt`)

| d | real only | sim paired (only where real remains) | raw sim, none at 20° | raw sim incl. 20° | transported incl. 20° |
|---|---|---|---|---|---|
| 5° | 61.95 | 62.07 | 62.06 | 62.07 | 62.00 |
| 10° | 61.40 | 61.57 | 61.58 | 61.60 | 61.60 |
| 15° | 60.54 | 60.83 | 60.86 | 60.87 | 60.98 |

| contrast (paired by seed) | d = 5° | d = 10° | d = 15° |
|---|---|---|---|
| **regularisation** (paired − real) | **+0.128 ± 0.007 (3/3)** | **+0.169 ± 0.004 (3/3)** | **+0.294 ± 0.010 (3/3)** |
| sim at the removed angles (raw none@20 − paired) | −0.012 ± 0.013 (1/3) | +0.009 ± 0.009 (2/3) | +0.028 ± 0.027 (2/3) |
| sim at the target (raw incl. − raw none@20) | +0.008 ± 0.011 (2/3) | +0.025 ± 0.006 (3/3) | +0.006 ± 0.021 (2/3) |
| realism (transported − raw, both incl.) | −0.073 ± 0.010 (0/3) | −0.002 ± 0.023 (1/3) | +0.115 ± 0.052 (3/3)* |

\* leaky upper bound, and so is the d = 10 entry: the map trained on real 15° (and 10°), which these levels remove
from the real channel, and the transported arm replaces the simulation at those angles too. Only d = 5 is clean.
(Corrected 2026-11-03: the first version of this section called d = 10 clean.)

**Reading.** Taking real data away costs 1.41 at 20° (d 5 → 15). Simulation's value rises with it, +0.13 → +0.29,
and **all of that rise is regularisation**: simulation restricted to the regimes where real data remains is worth as
much as simulation everywhere. Simulation at the removed angles (the ones bridging data and target) and at the target
itself is inside ±0.03 at every level; the one contrast that passes the read rule (target, d = 10, +0.025, 3/3) is at
the fold resolution and gone again at d = 15, so it is not an onset. Which regimes the simulation comes from does not
matter, only that there is simulation — the same thing the record found by scrambling it.

**Against the predictions.** d = 5 transfer ≈ 0 ✓; transported worse than raw at d = 5 ✓ (−0.073, more than the
mimic −0.032); real-only degrading steeply ✓. Transfer positive by d = 15 ✗ — though for raw simulation the statics
only predicted parity at d = 15 (mean flow 0.359 vs 0.373, fluctuations worse), so no crossover is consistent with
them. Transported > raw at d = 10 ✗ (−0.002), although both the statics (0.22 vs 0.31) and the leak favour it; at
d = 15 ✓ but leaky. So where time-averaged statistics say better-matched simulation should win, the forecaster does not
take it up even with leaked help: within d ≤ 15° the blocker is not the bracket alone, as §3c's last prediction framed
it. §3d has a likely reason: the statistics compared there are blind to the clock.

**Poster line.** Weakening interpolation makes simulation more valuable, but as a regulariser of the real data that
remains, not as a bridge to the regime that is missing. Bridging has not started 15° from the nearest measurement.

### 4d. Re-clocked simulation, 2026-11-03 18:40 UTC (20/20 runs; `report_clk_mimic.txt`, `report_thin_clk.txt`,
`clk_steps.txt`, `clock_read_clk.out`)
**Mimic folds:** `rbclk` − `rbbase` −0.011 ± 0.007 (0/4), inside noise. With dense real data the clock does not matter.

**20° fold, best-step read** (the rig's usual max over evals), paired by seed:

| contrast | d = 5 | d = 15 |
|---|---|---|
| re-clocked − raw, none at 20° | +0.011 ± 0.010 (2/3) | **+0.184 ± 0.029 (3/3)** |
| re-clocked − raw, incl. 20° | −0.004 ± 0.009 (2/3) | **+0.226 ± 0.063 (3/3)** |
| target, re-clocked (incl. − none) | −0.006 ± 0.014 (2/3) | +0.048 ± 0.014 (3/3) |
| target, raw | +0.008 ± 0.011 (2/3) | +0.006 ± 0.021 (2/3) |
| sim − real only: re-clocked / raw | +0.126 / +0.12 | +0.506 / +0.32 |

**At fixed steps** (`clk_steps.txt`) — needed because at d = 15 the best step is 1000 (re-clocked) vs 2000 (raw):
* re-clocked − raw (none at 20°) at d = 15 is positive at every step, 3/3 seeds: +0.24, +0.13, +0.16, +0.18, +0.25, +0.29,
  +0.30, +0.30 at steps 1000…8000. Real-only decays 60.53 → 60.15 and raw sim 60.86 → 60.67 from step 2000 to 8000; the
  re-clocked arm stays at 60.96–60.99. **Robust.**
* the target contrast with re-clocked simulation is +0.074 and +0.052 at steps 1000/2000 (3/3), then ≈ 0 and −0.03 from
  step 4000 on: **transient**, an early-training effect that the best-step read picks up.
* at d = 5, re-clocked − raw is +0.11 at step 1000, ≈ 0 by step 5000, −0.04 at 8000.

**Per cell at d = 15** (rel_l2, best checkpoints): the re-clocked gain sits where the raw clock was slow — −0.6 to
−1.4 % at Re 3750–6300 (raw clock 1.2–1.8×, right or fast), +3 % from Re 8850 rising to +8–9.5 % at Re 13950–19050
(raw clock 0.57–0.62×), +4–8 % at the top; 13/16 cells better. Target transfer per cell: −0.37 %, 5/16 (the fold-level
target gain is in MVPE, +0.14, and early).

**Against §3e.** Mimic within ±0.03 ✓. d = 5 ≈ raw ✓. d = 15 regulariser "at most slightly above raw" ✗ — it is +0.18
(best step) to +0.30 (final step), half again as much as the raw simulation's whole contribution; located at the
high-Re cells as predicted ✓. Target transfer ≈ 0 ✓ at the end of training, ✗ early (+0.05–0.07).

**Reading.** Yes, the clock was part of why simulation paid so little: wherever real data is sparse, a simulation
taught at the wrong per-frame pace was worth ~40 % less than the same simulation re-clocked (+0.32 vs +0.51 best step;
+0.52 vs +0.82 at step 8000), and the loss sat at the Reynolds numbers where the CFD ran slowest relative to the tunnel.
With dense real data it does not matter. Simulation at the target angle still does not transfer durably.

**Added 2026-11-03 18:45 UTC, pre-registered: `th15clkp`** (a20, re-clocked, `--sim-paired`: simulation only at 0–5°,
3 runs). clkx − clkp = what correctly clocked simulation at the removed angles (10°, 15°) is worth — bridging if
positive (the raw equivalent was +0.028 ± 0.027). Prediction: clearly positive this time, because the per-cell gain
tracks the clock and the removed angles are where the forecaster has no real dynamics to learn from.

**`th15clkp` result, 2026-11-03 19:55 UTC** (3/3 runs; `report_thin_clk.txt`, `clk_steps_paired.txt`). Logs confirm 30
real trajectories at 0–5° and the re-clocked simulation at the same 30 cells, equal mass.

| d = 15, paired by seed | best step | step 2000 | 4000 | 6000 | 8000 |
|---|---|---|---|---|---|
| re-clocked sim at the removed angles (clkx − clkp) | **+0.056 ± 0.007 (3/3)** | +0.002 | +0.002 | +0.083 (3/3) | +0.128 (3/3) |
| raw sim at the removed angles (rawx − rawp) | +0.028 ± 0.027 (2/3) | +0.028 | +0.009 | −0.003 | +0.048 (3/3) |
| re-clocked regularisation (clkp − real) | +0.450 ± 0.041 (3/3) | +0.458 | +0.628 | +0.684 | +0.687 |
| clock fix at matched regimes (clkp − rawp) | +0.156 ± 0.047 (3/3) | +0.156 | +0.190 | +0.201 | +0.221 |

Prediction (clearly positive) ✓ at the best step and from step 5000 on; ≈ 0 at steps 2000–4000, so not uniform over
training. Caveat: clkx draws from 80 simulated trajectories, clkp from 30, so part of the difference may be diversity
rather than the angles' content — but the raw pair has the same 80-vs-30 difference and reads ≈ 0, so whatever the
removed-angle simulation contributes appears with the clock fix.

**Decomposition at d = 15 (best step, re-clocked):** sim − real only ≈ +0.51 = regularisation at the regimes real data
still covers +0.45 (of which the clock fix +0.16) + simulation at the angles between data and target +0.06 +
simulation at the target ≈ 0 durably (+0.05 early, fades).

**Interpolation vs bridging, as it now stands.** With dense real data (mimic folds, 20° with real 5° away) the forecaster
interpolates and simulation is a regulariser whose regimes do not matter. As real coverage recedes, simulation's value
grows several-fold, still mostly as regularisation; a bridging component — correctly clocked simulation at the
unmeasured angles between data and target — appears only once the simulation runs on the measurement's clock, and is
modest (+0.06 to +0.13). Simulation at the target regime itself never transfers durably. What bridges is the simulated
*dynamics* at intermediate regimes, not the target's statistics — consistent with the CFD's poor static regime trend
(§3d) and with the transported set, which fixed the statistics and not the clock, never helping.

**Proposed, not run:** the clean transported arm at d ≥ 10 — retrain the flow map without real at the removed angles
(`tools/flow_sim2real.py` and `tools/translate_sim.py` at 951df90, `src/realistix/models/flow.py`).
