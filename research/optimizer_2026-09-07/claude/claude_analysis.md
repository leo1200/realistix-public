# Optimizer deep dive — Claude's analysis and pre-registration (2026-09-07)

> **This is the raw running record, kept for the per-run numbers.** The distilled,
> authoritative account is `docs/log/2026-09-optimizer.md`; where the two disagree the
> log is right, because it was written after the board scored 918211 and after Codex's
> checkpoint-age audit corrected the ensemble read. Sections below are in the order they
> were written, so early entries state things later ones overturn (notably: the ensemble
> read of +0.028, corrected to +0.009 age-matched; and the "target profile is flat"
> premise behind `--w-elead`, refuted by an oracle).

Companion to `../codex/brief1.md` (the joint brief) and `../codex/codex_report1.md` (Codex's round).
Screens queued 2026-09-07 ~10:50 UTC through `research/potential_2026-09-05/claude/launch_arms.sh`
(shipped recipe + one flag, snapshots kept, a100). Read with
`python research/potential_2026-09-05/claude/hist_read.py bt_<arm> kf_tke1em8 mim1,mim2,mim3,mim4 0,1 6000 6000`
(mimic, board geometry) and `screen_read.py` against `bt_r2_angle{05,10}_base_s{0,1}` (novel angles).

## What the optimizer is, in one paragraph
67 matrix-shaped tensors on Muon (Nesterov 0.95 → 5 Newton–Schulz steps in bf16 → aspect factor → lr 0.02
cosine over 8000 steps, wd 1e-4 i.e. a 2e-6 per-step shrink), 97 0/1-D tensors on AdamW (lr 3e-4, wd 1e-4,
gradients never zeroed in the shipped path), global clip 1.0 before both, bf16 autocast, EMA 0.998 of the
parameters from init, checkpoint = EMA at step 6000 of 8000 (lr at 14.6 % of peak). Batch 32 mixes real
and simulated windows by window count.

## Structural observations that were not in the record
1. **28 `nn.Embedding` tables are on Muon** (the 24 per-modality FiLM affines `(2, 2C)` plus the block-level
   modality tables; Codex's count of the inventory: 32 3×3 kernels, 5 1×1 skips, 2 ConvTranspose, 28 tables). A 2-row matrix's polar factor forces the real-row and sim-row
   updates to be orthonormal at every step, and the per-element step is 0.02/sqrt(2C) = 5e-4…1.4e-3, i.e.
   2–5× the AdamW lr those tables would otherwise get. Muon's own guidance sends embeddings to AdamW.
2. **The two `ConvTranspose2d` weights are flattened on the wrong axis** — `(C_in, C_out·16)` rather than
   `(C_out, C_in·16)`; the polar factor is of a different unfolding than for every other conv.
3. **Weight decay has never been tested above lr·wd = 6e-5 per step.** Under Muon the weights grow 5.5×, so
   for every GroupNorm-followed conv the *angular* step (∝ lr/‖W‖²) decays by an unmanaged factor on top of
   the cosine. Standard Muon practice is wd ≈ 0.1 at lr 0.02 (lr·wd = 2e-3).
4. **Order of momentum and polarisation.** The shipped rule is msign(EMA(g)); EMA(msign(g)) is a noise-aware
   polarisation (persistent directions keep σ ≈ 1, per-step-fresh directions average to ≈ sqrt((1−β)/(1+β))
   = 0.16). At batch 32 the gradient is noisy and the shipped order re-amplifies every surviving direction to
   unit gain. Momentum 0.98 helping members (+0.059) but not ensembles (+0.036) says the per-step noise
   in the update matters and is also part of the members' diversity — so this arm must be read at
   ensemble level if it is positive per member.
5. **Every averaging variant tried so far is post-hoc** (EMA, snapshot soups, raw windows) and each only
   reproduced an earlier effective checkpoint. Switch-EMA (periodically reset the online weights to the EMA)
   changes the training path instead, which is the one thing those estimators cannot do.

## Pre-registered screens (all queued 2026-09-07, mimic 4 folds × seeds 0/1 unless stated)
| arm | flag | mechanism | bar to continue | kill number |
|---|---|---|---|---|
| `bt_polar` | `--muon-order polar_first` | noise-aware polarisation (obs. 4) | ≥ +0.02 accuracy-only at 6000, ≥ 3/4 folds | ≤ 0 at 6000 and at own best step |
| `bt_polar04` | `… --lr-muon 0.04` | same; the effective spectral norm of EMA(msign) is < 1, so lr may need to double | same | same |
| `bt_polar` angle05/10 | same | angle read of the same arm | ≥ 1 % equal-angle rel_l2 cut | worse than base at both angles |
| `bt_embadam` | `--muon-embed-adamw` | embeddings off Muon (obs. 1) | ≥ +0.02, ≥ 3/4 | ≤ 0 |
| `bt_convt` | `--muon-convt-fix` | correct unfolding (obs. 2) | any consistent positive → bundle into hygiene | null → drop |
| `bt_wd03`, `bt_wd10` | `--wd-muon 0.03 / 0.1` | rotational equilibrium (obs. 3) | ≥ +0.02 at 6000 or own best | ≤ 0 at both |
| `bt_swema500` (+ angle05/10) | `--switch-ema 500` | training-path reset to the flat average (obs. 5) | ≥ +0.02 mimic or ≥ 1 % angle cut with mimic ≥ −0.02 | ≤ 0 on both instruments |

Priced honestly: the fold rig resolves ≈ 0.02–0.03 per paired comparison; anything at +0.02 is a lead, not a
result, and needs the ensemble read (`tools/fold_ensemble.py`) plus the angle and far-Re guards before a
build. Effects found by searching these same four folds are selected quantities (kfold.py `score_of`
docstring); a positive here must be confirmed on the angle suite or the board.

## Not implemented yet (waiting on Codex's verdict)
- H5 Dion-style low-rank polar with error feedback (explicit noise floor).
- H6 output-layer temporal metric Δ = S·msign(S·G).
- H11 modality-split polar update (two momentum buffers, real / sim).
- H10 the interval head's optimization procedure (lr response monotone to collapse; EMA/warmup/clipping unexamined).

## Round-1 cross-examination (Codex `../codex/codex_report1.md`, 2026-09-07 ~11:00 UTC)
Codex's independent ranking: **(1) polar-first, (2) embeddings → AdamW, (3) stem input-covariance
preconditioning** (its own candidate: right-multiply the stem momentum by (C + 0.1·tr(C)/40·I)⁻¹, C the 40×40
second moment of the stem input, before NS; a restricted Newton–Muon). Its 200-step common-init probe:
polar-first vs shipped, mean training loss over the last 50 steps 1.705 vs 1.789, four independent batches
1.699 vs 1.740 (4/4 lower); the clip is active on 100 % of steps (median multiplier 0.34 shipped, 0.085
polar-first); 98–99.7 % of the shipped convolution update's spectral energy sits below a heuristic
odd/even-half noise floor (FiLM tables 0 %) — a conservative heuristic, not a licence to truncate.

Corrections accepted:
- **H2 was confounded as first queued**: `--muon-embed-adamw` parked the tables in the never-zeroed AdamW group.
  Fixed: the moved embeddings' grads are set to None every step (the rest of the group keeps the shipped
  behaviour). The eight pending `bt_embadam` jobs were cancelled and re-queued after the fix.
- **H1's clip confound**: the rest group's accumulated gradient obeys q_t = c_t (q_{t−1} + h_t) with c_t the clip
  multiplier, and polar-first shrinks c_t 4×, so it also shortens the AdamW group's memory. Control queued:
  `bt_polarzr` = polar-first + `--zero-rest`, to be read against `bt_zerorest_mim*_s{0,1}`.
- **H4's argument was quantitatively wrong** (angular motion ≈ η‖U⊥‖/‖W‖, not η/‖W‖²; the 5.5× is Muon-vs-AdamW
  finished norm, not within-run growth; only the 14 `c1` kernels are gauge-invariant). The wd 0.03/0.1 arms
  stay queued as exploratory; wd 0.1 alone would shrink the initial weights to 0.0004 of themselves by step 6000.
- **H3 is a different metric, not a bug fix** (adjoint operator; either unfolding is a choice), and the
  output-first flatten lowers those two layers' ideal update norm by √2. Kept as its own arm.
- **H8 is unusable as a drop-in** (freezes the zero-init `out` and the zero-init tables). Dropped.
- **H10**: the head has no EMA (real), but the "unsaturated lr" reading is confounded by the later τ/no-spread
  re-selection. Reserve: a short late average of the head, read against an age-matched raw checkpoint.
- **H11**: "unit gain regardless of share" is wrong in general (Frobenius-orthogonal gradients need not have
  disjoint singular subspaces). Reserve, behind H1, with a random-partition two-buffer control.

Bit-identity of the default path: the HEAD trainer and the patched working tree both print
`step 40 loss 2.10270 | mean 69.27 | mvpe 91.136` on the same 40-step run (seed 0, GPU 1).

Additional arms queued after the cross-examination: `bt_polarzr` (mimic), `bt_embadam` (fixed, mimic),
`bt_stemcov` (`--muon-stem-cov 0.1`, mimic). `bt_polar` mimic moved to the idle local GPU
(`run_local.sh`, log `local_polar.log`) because the a100 pool was full; its pq jobs were cancelled.
Codex round 2 (`../codex/brief2.md`) is an implementation audit of every new flag, running now.

## Reads, 2026-09-07 (accuracy-only Δ vs `kf_tke1em8` at step 6000; hist_read.py)
| arm | read | verdict |
|---|---|---|
| `bt_polar04` (polar-first, lr 0.04) | **−0.061, 0/4** at 6000 (−0.060 at 5000, −0.067 at 8000); rel_l2 +0.7…+2.3 % on 8/8 | **dead** |
| `bt_polar` (polar-first, lr 0.02) | mim1 s0 −0.055, s1 −0.094 (rel +1.2/+1.8 %, tke worse, mvpe mixed); own end (8000) no better | **dead** — the remaining local runs were stopped after 2 of 8 (0/10 across both doses); `bt_polar10` stays queued only to settle the dose direction, `bt_polarzr` is left in the queue as the clip-memory control |
| `bt_convt` | −0.011, 1/4 at 6000 (−0.006 at 5000, −0.044 at 8000); the step-1000 rel_l2 lead (7/8) did not survive | **null** |
| `bt_wd03` | mim1 +0.005/+0.004, mim2 s0 **+0.045** (rel −1.4 %, tke −1.3 %, mvpe −1.6 %) at 6000; 5 runs pending | **lead** — moved with `bt_wd10` to the local GPU (`run_local_wd.sh`, `local_wd.log`) because the a100 pool emptied of our jobs again |

Codex round 2 (`../codex/codex_report2.md`): every flag passes; defaults reproduce HEAD to the tensor hash over
26 steps; recommended the polar second dose 0.1 instead of 0.04 (queued as `bt_polar10`); its combination-guard
diff (`../codex/audit2_fixes.diff`) is applied.

### Later reads (same protocol)
| arm | read | verdict |
|---|---|---|
| `bt_swema500` (Switch-EMA every 500) | **+0.002, 8 runs**, range −0.009…+0.021 | **null** — resetting the online weights to the EMA changes nothing the EMA itself did not already do |
| `bt_polarzr` vs `bt_zerorest` (polar-first with both groups zeroed, against the zero-rest baseline) | mim1 −0.059 / −0.088 | the clip-memory confound is **not** the cause; polar-first is dead on its own terms. Pending `bt_polarzr`/`bt_polar10` jobs cancelled |
| `bt_wd03` | 5/5 so far, mean +0.017 (mim3 s0 +0.015) | lead; wd 0.01 dose and seed 2 (for `fold_ensemble.py`) queued |

### wd 0.03 — 7 of 8 runs in (2026-09-07 ~12:45 UTC)
mim1 +0.005/+0.004, mim2 +0.045/+0.019, mim3 +0.015/+0.022, mim4 s0 +0.017 → **+0.018, 4/4 folds, 7/7 runs**,
rel_l2 −0.3…−1.4 %, tke and mvpe never worse by more than noise. Its stopping-step profile (vs base@6000):
4000 −0.026, 5000 +0.006, 6000 +0.018, 7000 +0.006, 8000 −0.019; the baseline's own is 5000 +0.005, 7000 −0.020.
So the arm's argmax stays at 6000 and its curve is flatter late (7000 costs 0.012 instead of 0.020) — consistent
with decay damping the late drift. Embedding→AdamW: −0.003/+0.001 on mim1 (null so far).
| `bt_embadam` (28 embedding tables → AdamW, grads zeroed) | **−0.003, 1/4** (mim1 −0.003/+0.001, mim2 +0.021/+0.010, mim3 −0.013/−0.016, mim4 −0.007/−0.015) | **null** — the shipped placement of the FiLM tables on Muon is not costing anything |

### Complete reads (2026-09-07 ~13:10 UTC), accuracy-only Δ vs `kf_tke1em8` @6000
| arm | per-run Δ (mim1 s0/s1, mim2, mim3, mim4) | mean | folds + | verdict |
|---|---|---|---|---|
| **`bt_wd03`** (`--wd-muon 0.03`) | +0.005/+0.004, +0.045/+0.019, +0.015/+0.022, +0.017/+0.016 | **+0.018** | **4/4 (8/8 runs)** | **real, small**: rel_l2 −0.3…−1.4 %, tke/mvpe never worse. Below the +0.06 bar, above the rig's 0.02–0.03 resolution only in aggregate. Ensemble read (seed 2) and angle/far guards queued |
| `bt_wd01` (`--wd-muon 0.01`) | −0.003/+0.004, ·/+0.005, ·/+0.011 (4 of 8) | +0.006 | 3/3 | dose response in the right direction (0.01 < 0.03); wd 0.1 running locally |
| `bt_stemcov` (Codex #3) | +0.009/+0.008, +0.022/−0.010, 0.000/+0.001, +0.009/+0.012 | +0.006 | 4/4 | **too small to matter** — consistent sign, but inside the rig's resolution; not funded further alone |
| `bt_embadam` | see above | −0.003 | 1/4 | null |
| `bt_swema500` | see above | +0.002 | 3/4 | null |
| `bt_convt` | see above | −0.011 | 1/4 | null |
| `bt_polar`, `bt_polar04`, `bt_polarzr` | −0.055/−0.094; −0.061 (0/4); −0.074 vs zero-rest | | 0/10 | dead |

## Round-3 cross-examination (Codex `../codex/codex_report3.md`, ~13:40 UTC)
- **Mechanism** (from the EMA snapshots, `diagnose3.py/json`): broad shrinkage of *every* non-table matrix
  (‖W‖_F ×0.39–0.60 at 6000, σ_max likewise, stable rank ≈ unchanged) — c2 residual-output and `out` shrink as
  much as the gauge-invariant c1 layers, so it is not a gauge-only angular-lr effect; plus a modestly flatter late
  trajectory (same-step Δ 4000 −0.017, 6000 +0.018, 7000 +0.027, 8000 +0.024). Note the 28 FiLM tables are on the
  decay arm too (they are Muon-group matrices).
- **Is it real?** fold means +0.005/+0.032/+0.018/+0.017, SE 0.0056 across folds, t₃ CI [+0.0002, +0.036];
  8/8 runs but 4 fold signs (p = 0.06). A credible small lead, not eight replications.
- **Dose curve** (fixed-6000 vs the same baseline): 1e-3 +0.006, 3e-3 +0.003, 0.01 +0.009 (6 runs), 0.03 +0.018.
  Not monotone at the low end; three fits disagree on 0.1 (+0.02…+0.06). Pre-read forecast for wd 0.1:
  +0.015 [−0.04, +0.05]; angle folds +0.01 [−0.03, +0.04].
- **Pricing**: rel +0.023 / tke +0.053 / mvpe +0.017 subscore locally → +0.006…+0.011 final from rel alone, ≈ +0.016
  if all three transfer, +0.027 with a fixed interval factor. **Pre-registered line (Codex): wd03-only vs 915299
  (81.754): +0.01 [−0.05, +0.06] final, no time change.** Against today's champion 917205 (mvpe03, 81.763):
  +0.001 [−0.06, +0.05]. Gates before any build: ensemble read (fold_ensemble, note it loads best.pt not step
  6000), angle guard (equal-angle Δ ≥ 0, neither angle < −0.02), far-Re guard (each ≥ −0.02), refitted head.
- Keep 6000; keep AdamW wd 1e-4; the one composition arm worth a line is wd 0.03 + `--w-mvpe 0.3` against mvpe03.
- On "closed": record dose range, recipe, estimand and reopening condition; the 3e-3 closure was a range, not a
  mechanism. Agent agreement on polar-first was correlated judgment, not validation.

### Dose and confirmation reads (rolling)
- `bt_wd01` complete: **+0.006, 3/4** (mim4 s1 −0.013 the one negative) — half of wd 0.03's read; dose curve 0.01 → 0.03 rising.
- `bt_wd10` mim1 s0: −0.008 (wd 0.03 gave +0.005 there) — over-regularised at 0.1, pending the other 7 runs.
- `bt_wd03` seed 2: mim2 +0.016 (9/9 runs positive so far).
- `bt_wd03` seed 2 complete: +0.037/+0.016/+0.012/+0.015 → **12/12 runs positive**, three-seed fold means
  +0.016/+0.027/+0.016/+0.016 (mean +0.019).
- `bt_wd10` mim1 s1 −0.008 (both mim1 seeds −0.008): 0.1 over-regularises.
- **Angle guard, first read**: angle05 s1 **+0.071 (rel −1.85 %)** at 6000 vs `bt_r2_angle05_base_s1`.
- **Far-Re guard, first read**: far_drop s1 **+0.021 (rel −0.95 %)** vs `bt_r4_screenbase_far_drop_s1`.
- **Ensemble read, partial** (`fold_ensemble.py`, best.pt per run, only mim2 had all three wd seeds finished):
  member +0.015 → ensemble **−0.007** on mim2 (base ensemble gain +0.021, wd03's −0.001). To be re-read on all four
  folds when the seed-2 runs reach 8000 — this is the number that decides the build.

### Confirmation reads, 2026-09-07 ~13:30 UTC — wd 0.03 passes every pre-registered gate so far
**Three-member ensemble read** (`fold_ensemble.py tke1em8 wd03 --seeds 0,1,2`, best.pt per run, ACC weighting):
| fold | base member → ens | wd03 member → ens | ensemble Δ |
|---|---|---|---|
| mim1 | 63.088 → 63.018 | 63.097 → 63.084 | **+0.066** |
| mim2 | 63.534 → 63.555 | 63.549 → 63.548 | −0.007 |
| mim3 | 63.210 → 63.231 | 63.238 → 63.259 | +0.028 |
| mim4 | 63.218 → 63.218 | 63.233 → 63.243 | +0.025 |
Per-member +0.017 → **ensemble +0.028, compression 1.68**, 3/4 folds, none below −0.02. No ensemble compression —
the failure mode that sank `--mults 1,2` is absent. Caveat: the members' per-element spread is ~7 % smaller under
wd 0.03 (spr 0.00215 vs 0.00233 on mim1 etc.); the interval head reads that spread, so the SPS width channel could
give back a little — the refitted head must be read before pricing SPS.

**Angle guard** (step 6000 vs `bt_r2_angle*_base`): angle05 s1 **+0.071** (rel −1.85 %), angle10 s0 **+0.043**
(rel −2.11 %); two runs still training. **Far-Re guard** (vs `bt_r4_screenbase_far_*`): far_drop −0.001 / +0.021,
far_keep +0.015 / +0.035 — all ≥ −0.02, rel never worse.

**Dose**: 0.01 +0.006 (3/4), 0.03 +0.018 (12/12 runs), 0.1 −0.008/−0.008 on mim1 (local runs stopped there to
free the GPU for a build; the decision does not need the rest of the 0.1 curve).

### Toward a candidate (2026-09-07 ~14:10 UTC)
The board's champion moved this morning to **917205 = mvpe03b (81.763)**, i.e. the shipped recipe + `--w-mvpe 0.3`.
A wd-only build would compete against it with ≈ 0 expected edge (Codex round 3), so the build that makes sense is the
**composition wd 0.03 + w_mvpe 0.3**, which is unmeasured. Queued its screen on the cluster: `bt_mv03_mim{1..4}_s{0,1}`
(w_mvpe 0.3 under the current recipe — the archived `kf_mvmu03` baseline is the *old* recipe, tke 0.5 / EMA 0.999)
against `bt_wd03mv_mim{1..4}_s{0,1}` (both knobs). Started the three-member build speculatively on GPU 1
(`WD_MUON=0.03 W_MVPE=0.3 NAME=wd03mv SUB=wd03mv CHAMPION=mvpe03b bash tools/train_final.sh`, log
`build_wd03mv.log`; `WD_MUON` is a new append-only knob in `train_final.sh`, `check_recipe.py` still green). The
build is a reversible GPU-hour; whether it is uploaded is the owner's call after the composition read.

### Composition screen, partial (~14:50 UTC)
`bt_mv03` (w_mvpe 0.3 alone, current recipe) vs base: **+0.003, 2/4** (8 runs) — mvpe-only is board-positive through
the mvpe channel, not the fold aggregate, as its manifest said. `bt_wd03mv` (wd 0.03 + w_mvpe 0.3) vs `bt_mv03`:
mim1 +0.026/+0.018, mim2 +0.010/+0.026 → **+0.020, 4/4 runs**, rel −0.3…−0.8 %, mvpe better on 3/4: wd's gain
carries over on top of the mvpe term unchanged. mim3/mim4 pending.

### Composition screen, complete (~15:20 UTC)
| comparison | Δ | folds + | runs + |
|---|---|---|---|
| `bt_wd03mv` vs `bt_mv03` (adds wd 0.03 on top of the champion's recipe) | **+0.016** | 4/4 | 8/8 (+0.026/+0.018, +0.010/+0.026, +0.011/+0.007, +0.005/+0.026) |
| `bt_wd03mv` vs base | +0.019 | 4/4 | |
| `bt_mv03` vs base | +0.003 | 2/4 | (its value is the board's mvpe channel, as its manifest predicted) |
| `bt_wd03mv` vs `bt_wd03` | +0.001 | 2/4 | the two knobs are additive, neither interferes |
rel −0.3…−0.8 % on all eight paired runs; tke and mvpe never worse by more than noise. **wd 0.03 composes cleanly with
`--w-mvpe 0.3`**, so the build under way (`wd03mv`) is the right object.

### Angle guard complete (~15:55 UTC)
angle05 s0 **+0.052** (rel −2.41 %), s1 **+0.071** (−1.85 %); angle10 s0 **+0.043** (−2.11 %), s1 **+0.063** (−2.72 %) —
**mean +0.057, 4/4**, equal-angle rel_l2 cut 1.9–2.7 %. This clears the angle suite's own promotion bar (≥ 2 % rel cut,
≥ 2/3 angles, none worse) on the two developed angles, and it is the strongest read wd 0.03 has produced anywhere:
the mechanism (shrinking every matrix, damping the late drift) helps most exactly where the record located the
open problem — late specialisation to the seen angles. 15° remains unopened (the suite's rule: lock before opening).

## Round-4 cross-examination (Codex `../codex/codex_report4.md`, ~16:20 UTC) — accepted
- The ensemble +0.028 was **checkpoint-age sensitive**: wd's best.pt sits at 6000 where the baseline's sits at 5000.
  Age-matched (wd read at the baseline's own steps) the ensemble Δ is **+0.009**, ensemble/member ratio 1.00, TKE-zeroed
  −0.002. No amplification; the honest ensemble expectation is the member-level gain. Baseline snapshots do not exist,
  so a clean fixed-6000 three-member comparison cannot be made without retraining the baseline with `--snapshots`.
- Prediction revised to **+0.00 [−0.06, +0.05] vs 917205**, adopted in the manifest.
- Before arming: the manifest's verification list was extended (args of every member, sps recorded against 51.008,
  extracted-zip smoke test). Codex's honest one-sentence claim for the report is recorded in its file.

## Build and verification (~16:40 UTC)
`wd03mv` built and packed: check_pack MATCH, zip 149.8 MB, standalone chunk-1 smoke test identical, inference code =
mvpe03b's. In-sample val_aoa 95.423 / 82.910 / 95.835 / 50.865 vs 917205's 95.483 / 83.223 / 95.902 / 51.008 — all
floors pass; every number a little lower, as a regulariser's in-sample fit should be, recorded as the risk in the
manifest. Not armed; the slot decision is the owner's.

# Round 2 of ideation — architecture and physics-inspired losses (2026-09-07, from ~14:30 UTC)
Owner's instruction: one more joint pass on architecture / physics losses; if nothing survives, ship wd03mv.
Codex brief `../codex/brief5.md`, report `../codex/codex_report5.md`.

## Two structural signatures found by probing the shipped members (`lead_profile.py`, `lead_anomaly.py`)
Fold mim1 held-out windows (201), `kf_tke1em8_mim1_s{0,1}` (both seeds agree; mim2 shows the same with the spike at a
different lead):
1. **Front-loaded fluctuation energy.** Prediction fluctuation energy (about its own window mean) / target: lead 0
   **1.37**, lead 5 0.74, lead 10 0.65, lead 17 0.47, lead 19 0.45. Target's own profile is the usual U (1.28 → 0.90 →
   1.28) — the same in real and sim, inputs and targets. The window-TKE term only sees the sum, so a front-loaded
   profile satisfies it as well as the target's. Relative to the L2-optimal amplitude (ρ² per lead) the model is
   ~2× over-energetic at every lead — the tke/rel frontier — with the excess largest early.
2. **A discontinuity at one of the last leads.** Pooled |p_k − p_{k−1}|² / mean: 1.41, 1.55, 1.76, 1.06, 0.88 … 0.25
   (lead 17), **3.80 (lead 18), 2.95 (lead 19)**; the target's is flat (1.00 ± 0.03). 72 of 201 windows put their
   energy maximum at lead 18, led by the extrapolation column 26700_0 where the lead-18 error is 0.137 against a window
   mean of 0.060. Output-conv row norms grow with lead (18/19 the largest in every run). Data are clean at that frame
   (real and sim, inputs and targets). **wd 0.03 halves the spike** (1.92 at lead 19) and **polar-first doubles it**
   (4.05/4.42 at leads 17/18) — consistent with both screens' outcomes and with wd's angle-suite gains (extrapolation).
   Rough price on mim1: fixing lead 18 alone is ≈ 1.5 % of rel_l2 ≈ +0.026 accuracy-only.

## New loss terms (inert at default), screens queued on the mimic folds × seeds 0/1
| arm | flag | term |
|---|---|---|
| `bt_elead1`, `bt_elead03` | `--w-elead 1.0 / 0.3` | `losses.lead_energy`: relative L2 between per-lead fluctuation-energy profiles (20-vector per window) — TKE-neutral, MVPE-neutral by construction |
| `bt_dlead1`, `bt_dlead03` | `--w-dlead 1.0 / 0.3` | `losses.diff_energy`: relative L2 between per-lead frame-difference energy profiles (19-vector) — targets the late-lead discontinuity and the over-smooth mid-horizon |
Bar (pre-registered): ≥ +0.02 accuracy-only at 6000 vs `kf_tke1em8`, ≥ 3/4 folds, none < −0.02; also read the
lead-18 spike on the trained arms with `lead_anomaly.py` (the mechanism check). If nothing clears, ship wd03mv.

## Round-2 verdict and ship decision (14:28 UTC; earlier "~UTC" stamps in this file ran ~2.5 h fast)
- Codex round 5 (`../codex/codex_report5_recovered.md`; the harness clobbered the original with its last message):
  **zero candidates.** Target-energy redistribution at fixed local TKE reads **−0.030** on 4/4 folds (kills the
  stationarity rationale of `lead_energy`: a finite-window mean makes the target profile U-shaped, and the conditional
  forecast's energy need not be flat); a normalized Reynolds-stress-shape correction reads +0.020 *when handed the
  actual future covariance* — no margin for a learned term; exact-support gate +0.001; learned 12-mode temporal basis
  −0.00002 (the head already lives in 12 directions); best archived width profile +0.012.
- Claude's late-lead discontinuity (`spike_oracle.py`): real per member, but **at three-member ensemble level the
  spikes sit at different leads and average out**; the neighbour-average oracle reads −0.023 / −0.000 / −0.030 / −0.002
  (it damps TKE). The +0.026 single-member estimate does not survive the ensemble. wd 0.03 already halves it.
- `bt_elead*` / `bt_dlead*` screens keep running as background reads; they do not gate the slot.
- **Shipped decision: `wd03mv` armed for 2026-09-08T00:00:30Z** (owner's instruction), prediction +0.00 [−0.06, +0.05]
  vs 917205. Log `submit_wd03mv.log`.
