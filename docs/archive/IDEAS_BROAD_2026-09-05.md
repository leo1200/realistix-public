# Broad idea search across fields — what survives, what is dead, and why

Joint Claude (Fable 5.1) / Codex (gpt-6-astra) pass, 2026-09-05, two independent catalogues and a
cross-examination round. Companion to `docs/POTENTIAL_2026-09-05.md` (which fixes the current picture: no
accuracy vein on the same input; ~30 % window-mean share; angle generalisation as the open axis). Artefacts:
`research/potential_2026-09-05/{claude/claude_broad.md, codex/codex_report_round3.md, codex/codex_report4.md}`.

Verdicts: **DEAD** (closed here or by mechanism), **LOW** (mechanism weak or ceiling below the +0.06 bar),
**SCREEN** (worth one pre-registered screen), **STRONG** (mechanism plus a measured signal already in hand).
Screen conventions: **B** = four mimic folds × seeds 0/1, paired, fixed step, then ensemble + honest head;
bar ≥ +0.06 direct, ≥ 3/4 folds, none below −0.02. **G** = the angle-only suite
`bt_r2_angle{05,10,15}_base_s{0,1}` (angle withheld from real *and* sim, snapshots kept); develop on 5°/10°,
lock before opening 15°; bar ≥ 2 % equal-angle rel_l2 cut over the strongest confirmed cheap baseline, ≥ 2/3
angles, none worse by 0.02; final guards mimic ≥ −0.02 and far-Re mvpe ≤ +1 %. Costs are H100 / A100
GPU-hours at 0.25 / 0.6 per member.

## 0. Corrections to the record established by the archive checks

| claim in the record | correction | evidence |
|---|---|---|
| `model_analysis.md` §3.5/§6: "the ScaledMomentum control at lr ≈ 0.006 has not been run" | **it was run**: `kf_nsgd6`, 4 folds × 2 seeds, −0.102 direct (0/4) best-step, −0.101 at fixed 6000 | Codex round 3 archive scan; §7 of that file already supersedes §3.5 |
| "batch 64 −0.164" (`docs/MODEL.md`) | real but narrowly scoped: `kf_mub64`, 8000 updates at the *same* lr and step count (2× the sampled examples, not compute- or exposure-matched), old loss (tke 0.5, no sps), EMA 0.999; −0.164 at 8000, −0.071 at 6000 | archive metadata |
| "add Nesterov to Muon" | already on (`Muon(nesterov=True)` default, never overridden) | `src/realistix/muon.py` |
| "weight averaging is dead" | what was measured is averaging each run's own **EMA snapshots at 3k/4k/5k** under EMA 0.999, then ensembling — not raw-iterate SWAD/LAWA and not cross-seed soups | `tools/swa_eval.py`, HANDOFF §7 |
| "attention is flat" | spatial only (`BottleneckAttention`, `TransformerBottleneck` over the 8×16 latent grid); **no temporal-weight-sharing, SSM, shooting or consistency arm exists** among 2628 recovered run configurations | archive scan (`round3_archive.py`) |
| "the 6000-step schedule was tried" | only `sched6k_s0`: old pre-Muon recipe, one seed, AoA-20 split; **zero Muon runs at steps = 6000** among 783 Muon configurations; the shipped recipe is an 8000-step cosine read at 6000, i.e. lr still at 14.6 % of peak | archive scan |
| warmup "never tried" (`docs/log/2026-08-ideas-sessions.md`) | `kf_warm250` ran: rel Δ −0.0001 ± 0.0003, mvpe −0.0005 vs `kf_tke1em8` at 6000 — **null** | histories |
| `kf_mu12k` (Muon, 12k cosine) "peaks mid-schedule then declines" | it also shows an **early transient** (mim1 val_aoa rel 0.086 → 0.113 at step 3000, recovered by 6000; `kf_mum90` likewise) and bottoms at 8000–9000 at 0.0764–0.0767 vs the 8k recipe's 0.0773 at 6000 — Muon at lr 0.02 without warmup sits near an early instability that the faster 8k decay hides; never compared at fixed compute | histories |

## 1. The catalogue

Both agents' verdicts agree unless a cell says otherwise. "Closed" cites the measurement; "no arm" means the
archive scan found no run of it.

### A. Optimisation theory
| idea | mechanism here | closed / near neighbour | verdict |
|---|---|---|---|
| Muon as spectral-norm steepest descent (Bernstein–Newhouse) | the polar update `UVᵀ` *is* the solution of steepest descent under ‖Δ‖₂ ≤ η, so "high-rank flat-spectrum steps" and "spectral-norm control" are one mechanism, not two; five Newton–Schulz steps approximate it | mechanism measured (update stable rank 84 vs 6.6); matched-magnitude control `nsgd6` −0.102 shows magnitude alone is not it | settled as far as it can be without a causal control; the open descendants are below |
| update spectral **clipping** instead of polarising | rescales large singular values, does not raise weak directions | `nsgd6` adverse; no arm | LOW |
| **weight** spectral cap (projection after the step) | bounds ‖W‖₂ against the 5.5× norm growth; distinct from the update geometry. Note: Kimi's "MuonClip" clips attention QK logits — a different mechanism; do not borrow the name | wd 1e-3/3e-3 only +0.005; no arm; no instability at 8k shown | LOW — Codex; Claude had SCREEN. Settling test: measure 4k → 6k spectral drift first; if < 10 % or the cap acts as scalar damping, stop |
| aspect factor √(fan_out/fan_in) unclamped | the shipped factor is √max(1, rows/cols); unclamped values here span 0.036–1.41, so a blind convention change re-scales some layers' lr 30× | no arm; not new information | LOW |
| **variance adaptation after the orthogonalisation** (AdaMuon/NorMuon-style, magnitude-controlled) | normalise the *orthogonalised* update's coordinate variance, then rescale to its Frobenius norm — the one AdamW ingredient Muon lacks; `soap.py`'s SOAP-Muon preconditions *before* orthogonalising, so its null does not test this order. NorMuon's row normalisation is redundant on wide full-row-rank matrices (65 of 67 here are wide) | SOAP −0.025/−0.043, SOAP-Muon ≈ Muon at its own best step but −0.070 at fixed 6000 | **SCREEN** (Codex top-5 #4): 500-step diagnostic first (direction change ≥ 1 %, stable rank ≥ 80 % of the plain update), then B |
| momentum 0.98 / 0.99 / 0.90, ns_steps 2 / 10, wd up | all measured: 0.98 +0.059 per member but +0.036 as a 3-ensemble vs 0.95's +0.064 (heavier momentum correlates members); ns 2 −0.010, ns 10 −0.001; wd 1e-3 +0.005 | measured | DEAD |
| AdEMAMix long-memory momentum before NS | a second gradient timescale; weight EMA is not gradient memory | momentum-diversity loss above; no arm | LOW |
| **6000-step full cosine vs the shipped 8000-step cosine read at 6000** | anneal to zero at the deployment step (WSD/cooldown literature); the lr at the shipped read is 14.6 % of peak | only the old-recipe single-seed `sched6k_s0` before today; **run 2026-09-05 (`bt_r4_cos6`, 8 runs): −0.064 mean, 8/8 negative** (§2) | **DEAD** — annealing to zero consolidates the late specialisation |
| schedule-free / WSD | different path, not just final lr | many longer schedules and EMA doses failed | LOW (only after the 6k screen, and it must beat it) |
| compute-matched batch/lr scaling | the −0.164 was 2× examples at the same lr and steps | see §0 | LOW |
| SOAP / Shampoo / K-FAC / PSGD / Sophia / natural gradient | curvature preconditioning; SOAP recovers two-thirds of Muon alone and adds nothing on top | SOAP measured; the rest have no arm and give up the matrix geometry that paid | DEAD (SOAP), LOW (rest) |
| **SAM / ASAM / GSAM** | weight-neighbourhood flatness — the DG literature's one reliable family; the training signal (perturbed-weight gradients) is absent from the recipe and from EMA | no arm; `model_analysis` §6's rejected *input*-smoothness rationale is a different thing | **SCREEN on G only** (Codex top-5 #2; 2× cost; matched random-perturbation control mandatory) |
| **raw-iterate dense averaging (SWAD / LAWA / Polyak tail 3001–5000, 4001–6000)** vs EMA 0.998 | EMA 0.998 has mean lag ≈ 499 steps and 98 % of its mass in the last 2000; a wide uniform window ending *before* the angle late-rise is a different estimator | what was called "weight averaging dead" averaged EMA snapshots under EMA 0.999 (§0) | **SCREEN** (Codex top-5 #1 secondary; needs raw weights carried in reruns — saved checkpoints hold EMA weights only) |
| LR warmup | edge-of-stability at lr 0.02 | `kf_warm250` null at 8k | DEAD at 8k; relevant only if schedules lengthen (§0, the 12k transient) |

### B. Deep-network theory and domain generalisation
| idea | mechanism | closed / near neighbour | verdict |
|---|---|---|---|
| **late specialisation to seen angles** (held-out-angle error rises after step 4000 in 16/16 runs, held-out-Re does not) | feature learning of angle-specific structure once shared structure is fit; EMA lag is the alternative reading — separable only with raw weights | measured today (§2 below) | **STRONG** as a phenomenon; lever = stopping step / averaging window / SAM / balanced batches |
| flat-minima *explanation* | parameterisation-dependent (Dinh et al.); simulation measurably *sharpens* minima here | `model_analysis` §0.5 | LOW (diagnostic only; G decides SAM, not a Hessian trace) |
| GroupNorm gauge / renormalising weights | the residual branch ends without GN, so norm growth is not a pure gauge; but wd-up bought +0.005 | measured | DEAD as a global renormalisation |
| LayerScale / residual scaling | closed by mechanism in `model_analysis` §6 (never measured directly) | argument | DEAD |
| muP / more width | flat 96–192 | measured | DEAD |
| angle-balanced ERM / group DRO | the sampler is uniform over window starts, not over angle groups (angles have 14–18 trajectories each); DRO's worst-group weighting conflates intrinsic high-angle difficulty with excess risk | no arm; regime bagging negative | balanced ERM = the control of the meta-update screen; DRO LOW alone |
| **cross-angle first-order meta-update** (MLDG-style, a step on angles A must still lower loss on training angle B) | ordinary ERM never evaluates post-step cross-angle transfer | proposed in `POTENTIAL_2026-09-05.md` §6; DomainBed caution: tuned ERM is hard to beat | **SCREEN** (Codex top-5 #5) with two controls: balanced ERM and random partitions with identical machinery |
| IRM / Fishr / V-REx literally | invariance can erase real angle physics | no arm | LOW (only after the meta-update result) |
| cross-seed soups / re-basin | permutation-aligned merging to save two forward passes | never tested (the "dead" result was within-run EMA snapshots) | LOW (time channel ≤ +0.10 and the single-member board losses are the adverse neighbour) |

### C. Multiple shooting and trajectory-level training
| idea | mechanism | closed / near neighbour | verdict |
|---|---|---|---|
| unrolled / recurrent single shooting | closed: −0.039/−0.074/−0.114 monotone in dose | measured | DEAD |
| full-trajectory latent multiple-shooting teacher (shooting nodes + continuity over 868 frames, distilled to a 20-frame encoder) | the classical cure for the exploding-gradient failure above; supplies long-horizon supervision | AE + latent flow reconstruction TKE 0.22–0.31; distillation anomalous on the board; the slow-mode premise weakened (30 % share, parity) | LOW — gate: the teacher's 20-frame-inferred state must cut blocked held-out slow-mode MSE ≥ 12 % before any student is trained |
| parallel-in-time | training wall-time only | n/a | DEAD as an accuracy idea |
| long-horizon auxiliary (frames 41–80 mean) | see `POTENTIAL` §6 candidate 2 | premise weakened | LOW; only with the shuffled-target control |
| **overlap consistency** (f(X_s)[k:] must equal f(X_{s+k})[:−k]) | the later window contains *newer observations*, so optimal forecasts legitimately differ — the exact constraint is mis-specified and would remove innovations (fluctuation energy) | argument; no arm | DEAD as an exact constraint (both agents; Claude's draft had it as a cheap screen) |
| consistency on a common information set (same observation cutoff, two views) | regularisation only | no arm; input-noise augmentation adverse | LOW |

### D. Dynamical-systems theory
| idea | mechanism | closed / near neighbour | verdict |
|---|---|---|---|
| Koopman / DMD / delay extrapolators | closed (+0.0002 incremental R²) | measured | DEAD |
| Takens embedding | the U-Net already has nonlinear access to the full 20-frame delay vector; embedding theorems do not certify observability of a noisy open 3D subsystem from a 2D plane | argument | DEAD as new information |
| symmetry reduction / slices / template shifts | no advection or phase error to remove (rigid-shift oracle removes < 6 %) | measured | DEAD |
| slow / inertial manifold chart | a deterministic closed manifold is not established; the mean head, low-frequency losses and DMD are flat | measured neighbours | LOW |
| Mori–Zwanzig / stochastic parameterisation | spread that cancels in the mean cannot change the mean's TKE (identity) | argument | DEAD as a stochastic fix |
| Lyapunov / finite-time predictability / analogues | analogue method failed (no analogue regime); diagnostic only | measured | LOW |
| 4D-Var latent encoder / EnKF | the U-Net is already an amortised assimilation map; explicit iteration needs a dynamics prior that does not exist legally beyond sim, and costs time | argument | LOW |
| convective-time canonicalisation of the *input* | genuinely different from time-dilation augmentation (coordinates of real observations, not synthetic examples) — but a fixed convective interval either needs frames before the window (high speed) or discards frames (low speed), so it is not lossless; keeping the native input plus a clock reduces to temporal conditioning/sharing (E below) | time-dilation null was about field transfer | LOW |
| enforce 2D incompressibility / streamfunction | in-plane divergence is measurably non-zero (3D wake, noise) | measured | DEAD |
| full Helmholtz features (curl + divergence + harmonic) as inputs | invertible transform the convs can already form; derivatives amplify PIV noise | argument | LOW |

### E. Current ML research
| idea | mechanism | closed / near neighbour | verdict |
|---|---|---|---|
| **temporal weight sharing in the stem** ((2+1)D: shared per-frame spatial conv → shared depthwise temporal kernel → fold to channels) | the shipped stem has independent weights per (input frame, output channel); a tied temporal operator imposes locality/stationarity in time that finite data and an untied optimiser need not discover; could generalise across shedding rates | attention/transunet nulls were spatial; **no temporal arm exists** | **SCREEN** (Codex top-5 #3) with the *untied* copy of the same computation as control, plus single-window timing |
| temporal attention / SSM (Mamba) over 20 frames | content-dependent mixing; long-context efficiency irrelevant at length 20 | no arm | LOW — only after the shared-stem result |
| MAE / JEPA / contrastive slow features | every frame is already a label; the slowest features are regime identity, i.e. the shortcut that hurts angle transfer | sequential sim pretraining −0.33 | LOW |
| MoE by inferred regime | hard conditioning; conditioning is the project's most robust negative | closed by mechanism | DEAD |
| legal test-time augmentation (input noise, sub-windows) | another smoothing ensemble; a full extra pass costs ≈ −0.114 of time, so it needs ≥ +0.174 gross | checkpoint-prediction averaging −0.03 | LOW/DEAD |
| target denoising / heteroscedastic loss | §9b; noise riders monotone worse | measured | LOW |
| curriculum / hard-regime mining / edge importance weighting | the hidden geometry is a hypothesis; balanced sampling is the cleaner version | no arm | LOW |
| distillation variants, privileged teachers | board-anomalous −0.146; joint fine-tuning −0.055 on angle | measured | DEAD for first-wave work |
| pressure beyond aux-p | null at two doses; note the old "pressure carries no information" *argument* is too strong (a 2D slice of a 3D flow does not determine p), the empirical null is the closure | measured | LOW |
| operator families / external pretrained models | closed; external weights breach the rules | measured / rules | DEAD |

### F. Scoring structure and legal inference
* The direct subscores apply 100/(1+0.5e) **after** averaging per-window relative errors (`scoring.py`), so there is no convexity arbitrage from redistributing error across windows. DEAD. (Claude's draft priced it LOW on curvature; the scorer's order of operations kills it outright.)
* Mean of ratios / denominator weighting: already exact in `losses.rel_l2` and friends. DEAD.
* End-to-end SPS-aware forecast objective (differentiate through the per-window prefactors and a frozen head): the exact-reward head read −0.069 and the head sits at 90–97 % of its reachable gap; needs +0.237 sps alone for +0.06. LOW.
* Per-window adaptive member count (early exit): bounded by the 3 → 1 time credit (~+0.10) and the single-member accuracy loss; offline oracle first. LOW.
* Cross-window pooling / adaptation: illegal. DEAD.
* More time engineering: ≤ +0.14 total, ≤ +0.11–0.13 after cugraph16t. LOW, outside the accuracy list.

## 2. The angle-only suite read, and the stopping-step conflict it exposes

The six pre-registered baseline runs (`bt_r2_angle{05,10,15}_base_s{0,1}`, shipped recipe, angle withheld
from real and sim, no Re holdout, 8000-step cosine, snapshots kept) finished 2026-09-05 ~13:30Z. Primary
comparison fixed in advance: **EMA snapshot at step 4000 vs 6000**. Direct accuracy Δ = Σ w_k (S(e_k^{4k}) −
S(e_k^{6k})), S(e) = 100/(1+0.5e), w = rel 0.474 / tke 0.106 / mvpe 0.093; the two held-out-angle halves
(val_aoa, test_aoa) averaged equally, then seeds, then angles. History reads (single models, no head):

| angle | rel_l2 4k / 5k / 6k / 8k | tke 4k / 6k | mvpe 4k / 6k / 8k | Δ(4k − 6k) s0 / s1 | Δ(5k − 6k) s0 / s1 |
|---|---|---|---|---|---|
| 5° | 0.0512 / 0.0514 / 0.0520 / 0.0532 | 0.479 / 0.489 | 0.0386 / 0.0397 / 0.0414 | +0.057 / +0.054 | +0.045 / +0.035 |
| 10° | 0.0843 / 0.0846 / 0.0853 / 0.0877 | 0.493 / 0.496 | 0.0814 / 0.0865 / 0.0929 | +0.044 / +0.064 | +0.030 / +0.052 |
| 15° | 0.1057 / 0.1060 / 0.1076 / 0.1097 | 0.489 / 0.499 | 0.0928 / 0.0987 / 0.1040 | +0.102 / +0.100 | +0.080 / +0.074 |
| **equal-angle mean** | | | | **+0.070** (6/6 positive) | **+0.053** (6/6) |

Every channel moves the same way at every angle and seed; the mvpe deterioration at 15° (0.093 → 0.104 from
4k to 8k) is the largest single term. The pre-registered promotion bar (≥ +0.06 mean, positive at all three
angles, none worse by 0.02) is **met by step 4000** on this instrument.

**The guards split it.** Same formula, archived shipped-recipe runs:

| instrument | Δ(4k − 6k) | Δ(5k − 6k) |
|---|---|---|
| mimic cell folds (`kf_tke1em8`, 4 folds × 3 seeds) | −0.006 / +0.007 / −0.029 / −0.010 → **−0.009** (1/4) | +0.006 / +0.014 / −0.004 / +0.001 → **+0.004** (3/4) |
| far-Re edge fold (`kf_farbase_far1`, 3 seeds; scores 24150/25425/26700 one to three steps past the training edge) | **−0.084 / −0.070 / −0.081** | −0.019 / −0.015 / −0.018 |

So the two Decision-Phase geometries pull the stopping step in **opposite directions**: novel angles want an
earlier stop (the network specialises to the five seen angles after step 4000), edge-Reynolds extrapolation
keeps improving to 6000 and beyond (the 27975 column in §0a's hidden-set hypothesis is exactly this case).
Step 4000 fails the far-Re guard outright; the shipped 6000 leaves ≈ +0.07 on the table at novel angles;
**step 5000 is the compromise** (+0.053 angle, +0.004 mimic, −0.017 far-Re) and the only setting that is
non-negative on the board instrument while clearing most of the angle gain. Whether a training change can
*dissolve* the conflict rather than split it — an averaging window, angle-balanced sampling, SAM, the
cross-angle meta-update — is the question the survivors in §1 exist to answer; the locked confirmation
protocol for the stopping step itself is in §3.

**Correction, 2026-09-05 evening — the far-Re guard was a vintage artefact.** The −0.078 above came from
`kf_farbase_far1` (EMA 0.999, tke 0.5, no sps term). The current-recipe far baselines queued for the
confirmation (`bt_r4_screenbase_far_{drop,keep}_s{0,1}`) read, single models, val_aoa:

| far fold (current recipe) | Δ(4k − 6k) | Δ(5k − 6k) | Δ(8k − 6k) |
|---|---|---|---|
| far_drop s0 / s1 | +0.006 / +0.027 | +0.004 / +0.017 | −0.082 / −0.091 |
| far_keep s0 / s1 | +0.010 / +0.008 | +0.028 / +0.011 | −0.071 / −0.064 |

Under the shipped recipe **early stopping is positive at the Reynolds edge as well** (mvpe 0.091 → 0.093 from
4k to 6k), and 8000 is −0.06 to −0.09 there. The "opposite pull" is therefore between novel-regime geometries
(angle and edge-Re, both earlier) and the interior mimic cells (flat: −0.009 at 4000, +0.004 at 5000), not
between angle and Reynolds. Step 4000 no longer fails a current-recipe far guard; the confirmation below reads
4000 as a declared secondary comparator alongside the locked 5000 vs 6000.

**Also read the same evening: the true 6000-step cosine is dead.** `bt_r4_cos6_*` (6k cosine read at 6000,
current recipe, seeds 0/1) against the shipped 8k cosine read at 6000: angle05 −0.046/−0.063, angle10
−0.070/−0.058, far_drop −0.088/−0.057, far_keep −0.066/−0.063 — **8/8 negative, mean −0.064**, and worse
still against 8k@5k or 8k@4k. Annealing to zero consolidates the late specialisation instead of helping; the
old `sched6k` negative was right for the right reason. Rank 1 in §3 loses its 6k-cosine arm.

Caveats. Single models, history reads, no refitted head; the size-matched controls of `POTENTIAL` §6 item 2
have not run, so "angle-specific" is still an inference; 15° was opened here for the *stopping* comparison
only (that comparison was locked before the suite ran) and stays closed for any mechanism screen.

### Confirmation read, 2026-09-05 evening — **miss; the shipped step 6000 stays**

Fresh seeds 3/4/5, **three-member ensembles** (predictions averaged, then scored), snapshots on the unchanged
8000-step path, common stride-40 windows, window → cell → equal mean; the scorer-native aggregate agrees to
0.005 everywhere (`research/potential_2026-09-05/claude/confirm_*.json`, `confirm_read.py`).

| geometry | Δ(5k − 6k) direct | Δ(4k − 6k) direct (secondary) | mvpe ratio 5k/6k |
|---|---|---|---|
| angle05 / 10 / 15 | +0.012 / +0.035 / +0.034 → **A = +0.027** | +0.020 / +0.037 / +0.039 → +0.032 | 0.99 / 0.97 / 0.98 |
| mim1 / 2 / 3 / 4 | −0.008 / −0.003 / **−0.022** / −0.014 → **M = -0.012** | −0.031 / −0.020 / −0.050 / −0.036 → -0.034 | 0.99–1.01 |
| far_drop / far_keep | +0.015 / +0.008 | +0.010 / −0.005 | 0.98 / 0.99 |

Against the locked gates: **A = +0.027 < +0.06** (primary, fails); all three angles positive (passes); mim3
−0.022 breaches the per-fold −0.02 guard by 0.002; far passes; J_novel = +0.012 ≥ 0 and
J_cells = -0.002 ≥ −0.02 pass. Heads were not fitted: at the fixed-R̄ pricing the sps prefactor
adds ≈ 0.6× the direct gain, so +0.027 direct cannot reach +0.06 net. **Verdict: a miss. Keep 8k@6000.**

What the miss teaches. The single-model angle gain (+0.053 at 5k, +0.070 at 4k) **halves at ensemble level**
(+0.027 / +0.032) — the same compression the record shows for every capacity- or schedule-type change
(`FINDINGS.md` §7, `fold_ensemble.py`) — while the interior-cell cost does not shrink. The late-specialisation
phenomenon is real (16/16 single runs, and the ensembles still read positive on all three angles and both far
folds), but it is worth ~+0.03 at novel regimes against ~−0.01 on interior cells, below the bar in every
weighting. The stopping-step axis is closed at ensemble level; the mechanism (late angle-specific fitting) is
what ranks 2–4 in §3 exist to attack directly.

### Rank-2 screen read, 2026-09-05 evening — sparse EMA-snapshot soup: **miss**

Same confirmation ensembles (seeds 3/4/5); per member an equal-weight average of the saved EMA snapshots, then
the three-member ensemble, against the step-6000 ensemble (`research/potential_2026-09-05/claude/soup_*.json`):

| soup | angle 05 / 10 / 15 → mean | mimic 1–4 → mean | far drop / keep |
|---|---|---|---|
| 3k+4k+5k+6k (pre-registered) | −0.013 / +0.045 / +0.021 → **+0.018** | −0.022 / −0.027 / −0.037 / −0.027 → **−0.028** | +0.019 / +0.007 |
| 4k+5k+6k | +0.013 / +0.033 / +0.023 → +0.023 | −0.010 / −0.004 / −0.028 / −0.022 → −0.016 | +0.017 / +0.010 |
| 5k+6k | +0.007 / +0.022 / +0.017 → +0.015 | −0.002 / +0.001 / −0.013 / −0.009 → −0.006 | +0.012 / +0.008 |

The soup behaves exactly like an earlier effective checkpoint (its centre): the same novel-regime gain and
interior-cell cost as steps 4000–5000, nothing extra. Below the bar; the dense raw-iterate average
(3001–5000, 4001–6000, 3001–6000, carried inside the `bt_rawavg_*` reruns queued tonight) is the last form
of this idea left to read.

### Rank-5 gate read, 2026-09-05 evening — spectral weight cap: **stops at its diagnostic**

`spectral_drift.py` on `bt_r2_angle05_base_s0` (67 matrix parameters, EMA snapshots): σ_max grows +149 % median
over steps 1000–4000, then **+7.0 % median (p90 +11.5 %, max +25.5 %) over 4000–6000** and +0.5 % over
6000–8000; Frobenius +9.3 %, stable rank +2.8 % over 4000–6000. The late angle deterioration coincides with
almost no spectral growth, so a weight cap has nothing to act on in the window that matters; the
pre-registered "< 10 % drift → stop" rule applies. Not built.

### Schedule / peak-lr factorial read, 2026-09-05 night — **dead under the current recipe**

`bt_fac_*` on mim1/mim2 × seeds 0/1 (history reads, single models, val_aoa) against `kf_tke1em8` at 6000:

| arm | read at | Δ direct (fold means) | rel_l2 change |
|---|---|---|---|
| 8k cosine, ⅔ peak lr | 6000 | −0.018 / −0.024 / −0.014 / −0.024 → **−0.020**, 0/2 | +0.3 to +0.7 % |
| 12k cosine, full lr | 9000 | −0.038 / −0.015 / +0.000 / −0.030 → **−0.021**, 0/2 | +0.1 to +1.0 % |
| 12k cosine, ⅔ peak lr | 9000 | −0.089 / −0.070 / −0.064 / −0.075 → **−0.075**, 0/2 | +1.2 to +1.7 % |

The archived +0.13 for the 12k horizon was a property of the old recipe (EMA 0.999, tke 0.5, no sps term); under
the shipped recipe neither a longer horizon nor a lower peak lr helps, and the two together hurt. The schedule
axis (length, terminal anneal, peak lr, stopping step) is now closed at every point measured today.

### Screen reads, 2026-09-05 night (single models paired by seed, cell-aggregated direct Δ vs the matching baseline)

**Rank 3 — SAM (ρ = 0.05, ASAM metric) — dead; loses to its own control.** Angle 5/10 × seeds 0/1, snapshot 6000:
adversarial +0.017 / +0.016 / −0.012 / +0.005 (mean **+0.007**, rel_l2 −0.5 / −0.3 / +0.2 / −0.2 %); the
random-direction control at the same radius +0.027 / +0.049 / −0.006 / +0.045 (mean **+0.029**, rel −0.3 / −1.0 /
−0.4 / −1.4 %). The SAM-specific gate (≥ 1 % rel_l2 over the random control) fails in the wrong direction. The
random perturbation itself reads a weak positive — a weight-noise regulariser worth ≤ +0.03 single-model, i.e.
nothing after the ensemble compression — and the late angle rise is unchanged in both arms (R_angle > 0).

**Rank 2 — dense raw-iterate averaging — recovers early stopping, nothing more.** Windows 3001–5000 /
4001–6000 / 3001–6000 vs the EMA at 6000 (angle 5/10/15 × seeds 0/1): +0.060 +0.050 / +0.041 +0.059 / +0.090 +0.087
→ **+0.065** for 3001–5000 (centre 4000); +0.031 +0.021 / +0.018 +0.032 / +0.042 +0.044 → +0.031 for 4001–6000
(centre 5000); +0.048 +0.035 / +0.033 +0.052 / +0.062 +0.061 → +0.048 for 3001–6000. Each average sits exactly
where the EMA checkpoint at its centre sits (step-4000 EMA read +0.070 on the same runs), so the incremental gain
over the best qualified checkpoint is ≈ 0 and the interior-cell cost will be the same. Closed. (The reruns also
replicated the baseline bit-for-bit at 4000 and 6000 — the averaging code is inert.)

**Rank 6 — variance adaptation after orthogonalisation — null.** Mimic 4 folds × seeds 0/1 vs `kf_tke1em8`:
**−0.013 at 6000 (0/4)**, +0.004 at 5000 (2/4); rel_l2 +0.1 to +1.0 %. Closed.

**Rank 4 — temporal-sharing stem — the one live result, below the bar.** Angle 5/10/15 × seeds 0/1, snapshot 6000,
vs the dense baseline (tied / untied):

| angle | seed 0 tied / untied | seed 1 tied / untied | tied rel_l2 change |
|---|---|---|---|
| 5° | **+0.042** / −0.022 | **+0.053** / +0.037 | −0.81 % / −1.14 % |
| 10° | −0.118 / −0.113 | **+0.028** / −0.105 | +0.78 % / −0.38 % |
| 15° | **+0.074** / −0.002 | **+0.095** / +0.002 | **−1.98 % / −2.01 %** |

Equal-angle mean **+0.029 (tied)** vs −0.034 (untied); tied beats untied in 5/6 runs by 1.15 % rel_l2 on average,
so the *attribution* gate (≥ 1 % over the identical untied computation) passes — the gain is temporal weight
sharing, not the extra stem depth. The *promotion* gate does not: 2/3 angles, +0.029 < +0.06, rel −0.9 % < 2 %.
The 10° failure is one seed: `stemT_angle10_s0` has tke 0.519 vs the baseline's 0.493 and mvpe 0.090 vs 0.085
(rel only +0.8 %), and both untied 10° runs show the same tke excess (0.514 / 0.521) — the new stems carry a
seed-sensitive fluctuation-amplitude instability at this fold that the dense stem does not. Inference cost:
15.7 k parameters vs 34.7 k, N = 1 eager latency 5.03 vs 5.02 ms (Codex's check, busy GPU). **Extended the same
night to the board-facing geometry:** `bt_stemT_mim{1..4}_s{0,1}` and `bt_stemT_far_{drop,keep}_s{0,1}` (12 runs);
read below when done. A promotion would additionally need the untied control there, three-member ensembles and
the tke instability understood.

### Size-matched controls, 2026-09-05 night — **the angle penalty is angle-specific, and it grows late**

`bt_szctl_a{05,10,15}_q{1,2}_s{0,1}` vs the whole-angle models on exactly the Qj cells (single models, same
seed, same step, per-cell errors → equal-cell mean; `szctl_read.py`, `szctl_*.json`):

| angle | whole-angle excess over the size-matched control, step 6000 | step 4000 |
|---|---|---|
| 5° | rel_l2 **+6.1 %** (5.9–6.3 over 4 pairs), direct −0.20 | +4.1 %, −0.14 |
| 10° | **+7.5 %** (6.7–8.2), direct −0.31 | +5.4 %, −0.25 |
| 15° | **+9.4 %** (9.1–9.9), direct −0.41 | +6.8 %, −0.30 |

Both models train on the same number of real trajectories (67/63/64); **but — Codex audit, round 6 — the
controls kept 86/82/83 simulated trajectories against the whole-angle fold's 80, including the target angle's
simulation-only cells**, so the excess below mixes same-angle real neighbours with extra target-angle simulation.
The corrected controls (`bt_szctl2_*`: the 6/2/3 simulation-only cells at the target angle added to the held-out
list, so sim = 80 and no target-angle simulation anywhere) **read identically**: whole-angle excess **+6.0 / +7.8 /
+9.5 %** at 6000 (ranges 5.5–6.4, 7.2–8.6, 9.2–9.8 over 4 pairs) and +3.9 / +5.6 / +6.9 % at 4000, direct −0.20 /
−0.32 / −0.40. The extra target-angle simulation contributed nothing, so the penalty is the absence of same-angle
*real* neighbours — angle coverage, not training size and not simulation. The excess **grows from step 4000 to 6000 in all 12 pairs**
(+4.1/5.4/6.8 % → +6.1/7.5/9.4 %), which is the late-specialisation mechanism measured directly rather than
inferred from curves. The target of ranks 3, 4 and 7 is real; SAM (above) does not reach it, the temporal stem
reaches part of it at 5° and 15°, and the cross-angle meta-update remains the untested mechanism aimed squarely
at it. (Direct deltas here are on the held-out cells only and are single-model; halve them for the ensemble
compression before pricing anything.)

**Temporal stem on the board-facing geometries (read 2026-09-05 late night).** Mimic 4 folds × seeds 0/1 vs
`kf_tke1em8` at 6000 (history, single models): +0.022 / +0.004 / +0.016 / −0.015 / +0.011 / +0.004 / +0.014 /
+0.005 → **+0.008, 4/4 folds positive** (rel_l2 −0.7 to +0.4 %); at 5000 +0.011. Far-Re vs the current-recipe far
baselines at 6000: +0.004 / +0.010 (drop), +0.000 / +0.023 (keep); at 4000 +0.009 to +0.042. So the stem is
non-inferior everywhere and weakly positive in every geometry (angle +0.029 with one seed's tke excess, mimic
+0.008, far +0.009), never at the +0.06 bar. Its status (wording corrected after Codex's round-6 audit): a weak positive on 5°, 15°, mimic and far, but
**angle 10 averages −0.045 direct and breaches the per-angle −0.02 guard**, so it is not "consistent across all
nine geometries"; the tied-over-untied rel_l2 gain is 1.15 % over all angles but only 0.8 % on the 5°/10°
discovery angles (15° supplies the threshold crossing); and it carries a fluctuation-amplitude excess at one
angle-10 seed that has to be understood before any ensemble read. (Traced in
the histories: `stemT_angle10_s0` sits at tke 0.61 → 0.52 → 0.51 from step 1000 onward against the baseline's
0.57 → 0.49 → 0.50, i.e. it is an *early* seed-dependent choice of tke operating point at that fold, not a late
blow-up; `stemU_angle10_s{0,1}` show the same, `stemT_angle10_s1` does not, and no other fold shows it.) Not
promoted; parked as the first candidate for a three-member ensemble read if the Decision-Phase package has room
for one more measured component.

**Stem ensemble diagnostic at angle 10 (Codex's requested next measurement, zero training).** Two-member
ensembles of seeds 0/1 at step 6000, `stem_ens_read.py`: dense 62.906 direct (rel 0.0834, tke 0.500), tied
62.859 (**−0.047**; rel 0.0831, tke 0.513), untied 62.784 (−0.122; tke 0.529). The tied stem's rel_l2 is not
worse at 10° — its tke is (fluctuation-energy ratio 0.730 vs 0.720, member-error correlation 0.90 vs 0.92): the
new stems settle at a slightly higher fluctuation amplitude at this fold and pay it through the tke/rel frontier.
That is the amplitude operating point, not a generalisation failure, and it says a fair comparison of stems would
need the tke weight re-selected per stem — exactly the "constants do not transfer between configurations" rule
(`FINDINGS.md` §8). Parked with that note.

### Two late-night findings (2026-09-05)

**Rank 7 — cross-angle meta-update at α = 10⁻³ relative displacement: unstable, re-queued at 10⁻⁴.** Angle 5 × seeds
0/1 (angle 10 pending at the time of reading): the *random-partition control* degrades monotonically — rel_l2 +39 %
at 4000, +47 % at 6000, tke error 0.53 → 0.69 — while the angle arm reads +15 % at 4000 recovering to +0.8 % / +0.1 %
at 6000. Both arms zero both optimiser groups (see below). The signature — smooth drift in the amplitude channel,
worse when support and query are iid — is a first-order lookahead overshoot in high-curvature directions
(g_B(θ′) ≈ g_B − αHg_A with αλ ≳ 1), not a divergence; the implementation is under Codex audit and the screen was
re-queued at α = 10⁻⁴ (`bt_meta1e4_*`, `bt_metarnd1e4_*`). No verdict on the mechanism yet.

**The shipped recipe's AdamW group trains on accumulated gradients.** `train_mixed.py`'s baseline path calls
`opt.zero_grad()` on the Muon optimiser only; `opt_rest` (AdamW over the 1-D parameters — biases, GroupNorm affines,
per-modality FiLM embeddings) is never zeroed, so its `.grad` is the running sum of every step's gradient (verified
with a three-line reproduction: 2, 4, 6…). Adam then follows the sign of the *cumulative* gradient at ~lr per step.
This has been true of every Muon run since the optimiser was introduced, including the shipped members and the
bit-identical reproduction in `REPRODUCE.md` — so it is part of the recipe, not a divergence between build and
ship. It also confounds tonight's SAM screen: both SAM arms zeroed both groups, so the random control's +0.029 is
"zero-grad fix + noise" rather than "noise". Added `--zero-rest` (off = shipped behaviour, bit-identical at the
smoke) and queued the clean screen `bt_zerorest_{mim1..4,angle05,angle10}_s{0,1}` (12 runs). Expected size: small
(0.1 % of parameters), sign unknown; the Decision-Phase retrain reproduces whatever `train_final.sh` does either
way.

### Final reads of the night (2026-09-05 → 06)

**`--zero-rest` (the gradient-clearing fix) — null on the board geometry, weak positive on angles.** Mimic 4 folds ×
seeds 0/1 vs `kf_tke1em8` at 6000: −0.021 / −0.000 / +0.015 / +0.013 / −0.010 / +0.010 / +0.008 / −0.011 → **+0.000
(1/4)**. Angle 5/10 × seeds 0/1 at 6000 vs the shipped-behaviour baselines: +0.040 / +0.011 / −0.000 / +0.029 →
**+0.020**, rel_l2 −0.04 to −1.45 %. So the accumulated AdamW-group gradient is harmless on interior cells and mildly
unhelpful at novel angles; not a candidate on its own, but every "machinery" arm tonight that cleared both groups
(SAM-random +0.029, meta-random below) inherits this ≈ +0.02.

**Rank 7 — cross-angle meta-update at α = 10⁻⁴ — stable now, and the mechanism is dead.** Angle 5/10 × seeds 0/1 at
6000 vs baseline: angle-episode arm +0.038 / +0.057 / −0.009 / +0.075 → **+0.040** (rel −0.7 / −1.45 / −0.1 / −1.55 %);
**random-partition control +0.059 / +0.069 / +0.009 / +0.051 → +0.047** (rel −1.3 / −1.4 / −0.2 / −1.1 %). The control
matches or beats the cross-angle episodes, so nothing is attributable to the angle structure — the same verdict as
SAM vs its random control. What both arms share (two half-batch gradients, a tiny lookahead, both optimiser groups
cleared) reads +0.04–0.05 single-model on angle folds, above the zero-rest-only +0.02: a weak implicit regulariser,
unpriced on the mimic geometry, not funded further.

**Where that leaves the day's ranked list.** Every mechanism screened is closed or parked: stopping step (ensemble
miss), 6k cosine, factorial, soup, raw averaging (recover early stopping), SAM (≤ random), adavar (null), spectral
cap (gate), meta-update (≤ random), zero-rest (null on board, +0.02 angles), temporal stem (weak positive, breaches
the 10° guard through a tke operating point). Codex's round-6 closing verdict stands: **keep 8k-cosine@6000, freeze
deliberately, and spend the remaining days on packaging and the retrain verification**; the only follow-ups worth a
line are the repaired size controls (`bt_szctl2`, real-only attribution of the angle penalty) and, if there is room
for one more measured component, a three-member ensemble read of the temporal stem with its tke weight
re-selected.

## 3. Merged ranking after cross-examination, and the locked stopping-step protocol

One ordering of everything that survived both catalogues. Ranks 1–5 are funded in that order; 6–7 are reserves.
Each row names the single number that kills it. Costs H100 / A100 GPU-hours; deployment cost is the change to
the packed three-member artefact.

| rank | locked experiment | the number that kills it | screen cost | deployment cost |
|---|---|---|---|---|
| **1** | **Stopping step / schedule — CONFIRMATION MISSED 2026-09-05 (see §2): ensemble A = +0.027 < +0.06, one mimic fold −0.022; 8k@6000 stays. 6k cosine dead (8/8). Only the 12k × ⅔-lr factorial remains open on this axis.** Original entry: confirm 8k-cosine@5000 (C) vs 8k@6000 (B) on fresh seeds 3/4/5 as three-member ensembles with honest heads over nine geometries (angle 5/10/15 with real+sim withheld; mimic 1–4; far-Re with sim dropped and with sim kept). Separately screen a true 6000-step cosine and the 12k-horizon × ⅔-peak-lr factorial (§4). | angle honest net < +0.06, or any mimic/far fold below −0.02 direct or net, or far pooled mvpe error ratio > 1.01 (also per Re-distance band) | 27 paths **6.75 / 16.2** (C and B share paths); 6k-cosine arm 1.5 / 3.6; factorial 5 / 12 for two folds | none: same models, same head, same latency; production stops at 5000 on the unchanged 8000 horizon |
| **2** | **Within-run averaging — sparse soup MISSED 2026-09-05 (angle +0.018, mimic −0.028; see §2); raw-iterate averaging queued.** Cheap first: an equal-weight soup of the saved EMA snapshots 3k/4k/5k/6k on the angle suite (zero training; *not* SWAD). Then, only if it beats the best qualified checkpoint, dense raw-iterate averaging over 3001–6000 carried inside reruns (saved checkpoints hold EMA weights only). | incremental angle net < +0.06 over the strongest qualified cheap checkpoint (beating the 6k endpoint alone does not count) | soup 0.1 / 0.25; raw reruns 1.5 / 3.6 | none: weights averaged offline, three members retained |
| **3** | **Scale-aware SAM — DEAD 2026-09-05 (adversarial +0.007 < random control +0.029; §2).** Was: on the angle suite: same minibatch at θ and θ+ε with ε = 0.05·D²g/‖Dg‖₂, D_i = |θ_i| + 0.01·RMS(θ) (ASAM-style), one Muon/AdamW update, momentum and EMA updated once; control = identical machinery with a seeded random direction at the same radius. Dissolution criterion: the within-arm angle late-rise R = D(4k) − D(6k) ≤ 0 with the 6k endpoint ≥ the baseline's 4k and far-Re 6k within −0.02 | < 1 % angle rel_l2 gain over the random-perturbation control (a flatter minimum alone is a negative) | 5°/10° × 2 seeds × 2 arms 4 / 9.6; far screens +4 / 9.6; missing current far baselines 1 / 2.4 | none at inference; ≈ 2× training |
| **4** | **Temporal-sharing stem — the one live result (angle +0.029, beats untied 5/6, −2 % rel at 15°; below the bar; extended to mimic/far; §2).** Spec: shared per-frame 2→8 spatial 3×3 conv, shared depthwise temporal kernel (length 3) + 8→8 pointwise, fold 20×8 → 96; rest of the U-Net and the last-frame anchor unchanged; control = the identical computation with untied per-frame weights | < 1 % angle rel_l2 gain over the untied stem (then the win is stem depth, not sharing); or > 10 % one-window latency or < +0.06 net after time | two six-run arms 3.3 / 8.0 | extra kernels in the timed region — measure on A100 at N = 1 before believing any net |
| **5** | **Spectral weight constraint under Muon — STOPPED at its gate 2026-09-05 (σ_max drift 4k→6k +7 % median; §2).** Was: conditional: first instrument singular-value drift over steps 1–6k and function sensitivity; then selective clipping vs a matched scalar-shrink control and a lower peak lr. (Kimi's "MuonClip" clips attention logits — a different mechanism; do not borrow the name.) | < +0.06 angle honest net vs the scalar-rescaling / lower-lr control; suppressing the 12k transient without moving the endpoint is not a win | diagnostic 0.1 / 0.2; two arms 3.6 / 8.6 (SVD cost must be timed first) | none (ordinary weights saved) |
| 6 | **Variance adaptation after orthogonalisation — NULL 2026-09-05 (−0.013, 0/4; §2).** (AdaMuon-inspired local ablation: V = 0.95V + 0.05 O², U = O/√V̂ rescaled to ‖O‖_F per tensor; Nesterov kept). 500-step diagnostic first: median direction change ≥ 1 %, stable rank ≥ 80 % of the plain update | M < +0.06 direct on the four mimic folds (a member-only gain is insufficient — the momentum-0.98 precedent) | 2.3 / 5.5 + diagnostic | none |
| 7 | **Cross-angle first-order meta-update — DEAD 2026-09-06 (α 10⁻³ unstable; α 10⁻⁴ +0.040 vs random control +0.047; §2).** (θ′ = θ − α g_A on training angles A; outer gradient (g_A + g_B(θ′))/2 on a withheld *training* angle B; α fixed from training-only gradients) vs balanced ERM and random-partition episodes | < 1 % angle rel_l2 gain over the better of the two controls | 7.5 / 18, plus the size-matched controls 3 / 7.2 if not already run | none at inference; ≈ 2× training |

**Why 5000 and not 4000.** A geometry-blind recommendation of 4000 is untenable: it fails the far-Re direct guard
(−0.078) and one mimic fold (−0.029). The illustrative mixture Δ(p) = 0.070p − 0.078(1−p) breaks even only at
an angle share of 53 %; Δ₅(p) = 0.053p − 0.017(1−p) breaks even at 25 %. Under strict minimax over unknown
mixtures nothing beats the shipped 6000 (its worst-case delta is zero), so **5000 is the compromise to
confirm, not the recipe to ship**; a miss keeps 6000. The alternatives are not interchangeable — 8k@4k, 8k@5k,
8k@6k, 7k@5k, 6k@6k have terminal lr 0.50/0.31/0.15/0.19/0 of peak and integrated lr doses of 3273/3676/
3900/3371/3000 peak-steps — so 7k@5k is deferred until the clean 6k-cosine result exists, and `--steps 5000`
must not be used to imitate 8k@5k (it changes the horizon).

**Caveat on the far guard.** `kf_farbase_far1` is an older vintage (EMA 0.999, tke 0.5, no sps term); its
pooled mvpe changes at 4000 are +0.14 / −0.79 / −1.40 %, so the *original* "mvpe ≤ 1 %" guard does not
literally fail there — the direct-accuracy guard does. Current-recipe far baselines are part of the launched
confirmation (both simulation policies), which is what makes the trade measurable rather than inferred.

**Locked protocol (Codex `launch_round4.py`, 27 training jobs, all dry-run validated).** C = EMA 0.998 at
`step5000.pt`, B = EMA 0.998 at `step6000.pt`, same 8000-step path; seeds 3/4/5 paired; nine fold environments
(angle05/10/15 real+sim withheld, no Re holdout; mim1–4; `far_drop` = `FAR_FOLDS[0]` with sim dropped at the
cells; `far_keep` with `REALISTIX_SIM_KEEPS_CELLS=1`); ensemble endpoint = average the three members'
predictions first, then score; common stride-40 starts, window → cell → equal angles/folds, plus the
scorer-native aggregate (a promotion must not reverse sign under it); heads: the shipped no-spread width-96
head, lr 8e-4, τ 0.05, 4000/4000, head seeds 0/1/2 paired, fitted on `partition('real')['train']` only, multiplier
1, no search; bootstrap whole cells with angle strata fixed; gates as in rank 1 plus `J_novel = 0.5A + 0.25M +
0.25F_keep ≥ 0` and `J_cells = 0.5M + 0.5F_keep ≥ −0.02` as declared stress weights (not inferred organiser
frequencies). The +0.06 is an *affected-angle* gate, not a promised whole-final-score gain.

## 4. The Muon 12k transient, re-read

All eight `kf_mu12k` histories against `kf_mu02` (same folds, seeds and old-loss vintage), fixed steps:

| mimic fold | 12k@6k − 8k@6k | **12k@8k − 8k@6k** | 12k@9k − 8k@6k |
|---|---|---|---|
| 1 / 2 / 3 / 4 | −0.112 / −0.263 / −0.180 / −0.200 | **+0.134 / +0.218 / +0.104 / +0.070** | +0.121 / +0.218 / +0.093 / +0.056 |
| mean | −0.189 | **+0.131 (4/4)** | +0.122 |

So the longer horizon paid +0.13 direct on every fold at its own deployment step — under the *old* recipe, at
50 % more updates, never confirmed, never at ensemble level — and it comes with a transient at steps 2000–4000
where the 12k cosine still runs at 85 % of peak lr (the 8k cosine is at 69 %). That is consistent with a
high-lr transient followed by useful learning, not proof of an edge-of-stability regime (the archived curves are
held-out EMA errors; plain-GD `ηλ_max ≈ 2` does not transfer to Nesterov-Muon with approximate Newton–Schulz,
clipping, two optimiser groups and EMA). The bounded test: a 2 × 2 factorial, horizon 8k/12k × peak scale
1/⅔ (both groups), current recipe, endpoints 8k@6k and 12k@9k plus matched-update reads at 6k and 8k,
training-only instrumentation every 250 steps over 1–4k (raw and EMA loss, activation RMS, per-layer σ_max,
relative update size, clip frequency, loss along the actual update). Two folds × two seeds 5 / 12 GPU-hours;
kill unless ≥ +0.06 direct on ≥ 3/4 mimic folds with none below −0.02, then the angle/far gates, then the
ensemble. Compare against GPU-seconds too: 12k@9k is 50 % more training for the Decision-Phase retrain.

## 5. Agreements, remaining disagreements, and what settles them

Agreed: no accuracy breakthrough exists on the same input; the record's closures of ScaledMomentum, wd, ns_steps,
momentum, SOAP, batch 64, warmup and (sparse EMA) averaging are reproduced from the archive; attention nulls
were spatial and no temporal-sharing arm exists; overlap consistency is mis-specified as an exact constraint;
score-map convexity offers no arbitrage; stochastic spread cannot raise the mean's TKE; angle generalisation is
the open axis and the stopping step is its cheapest lever; 5000 is the compromise to confirm.

| still in dispute | Claude | Codex | settles it |
|---|---|---|---|
| weight spectral cap | SCREEN | conditional SCREEN (rank 5) | singular-value drift over 1–6k; selective clip vs scalar shrink vs lower lr |
| SWAD as "the DG method that beats ERM" | a reliable positive | too universal; DomainBed is a selection warning | sparse soup now; raw-iterate reruns only if it beats the best checkpoint |
| LayerScale, weight standardisation | LOW | DEAD / LOW-with-diagnostic | branch-scale intervention ≥ 2 % train-only optimisation gain before any budget |
| soft MoE, legal TTA, denoised targets | DEAD | LOW (not literally measured) | no budget either way; listed so nobody re-derives them as new |
| the 12k horizon | an instability to fix | +0.13 fixed-step on 4/4 old folds: a bounded re-test | the 2 × 2 factorial in §4 |

## 6. Reproduce
Catalogues: `research/potential_2026-09-05/claude/claude_broad.md`, `codex/codex_report_round3.md` (fields,
archive scan: `round3_archive.py` → `round3_archive_index.json.gz`, `round3_checks.txt`, `round3_extra.txt`,
`round3_shapes.txt`), `codex/codex_report4.md` (ranking, protocol; `round4_audit.py/.json/.txt`,
`launch_round4.py`, `round4_confirm_manifest.json`, `round4_launch_validation.txt`). Angle-suite histories:
`$RUNS/bt_r2_angle{05,10,15}_base_s{0,1}/history.json` (+ `step*.pt` snapshots). Guard reads:
`kf_tke1em8_mim*_s*`, `kf_farbase_far1_s*`, `kf_warm250_*`, `kf_mu12k_*`, `kf_mu02_*` histories.
Launch the confirmation with `python research/potential_2026-09-05/codex/launch_round4.py --submit --type a100`
(dry-run without `--submit`; `--study cos6` / `--study screenbase` for the schedule discovery arms).

## 7. Implementation and screens queued, 2026-09-05 evening

Trainer flags added to `tools/train_mixed.py` (all inert at their defaults; 20-step smoke tests on the shipped
recipe reproduce the baseline's validation numbers bit-for-bit for `--raw-avg` and `--sam-random`, and every
arm reloads through `build_model`):

| flag | mechanism | screen queued (pq, A100, shipped recipe, snapshots kept) |
|---|---|---|
| `--sam-rho 0.05` / `--sam-random` | scale-aware SAM (ASAM metric D_i = \|θ_i\| + 0.01·RMS, ε = ρD²g/‖Dg‖), same minibatch re-evaluated at θ+ε, one Muon/AdamW step; abort on non-finite or >1.2× loss for 200 steps; random-direction control at the same radius | `bt_sam005_angle{05,10}_s{0,1}`, `bt_samrnd_angle{05,10}_s{0,1}` (8 runs, 2× cost) |
| `--raw-avg 3001-5000,4001-6000,3001-6000` | uniform average of the RAW iterates over each window, saved as `rawavg_*.pt` (the first dense raw-weight average in this repo) | `bt_rawavg_angle{05,10,15}_s{0,1}` (6 runs; also baseline replicates) |
| `--muon-adavar` (`realistix.muon.Muon(adavar=True)`) | coordinate variance adaptation *after* the Newton–Schulz orthogonalisation, rescaled to ‖O‖_F per tensor | `bt_adavar_mim{1,2,3,4}_s{0,1}` (8 runs; board-facing, vs `kf_tke1em8`) |
| `--stem temporal` / `--stem untied` (Codex, `models/unet.py`: `TemporalStem`, shared per-frame 2→8 3×3 conv → depthwise temporal k=3 + 8→8 pointwise → fold 160→96; untied = identical graph with per-timestamp weights, identical init) | temporal weight sharing in the stem; 15.7k / 20.6k parameters vs the dense stem's 34.7k; dense stem bit-identical and every old checkpoint loads strict | `bt_stemT_angle{05,10,15}_s{0,1}`, `bt_stemU_angle{05,10,15}_s{0,1}` (12 runs) |
| (no flag) 12k horizon × ⅔ peak lr factorial | `--steps 12000`, `--lr-muon 0.01333 --lr 0.0002` arms | `bt_fac_{h8_lr23,h12_lr1,h12_lr23}_mim{1,2}_s{0,1}` (12 runs; baseline = `kf_tke1em8`) |

Reads planned (all single models paired by seed against the matching baseline at fixed steps 4000/5000/6000, cell-
aggregated direct score; gates as in §3): SAM adv vs random and vs baseline; raw averages vs 4k/5k/6k EMA; adavar
vs `kf_tke1em8` on mimic; temporal vs untied vs dense; factorial at 8k@6k and 12k@9k plus matched-update reads.

Also queued the same evening: the **training-size-matched controls** of `POTENTIAL_2026-09-05.md` §6 item 2
(`launch_size_controls.py`, 12 runs `bt_szctl_a{05,10,15}_q{1,2}_s{0,1}`): for each angle its cells are split
into Q1 (Re ∈ `TEST_AOA_RE`) and Q2; control j withholds Qj plus enough seeded, stratified real cells at *other*
angles (never a whole Re column) that real and simulated training counts equal the whole-angle fold's. Read:
whole-angle model vs control on exactly Qj (`szctl_read.py`). If the whole-angle model's excess on Qj vanishes,
the "angle penalty" is a training-size effect and ranks 3 and 7 lose their target.

## 8. The bold round (2026-09-06, owner's request: one more ambitious attempt)

Target: the measured, real-coverage-specific novel-angle penalty (6.0 / 7.8 / 9.5 % rel_l2 at 5° / 10° / 15°,
mvpe hit hardest). Three mechanisms built the same night, all inert at their defaults, all smoke-tested to leave the
shipped path bit-identical, all screened on the angle-only suite (seeds 0/1, pre-registered gates: ≥ 2 % equal-angle
rel_l2 cut, ≥ 2/3 angles, none worse than −0.02 direct; mimic non-inferiority for anything that survives).

1. **CFD-guided angle morphing** (`--morph-prob`, `--morph-angles`, `--morph-extrap`; Claude). A real window's
   input *and* target are shifted on their valid cells by f·[mean_sim(Re, a_nb) − mean_sim(Re, a)] in normalised
   units — the paired simulation's mean-flow difference between the window's angle and a neighbour — producing
   pseudo-real windows whose mean flow sits at an intermediate angle (f = |a′ − a|/5). Sim means are per-cell
   time-means of the GPU-resident sim trajectories (RMS-speed normalised), 80 of them when the target angle's sim
   is withheld. A transform of released samples (permitted for training). Mean-flow difference between 5° and 10°
   at Re 13950 is 9 % of the mean field's RMS. Two geometries: **novel** (sim withheld at the target angle, the
   `bt_r2` baselines) — morphing toward the target needs the neighbour's simulation on the far side, so
   `--morph-extrap` continues the mean-flow trend from the near side instead; and **sim-kept** (`bt_basek_*` new
   baselines, target-angle sim available, the hidden-dev-set geometry). Arms: `morphx` / `morphk` (targets = the
   held-out angle itself), `morphxi` / `morphki` (deployment targets 2.5/7.5/12.5/17.5), probability 0.5 per real
   window. 24 runs + 6 baselines.
2. **Geometry as input** (`--geo-in`; Claude, `unet.py` forward): two channels computed from the window's own
   exact-zero mask — the invalid mask and a 12-dilation quantised distance-to-invalid field — appended to the stem
   (42 input channels), so the foil/shadow geometry is an explicit smooth variable instead of something inferred
   from wake position. 6 angle + 8 mimic runs.
3. **Foil-frame residual forecasting** (`--foil-frame foil|reverse`; Codex, `models/foil_frame.py`,
   `tools/check_foil_frame.py`). The foil angle is inferred from the observed body mask by fitting the NACA4418 upper
   profile (median error 0.6° at 0°, but a 3.9° bias at 20° on real masks — the weak link, stated up front); the
   window is pulled back into the foil frame (grid resample + velocity rotation about the quarter-chord), the
   shared trunk forecasts the residual there, the residual is rotated back and blended with the lab-frame forecast
   under a coverage weight; the last-frame anchor stays in the lab grid so persistence is exact at every angle and a
   zero transform is bit-identical. Two trunk evaluations per forecast: N = 1 graph latency 1.71 → 3.28 ms, so even a
   real gain must pay ~−0.1 of time. `reverse` negates the inferred angle with identical machinery as the
   attribution control (≥ 1 % rel_l2 over it required). 6 + 6 runs.

All three are structurally new to this repository: nothing in the closed table touches the *geometry* of the problem
(the mask was only ever a scorer detail and a "low" untried input channel; rotation was rejected as an augmentation,
which this is not). Codex's pre-registration for 3 is in `research/potential_2026-09-05/codex/round7_screen.json`.
Results are appended below as they read.

### Bold round, read 1 (2026-09-06): CFD-guided angle morphing — the most consistent angle result so far, below the bar

Single models, seeds 0/1, cell-aggregated direct Δ vs the matching baseline at 4000 / 5000 / **6000**:

| arm (novel geometry, sim withheld; morph by trend extrapolation) | 5° s0 / s1 | 10° s0 / s1 | 15° s0 / s1 | equal-angle mean @6k |
|---|---|---|---|---|
| `morphx` — pseudo-windows at the held-out angle, p = 0.5 | +0.043 / +0.074 | +0.012 / +0.020 | +0.065 / +0.081 | **+0.049**, 6/6 positive, rel_l2 −0.7 / −2.0 / −1.0 / −1.2 / −1.5 / −1.8 % (mean −1.35 %) |
| `morphxi` — deployment targets 2.5/7.5/12.5/17.5 | (s0 failed) / +0.018 | +0.015 / +0.042 | +0.058 / +0.043 | +0.035, 5/5 positive |

| arm (sim-kept geometry, vs `bt_basek`) | 5° | 10° | 15° | mean |
|---|---|---|---|---|
| `morphk` (target angle, real sim delta) | −0.032 / +0.018 | +0.022 / +0.019 | +0.071 / +0.089 | +0.031 |
| `morphki` (deployment targets) | −0.046 / −0.017 | +0.023 / +0.026 | −0.001 / +0.101 | +0.014 |
| (`bt_basek` itself vs the sim-withheld baseline: target-angle *simulation as plain data* is worth +0.02 to +0.09 at 6000, rel −0.5 to −2.3 % — larger here than at the Re edge) | | | | |

Reading. In the geometry that matters for a truly novel angle, the morph is positive on every run and every angle,
and — unlike anything else tried — its gain **grows through the late schedule** (15° s0: −0.006 at 4000 → +0.065 at
6000), i.e. it counteracts the late specialisation directly by keeping the mean flow at the missing angle in the
training distribution. Against the gate it is short: −1.35 % rel_l2 vs ≥ 2 %, +0.049 vs +0.06. In the sim-kept
geometry the plain simulation at the angle already supplies most of what the morph adds, and 5° turns negative.
Next reads queued: the interior-cell guard (`bt_morphxi_mim{1..4}_s{0,1}`, deployment targets with real sim deltas)
and a full-dose arm (`bt_morphx1_*`, p = 1.0). If the guard holds, the confirmation is the usual: seeds 3/4/5 as
three-member ensembles with honest heads on angle + mimic + far.

### Bold round, read 2 (2026-09-06): foil-frame dead, geometry-as-input alive

**Foil-frame residual forecasting (Codex) — dead.** vs the dense baseline at 6000: 5° −0.087 / −0.055, 10° −0.049 /
−0.048, 15° +0.027 / +0.026 (rel_l2 +3.5 / +2.2 / +1.8 / +1.9 / −1.4 / −0.4 %); the **reverse control is
indistinguishable** (−0.121 / −0.063, −0.069 / −0.018, +0.009 / +0.000), so nothing is attributable to body alignment
— the resampling cost (white-noise round-trip rel_l2 0.54 in Codex's own check) and the 3.9° mask-angle bias eat it.
Closed; the doubled inference cost would have killed it anyway.

**Geometry as input (`--geo-in`) — alive, same size as the morph.** Angle folds at 6000: 5° +0.011 / +0.052,
10° +0.003 / +0.035, **15° +0.119 / +0.059** (rel_l2 −0.5 / −1.5 / −0.5 / −0.3 / **−3.4** / −1.5 %) → equal-angle mean
**+0.047, 6/6 positive**, growing late (15° s0: +0.060 → +0.092 → +0.119 over 4k/5k/6k); mimic 4 folds × 2 seeds vs
`kf_tke1em8`: **+0.001 (2/4), neutral**. Two mask-derived channels, no inference-time cost to speak of (an invalid mask
and a twelve-dilation distance field). Below the gate alone (rel −1.3 % mean, +0.047), like the morph — but the two
mechanisms are independent (one changes the *input geometry*, the other the *training distribution of angles*) and
both are neutral on interior cells. **Combined arm queued**: `bt_geomorph_angle{05,10,15}_s{0,1}` (geo-in + morph
toward the held-out angle) and `bt_geomorphi_mim{1..4}_s{0,1}` (geo-in + deployment morph, interior guard).

### Bold round, read 3 (2026-09-06): the morph's guard and dose

* **Interior guard (deployment morph, p = 0.5, targets 2.5/7.5/12.5/17.5, mimic 4 × 2 vs `kf_tke1em8`):** −0.004 /
  +0.006 / +0.019 / −0.003 / −0.022 / −0.026 / −0.027 / −0.030 → **−0.011 (2/4)**, mim3 and mim4 below −0.02 (rel_l2
  +0.4 to +0.8 %). The pseudo-windows help at the missing angle and cost a little on interior cells — the early-
  stopping trade in miniature.
* **Full dose (p = 1.0 toward the held-out angle):** 5° −0.025 / +0.041, 10° −0.083 / −0.034, 15° −0.012 / −0.016 →
  negative; the training distribution becomes mostly synthetic hybrids. p = 0.5 is the dose, and there is no more
  to take.
* The relaunched `morphxi_angle05_s0` reads +0.011; the deployment variant's equal-angle mean is +0.031 over all six.
* Codex's audit (round 8, `codex_report8.md`): implementation and comparators correct, no leakage path (the sim mean
  table is training-partition simulation only; the held-out real cells are never read); "grows late" is *reduced
  deterioration* (morph 63.098 → 63.060 vs baseline 63.082 → 63.011 from 4k to 6k), not a cure; the honest claim is
  "CFD-guided mean-flow differences as training-time augmentation across angle of attack; held-out-angle gains
  below the promotion threshold".

**Verdict on the morph:** a reproducible +0.05 single-model gain at novel angles with a −0.01 interior cost, dose-
limited, below the gate before ensemble compression. Not promotable alone. Geometry-as-input is the cleaner of the
two (+0.047 angles, +0.001 interior); the combined arm decides whether they add.

**Two-member ensemble reads (seeds 0/1, step 6000, vs the two-member dense baseline; zero training):**

| angle | geo-in ensemble (singles) | morph ensemble (singles) |
|---|---|---|
| 5° | −0.002 (+0.032) | **+0.113** (+0.059) |
| 10° | +0.021 (+0.019) | +0.024 (+0.016) |
| 15° | +0.083 (+0.089) | +0.063 (+0.073) |
| mean | +0.034 | **+0.067** |

The morph does **not** compress at ensemble level — at 5° it doubles — which is what a training-distribution
augmentation with independent random draws per member should do: it decorrelates the members' angle errors.
Geo-in compresses at 5°. On the angle instrument the morph ensemble clears +0.06; the interior cost (−0.011
single-model, mim3/mim4 at −0.02 to −0.03) is the open question, and a three-member confirmation with honest
heads over the nine geometries (Codex's locked protocol, round 8) is the test that decides it.

### Bold round, read 4 (2026-09-06): the combination, and the locked confirmation

**Geo-in + morph (`bt_geomorph`, angle folds, vs the dense baseline at 6000):** 5° +0.061 / +0.077, 10° −0.002 /
+0.015, 15° **+0.109 / +0.105** (rel_l2 −0.7 / −1.9 / −0.1 / −0.0 / **−3.3 / −2.2 %**) → equal-angle mean **+0.061**,
rel −1.4 %; two-member ensembles +0.118 / +0.006 / +0.101 → **+0.075**. Interior guard (geo-in + deployment morph,
mimic 4 × 2): −0.009 (1/4; one run −0.045). So the combination is the morph's ensemble result with geo-in's 15°
strength; 10° remains the weak angle in every arm, and the interior cost is unchanged. The two mechanisms are not
additive beyond noise, and geo-in alone is the cleaner interior citizen.

**Where the round ends up.** Against the pre-registered single-model gate the morph and geo-in each miss (rel −1.3 to
−1.4 % vs 2 %); as two-member ensembles on the angle instrument both the morph (+0.067) and the combination
(+0.075) clear +0.06, because the morph decorrelates members. The interior cost (−0.009 to −0.011 single-model,
individual folds to −0.03) is the thing that has killed every earlier angle lever at confirmation. **Launched the
locked confirmation from Codex's round-8 protocol**: `bt_r9morph_{angle05,angle10,angle15,mim1..4,far_drop,
far_keep}_s{3,4,5}` — morph p = 0.5, deployment targets 2.5/7.5/12.5/17.5, extrapolation on, shipped recipe
otherwise — to be scored as three-member ensembles at step 6000 against the matching `bt_r4_confirm` baselines
(same seeds, same nine geometries). Gates: angle A ≥ +0.06 direct (or honest-head net), ≥ 2/3 angles positive, no
angle/mimic/far fold below −0.02, far mvpe ratio ≤ 1.01, scorer-native sign agreement. Heads only if the direct
gates pass. 27 runs, ≈ 7 H100-hours.

### Bold round, read 5 (2026-09-06): the locked morph confirmation — **miss**

`bt_r9morph_*` (morph p = 0.5, deployment targets, extrapolation on; seeds 3/4/5) vs `bt_r4_confirm_*` (same seeds,
shipped recipe), **three-member ensembles at step 6000**, window → cell → equal mean (`confirm_read.py`):

| geometry | Δ direct | rel_l2 change | tke error arm vs base | mvpe ratio |
|---|---|---|---|---|
| angle05 | -0.072 | -2.51 % | 0.509 vs 0.479 (+6.4 %) | 0.997 |
| angle10 | +0.017 | -0.95 % | 0.503 vs 0.499 (+0.9 %) | 0.960 |
| angle15 | +0.036 | -1.03 % | 0.497 vs 0.497 (-0.2 %) | 0.973 |
| mim1 | -0.003 | +0.41 % | 0.508 vs 0.507 (+0.2 %) | 0.982 |
| mim2 | -0.038 | -0.20 % | 0.472 vs 0.461 (+2.5 %) | 1.009 |
| mim3 | -0.010 | +1.01 % | 0.459 vs 0.463 (-0.7 %) | 1.011 |
| mim4 | -0.029 | +0.52 % | 0.483 vs 0.478 (+1.0 %) | 1.017 |
| far_drop | +0.034 | -0.54 % | 0.570 vs 0.570 (+0.0 %) | 0.940 |
| far_keep | +0.034 | -0.86 % | 0.565 vs 0.563 (+0.4 %) | 0.939 |

Equal-angle **A = −0.007**, mimic **M = −0.020** (mim2 −0.038, mim4 −0.029), far +0.034 / +0.034; J_novel +0.000,
J_cells +0.007. Against the locked gates: A ≥ +0.06 fails; angle 5 at −0.072 and two mimic folds below −0.02 fail
the per-fold guard; far passes. **Miss — the shipped recipe stays.**

What the miss teaches is precise: **the morph buys relative-L2 everywhere** (−2.5 / −1.0 / −1.0 % at the three
novel angles, −0.5 / −0.9 % at the Reynolds edge, ≈ 0 on interior cells) **and pays for it in fluctuation
amplitude** — tke error at 5° 0.509 vs 0.479 (+6.4 %), and +1–2 % on three of four
mimic folds. Shifting a window's mean flow without touching its fluctuations teaches the network a slightly damped
fluctuation/mean relationship, which is exactly the tke/rel frontier this repository has measured at zero five
times; the two-member seed-0/1 ensembles that read +0.067 on the angle instrument had not yet paid it. A
tke-weight re-selection under the morph is the only untested knob left on this mechanism, and it is a constant
re-selection, not a new idea.

**Where the bold round ends.** Three geometry-aimed mechanisms built and screened in one night: foil-frame
canonicalisation (dead, no alignment attribution), geometry-as-input (weak positive at angles, +0.001 interior, not
confirmed at three members), CFD-guided angle morphing (rel_l2 gain at every novel regime, confirmed as an ensemble,
paid back through tke; miss). The angle penalty is measured, real-coverage-specific and, so far, not buyable with
anything that leaves the tke/rel frontier where the shipped loss puts it.

## 9. Geometry-as-input: robustness assessment and shipping question (2026-09-06)

The owner asked whether `--geo-in` is robust enough to ship despite missing the bar. What is known, and what is
queued to find out:

**Known.** Angle folds (seeds 0/1, single models at 6000): +0.011 / +0.052, +0.003 / +0.035, +0.119 / +0.059 — all
six positive, mean +0.047, rel_l2 −0.5 to −3.4 %, gain growing over 4k → 6k on every run. Two-member ensembles:
−0.002 / +0.021 / +0.083 (mean +0.034; the 5° gain compresses). Interior mimic folds (seeds 0/1 vs `kf_tke1em8`):
+0.001, 2/4, no fold below −0.016. Mechanism is deterministic and inference-only: two channels from the window's
own exact zeros (invalid mask, 12-dilation distance field), identical convention on the hidden data (masks are
exact zeros there too). Inference cost, fp16 channels_last member at N = 1 (H100, shipped layout): eager 3.27 → 3.63
ms, **CUDA graph 1.38 → 1.47 ms** (+0.09 ms per member if computed inside each forward; computed once per call in
the packed `_pre` it is ≈ +0.1 ms per call, ≈ −0.007 final at the container's 5.3 ms). Retrain-safe: no new data,
no labels, no tuned constants; old checkpoints still load; the Decision-Phase retrain reproduces it from the flag.

**Not yet known — queued as `bt_r10geo_{9 geometries}_s{3,4,5}` (27 runs), read as three-member ensembles against
`bt_r4_confirm` with the same gates as every confirmation:** fresh-seed angle gain, far-Reynolds behaviour (never
measured), interior cost at ensemble level, per-fold guard (−0.02), scorer-native sign. Honest heads and the packed
N = 1 timing follow only if the direct read holds. The shipping bar under the owner's "ship best on hand" rule is
not +0.06 but *non-negative on the interior and far geometries with a positive angle read*; the confirmation is the
instrument for exactly that.

**Shipping path if it holds.** (1) `train_final.sh` with `--geo-in` for three all-81 members + the no-spread head;
(2) the packed `submission.py` carries its own UNet copy — add the two channels in `_pre` once per call (or the
forward-side branch, which the graph capture handles) with stem `cin` 42; (3) `check_pack` / kit scorer at chunk 1
and 64; (4) pre-registered prediction and an idle slot. A Decision-Phase package must then include the flag in the
recipe and the channel computation in the inference code.

**Morph with a re-selected tke weight** (owner's request): queued `bt_mtke13_*` / `bt_mtke16_*` (morph p = 0.5,
deployment targets, extrapolation on, `--w-tke 1.3` / `1.6`) and the control `bt_tke13_*` (weight only) on angle
5/10/15 and mimic 1–4, seeds 0/1 (42 runs). The morph's confirmation miss was a tke payment (+7 % at 5°, +1–2 %
interior) for a rel_l2 gain at every novel regime; the question is whether a higher tke weight recovers it without
giving the rel_l2 back.

### Geometry-as-input confirmation (2026-09-06): **not robust — miss, negative on angles at fresh seeds**

`bt_r10geo_*` (seeds 3/4/5) vs `bt_r4_confirm_*`, three-member ensembles at step 6000, nine geometries:

| geometry | Δ direct | rel_l2 | tke | mvpe |
|---|---|---|---|---|
| angle05 / 10 / 15 | **−0.094 / −0.030 / −0.011 → A = −0.045** | −1.6 / +1.3 / +0.5 % | **+6.8** / +0.4 / +0.3 % | +1.6 / 0.0 / −1.1 % |
| mim1 / 2 / 3 / 4 | −0.027 / +0.002 / +0.005 / +0.015 → M = −0.001 | −0.4 / +0.2 / +0.2 / +0.4 % | +2.2 / −0.4 / −0.6 / −1.3 % | |
| far_drop / far_keep | +0.025 / +0.020 | −0.5 / −0.6 % | 0.0 / +0.7 % | −3.6 / −5.1 % |

At steps 4000/5000 the angle read is still positive at 10° and 15° (+0.022 / +0.034 at 4000) and gone by 6000: the
channels *delay* the late specialisation for some seeds, they do not remove it. The seeds-0/1 result (+0.047, 6/6)
does not replicate at seeds 3/4/5 as an ensemble; 5° pays the same +7 % tke that the morph paid there. **Answer to
"can we ship it": no.** It is neutral on interior cells and +0.02 at the Reynolds edge, but the one thing it was for
— novel angles — reads −0.045 on fresh seeds, and a shipping decision under "ship best on hand" still needs a
non-negative read on the target geometry. Closed as a shipping candidate; recorded as a mechanism that is
seed-fragile.

*Operational note, 2026-09-06 ~15:00Z: the home filesystem hit its 101 GB quota (a 12 GB pip cache was the
removable part; `~/.local` 33 GB and another project 19 GB are the fixed part). The pq agents spool under `~/q`
and stalled 28 of the morph-tke jobs as "CLAIMED, NO META"; the cache was removed (12 GB free) and the jobs
requeued. `submissions/` holds 7.5 GB of packed `model.pth` files inside the repo in `$HOME`, against
`HANDOFF.md`'s "nothing big may live in `$HOME`" — worth moving to `<data-disk>` before the freeze.*

### Morph with a re-selected tke weight (2026-09-06): the payment is gone at single-model level

Angle folds, seeds 0/1, single models at 6000 vs the dense baseline (rel / tke / mvpe relative changes):

| arm | 5° s0 / s1 | 10° s0 / s1 | 15° s0 / s1 | mean | channels (typical) |
|---|---|---|---|---|---|
| morph p 0.5 + **w_tke 1.3** | +0.060 / +0.051 | +0.004 / +0.055 | +0.098 / +0.088 | **+0.059**, 6/6 | rel −0.7…−1.5 %, **tke −1.4…−3.2 %**, mvpe −0.7…−4.5 % |
| morph p 0.5 + w_tke 1.6 | +0.055 / +0.049 | +0.020 / +0.057 | +0.066 / +0.093 | +0.057 | rel −0.3…−0.6 %, tke −0.8…−3.3 %, mvpe −1.6…−6.8 % |
| w_tke 1.3 alone (control) | +0.036 / +0.083 | −0.013 / +0.038 | +0.038 / −0.010 | +0.029 | rel −1.1…+0.8 %, tke −0.2…−3.6 % |
| morph p 0.5 alone (for reference) | +0.011 / +0.018 | +0.015 / +0.042 | +0.058 / +0.043 | +0.031 | rel −0.1…−1.5 %, tke −1.2…**+1.0 %** |

Interior (mimic, history vs `kf_tke1em8` at 6000; several runs were still finishing after the queue stall):
morph + 1.3 −0.007, morph + 1.6 −0.025, 1.3 alone −0.004. So the tke weight at 1.3 does exactly what it was asked
to: the morph's fluctuation-amplitude payment (tke +1 % single, +7 % at 5° in the ensemble confirmation) turns into
a −1.4…−3.2 % tke *gain* while the rel_l2 gain is retained at two-thirds of its size, and mvpe improves as well —
every channel now moves the right way at every angle and seed. The tke weight alone is worth about half of it and
the morph alone about half; together they read +0.059, the highest single-model angle mean of any arm to date.
1.6 gives nothing more and costs the interior. **Confirmation launched: `bt_r11mtke13_*` (seeds 3/4/5, nine
geometries, the same gates as every other candidate).**

### Morph + tke 1.3 confirmation (2026-09-06 evening): **miss at the gates, and the mechanism of the miss is now exact**

`bt_r11mtke13_*` (seeds 3/4/5) vs `bt_r4_confirm_*`, three-member ensembles at 6000:

| geometry | Δ direct | rel_l2 | tke | mvpe |
|---|---|---|---|---|
| angle05 / 10 / 15 | **−0.065 / +0.040 / +0.031 → A = +0.002** | −2.4 / −0.3 / −0.4 % | **+6.1** / −0.8 / −0.9 % | −0.9 / −5.7 / −1.5 % |
| mim1 / 2 / 3 / 4 | −0.017 / +0.011 / +0.013 / −0.033 → M = −0.006 | +1.0 / +1.1 / +2.3 / +0.9 % | +0.3 / −1.8 / −3.7 / +0.9 % | |
| far_drop / far_keep | +0.028 / +0.018 | +0.5 / 0.0 % | −1.2 / 0.0 % | −4.6 / −5.0 % |

Gates: A < +0.06 fails; 5° −0.065 and mim4 −0.033 fail the per-fold guard; far passes; J_novel +0.004 and J_cells
+0.006 pass. Interior single models (seeds 0/1, after the relaunch) −0.014. **Miss.**

**The 5° tke excess is an ensemble effect, not a member effect, and it is the same in all three fresh-seed
confirmations (morph 0.509, geo-in 0.511, morph+tke 0.508 vs baseline 0.479).** Single members at 5°: baseline
tke 0.493 / 0.491 / 0.508 (mean 0.497) → **ensemble 0.479** (averaging *lowers* the baseline's tke error by 0.018);
arm 0.494 / 0.491 / 0.487 (mean 0.491, i.e. *better* singly) → **ensemble 0.508** (averaging *raises* it by 0.017).
The baseline members over-predict fluctuation amplitude at 5° and averaging damps them toward the truth; the
arms' members sit closer to the truth singly — that is where their rel_l2 gain comes from — and averaging then
damps them below it. It is the tke/rel frontier again, this time between the member level and the ensemble level,
and a member-level tke weight (1.3) cannot see it: the constant that would have to move is the *ensemble's*
amplitude at 5°, which is exactly the class of post-hoc/dose interventions the record closed. 10° and 15° do not
show it (arm ensemble tke 0.495 / 0.493 vs 0.499 / 0.497) and read +0.040 / +0.031 with mvpe −5.7 / −1.5 %.

**Closing the thread.** Three angle mechanisms, three fresh-seed ensemble confirmations, three misses with the
same fingerprint. What is established beyond the misses: (i) the novel-angle penalty is real-coverage-specific
(6–10 % rel_l2) and grows late; (ii) CFD-guided morphing and geometry channels each buy 1–2.5 % rel_l2 and several
per cent mvpe at every novel regime, robustly across seeds, and both pay for it at ensemble level through
fluctuation amplitude at the lowest angle; (iii) a member-level tke weight re-selection removes the payment for
single models but not for the ensemble. The shipped recipe stays. If this is picked up again, the experiment is
the ensemble-level one: three morph members with the tke weight re-selected *on the three-member prediction* at
5° — a re-selection, priced by the record at hundredths, not a new idea.
