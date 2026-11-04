# What the shipped model actually is — the 2026-08-28 anatomy

> **How to read this document (added 2026-08-31).** This is a point-in-time
> analysis with its own adversarial audit: **§0.5 is authoritative wherever it
> disagrees with the original text**, and five section headings carry
> REFUTED/WEAKENED/RETRACTED tags where the audit overturned them. The follow-up
> measurements it spawned (the scramble arms, the optimizer-mechanism arms, the
> time-model probe triangle, MIMO) are recorded in `docs/log/` and summarised in
> `FINDINGS.md` §3a and §9a; the durable component numbers live in
> `docs/MODEL.md`. Kept intact rather than rewritten because the audit trail —
> what was claimed, what survived re-measurement, and why — is the document's
> main value.

An anatomy of the 81.597 forecaster, written to be a starting point for improvements
rather than a record of one. Four questions, each answered with a new measurement
rather than with a re-reading of `HANDOFF.md`, plus a drawing of the thing.

Every number below is reproducible from a checkpoint already on disk. The tools are
new and self-documenting:

| tool | what it measures | cost |
| --- | --- | --- |
| `tools/norm_audit.py` | what per-window normalisation aligns and what it leaves; value distributions; radial and temporal spectra; the decimation floor | ~8 min, CPU |
| `tools/flow_anatomy.py` | input-frame attribution, occlusion, the spectral transfer matrix, receptive field, anchor/delta split, pathway ablation | ~3 min, 1 GPU |
| `tools/cond_anatomy.py` | real-vs-simulated gradient geometry along training, what the per-domain affine computes, loss curvature, Jacobian norm | ~6 min, 1 GPU |
| `tools/muon_anatomy.py` | weight singular-value spectra, per-layer step sizes under three update rules, the paired error distribution | ~15 min, 1 GPU |
| `docs/model.tex` | the architecture diagram (`tectonic docs/model.tex`) | — |

`--sim-scramble {none,time,phase,mean}` was added to `tools/train_mixed.py` for §1.4; it
is the only change to a training path, and it is inert at its default. The fold arms it
drives are the one expensive measurement here — 12 runs per arm at ~20 min on one A100 —
and they are launched the ordinary way:

```
RECIPE="--opt muon --lr-muon 0.02 --muon-momentum 0.95 --w-sps 1.0 --w-tke 1.0 --ema 0.998"
python tools/kfold.py --tag scphase --seeds 0,1,2 --fold-set mimic --folds mim1,mim2,mim3,mim4 \
    --args "$RECIPE --sim-scramble phase" --pq --pq-type a100
python tools/kfold.py --report tke1em8 sctime scphase --fold-set mimic \
    --folds mim1,mim2,mim3,mim4 --seeds 0,1,2
```

Raw outputs are under `analysis/` and
`docs/norm_audit.json`.

---

## 0. The one-paragraph summary

**Read §0.5 first.** This section and §1–§4 are the original analysis; §0.5 is the
adversarial re-measurement of it and is authoritative where they disagree.

**The headline.** The simulation axis is **small**. On the four mimic folds under the
current recipe, removing simulation entirely costs **0.053 ± 0.003** on the accuracy
channel; calibrated against the Muon change on the same folds in the same units (+0.037
→ +0.074 of final on the board), that puts the whole axis near **0.10 of final**. The
archived `simval` arms agree at +0.044. The +0.71 this repository quotes could not be
reproduced from any artefact on disk. Everything downstream — seven failed transfer
mechanisms, the conditioning design, the transport map, the coverage argument — has been
competing for about a tenth of a point.

Within that: replacing simulated windows by spectrum-matched noise *pairs* retains
**55–60%** of the value, so most of it is in the simulated **inputs** rather than in the
simulated task. And simulation is not a regulariser in the mechanical sense: it leaves
both a **sharper** weight-space minimum (5/5 seeds) and a **more responsive** learned
operator (Jacobian on the delta 1.21x, 5/5 seeds). It behaves like extra, biased data.

**The rest.** The forecaster is a scale-selective operator sitting on a persistence
anchor. About half of what it emits is driven by the input window's 20-frame mean and
almost all the rest by the last four or five frames; the oldest ten frames contribute
2.5% of the occlusion effect. It is over-amplified relative to the squared-error optimum
in every band, from 1.12x at the largest scales to 1.58x at the smallest, and correcting
that per mode would remove 6.25% of fluctuation error — about 0.02 of final, gross, and
it would cost TKE. Its receptive field saturates the domain (computed half-width 80 cells
on a 32x64 grid) while 90% of its gradient mass sits within 15–26 cells, so it is not
reach-limited. Muon leaves a flatter, higher-rank weight spectrum (sigma90/sigma10 6.76 →
3.72, update stable rank 84 against 6.6) at 5.5x the weight norm, and as an operator it
raises band coherence in 6/6 bands. And per-window normalisation aligns the one component
that never differed — mean flow, to 1.2% — while leaving fluctuation energy at 2.6x and
cross-flow at 5.7x.

---

## 0.5 Stress test: every claim, and what survived

> **Provenance warning, 2026-09-01.** `tools/stress_checks.py` — the tool this section
> *is* — shared the import-order bug of `FINDINGS.md` §10: it set
> `REALISTIX_HELDOUT_*` after `realistix.splits` had frozen them, so `--fold` was inert
> and every run scored the default AoA=20 holdout rather than the fold's mimic cells.
> The same applies to `flow_anatomy.py` (§2) and `muon_anatomy.py` (§3). The tools are
> fixed or failing loudly and the guard is in `splits.partition`, but the numbers in
> those sections were measured on a holdout other than the one they name, and for some
> folds on cells that fold's members had trained on. **Re-measure before quoting.** The
> conclusions may well survive; their provenance does not.

Written after the fact, because the first pass got several things wrong and the
corrections are more useful than a clean narrative. `tools/stress_checks.py` re-measures
each claim against the confound that could have produced it. Sections 1–4 below are the
original text with the worst overclaims marked; **this table is authoritative where they
disagree.**

| § | claim as first written | verdict | corrected statement |
| --- | --- | --- | --- |
| 1.1 | simulation's gradient *opposes* the real gradient through steps 2000–4000 — a "constraint phase" | **retracted** | the three points are −0.019, −0.031, −0.034 with a per-step SE of ~0.05. Indistinguishable from zero. What survives: cos = +0.367 at step 1000, ~0 thereafter |
| 1.1 | the simulated gradient has "half the spread" of the real one, so it is a persistent force | **not established** | variance ratio 4.22 against an F(7,7) 5% critical value of 4.99 |
| 1.1 | "99.1% of g_sim is orthogonal to g_real" | **not independent evidence** | orth frac = sqrt(1 − cos²); it is a restatement of the small cosine, not a second measurement |
| 1.2 | simulation leaves the operator **smoother** — a function-space regulariser | **refuted, sign inverted** | measured on the *delta*, plus_sim's Jacobian is **1.21x higher, 5/5 seeds**. The full-output figure (0.97, lower on 4/5) was the anchor path cancelling the delta's response |
| 1.2 | simulation leaves a *sharper* weight-space minimum | **holds** | 5/5 seeds on the filter-normalised probe, 4/5 on the Hutchinson trace |
| 1.3 | the affine's per-band gain reconstructs the sim/real spectral gap | **weakened** | the original ratio was against the mean-flow-dominated prediction, which rises with k by construction. Like-for-like: Spearman +0.77 (n = 6, p ≈ 0.07), the affine **under-corrects by 10–35%** and has the **wrong sign** in k 2–4, the band carrying most fluctuation energy |
| 1.4 | the +0.71 in the repository is wrong by 13x | **overstated** | +0.053 is a mimic-fold *accuracy-channel* delta and +0.71 is a final-score figure; they are different units. Rig-calibrated (below) the axis is ~0.10 of final. The +0.71 could not be reproduced from any artefact on disk, so its provenance needs checking — not a declaration that it is wrong |
| 1.4 | a static simulated field is 34x worse than none — "fluctuation level is the load-bearing statistic" | **retracted: loss pathology** | `tke_rel_l2` divides by the target's TKE, which for a static target is **exactly zero** → clamped to 1e-8. The arm ran with an effectively infinite weight on "emit no fluctuation" and `val_aoa._tke` is **1.000 at every step**. It measures a divide-by-zero, not physics |
| 1.4 | 55–60% of simulation's value is carried by its spectrum | **holds, narrower reading** | the arms randomise input and target independently, so they also destroy *learnability*. What is established: over half the value is in the simulated **inputs**, not in the input→output mapping |
| 2.1 | 48.8% of the delta survives with the input fluctuation removed | **flagged** | that input is constant in time and far out of distribution. The occlusion numbers (41.3% / 89.4% / 2.5%) are in-distribution and carry the claim |
| 2.1 | the model responds more to fluctuation (1.049) than to the mean (0.801) | **ambiguous** | the two perturbations carry equal *total* energy but the mean's sits in one temporal mode against the fluctuation's nineteen. The comparison depends on a normalisation choice that was not argued |
| 2.2 | over-amplified in every band, monotonically more at small scales | **holds** | the per-band estimator had a Jensen bias (energy-weighted *mean* coherence against energy-weighted *RMS* amplitude) growing with band width, i.e. with k. Redone per mode with no pooling: amp/rho = 1.12, 1.15, 1.22, 1.30, 1.42, 1.58. Same trend, slightly attenuated |
| 2.2 | the per-scale amplitude axis is worth ~0.016 of final | **holds, refined** | per-mode, an optimal rescale removes **6.25%** of fluctuation error energy → ~0.02 of final gross, and the optimal gains are 0.61–0.89, so taking it damps TKE |
| 2.3 | theoretical receptive field ~44 cells | **wrong, and the conclusion is stronger** | computed half-width along the longest path is **80 input cells** on a 32x64 grid — the field saturates the domain. Measured 90% extent is 15–26 cells |
| 2.3 | 68.3% of gradient mass upstream ⇒ advection prior learned | **mostly artefact** | the cell sat at x = 40 of 64, so a symmetric kernel already gives 62.5%. Against the area baseline the bias is **+5 to +8 points**; locally (±8 columns, both sides in frame) 0.55–0.70 against 0.50 |
| 2.4 | the deepest skip and the bottleneck are redundant with each other | **hypothesis wrong** | jointly −0.316 against −0.009 and −0.267 alone: additive, not redundant |
| 2.4 | "half the parameters sit in low-value pathways" | **overstated** | only the bottleneck's 5.32 M (23%) is ablated. `enc2` and `dec2` stay in the main path and are never tested by this measurement |
| 3.2 | Muon's 5.5x weight norm is "mostly gauge" | **refuted, better mechanism** | the GroupNorm affines did not shrink to compensate — they *grew* 1.18x. So the 28% weaker residual branch cannot be gauge. It is the rank: a higher-rank weight matrix of larger Frobenius norm has *smaller* realised gain on low-rank activations |
| 3.5 | `ScaledMomentum` shows the orthogonalisation is load-bearing | **holds, confounded** | it ran at Muon's lr and took a 3.3x larger relative step. The clean arm is `lr ~0.006`, matched on realised step. What is unconfounded: update stable rank **84.3 vs 6.6** |
| 4.2 | the cross-flow Reynolds trend is physics, not a PIV noise floor | **ambiguous** | the (high k, high omega) corner was the wrong discriminant — the interrogation window low-passes the noise. On omega alone: ratio 0.49 in the wake band (omega ≤ 3) and **0.36** in the noise band (omega ≥ 7). Noise contributes and is not the whole story; not separable with this test |
| 4.1, 4.4 | the alignment table and the spectra | **hold** | direct measurements of the data, no model involved |
| 4.5 | decimation aliasing is 0.2% | **holds** | |

**The rig calibration that §1.4 needed.** On the same four mimic folds, same seeds, same
units, the change that became submission 898763:

| change | mimic-fold accuracy-only Δ | what it paid on the board |
| --- | --- | --- |
| AdamW → Muon (`mimbase` → `mu02`) | +0.037 ± 0.007 (4/4) | **+0.074** of final |
| simulation (`real_only` → reference) | **+0.053 ± 0.003** (0/4 for its removal) | — |

So simulation is worth about 1.4x Muon *in the rig's own units*, and Muon paid +0.074.
That puts the simulation axis near **0.10 of final** — the same order as a single recent
recipe change, and well below +0.71.

> **How much weight this conversion carries (added 2026-09-01).** Less than the
> confident "~0.10 of final" repeated across six documents suggests. The step from
> +0.053 to +0.10 is a single-point ratio transfer of exactly the kind the developer
> guide forbids: it assumes simulation and Muon share an aggregate local→board
> coefficient, while this project's own record puts that coefficient at 0.52, 1.66,
> 1.74 and 3.29 over four clean paired observations, with tke changing sign. It also
> crosses conventions — +0.037 is an accuracy-only fold delta and +0.074 is a
> final-score move that includes sps and runtime. What is measured is
> **+0.053 ± 0.003 on the mimic-fold accuracy channel**; the board value of the
> simulation axis has never been measured and could plausibly be anywhere from
> ~0.03 to ~0.17 on the same spread of coefficients. The useful content of "~0.10"
> is its *order* — small, and far below +0.71 — not its value. Independently, the archived `simval` arms (width 64,
no conditioning, the original angle/Re splits) give +0.044 accuracy-only and +0.177 with
the sps proxy over five seed pairs, 5/5 positive — consistent with 0.053 and also nowhere
near 0.71.

**What the stress test cost the conclusions.** Two candidate improvements lost their
justification outright (A1, and the LayerScale idea), one headline was retracted
(`scmean`), one was inverted (§1.2), three were weakened (§1.3, §2.3, §4.2), and three
strengthened (§2.2's trend survived a real bias, §2.3's receptive-field argument got
stronger, §3.2 gained a mechanism). The single most useful output is unchanged and was
not what the analysis set out to find: **the simulation axis is small.**

---

## 1. Conditioning versus pre-training, and what the regularisation is

The standing account (`FINDINGS.md` §3): joint training on simulation is worth +0.71,
a sequential curriculum recovers 62% of that on held-out angles and *nothing* at
held-out Reynolds numbers, and seven mechanisms for using simulation as regime
information are negative. The stated explanation is that simulation is "a continuous
regulariser, not an initialisation". That is a claim about training dynamics, and
nothing had measured the dynamics.

Two things come out of doing so. §1.1–§1.3 sharpen the mechanism — and §1.4 finds that
the +0.71 is 0.053 under the current recipe on hidden-set-shaped folds, which reprices
the whole section. Read §1.4 before acting on §1.1–§1.3.

### 1.1 The simulated gradient is nearly orthogonal, and it is orthogonal on purpose

`tools/cond_anatomy.py` takes real-batch and simulated-batch gradients of the run's
own objective at the same parameters. The number that makes it readable is the real
gradient's own batch-to-batch cosine, which sets the scale for "aligned".

At the shipped solution (`cleanroom_m0`, 8 independent batch triples, batch 32):

| | value |
| --- | --- |
| cos(real, real′) | **+0.046** ± 0.221 |
| cos(real, sim) | **+0.084** ± 0.107 |
| ‖g_sim‖ / ‖g_real‖ | 1.47 |
| fraction of g_sim orthogonal to g_real | **0.991** |

Two readings, and the second is the interesting one. First, at convergence a real
minibatch gradient is essentially noise — 0.046 ± 0.221 is a cosine indistinguishable
from zero with enormous spread. Second, the *simulated* gradient has **half the
spread** of the real one (0.107 against 0.221) at 1.47x the norm. Simulation is not a
noisier version of the real signal; it is a **persistent, near-orthogonal force** of
larger magnitude than the real gradient. That is the functional definition of a
penalty term, not of extra data.

Along the training trajectory (same member's step snapshots):

| step | cos(real, real′) | cos(real, sim) | ‖g_s‖/‖g_r‖ |
| --- | --- | --- | --- |
| 1000 | +0.865 | **+0.367** | 1.50 |
| 2000 | +0.348 | **−0.019** | 3.00 |
| 3000 | +0.318 | **−0.031** | 2.37 |
| 4000 | +0.173 | **−0.034** | 1.52 |
| 5000 | +0.147 | +0.105 | 1.41 |
| 6000 | +0.025 | +0.089 | 1.42 |
| 7000 | −0.017 | +0.034 | 1.31 |
| 8000 | −0.003 | −0.004 | 1.41 |

Three phases, and they explain the pre-training result exactly:

* **Early (to ~1000 steps)** the simulated gradient is genuinely *aligned* with the
  real one, at 0.42 of the real gradient's self-consistency. Simulation is teaching
  the same thing measurement is. This is the part a sequential curriculum captures,
  and it is why pretraining is worth 62% of joint training on the *easy* axis.
* **Middle (2000–4000)** cos(real, sim) turns **negative** while the simulated
  gradient is at 2–3x the real gradient's norm. Simulation is now actively pulling
  against the real objective — it is holding the model away from the real-only
  optimum. This phase has no analogue in a curriculum: fine-tuning removes it.
* **Late** both are noise and simulation's contribution is a small orthogonal push.

So "continuous regulariser" is right, and the sharper statement is available: the
value is concentrated in a *mid-training constraint phase* that only exists while both
datasets are in the same batch. Pretraining does not fail because the prior is
forgotten — it fails because the constraint was never applied.

Per pathway at the solution, the alignment is not uniform: the decoder (cos 0.078) and
the GroupNorm parameters (0.192) take a far more coherent simulated signal than the
encoder (0.045) or the bottleneck (0.006). The simulated batch acts mostly where output
statistics are set — the same localisation the conditioning shows in §1.3.

### 1.2 The regularisation is in function space, not in weight space — REFUTED, see §0.5

The textbook signature of a regulariser is a flatter minimum. Measured on the
`simval` pair — width-64 models trained identically except for the simulated windows,
probed under the same objective with filter-normalised random directions:

| | real_only | plus_sim | |
| --- | --- | --- | --- |
| real-data loss | 0.3517 | 0.3589 | |
| loss rise at 2% relative radius | +0.0007 | **+0.0009** | plus_sim is *sharper* |
| Hutchinson tr(H) | 2.88 | **3.37** | plus_sim is *sharper* |
| ‖J<sup>T</sup>v‖ (input-output Jacobian) | 0.747 | **0.703** | plus_sim is **smoother** |

The flat-minimum story is refuted and something more specific is true: simulation
leaves the weight-space minimum *more* curved — which is what adding a constraint to
an optimisation problem does — and the input-output map **6% less sensitive**. The
regularisation acts on the function, not on the parameters.

Replicated over all five `simval` seed pairs. Read the sign consistency rather than
the means: the Hutchinson trace is a noisy estimator on a small trace and returns
−0.26 for one real_only seed.

| | plus_sim | real_only | plus_sim higher on |
| --- | --- | --- | --- |
| loss rise at 2% radius | 0.0010 | 0.0006 | **5/5** |
| Hutchinson tr(H) | 3.54 | 2.14 | **4/5** |
| ‖J<sup>T</sup>v‖ | 0.720 | 0.744 | 1/5 — i.e. **lower on 4/5**, −3.2% |
| real-data loss | 0.3583 | 0.3546 | 4/5 |

Weight-space sharpness up on 5/5 by the robust probe and 4/5 by the noisy one;
function-space sensitivity down on 4/5. The real-data loss being *higher* under
plus_sim is expected and is the point: the model is fitting a constraint as well as
the data. These are width-64 AdamW models, where the real gradient's own batch-to-batch
cosine is a healthy +0.298 rather than the shipped Muon member's +0.046; averaged over
the five, cos(real, sim) = +0.058 at 98.4% orthogonal and 1.23x the norm — the same
picture at a different operating point.

Two caveats on the size of the effect. −3.2% on a projected Jacobian norm is a small
number measured on n=5, and the probe is a single random projection per repeat rather
than a full Frobenius estimate. And these models predate four recipe changes. The sign
is consistent; the magnitude should not be quoted as a coefficient.

This matters because it names a substitute. If what simulation buys is operator
smoothness, then an explicit smoothness penalty on real data — a Jacobian-norm term,
or consistency under a small input perturbation — targets the same quantity directly,
can be *dosed* (simulation cannot: it saturates at a quarter of the release, which is
exactly what a regulariser at full strength looks like), and costs no extra forward
passes on simulated windows. That is the first mechanism-derived candidate this
analysis produces, and it is cheap to screen.

### 1.3 What the per-domain affine computes (band-profile claim WEAKENED, see §0.5)

Switching only the modality token on the same real window changes the output by
**15.2%** of its norm. Decomposed:

* predicted **fluctuation** amplitude, sim token / real token: **1.311**
* predicted **mean-flow** amplitude, sim token / real token: **0.973**
* the gain rows alone account for 14.0% of the prediction norm, the bias rows 5.1%

So the affine is almost purely a fluctuation-energy control, implemented mostly as a
gain — exactly what the design intended. The new part is *where in `k`* it acts, read
against the gap it exists to represent (`tools/norm_audit.py`, 81 paired regimes):

| band | affine's effect, as a fraction of the local prediction | measured sim/real fluctuation **amplitude** ratio |
| --- | --- | --- |
| k < 2 | 0.070 | 1.23 |
| 2–4 | 0.377 | 1.37 |
| 4–6 | 0.637 | 1.73 |
| 6–8 | 0.645 | **1.95** |
| 8–12 | 0.605 | **2.01** |
| 12–40 | 0.490 | 1.47 |

The two columns rise together, peak in the same band and fall together. **The sim/real
gap is not one number, it is a spectrum**, and the learned per-domain affine has
reconstructed that spectrum. This is the mechanism behind the `--cond norm` versus
`--cond enc` result (+0.17 against +0.03) at a level the record did not have: a
U-Net's levels *are* a coarse scale decomposition, so per-level per-domain affines can
express a `k`-dependent gain; and the sites that separate most are `ups.2.0.f2` (0.67),
`ups.2.1.f2` (0.48) and `ups.1.0.f2` (0.43) — the last two decoder levels, at full and
half resolution, which is where fine-scale output energy is set.

It also says the shipped conditioning is *coarse* for the job: three resolutions
against a ratio that varies by 1.6x across the resolved bands, with the peak sitting
between two of them.

### 1.4 How much of simulation's content does the gain need? (`mean` arm RETRACTED, see §0.5)

Every previous measurement removed or re-levelled simulation. None asked the
complementary question. `--sim-scramble` degrades the simulated half of each batch
while keeping stated statistics exact, leaving real supervision bit-identical:

* `time` — the 20 frames of each simulated window are independently permuted, in the
  input and in the target. Dynamics, transport and shedding phase are destroyed; every
  per-frame field and the window-summed spectrum are preserved exactly (verified).
* `phase` — Fourier phases randomised per frame with the Hermitian constraints
  satisfied by construction, so the amplitude spectrum survives to 1e-4 and the mean
  and energy exactly, while all spatial structure is destroyed.
* `mean` — each simulated window replaced by its own time-mean.

Five arms, four mimic folds x 3 seeds each, all against the existing `kf_tke1em8`
reference — same recipe, same folds, same seeds, same ten held-out cells per fold. The
`real_only` arm is the denominator, and measuring it is what makes the rest readable:

| arm | mim1 | mim2 | mim3 | mim4 | accuracy-only Δ | folds positive |
| --- | --- | --- | --- | --- | --- | --- |
| reference (`tke1em8`) | 74.771 | 77.031 | 75.014 | 75.506 | — | — |
| **`real_only`** (no simulation at all) | 74.723 | 76.928 | 74.950 | 75.393 | **−0.053 ± 0.003** | 0/4 |
| `time` (dynamics destroyed) | 74.767 | 76.963 | 74.927 | 75.446 | −0.024 ± 0.006 | 0/4 |
| `phase` (all spatial structure destroyed) | 74.764 | 76.954 | 74.966 | 75.460 | −0.021 ± 0.005 | 0/4 |
| `mean` (simulation reduced to its time-mean) | 71.066 | 73.232 | 70.614 | 71.665 | **−1.794 ± 0.055** | 0/4 |

**The denominator is the headline, and it is not the number this repository quotes.**
Under the current recipe, on hidden-set-shaped folds, removing simulation entirely costs
**0.053** — not 0.71. The +0.71 in `README.md` and `FINDINGS.md` §3 was measured on
`val_aoa`/`val_re` under a pre-Muon, pre-`w_sps`, pre-`w_tke 1.0` recipe, and it does not
reproduce here. Two plausible reasons and they compound: the mimic folds train on 88% of
the release against the angle folds' 71%, and a regulariser is worth less with more data;
and four recipe improvements later, some of what simulation was correcting is now
corrected by the loss and the optimizer. That is the "convergence accelerator" signature
this project has seen before (`HANDOFF.md`, ConFIG: +2.35 → +0.14 → −0.001 as training
proceeds). **The simulation axis is nearly closed by size, not by mechanism** — and that
is a much more useful thing to know than anything the scramble arms say.

Against that denominator, the scramble arms read:

| what survives in the simulated window | value delivered | as a share of simulation's 0.053 |
| --- | --- | --- |
| everything (reference) | 0.053 | 100% |
| per-frame fields, spectrum, energy — no dynamics (`time`) | 0.029 | **55%** |
| spectrum, mean, energy — no structure at all (`phase`) | 0.032 | **60%** |
| a static field (`mean`) | **−1.741** | **−3300%** |

So the honest answer to "how much of simulation's content does the gain need" is:
**a little over half of it is pure second-order statistics, and the other half does
need structure.** That is a real result and a much weaker one than the scramble deltas
looked like in isolation — the arms cost 0.02 each because the whole axis is only worth
0.05, not because 96% of a large effect survived.

The `mean` arm is the sharpest thing here and it was the throwaway control.
**A non-fluctuating simulated field is 34x worse than no simulated field at all**
(−1.794 against −0.053). With simulation at ~56% of every batch, constant-in-time
windows teach the network that the future equals the mean, and it obliges. So the
requirement on the simulated set is not "correct physics" and not even "correct spectrum"
— it is **"fluctuates at roughly the right level"**, and everything past that is worth
about 0.02.

Three consequences, all smaller than the ones this section originally claimed.

*The sim-to-real information program is closed by arithmetic.* Seven mechanisms for
making simulation informative are on the record as negative, against the puzzle that a
flow-matching transport map proves the information is there and generalises (width 0.96,
deficit 0.99, cross-regime correlations 0.98). The resolution is now quantitative: the
*entire* axis those mechanisms competed for is worth 0.053, of which at most 0.024 is
structure-dependent — i.e. at the fold rig's own resolution limit. There was never enough
there to win.

*Synthetic replacement drops in priority accordingly.* A generated field with the right
spectrum would recover ~60% of 0.053, or about 0.03. Not worth building for its own sake.
The coverage argument (spectra interpolated to regimes the release does not cover) is a
different quantity and is not bounded by this measurement — but it now has to be
justified on its own, against a base rate of 0.05 rather than 0.7.

*And read with §1.2, the regularisation account survives intact.* Note what the
`real_only` arm actually shows: it sees **2.3x more real windows per step** than the
reference (32 real against ~14 real + 18 simulated at the same step count) and still
loses. So simulation's regularisation more than compensates for cutting real throughput
by more than half — the effect is small in absolute terms but not weak in mechanism. That
also means 0.053 is a *lower bound* on simulation's value at matched real-data throughput,
and the archived +0.71 shares the same confound, so the comparison between them is fair.

### 1.5 Where the mechanism can be improved

**Superseded by §0.5 and §6.** This subsection originally proposed an explicit
smoothness penalty as a substitute for simulation, a curriculum built on the
"constraint phase", and a finer conditioning. The stress test removed the basis for the
first two — simulation leaves the operator *more* sensitive, not less, and the
constraint phase is not statistically distinguishable from zero — and §0.5's rig
calibration caps the whole axis at ~0.10 of final. The surviving candidates and their
screens are in §6; nothing on the simulation axis is in the "worth doing" tier.

---

## 2. Information flow and sensitivity

All from `tools/flow_anatomy.py` on `kf_tke1em8_mim1_s0`, scored on fold `mim1`'s
held-out cells.

### 2.1 The forecast is the window mean plus about five frames

Gradient of each output lead's *delta* with respect to each input frame, as a share of
gradient mass:

| lead | t−20…t−16 | t−15…t−11 | t−10…t−6 | t−5…t−2 | t−1 |
| --- | --- | --- | --- | --- | --- |
| t+1 | 0.060 | 0.117 | 0.215 | 0.361 | 0.248 |
| t+10 | 0.043 | 0.082 | 0.170 | 0.377 | **0.329** |
| t+20 | 0.045 | 0.089 | 0.177 | 0.369 | 0.320 |

And functionally, replacing one input frame by the window's time-mean (reference
rel-L2 0.0937):

| replaced frame | t−20 | t−15 | t−10 | t−5 | t−3 | t−2 | t−1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Δ rel-L2 | +0.0008 | +0.0009 | +0.0013 | +0.0607 | +0.0411 | +0.0587 | **+0.1471** |

The last frame alone carries **41.3%** of the total occlusion effect (0.356 summed over
all twenty), the last five carry **89.4%**, and the oldest ten carry **2.5%**. Note what this test can and cannot say: replacing a frame by the
window mean leaves the window mean intact, so it measures the *instantaneous deviation*
of that frame, not the frame's contribution to climatology. The complementary
measurement supplies the other half — feeding the model its input window with the
fluctuation removed entirely (every frame replaced by the window mean) still produces
**48.8%** of the delta's norm.

So: **roughly half the emitted correction comes from the window's 20-frame mean, and
almost all the rest from the last four or five frames.** Frames t−20…t−10 contribute
their share of that mean and essentially nothing else. Under an equal-energy
perturbation the model is if anything *more* responsive to the fluctuation component
(1.049) than to the mean component (0.801) — it is not that the network ignores
instantaneous structure, it is that at 0.4 s lead there is little of it left to use.

Consequence, and it is about capacity rather than speed: 30 of the stem's 40 input
channels are carrying content the network does not read. That is free width for a
different input — the window's own fluctuation field on its own scale (§4), or its
per-cell TKE map — at zero inference cost, since the stem is 0.035 M of 22.65 M.

### 2.2 The spectral transfer function, and the per-band operating point

Injecting band-limited noise into the input and decomposing the response into bands
gives the operator's transfer matrix. With the anchor's unit-gain path removed, so
that this is the *learned* operator:

| b_in ＼ b_out | 0–2 | 2–4 | 4–6 | 6–8 | 8–12 | 12–40 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 | **1.47** | 1.29 | 0.82 | 0.52 | 0.43 | 0.31 |
| 2–4 | 1.03 | **2.26** | 1.19 | 0.68 | 0.55 | 0.37 |
| 4–6 | 0.63 | 1.30 | **1.63** | 0.71 | 0.54 | 0.34 |
| 6–8 | 0.42 | 0.76 | 0.77 | **1.12** | 0.49 | 0.30 |
| 8–12 | 0.24 | 0.44 | 0.43 | 0.40 | **0.97** | 0.27 |
| 12–40 | 0.21 | 0.42 | 0.32 | 0.22 | 0.19 | **0.87** |

Two structural facts. The operator **amplifies** at k < 6 (in-band gain 1.5–2.3) and
is roughly neutral above — it is not a smoother. And it is heavily **scale-coupled**:
an input perturbation at k < 2 produces a response at 2–4 of gain 1.29, comparable to
its own in-band gain, so the network's bands are not independent channels. (Caveat: the
probe is temporally white, so this is the operator's response to unphysical input.)

What the model achieves against the target, on the **fluctuation** field — the one TKE
scores, and the one the coherence numbers in `spectral.py` refer to:

| band | coherence with target | predicted/target amplitude | amp / coh |
| --- | --- | --- | --- |
| 0–2 | 0.851 | 0.960 | 1.13 |
| 2–4 | 0.899 | 1.022 | 1.14 |
| 4–6 | 0.763 | 0.917 | 1.20 |
| 6–8 | 0.619 | 0.814 | 1.32 |
| 8–12 | 0.471 | 0.710 | **1.51** |
| 12–40 | 0.324 | 0.562 | **1.74** |

The last column is the useful one, because for a conditional-mean predictor the
amplitude ratio equals the coherence *identically* (E[(Y−P)P\*] = 0 forces it). So
`amp / coh` is exactly how far each band sits from the squared-error optimum, and the
shipped model is **over-amplified in every band, monotonically more so at small
scales** — which is what `w_tke` and `w_spec` are for, expressed per scale for the
first time.

It also prices the whole axis. Relative error energy in a band is 1 + a² − 2·a·ρ,
minimised at a = ρ, so the excess is (a − ρ)²:

| band | excess band error | share of fluctuation energy | weighted |
| --- | --- | --- | --- |
| 0–2 | +0.012 | 32.1% | +0.0038 |
| 2–4 | +0.015 | 39.6% | +0.0060 |
| 4–6 | +0.024 | 16.3% | +0.0039 |
| 6–8 | +0.038 | 5.7% | +0.0022 |
| 8–12 | +0.057 | 3.4% | +0.0019 |
| 12–40 | +0.057 | 2.9% | +0.0016 |
| | | **total** | **+0.019** |

That is an excess of 1.9% *of the fluctuation's own energy*. Fluctuation is 0.74% of a
window's energy and the total squared relative error is 0.0088, so the entire
over-amplification costs **1.6% of squared error**. Carried through the scorer's own
mapping — `rel_l2` 0.0937 → 0.0930, `rel_l2_score` +0.034, weight 0.4738 — that is
**+0.016 of final**, and it is the *gross* figure: moving each band's amplitude down to
its coherence is exactly what lowers TKE, so the net is smaller than 0.016 by whatever
`tke_score` gives back. **So the per-band amplitude profile is bounded at 0.016 gross
and is not a vein.** This is the fifth independent route to the tke/rel frontier's price
and the first to bound it quantitatively per scale; it agrees with the other four at
approximately zero. Do not expect accuracy from per-band conditioning (§6 B1).

### 2.3 Receptive field: upstream-biased, and using a third of what it has — WEAKENED, see §0.5

Gradient of one output cell (y = 16, x = 40) at lead t+11 over the whole input field:

* gradient centroid at x = 35.7, i.e. **4.3 cells upstream** of the cell
* **68.3%** of the gradient mass lies upstream
* 90% of the mass within **17 cells**, against a theoretical receptive field of ~44 for
  the shipped topology

The advection prior is learned rather than imposed — two thirds upstream is what an
advection-dominated wake should give — and the network is nowhere near receptive-field
limited. That is a fifth, independent nail in the long-range-mixing coffin
(attention −0.03, transunet −0.020, FNO lost, XScan null): the model is not using the
reach it already has, so giving it more cannot help. Worth recording explicitly, since
"the receptive field spans the wake" was an assumption in `unet.py`'s docstring, not a
measurement.

*Docstring correction found on the way:* `unet.py` says "three downsamplings take it to
4x8". With the shipped `--mults 1,2,4` there are **two** downsamplings and the
bottleneck is **8x16**, since `pools` is `Identity` at the last level. The theoretical
receptive field follows from the corrected depth.

### 2.4 Pathway ablation

Zeroing one internal pathway and re-scoring on the held-out cells (reference
rel_l2 = 95.37, tke = 83.20, mvpe = 95.70, sps = 44.65):

| pathway | rel_l2 | tke | mvpe | sps | mean4 |
| --- | --- | --- | --- | --- | --- |
| `skip0` (32x64, full res) | −3.153 | −9.988 | −4.289 | −14.592 | **−8.005** |
| `skip1` (16x32) | −1.586 | −6.422 | −1.718 | −8.247 | **−4.493** |
| `skip2` (8x16, deepest) | −0.011 | +0.015 | −0.009 | −0.031 | **−0.009** |
| `mid` (both bottleneck blocks bypassed) | −0.061 | −0.377 | −0.084 | −0.545 | **−0.267** |

The two shallow skips are the model. The deepest skip is **free** — removing it costs
0.009 of mean4, which is a third of the *seed* noise — and bypassing both bottleneck
blocks costs 0.267. Those two pathways together hold **5.3 M of 22.65 M parameters
(23%)** and their trained contribution is under 0.3 of mean4, while `enc2` + `dec2`
push the deep half past 50% of the parameter count.

Read this carefully, because the obvious inference has already been tried and lost.
Ablating a pathway in a trained network is not the same as training without it — the
network can reallocate — and `--mults 1,2` (drop the deepest level) read +0.062 on 7/7
angle folds and **−0.013 on the board** (891698), with `tke` falling 0.288. So this is
not a licence to delete the deep levels. What it does say, and what has never been
tested, is that every width sweep in this project scaled all three levels *together*:
the untried move is to **reallocate rather than remove** — hold the parameter count and
shift it toward the shallow levels where the function demonstrably lives (e.g. width 128
at mults 1, 1.5, 2 against the shipped width 96 at 1, 2, 4).

### 2.5 What §2 suggests

* The unused input capacity (§2.1) is the cheapest structural opportunity in the model:
  a second input stream that is *not* redundant with the window mean.
* The per-band amplitude axis is bounded at +0.016 of final gross, less net (§2.2).
  Read this before any further spectral surgery.
* Long-range mixing is closed a fifth time, now from the receptive field rather than
  from a failed experiment (§2.3).
* Half the parameters sit in pathways whose trained contribution is under 0.3 of mean4
  (§2.4). Not a licence to delete them — that already lost on the board — but a reason
  to sweep the *width profile* rather than the width, which has never been done.

---

## 3. AdamW → Muon

The consequence is on the record: +0.038 per member and +0.064 at ensemble level on the
mimic folds, board +0.074 of final while `rel_l2` went slightly **negative**, and 60% of
the local gain sitting in `R_const` — the best *constant* interval band, which is a
functional of the bulk of the error distribution rather than of its tail
(`HANDOFF.md` §9.7). What follows is the
cause, measured on the cleanest paired member sets in the repository:
`kf_mimbase_mim{1..4}_s{0,1,2}` against `kf_mu02_mim{1..4}_s{0,1,2}`, 12 members each,
differing in **nothing** but the optimizer (verified by diffing the saved args).

### 3.1 Muon leaves a flatter, higher-rank weight spectrum

Singular values of every matrix-shaped parameter on Muon's own `(out_channels,
everything_else)` reshape, pooled over 67 tensors x 12 members:

| quantity | AdamW | Muon | ratio |
| --- | --- | --- | --- |
| stable rank ‖W‖²_F / ‖W‖²₂, as a fraction of its maximum | 0.394 | **0.494** | 1.25 |
| spectral entropy, as a fraction of its maximum | 0.862 | **0.927** | 1.08 |
| σ₉₀ / σ₁₀ | 6.76 | **3.72** | 0.55 |
| ‖W‖_F | 7.57 | 41.6 | **5.50** |
| σ_max | 1.93 | 6.90 | 3.58 |

The first three rows are the update rule doing exactly what it says on the tin: the
nearest-semi-orthogonal-matrix update accumulates into a spectrum whose 90/10 ratio is
**45% smaller** and whose numerical rank is 25% higher. Per level, the effect is
*equalising* rather than uniform — AdamW leaves stable-rank fractions ranging from 0.17
(`upsample`) and 0.34 (`mid`) to 0.47 (`dec2`), and Muon brings almost everything to
~0.50:

| level | AdamW | Muon | |
| --- | --- | --- | --- |
| upsample | 0.166 | 0.312 | **+88%** |
| mid (bottleneck) | 0.342 | 0.593 | **+73%** |
| dec0 (full res) | 0.352 | 0.508 | +44% |
| enc2 | 0.360 | 0.510 | +42% |
| pool | 0.264 | 0.334 | +27% |
| enc0 | 0.439 | 0.522 | +19% |
| enc1 | 0.465 | 0.529 | +14% |
| dec1 | 0.464 | 0.498 | +7% |
| dec2 | 0.470 | 0.465 | −1% |
| stem | 0.207 | 0.179 | **−14%** |

### 3.2 The 5.5x weight norm, and the functional effect of the opposite sign ("gauge" REFUTED, see §0.5)

The norm row is the surprise, and it is not an averaging artefact: per tensor, matched by
name over four paired members, the median ‖W‖_F ratio is **5.55** (p10 2.67, p90 8.67,
min 1.18) and every level is above 2.9. Muon at lr 0.02 over an 8000-step cosine simply
grows the weights, because its update has a *fixed* RMS by construction while
`weight_decay` at 1e-4 removes a total factor of 0.984 over the whole run.

That could mean two very different things, because GroupNorm makes a conv's scale a gauge
freedom wherever a normalisation follows it — and the second convolution in each `Block`
feeds the residual sum directly, where scale is *not* gauge. So the question is settled
by measuring the actual activations. Per `Block`, on a real batch,
‖residual branch‖ / ‖skip path‖:

| | median | p10 | p90 |
| --- | --- | --- | --- |
| AdamW | 0.767 | 0.433 | 1.345 |
| Muon | **0.553** | 0.288 | 0.933 |

**Muon's blocks are 28% more identity-like, not less**, despite carrying 5.5x the weight
norm — so most of the norm growth is absorbed downstream and the functional change runs
the other way. The largest reductions are at the final decoder level, which §1.3 and §2.4
both identify as the part of the network that matters most:

| block | AdamW | Muon | ratio |
| --- | --- | --- | --- |
| `ups.2.0` (full-res decoder) | 1.978 | 0.431 | **0.22** |
| `ups.2.1` | 0.936 | 0.235 | **0.25** |
| `downs.2.0` | 1.194 | 0.429 | 0.36 |
| `mid1` | 0.424 | 0.259 | 0.61 |

### 3.3 The error distribution: every quantile improves, the low ones more

`tools/muon_anatomy.py` puts 13.4 M scored elements per fold through the same
decomposition on all four mimic folds. Fold mim1, and the other three agree to within a
third of these deltas:

| |e| / σ_global | AdamW | Muon | Δ |
| --- | --- | --- | --- |
| median | 0.0477 | 0.0458 | −0.0019 |
| mean | 0.1119 | 0.1066 | −0.0053 |
| p90 | 0.2800 | 0.2659 | −0.0140 |
| p99 | 0.9381 | 0.8916 | −0.0464 |
| rms | 0.2187 | 0.2082 | −0.0105 |
| `R_oracle` = mean exp(−\|e\|/σ) | 0.9066 | 0.9103 | **+0.0037** |
| `R_const` (best single band) | 0.5921 | 0.6000 | **+0.0079** |

Shift in the share of elements per decile of the pooled \|e\| (Muon − AdamW, in points),
mim1:

```
+0.32 +0.33 +0.28 +0.20 +0.10 +0.02 −0.04 −0.22 −0.30 −0.68
```

Elements move *out of* the top three deciles and *into* the bottom four, monotonically.
Muon is not trading the tail for the bulk — every quantile improves — but the *reward*
that `exp(-|e|/σ)` pays is collected where the mass moved, which is why `R_const`
(+0.0079) outruns the rms improvement, and why 60% of the local gain showed up there.

### 3.4 As an operator: coherence up in every band, at an unchanged operating point

Running `tools/flow_anatomy.py` on one member of each set (fold mim1; single members, so
read the signs, not the magnitudes) says where the accuracy came from:

| band | coh AdamW | coh Muon | amp/coh AdamW | amp/coh Muon |
| --- | --- | --- | --- | --- |
| 0–2 | 0.865 | **0.879** | 1.09 | 1.14 |
| 2–4 | 0.899 | **0.905** | 1.12 | 1.16 |
| 4–6 | 0.759 | **0.775** | 1.18 | 1.21 |
| 6–8 | 0.617 | **0.639** | 1.25 | 1.28 |
| 8–12 | 0.469 | **0.499** | 1.38 | 1.40 |
| 12–40 | 0.324 | **0.346** | 1.56 | 1.56 |

Coherence with the target rises in **6/6 bands** — by 1.6% at the largest scales and
6.4–6.8% at k ≥ 8 — while `amp/coh`, the position on the amplitude-versus-optimality
trade of §2.2, is essentially unchanged. So Muon did not move the operating point; it
moved the whole curve. That is a real accuracy gain rather than a reallocation, and it is
distributed across scales rather than concentrated — which is exactly the profile that
shows up as a uniformly tighter error distribution.

Two further differences, same caveat:

* the learned operator's in-band gain is uniformly **larger** under Muon (1.13 → 1.68 at
  k < 2; 0.80 → 0.91 at k ≥ 12), and the model is more sensitive to its input overall
  (occlusion sum 0.199 → 0.375);
* reliance on the deepest skip drops by an order of magnitude (ablating `skip2` costs
  0.757 of mean4 under AdamW and **0.105** under Muon) while `skip1` matters more
  (−4.75 → −6.12). Muon shifts the load off the deep lateral path.

### 3.5 What the mechanism is, and what it is not

Putting §3.1–§3.4 together, and being careful about which way each arrow points:
flatter spectra and higher stable rank (§3.1), weaker per-block residual branches
(§3.2), yet a *more* responsive end-to-end operator whose coherence is higher in every
band (§3.4). Those are consistent, and the reading they support is that **Muon
distributes the same map over more blocks and more directions**: each block does less
and passes more through, and the composition is less lossy than AdamW's small number of
high-gain, low-rank steps.

What that rules out is the tempting simpler story. It is *not* "Muon makes smaller, less
confident corrections and therefore trades the tail for the bulk" — §3.3 shows p99
improving alongside the median, and §3.4 shows the amplitude/coherence operating point
unmoved. Any account resting on damping alone is contradicted by the operator
measurement.

**So the facet to test first is the rank, not the damping** — and which of the two is
load-bearing is directly testable rather than a matter of taste, because they can be
separated. `ScaledMomentum`
in `tools/muon_anatomy.py` is Muon with the orthogonalisation removed and *nothing else
changed*: same momentum, same Nesterov, same weight decay, same learning rate, and the
update rescaled to exactly the Frobenius norm the orthogonalised update would have had.
Same magnitude, unmodified spectrum. If that arm matches Muon, per-layer step-size
normalisation is the whole story and a cheaper rule buys it; if it does not, the
orthogonalisation is load-bearing and the axis to push on is rank.

A 500-step probe from a common initialisation, same seed, same batches in the same
order, the recipe's own loss (`--steps 500`, one member's args):

| arm | loss@10 | loss@end | rel step, mean | sd/mean over layers | update stable rank |
| --- | --- | --- | --- | --- | --- |
| AdamW | 2.370 | 1.258 | 0.274 | 0.365 | 6.6 |
| Muon | 2.303 | **1.147** | 3.537 | **0.236** | **84.3** |
| `ScaledMomentum` | 3.631 | 1.932 | 11.50 | 0.389 | 1.5 |

Three things, and one of them is the answer.

**Muon does equalise per-layer step sizes**, as advertised: sd/mean over layers is 0.236
against AdamW's 0.365 and `ScaledMomentum`'s 0.389.

**But the orthogonalisation is what matters, by an order of magnitude.** The *accumulated
update's* stable rank is **84.3** for Muon against **6.6** for AdamW — a 13x difference,
against the 1.25x difference the finished *weights* show (§3.1). So the rank signature is
far stronger in the update than in the solution, which is what one would expect if the
solution is a sum of many high-rank steps that partly cancel.

**And matching Muon's magnitude without its spectrum is worse than doing nothing:**
`ScaledMomentum` ends at 1.932 against AdamW's 1.258 and Muon's 1.147.

That last row needs an honest caveat, and the caveat turns out to be the mechanism.
`ScaledMomentum` was run at Muon's own learning rate and took a **3.3x larger relative
step** (11.50 against 3.537), so part of its deficit is simply a bad learning rate rather
than the missing orthogonalisation. But *why* it takes a larger step is the point: raw
momentum here has stable rank **1.5** — it is nearly rank-one — so rescaling it to the
Frobenius norm an orthogonalised matrix would have had puts that entire norm into a
single direction. You cannot match Muon's step size without orthogonalising, because the
norm Muon's update carries is only a sensible size when it is spread across ~84
directions.

So the clean version of this arm is `ScaledMomentum` at a learning rate scaled to match
Muon's realised relative step — roughly `lr 0.006` — and that is the one sweep that
would settle the magnitude-versus-shape question properly. Until it is run, the
statement supported here is the weaker one: **the update's rank is where Muon differs
most from AdamW, and it is not reproducible by rescaling.**

**And one axis the norm growth reopens: weight decay upward.** `wd` under Muon was tested
at **0** (`kf_muwd0`) and read as "plateaued"; the sweep never went up. The 5.5x norm
growth is a reason it might matter now when it did not under AdamW — a growing ‖W‖
against a fixed-RMS update is an implicit learning-rate decay stacked on the cosine, and
it is currently unmanaged. It is also the most likely explanation on offer for the two
Muon scheduling oddities on record: the optimum sitting at the schedule's end rather
than mid-schedule, and a 12k cosine peaking mid-schedule then declining. One sweep,
`--wd 1e-3` and `3e-3` on the mimic folds.

A weaker candidate, kept for honesty about how it was arrived at: **LayerScale** — a
learned scalar per block on the residual branch, ~14 parameters, exact identity at init,
folds away at export. It addresses §3.2 directly, and §3.2 was where this section was
heading before §3.4 was measured. It now looks like the *less* promising of the two
facets, because damping alone cannot produce coherence gains in every band. Screen it
after the `ScaledMomentum` arm, not before.

---

## 4. Normalisation, value distributions, and spectra

`realistix.normalize` divides each window by one number: RMS speed over the window's
valid cells. From `tools/norm_audit.py`, 1193 real and 1377 simulated windows over the
81 paired regimes.

### 4.1 The shipped scale measures the component that never differed

The scale squared is mean-flow energy plus fluctuation energy, and:

* fluctuation's share of it is **0.7%** for real windows, 2.0% for simulated ones
* turbulence intensity is **0.086** [0.041, 0.127] real against **0.143** [0.121, 0.165]
  simulated

So `s` is 99.3% a measurement of the mean flow. Dividing by it aligns the mean flow —
and nothing else. Sim/real ratio per paired regime, by normaliser (1.00 = a simulated
window and its real partner look alike to the network; `|log|` is the mean absolute log
ratio, i.e. how much misalignment is left):

| normaliser | mean-flow energy | fluctuation energy | cross-flow energy |
| --- | --- | --- | --- |
| **total** (shipped) | 0.988 (\|log\| 0.012) | **2.598** (1.186) | **5.653** (2.036) |
| per-channel | 1.026 (0.027) | 2.706 (1.226) | **0.861 (0.362)** |
| fluctuation-split | 1.000 (0.000) | **1.000 (0.000)** | 2.009 (0.852) |
| mean-flow only | 1.000 (0.000) | 2.637 (1.198) | 5.719 (2.049) |

The shipped normaliser leaves a 2.6x fluctuation-energy misalignment and a 5.7x
cross-flow misalignment, and the learned per-domain affine of §1.3 is what covers the
difference. Two alternatives close them:

* **per-channel scaling** collapses the cross-flow misalignment from 5.7x to 0.86x
  (mean |log| 2.04 → 0.36) at the cost of one extra number per window, and does nothing
  for the fluctuation gap;
* **carrying mean flow and fluctuation on separate scales** makes the fluctuation gap
  exactly 1.0 by construction and halves the cross-flow one.

**But note what those two do, because the table understates the price.** Each one
achieves alignment by *dividing out a physical variable*: per-channel scaling divides
out the flow's anisotropy, and a fluctuation split divides out its turbulence
intensity. Both are real state variables — and they are exactly the two quantities in
which simulation and measurement differ — so normalising them away without supplying
them back discards information rather than aligning representations. §4.3 shows the
second half of the price: both alternatives produce heavy-tailed inputs.

The principled form is therefore not "swap the normaliser" but **keep the single scale
and hand the network the two dimensionless ratios** `s_fluct / s_mean` and `s_v / s_u`
as scalars, the way `log s` is already handed to it. Two numbers, a strict
generalisation of the shipped model, no change to the field's distribution.

That candidate has a serious prior against it, which has to be stated: it is a member
of the family `--cond norm_re` belongs to, and *"never condition on a parameter you
must extrapolate over"* is this project's most robustly established negative result
(`FINDINGS.md` §4, monotone over four designs: none > window-inferred > explicit
labels; `norm_re` measured −0.092 on `val_re`). §4.2 shows that `s_v / s_u` carries a
strong Reynolds trend, so feeding it in *is* handing the network a Reynolds coordinate
by another name. Anyone screening this should expect it to lose, and should screen it
on `val_re`-shaped holdouts first, where it will lose fastest if it is going to.

**And §1.4 reframes this subsection twice, in opposite directions.** The reason to align
the domains has always been argued representationally — shared convolutions should see
comparable inputs. §1.4 says that over half of what simulation delivers survives having
all of its structure destroyed, so a large part of the payload really is the second
moments measured above: the misalignment is not a nuisance beside the signal, it is
partly *of* the signal. The `mean` arm sharpens that to a specific requirement — a
simulated field that does not fluctuate is 34x worse than no field at all — so
fluctuation *level* is the load-bearing statistic, which is exactly the column the
shipped normaliser leaves at 2.6x.

Then the same section takes it back: the whole axis is worth **0.053**, so the most a
perfect alignment of simulated levels can buy is a fraction of that. Both readings are
true and the second dominates the decision. Align the domains if it is nearly free
(it is: a scalar per window); do not build a transport map for it.

### 4.2 The normalised field is Re-invariant in level and not in shape

Per-window normalisation removes the velocity scale, which is what tracks Reynolds
number (corr(Re, s) = +0.99). It does not follow that what remains is Re-invariant.
Correlations with Re over real windows, and the within-angle average that removes the
grid's AoA confound:

| quantity | corr(Re, ·) | within-AoA | low-Re → high-Re median |
| --- | --- | --- | --- |
| turbulence intensity | +0.111 | **+0.051** | 0.083 → 0.091 (1.10x) |
| fluctuation / mean energy | +0.189 | **+0.060** | 0.0069 → 0.0084 (1.22x) |
| cross-flow share of fluctuation | −0.690 | **−0.725** | 0.412 → 0.200 (**0.49x**) |
| the scale itself | +0.992 | +0.998 | 0.111 → 0.252 (2.27x) |

The normalisation delivers what it promised for the *level* of fluctuation (within-AoA
correlation 0.05) and fails completely for its **anisotropy**: the cross-flow share of
fluctuation energy halves from the low-Reynolds half of the grid to the high one, at
within-angle correlation −0.73.

The obvious alternative explanation is the PIV noise floor — velocities are 2.3x
smaller at the low-Re end, so the same absolute noise is a larger and isotropic share
there. It is testable, because measurement noise is broadband and temporally
uncorrelated while wake fluctuation is not:

| (k, ω) region | low-Re | high-Re | ratio |
| --- | --- | --- | --- |
| everything | 0.429 | 0.221 | 0.51 |
| **k < 6 and ω ≤ 3** (where the wake lives) | 0.522 | 0.244 | **0.47** |
| k ≥ 8 and ω ≥ 6 (where noise lives) | 0.222 | 0.101 | 0.45 |

The trend is exactly as strong in the energetic, slow, large-scale band as in the noise
corner. **It is not a noise floor.** The wake's fluctuation really is about twice as
cross-flow-dominated at the low-Reynolds end of this grid, and the shipped normaliser
passes that shift through untouched — on the one axis the private test set is
guaranteed to extrapolate along.

How much of that a per-channel scale would remove: under the shipped single scale the
cross-flow share runs 0.412 → 0.200 across the Reynolds range (a factor of 2.06);
under per-channel scaling it runs 0.987 → 0.981, a factor of 1.01, and the within-AoA
correlation falls from −0.725 to −0.406. So per-channel scaling removes the shift
almost entirely — by removing the quantity. That is the trade §4.1 describes, stated
in numbers: the shift is a *normalisation choice*, not something the network is forced
to learn, and the choice is between passing a real Reynolds-dependent shift through to
the convolutions or discarding a real variable.

### 4.3 Value distributions

What the first convolution sees, over valid cells, u and v pooled:

| normaliser | | sd | skew | kurtosis | p01 | p99 | max abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| total (shipped) | real | 0.541 | +0.37 | 1.34 | −0.23 | +1.35 | 1.7 |
| | sim | 0.557 | +0.23 | 1.53 | −0.50 | +1.32 | 1.7 |
| per-channel | real | 0.900 | −0.66 | 5.73 | −2.28 | +2.35 | 8.7 |
| | sim | 0.898 | −0.85 | 4.18 | −2.24 | +2.18 | 6.6 |
| fluctuation-split | real | 0.669 | −0.28 | 10.52 | −1.82 | +1.75 | 12.4 |
| | sim | 0.672 | −0.15 | 8.89 | −1.85 | +1.82 | 8.6 |

The shipped scheme is the best-behaved of the four by a wide margin: sd 0.54, kurtosis
1.34, and a bounded range (|x| ≤ 1.7 — the field cannot exceed 1.7x its own RMS speed).
The two better-aligned alternatives buy that alignment with heavy tails: per-channel
reaches |x| = 8.7 because s_v is small wherever cross-flow is weak, and the
fluctuation split reaches 12.4 at kurtosis 10.5 for the same reason.

**That is the real cost of the §4.1 fixes and it has to be paid explicitly** — a
per-channel or fluctuation-normalised input needs a floor on the divisor (e.g.
`s_v ← max(s_v, 0.2·s_u)`), or the network will spend capacity on outliers created by
the normaliser. It is also a candidate explanation for why level-correcting simulation
read as a null: correcting levels without controlling the tails trades one problem for
another.

Marginal distributions are otherwise remarkably well matched between the domains under
the shipped scheme (sd 0.541 vs 0.557, kurtosis 1.34 vs 1.53). The domain gap is not a
marginal-distribution gap; it is a second-moment-in-time gap, which is why it survives
a normaliser that matches marginals.

### 4.4 Spectra

The full-field radial spectrum is useless for this question — k < 2 holds **99.5%** of
a velocity window's energy, because the mean flow is the flow. On the fluctuation about
each window's own mean, normalised by the shipped scale:

| band | real | sim | energy ratio | amplitude | real share |
| --- | --- | --- | --- | --- | --- |
| 0–2 | 0.128 | 0.194 | 1.52 | 1.23 | 32.1% |
| 2–4 | 0.157 | 0.296 | 1.88 | 1.37 | 39.6% |
| 4–6 | 0.065 | 0.194 | 2.99 | 1.73 | 16.3% |
| 6–8 | 0.023 | 0.086 | 3.79 | **1.95** | 5.7% |
| 8–12 | 0.013 | 0.054 | 4.06 | **2.01** | 3.4% |
| 12–40 | 0.011 | 0.024 | 2.15 | 1.47 | 2.9% |

This is the table §1.3 reads against, and it is worth stating on its own: the
famous "median 3.8x energy" sim/real gap is a **band-average of a ratio that runs from
1.5x to 4.1x**. Any single-number correction — a scalar gain, a global affine, a moment
match — is fitting the average of a curve. 72% of real fluctuation energy is below
k = 4 where the gap is only 1.5–1.9x.

Temporally, per 20-frame window (ω = 0 is the mean, 99.6% of energy):

| ω band | real | sim | ratio |
| --- | --- | --- | --- |
| 1–3 (shedding/meandering) | 0.0333 | 0.0898 | **2.70** |
| 3–6 | 0.0038 | 0.0083 | 2.16 |
| 6–11 | 0.0015 | 0.0037 | 2.50 |

The gap is flatter in ω than in k, and largest in the ω ∈ [1,3] band — the same band
`losses.lowfreq_rel` was built for and which carries 56% of held-out error.

### 4.5 Decimation is not a problem

The release is 64x128 and both training and scoring take every second point with no
prefilter, which aliases whatever sat above the 32x64 Nyquist into the resolved band.
Measured on the raw frames: **0.2%** of energy [0.2%, 0.2%]. Negligible. The PIV
interrogation window has already low-pass-filtered the field well below its own grid
Nyquist, so the decimation is effectively free and there is no aliasing floor on
high-k predictability. Closing this before it became an idea.

---

## 5. The diagram

`docs/model.tex` → `docs/model.pdf`, `docs/model.png`. Three panels: the scored call,
one member's internals with grid sizes and channel widths, and the training recipe.
Every number is read off the code; the header comment says how to regenerate the
parameter counts. Builds with `tectonic docs/model.tex` (tikz only, no other packages).



---

## 6. Candidate improvements, ranked

Rewritten after §0.5. Ranked by (strength of the surviving mechanism) x (cheapness of the
screen). Nothing here is a recommendation to ship; each line names the screen it should
die on.

**The two ceilings that bound most of the list:**

* the **simulation axis** is ~**0.10 of final** by rig calibration (§0.5), not +0.71.
  Everything about simulation, conditioning, domain alignment and regime coverage
  competes inside it;
* the **per-scale amplitude axis** is ~**0.02 of final gross**, less net (§2.2 as
  corrected).

**Not bounded by either:** the optimizer, and the width profile.

### A. Worth doing

**A1. Separate Muon's two mechanisms, then push on the rank.**
The strongest surviving result in the document. The accumulated update's stable rank is
**84.3 under Muon against 6.6 under AdamW** — a 13x difference, against only 1.25x in the
finished weights (§3.1, §3.5) — and §3.2 now has a mechanism for the functional
consequence: a higher-rank weight matrix of larger Frobenius norm has *smaller* realised
gain on low-rank activations, which is why the residual branches measure 28% weaker at
5.5x the norm while the end-to-end operator is more responsive and more coherent in 6/6
bands (§3.4). Muon paid +0.074 on the board, so this axis has a demonstrated size.
Three knobs, none ever touched:
* `ns_steps`, the Newton-Schulz iteration count, fixed at 5 since the module was written.
  It sets how flat the update's spectrum is — the exact quantity above.
* extending Muon to the 1-D parameters (biases, GroupNorm scales, per-domain affines)
  currently parked on AdamW.
* the clean `ScaledMomentum` control at `lr ~0.006`, matched on realised relative step,
  which is what makes the rank claim airtight rather than merely well-supported.
Screen: mimic folds, three seeds, accuracy channel *and* interval channel — §3.4 says
Muon's gain is band coherence, which is an accuracy quantity, while §3.3 says the reward
is collected in the bulk.

**A2. Weight decay upward under Muon.**
Muon grows the median weight norm **5.5x** over the run while `--wd 1e-4` removes a total
factor of 0.984 (§3.2). A growing ‖W‖ against a fixed-RMS update is an unmanaged implicit
learning-rate decay stacked on the cosine, and `wd` under Muon was only ever swept
*downward* (`kf_muwd0`, wd = 0). It is also the most likely explanation on offer for the
two Muon scheduling oddities on record: the optimum sitting at the schedule's end, and a
12k cosine peaking mid-schedule then declining.
Screen: `--wd 1e-3` and `3e-3`, mimic folds, three seeds. Then re-read the `--fixed-step`
argmax off the fold histories — this changes late-schedule dynamics, which is the
situation that has moved that constant four times.

**A3. Sweep the width *profile*, not the width.**
§2.4: the bottleneck's two blocks are **5.32 M parameters (23% of the model)** for
**0.267** of mean4, and the deepest lateral skip is free (−0.009); jointly −0.316. Every
width sweep in this project (64 → 96 → 128 → 160 → 192) scaled all three levels together,
so "the same parameter budget, distributed differently" has never been tested — e.g.
width 128 at mults (1, 1.5, 2) against width 96 at (1, 2, 4).
Caveat that must be respected: §0.5 notes the ablation never removes `enc2`/`dec2`, only
the bottleneck blocks and the deep skip, so this is not evidence that the deep *level* is
idle. And it is a capacity change, which the fold rig systematically over-predicts —
`--mults 1,2` read +0.062 on 7/7 angle folds and **−0.013** on the board. Screen on mimic
folds *and* `tools/honest_pipeline.sh`, never on angle folds, and watch `tke`.

### B. Cheap, with the ceiling known

**B1. Per-mode amplitude rescale of the predicted fluctuation, for the interval channel.**
§2.2 as corrected: an optimal per-mode gain removes 6.25% of fluctuation error energy,
worth ~0.02 of final on `rel_l2` gross, and the optimal gains are 0.61–0.89 so it damps
TKE. As an accuracy play it is priced at approximately zero — the fifth independent route
to that conclusion. But `R_const`, the bulk of the error distribution, is a different
functional and is *not* bounded by that arithmetic, and it is 60% of what Muon bought.
`tools/spread_info.py` answers whether per-mode damping moves it, in four minutes on
existing members and with no training.

**B2. Use the free input capacity.** §2.1: the oldest ten input frames contribute 2.5% of
the occlusion effect, and the stem is 0.035 M of 22.65 M, so extra input channels are
free at inference. Candidates: the window's per-cell TKE map, or its time-mean supplied
explicitly. Tempered by the same section — the network already recovers most of what it
needs from the mean and the last few frames, so the marginal information is small.

### C. Cheap, and only worth it if something else pays for the trip

**C1. Per-channel or fluctuation-split normalisation.** §4.1: the shipped scale is 99.3%
a mean-flow measurement and leaves fluctuation energy misaligned 2.6x and cross-flow 5.7x
between the domains. Per-channel scaling collapses cross-flow to 0.86x. But it competes
inside the 0.10 ceiling, both alternatives produce heavy-tailed inputs (§4.3: max |x| 8.7
and 12.4 against the shipped 1.7, so a floor on the divisor is mandatory), and the
principled variant — hand the network `s_fluct/s_mean` and `s_v/s_u` as scalars — is in
the family "never condition on a parameter you must extrapolate over", which has failed
four times in a monotone ordering. Screen on `val_re`-shaped folds, where it will lose
fastest.

### Closed — do not spend on these

* **An explicit smoothness / Jacobian penalty as a substitute for simulation.** This was
  A1 in the first draft and §0.5 removed its entire basis: simulation leaves the operator
  **more** sensitive, not less (Jacobian on the delta 1.21x, 5/5 seeds), and a *sharper* minimum. There
  is no smoothing effect to substitute for.
* **LayerScale on the residual branches.** Proposed from §3.2's damping measurement, and
  §3.4/§0.5 show damping is not the mechanism — the end-to-end operator is more
  responsive, not less.
* **Making simulation informative.** Closed by arithmetic, not by an eighth failed
  mechanism: the entire axis is ~0.10 of final and at most 45% of it depends on
  simulation having any structure at all.
* **Synthetic spectrally-matched fields.** As a *replacement* they recover ~60% of a 0.10
  axis. The coverage argument is not bounded by this measurement, but it now has to
  justify itself against a base rate of 0.10 rather than 0.7, and coverage arguments have
  lost seven times here.
* **Per-band levelling of the simulated set.** Same ceiling, plus it writes a transformed
  training set that carries an unresolved rules question, plus it is adjacent to the
  translated-sim null.
* **Long-range mixing, a fifth time.** The computed receptive field *saturates the
  domain* (half-width 80 on a 32x64 grid) while 90% of the gradient mass sits within
  15–26 cells (§2.3 as corrected). The model is not reach-limited, so more reach cannot
  help — and this closes the family from the architecture rather than from a fifth failed
  experiment.
* **Prefiltering before decimation.** 0.2% of energy above the 32x64 Nyquist (§4.5).
* **Deleting the deep levels.** Already refuted on the board (891698).

### 7. Follow-through: the arms ran, and five of six threads closed

Code added for these arms, all inert at their defaults: `--muon-ns-steps`,
`--muon-1d`, `--opt nsgd` (`realistix.muon.ScaledMomentum`), fractional `--mults`,
and `tools/mode_gain.py`. `tools/check_recipe.py` passes and correctly reports the new
flags as postdating the shipped run.

Seven arms, four mimic folds x two seeds, paired against `kf_tke1em8`:

| tag | arm | accuracy-only Δ | folds + | verdict |
| --- | --- | --- | --- | --- |
| `wd1e3` | `--wd 1e-3` | +0.005 ± 0.001 | 4/4 | real, and **10x below the 0.05 bar** |
| `wd3e3` | `--wd 3e-3` | +0.003 ± 0.002 | 3/4 | noise |
| `ns2` | `--muon-ns-steps 2` | −0.010 ± 0.004 | 1/4 | small loss |
| `ns10` | `--muon-ns-steps 10` | −0.001 ± 0.005 | 1/4 | nothing |
| `nsgd6` | `--opt nsgd --lr-muon 0.006` | **−0.102 ± 0.011** | 0/4 | the control, and it worked |
| `mu1d` | `--muon-1d` | −0.018 ± 0.002 | 0/4 | consistent loss |
| `wprofF` | `--width 128 --mults 1,1.25,2` | **+0.011 ± 0.002** (3 seeds) | **4/4** | real and consistent; **+0.029 (4/4) on the sps proxy** |

Per subscore, fold-averaged (reference: rel_l2 96.424, tke 81.309, mvpe 97.021,
sps 48.709):

| arm | rel_l2 | tke | mvpe | sps |
| --- | --- | --- | --- | --- |
| `wd1e3` | +0.003 | +0.029 | +0.012 | +0.024 |
| `ns2` | −0.008 | −0.057 | −0.006 | −0.116 |
| `ns10` | +0.001 | −0.014 | +0.004 | −0.004 |
| `nsgd6` | −0.093 | **−0.479** | −0.076 | **−0.719** |
| `mu1d` | −0.017 | −0.074 | −0.026 | −0.129 |
| `wprofF` | **+0.012 (4/4)** | **+0.006 (2/4)** | **+0.013 (4/4)** | **+0.066 (4/4)** |

**The control is the result.** `nsgd6` — Muon's per-layer step *size* with the
orthogonalisation removed, at the learning rate matched on realised relative step —
costs **0.102** against the Muon reference. Since AdamW → Muon is +0.037 on these same
folds, `ScaledMomentum` lands about **0.065 below plain AdamW**. So matching Muon's
step magnitude without its spectrum is worse than not trying: **the orthogonalisation
is load-bearing and per-layer step-size normalisation is not the mechanism.** The
stress test's tentative reading is now confirmed with a properly tuned control. (One
lr, not a sweep — but the margin is wide enough that a different lr is unlikely to
change the sign.)

**And the mechanism being identified does not make it actionable.** Both knobs that
follow from it are null or negative: `ns_steps` at 2 loses 0.010 and at 10 gains
nothing, so the value 5 chosen when the module was written is already at its optimum;
and putting the 0/1-D parameters on Muon costs 0.018 on 0/4 folds, so the original
split (matrices on Muon, everything else on AdamW) was right. **The rank axis is
identified and closed.**

**Weight decay: hypothesis answered, negatively.** The 5.5x norm growth is real but it
is not costing anything measurable — `--wd 1e-3` buys +0.005, which is consistent
(4/4, 5 SE) and an order of magnitude below what a submission needs. Closed.

**`tools/mode_gain.py` closes the per-mode amplitude axis on both channels.** Fitting
the optimal per-mode real gain on the fold's `train` split and applying it to the
holdout (fold mim1; mim2 agrees to within 0.001 on every column):

| | rel_l2 | tke | mvpe | sps | R_const | R_oracle |
| --- | --- | --- | --- | --- | --- | --- |
| as trained | 95.475 | 83.056 | 95.773 | 45.035 | 0.5974 | 0.9087 |
| per-mode gain, out of sample | 95.534 | **79.072** | 95.773 | 43.895 | 0.5972 | 0.9089 |
| Δ | **+0.059** | **−3.985** | 0.000 | −1.140 | **−0.0002** | +0.0002 |

The `rel_l2` gain is +0.059 of subscore, i.e. **+0.028 of final** at weight 0.4738 —
which confirms §2.2's ~0.02 prediction. But TKE falls **four points**, i.e. −0.42 of
final at weight 0.1057, so the *net* is about **−0.39**. And the reason this screen
existed — the hope that `R_const` is not bounded by the L2 arithmetic because it is a
different functional — is **refuted**: `R_const` moves by −0.0002, which is nothing.
Per-mode amplitude control is dead on the accuracy channel and dead on the interval
channel. Sixth independent route to the tke/rel frontier, sixth agreement at zero.

**One live thread: `wprofF`.** At a third seed it firms up to **+0.011 ± 0.002 on
4/4 folds** (+0.029 ± 0.003 with the sps proxy) — statistically real at 5.5 SE, and
carrying **0.56x the parameters at measured parity on cost**: timed on an A100 at
batch 256 in fp16, 0.2028 ms/sample per member against the shipped 0.2062, i.e. 1.6%
*faster*, which confirms the FLOP proxy the arm was designed against. The detail that matters is what *did not* happen: `tke` is
+0.006, not down. That is the exact failure mode that sank `--mults 1,2`, whose `tke`
fell 0.288 — and dropping a whole level is a different intervention from reallocating
across three, so the difference is mechanically plausible rather than lucky.

**And it passes the screen that `--mults 1,2` failed.** Capacity changes have to be
scored as an *ensemble*, because the fold rig averages single models while the board
scores three: `--mults 1,2` read +0.005 per member and **−0.053 as an ensemble**, a
compression of −10.6. `tools/fold_ensemble.py tke1em8 wprofF` over the four folds:

| | per-member Δ | ensemble Δ | compression |
| --- | --- | --- | --- |
| `wprofF` | +0.025 | **+0.029** | **1.19** |

No compression at all — the ensemble delta is marginally *larger*. So the instrument
artefact that made the previous capacity reduction look good on folds and lose on the
board is absent here, and that is the single most informative thing measured about this
arm.

**What it is and is not worth.** At +0.011 accuracy-only it is **below the ~0.06 bar
`ideas.md` sets for spending a submission**, and it should not be shipped on its own.
What it *is*, at a small but consistent accuracy *gain* and measured cost parity, is a
**44% smaller model**:

| | member | 3 members + head, fp16 |
| --- | --- | --- |
| shipped | 22.65 M | 162.1 MB of 256 |
| `wprofF` | 12.57 M | **101.6 MB of 256** |

That matters for two reasons that are not accuracy. The Decision Phase retrains the
shortlist from scratch, so a smaller model is less exposed to whatever their environment
does differently; and 60 MB of freed checkpoint budget is optionality for anything later
that wants parameters somewhere else. Treat it as a **better baseline to build on**
rather than as a submission — and put it through `tools/honest_pipeline.sh` before it
becomes one, since the fold rig trains on 88% of the release against the shipped 100%
and that is exactly where a capacity reduction flatters itself.

One thing the rig cannot see either way: it scores **no time term**. This arm was built
FLOP-matched for that reason. The parameter-matched version of the same reallocation
(width 160 at mults 1,1.5,2, 0.95x parameters) is a **1.68x slower network**, and
nothing in `kfold.py` would have told us.

### Two documentation fixes

**The +0.71.** `README.md` ("Simulation pays only as a continuous regulariser — worth
+0.71"), `FINDINGS.md` §3 and `HANDOFF.md` §8.3 all quote it, and it is the number anyone
prioritising work will read. Two independent measurements here — mimic folds under the
current recipe, and the archived `simval` arms on their own splits — put the axis at
+0.044 to +0.053 on the accuracy channel and ~0.10 of final after rig calibration. The
old figure is on a different scale from either and I could not reconstruct which; the
right action is to find its provenance and state the units next to it, not to overwrite
it. Note this is not a claim that the old measurement was wrong — only that it is not
usable for prioritisation as quoted.

**The bottleneck's shape.** `unet.py`'s module docstring says "three downsamplings take it
to 4x8". Under the shipped `--mults 1,2,4` there are **two** downsamplings and the
bottleneck is **8x16** (`pools` is `Identity` at the last level). The receptive-field
argument in the same docstring is derived from the wrong depth — and §2.3 shows the true
field saturates the domain, so the argument was not load-bearing anyway.
