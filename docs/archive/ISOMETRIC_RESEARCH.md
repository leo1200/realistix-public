# A test of refinement at fixed forecast statistics

2026-09-05. Research candidate; no breakthrough or score gain established.

The proposed mechanism preserves two already learned forecast quantities exactly
while learning the remaining temporal structure. At each spatial cell, write the
20-frame, two-component reference forecast as p = m + f. The two component means
in m determine MVPE, and ||f||² / 40 determines the cell's TKE. Holding these three
numbers fixed leaves a sphere of dimension 37 inside the 40-dimensional output
space. Neither metric fixes the direction on this sphere.

For a network proposal q, let z = q - mean_time(q). Return

    p_new = m + ||f|| z / ||z||.

The zero-radius and zero-direction cases retain p. This is the Euclidean closest
point to q on the constraint set; applying it to the target gives an analytic
L2 oracle for this intervention class. The oracle is a diagnostic only. Inference
uses solely the current input window and its frozen ensemble's predictions.

The proposed contribution is **learning temporal corrections on the joint level
set of forecast statistics**. It is not a claim that projected networks, sphere
normalization, or energy preservation are new. Nor is preserving a forecast's
TKE a conservation law of the real flow. Related work already includes
[energy-conserving turbulence closures](https://arxiv.org/abs/2301.13770) and
[ECO's quadratic projection for bounded chaotic operators](https://proceedings.mlr.press/v331/goertzen26a.html).
A literature search did not establish priority for this particular construction.

This differs from the repository's TKE-shape gate: that changed energy allocation
across space while retaining temporal direction. This does the complementary
operation: every cell's energy stays fixed and its temporal direction changes.
It also differs from a global time-shift or a linear delay-embedding correction.

## Screen fixed before trained results

- Four mimic folds, frozen existing tke1em8 ensembles, seeds 0/1/2.
- Width-32, depth-2 combiner reading members and the input window; zero output init.
- Two heads with identical initial weights and batches: unconstrained vs projected.
- 1,500 steps, batch 32, AdamW 3e-4, cosine to zero, weight decay 1e-5;
  loss 8*relative_L2 + SPS_bulk + TKE_relative_L2 in both arms.
- Each correction trains on its own fold's training trajectories only. Fold env
  is applied before importing split-dependent modules. Evaluate stride 40.
- Save the fixed final step, never the best validation step. Oracle and identity
  evaluated through the same scorer. SPS reported here is only the constant-band
  proxy, not the learned interval head and not a final-score estimate.
- Continue only if projected refinement cuts pooled relative-L2 error by at least
  2%, improves at least three folds, and beats the matched unconstrained head on
  accuracy excluding SPS. Require raw TKE/MVPE changes below 1e-6. This is a screen,
  not confirmation: any survivor needs fresh seeds/regime splits, a refitted
  interval head and runtime measurement before promotion.

The historical ensemble checkpoints do not contain every modern provenance field.
The runner preserves those warnings in its results; a positive result cannot be
called rigorously confirmed from directory names. The queue record for mim1 seed 2
does explicitly record the expected cell holdout. Any survivor needs the remaining
archive provenance verified or its baselines retrained with full stamps.

## Reproduce

Use the repository's `realistix` environment and allocated GPUs via `pq`:

```bash
python tools/check_isometric.py
pq sub -t a100 --name iso_mim1 -- python tools/train_isometric.py \
  --fold mim1 --out research/isometric/mim1
```

Repeat for mim2/mim3/mim4. Existing output paths are refused. The runner writes
`result.json` and a research-only `heads.pt` to each output directory. These are
not compatible submission packs. The champion and its training recipe are unchanged.

The invariant check covers float32/float64 temporal means and energy, bit-exact
zero-update identity, zero-energy degeneracies, finite derivatives and gradcheck.
It caught and fixed an inactive-branch overflow in the initial projection formula.


## Completed result

All four paired runs finished successfully. These are equal-fold summaries;
folds overlap in some regime cells, so four positive signs are not four fully
independent replications. One correction seed was tested.

| Fold | Relative-L2 error reduction | L2-oracle reduction | Rel-L2 subscore gain |
|---|---:|---:|---:|
| mim1 | 0.1613% | 35.2109% | +0.00562 |
| mim2 | 0.3528% | 33.1182% | +0.01057 |
| mim3 | 0.1094% | 35.9552% | +0.00389 |
| mim4 | 0.0862% | 33.2399% | +0.00292 |
| Equal-fold mean | **0.1774%** | **34.3810%** | **+0.00575** |

Raw TKE error changes by at most 5.96e-8; raw MVPE by at most 7.45e-9.
The field-level temporal means agree to 7.16e-7 in normalized units and TKE maps
to 2.99e-8. These are float32 rounding effects, consistent with the invariance.
All four zero-initialization comparisons were bit-identical in the scorer.

The equal-fold accuracy gain excluding SPS is +0.00272 for the projected
head against +0.04574 for the unconstrained control; it loses this comparison
on all four folds. These are weighted local subscores, not final scores.

**The predeclared screen fails.** Every fold is below the required 2% error
reduction, so no pooling convention could rescue it. The unconstrained control
loses relative-L2 on every fold but gains TKE; projected refinement gains a tiny
amount of fidelity while retaining TKE. This supports the mechanical distinction,
not a large predictive advance. No inference benchmark, learned-head SPS claim,
final-score conversion, shipping build, or leaderboard submission is warranted.

The target-informed oracle demonstrates a substantial geometric opportunity; it
is not a predictability bound or evidence that the opportunity is learnable. The
small correction trained here captures only about half a percent of the available
oracle reduction. This does not rule out training a representation from scratch
on the same manifold, but it does reject this particular frozen-ensemble refiner
as a breakthrough candidate. Priority/novelty in the literature remains unproven.

Machine-readable results are in `research/isometric/mim{1,2,3,4}/result.json`.
Queue jobs: 1788594665439-iso_mim1, 1788594770057-iso_mim2,
1788594770254-iso_mim3, 1788594770448-iso_mim4. The first two used H100s;
the latter two A100s. Both arms within each fold shared its GPU. Training took
51–69 seconds per fold after loading/evaluation setup, 12,000 correction updates
total across eight heads. Existing championship weights and recipes were not edited.
