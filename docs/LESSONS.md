# Lessons — what this project learned about ensembling, selection and training (2026-09-17)

*Written for the next project as much as for this one. Every claim carries the measurement behind it; where the
record is thin it says so. Joint Claude + Codex (Codex's independent essay: `research/audit_2026-09-17/codex/round3/INSIGHTS_codex.md`;
disagreements are marked).*

## 1. Ensembling: what averaging buys, what it costs, and where the next gain was hiding

**The member ensemble bought accuracy and, through accuracy, the SPS prefactor.** (Codex's caution: neither
board K-comparison is a pure ablation — onelean also changed the loss, K4 also refitted the head — so the member
count's value is bracketed, −0.117…−0.171 non-time for one member vs three, +0.039 for four vs three, not identified.) Three members against one were
worth ≈ +0.12 of final on the board under two recipes (893273, 910411), and the fourth member +0.039 exact (923184).
A fifth was killed on its sentinel (in-sample SPS 51.016 vs 51.020): the K-axis is concave and the size cap
(256 MB) closes it at four width-96 members anyway. The spread input to the interval head, the mechanism the
ensemble was *supposed* to feed, measures ≈ 0 under the current members — as the members improved, their errors
became predictable from (window, prediction) alone. The ensemble's value is the mean's accuracy, full stop.

**Measure what the ensemble removes before buying more of it.** Members trained with different seeds have error
fields correlated at 0.88–0.94; the four-member mean removes 4–9 % of the squared error, matching ρ + (1−ρ)/K exactly.
That is worth ≈ 3 % of rel and MVPE, and it costs TKE amplitude. Everything else the ensemble is worth on the board is
SPS through the accuracy prefactor and the head's calibration. A fifth member added nothing because there was almost
nothing independent left to average; a better single model would replace the mean; and "ensemble-level training" cannot
decorrelate a shared recipe bias (measured 2026-09-17, `research/lean_2026-09-17/k_ladder_*.json`).

**Averaging damps fluctuation amplitude, and the scorer's TKE term charges for it.** Members disagree on the phase of
the fluctuations; their mean has less temporal variance than any member and less than the truth. Every *trained*
lever that raised TKE (w_tke 1.5/2.0 per member, DropPath, time dilation, mean-flow mixing, the SPS-prefactor
objective) was buying amplitude at a rel/MVPE cost, and the ensemble mean did not inherit the per-member TKE gain
(ratios 0.97–1.04, sign inconsistent across folds) while the cost survived averaging. The right fix was post hoc and
target-free: rescale each cell's fluctuation toward the members' own average variance (dose 0.5, board tke +0.155,
rel −0.024, MVPE invariant to seven digits, exact N +0.0094). Once the gain is applied, nineteen closed forecaster arms
collapse onto the incumbent (best +0.009): what they differed in was amplitude, which the gain now supplies for free.
What remains is *pattern* — realised-fluctuation skill — and no arm moves it.

**Member diversity from training settings did not help; averaging *estimators* did.** Shorter EMA, lower weight
decay, different loss weights (39 + 39 runs at K = 3/4) read −0.003 to −0.037 as ensembles: the diversity averaging
needs is decorrelated *error*, not decorrelated training. The one place estimator noise had never been averaged was
the interval head: two independently fitted heads' *output bounds* averaged 50:50 paid +0.0807 board SPS (926634);
the fold K-ladder reads +0.100/+0.125/+0.135/+0.149 at K = 2/3/4/6, i.e. G(K) ≈ G∞(1 − 1/K) with G∞ ≈ 0.18 — the
signature of independent fitting noise on a shared, already-optimal policy (c* = 1.000 on every rescaling family).
Weight soups of the same heads are catastrophic (−4.5): independently initialised features are unaligned, so
averaging must happen in output space — or be *distilled*: a student regressed onto the six-head average carries
82 % of it at one head's cost (fold +0.122; board 43.945 vs the pair's 43.941).

**Distilling estimators succeeded exactly where distilling the forecaster failed** (−0.145 board): the target was a
fixed, smooth average of functions on the same inputs, not a harder learning problem than the original.

**Read the K-ladder as a variance/shared-error split.** Fixed-prefix averaging saturates near 0.18 fold SPS; the
residual (the oracle gap 0.612 → 0.842 W̄) is shared error no amount of head averaging removes. Codex's caveat
stands: the prefixes share heads and the fit is a heuristic, not a ceiling. The same arithmetic prices the
organisers' retrain: it is one draw of members *and* heads, and the devices that reduce our variance (four members,
the distilled student, fixed steps instead of validation argmax) reduce theirs. A one-off "lucky" seed is worthless
there; a low-variance recipe is not.

## 2. Parameter selection: every constant is a property of the rest of the recipe

**Constants re-selected under one recipe did not survive the next.** EMA 0.998 measured −0.007 under plain Muon and
+0.047 once w_sps existed; w_tke 2.0 cost 0.28 under the pre-w_sps loss and w_tke 1.0 paid +0.080 after it; the
stopping step moved 6000 → 8000 → 7000 → 6000, tracking the objective each time. Component attributions do not
transfer either: sim +0.71 → +0.053, conditioning +0.17 → ≈ 0, EMA +0.31 → +0.047, width +0.29 → +0.015 as the loss
re-derivation and the optimiser absorbed what they used to carry. Rule: after any recipe change re-read every
argmax off the fold histories; a stale vintage is a re-measure order, not a citation.

**Selection on a validation split the artefact will be scored on is a leak, and a fixed step is the cure.** The
head's val_aoa argmax kept narrowing intervals past what held-out data pays for; `--fixed-step` (4000 for heads, 6000
for members) transferred from runs that *could* validate. The same applies to the student (last step, no selection)
and to checkpoint choice at ensemble level (5000 loses to 6000 at K = 3 and 4).

**Gates must be priced with the funding channel removed — through both paths.** TKE enters the score directly and
through the SPS prefactor `0.3/(1+2e_tke)`; holding the displayed TKE subscore fixed is *not* N₀ (Codex,
`EXACT_DECISIONS.md`). Define N₀ by holding baseline TKE and substituting its per-window SPS prefactor. A gate cleared by TKE funding is not cleared: onelean
(910411) passed a tail gate at +0.26 with +0.57 of TKE and lost −0.047 on the board; coop25 gained board tke +0.137
and lost −0.023. Price with TKE zeroed for trained arms; for deterministic post-averaging transforms the k4g rule
(N ≥ +0.010 under every fit, channel floors, MVPE invariant) is the calibrated one — and under the exact score k4g
read +0.0094, so even that rule was met only approximately.

**Know the instrument's resolution and transfer.** The mimic folds resolve ≈ 0.02–0.03 on the accuracy channel
(seed sd 0.009 per run) and transfer rel ≈ 1×, MVPE ≈ 5× (under-priced locally), TKE sign-unstable; member-level
gains compress at ensemble level (`--mults 1,2`: +0.005 per member, −0.053 as an ensemble). Screen on rel_l2 and the
sign of the rest; score ensembles, not members, for anything touching capacity or diversity; a closure is a swept
range on a geometry, not a refuted mechanism.

**Pre-register, with a kill number, and read the miss honestly.** (Codex extended exact pricing to 43 named board
comparisons and 64 co-mentioned pairs: **no named board accuracy sign flips**; the half-dose gate is the one close
call, +0.0104 → +0.0094; Muon's non-time value +0.0544 → +0.0493, the TKE/EMA bundle +0.0895 → +0.0849.) Of the registered predictions, the mechanism
predictions mostly landed (the head pair +0.065 [−0.02, +0.15] → +0.081; the gain's channel signs all correct) and
the *clock* predictions mostly missed (924146 time band [−0.08, +0.08] → −0.25; 929146 final band missed on a 6.0 ms
draw). Band-hit counts are not a validated transfer model; the failures were transfer and clock, not score-fit noise.

## 3. Training strategies: derive the objective from the scorer, then optimise well

**The largest gain in the project was reading `scoring.py`.** Per-window relative L2 at weight 8 against MSE 1
(+0.48 board), TKE at 1.0 (+0.080), an SPS-bulk term `1 − mean exp(−|e|/σ)` (+0.057), MVPE at 0.3 (+0.009): the loss
is honestly `rel + tke + sps` and slightly better without the dead terms (lean +0.010, 4/4, unshipped). The one
convexity in the scorer that looked exploitable (the per-window SPS prefactor) is a 0.3–1.5 % Jensen gap and not
trainable (`spsoracle`).

**Muon was the optimiser find** (+0.070 ± 0.007 fold, +0.074 board): orthogonalised momentum on matrix parameters
(update stable rank 84 vs AdamW's 6.6); SOAP recovers two thirds, SOAP + Muon = Muon. EMA 0.998 (+0.047), Muon-group
weight decay 0.03 (+0.018 fold, +0.057 on the angle suite; board −0.008 accuracy, TKE-funded loss), batch 32,
8000-step cosine read at 6000. Width saturates at 96; depth, blocks, attention, FNO/CNO/Transolver are flat or worse.

**Simulation is a regulariser, not a teacher.** Joint training on 100 simulated trajectories is worth +0.053 fold and
more than half survives destroying sim's structure; the per-modality affine's marginal is ≈ 0 under the current loss.
Every attempt to *use* sim's information (translation objectives −0.6 to −1.4, curricula −0.068, regime bagging −0.107,
conditioning −0.09 to −0.17, mean-flow mixing: far-Re +0.011…+0.020 paid −0.013…−0.024 at a novel angle, frontier
upweighting, covariance augmentation) failed; the sim SPS term's non-dimensional units are wrong and removing them
*loses* 0.025 at a novel angle. Distillation of forecasters (−0.146 board), SAM, meta-updates, soups, spectral caps,
input masking, time dilation: closed.

**Generalisation costs are measurable and asymmetric.** A novel angle costs +16–20 % rel_l2 (6–9.5 % coverage-
specific, growing after step 4000); Reynolds extrapolation costs rel +0.006/+0.011/+0.015 and MVPE +0.005/+0.013/
+0.025 at 1/2/3 steps — MVPE drifts first and the board prices it hardest. Regularisation (wd 0.03, early stopping)
is the one repeatable weak positive at novel regimes, and it compresses at ensemble level through 5° amplitude.

## 4. The exact score, and what it changes

**`F = (0.5·rel^p + 0.2·sps^p + 0.1·tke^p + 0.1·time^p + 0.1·mvpe^p)^(1/p)`, p = 0.6854**, reproduces all 41
scored cards to 9e-11. The linear fits (14- and 34-card) were secants at our operating point and drifted with the
card population, which is why "the weights moved" was a recurring puzzle. Marginals at 929146: rel 0.477, sps 0.243,
tke 0.101, time 0.0965, mvpe 0.0957 — within 0.01 of the locked fit, so **no past decision flips by more than its
noise** (closest: k4g's N = +0.0094 against a +0.010 rule; wd03mv was a time-funded win at −0.0100 accuracy).
For a frozen forecaster at fixed runtime, maximising F over the head is exactly maximising SPS — the power mean
creates no new head optimum. Training acts on *errors*, not subscores: the direct error derivatives at A2 are ≈ −21.5
(rel), −3.2 (tke), −4.2 (mvpe) per unit error, plus the SPS pathways. **No re-derivation of the loss weights is warranted**: the forecaster's loss constants were selected on held-out
scores, not on the linear weights, and a 4 % change in one marginal is far inside the fold rig's resolution.
What the concavity *does* imply: (i) a point on the weakest subscore is worth most — SPS at 43.9 carries 0.243 for a
nominal 0.2; (ii) on a private set with unseen regimes, where rel and tke will be lower, the marginals shift toward
those weak channels, so robustness of rel/tke is worth relatively more there than here; (iii) gate rules should be
stated in exact terms (`dF/ds_j = w_j (F/s_j)^(1−p)`), and clock costs priced as `−0.0631 · Δms` at 5.7 ms.
The score is a mean of per-window means inside a power mean of subscores: no per-window convexity to arbitrage
(the archive checked this on 09-05; the SPS prefactor is the only per-window nonlinearity and it is small).

**The clock is engineering, and the container is not your workstation.** The gain's in-graph kernels cost +0.05 ms
quiet and up to +0.6 ms under CPU contention; a captured graph is not free of host cost. Recomputing the per-window
normalisation inside each of four member graphs was 4 × 17 redundant nodes — computing it once removes more than the
whole penalty (`k4a2portable`). And the evaluation image has no C compiler while Triton keys its cache on the
`libtriton.so` hash: the pip build's launchers never matched the conda image's, which is how the 09-06 Triton artefact
silently fell back and drew +0.7 ms. Verify deployment against the *exact image contents*, not a version string.

## 5. Method lessons that generalise
* "Accuracy has been flat" was the wrong summary: since 08-28 the exact attribution is +0.078 accuracy and +0.132
  time — two different achievements needing different controls (Codex).
* Two independent passes with a cross-examination round catch what one does not (the missing API page, the
  wrong accuracy attribution, the "no CPU cost inside a graph" assumption — all caught on 2026-09-17).
* Every number carries its vintage and its instrument; re-check stale numbers after a recipe change.
* The clock is a measurement of the platform, not of the model: control with byte-identical repeats, never infer a
  slope from a pair, and prefer candidates that are cheaper *and* better so the slope never has to be resolved.
* Packing fidelity is not training provenance: a checker must be shown mutants (a zeroed head, a step-2000
  snapshot, a substituted source) and reject them; only a replay authenticates a training run.
