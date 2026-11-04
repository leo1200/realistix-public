# Improvement screens — wave 1 (2026-09-10), Claude + Codex

**Question.** Do any of the audit's proposed levers, or the owner's two ideas, beat the shipped recipe at step
6000 on three geometries (mim1 interior cells, angle05 whole-angle real+sim withheld, far_drop edge-Re)?

**Arms (seed 0 first; `research/improve_2026-09-10/claude/arms_wave1.json`), all on `kfold.CURRENT_BASE`
(= 918211's member recipe) + `--stop-at 6000 --snapshots`, run dirs `bt_w1_<arm>_<fold>_s0`:**
B (control) · Z (`--zero-rest`, the gradient-clearing control every noise arm inherits) · R (`--sps-domain real`:
real-only sps_bulk numerator, whole-batch denominator) · U (`--sim-sps-scale 0.18`: sim errors in real-like units
inside sps_bulk only) · T (`--time-dilate 0.25 --dilate-factors 2 --dilate-domain real`: 25 % of real windows
read at double speed, model must infer the clock) · D03 / D08 (`--zero-rest --drop-path 0.03 / 0.08`, DropPath on
Block residual branches, 0 / p/2 / p at widths 96 / 192 / 384).

**Cut for wave 1 (Codex round 3):** channel dropout, input-frame masking, half-speed interpolation (noise TKE
−31 %, adjacent-frame noise correlation), sim-wide dilation, compound loss arms.

**Gates (Codex round 3/4).** Read history at step 6000 (provisional, window-pooled) with
`read_wave.py`; direct Q = .4738 S(rel) + .1057 S(tke) + .0926 S(mvpe). Continue an arm only if equal-fold mean
> +0.015 and no fold < −0.020; R/U every fold > 0; T far ≥ +0.020 and mimic ≥ −0.010 vs B; D angle ≥ +0.020 and
mimic ≥ −0.010 vs Z and also passing vs B. At most one survivor goes to seed 1 with B and Z (9 runs); then
ensembles + honest heads on the audit's thirteen-fold suite before any promotion talk.

**Implementation record.** Flags added to `tools/train_mixed.py`, `models/unet.py`, `gpudata.py`, `losses.py`
(patches: `claude_changes_{src,trainer}.patch`; pre-change tree reconstructed at /tmp/prepatch). Identity
verified: pre-patch vs live at defaults, 20 steps, same seed/fold — history metrics identical and all 164 tensors
of best.pt equal (`smoke3.log`). Every new flag runs and evaluates finite (`smoke.log`). Codex round-4 audit
(`codex/codex_report4.md`) found and we fixed: a broken pre-patch reconstruction (`_finish` dropped), a dilation
start off-by-one (790 legal starts, not 789), missing dilation telemetry (now `dilate.*` keys in history.json),
launcher gaps (SIM_KEEPS_CELLS reset on every fold, `unset REALISTIX_SIM_DIR`, explicit per-arm allow-lists with
a full-field recipe comparison, duplicate-job check, JSONL job log, T runs far_drop first). CPU unit checks:
dilated rows stay inside the native frame bound, floor/ceil pairing for 0.5×, keep-denominator SPS gradient
semantics (`smoke`/unit outputs in this directory's logs).

**Mechanism notes.** Time dilation: the closed 2026-09-01 experiment synthesised intermediate-Re fields (TKE field
unchanged → "hybrids"); this arm tests speed-inference as an invariance/regulariser, so the old closure does not
apply, but Re is not a pure time scaling and the augmentation removes a valid spatial→clock cue. Codex's clock
probe: native lag-1 fluctuation correlations 0.80–0.99; Re 26700 at stride 2 reads lag-2 0.60–0.69, below every
native value (a clock faster than anything seen); stride-2 target TKE ×1.20 median and a flatter per-lead energy
profile. SPS: sps_bulk prices sim in non-dimensional units against the real sigma (median real/sim scale
0.177); sim carries 87.5 % of the term's parameter-gradient norm at the incumbent checkpoint.

## Wave 1 read (seed 0, 2026-09-10 evening) — no arm passes its pre-registered gate; B stays

Cell-aggregated direct Q vs B (vs Z), `wave1_seed0_cell_read.txt`, Codex's independent recomputation
`codex/round5_read.json` agrees to 1e-6:

| arm | mim1 | angle05 | far_drop | mean vs B | mean vs Z | verdict |
|---|---|---|---|---|---|---|
| Z (`--zero-rest`) | +0.000 | +0.010 | **−0.037** | −0.009 | — | fails far floor; the far loss sits at Re 26700 (mvpe +12 %) |
| R (real-only sps_bulk) | −0.001 | **−0.030** | −0.008 | −0.013 | −0.004 | stop: rel +1.6 %, mvpe +3.4 % at the novel angle |
| U (sim sps in real-like units) | −0.001 | **−0.023** | −0.016 | −0.014 | −0.005 | stop: rel +1.4 %, mvpe +3.5 % at the novel angle |
| T (25 % real windows at 2× speed) | −0.012 | +0.010 | −0.008 | −0.003 | +0.005 | stop: fails mimic ≥ −0.010 and far ≥ +0.020; far loss at the 3-step row (tke +0.016, mvpe +3.3 %) |
| D03 (Z + DropPath 0.03) | +0.005 | +0.007 | −0.021 | −0.003 | +0.006 | stop |
| D08 (Z + DropPath 0.08) | **+0.017** | +0.001 | +0.001 | +0.006 | +0.015 | fails the D angle gate (−0.009 vs Z); the only arm non-negative on all folds vs B |

What the reads say. (i) **The sim SPS-bulk term is load-bearing at novel angles**: deleting it or scaling sim errors
into real-like units both cost ~1.5 % rel_l2 and ~3.5 % mvpe at angle05 — the "different loss treatment of sim and
real" idea closes for the SPS term under this recipe (one seed, two interventions; up-weighting sim SPS or a
common normalised-unit sigma are not licensed by a loss that only shows removal hurts). (ii) **Time dilation** was a
fair test of the owner's idea at p = 0.25, factor 2, real only (25.17 % of real draws dilated, all training
trajectories exposed, no fallbacks; 37/71 far-training trajectories had stride-2 correlations below the native
minimum, so the clock intervention was substantial): it helps rel_l2 a little everywhere (−0.2 to −0.9 %) but pays
in tke and mvpe at the far row where the clock-inference mechanism was supposed to help. p = 0.5, real+sim or factor
3 are new hypotheses, not funded. (iii) **Zero-rest costs −0.037 at far-Re on seed 0** (never measured there
before); the D arms inherit it, and D08 recovers it. (iv) **DropPath 0.08** is the one lever with a non-negative
read on every fold vs B and +0.017 on interior cells (wd 0.03, which shipped, read +0.018 per member on the same
rig), but its angle gain vs Z is negative and the dose trend (0.03 → 0.08) improves mimic/far while angle
deteriorates.

Running (submitted before Codex's round-5 verdict): seed 1 of B, Z, D08 (9 runs — B/Z resolve whether Z's far loss
is a seed accident; D08 is an **exploratory exception**, thresholds below) and D15 seed 0 (a dose-response
diagnostic only; declared now: no choosing between D08 and D15 on seed-1 outcomes; any DropPath promotion path needs
a fresh pre-registered dose + seeds 11/12/13 + the thirteen-fold ensemble read). Codex's seed-1 requirements for
D08 to earn the ensemble read (pooled two-seed floors): D08−Z three-fold mean > +0.01497 with angle05 ≥ +0.049;
D08−B mean > +0.02364 with every fold ≥ −0.041…−0.057; both from `codex/round5_read.json`.

## Seed 1 + D15 (2026-09-10 night) — pooled reads, final wave-1 verdict in `docs/IMPROVE_2026-09-10.md`
Z vs B seed 1: −0.001 / +0.006 / +0.016 (far sign reversed between seeds; recurring harm unconfirmed). D08 vs B seed 1: −0.002 / +0.030
/ +0.016; pooled two-seed +0.007 / +0.016 / +0.009, mean **+0.011**, 5/6 pairs positive; vs Z pooled +0.012 (angle
+0.007 < +0.020 required). D15 seed 0 vs B: +0.004 / +0.020 / +0.005 (mean +0.009; vs Z +0.018). Both DropPath
doses fail Codex's pre-registered floors; recorded as a consistent sub-bar positive of the regularisation class.

Codex round 6 (`codex/codex_report6.md`) recomputed all 33 caches (agreement 1e-14) and applied the frozen rules:
D08 seed-1 means +0.0148 vs B / +0.0080 vs Z (required > +0.0236 / > +0.0150), angle vs Z +0.024 (≥ +0.049) — stop;
D15 mean vs B +0.009, angle vs Z +0.010 — stop. Wording corrected in docs/IMPROVE_2026-09-10.md accordingly.

## Wave 2 (2026-09-10 night) — the owner's continuous sim/real fraction, and input-frame masking
Mechanism tested (`--mix-prob p --mix-alpha a --mix-cond alpha|real`): with probability p a real window's input
and target are shifted on valid cells by α·(mean_sim(cell) − mean_real(cell)), α ~ U(0, a), both means per-cell
time-means in trajectory-normalised units (the morph convention); the modality FiLM sees α (rows interpolated,
`_apply` float branch — exact at 0/1, unit-tested) or, in the `real` variant, nothing (the network must infer the
shifted mean flow). Pointwise mixup of unaligned sim/real fields was not built: fluctuations of unaligned fields add
in quadrature, so the hybrid's TKE is wrong by construction. Arms: MIXa (p .5, a .5), MIXb (p .5, a 1.0), MIXr (p
.5, a .5, cond real), FM05 (`--frame-mask 0.05`, anchor frame kept). Same folds, seed 0, controls B/Z from wave 1,
same gates (mean > +0.015, no fold < −0.020; a generalisation arm needs angle05 or far ≥ +0.020). Prior: ≤ +0.02;
the per-modality affine's marginal is ≈ 0 and simulation's whole value is +0.053, so a continuous fraction can
regularise but cannot add information. Smoke: default run bit-identical to the earlier identity checkpoint;
mix arms finite (`smoke4.log`).

### Wave 2 exploratory reads (seed 0, cell-aggregated dQ vs B) — and a defect found by Codex round 8
| arm | mim1 | angle05 | far_drop | mean | mvpe at far |
|---|---|---|---|---|---|
| MIXa (p .5, α≤.5, FiLM sees α) | +0.005 | −0.003 | +0.016 | +0.006 | −1.5 % |
| MIXb (p .5, α≤1.0, FiLM sees α) | +0.011 | +0.012 | **+0.020** | **+0.014** | −2.7 % |
| MIXr (p .5, α≤.5, network told nothing) | +0.008 | −0.011 | **+0.030** | +0.009 | −3.5 % |
| FM05 (input-frame masking 0.05) | −0.018 | **−0.055** | **−0.079** | −0.051 | +7.1 % |

Frame masking is dead (rel +2–3 %, mvpe +4–7 %; corrupting temporal evidence hurts as dilation did). The mean-flow
mixing arms are the first arms of the day positive on far_drop by ≥ +0.016 with MVPE improving 1.5–3.5 % there —
the extrapolation channel where regime-mean bias lives — and MIXb is positive on all three folds (+0.014 mean, still
below the +0.015 gate, angle +0.012 < +0.020). **These runs carry a defect** (Codex `codex_report8.md`): the shift
was masked by "cell observed in any input frame", so ~2.7 % of exact-zero (unobserved) target entries became
non-zero fabricated targets that `sps_bulk` then scored. Fixed: the shift is confined to cells observed in every
frame of both trajectories (`stable` mask), plus guards (no tail holdout / phase-aug / unroll / out-gain). The
defective runs are kept as exploratory diagnostics only; corrected reruns under new names (MIX2*) are the read.
Also queued (Codex round-7 rank 2): the regularisation × duration factorial Z12k / D0812k (12k cosine read at 9k).

## Timing probes built (2026-09-10 night) — not submitted
`submissions/timing_p0/submission.py` (persistence forecast, explicit default-band bounds, our payload class: one
H2D, fp16 prediction + two fp16 bound arrays, three D2H, GPU path) and `timing_p1/` (identical outputs plus
EXTRA_WORK = 32 identity 3×3 convolutions on a fixed 512-channel fp16 buffer, on the same stream before the copies).
Local H100 reads were taken while another user's job saturated all four local GPUs (99–100 %), so they are upper
bounds: P0 1.43 ms/call, +0.045 ms per extra conv; the incumbent read 5.1 ms under the same contention against
Codex's 2.65 ms on an idle card. A clean A100 measurement is queued on pq (`timing_probe_a100_v2.log`); P1's
EXTRA_WORK is to be set so g1 − g0 ≈ 2 ms on A100 before any upload. Submission of P0/P1 needs the owner's go
(two slots; each scores far below the incumbent by design and does not touch the best).

### Regularisation × duration factorial (Codex round-7 rank 2) — dead on seed 0
12k cosine read at 9000 (history, native pooling) vs the 8k recipe read at 6000: Z12k −0.010 / −0.033 / −0.033 vs B
(mean −0.025); D0812k +0.008 / +0.003 / +0.002 vs B (mean +0.004) and **−0.002 vs D08@6000** — the longer horizon adds
nothing on top of DropPath, and without DropPath it hurts (the 09-05 "12k dead under the current recipe" reading
stands with strong regularisation too). Both 12k curves peak near step 8000 and decline to 9000. Not funded further.

### Corrected mixing reruns (MIX2*, stable-support mask), seed 0, cell dQ vs B — complete
| arm | mim1 | angle05 | far_drop | mean | mvpe far |
|---|---|---|---|---|---|
| MIX2b (α≤1, FiLM sees α) | +0.002 | −0.013 | +0.011 | 0.000 | −2.2 % |
| MIX2rb (α≤1, hidden) | +0.003 | −0.013 | +0.020 | +0.003 | −2.6 % |
| MIX2a (α≤.5, FiLM sees α) | +0.021 | −0.024 | +0.015 | +0.004 | −1.8 % |
The defective runs' angle05 gains (+0.012 for α≤1) do not survive the support fix (−0.013): part of that read came
from the fabricated targets. What survives across all six mixing runs so far: **far-Re +0.011…+0.030 with MVPE
−1.5…−3.5 %**, interior ≈ 0…+0.02, novel angle ≈ −0.01. Reading: shifting the mean flow by the sim/real level
difference teaches robustness to Reynolds-like level errors of the mean flow (which is what extrapolation in Re
gets wrong first — FINDINGS §0c), not to the shape changes a new angle brings. Fails the gate as a whole; the
far-Re channel is the specific signal to think about.

### A100 timing measurement (pq, A100-PCIE-40GB, idle card): the payload-class floor is small
P0 (persistence, our payload class, GPU path): **0.40 ms/call wall (0.37 GPU-event)**; extra identity convs
(512 ch): ≈ +0.10 ms each (16 → 1.97 ms); **incumbent wd03mv graphed: 3.98 ms/call (3.93 GPU)**. Against the board's
5.25 ms/sample that leaves ≈ 1.3 ms of container/harness overhead on top of our own GPU chain on this card class
(the evaluation A800-SXM is faster than a PCIE A100, so the true F is likely larger than 1.3 ms but well below the
3.4 ms two-point inference). The chain itself — three members + head at ~3.9 ms — is the bulk of the per-sample
time: a 2 ms path would be worth ≈ +0.2 in both phases *if* F is ~1.3 ms. P0/P1 on the board settle F:
t0 = F + a·0.4, t1 = F + a·(0.4 + 32·0.1). Zips built: build/timing_p{0,1}.zip
(submission.py only; standalone smoke passed: shapes, finiteness, ordered bounds, persistence, class path,
P0 ≡ P1 outputs). Awaiting the owner's go for two slots.

### Wave 4 — LEVEL10 (Codex round 9's last bounded screen), queued 2026-09-11
`--level-prob 0.5 --level-beta 0.10`: with probability .5 a real window's input and target are shifted by
β·(input temporal-mean field), β ~ U(−.1, .1), on cells observed in every frame of the trajectory; no simulation
involved, tags/wscale unchanged, private RNG. Tests whether the CFD direction in the mixing arm matters at all or
generic mean-level variation suffices. Pre-registered gate (Codex): mean > +0.015 vs B, every fold ≥ −0.020,
angle05 ≥ 0, one-step far row ΔQ ≥ +0.010 with MVPE non-increasing; a pass earns a joint high-Re/AoA-5 guard fold
and a seed-1 replicate before anything else. Prior ≈ 0. Smoke: default bit-identical; both doses finite; 146/260
real windows shifted in 20 steps (`smoke6.log`). Run dirs bt_w1_LEVEL10_{mim1,angle05,far_drop}_s0.
**LEVEL10 read (seed 0, cell dQ vs B): +0.006 / −0.004 / −0.013, mean −0.004 — fails (far worsens: MVPE +2.4 %, the
3-step row −0.040 / MVPE +5.8 %).** Generic scalar level variation of the mean flow does not reproduce the mixing
arm's far-Re gain and hurts where the mixing arm helped; the CFD sim/real mean-flow *direction* was the operative
ingredient. This ends the branch under the pre-registered rule.
