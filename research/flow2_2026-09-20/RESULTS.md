# The flow as a donor — zero-training pre-test (2026-09-20)

FM1's fold checkpoints (Codex round 8, trained for *intervals* and killed there), K = 16 / S = 8, sample mean scored
against the frozen **gained** four-member ensemble on each fold's held-out cells; exact marginals, cell-balanced.

| fold | ρ(error, ensemble error) | RMS ratio | MSE-optimal w* | rel at w* | TKE at w* | exact ΔQ at w* |
|---|---|---|---|---|---|---|
| mim1 | 0.855 | 0.991 | 0.53 | −4.3 % | +22 % | −0.268 |
| far_drop | 0.874 | 0.997 | 0.51 | −3.2 % | +15 % | −0.181 |
| angle05 | 0.810 | 0.972 | 0.57 | −4.3 % | +17 % | −0.224 |

**Why this matters.** Every donor this project ever measured was either correlated (seeds 0.94; temporal stem, frame,
MIMO, XScan 0.88–0.97) or much worse (FNO 0.86–0.90 at +5–7 % error; attention 0.69 at +27 %). The flow's sample mean
is the first that is **both decorrelated (0.81–0.87) and equally accurate (0.97–1.00)** — the MSE-optimal weight is a
50/50 partnership, and it improves rel by 1.2–4.3 % and MVPE slightly. This is the "different function class" the
shared-error decomposition (`FINDINGS.md` §2: ρ = 0.94 between seeds, 90 % shared error) said was needed.

**Why it does not pay yet.** Blending two decorrelated fields damps the fluctuation amplitude further, so TKE degrades
15–22 % and every blend is net negative. Counterfactually, with TKE held at base the same blends read **+0.021 (w = 0.1)
to +0.063 (w*) exact Q** — at or above the FM2 continue gate.

**The amplitude repair is the crux, and it is not available today.** Gain-target variants on the *unblended* forecast
(`blend2_<fold>.json`): member variance (shipped) 0.000 · flow sample amplitude **−0.023 / −0.058 / −0.058** · elementwise
max **+0.007 / +0.008 / +0.005**. The flow *under-produces* fluctuation energy, exactly as the archive recorded
("plain flow matching under-produces it, 0.51 of the measured value"), so its amplitude estimate is biased low. The best
blend with the max target is +0.001 on the fold mean.

## The dose sweep (the by-product that matters most)
Repairing the blend's amplitude needs the gain at **dose 1.0**, not 0.5: at dose 1 the gain sets the variance *to* the
target regardless of the prediction's own damping, so a more-damped blend is fully corrected. Sweeping (weight × dose)
with official SPS on the held-out cells (`sweep_sps_<fold>.json`), exact ΔF vs the shipped dose 0.5:

| setting | mim1 | far_drop | angle05 | mean |
|---|---|---|---|---|
| no flow, dose 0.9 / **1.0** / 1.1 | +0.013 / **+0.014** / +0.014 | +0.016 / **+0.017** / +0.015 | +0.040 / **+0.043** / +0.042 | +0.023 / **+0.0245** / +0.024 |
| flow w = 0.1, dose 1.0 | +0.024 | +0.029 | +0.051 | +0.034 |
| flow w = 0.2, dose 1.0 | +0.029 | +0.036 | +0.054 | +0.040 |
| flow w = 0.3, dose 1.0 | +0.031 | +0.037 | +0.053 | +0.040 |

**Two separable findings.** (1) **The dose alone is +0.0245**, needs no flow, and is one character in the shipped file —
armed as `canon0918d10` for 09-21. (2) **The flow donor adds +0.015 on top**, all folds positive, but needs a second
model, 30 ms/window and the 180 s wall (≈ 3 k windows max). The flow is worth research; the dose is worth a slot.

**Where this leaves FM2.** The premise is half-confirmed: the donor exists, the point-forecast gain is real, the energy
calibration is the gate. The archive records a fix that worked (a per-band spectral penalty differentiated through the
whole solve, 2000 fine-tune steps), so Codex round 10 was redirected to calibrate energy *first* and to re-test the
amplitude target before spending the training budget on the conditioned residual flow.
