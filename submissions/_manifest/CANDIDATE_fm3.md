# Research FM3 — distil the flow's sample mean into one network, blend it component-wise (pre-registered 2026-09-21, before the build)

**Why.** The flow's K = 16 sample mean is the only donor this project has found that is decorrelated from our ensemble
(ρ 0.81–0.87 vs 0.94 between seeds) *and* equally accurate (RMS 0.97–1.00) — `research/flow2_2026-09-20/RESULTS.md`.
Naively blending it damps fluctuation amplitude and loses; the dose-1.0 repair was killed on the board (−0.0154 exact:
amplifying a wrong pattern). **The fix that survives is component-wise blending at the shipped dose 0.5**
(`split_blend.json`): blend the time-mean directly, blend the fluctuation *pattern* at renormalised variance, so **TKE
is exactly invariant by construction** and MVPE moves only through the time-mean:

| variant (w = 0.3) | mim1 | far_drop | angle05 | mean fold ΔQ |
|---|---|---|---|---|
| A time-mean only (TKE invariant) | +0.0077 | +0.0140 | +0.0008 | +0.0075 |
| B fluctuation pattern only (MVPE invariant) | +0.0033 | −0.0002 | +0.0053 | +0.0028 |
| **C both** | **+0.0110** | **+0.0138** | **+0.0061** | **+0.0103** |

**The efficiency.** Only the sample *mean* is used, and it is a deterministic smooth function of the input, so it is
distillable: one forward pass (~1 ms) instead of K × S = 128 (~45 ms), which also removes the 180 s wall constraint
(≈ 3 k windows at K16/S8). This is the argument that made the interval student work (82 % of a six-head average at one
head's cost) where forecaster distillation failed — and here the student's target is a *correction blended at fixed
variance*, not a damped mean, which is what killed DS1/DS2/DS3.

**Arms** (folds mim1 / angle05 / far_drop; code under `research/flow3_2026-09-21/`, frozen tree untouched):
S0 = the teacher blend (variant C at w ∈ {0.2, 0.3}, K = 16 / S = 8) — the ceiling · **S1 = a distilled student**
(small U-Net or the shipped trunk, regressing the flow sample mean from the window; 4000 steps) used in the same
component-wise blend · S2 = S1 with the blend weight fitted on outer-training windows only.

**Gates (fixed now).** Continue to an all-data build and a slot: **mean fold ΔQ ≥ +0.015 with every fold ≥ 0**, TKE
ratio within 1e-3 of 1 (the construction must hold), MVPE ratio ≤ 1.000 on the mean, and the student ≥ 0.7 × the
teacher's gain. **Kill:** mean ≤ +0.005, or any fold < −0.005, or student capture < 0.5. Between: documented, no slot.
**Prior:** teacher +0.010 (measured), student +0.008 [0, +0.015]; P(clear +0.015) ≈ 0.15.
**Whatever the read, the frozen method (934147 bytes, dose 0.5) is the deliverable unless a board slot confirms FM3.**

## Read (2026-09-21, Codex round 11; `research/audit_2026-09-17/codex/round11/REPORT_full.md`, `summary.json`)
Official SPS from the fold A2 students, exact marginals, cell-balanced (far_drop Re-row); the gained forecast's
containment guard applied; TKE invariance verified numerically (max deviation 3.5e-8, i.e. the construction holds).

| arm | mim1 | far_drop | angle05 | mean ΔQ | capture of the teacher |
|---|---|---|---|---|---|
| S0 teacher, w = 0.2 / 0.3 (K16/S8 flow) | +0.0153 / +0.0173 | +0.0188 / +0.0199 | +0.0113 / +0.0095 | **+0.0151 / +0.0156** | — |
| **S1 distilled student, w = 0.3** | +0.0124 | +0.0099 | +0.0185 | **+0.0136** | **87.3 %** |
| S2 student, weight fitted on outer training | +0.0092 | +0.0077 | +0.0097 | +0.0089 | 57.1 % |

**Verdict: BETWEEN — documented, no slot** (continue ≥ +0.015, kill ≤ +0.005). Every fold is positive on every arm,
MVPE ratios are ≤ 1.000, and the TKE-invariance construction holds to 1e-7. The distillation worked as predicted: one
forward pass at **1.78 ms/window** (against 45 ms for the K16/S8 sampler) captures **87 %** of the teacher, and the
student is *more* decorrelated from our ensemble than the teacher (ρ 0.84–0.89 vs 0.81–0.87) while staying equally
accurate. Channel decomposition at w = 0.3: the gain is rel (+0.008 / +0.003 / +0.011) and SPS through the prefactor
(+0.004 / +0.003 / +0.007), with MVPE ±0.004 and TKE exactly 0 by construction.

**Why it stops here.** +0.0136 fold is below the registered continue gate, and the honest transfer prior after the
dose-1.0 failure (fold +0.0245 → board −0.0154) does not support spending a slot, an extra model in the retrain
contract and a second training stage on it. Fitting the blend weight honestly (S2) costs a third of the gain, which is
the usual sign that the effect is near the instrument's resolution (0.02–0.03).

**What it establishes for the record.** A flow-matched model is the only donor this project found that is both
decorrelated (ρ 0.81–0.89) and equally accurate (RMS 0.97–1.00); its value survives distillation into **one forward
pass**; and blending it component-wise (time-mean directly, fluctuation pattern at renormalised variance) is the way to
use a donor without paying the amplitude tax that killed every previous attempt. That is the transferable design, and
it is a follow-up project, not a pre-deadline change.

## Sample-count scaling (2026-09-21, `research/flow2_2026-09-20/kscale_<fold>.json`; point-channel ΔQ, no SPS)
Nested prefixes of one noise bank, same checkpoints, no training. Distillation makes K free at inference, so the
teacher's K is a design choice, not a cost.

| K | mim1 (w = 0.3) | far_drop | angle05 | mean | donor time-mean ρ / RMS ratio |
|---|---|---|---|---|---|
| 16 | +0.0114 | +0.0131 | +0.0062 | +0.0102 | 0.89 / 1.04–1.15 |
| 32 | +0.0123 | +0.0142 | +0.0068 | +0.0111 | 0.89 / 1.04–1.14 |
| 64 | +0.0130 | +0.0148 | +0.0072 | +0.0117 | 0.89 / 1.03–1.14 |

Logarithmic saturation: 4× the samples buys +0.0015 (≈ +0.002 with SPS, i.e. the teacher would read ≈ +0.018 and the
student ≈ +0.016 at 87 % capture — the gate, but only just). **The binding limit is the donor's own accuracy**: its
time-mean is 3–15 % *worse* than ours, which caps the optimal weight near 0.3. At RMS parity with ρ 0.89 the optimal
weight would be 0.5 and the MSE reduction 5.5 % instead of 2.4 % — roughly double the blend gain.

## FM4: conditioning the flow on our forecast (2026-09-21/22, Codex round 12) — the axis closes
Trained the FM2 design for *this* job (inputs = window + the frozen gained forecast; target = the residual), 8000
steps, three folds, ≈ 1.2 A100-h. The diagnostic that was registered as decisive:

| fold | time-mean ρ: FM1 → FM4 | RMS ratio: FM1 → FM4 |
|---|---|---|
| mim1 | 0.892 → **0.977** | 1.036 → **0.992** |
| far_drop | 0.891 → **0.982** | 1.041 → **0.998** |
| angle05 | 0.809 → **0.936** | 1.143 → **1.032** |

**Conditioning did exactly what it was meant to and destroyed the reason to use it.** FM4 is more accurate than FM1 on
every fold — it reaches RMS parity with our ensemble, the condition that would have doubled the blend's value — but by
seeing our forecast it *inherits our errors*: ρ rises from 0.81–0.89 to 0.94–0.98. Best teacher blend **+0.0089** (vs
FM1's +0.0156), so no arm reached distillation. **This is the donor trade-off stated as a measurement: accuracy and
decorrelation are bought against each other, and the product (what the blend actually pays for) does not improve.**

**The flow axis is closed for this competition.** Four registered screens — FM1 intervals (−1.75 SPS), FM2 amplitude
(gate failed), FM3 component-wise blend (+0.0136, below gate), FM4 conditioned donor (+0.0089) — and the K-ladder.
What survives for the record is the design and the trade-off curve, not a candidate.
