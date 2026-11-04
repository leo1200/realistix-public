# Screen T1 — fixed 12-mode temporal DCT decoder (owner-approved 2026-09-17; feasibility gate read before the arm was built)

**Feasibility gate (PREREG_FUTURE.md T1; `research/distil1_2026-09-17/t1_feasibility.py`, `t1_<fold>.json`).** Best
orthogonal projection of each held-out target's residual about the last input frame onto the first R of 20 orthonormal
temporal DCT-II modes (analytic basis, no fitting), rel_l2 floor vs the B four-member mean's rel error:

| fold | B mean rel error | floor R=4 | R=8 | **R=12** | R=16 | ratio R=12 / B | TKE relL2 of the R=12 projection |
|---|---|---|---|---|---|---|---|
| mim1 | 0.0742 | 0.0371 | 0.0130 | **0.0100** | 0.0057 | **0.135** | 0.022 |
| angle05 | 0.0488 | 0.0234 | 0.0090 | **0.0072** | 0.0042 | **0.148** | 0.023 |
| far_drop | 0.1029 | 0.0500 | 0.0174 | **0.0131** | 0.0074 | **0.127** | 0.020 |

**PASS** (kill was a floor above 1.3 × B's error): twelve temporal modes lose ≈ 13–15 % of B's *current* error as a
representational floor, and TKE damping of the projection is 2 %. The restriction costs almost nothing in
representable accuracy, so the arm tests only whether it changes generalisation (the mechanism), not capacity.
Target-assisted bound; it does not predict trainability.

**Arm (to be built).** `--temporal-rank 12` in the U-Net: the output convolution emits 12 coefficient fields per velocity
channel, decoded by the fixed 20 × 12 DCT matrix to the 20 frames; last-frame anchor and zero-initialised output kept;
every other constant B's. Inert at the default (rank 0 = the shipped 20-frame output) and identity-smoked before the
screen. Folds mim1 / angle05 / far_drop, seeds 0 and 1; 6 × 0.45 A100-h.

**Read and gates (from PREREG_FUTURE.md T1).** As a weighted extra member (w = 0.10) with the member-variance gain
(dose .5, cap 3) vs B + gain, and as a replacement (single T1 model vs a single B member); ρ vs the B mean and RMS ratio
reported. Prediction: ρ 0.85–0.98, RMS ratio 0.98–1.2, direct ΔQ 0 [−0.03, +0.02]; P(useful pass) ≤ 5 %. Pass to
seed replication: mean ΔQ ≥ +0.025, each fold ≥ 0, MVPE ≤ 1.005; kill: mean ≤ +0.01 or any fold < −0.02 or MVPE > 1.01.
No rank sweep after the read.

## Read (2026-09-17 ~16:00 UTC; `research/distil1_2026-09-17/donors_<fold>.json`; six runs, seeds 0/1)
| fold / seed | ρ vs B mean | RMS ratio | as donor w = 0.10 (with gain) ΔQ vs gained B | as replacement ΔQ vs a single B member | rel ratio | TKE ratio | MVPE ratio |
|---|---|---|---|---|---|---|---|
| mim1 s0 / s1 | 0.933 / 0.970 | 1.018 / 1.025 | −0.006 / +0.000 | +0.001 / −0.012 | 1.018 / 1.025 | 0.984 / 0.987 | 1.041 / 1.033 |
| angle05 s0 / s1 | 0.887 / 0.878 | 1.055 / 1.079 | −0.010 / −0.001 | −0.029 / −0.037 | 1.046 / 1.055 | 0.973 / 0.969 | 1.021 / 1.057 |
| far_drop s0 / s1 | 0.964 / 0.967 | 1.025 / 1.028 | +0.001 / −0.000 | +0.001 / −0.020 | 1.027 / 1.029 | 0.970 / 0.976 | 1.037 / 1.052 |
| **mean** | 0.93 | 1.04 | **−0.003** | **−0.016** | | | |

**Verdict: KILL** (donor mean ΔQ −0.003 ≤ +0.01; replacement worse than a single member on 4/6 runs; MVPE ratios 1.02–1.06
exceed 1.01 on every fold). The restriction does what the feasibility gate predicted for capacity — TKE 1.3–3 % *better*
(the low-rank basis suppresses noisy lead-specific fitting) — but it costs rel 2–5 % and MVPE 2–6 %, and its decorrelation
from the U-Net (ρ 0.88–0.97, lowest at the novel angle) is far from the ρ < 0.8 a useful donor needs. The mechanism
(different temporal inductive bias → different generalisation) is real in sign but too small and paid for in the
time-mean. Closed; no rank sweep. The `--temporal-rank` flag stays inert in the trainer.
