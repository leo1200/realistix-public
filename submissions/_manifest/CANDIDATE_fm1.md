# Screen FM1 — full-field conditional-flow interval model (open round, 2026-09-18; Codex round 8) — KILLED
Design and registration: `research/audit_2026-09-17/codex/round7/REPORT_full.md`, `round8/PREREG.md`; code under
`research/flow_2026-09-18/`; read `round8/REPORT_full.md`, `screen_summary.json`. Residual flow matching about the last
frame, time-conditioned U-Net (zero-init FiLM), 8000 steps, seed 0, three folds; interval-only read (gained incumbent
forecast fixed) with three readouts, the smoothed sample-CDF SPS solver selected inside outer training on every fold.

| fold | ΔSPS vs the fold A2 student (gate aggregation) | K8/S4 | K16/S4 | K16/S8 (primary) | K32/S8 |
|---|---|---|---|---|---|
| mim1 | **−1.867** | −3.78 | −2.74 | −1.87 | −1.73 |
| angle05 | **−1.258** | | | | |
| far_drop | **−2.137** | | | | |
| mean | **−1.754** | | | | |
All 34 held-out cells negative vs A2 (−0.96 … −2.79). Step 6000 reads are worse than 8000 (−2.18 / −1.25 / −2.43).

**Verdict: KILL** (gate kill ≤ +0.05 or any fold < −0.10). Sample-based intervals from a learned joint distribution of
the residual are 1.3–2.1 SPS below the reward-trained head after a full 8000-step budget; more samples and steps help
monotonically but converge far short (K32/S8 −1.73). Compute: 1.18 A100-h + 0.84 local GPU-h. A first recipe on a
short horizon, not a theorem against generative forecasting — but it closes the "learned predictive distribution
beats the reward-trained width" hypothesis at the budget we can afford before the freeze, as D1 did for the marginal
law. Codex's fresh-ordinary-head controls in this round (≈ 3.8 SPS below A2) are not the shipped head protocol and are
not used for the verdict; the A2 comparison is.

**One observation worth keeping (not a candidate).** As a *point* forecaster the flow's K = 16 sample mean has rel error
0.07425 / 0.04923 / 0.10412 against the gained incumbent's 0.07580 / 0.04932 / 0.10416 — equal or 2 % better on mim1 —
with MVPE 5–15 % worse and TKE far worse (damped sample mean). A generatively trained U-Net reaches the deterministic
ensemble's rel from a *different* training objective; as a low-weight donor or after a longer horizon it is the kind of
"different function class" the shared-error decomposition asked for. Follow-up project, not pre-freeze work.
Inference cost K = 16 × S = 8: 45 ms GPU per window on an A100 (the eager fallback would breach the 180 s wall for the
hidden set's size only if N ≫ 3000; measure before any such artefact).
