# Research FM2 — residual conditional flow as a POINT forecaster (owner-approved 2026-09-20, post-freeze research; pre-registered before the build)

**Why, and how it differs from FM1.** FM1 used a flow's samples for *intervals* and lost 1.75 SPS. Its one useful
by-product was the opposite: the K = 16 **sample mean matched or beat our four-member ensemble on rel** (0.07425 /
0.04923 / 0.10412 vs 0.07580 / 0.04932 / 0.10416) from a different objective, while its TKE and MVPE were worse.
That is the signature of mechanism 1 in the leader's trajectory (`research/leap2_2026-09-18/`): inference-time compute
that keeps paying on the *accuracy* channels. FM2 therefore (i) conditions the flow on the frozen ensemble's gained
forecast, so it learns the *residual correction distribution* around our best deterministic prediction rather than the
field from scratch — the one thing aimed at the ~90 % shared error (`FINDINGS.md` §2); (ii) reads the **sample mean as
the point forecast** (Monte-Carlo variance reduction, the task FM1's sample mean already did well); (iii) reads a
**per-cell amplitude from the samples** as the gain's variance target, replacing the member-variance proxy — the
learned-variance-target lever that already read +0.012 held out (`research/audit_2026-09-17/codex/round4/`); (iv) keeps
the shipped interval estimator, refitted on the new prediction. Intervals are no longer the mechanism.

**Arms** (folds mim1 / angle05 / far_drop; the frozen members are the conditioning input and the control):
F0 = frozen gained ensemble (control, exists) · **F2a** = F0 + flow residual, sample mean (K = 16, S = 8), gain from the
member variance as shipped · **F2b** = F2a with the gain's variance target from the sample amplitude · seed 0 first,
seed 1 on a pass. Trained with CFM on the residual `(y − gained_mean)/q`, q a frozen training-only scale; 8000 steps,
incumbent data/EMA/batch; code under `research/flow2_2026-09-20/` (the frozen `src/`, `tools/`, `submissions/` untouched).

**Reads and gates (fixed now).** Per-fold rel / TKE / MVPE and exact-marginal Q at a **fixed clock**, cell-balanced
(far_drop Re-row), against F0; interval SPS from a fresh A2-protocol student refitted on each candidate's prediction,
with a matched fresh student on F0 as the control. **Continue to seed 1:** mean exact ΔF ≥ **+0.06** with every fold ≥ 0,
no fold's MVPE ratio > 1.005. **Kill:** mean ≤ +0.02, or any fold < −0.02. **Replacement of the frozen method needs**
both seeds ≥ +0.06, the fresh-geometry split with the same sign, the full retrain contract implemented, and the wall
budget below — otherwise the frozen method ships and FM2 is recorded as research.

**The wall, not the clock, is the binding constraint.** The prediction stage has 180 s; the evaluation set size N is
unpublished (the ingestion log prints only the per-sample mean, `research/audit_2026-09-17/codex/api_logs/`). Measured
A100 cost: 45 ms/window at K = 16 / S = 8, 22.6 ms at K = 16 / S = 4, 16.4 ms at K = 8 / S = 4 (`round8/a100_cost.json`).
Safe-N table (80 % of the wall, 20 s setup allowance): K16/S8 ≤ 3.1 k windows · K16/S4 ≤ 6.2 k · K8/S4 ≤ 8.5 k · the
frozen artefact ≈ 30 k. **Any shipped flow artefact must (a) pick K/S from a measured worst-case N, (b) carry a
deadline check that falls back to the frozen deterministic path, and (c) be re-verified against the organisers'
example when it appears (~09-22).** Prior: P(pass the continue gate) ≈ 0.2; P(a shippable replacement by 09-24) ≈ 0.07.

## Zero-training pre-test of the premise (2026-09-20, Claude; `research/flow2_2026-09-20/blend{,2}_<fold>.json`)
FM1's existing fold checkpoints, K = 16 / S = 8, sample mean vs the frozen **gained** ensemble on the held-out cells:

| fold | ρ(error, ensemble error) | RMS ratio | MSE-optimal blend w* | rel at w* | TKE at w* | exact ΔQ at w* |
|---|---|---|---|---|---|---|
| mim1 | **0.855** | **0.991** | 0.53 | 0.0746 → 0.0714 (−4.3 %) | 0.485 → 0.593 | −0.268 |
| far_drop | 0.874 | 0.997 | 0.51 | 0.1035 → 0.1002 (−3.2 %) | 0.564 → 0.652 | −0.181 |
| angle05 | 0.810 | 0.972 | 0.57 | 0.0493 → 0.0472 (−4.3 %) | 0.484 → 0.567 | −0.224 |

**The donor is real and it is the first of its kind in this project**: equally accurate (RMS 0.97–1.00) *and* genuinely
decorrelated (ρ 0.81–0.87 vs 0.94 between our seeds; the FNO was 0.86–0.90 at 5–7 % worse accuracy), so the MSE-optimal
weight is ≈ 0.53 — a true 50/50 partner. Blending improves **rel by 1.2–4.3 %** and MVPE slightly.
**The whole loss is amplitude**: blending two decorrelated fields damps the fluctuation further, and with TKE held at
its base the same blends would read **+0.021 (w = 0.1) … +0.063 (w*) exact Q** — at or above this screen's continue gate.
**No available amplitude estimator repairs it.** The flow's own sample amplitude is an *under*-estimate (the archive:
"plain flow matching under-produces fluctuation energy, 0.51 of the measured value"): as the gain's variance target it
reads **−0.023 / −0.058 / −0.058**; the elementwise max of member variance and flow amplitude reads **+0.007 / +0.008 /
+0.005 with no blending at all**, and the best blend (w = 0.1, gain = max) is **+0.001** on the fold mean.

**Consequence for FM2 (registration amended before the build):** the binding problem is not the point forecast, it is
the **fluctuation energy of the generative model**. FM2 must calibrate the sampler's energy first — the archive records
that a per-band spectral penalty differentiated through the *whole* solve fixes it cheaply (2000 fine-tune steps, 670 s;
one-step penalties and noise injection do not) — and only then is the sample amplitude a candidate gain target and the
blend a candidate point forecast. Continue gate unchanged (+0.06 exact ΔF, every fold ≥ 0, MVPE ≤ 1.005).
