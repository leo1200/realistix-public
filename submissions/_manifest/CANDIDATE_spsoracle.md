# Screen spsoracle — forecaster trained on the SPS reward TIMES the scorer's per-window accuracy prefactors (owner's question 2026-09-13; gate fixed BEFORE the read)

**Owner's question.** The subscores map error to score through 100/(1 + e/2): is there a convexity to exploit, and is
training on the evaluation terms allowed? **Facts:** `scoring.py` applies the map to the MEAN of the per-window errors
(lines 312–313, 150–153), so rel/tke/mvpe have no per-window convexity to arbitrage (the archive noted this on 09-05).
The one per-window nonlinearity is the SPS prefactor W_s = 0.5/(1 + 2 m_rel) + 0.3/(1 + 2 m_tke) + 0.2/(1 + 2 m_mvpe),
which multiplies every element's interval reward: its derivative is larger on already-accurate windows, so the
scorer's SPS branch rewards accuracy unevenly across windows. Rules: the metric is public in the starting kit;
prohibited are reading evaluation files, encoding evaluation data in outputs, and cross-window information; deriving
the loss from the published scorer is what the whole recipe already does. Allowed.

**Arm.** `--sps-objective oracle` (`losses.sps_oracle`: bulk reward × the three per-window prefactors, both receiving
gradients; physical-unit floors as in the scorer), w_sps 1.0 as shipped; everything else the incumbent recipe.
**Never run before** (0 of ~4300 archived runs). Seeds 0/1/2, folds mim1 / angle05 (drop) / far_drop, step 6000,
read as three-member means with the enstke gate (`CANDIDATE_enstke.md`): means D ≥ +0.025, D₀ ≥ +0.025, A₀ ≥ +0.040;
every fold D ≥ −0.020, D₀ ≥ −0.010; rel/MVPE ratios ≤ 1.005, far MVPE ≤ 1.000; TKE-free SPS prefactor mean ≥ 0; all
three weightings. **Mechanism-specific read:** the per-window prefactor W̄ itself (mean over held-out windows) must rise
vs B — that is the quantity the arm targets; a W̄ gain without an accuracy loss is the pass signature.
Prior: +0.005 [−0.02, +0.02] direct; W̄ +0.1–0.3 %. Cost 9 runs ≈ 1.8 A100-h.

## Read (2026-09-13, pq A100; three-member means, seeds 0/1/2) — FAILS the gate under all three weightings
| fold | D | D₀ | rel ratio | tke ratio | mvpe ratio | W̄ (B → SO) |
|---|---|---|---|---|---|---|
| mim1 | −0.011 | −0.009 | 1.002 | 1.001 | 1.018 | 0.7672 → 0.7668 |
| angle05 | **+0.092** | −0.020 | 1.018 | **0.936** | 0.997 | 0.7926 → 0.7964 |
| far_drop | −0.014 | −0.010 | 1.001 | 1.002 | 1.021 | 0.7290 → 0.7286 |
Mean D +0.022 (below +0.025), D₀ −0.013, A₀ −0.022; mim1/far MVPE ratios > 1.005. **The mechanism did not act:** the
per-window prefactor W̄ the objective targets is unchanged on mim1 and far_drop (−0.0004) and rises 0.4 % on angle05 only
through TKE. What the arm did instead is keep 6 % more fluctuation amplitude at the novel angle (tke ratio 0.936, rel
+1.8 %) and pay MVPE +2 % elsewhere — the trained-TKE frontier again, in a new coat. Single models agree (mixed signs,
MVPE worse on 7/9). **Closed:** the per-window SPS-prefactor convexity is not reachable by training on it; the Jensen gap
(0.3–1.5 %) is the ceiling and the loss cannot move W̄ within it. Cost 9 runs ≈ 2 A100-h. The systematic re-read
(`reopen_table.py`) will still show SO with the gain applied, for completeness.
