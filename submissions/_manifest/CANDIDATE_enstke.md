# Candidate enstke — loss constants re-selected at ENSEMBLE level (owner's idea, 2026-09-11 night; gates locked by Codex round 6 BEFORE any read)

**Screen (running on pq A100).** Arms T07 / T15 / T20 (w_tke 0.7 / 1.5 / 2.0), S05 (w_sps 0.5) vs B (incumbent recipe);
seeds 0/1/2; folds mim1, angle05 (sim dropped), far_drop; EMA step 6000; predictions averaged BEFORE scoring;
cell-balanced, far Re-row-balanced; S(e) = 100/(1 + e/2) after aggregation. Reader: `research/audit_2026-09-11/claude/
{read_ens_screen.sh, ens_summary.py}`. Closed neighbour: per-member dose sweeps (w_tke peak ≈ 1.15 worth +0.001;
w_sps ≥ 2 damps amplitude). What differs: the dose is selected on the three-member mean.

**Quantities per arm and fold.** D = .4738 ΔS_rel + .1057 ΔS_tke + .0926 ΔS_mvpe (ensemble); D₀ = D without the TKE term
(not renormalised); A / A₀ = the same through `weights.acc_effective` (SPS prefactor at K = 0.611, derivatives at each
B ensemble's fold subscores); per-seed paired member D with mean M, M₀; compression C = D/M, C₀ = D₀/M₀ (|denominator|
< 0.005 → "unresolved"); channel ratios r = e_arm/e_B; e_ensemble / mean_seed(e_member) per channel; far-row deltas.

**Promotion to the thirteen-fold confirmation requires ALL of:** three-fold means D ≥ +0.025, D₀ ≥ +0.025, A₀ ≥ +0.040;
every fold D ≥ −0.020 and D₀ ≥ −0.010; mean rel and MVPE subscore changes each ≥ 0; each fold's rel and MVPE error ratio
≤ 1.005 and far MVPE ≤ 1.000; predicted TKE-free SPS-prefactor change mean ≥ 0 and every fold ≥ −0.05 sps points; all of
this under the locked weights, DELTA, and the 34-card fit (.462/.252/.102/.111); scorer-native mean D₀ > 0.
**A D > 0 financed by TKE while D₀ misses is a failed gate.** At most one passer advances (highest worst-fold A₀, then
mean A₀, then T07/T15/T20/S05); no interpolated dose, no mixed-dose ensemble, no seed selection.
TKE dose response counts as resolved when T20 vs T07 moves ensemble TKE error ≥ 1 % in the same direction on ≥ 2 folds.

**Confirmation (only for a passer).** Fresh members 11/12/13, honest heads 0/11 fitted on outer-training data, the
thirteen manifests with family weights ¼ each; N (full non-time incl. learned SPS once) and N₀ (TKE zeroed, candidate's
per-window TKE prefactor replaced by B's) both ≥ +0.04, every family ≥ 0, every fold ≥ −0.02, for both head seeds and all
weightings; rel/MVPE/SPS family deltas ≥ 0; learned SPS loses ≤ 0.05 per fold. A slot additionally needs N₀ ≥ +0.06 and
the adverse scenario (board TKE −0.264 incl. its SPS cost ≈ −0.046) printed. Follow-up ladder if a passer exists:
w_sps 0.5/1.5, fixed step 4000/5000, EMA, w_rel, w_mvpe, Muon wd — one coordinate at a time, same gates, 3.6 A100-h per pair.

**Prediction (conditional on a confirmed three-member arm; Codex round 6 text):** non-time Δ +0.06 [−0.04, +0.10];
total Δ +0.06 [−0.08, +0.14] vs 918211; TKE receives no promotion credit, including through SPS. Kill deployment if board
non-time or the registered TKE-stripped proxy H₀ ≤ 0 under any weighting, rel/MVPE/SPS regress, or time consumes the
gain. If K4 becomes the incumbent first, the confirmation must be re-run as paired four-member ensembles. Slots: K4
(armed 09-12), 09-13 incumbent repeat, 09-14 P0, 09-15 meanfix, 09-16 final, 09-17 cc11, 09-18 cc11h; enstke may take
09-19 only after qualification.

## Read (2026-09-12 ~00:00 UTC, `readout/ens_screen_read.log`, `readout/ens_summary.txt`) — ALL ARMS FAIL, axis closed at ensemble level
Three-member means at step 6000 (cell-balanced; ensemble error ratios vs B, D = weighted ΔQ, D₀ = TKE zeroed):
| arm | fold | rel ratio | tke ratio | mvpe ratio | D | D₀ |
|---|---|---|---|---|---|---|
| T07 (w_tke 0.7) | mim1 / angle05 / far | 0.986 / 1.013 / 0.987 | 1.013 / 0.951 / 1.028 | 1.012 / 1.001 / 1.022 | −0.003 / +0.070 / −0.032 | +0.017 / −0.014 / +0.022 |
| T15 (1.5) | | 0.999 / 0.999 / 1.014 | 1.035 / 1.007 / 0.987 | 1.012 / 0.997 / 1.011 | −0.060 / −0.011 / −0.011 | −0.008 / −0.001 / −0.034 |
| T20 (2.0) | | 1.017 / 1.014 / 1.018 | 1.011 / 0.972 / 1.005 | 1.016 / 1.000 / 1.010 | −0.050 / +0.033 / −0.052 | −0.032 / −0.015 / −0.043 |
| S05 (w_sps 0.5) | | 0.991 / 1.002 / 1.001 | 1.039 / 0.997 / 0.994 | 1.003 / 1.000 / 1.014 | −0.050 / +0.002 / +0.004 | +0.007 / −0.002 / −0.001 |
Means: T07 D +0.012 / D₀ +0.008 / A₀ +0.010; T15 −0.027 / −0.012 / −0.018; T20 −0.023 / −0.030 / −0.047; S05 −0.015 / +0.002 / +0.001
(identical verdicts under locked, DELTA and 34-card weights). Gate misses everywhere: rel/MVPE ratio floors (T07 far MVPE
+2.2 %, T20 rel +1.7 %), fold floors (T15 mim1 −0.060), means far below +0.025.
**Mechanism, measured:** per member, w_tke 1.5/2.0 cut TKE error by 0.005–0.013 on every seed and fold; the three-member
mean does not inherit it (ensemble TKE ratios 0.97–1.04, inconsistent in sign across folds: T20 vs T07 −0.2 % / +2.2 % /
−2.3 %), while the rel cost (+1–3 %) survives averaging intact. The TKE dose response is unresolved-to-absent at ensemble
level; the loss weight moves member amplitude that averaging removes. Lower w_tke (T07) buys rel −1.4 % and pays TKE and
MVPE. w_sps 0.5 is neutral. **Closed: re-selecting w_tke or w_sps on the ensemble does not pay; no confirmation, no slot;
the follow-up ladder (step, EMA, w_rel, w_mvpe, wd) is not opened** — its premise was a TKE slope that does not exist.
Cost: 39 A100 runs ≈ 8 A100-h.
