# Screen div4 — diversity-favouring member settings read at K = 4 (owner's go 2026-09-12; gates fixed BEFORE the read)

**Mechanism.** An ensemble can prefer members that are individually worse but less correlated. Settings rejected at
member level because they raise member variance are the ones whose ensemble-level optimum can differ: EMA 0.995 and
0.99 (vs 0.998; shorter averaging window → noisier members) and Muon weight decay 0.01 (vs 0.03; less shrinkage).
Closed neighbours: EMA 0.998 selected per member (08-28); wd 0.03 selected per member (09-07, board tke −0.264).
**Arms** E995, E990, W01 vs B; seeds 0/1/2/3; folds mim1, angle05 (sim dropped), far_drop; step 6000; scored as
four-member means (and three-member for comparison) with `ens_read.py` / `ens_summary.py`-style quantities.
**Gate (same as `CANDIDATE_enstke.md`, pre-registered by Codex round 6):** three-fold means D ≥ +0.025, D₀ ≥ +0.025,
A₀ ≥ +0.040; every fold D ≥ −0.020 and D₀ ≥ −0.010; mean rel and MVPE subscore changes ≥ 0; per-fold rel and MVPE error
ratios ≤ 1.005, far MVPE ≤ 1.000; TKE-free SPS prefactor mean ≥ 0 and every fold ≥ −0.05; identical verdict under the
locked, DELTA and 34-card weights. A passer goes to the thirteen-fold confirmation (fresh seeds, honest heads) before any
slot. Cost 37 runs ≈ 7.4 A100-h. Prior: +0.01 [−0.03, +0.03] per arm; one-in-four chance of a real positive.

## Read (2026-09-12, pq A100; `~/q/spool/logs/*div4read*.out`) — ALL ARMS FAIL at K = 4 and K = 3, under all three weightings
Four-member means, D vs B (mim1 / angle05 / far_drop; mean):
| arm | D | D₀ | A₀ | note |
|---|---|---|---|---|
| EMA 0.995 | +0.001 / −0.002 / −0.008 (−0.003) | −0.006 | −0.010 | flat; MVPE slightly worse everywhere |
| EMA 0.99 | −0.002 / −0.004 / −0.012 (−0.006) | −0.008 | −0.013 | flat-to-negative; rel ratio 1.005 at angle05 |
| wd 0.01 | −0.052 / −0.068 / +0.009 (−0.037) | −0.003 | −0.007 | TKE +3.8 % / +4.0 % worse on mim1 / angle05 |
Ensemble TKE ratios for the EMA arms are 0.997–1.000: shorter averaging does not make the members more useful to the mean;
member diversity from EMA is not the diversity averaging needs. Lower decay reproduces the per-member wd finding with the
sign the board showed (wd 0.03 → less TKE error). **Closed: diversity-favouring member settings do not pay at K = 4.**
Side read: B at K = 4 vs K = 3 = −0.007 / +0.026 / −0.013 direct — the fourth member's direct accuracy effect is mixed by
fold, consistent with the sentinel; its board gain came through SPS. Cost 39 runs ≈ 8 A100-h.
