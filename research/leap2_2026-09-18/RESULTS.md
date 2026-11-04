# Open round, 2026-09-18 — where the SPS oracle gap actually is (`sps_gap_{mim1,far_drop}.json`)

Held-out folds, B members s0–3, the fold A2 student's bounds; per-element reward r = exp(−w/σ)·covered against the
realised-|error| oracle (width 2|e| about the mean) — element-weighted, prefactor omitted.

| group (mim1 / far_drop) | share of elements | reward | oracle | coverage | width / σ | mean \|e\| / σ | share of the oracle gap |
|---|---|---|---|---|---|---|---|
| all | 100 % | 0.68 / 0.59 | 0.84 / 0.77 | 0.88 / 0.81 | 0.31 / 0.40 | 0.12 / 0.19 | 100 % |
| calm cells (bottom 70 % of target variance) | 70 % | 0.79 / 0.71 | 0.92 / 0.88 | 0.95 / 0.92 | 0.20 / 0.27 | 0.044 / 0.074 | **58 / 64 %** |
| wake cells (top 30 %) | 30 % | 0.44 / 0.31 | 0.67 / 0.53 | 0.72 / 0.57 | 0.56 / 0.70 | 0.29 / 0.48 | 42 / 36 % |
| \|e\| < 0.5 σ | 95 / 90 % | 0.71 / 0.65 | 0.88 / 0.84 | 0.92 / 0.89 | 0.28 / 0.35 | 0.07 / 0.10 | ≈ 99 % |
| \|e\| > 1 σ | 1.5 / 3.8 % | 0.00 | 0.05 | 0.02 | 0.95 | 1.6 | **0.6 / 1.2 %** |
| missed elements | 12 / 19 % | 0 | — | 0 | — | — | 33 / 39 % |
| leads 10–19 | 50 % | | | | | | 53 / 52 % |
| target-side placement of the same width (one-sided bound) | | +0.056 / +0.074 over the shipped reward | | | | | |

**Reading.** The gap is not in large-error or bimodal cells: elements with |e| > σ are 1.5–4 % of the field and carry
≈ 1 % of the gap. It sits in the **calm 95 % of elements whose realised error is 0.04–0.10 σ while the interval is
0.2–0.35 σ wide** — three times the realised error. Is that width wrong? For a Gaussian residual of scale s the
reward-optimal symmetric width maximises exp(−u)·erf(u σ /(2√2 s)); at s ≈ 0.055 σ (calm mim1) the optimum is
**u ≈ 0.21 σ — the head's 0.20 σ**. The head is at the optimum for the *error scale it can predict*; the oracle gap is
the randomness of the realised error around that scale, i.e. the forecaster's error itself. Only a lower error, or a
better per-element prediction of the error *magnitude*, moves it — and the archive measured a conditional flow's
spread as a better error predictor than seed spread (0.57 vs 0.51 correlation) yet **−0.14 SPS** when fed to the head.

**Consequence for the leap hypothesis G1 (generative, mode-seeking intervals).** Refuted as the main mechanism: a
one-mode interval helps only where modes are separated by ≫ σ, and those elements carry ~1 % of the gap. The
target-side bound caps everything "centring" could ever buy at +0.06–0.07 reward ≈ +2.5 SPS *with target knowledge*;
our pair/student already harvested the learnable part (2/3 of +0.1 SPS was centring). A generative forecaster could
still pay through TKE amplitude and a better error-scale predictor, but the archive priced both and found them
negative or amplitude-equivalent.

**What remains consistent with 81.96:** a forecaster with lower error (rel −6 %, which also narrows every interval and
raises the SPS prefactor), or a harness/time effect (probe armed for 09-20). Not an interval trick.

## Codex round 7 (`../audit_2026-09-17/codex/round7/REPORT_full.md`) — corrections and the merged build order
* **My arithmetic was wrong by 7×:** a quarter of the W̄ gap (0.61 → 0.84) is ≈ +4.1 SPS at R̄ = 0.716, not +0.6; the
  leader needs only **+0.64 SPS = 3.9 % of that gap**. The interval channel has ample absolute headroom; what my
  decomposition showed is only that the *far-tail / bimodal* story carries ≈ 1 % of it and that the current head is at
  the Gaussian optimum for the error scale it predicts. Codex's independent read on the gained/guarded path agrees
  (|e| > σ carries 0.56 / 0.07 / 1.21 % of the gap; a crude four-member mode selector loses 0.27 / 0.21 / 0.41 SPS on
  every cell; target-side placement at the current width is +4.3 / +3.2 / +5.3 SPS as an *oracle*).
* **What survives:** learning a better *conditional distribution* of the residual from the raw window — not more
  post-processing of the same predictions. Two cheap-to-medium tests, in order: **(1) a marginal mixture / CDF head**
  (3 logistic or Gaussian components per element, free centres, NLL or CRPS, then the reward-optimal contiguous interval
  from the analytic CDF; interval-only, point forecast and clock unchanged; ≈ 1 day, 0.5–1.5 GPU-h; prior ΔSPS −0.2 …
  +0.6); **(2) a bounded full-field conditional-flow screen, interval-only** (K = 8–16 samples × 4–8 Euler steps ≈
  8–23 ms/window graphed on H100 — *not* free if the clock charges it; prior ΔSPS −0.5 … +0.8; P(+0.3 SPS by 09-24)
  10–20 %). **(3) input-only test-time adaptation** as the other genuinely different idea (1–2 days). The 09-20 payload
  probe stays the independent harness discriminator.
* Gates (Codex, registered): continue to a second seed only at mean interval-only ΔSPS ≥ +0.15 with every fold ≥ 0;
  kill ≤ +0.05 or any fold < −0.10; deployment only at two seeds ≥ +0.30 and a private-geometry split with the same sign.
* None of this changes the 09-20 modelling freeze or the 09-25 package: the frozen method is the deliverable; these are
  research arms whose success would be a separate owner decision.

## D1 read (2026-09-18): KILL — ΔSPS −0.31 / −0.42 / −0.31 vs A2, 0/34 cells up (`../density_2026-09-18/RESULTS.md`). A learned conditional density with free centres does not beat the reward-trained width head; the interval channel's headroom is not reachable by a better marginal law from the same inputs.

## FM1 read (2026-09-18, Codex round 8): KILL — sample-CDF intervals from an 8000-step conditional flow read −1.87 / −1.26 / −2.14 SPS vs A2, all cells negative, K32/S8 still −1.7 (`CANDIDATE_fm1.md`). **The open round's conclusion:** three routes to the interval channel — far-tail/bimodal (≈ 1 % of the gap), a learned marginal law (D1, −0.35), a learned joint law (FM1, −1.75) — all lose to the reward-trained width head. The leader's +0.15 is therefore most consistent with a better point forecaster (which also lifts SPS through narrower intervals and the prefactor) and/or a harness effect (probe 09-20). Input-only test-time adaptation remains the one unmeasured idea.

## TTA read (2026-09-18 20:50, Codex round 9): KILL — adaptation cuts the auxiliary loss 6–11 % and raises future rel/TKE/MVPE on every fold (A2 − A1 −0.32 / −0.19 Q; MVPE up to +42 %); the auxiliary task costs the base 0.04 Q (`CANDIDATE_tta.md`). The open round is closed: four routes, four registered kills.

## Two last target-free checks (2026-09-18 evening; `overlooked_<fold>.json`)
| check | mim1 | far_drop | angle05 | verdict |
|---|---|---|---|---|
| MVPE: replace the forecast's time-mean by the input window's time-mean (persistence) | −0.51 Q (MVPE 0.062 → 0.086) | −0.34 | −0.62 | the model's slow mode is far better than persistence |
| 25 % blend toward the input mean | −0.032 | +0.006 | −0.046 | no |
| TKE: add white noise with the input window's noise-floor variance (calmest 20 % cells), dose 0.5 / 1 / 2 | −0.001 / −0.002 / −0.003 | −0.001 / −0.001 / −0.002 | −0.001 / −0.003 / −0.005 | the PIV noise floor is 0.1–0.5 % of the target variance — nothing to restore |
