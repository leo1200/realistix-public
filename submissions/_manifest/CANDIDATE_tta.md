# Screen TTA — input-only test-time adaptation (owner-approved 2026-09-18; pre-registered before the build)

**Mechanism.** At inference the 20 observed frames are a small supervised set about the current regime. An auxiliary
self-supervised task trained into the forecaster — predict frames 11–20 from frames 1–10 (input frames 11–20 replaced by
frame 10 repeated, plus one indicator channel that is 1 for the auxiliary task and 0 for the ordinary 20 → 20 task) —
lets a small adapter (the per-channel affine at every GroupNorm site, ≈ the existing modality-affine shape, reset from a
frozen initial value for every window) take 1 or 3 gradient steps on the observed frames only; the adapted model then
forecasts frames 21–40 and the adaptation is discarded. No target, no cross-window state, no persistent weight change.
It targets the ~90 % shared regime bias (FINDINGS §2) by re-estimating part of the function on the regime at hand —
the private set's mechanism par excellence. Not in FINDINGS §7 (regime *conditioning* is the closed neighbour: that
conditioned the network on a regime *label*; this adapts on the window's own dynamics with a self-supervised loss).

**Arms (folds mim1 / angle05 / far_drop, seed 0; incumbent recipe otherwise; research trainer copy under
`research/tta_2026-09-18/`, the frozen `tools/train_mixed.py` untouched):**
A0 = B member (exists) · **A1 = auxiliary-trained member, no adaptation** (50/50 task sampling, indicator channel;
the decisive control) · **A2 = A1 + adaptation** with 1 step and with 3 steps (Adam, lr registered after a 5-window
pilot on *training* windows, never on held-out ones), adapter = norm affines only, reset per window.

**Reads.** Per-member rel / TKE / MVPE on the fold's held-out stride-40 windows (equal-cell; far_drop equal-Re-row),
Q under the 09-13 weights and exact marginals; auxiliary-task error on the observed frames before/after adaptation
(the mechanism must move its own target); adaptation cost per window (forward + backward + optimiser; A100).
**Gates (fixed now).** Continue (seed 1, then K = 4 + gain + head read): A2 − A1 ≥ +0.020 Q on the fold mean with every
fold ≥ 0, A1 ≥ A0 − 0.010 on the mean (the auxiliary task must not cost the base forecast), MVPE ratio A2/A1 ≤ 1.005
every fold. Kill: A2 − A1 ≤ +0.005 on the mean, or any fold < −0.020, or A1 < A0 − 0.020. Prior (judgement): A2 − A1
0 [−0.05, +0.04]; A1 − A0 −0.005 [−0.03, +0.01]; P(continue) ≈ 0.2. Budget: 3 fold trainings ≈ 1.4 A100-h, reads ≤ 1 h.
Timeline: build + smoke 09-18, folds overnight, read 09-19; a continue leads to seed 1 and the ensemble read by 09-20/21.
This is research after the modelling freeze; a positive result is an owner decision on the package, not an automatic change.

## Read (2026-09-18 20:50 UTC; Codex round 9, `research/audit_2026-09-17/codex/round9/REPORT_full.md`, `summary.json`)
| fold | arm | rel err | TKE err | MVPE err | Q | ΔQ vs A1 | MVPE / A1 |
|---|---|---|---|---|---|---|---|
| mim1 | A0 (B member) | 0.0759 | 0.4855 | 0.0635 | 63.128 | +0.040 | 0.970 |
| mim1 | A1 (aux-trained, no adaptation) | 0.0765 | 0.4905 | 0.0655 | 63.088 | 0 | 1 |
| mim1 | A2 1 step / 3 steps | 0.0869 / 0.0815 | 0.5016 / 0.5127 | 0.0805 / 0.0686 | 62.759 / 62.890 | **−0.330 / −0.198** | 1.23 / 1.05 |
| angle05 | A0 / A1 / A2₁ / A2₃ | | | | 63.802 / 63.798 / 63.409 / 63.588 | +0.004 / 0 / **−0.389 / −0.210** | 0.97 / 1 / 1.42 / 1.15 |
| far_drop | A0 / A1 / A2₁ / A2₃ | | | | 62.137 / 62.062 / 61.818 / 61.900 | +0.075 / 0 / **−0.244 / −0.162** | 0.95 / 1 / 1.16 / 1.03 |
| **mean** | A2 − A1 | | | | | **−0.321 (1 step) / −0.190 (3 steps)**; A1 − A0 **−0.040** | |

**Verdict: KILL on every clause** (continue needed A2 − A1 ≥ +0.020 and A1 ≥ A0 − 0.010). The mechanism *acted* — three
adaptation steps cut the auxiliary loss on the observed frames by 6–11 % on 91–99 % of windows (lr 1e-3, chosen on
training windows: forecast movement ≤ 1.2 % there) — and the improvement transferred **negatively** to the future:
rel +5–8 %, TKE +2–6 %, MVPE +5–42 % worse, on every fold. The auxiliary task itself costs the base member 0.04 Q. The
adapter overfits the observed window's noise and phase and erases learned dynamics, the failure mode the registration
named. Cost: 3 × 0.25 A100-h + 43–101 ms per window for the adaptation (eager; would also break the CUDA-graph path).
A negative for this adapter and budget, not a theorem — but the last legal, untested mechanism of the open round is
now measured, and it points the same way as all the others: the frozen four-member method stands.
