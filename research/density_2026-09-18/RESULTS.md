# D1 — marginal mixture-density interval head (2026-09-18; registered gates in `research/leap2_2026-09-18/RESULTS.md`)

Three-component logistic mixture per element (logits, free centres, log-scales in σ units; scale floor 0.02 σ), same trunk
and inputs as the shipped head, NLL on the residual about the plain mean, 4000 steps, lr 8e-4, seed 0, outer-training
windows of each fold; the interval is the reward-optimal contiguous segment from the analytic CDF over 26 widths ×
(k+1) centres × 5 shifts, free to exclude the point forecast. Read against the fold A2 student on the held-out stride-40
windows with the gained forecast fixed (`read_density.py`, `read_<fold>.json`).

| fold | ΔSPS vs A2 (element-weighted) | equal-cell Δ | coverage D1 / A2 | mean width σ D1 / A2 | cells up |
|---|---|---|---|---|---|
| mim1 | **−0.310** | −0.308 | 0.870 / 0.883 | 0.296 / 0.310 | 0 / 10 |
| angle05 | **−0.418** | −0.413 | 0.932 / 0.940 | 0.201 / 0.203 | 0 / 14 |
| far_drop | **−0.315** | −0.316 | 0.795 / 0.813 | 0.378 / 0.400 | 0 / 10 |
| mean | **−0.348** | | | | 0 / 34 |
Guarding the density intervals to contain the gained forecast changes nothing (−0.33 / −0.42 / −0.33).

**Verdict: KILL** (gate: continue ≥ +0.15, kill ≤ +0.05). The plug-in decision from a likelihood-trained density is
systematically slightly too narrow (coverage −1.3 to −1.8 points for 1–5 % less width) — the same lesson the head's
history taught in August: proxies trained on anything but the reward lose to a reward-trained width, because the
reward's asymmetry (coverage lost outright vs width paid smoothly) is not what NLL prices. A reward-trained mixture
would collapse toward the shipped objective. Learned conditional shape (free centres, three components) did not pay on
any of 34 cells: no evidence of exploitable multimodality or skew in the residual law at this resolution. Closed.
