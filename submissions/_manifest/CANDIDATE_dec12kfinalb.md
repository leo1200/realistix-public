# Candidate dec12kfinalb — dec12kfinal with cuDNN benchmark-mode algorithm selection (registered 2026-09-25 22:05 UTC, before upload)

**What it is.** `submissions/dec12kfinalb/` = `submissions/dec12kfinal/` (the contract's CHAMPION) plus two lines at import:
`torch.backends.cudnn.benchmark = True` and `torch.backends.cudnn.deterministic = True` — each convolution's cuDNN
algorithm is chosen by timing the (deterministic) candidates once during setup, before any graph is captured; the
graphs replay the choice. Owner's call (2026-09-25): try it on the board.

**Measured.** A100, paired fresh-process lifecycle (contended node, 8 rounds): call 2 **4.55 vs 4.73 ms** (−0.18; a quieter
node read −0.4 on 09-24), import **+3.6 s (+19 %)**. Held-out r3joint parity vs dec12kfinal (292 windows): rel +2e-6 /
tke +9e-6 / mvpe −1.8e-5 / sps −2.7e-4; max |Δ| 4.9e-4 / 7.3e-4 (fp16 ulps). Outputs can differ **between processes**
by fp16 rounding (the algorithm choice is timed per process) — the standalone zip reads bitwise-different from a
second process, as expected. Validation battery: see the read below.

**Why only the board.** Not adopted into the retrain contract: it lengthens every fresh process (≈ +0.8 s on the A800 →
≈ 24 instead of 26 processes per 180 s) and gives up cross-process bitwise reproducibility, for a few thousandths.

**Prediction.** Accuracy = 943854's digits within ≈ 3e-4; our call ≈ 4.4–4.6 ms in the log; import ≈ 5.3 s. F is the
platform's clock draw: 81.945 at 09-25's harness share, 81.969 at 09-24's, **81.957 [81.938, 81.977]** at ± 1 sd — about
+0.003 over the same draw of dec12kfinal. New best iff t < 4.187 ms/sample. **Kill:** accuracy off by > 1e-3,
`fused=False`, call 2 ≥ 6.2 ms, or import ≥ 7 s in the log.
