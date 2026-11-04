# Candidate k4a2f32 — 929146's artefact returning float32 arrays (pre-registered 2026-09-18; CONDITIONAL on the 09-20 payload probe)

**What.** `submissions/k4a2f32/`: `model.pth` byte-identical to `submissions/k4a2/model.pth`; `submission.py` differs
from k4a2's in two lines — the fp16 output tensors are upcast to float32 on the GPU before the host copy (graphed and
eager paths). Values are the same fp16-rounded numbers, so every decoded output equals 929146's bit for bit (verified
below); only the returned dtype (and the payload size, 737,280 → 1,474,560 bytes per sample) changes.

**Why.** If the harness converts fp16 outputs to float32 inside the timed region (≈ 1.1 ms per sample on a CPU), a
float32 payload removes that cost; if it serialises bytes, it adds ≈ 0.1–0.2 ms. The 09-20 P0-F32 probe
(`CANDIDATE_timing_p0f32.md`) decides which: this artefact is uploaded on 09-21 **only if** the probe reads
tP ≤ tC − 0.5 ms and ≤ 4.75 ms with P0's accuracy digits.

**Prediction (exact power mean).** Accuracy = 929146's digits (rel 94.8293790855 / tke 79.1320014641 / mvpe
94.1013126942 / sps 43.9453064676; any departure > 1e-5 quarantines the read as a build fault). Clock: the probe's
saving applied to the day's level, centre tC − 0.6 ms [tC − 1.1, tC − 0.3]; at tC = 6.0 → 5.4 ms → **final ≈ 81.83**
[81.80, 81.86] — a new best iff the clock draws below 5.717 ms. Kill: clock ≥ tC − 0.2 (no payload effect on the
trained artefact even though the probe showed one — then the cost was probe-specific); accuracy quarantine as above.

**Verification (2026-09-18, local H100).** Decoded outputs identical to k4a2 at N = 1 / 3 / 8 / 48; float32 dtypes;
nine graphs; N = 1 median latency within noise of k4a2 (below). Zip `build/k4a2f32.zip`.
| local H100, N = 1 median (200 calls, same process) | k4a2 3.130 ms | **k4a2f32 3.387 ms** (+0.26 ms: three GPU upcasts + twice the device-to-host bytes) |
|---|---|---|
Zip sha256 `4d6ede18eb4ebfa8a80336f6bac5d0bff4855d84f3be4b20ade6f895d275e8cb` (191.6 MB; 207.4 MB extracted; two files).
The +0.26 ms local cost is the same structure the P0-F32 probe carries, so the probe's board reading already nets it
against any harness saving; the trained artefact adds nothing the probe does not.
