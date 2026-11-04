# Probe P0-F32 — P0's persistence payload returned as float32 from one host buffer (pre-registered 2026-09-17, Codex round 4 + Claude; NOT armed)

**What.** `research/audit_2026-09-17/codex/round4/payload_probe.zip` (copy `build/timing_p0f32_codex.zip`),
sha256 `47ebb223a80821ce69f023d789171117c29a1d7a30f09906b26a6ecf72edd20e`, 1,454 bytes of source, no weights.
Identical to P0 (921738: persistence forecast, ±5 % default band, fp16 rounding on the GPU, one window per call,
stateless, input-only) except that prediction/lower/upper are stacked, upcast to **float32 on the GPU** and returned as
three contiguous float32 views of **one** host allocation. Decoded values equal P0's exactly at N = 1/4/9 (Codex,
`payload_gpu.json`, `payload_a100.json`).

**Why.** The container charged P0 5.338 ms per sample with almost no GPU work, which the record has read as a
payload-independent harness floor (~5.2 ms). That floor was measured with *our* payload (fp16, three arrays). If the
harness casts outputs to float32 or serialises them per call inside the timed region, the dtype/layout of what we return
is part of the clock. Local surrogates say the two directions are both plausible: fp16→fp32 conversion on the CPU costs
≈ 1.1 ms per N = 1 payload (P0-F32 saves 0.65–0.86 ms per call if the harness converts), while fp32 doubles the bytes
(737,280 → 1,474,560) and costs +0.09–0.19 ms per call locally if it does not. A team on a lower payload floor is one
of the few legal explanations of the leader's 81.96 that our modelling cannot reach; this is the single-slot test.

**Prediction (exact power mean).** Accuracy **identical to 921738 to every digit** — rel 91.8687736326 / tke
66.6666666667 / mvpe 89.8859533408 / sps 12.4681202413 (the values are P0's; any deviation > 1e-6 **quarantines** the
read). Clock, three hypotheses: **payload-insensitive** → the current level, 5.83 ms [5.40, 6.25] (or 5.34 if the 09-18
control shows the pre-09-13 level is back); **harness converts to float32 inside the timer** → ≈ 4.7 ms [4.3, 5.2]
(P0's read minus ≈ 0.65 ms); **harness serialises bytes inside the timer** → ≈ 6.4 ms [6.0, 7.0] (+1 ms). The final
score of the probe is irrelevant (≈ 70); only time_score is read.

**Decision rule.** ≤ 5.0 ms → payload-dependent, conversion-bound: build the trained artefact with float32 outputs
(k4a2portable + float32 return) and register it; ≥ 6.0 ms → byte-bound: keep fp16, consider whether any smaller
legal payload exists (none known); between → insensitive within noise, the floor stands, no artefact change.
One read; the interpretation is conditional on the 09-18 control having identified the day's level.

**Slot.** Not before the 09-18 control and the conditional 09-19 `k4a2portable`. Candidate for **09-20** if the
freeze-confirmation artefact is not ready, or for any later Development-Phase day (the board keeps the best, so a probe
costs nothing but the day). Owner's call.

## Disarmed 2026-09-19 05:20 UTC — replaced on the 09-20 slot by `canon0918f32` (the clock proved to be a wide draw, not a level, so the probe's level-based rule cannot be read). Registration retained for a later paired design.
## Armed (owner's go, 2026-09-18 ~04:40 UTC; superseded)
`research/audit_2026-09-17/claude/arm_timing_p0f32.sh` — waits for **2026-09-20T00:00:30Z**, verifies sha256
`47ebb223…dd20e` before sending, retries to 23:45Z; log `logs/submit_timing_p0f32.log`.
Reference level tC for the decision rule: the 09-18 control (6.110 ms) and the 09-19 canon0918 read. **Payload-dependent**
iff tP ≤ tC − 0.5 ms and tP ≤ 4.75 ms (accuracy = 921738's digits within 1e-5); then 09-21 carries the trained artefact
with float32 outputs (one-line change to `k4a2`'s `out_dt`, parity-checked like the portable build). Kill the large-saving
hypothesis if tP ≥ tC − 0.2 ms; keep fp16.
