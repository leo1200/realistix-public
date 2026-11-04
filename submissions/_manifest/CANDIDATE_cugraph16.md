# Candidate: cugraph16 — fp16 modules in channels_last, graph-replayed (2026-09-04)

**What it is.** 914270's `model.pth` byte-for-byte (md5 `137325e2…`) and its CUDA-graph
`submission.py`, with one change in `_load()`: on the GPU the three members and the head
are cast to fp16 and laid out channels_last, and the input is packed channels_last to
match. Nothing else moves. Off CUDA everything stays fp32.

**Why.** 914270 read 5.69 ms/sample with graphs against 6.65 without, a ratio of 0.86,
which together with the earlier points identifies the container regime as **one window
per call**: the clock is the GPU-side chain of kernels, not CPU launch cost. A kernel
profile of one member at one window shows ~300 kernels for 1.1–1.7 ms of GPU time, of
which 60–66 are the NCHW→NHWC layout conversions cuDNN inserts to reach its
convolution kernels; channels_last removes them, and fp16 arithmetic trims the rest.
The fp16 cold-shape penalty that sank 891095 does not apply because every shape is
captured at import.

**Measured, one-window graph replay, ms per call:**

| | cugraph (914270) | cugraph16 |
|---|---|---|
| H100 | 3.57 | **3.10** (0.87×) |
| A100, end-to-end replay | 4.12 | **3.49** (0.85×) |
| A100, three members alone | 2.95 | — |

Fresh process, first call, H100: N=1 3.9 ms, N=2 5.9 ms, N=187 1.18 ms/sample; setup
6.8–7.7 s (unchanged). `tools/run_and_score.py` on `val_aoa` at chunk 64 **and** at
chunk 1: rel_l2 95.474 / tke 83.272 / mvpe 95.815 / sps 51.141 — identical to 914270's
local read to three decimals. Outputs differ from 914270's by 1.8e-4 to 4.9e-4 absolute
(one to two fp16 ulps of the returned arrays; 4e-4 relative), the arithmetic being fp16
now rather than fp32 on fp16-rounded weights. Zip `build/cugraph16.zip`, 162.0 MB
extracted, smoke-tested standalone through both interfaces.

**Prediction, recorded before the score.** Against **914270's 81.73109 (time 91.88201,
5.690 ms/sample)**, day-adjacent, weights identical. Scaling the container's 5.69 by the
measured ratio 0.85–0.87 gives 4.8–5.0 ms/sample → `time_score` 92.4–92.3 →
**+0.05 to +0.06 of final**. If the container's remaining time is partly host work the
change does not touch, the gain is smaller. **Central +0.05, band [0.00, +0.10]** → board
**81.78 [81.73, 81.83]**. Accuracy subscores must come back within **0.005** of 914270's
on every channel (the arithmetic changed by 4e-4 relative; locally the subscores did not
move at three decimals).

**Falsifier.** `time_score` at or below 91.88 with accuracy inside tolerance: the
container's clock is not the kernel chain we measured, and the kernel-level track
closes. Accuracy outside 0.005 on any channel: fp16 arithmetic is not free on the hidden
set; revert to the fp32 graphs (914270).

**Armed 2026-09-04 for 2026-09-05T00:00:30Z.** Log `submit_cugraph16.log` next to the
runs directory; zip md5s: submission.py `249fd785…`, model.pth `137325e2…`.

## Outcome — 915299, scored 2026-09-05T00:00Z

**81.75365 against 914270's 81.73109: +0.023, inside the band [0.00, +0.10], below the
+0.05 central. New best; rank 2 of 50, 0.008 behind np-user.**

| | 914270 | 915299 | Δ |
|---|---|---|---|
| rel_l2 | 94.82381 | 94.82385 | +0.00003 |
| tke | 79.05878 | 79.05839 | −0.00039 |
| mvpe | 93.91028 | 93.91058 | +0.00030 |
| sps | 43.74379 | 43.74508 | +0.00129 |
| time | 91.88201 | **92.11282** | +0.231 (5.69 → 5.34 ms/sample) |

Accuracy inside the 0.005 tolerance on every channel: fp16 arithmetic is free on the
hidden set. The time ratio is **0.94** where the GPU chain shrank by 0.85–0.87 locally
(A100: 4.12 → 3.49 ms, −0.63; container: −0.345). Read as `t = fixed + chain`: the
container's chain is about 55 % of our loaded A100's, ≈ 2.0 ms, and **≈ 3.4 ms of every
call is fixed overhead our kernels never touch** — host-side work inside the harness's
timer (the input and the three output arrays cross a process boundary per call). Two
consequences: the fused-kernel candidate armed for 09-06 is worth about +0.02, not
+0.06 (its manifest is revised below the line, before its score); and the remaining
time pool is small — a network costing nothing would read ≈ 3.4 ms, `time_score` 93.6,
about +0.15 of final.
