# Candidate: cugraph — CUDA-graph replay of the bucketed batch shapes (2026-09-03)

**What it is.** 912717's `model.pth` byte-for-byte (md5 `137325e2…`, the 911563 /
nospreadhead pack) with one change to `submission.py`: at import, after the same
`_load()` and eager warm-up 912717 does, each bucketed batch shape is captured as a
CUDA graph — 1, 2, 3, 4, 6 and 8 windows with the three members replayed concurrently
on three streams and the head after them; 16, 32 and 64 as one sequential graph — and
`predict()` replays the graph for its call size (padding by repeating the last window
up to 8; exact bucket sums above 8, because padding a compute-bound batch is real
work). A self-check at setup compares the smallest and largest bucket against the
eager path; any exception in capture, the self-check or replay leaves the eager path
in charge, which is exactly 912717. The class interface is kept.

**Why.** `docs/archive/FINDINGS_2026-09-10.md` §9c. After 912717 the container reads 6.65 ms/sample. Locally
the same `predict()` is 1.1 ms/sample on a large batch and **9–18 ms per call** on one
to eight windows — flat in N — because the forward is ~640 small kernels and below
~16 windows the clock is the CPU launching them. The container's number is what two
or three windows per call looks like on an A100-class node. Graph replay launches the
identical kernels from one call.

**Measured, warm, ms per call (eager → graph):**

| windows | H100 | A100 (idle) | A100 (loaded node) |
|---|---|---|---|
| 1 | 13.0 → **3.6** | 18.1 → **4.4** | 19.8 → 7.5 |
| 2 | 13.0 → **4.1** | 18.4 → **5.3** | 21.8 → 8.9 |
| 3 | 9.8 → **4.9** | 18.9 → **6.3** | 22.0 → 11.2 |
| 5 | 10.4 → 6.0 | — | 25.2 → 13.8 |
| 8 | 11.6 → 8.1 | 20.0 → 11.5 | — |
| 9 / 17 / 33 / 65 / 100 / 187 | 1.0 / 0.9 / 0.9 / 0.9 / 0.9 / 1.0 × | — | 1.1 / 0.8 / 1.1 / 1.3 / 1.0 / 1.1 × |

Fresh process, one call (the board's shape), A100: N=2 first call **8.3 ms against
76.8 ms** eager (the shipped code pays ~65 ms for the first call at every batch shape
it was not warmed at; the graphs are all pre-warmed). Setup 7.5–8.9 s per fresh
process against 5.2–5.4 s; 1.65 GB of device memory.

**Outputs.** Identical to the eager path to one fp16 ulp: max |Δ| ≤ 2.44e-4, 0–10% of
elements differing by one ulp depending on N (cuDNN picks per-batch-shape algorithms;
the padded shapes differ from the eager shapes). `tools/run_and_score.py` on `val_aoa`
at chunk 64 and at chunk 2: rel_l2 95.474 / tke 83.272 / mvpe 95.815 / sps 51.141,
identical to 912717's local read to three decimals; time 96.197 and 94.467 locally.
Zip `build/cugraph.zip`, 162.0 MB extracted, smoke-tested standalone from a temp dir
through both interfaces and with a float64 input.

## Prediction, recorded before the score

Against **912717's 81.67313 (time 91.28116, 6.651 ms/sample)**, day-adjacent, weights
identical, accuracy pinned by construction:

| container regime | gain ratio (graph/eager) | ms/sample | time_score | Δ final |
|---|---|---|---|---|
| 2–3 windows per call (the reading) | 0.30–0.50 | 2.0–3.3 | 94.9–93.7 | **+0.24 to +0.36** |
| 1 window per call | 0.38–0.42 | 2.5–2.8 | 94.4–94.2 | +0.29 to +0.31 |
| 8 windows per call | ~0.6 | ~4.0 | 93.3 | +0.20 |
| ≥ 32 windows per call | 0.9–1.0 | 6.0–6.7 | 91.3–91.6 | 0.00 to +0.03 |
| capture fails in the container → eager fallback | 1.0 | 6.65 | 91.28 | 0.00 |

**Prediction: +0.25 central, band [0.00, +0.40]** → board **81.92 [81.67, 82.07]**.
The board read also *measures* the container's call size through the gain ratio, which
is the second thing this slot buys. Accuracy subscores must come back within **0.002**
of 912717's on every channel (one fp16 ulp on a few percent of elements is the only
difference allowed); anything larger means the archive is not what it claims.

**Falsifier.** `time_score` at or below 91.28 with accuracy unchanged: the container
is not launch-bound (or capture fell back to eager, which the board cannot
distinguish). Either way the small-call model of §9c is wrong as stated and the axis
closes on this submission; revert to 912717's code.

**Tail risk, stated.** A capture or replay fault that is *sticky* (a CUDA illegal-access
error poisons the context) would fail the run rather than fall back — the eager
fallback catches Python exceptions, not a dead context. Both cards tested here (H100,
A100-PCIE, torch 2.2.2+cu121, the container's own pinned stack) capture and replay
cleanly, including a standalone import from an unpacked zip. The board keeps the best,
so the cost of that failure is one slot. Setup +2.5 s per fresh subprocess is inside
the 180 s wall unless the harness spawns more than ~40 subprocesses.

**Armed 2026-09-03 08:52Z for 2026-09-04T00:00:30Z** on the owner's go, after a rules
check: no rule touches inference-side graph capture; every prediction depends only on
its own window (padding rows are copies from the same call and are discarded); outputs
are deterministic to one fp16 ulp; setup runs at construction, which the FAQ places
outside the timed region. Log `runs/../submit_cugraph.log`; zip `build/cugraph.zip`
(submission.py md5 `1d19ba5b…`, model.pth md5 `137325e2…`). The command used:

```bash
RETRIES=3 setsid nohup bash tools/submit_at.sh 2026-09-04T00:00:30Z \
    build/cugraph.zip "cugraph" \
    &> submit_cugraph.log &
```

## Outcome — 914270, scored 2026-09-04T00:00Z

**81.73109 against 912717's 81.67313: +0.058, inside the band [0.00, +0.40], well below
the +0.25 central.** New best; rank 3, 0.030 behind np-user and 0.016 behind doomduke2.

| | 912717 | 914270 | Δ |
|---|---|---|---|
| rel_l2 / tke / mvpe / sps | identical | identical | 0 |
| time | 91.28116 | **91.88201** | +0.601 (6.651 → 5.690 ms/sample) |

Accuracy identical to the last digit, as required. The gain ratio graph/eager is **0.86**.
In the prediction table that sits between "8 windows per call" (0.57–0.62) and "≥ 32"
(0.9–1.0) *if* the container's eager cost scaled like our nodes'. It does not: the row
that fits all three numbers at once — 6.65 ms/sample eager, 5.69 with graphs, and the
2.27 ms/sample of setup 912717 removed — is **one window per call on a fast, unloaded
CPU**. Eager at N=1 is CPU launch cost, ~640 kernels at ~10 µs each ≈ 6.6 ms on an idle
modern core (our shared nodes measured 12–18 ms because they were loaded); graph replay
removes the launch cost and leaves the GPU-side latency floor of ~640 kernels executed
in sequence per member chain plus the head chain, which is 4.4 ms on an idle A100, plus
~1 ms of host-to-device copy, three device-to-host copies and Python. The prediction's
mistake was assuming the container's eager cost scaled from our loaded nodes; the band
was wide enough to hold, the central was not.

Consequences: the clock is now the *GPU-side kernel latency chain*, not the CPU. Fusing
the three members into one grouped network does not shorten that chain — they already
run concurrently on three streams, so the critical path is one member's kernels plus the
head's. What would: fewer kernels on that path (fused GroupNorm+SiLU, conv epilogues),
faster per-kernel algorithms at batch 1 (cuDNN benchmark mode before capture), and one
device-to-host copy instead of three.
