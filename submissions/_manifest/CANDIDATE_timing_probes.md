# Timing probes P0 / P1 — container fixed-cost measurement (armed 2026-09-10, owner's go: "let's try the time stuff first")

**What they are.** Two deterministic persistence forecasts with explicit default-band bounds (pred ± 0.05|pred|),
produced on the GPU through our real submissions' interface and payload (one H2D, fp16 prediction + two fp16 bound
arrays, three D2H, one window per call). P1 is byte-identical in outputs to P0 and adds 32 identity 3×3
convolutions on a fixed (1, 512, 32, 64) fp16 buffer on the same stream (≈ +3.2 ms of GPU work on an A100-PCIE,
≈ +2 ms expected on the A800). Ordinary reproducible baselines under the rules: input-only, no evaluation data
read, no cross-window information, no state between calls. Zips `build/timing_p0.zip` (1586 B), `timing_p1.zip`
(1626 B); source `submissions/timing_p0/`, `timing_p1/`; standalone smoke (shapes, finiteness, ordered bounds,
persistence, class path, P0 ≡ P1 outputs) passed; kit smoke test run before upload (below).

**Why.** The record priced the container's per-call fixed cost at F ≈ 3.3–3.4 ms from two board points, which
bounds the time channel at ≤ +0.14. On an idle A100-PCIE (pq, 2026-09-10) the incumbent's graphed chain costs
3.98 ms/call and P0 costs 0.40 ms (0.37 GPU-event), so against the board's 5.25 ms/sample the harness overhead on
that card class is ≈ 1.3 ms. Whether our own kernel chain (≈ 3.9 ms) or the harness dominates the timed region
decides whether a fused ~2 ms path is worth ≈ +0.2 in both phases or nothing.

**Model.** Board per-sample time t = F + a·g, with g the local A100 GPU time of the call and a the A800/A100 speed
ratio (expected 0.6–1.0). P0: g0 = 0.373 ms; P1: g1 ≈ 3.6 ms (A100-PCIE, k = 32; exact value in
`research/improve_2026-09-10/claude/timing_probe_a100_k32.log`). Then a = (t1 − t0)/(g1 − g0), F = t0 − a·g0.

**Predictions, recorded before upload.** time_score = 100/(1 + sqrt(t/728.96 ms)).
| scenario | F (ms) | P0 t0 | P0 time_score | P1 t1 (a = 0.8) | P1 time_score | reading |
|---|---|---|---|---|---|---|
| record's two-point inference | 3.4 | ≈ 3.7 | ≈ 93.4 | ≈ 6.3 | ≈ 91.5 | kernel work ≤ +0.14 left; stop engineering |
| A100 proxy | 1.3 | ≈ 1.6 | ≈ 95.5 | ≈ 4.2 | ≈ 92.9 | our chain is ~¾ of the timed region; a 2 ms path ≈ +0.2 |
| near-zero harness cost | 0.5 | ≈ 0.8 | ≈ 96.8 | ≈ 3.4 | ≈ 93.6 | same, larger |
Central expectation F ≈ 1.5 ms [0.5, 3.4]. Accuracy subscores are persistence's and irrelevant (tke exactly
66.67); the final will sit far below the incumbent by design and does not touch the retained best. Falsifiers:
a slope a outside [0.4, 1.5] or F < 0 means the two reads are not on one line (platform drift between days) —
repeat P0 on a third slot before concluding anything. Decision rule: F ≥ 3.0 ms closes kernel work for good;
F ≤ 2.0 ms funds one bounded engineering week on a fused inference path (parity-checked, eager fallback kept),
priced at up to +0.27 for a 2 ms end-to-end call.

**Schedule.** P0 uploaded 2026-09-10 (today's slot was unused); P1 armed for 2026-09-11T00:00:30Z via
`tools/submit_at.sh` (log `research/improve_2026-09-10/claude/submit_p1.log`). Results appended below.

## Upload record
* **P0 → submission 921738**, uploaded 2026-09-10 21:13 UTC (`research/improve_2026-09-10/claude/submit_p0.log`).
  Local leaderboard-style check (`tools/run_and_score.py`, val_aoa, chunk 1, shared H100): time_score 95.575, sps 12.751.
* **P1 armed** for 2026-09-11T00:00:30Z (`submit_p1.log`); local check time_score 93.743, sps 12.751 (identical accuracy).
* A100-PCIE dose measurement at k = 32 (`timing_probe_a100_k32.log`, idle pq card): **g0 = 0.317–0.373 ms, g1 = 3.56 ms**
  (P1 − P0 = +3.2 ms of GPU work), incumbent graphed 3.62–3.93 ms on the same card class. So P1's GPU time is
  about the incumbent's: the pair brackets our operating point. F = t0 − a·g0 with a = (t1 − t0)/3.2 ms.

## P0 scored 2026-09-10 (921738): time_score **92.117 → 5.34 ms per sample** — the same as the incumbent's 5.25
final 69.982 (rel 91.869, tke 66.667, mvpe 89.886, sps 12.468 — persistence, as designed). A call whose GPU work is
0.35 ms is timed at 5.34 ms; the incumbent's, whose GPU work is ≈ 3.9 ms on an A100, at 5.25 ms. So the container's
per-call cost is ≈ 5.0–5.3 ms **independent of our kernels** (F ≈ 5.0 ms, outside the pre-registered [0.5, 3.4]
range on the high side) and the two-point inference of 3.4 ms was itself too low. Reading, pending P1: our own
kernel chain contributes at most a few tenths of a millisecond to the timed region; there is nothing to fuse.
If P1 (+3.2 ms of GPU work) also reads ≈ 5.3 ms, GPU work is invisible to the clock (a ≈ 0) and every past
"time" gain that was not setup relocation was platform drift; if P1 reads ≈ 7.5–8.5 ms (a ≈ 0.7–1.0), the timer
sees GPU work and the incumbent's chain must run far faster in the container than locally — either way the
remaining kernel-side headroom is ≤ +0.03. **The time channel closes on this measurement.**

## P1 scored 2026-09-11 (921760): time_score **91.376 → 6.49 ms per sample**; final 69.914 (accuracy identical to P0)
Slope a = (6.49 − 5.34)/(3.56 − 0.35) = **0.36** board-ms per local-A100 GPU-ms (the A800 plus its harness run our
kind of kernel ~2.7× faster than the A100-PCIE measurement suggests); **F = 5.34 − 0.37·0.35 ≈ 5.2 ms**. Cross-check
on the incumbent: (5.25 − 5.2)/0.37 ≈ 0.1 ms of implied GPU work against 3.9 ms measured locally — i.e. the
incumbent's chain, or most of it, overlaps with the ≈ 5 ms harness cost rather than adding to it. Either way the
pre-registered decision rule fires on the closing side: **F ≥ 3.0 ms → kernel work is closed for good.** Saving
1 ms of local-A100 GPU work is worth ≤ +0.03 of final and the incumbent has < 1 ms visible to the clock.
Falsifier check: slope inside [0.4, 1.5]? No — 0.36 sits below the band, and the incumbent does NOT sit on the
P0/P1 line (predicted 6.6 ms, observed 5.25), so the timing model is not identified and the pre-registered repeat
of P0 on a third slot is owed before the slope is quoted. The robust conclusion needs neither: a call with almost no
GPU work costs the same as the incumbent, so kernel work on the incumbent's chain has at most a few hundredths to
give.
