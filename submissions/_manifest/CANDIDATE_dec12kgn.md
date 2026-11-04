# Candidate dec12kgn — dec12kfast + a fused GroupNorm+SiLU kernel in the captured graphs (registered 2026-09-24 16:40 UTC, before any upload)

**What it is.** `submissions/dec12k/model.pth` byte for byte; `submissions/dec12kfast/submission.py` with one
inference-only change, built reproducibly by `research/last_2026-09-24/claude/build_gn.py` from
`research/last_2026-09-24/claude/fastgn.py`:
* Every `silu(group_norm(x))` pair in the four members and the interval head (29 per member, 15 in the head — every
  GroupNorm in both networks is followed by a SiLU) runs, **inside the captured CUDA graphs only**, as a custom CUDA
  kernel on the fp16 channels_last activations: one block per (window, group), fp32 shifted moments, fixed-order
  warp/block reductions (bitwise repeatable), then scale/shift + SiLU in one pass with one fp16 rounding; for the one
  5×192×32×64 site a two-kernel chunked version (fixed-order, no atomics). The eager path stays torch's native ops.
* No compiler at evaluation time: the kernel source lives in `submission.py`; the archive ships the cubins NVRTC 12.1
  (torch 2.2.2's own `nvidia-cuda-nvrtc`) compiled from that exact source (`fastgn_<sha12 of source>_sm80/sm90.cubin`);
  at import the cubin for the device's arch is loaded through the CUDA driver API (libcuda), else the source is
  compiled with NVRTC, else the graphs are captured with native ops. Kernels are launched with `cuLaunchKernel` on
  torch's current stream, so they are captured into the graphs like torch's own kernels. Every bucket's fused graph is
  checked against the native eager path on synthetic data at import (tolerance 2e-3, the native graphs' own bar;
  measured 2.4e-4…3.7e-4 = one fp16 ulp on H100 and A100); any failure → that bucket and all later ones are captured
  natively (= dec12kfast).

**Why it pays.** torch 2.2's GroupNorm on channels_last fp16 costs 66–80 µs at 5×96×32×64 on an A100 where the 3×3 conv
beside it costs 17 µs: GN+SiLU was 1.17 of a member's 2.26 ms. Fused (A100, graph-replayed, B = 5): 80→26 µs at
C96 32×64, 74→32 at C192 32×64, 28→14 at C192 16×32, 20→9 at C384 8×16 (`mb_fastgn.py`, pq logs mbfastgn2).

**Measured (A100-PCIE-40GB, node7, fresh process per trial, call 2 = the harness's timed call, `callseq2.sh`):**
| round | dec12kfast | dec12kgn |
|---|---|---|
| 1–5 | 7.86 / 7.84 / 7.56 / 7.85 / 8.07 | **5.58 / 5.89 / 5.28 / 5.71 / 5.31** |
Steady state (`profile_gn.py`): members on 4 streams 5.38 → 3.48 ms, post graph 1.67 → 1.31 ms, full predict 8.28 → 5.99 ms.
Import time unchanged (12.7 vs 13.5 s on the pq node); cubin load ≈ 1 ms (NVRTC fallback 0.25 s).

**Accuracy (held-out r3joint pack, 292 windows in N = 5 calls, official scorer, `parity_gn.py`):** vs dec12kfast
rel −2.1e-5, tke +1.7e-4, mvpe −1.4e-5, sps −1.8e-4 (exact ΔF ≈ −4e-5); max |Δ| 4.9e-4 (prediction) / 9.8e-4 (upper),
p99 2.4e-4 (1–2 fp16 ulps). Validation (`validate_gn.sh`, A100): N = 0, 1, 2, 5, 7, 8, 9, 13 (…) — shape, float32,
finite, ordered, pressure zero, **repeat-equal at every N** (dec12kfast itself is *not* repeat-equal at N = 7: native
GroupNorm/cuDNN nondeterminism, harmless but noted); returned arrays never overwritten; injected loss of cubin+NVRTC →
bitwise dec12kfast; corrupt cubin → NVRTC path, bitwise equal to the cubin path; standalone zip extraction on an A100
(`zipcheck.sh`): origin = cubin, outputs bitwise equal to the validated build. An earlier build (dec12kgn2) with
shared-memory atomics was caught *not* repeatable at N = 1 and replaced before registration.

**Prediction (board, exact power mean, at 942498's harness share of 2.924 ms/sample).** The A800 ran dec12klog's call 2
at ≈ 0.745× our A100; so dec12kgn's logged call 2 ≈ **4.1 ms [3.6, 4.8]** (dec12k 8.4–8.9). Per-sample
t ≈ 3.74 ms [3.64, 3.88] → time 93.31 → **F ≈ 81.977 [81.966, 81.985], ΔF vs 942498 ≈ +0.067 [+0.056, +0.075]** at a fixed
platform level (dec12kfast alone was registered at +0.035); accuracy digits within 1e-4 of 942498 (rel/mvpe), sps ± 0.003.
The board clock (sd ≈ 0.25 ms/sample ≈ ±0.018 final) overlays this, so **the read is the logged call-2 time and the
`fused=True/cubin` import line first, the final score second.** Rank thresholds at this share: 5th needs call < 7.07 ms,
4th < 4.64 ms, 3rd < 1.71 ms.
**Kill:** log shows `fused=False` (the evaluation image cannot load the kernel — then this *is* dec12kfast and is read
under dec12kfast's rule), or logged call 2 ≥ 6.0 ms, or any accuracy subscore off by > 1e-3 → revert to dec12kfast /
dec12k's source for the Decision Phase. **Adopt** as the contract's CHAMPION (with its cubins in the package) if
`fused=True`, call 2 ≤ 5.0 ms, and accuracy within the tolerances above.

**Slot.** 2026-09-25 (replaces the armed dec12kfast, which is a strict subset of this change; `VETO_2026-09-25` stops
`arm_fast.sh`). Zip `build/dec12kgn.zip`, sha256
`c20e101267c992761c8262e8c49ab803dde11084e41eff43b6b3bb356d701a29`.

## Amendment 2026-09-24 16:50 UTC (before any upload): the A800 ratio was fault-contaminated — prediction lowered
The 2026-09-24 audit (harness dimension, `research/last_2026-09-24/agents/harness/`, findings H1/H3) showed that the
0.745 A800/A100 ratio above came from `callseq.py`, where dec12k's call 2 carried ~1800 minor page faults (≈ 2.8 ms) from
keeping the previous result alive, and that call-2 time tracks the GPU boost clock (765 MHz idle floor vs 1410 MHz). At
matched boost clocks in a harness-like child: dec12klog 9.14 ms, dec12kfast 8.28 ms on the A100; the board's dec12klog
call 2 (8.4–8.9, mean 8.56) matches the boosted figure, so **A800 ≈ 0.94× A100 at matched clocks.** My dec12kfast-vs-
dec12kgn comparison is unaffected (both use the pinned buffers, no faults): ratio 5.55 / 7.83 = 0.709. Chain:
8.56 × (8.28/9.14) × 0.709 = **5.5 ms [4.8, 6.0] logged call 2** (was 4.1 [3.6, 4.8]) → t ≈ 4.02 ms/sample →
**F ≈ 81.955 [81.948, 81.966], ΔF vs 942498 ≈ +0.045 [+0.038, +0.056]** at a fixed platform level. (The same chain puts
dec12kfast alone at ≈ 7.75 ms, ΔF ≈ +0.012 — the audit's H3.) Kill / adopt thresholds re-anchored: **kill** if the log
shows `fused=False` or call 2 ≥ 6.8 ms (≥ dec12kfast's expected level: the kernel did not pay in situ) or any accuracy
subscore off by > 1e-3; **adopt** if `fused=True`, call 2 ≤ 6.2 ms, accuracy within tolerance. 6.2–6.8 ms: adopt only if
the log's call 1 − call 2 pattern shows boosted clocks (call 2 < call 1).

## Amendment 2 — 2026-09-24 17:10 UTC (before any upload): the upload carries the v2 kernel
**What changes.** The same inference change with a better kernel: `submissions/dec12kgnv2/` (built by
`FASTGN=fastgn2 build_gn.py` from `research/last_2026-09-24/claude/fastgn2.py`). One kernel for every GroupNorm site
(the two-kernel chunked path is gone): one block per (window, group), each thread owns a fixed 4-channel slot of the
group (8-byte loads, no per-element division, loads issued four at a time), fixed-order reductions, 96…960 threads by
work. A100, graph-replayed, B = 5 (`mb_fastgn.py`, pq mbgnv2): 26→15 µs at C96 32×64, 14→8 at C192 16×32, 24→12 at
C384 16×32, 13→8 at C768 8×16 against the v1 kernel (native: 80 / 28 / 44 / 26).
**Measured.** Fresh-process call 2 on A100s, three independent jobs: v2 4.92–5.40 ms vs v1 5.17–5.54 vs dec12kfast
7.43–8.01 (v2/dec12kfast = 0.66 in all three). Parent-measured start→exit wall unchanged (17.1–17.9 s vs 16.9–17.5).
Validation identical to v1's battery (`validate_gn.sh dec12kgnv2`): every N in 0…1000 shape/dtype/finite/ordered/
pressure-zero; edge battery (NaN/Inf/zero/masks/scales/permutation/batch independence/graph-eager ≤ 2.4e-4/forced graph
failure/empty) identical in profile to dec12k's; CPU-only path; mutants (no cubin + no NVRTC → bitwise dec12kfast;
corrupt cubin → NVRTC path bitwise = cubin path); held-out r3joint parity vs dec12kfast rel −2e-5 / tke +1.0e-4 /
mvpe +1.3e-5 / sps +7e-6; standalone zip on an A100: origin = cubin, bitwise equal to the validated build.
**Repeatability (all variants, `detprobe2.py`, 30 repeated predict() calls on an A100):** N = 5 — the harness's size —
is bitwise repeatable in dec12kfast, v1 and v2; at N = 7 / 8 all three (dec12kfast included) show occasional one-ulp
differences (≤ 1.8e-4), i.e. a pre-existing cuDNN algorithm effect at those sizes, not the kernels. *(Qualified 18:05 UTC by the pre-flight kernel review, K1: under GPU contention on an A100, graph replays at every bucket including N = 5 occasionally differ by one fp16 ulp on ≤ 2 % of elements; the first diverging op is torch's native ConvTranspose2d (cuDNN dgrad), in dec12kfast too and even with deterministic cuDNN — a pre-existing property of the native path, not of the fused kernels, which repeat bitwise over 300–600 relaunches.)*
**Prediction.** 8.56 × (8.28/9.14) × 0.663 = **5.1 ms [4.6, 5.8] logged call 2** → t ≈ 3.95 ms/sample →
**F ≈ 81.961 [81.951, 81.969], ΔF vs 942498 ≈ +0.051 [+0.041, +0.059]** at a fixed platform level. Kill/adopt rule as in
Amendment 1 (kill: `fused=False`, or call 2 ≥ 6.8 ms, or an accuracy subscore off by > 1e-3; adopt: `fused=True`,
call 2 ≤ 6.2 ms). Zip `build/dec12kgnv2.zip`, sha256
`21e75b9fa3f7f10fe670f01ddfd4d0c7ea061523dfd6b40748ddd51661907e01`; the v1 arm (dec12kgn.zip) is vetoed.

## Amendment 3 — 2026-09-24 17:35 UTC (before any upload): the upload carries dec12kgnv3lean
**What changes against v2** (`submissions/dec12kgnv3lean/`, built `FASTGN=fastgn3 build_gn.py` then
`build_lean.py … --no-pre --no-meta`; source sha of the kernel `d3cb1a5df09b`):
1. **v3 kernels** (`fastgn3.py`): the convolutions in the captured path run bias-free and their biases are folded into
   the fused kernels — c1's bias into n2's GroupNorm (the GN kernel takes an fp32 pre-bias), c2's plus the 1×1 skip's
   into one residual-add kernel, the output conv's into the last-frame anchor (members) and into bias+clamp+exp (head):
   ~33 kernels fewer per member, ~23 in the head. Eager path unchanged (native).
2. **Lean setup** (ablated in `lifecycle.py`, rejected parts dropped): one CPU thread when a GPU is present; the
   checkpoint loads on the CPU; the redundant eager `_WARM` passes are gone (each bucket's self-check runs eager at its
   shape). All 11 buckets are still captured at import (no lazy capture, no keep-warm spin). *(Correction 18:00 UTC,
   pre-flight review RH2: the "NVML shows 1305–1410 MHz at every call" read NVML index 0, which on the multi-GPU pq
   nodes is not necessarily the job's GPU — that evidence is withdrawn; the call-2 timings themselves are unaffected.
   In the single-GPU evaluation container index 0 should be the scored card, so the board's `sm=` fields are provisional.)*
3. **Diagnostics** on stderr (FAQ-permitted): epoch time at import start/end, NVML SM clock and minor-fault delta per
   call — ~260 B per process, ~4.4 KB for 17 processes (cap 10 KB). Purpose: the board's per-process wall and clock
   state, which price the Decision-Phase 180 s wall.
**Measured (A100s).** Call 2, same jobs: v3lean 4.50 [4.45–4.74] vs v2 5.29 [4.83–5.67] vs dec12kfast 7.58 (cseq_v3, 8
rounds); 4.57 vs v2 5.19 (life_v3, 8 rounds); 4.71–5.19 vs dec12kfast 7.77–8.00 (validation, 6 rounds). Parent-measured
start→exit wall **10.1 s vs 16.6 s** for v2 (−39 %). Validation battery all green (N = 0…1000 incl. repeat-equal at
every N, edge set identical in profile, CPU-only, fallback mutants bitwise, standalone zip: origin = cubin, bitwise =
validated build); held-out r3joint parity vs dec12kfast rel +1.0e-5 / tke −7.7e-5 / mvpe −1.3e-5 / sps +5.8e-5
(max |Δ| 4.9e-4 / 9.8e-4). `tools/build_cubins.py --check` passes.
**Prediction.** Paired ratio to v2 0.866 → **logged call 2 ≈ 4.5 ms [4.0, 5.2]** → t ≈ 3.82 ms/sample →
**F ≈ 81.970 [81.960, 81.979], ΔF vs 942498 ≈ +0.060 [+0.050, +0.069]** at a fixed platform level; accuracy digits
within 1e-4 (rel, mvpe), sps ± 0.003 of 942498. **Kill:** `fused=False`, or call 2 ≥ 6.2 ms, or an accuracy subscore
off by > 1e-3. **Adopt** as CHAMPION if `fused=True`, call 2 ≤ 5.6 ms and accuracy within tolerance. The log's
`t=` stamps give the per-process wall; its `sm=` fields say whether the timed call ran boosted.
Zip `build/dec12kgnv3lean.zip`, sha256
`6a14117eb6684fb2d8216b635ba0bd2165a38d198f0c1f6f37f6210495900190`; the v2 arm is vetoed.

## Amendment 4 — 2026-09-24 18:25 UTC (before any upload): dec12kgnv3leanb, the pre-flight review's fixes
Two independent pre-flight reviews of the v3lean bytes ran 17:40–18:10 UTC (three-lens workflow: kernels / integration /
rules-harness, all **go**, `research/last_2026-09-24/agents/preflight/`; Codex round 2, **no-go for the unchanged zip**,
`research/last_2026-09-24/codex2/REPORT.md`). The upload now carries `submissions/dec12kgnv3leanb/` (kernel source sha
`649e827b103d`; `FASTGN=fastgn3b build_gn.py` then `build_lean.py … --no-pre --no-meta --diag2`), which differs from v3lean
only by:
1. **`__syncwarp()`** after the warp reduction in the GroupNorm kernel, before lane 0 reuses `red[0..1]` (Codex #1:
   Compute Sanitizer racecheck flagged a potential write-after-read; shuffles are not memory barriers). Re-measured
   here: racecheck **32 hazards → 0**, synccheck and memcheck 0 errors (`racecheck.sh`).
2. `add_bias_res` no longer declares its in-place operand `__restrict__` / reads it through `__ldg` (kernel review K2).
3. Diagnostics: the NVML handle is the CUDA device's (matched by PCI bus id — Codex #2, RH2, INT3), and the SM clock is
   read only at import end and after the untimed call 1, never inside the timed call (RH1); the import line reports the
   number of fused buckets `fb=` (INT2); fallback log lines are capped at 100 characters (RH3).
**Evidence on the final bytes.** Held-out r3joint parity vs v3lean: **bitwise identical** on all 292 windows (every
channel, prediction and bounds). Validation battery (`validate_gn.sh dec12kgnv3leanb`, on the pre-`fb=` build; the `fb=`
change is log-only): every N in 0…1000 OK, edges identical in profile, CPU-only, fallback mutants bitwise. Standalone
zip on an A100: `fused=True/cubin fb=11`, bitwise equal to the validated build. Call 2 on an A100, same job: 4.76 ms vs
v3lean 4.73 vs dec12kfast 7.86 (8 rounds) — the barrier costs nothing measurable.
**Prediction unchanged:** logged call 2 ≈ 4.5 ms [4.0, 5.2] → **ΔF vs 942498 ≈ +0.060 [+0.050, +0.069]**. Kill / adopt
as in Amendment 3, with the pre-flight's clarification (INT2): a log line `fused GN disabled at bucket b` with b > 5
still means the N = 5 timed call ran fused; `fb=` counts the fused buckets. `sm=` is the call-1-end clock of the scored
GPU. Correction to the registration's wording (INT1): a *driver-level* launch error during capture would drop every
graph and run eager (correct outputs, slow); only Python-level failures and self-check failures fall back to native
graphs. Zip `build/dec12kgnv3leanb.zip`, sha256
`af1f46691cd7f2fcc2f5b9f3d98f7edc5ef8eab294192a75324f433c6f7944c4`; the v3lean arm is vetoed.

## Scored 2026-09-25 00:21 UTC: 943854 = **81.94269, new retained best (+0.03277 vs 942498), rank 5** — ADOPT
The upload survived three platform HTML errors (00:00:53, 00:06:16, 00:11:41 — `arm_zip.sh` retried; the old
`arm_fast.sh` would have stopped for the day) and went in at 00:16:50. Subscores: rel 94.8462108 (+5e-6) / tke
79.2325971 (−2.7e-4) / mvpe 94.0910421 (+6.7e-5) / sps 44.0015022 (−1.1e-4) / time 92.9554809 (+0.341), t = 4.187
ms/sample. **Ingestion log** (`research/last_2026-09-24/logs_943854/`, `read_board.py`): 17 processes, all
`fused=True/cubin fb=11`; **call 2 median 4.60 ms (mean 4.76, 4.5–6.9)** against the registered 4.5 [4.0, 5.2] and
dec12klog's 8.4–8.9 — the kernel delivered in situ exactly as measured; SM clock 1410 MHz at import end and after call 1
(boosted). The shortfall against the registered ΔF is the platform's: harness share = 4.187 − 0.952 = **3.24
ms/sample vs 2.92 on 09-24** (+0.31, about one clock-draw sd). At 09-24's harness share this build reads ≈ +0.061.
**Per-process wall (new, from the `t=` stamps):** import 4.5 s (dec12klog 3.9), 1.82 s from call 2 to the next import
(teardown + spawn), 6.70 s per process, 107 s for 17 → a 180 s stage fits ≈ 26 processes before overheads.
**Rule:** `fused=True`, call 2 4.60 ≤ 5.6 ms, accuracy within 3e-4 on every subscore → **adopt**. The contract CHAMPION
becomes `dec12kfinal` (the same computation bitwise on finite inputs, with the input sanitise and honest comments;
check_recipe OK and 18/18 mutants with it, 2026-09-24).
