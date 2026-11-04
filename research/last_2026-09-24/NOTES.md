# Last round, 2026-09-24 (afternoon/evening UTC) — audit, and the fused GroupNorm kernel

Joint Claude + Codex. Board at 16:10 UTC: 1 roysegal 82.15105 · 2 modu-lemon 82.06952 · 3 skabob 82.01667 · 4 np-user
81.96840 · 5 PhysicsOracle 81.93145 · **6 us 81.90992** (942498) · 7 JLShen 81.86335 · 8 doomduke2 81.85937 · 10 hituc
81.76452 · 11 pone7 81.74229.

## 1. Audit (six dimensions, 27 agents, every material finding adversarially verified; Codex independently)
Synthesis: `agents/AUDIT_SYNTHESIS.md`; Codex: `codex/REPORT.md`. **No critical finding; nothing touches the shipped
outputs.** What mattered:
* **The timed call is the GPU's clock state** (harness H1, Codex §4, NVML): import-time capture leaves the GPU boosted
  (~1400 MHz) for 1–2 s; lazy capture / a 2 s idle gap puts call 2 at 765 MHz, +5 ms. The board's dec12klog call 2
  (8.4–8.9 ms) matches the boosted A100 figure (9.14 ms) → **A800 ≈ 0.94× A100 at matched clocks**, not 0.745.
* **dec12kfast's +0.035 was mostly a probe artefact** (H3): `callseq.py` kept results alive, putting ~1800 minor faults
  (~2.8 ms) into dec12k's call 2. At matched clocks dec12kfast gains ~0.86 ms/call → ≈ +0.012.
* **The 180 s wall** (H2, Codex §3): 17 strictly sequential fresh processes use ~40–60 % of it on the dev set; a private
  set ≥ ~2× the dev set under the same lifecycle would fail. Per-process cost is ~3× a lean torch+CUDA floor.
* The upload script could not recover from a transient "Failed" or a quota-call error (M1) → replaced by
  `arm_zip.sh` (re-checks hash + veto before every attempt, retries API errors, resubmits after a platform Failed).
* Judge-facing docs describe an older model (M4: `docs/REPORT.md` is the 09-03 draft); environment pin thin (M5);
  contract adoption of a new inference source needs coordinated edits (L1); retained best not pinned (L2); stale
  docstrings (L3); NaN input zeroes a window (L4, Codex patch 05); fp16-packing wording (L5); organiser competition
  titles in two tracked API dumps (L8, redacted in ad27afb; untracked evidence copies left untouched).
* Codex: no accuracy idea clears a slot; the clean harness-share levers are used up; lazy+spin setup rejected by the
  synthesis as fragile and gaming-prone (L7).

## 2. The fused GroupNorm+SiLU kernel (new; the round's result)
The kernel-level profile showed torch 2.2's channels_last fp16 GroupNorm at 66–80 µs per call at 5×96×32×64 on an A100
(the 3×3 conv beside it: 17 µs), ~20× off its memory roofline — GN+SiLU was 1.17 of a member's 2.26 ms. The record had
closed "hand-written GroupNorm" because torch-op compositions (8–13 small kernels) were slower; a real kernel was never
tried because the image has no C compiler (Triton cannot build its launcher). **Route found:** CUDA source embedded in
`submission.py`, compiled by NVRTC (torch's own `nvidia-cuda-nvrtc` 12.1) or loaded from a shipped cubin named by the
source hash (`tools/build_cubins.py` regenerates/verifies; NVRTC is deterministic), launched with libcuda's
`cuLaunchKernel` on torch's current stream — captured into the CUDA graphs like torch's own kernels. Kernels run only
inside captured graphs; every bucket is self-checked against the native eager path at import; any failure → native
graphs (= dec12kfast, bitwise).

| build | kernel | A100 call 2 (8 fresh processes, same job) | per-process wall |
|---|---|---|---|
| dec12kfast (armed until 16:53) | native | 7.58 [7.50–7.83] ms | ~17 s |
| dec12kgnv2 (armed 17:10–17:35, superseded) | one GN kernel (4-ch slots, 8-byte loads) | 5.29 [4.83–5.67] | ~17 s |
| dec12kgnv3 | + conv biases folded into GN / residual / anchor / head epilogue | 4.82 [4.47–4.92] | ~17 s |
| **dec12kgnv3lean (armed for 09-25)** | v3 + lean setup + diagnostics | **4.50 [4.45–4.74]** | **~12 s** |

Lean setup (ablated, `lifecycle.py`): one CPU thread when a GPU is present, checkpoint loaded on the CPU, the redundant
eager `_WARM` passes dropped (every bucket's self-check runs eager at its shape anyway); import-time capture of all 11
buckets kept (the clock should stay boosted as in dec12kfast; the NVML 1305–1410 MHz read in the ablation came from NVML
index 0, which on multi-GPU pq nodes need not be the job's GPU — withdrawn, pre-flight RH2). Rejected in the ablation: a shared `_pre` graph
(+capture time, no call-2 gain) and meta-device construction (no gain). Diagnostics (stderr, FAQ-permitted): epoch time,
NVML SM clock and minor faults at import end and per call, ~260 B per process.

Accuracy: held-out r3joint pack, 292 windows in N = 5 calls, official scorer — every variant within fp16 ulps of
dec12kfast (v2: rel −2e-5 / tke +1.0e-4 / mvpe +1.3e-5 / sps +7e-6; v3: +1e-5 / −8e-5 / −1e-5 / +6e-5); max |Δ| 4.9e-4
prediction, 9.8e-4 bounds. Validation battery (`validate_gn.sh`): N = 0…1000, edge set (NaN/Inf/zero/masks/scales/
permutation/batch independence/forced graph failure/empty), CPU-only, fallback mutants (no cubin + no NVRTC → bitwise
dec12kfast; corrupt cubin → NVRTC path = cubin path), standalone zip (`zipcheck.sh`). Repeatability (`detprobe2.py`, 30
calls): N = 5 bitwise in every variant; N = 7/8 show rare 1-ulp differences in *all* variants including dec12kfast
(cuDNN algorithm effect; deterministic cuDNN costs +0.4 ms and was not adopted). *(Qualified 18:05 UTC by the pre-flight kernel review, K1: under GPU contention on an A100, graph replays at every bucket including N = 5 occasionally differ by one fp16 ulp on ≤ 2 % of elements; the first diverging op is torch's native ConvTranspose2d (cuDNN dgrad), in dec12kfast too and even with deterministic cuDNN — a pre-existing property of the native path, not of the fused kernels, which repeat bitwise over 300–600 relaunches.)* Two defects caught before any upload:
an atomics reduction (non-repeatable at N = 1) and a rename that changed the kernel source hash (cubin not found →
NVRTC fallback); the builder now asserts the shipped source hashes to the cubin name.

## 3. Slots and registrations (`submissions/_manifest/CANDIDATE_dec12kgn.md`, registration + Amendments 1–3)
09-25: **dec12kgnv3leanb** (= v3lean + the pre-flight fixes: `__syncwarp` in the GN reduction — racecheck 32 hazards → 0 —,
no `__restrict__` on the in-place operand, NVML by PCI bus id and never in a timed call, `fb=` count, short fallback logs;
bitwise identical outputs to v3lean on the 292 held-out windows), registered **+0.060 [+0.050, +0.069]** (logged call 2
≈ 4.5 ms [4.0, 5.2] on the A800), armed by `arm_zip.sh` (tag gnv3leanb; the dec12kfast, v1, v2 and v3lean arms are vetoed). Read it with `read_board.py <id>` (log first).
09-26 / 09-27: a clean final-source build (v3lean's numerics + the input sanitise of audit L4 + the stale-comment and
dead-code cleanup of L3/RH4), then a repeat of the best-reading bytes.

## 4. Measured and not adopted
* **cuDNN benchmark mode** (`cseq_bench`, 8 rounds): call 2 4.85 → 4.45–4.48 ms (≈ −0.4 ms, ≈ +0.005), import +1 s; the
  choice of algorithm can differ between processes on timing noise, which would end the bitwise cross-process
  reproducibility the contract relies on (the organisers also ask for deterministic inference). Owner's call.
* **Post-graph glue fusion** (mean/std/gain/assembly, ~40 kernels ≈ 0.3 ms per call on the A100, ≈ +0.004): not built.
* **Pre-flight review** (three lenses + Codex round 2, `agents/preflight/`, `codex2/`): see REPORT files; the
  rules/harness lens is *go* with only later-class findings (NVML reads inside the timed call cost 0.02–0.05 ms with a
  rare stall tail; NVML index 0 need not be the CUDA device on multi-GPU hosts; fallback log lines could approach the
  10 KB cap; stale comments).

## 5. Board read 2026-09-25 00:21 UTC: 943854 = 81.94269, rank 5 — adopted
Accuracy unchanged to 3e-4 on every subscore; `fused=True/cubin fb=11` in all 17 processes; timed call median 4.60 ms
(mean 4.76) vs dec12klog 8.5 — the in-situ gain is the measured one. Harness share 3.24 ms/sample (09-24 2.92): the
platform drew ≈ +1 sd, so ΔF +0.0328 instead of ≈ +0.061 at a fixed platform. Per-process wall 6.70 s (import 4.5 s,
call-2 → next import 1.82 s), 107 s for 17 processes. The upload hit three platform HTML errors and `arm_zip.sh`
retried through them. CHAMPION → `dec12kfinal`; 943854 pinned (`tools/verify_best.py`, 15 pins).

## 6. Wall-budget trim measured and rejected; the last two slots armed
Capturing the 16/32/64 buckets lazily (never used at N = 5) cut the A100 per-process wall 9.54 → 8.05 s but raised call 2
4.92 → **7.41 ms** (`life_lazy`, 8 rounds): the heavy import-time captures are what leave the GPU boosted for the timed
call. Rejected; the wall budget stays a documented limitation (≈ 26 fresh processes per 180 s, ≈ 44 per 5 min at the
in-situ 6.7 s). Slots: **09-26 `dec12kfinal`** (`CANDIDATE_dec12kfinal.md`, 81.954 [81.936, 81.974]); **09-27**
`arm_27.sh` repeats dec12kfinal if its 09-26 accuracy digits match 943854's within 1e-3, else 943854's bytes.

## 7. 09-25 evening: benchmark mode (owner's request) and the post graph's glue fused
**cuDNN benchmark mode** (`dec12kfinalb`, `CANDIDATE_dec12kfinalb.md`): `cudnn.benchmark = True` with
`deterministic = True` before any capture. Paired A100 lifecycle: call 2 4.55 vs 4.73 ms, import +3.6 s (+19 %); held-out
parity to fp16 ulps; outputs differ between processes by fp16 rounding (the algorithm choice is timed per process).
Board-only — not in the retrain contract (per-process wall, cross-process reproducibility).
**Glue kernels** (`dec12kfinal4`, `claude/build_v4.py`, kernels in `claude/fastgn4.py`, source sha12 `7d074a23c378`;
`CANDIDATE_dec12kfinal4.md`): in the parallel buckets' post graph, `head_prologue` writes the members' mean, their
unbiased spread (sequential Welford, torch's order) and the head's 161-channel input in one pass; `post_assemble` does the
gain, scaling, bounds, non-finite guard, containment and the float32 output layout in one pass (~40 small kernels
replaced). Unit tests ≤ 1 fp16 ulp on 0.02 % / 0.001 % of elements; Compute Sanitizer clean; post-graph device time
1056 → 800 µs; A100 call 2 **4.43 vs 4.88 ms** (4.09 with benchmark mode), import unchanged; held-out r3joint parity
rel/tke/mvpe identical to six decimals. Registered dec12kfinal4b **81.964 [81.944, 81.983]**, dec12kfinal4 81.961.
**Slots:** 09-26 carries **dec12kfinal4b** (the dec12kfinal and dec12kfinalb arms are vetoed); 09-27 `arm_27.sh`
repeats dec12kfinal4b if its 09-26 accuracy is within 1e-3 of 943854's, else 943854's bytes.
**Contract:** `CHAMPION=dec12kfinal4` (the deterministic v4) after `CHAMPION=dec12kfinal4 check_recipe` (kernel images
byte-identical to a fresh NVRTC compile) and the mutation suite (18/18) passed; `SHIPPED_PACK` and the mutation
default moved with it; `research/last_2026-09-24/claude/adopt_final4.py` made the edits.

## 8. 09-25 late: where the timed call goes, and v5 (the prediction's copy overlapping the head)
**Wall split** of one N = 5 call of v4 on an A100 (`claude/xfer.py`): input copy 0.22 ms (pageable is already the
cheapest: staging through page-locked memory costs 0.15 ms of CPU copy + 0.11 ms DMA, pipelining 2–16 chunks 0.25–0.43),
member graphs 2.97 (four 1135 µs-device-time graphs on four streams; per member: 3×3 convs ~60 %, GroupNorm+SiLU 22 %),
post graph 0.84, output copy 0.34 (the prediction plane alone 0.13). **The time channel is nearly spent:** F moves
≈ 0.0015 per 0.1 ms of our call (the harness's ≈ 3.1 ms/sample share dominates t), so a zero-time call would add only
≈ +0.065 — the 0.056 gap to 4th cannot be closed by inference speed.
**Benchmark-mode variants rejected** (`claude/life_bench.sh`, 8 rounds): `benchmark_limit = 0` call 2 4.21 vs 4.11 ms
for v4b and import 17 s (vs 10); `deterministic = False` 4.12. v4b's two lines stay.
**v5** (`CANDIDATE_dec12kfinal5.md`, `claude/build_v5.py`, `fastgn5.py`, sha12 `93aac3a64160`): the post graph split so
the prediction plane (post_assemble part 0) is copied to the host on a side stream while the head runs; member 0's
graph hands over the normalised input and scale. Parts 0 + 1 bitwise v4's kernel on 189 M elements; held-out
differences at v4's own cross-process floor (0.11–0.15 % of elements, one ulp, native ConvTranspose); 42 edge cases
identical to v4's; batteries green; import +0.1 s (noise). Call 2: v5 4.12 / 4.23 / 4.24 vs v4 4.47 / 4.51 / 4.50 ms;
v5b 3.93 vs v4b 4.11. Registered v5b 81.966 [81.953, 81.979] (+0.002–0.004 over a v4b draw).
**09-27:** `arm_27b.sh` (replaced `arm_27.sh`) uploads dec12kfinal5b if the 09-26 read passes and
`GREEN_final5b` exists (written after the registration commit), else v4b, else 943854's bytes.

## 9. Read 2026-09-26: 945471 (dec12kfinal4b) = 81.94157; v5 adopted; 09-27 = dec12kfinal5b
Accuracy = 943854's within 2.2e-4; `fused=True/cubin fb=11` in 17/17; call 2 median 4.00 ms (vs 4.60 for 943854);
import 5.8 s (benchmark mode, board-only). The harness share drew 3.39 ms/sample (third day rising: 2.92, 3.24, 3.39),
so F = 81.94157, −0.0011 vs the retained best. We slipped to **6th** (doomduke2 81.96799 moved up; np-user and skabob
passed modu-lemon). **Contract:** an independent review of v5 found no correctness defect (two low-severity notes,
`CANDIDATE_dec12kfinal5.md` Amendment 1); `CHAMPION=dec12kfinal5` with `check_recipe` OK (cubins byte-identical),
18/18 mutants, 15 pins intact (`claude/adopt_final5.py`; `--revert` restores dec12kfinal4). **09-27:** the read passes,
`GREEN_final5b` is present, so `arm_27b.sh` uploads dec12kfinal5b: expected 81.944 at today's share, 81.955 at 3.24,
81.968 at 3.08 — a new best is roughly a coin flip, 5th place (81.968) needs the share back near 09-24's.

## 10. Read 2026-09-27: 947208 (dec12kfinal5b) = 81.99600 — new retained best, rank 5; the Development Phase is spent
Accuracy = 943854's to 2e-4; `fb=11` 17/17; our call 2 median 3.80 ms (945471: 4.00; 943854: 4.60; 942498: 8.5); import
6.3 s. The harness share drew 2.74 ms/sample (the four days: 2.92, 3.24, 3.39, 2.74) — the low draw carried F above the
registered band. Across the round the time channel took the same model from 81.90992 to 81.99600 (+0.086) with no
accuracy change. Board at 01:15 UTC: 1 roysegal 82.15105 · 2 modu-lemon 82.12771 · 3 np-user 82.12590 · 4 skabob 82.09701
· **5 us 81.99600** · 6 syouya_tobita 81.98173 · 11 andychang 81.80098. Quota for 09-27 used; the phase ends 23:59 UTC.
Pinned 947208 read-only on both copies (16 pins). The Decision Phase (09-28 → 10-25) is organiser-controlled: the top
10 are retrained from their code and scored on a private set with unseen angles / Reynolds numbers; the package format
was still unannounced at 01:30 UTC (all six pages identical to the 09-24 snapshot).
