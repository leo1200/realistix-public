# Candidates dec12kfinal4 / dec12kfinal4b — the post graph's glue fused (registered 2026-09-25 22:20 UTC, before upload)

**What.** `submissions/dec12kfinal4/` = dec12kfinal (the contract CHAMPION) + two CUDA kernels in the parallel buckets'
post graph (`research/last_2026-09-24/claude/build_v4.py`, kernels in `fastgn4.py`, source sha12 `7d074a23c378`):
* `head_prologue` — the members' plain mean, their unbiased spread (sequential Welford, torch's order) and the interval
  head's 161-channel input (window, mean, log scale, spread, log spread) in one pass, replacing the stack / float /
  mean / std / log / cat kernels;
* `post_assemble` — the member-variance-matched gain, the window scaling, the bounds, the non-finite guard, the
  containment `lower <= prediction <= upper` and the float32 (3, N, 20, 32, 64, 3) layout in one pass, replacing
  ~30 small reduction / elementwise / copy kernels.
Everything else is dec12kfinal's: the eager path, the sequential 16/32/64 buckets, the self-check against eager at
import, the native-capture fallback. `dec12kfinal4b` adds the benchmark-mode lines of `CANDIDATE_dec12kfinalb.md`.

**Measured.** Unit tests vs the native computation: ≤ 1 fp16 ulp on 0.02 % (head input) / 0.001 % (outputs) of
elements. Compute Sanitizer racecheck / synccheck / memcheck on all v4 kernels: 0 hazards, 0 errors. A100 kernel
profile: post-graph device time **1056 → 800 µs**. Paired fresh-process lifecycle (quiet A100, 8 rounds): call 2
**4.43 vs 4.88 ms** for dec12kfinal (−0.45), import unchanged (8.09 vs 7.94 s). Held-out r3joint parity vs dec12kfinal
(292 windows): rel / tke / mvpe identical to six decimals, sps −7e-5; prediction differs on 3e-5 of elements, bounds on
≈ 4.6 % by one ulp (≤ 2.4e-4). Standalone zip: origin = cubin, bitwise equal to the validated build.

**Prediction (943854's accuracy digits, our call ≈ 4.34 ms for v4, ≈ 4.17 ms for v4b).** v4b: F = 81.952 at 09-25's
harness share, 81.976 at 09-24's, **81.964 [81.944, 81.983]** at ± 1 sd; v4: **81.961 [81.942, 81.981]**. New best iff
t < 4.187 ms/sample. **Kill:** accuracy off by > 1e-3, `fused=False` / `fb` < 5 buckets, call 2 ≥ 6.2 ms, import ≥ 7 s.
**Slot:** 09-26 carries **dec12kfinal4b** if its checks are green by 23:40 UTC (it supersedes dec12kfinalb, which it
contains); the contract CHAMPION moves to the deterministic dec12kfinal4 only after `check_recipe` and the mutation
suite pass with it.

**Read 2026-09-26 00:07 UTC: 945471 (dec12kfinal4b) = 81.94157 — kill rule passes, not a new best.** Accuracy vs 943854:
rel +1.2e-5, tke −2.2e-4, mvpe +1.0e-4, sps +8.5e-5 (all within 1e-3). `fused=True/cubin fb=11` in all 17 processes.
Our call 2 **median 4.00 ms** (mean 4.08; range 3.9–4.3) vs 943854's 4.60 — faster than the registered ≈ 4.17.
Import median 5.8 s (max 6.7; benchmark mode's +1.3 s in situ). But the harness's own share drew **3.39 ms/sample**
(09-25 3.24, 09-24 2.92), beyond the registered ± 1 sd: t = 4.202 ms/sample vs 943854's 4.187, so F came in
−0.0011 below the retained best and below the registered band [81.944, 81.983]. The miss is the platform draw; the
kernels delivered more than predicted. 943854 stays the retained best. Board at the read: 1 roysegal 82.15105 ·
2 np-user 82.11219 · 3 skabob 82.09701 · 4 modu-lemon 82.06952 · 5 doomduke2 81.96799 · **6 us 81.94269** ·
7 PhysicsOracle 81.93145 · 11 hituc 81.76452 (cushion 0.178). Log: `research/last_2026-09-24/logs_945471/`.
