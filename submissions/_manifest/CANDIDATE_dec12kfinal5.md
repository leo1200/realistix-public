# Candidates dec12kfinal5 / dec12kfinal5b — the prediction's copy overlapping the head (registered 2026-09-25 23:24 UTC (commit 2c4a765), before upload)

**What.** `submissions/dec12kfinal5/` = dec12kfinal4 (the contract CHAMPION since 09-25 23:00 UTC) with the parallel
buckets' post graph split in two (`research/last_2026-09-24/claude/build_v5.py`, kernels in `fastgn5.py`, source sha12
`93aac3a64160`):
* the first post graph writes the prediction plane (`post_assemble` part 0: the gain needs only the members) and the
  head's input; the prediction's copy to the page-locked landing buffer then runs on a side stream while the second
  graph runs the interval head and writes the bounds (part 1), whose copy follows;
* member 0's graph also returns the normalised input and window scale it computed, so the post graph no longer
  recomputes them (~10 small kernels on the serial path);
* the import self-check also runs the page-locked path the harness takes, on every parallel bucket.
Everything else is dec12kfinal4's. `dec12kfinal5b` adds the benchmark-mode lines of `CANDIDATE_dec12kfinalb.md`
(byte-for-byte the v4 → v4b hunks).

**Measured (A100).** Wall split of one N = 5 call of v4 (`xfer.py`): input copy 0.22 ms, member graphs 2.97, post graph
0.84, output copy 0.34 (the prediction plane 0.13). Kernel: parts 0 + 1 bitwise v4's `post_assemble` on 189 M
elements (`ut_part2.py`; part 2, not used by the graphs, differs on 64 bound elements by one fp16 ulp — NVRTC contracts
the merged loop differently); Compute Sanitizer racecheck / synccheck / memcheck clean. Held-out r3joint (292 windows):
v5 vs v4 differs on 0.11 % / 0.15 % / 0.15 % of elements by one ulp, exactly the floor of v4 vs v4 in one process
(0.11 / 0.15 / 0.14 %; native ConvTranspose); rel identical, tke −1.8e-6, mvpe +7e-7, sps −1.2e-5. Paired fresh-process
lifecycle, call 2: v5 **4.12 vs 4.47**, **4.23 vs 4.51**, **4.24 vs 4.50** ms for v4 (8–10 rounds each); v5b **3.93 vs
4.11** for v4b. Import: per-bucket setup equal, module import +0.1 s median (alternating order, `imptime2.sh`).
Validation battery (`validate_gn.sh`) green for both: batch sizes 0–1000 vs dec12kfast, 42 edge cases (identical to v4's
for v5), CPU path, both fallback mutants (native fallback bitwise dec12kfast for v5; not bitwise for v5b, by benchmark
mode, as for v4b), fresh-process call sequences. Standalone zips: `fused=True/cubin fb=11`; v5 bitwise equal to its
build dir, v5b not (benchmark mode, as for v4b).

**Prediction (943854's accuracy digits; our in-situ call ≈ 4.17 − 0.2 = 3.97 ms for v5b).** v5b: F = 81.953 at 09-25's
harness share, 81.979 at 09-24's, **81.966 [81.953, 81.979]**; about +0.002–0.004 over a draw of v4b — the clock draw
dominates. New best iff t < 4.187 ms/sample (or below the 09-26 read's t if that sets a new best).
**Kill:** any accuracy subscore off by > 1e-3 from 943854's, `fused=False` or `fb` < 11 in any process, call 2 ≥ 6.2 ms,
import ≥ 7 s.

**Slot and contract.** 09-27 carries **dec12kfinal5b** through `research/last_2026-09-24/arm_27b.sh` if the 09-26 read of
dec12kfinal4b passes its kill rule (else 943854's bytes); the flag `research/last_2026-09-24/GREEN_final5b` is written
only after this file is committed. The contract CHAMPION moves to the deterministic **dec12kfinal5** once
`CHAMPION=dec12kfinal5 check_recipe` and the mutation suite pass; if the 09-27 read fails its kill rule, it moves back
to dec12kfinal4.

**Amendment 1 (2026-09-25 23:55 UTC, before upload) — independent review.** The Codex pass died on a rejected API key
(`research/last_2026-09-24/codex_v5/run.log`); the same brief went to an independent Claude reviewer (static review of
torch 2.2.2's stream / graph / pinned-copy semantics, cuobjdump of both shipped cubins, GPU tests on an idle H100 with
the sm90 cubin; scripts in `research/last_2026-09-24/review_v5/`). **No correctness defect** in stream ordering, graph
capture, fallbacks, landing buffers or the kernel's `part` argument (the part-0 path issues no loads; part 2 is
unused); a NaN-filled, alternating-input ordering test passed 40 iterations at buckets 1, 5, 8. Two low-severity
weaknesses, neither changing v5's outputs: (1) the import self-check's page-locked run reuses the same input, so it
could not catch a copy-ordering race (a mutant without `cs.wait_stream` passes it; the shipped ordering is correct by
reading and by the NaN-fill test); (2) containment around the prediction relies on parts 0 and 1 computing it
identically — true in both shipped cubins (same SASS sequence; 0 of 188.7 M elements differ), not guaranteed by
construction for a different NVRTC; the scorer rejects only non-finite or reversed bounds (`lower > upper`, which part 1
excludes by construction), so a one-ulp-outside prediction would at most move coverage. The registered bytes stand.

**Read 2026-09-27 00:06 UTC: 947208 (dec12kfinal5b) = 81.99600 — new retained best (+0.0533 vs 943854), rank 5.**
Accuracy vs 943854: rel +3e-6, tke −2.0e-4, mvpe +8.4e-5, sps +1.5e-4 (all within 1e-3). `fused=True/cubin fb=11` in all
17 processes. Our call 2 **median 3.80 ms** (mean 3.85; range 3.7–4.3; v4b's 945471: 4.00) — the registered in-situ
≈ 3.97 was conservative. Import median 6.3 s (max 6.8 < 7). The harness share drew **2.74 ms/sample** (the lowest of the
four days: 2.92, 3.24, 3.39), t = 3.511 ms/sample → time 93.510, F 81.99600, above the registered band [81.953, 81.979]
because the platform drew low. Board at 01:15 UTC: 1 roysegal 82.15105 · 2 modu-lemon 82.12771 · 3 np-user 82.12590 ·
4 skabob 82.09701 · **5 us 81.99600** · 6 syouya_tobita 81.98173 · 7 doomduke2 81.96799 · 10 JLShen 81.88545 · 11 andychang
81.80098 (cushion 0.195). Kill rule passes: CHAMPION stays dec12kfinal5. Log: `research/last_2026-09-24/logs_947208/`.
