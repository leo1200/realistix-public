# Final round, 2026-09-24 — audit, deliverable, and the time channel

## 1. Deliverable integrity (dec12k)
* `check_recipe --replay-student` on `submissions/dec12k/model.pth`: **80/80 bitwise**, replayed on node7 a day after
  the node10 build (`replay_dec12k.sh`, pq `1790226952794-replay_dec12k`).
* dec12k's four packed members are **bitwise the 09-19 canonical members (544/544)**, its six teachers bitwise the
  canonical teachers (80/80 each): member and teacher training reproduce across a week and across nodes.
* Its student is 0/80 against the two 12k students of 09-23 ~02:00 UTC — those came from the pre-commit working tree,
  which is exactly the drift REPLAY_FINDING.md describes; the committed tree now replays its own output.

## 2. The board read and the harness, from its own logs
942498 (dec12k + stderr logging, `CANDIDATE_dec12klog.md`) = **81.90992**, new retained best, rel/MVPE on the registered
digits. `logs_942498/prediction_ingestion_stdout.txt`: **A800-SXM4-80GB**; **17 fresh processes, each one untimed
warm-up + one timed call of N = 5**; import 3.9 s; our timed call 8.4–8.9 ms, the harness's per-sample mean 4.6 ms.

## 3. The timed call (`callseq.py` — fresh process, eight N = 5 calls; call 2 is what the harness times)
Local H100: dec12k 9.8 / **10.2** / 7.4 / 5.9 … vs pinned-output variant 5.4 / **5.0** / 5.4 … — the early calls of a
fresh process pay for new host allocations (three pageable `.cpu()` copies). Ablation on pq A100 (10 processes each):

| variant | call 2, ms |
|---|---|
| dec12klog | 11.4–12.3 (node8), 10.9–11.7 (node9) |
| + exact N = 5 bucket | 10.8–11.2 |
| **+ in-graph fp32 stack + one pinned copy (dec12kfast)** | **7.7–8.2**, **7.6–8.0** |
| same + lazy capture on the warm-up (Codex patch 03) | 12.9–13.1 — **rejected** |

Kernel-level levers screened and dropped: grouped-conv member fusion (grouped convs 1.3–4.6× slower than four separate
convs at these shapes, `mb_grouped.py`); hand-written GroupNorm (every variant slower than native in-graph,
`mb_gn*.py`; native channels_last GN costs 66–77 µs at 96×32×64 vs 17 µs for the 3×3 conv beside it — the one real
kernel cost left, and only NCHW helps it); `cudnn.benchmark` (−0.3 ms but per-process algorithm choice, i.e.
non-deterministic outputs — not shipped). Held-out parity of dec12kfast (`fold_compare5.py`, 292 r3joint windows in
N = 5 calls): identical to noise. Edge set N = 0…1000, CPU, injected graph failure, array retention: pass.

## 4. Head arms (`CANDIDATE_head2x.md`): harvested
T12-12k +0.0067 SPS (floor +0.010), T6-24k −0.059, T12-24k +0.0001 with a cell at −0.083; the 24k schedule diverges
near step 8000 at lr 8e-4. The 12k student already beats even a 12-teacher output average.

## 5. Audit (Codex, `codex/REPORT.md`) and what was done with it
Fixed in 3afa023: untracked manifest pair and mutation suite (clean export failed); CHAMPION literal k4a2 vs shipped
canon0918f32; check_recipe blind to member seeds and to `submission.py` (now bound to the CHAMPION source byte for
byte). Not adopted: removing this host's conditional path fallbacks (harmless elsewhere, needed here); lazy capture
(measured +5 ms on the timed call); NaN sanitising (release data carries no NaN; unmeasured on held-out). A pure
git-archive build of d490ad1 into an empty root is running (`clean_build.sh`, pq `1790232998337-cleanbuild`).

## 6. Release acceptance: a pure git-archive build reproduces the deliverable bitwise (2026-09-24 ~09:30 UTC)
`git archive d490ad1` into an empty directory, no repo on the path, fresh run/build/eval roots, explicit
`REALPDE_DATA`/`PY`, `bash tools/train_final.sh` (pq `1790232998337-cleanbuild`, node14, A100-PCIE-40GB, driver
580.126.09 — dec12k was built on node10, driver 595.71.05): **RC 0**; data manifest OK; the parity eval set was cut
fresh from the release; `check_pack` MATCH; `check_recipe` OK. Against dec12k (`compare_clean.py`): **members 4 × 136/136,
student 80/80, teachers 6 × 80/80 — bitwise identical.** The committed tree alone regenerates the shipped model.
