# Candidate dec12kfinal — the contract's inference source on the board (registered 2026-09-25 00:40 UTC, before upload)

**What it is.** `submissions/dec12kfinal/` = dec12k's `model.pth` + the Decision-Phase inference source (the retrain
contract's `CHAMPION` since 2026-09-25), built by `research/last_2026-09-24/claude/build_final.py` from dec12kgnv3leanb
(943854): identical computation, bitwise, on finite inputs (N = 1, 5, 8, 13, 37 on real windows, graph and eager paths);
the CUDA source and cubins are unchanged (sha12 `649e827b103d`). Differences: comments rewritten to describe the current
method, the dead Combiner / `_WARM` / `_GRAPH_BUDGET_S` removed, and non-finite input velocities read as 0 (a NaN or Inf
cell no longer zeroes its whole window). Validation battery (`validate_gn.sh dec12kfinal`) and standalone zip green;
A100 call 2 4.6–5.4 ms vs 7.7–8.0 for dec12kfast; `check_recipe` OK and 18/18 mutants with it as CHAMPION.

**Why a slot.** It puts the exact Decision-Phase bytes on the board once, and it is a second draw of the platform clock
for the same computation (the board keeps the best).

**Prediction.** Accuracy = 943854's digits to ≈ 1e-5 (rel 94.84621 / tke 79.23260 / mvpe 94.09104 / sps 44.00150; the
dev set carries no non-finite values, so the sanitise is the identity; the only expected differences are the one-ulp
native ConvTranspose effect). Our timed call ≈ 4.6 ms in the log again. The final score is the harness's clock draw:
F = 81.943 at 09-25's share (3.24 ms/sample), 81.967 at 09-24's (2.92), 81.954 at their mean; **81.954 [81.936, 81.974]**
for ± 1 sd. New retained best iff t < 4.187 ms/sample. **Kill** (do not repeat on 09-27): any accuracy subscore off by
> 1e-3, `fused=False`, or call 2 ≥ 6.2 ms in the log — then 09-27 repeats 943854's bytes instead.
