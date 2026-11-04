# Candidate k5 — fifth ordinary member on the 923184 recipe (pre-registered 2026-09-12 02:15 UTC, before the build finishes; NOT uploaded)

**What.** The four members of 923184 (seeds 0–3) plus an ordinary seed-4 member (identical argv, H100, checkpoint 6000)
and a width-96 spread head refitted on five members (seed 0 exported; seed 11 as the nuisance read). Same inference
source. Tensor bytes 5 × 45.2 + 26.2 ≈ 252.6 MB; the extracted size must be checked against 256,000,000 bytes.

**Why.** K4 paid +0.118 sps on the board (+0.088 in-sample): the head's capture of an added member's spread transferred
at ≥ 1×. The archive's dispersion model (relative noise 1/√(2(K−1)); 2→3 vs 3→4 measured 2.07× against 2.25× predicted)
prices 4→5 at ≈ 0.6× of 3→4. Time is ≈ free at one to three windows (+0.29 ms local per member ≈ −0.007 final).

**Prediction vs 923184.** sps +0.05 [0.00, +0.09]; rel/tke/mvpe each +0.00…+0.005; time 0 [−0.01, 0]; non-time Δ
+0.015 [−0.01, +0.03]; total Δ +0.01 [−0.02, +0.03]. **Kill:** non-time Δ ≤ 0 under either weight fit, extracted size
≥ 256,000,000 bytes, or the time loss ≥ the non-time gain. Pass → five members become the frozen default; fail → the
member-count axis closes at four with two board measurements behind it.

**Before any slot.** `check_recipe --pack` with `MEMBERS="0 1 2 3 4"`, `check_pack` (prediction + intervals) MATCH,
standalone extraction in a fresh process, sentinel: mim1/far_drop five- vs four-member ensemble means ≥ −0.02
(needs `bt_w1_K4B_{mim1,far_drop}_s4`), paired A100 N = 1 latency. Slot: owner's call (earliest 09-13).

## Sentinel read (2026-09-12 02:50 UTC, held-out ensemble means at step 6000, B s0/s1 + K4B s2/s3 [+ s4])
| fold | K=4 | K=5 | Δ |
|---|---|---|---|
| mim1 | 63.1459 (tke 0.4933) | 63.1207 (tke 0.5038) | **−0.025** |
| far_drop | 62.1408 | 62.1451 | +0.004 |
**mim1 breaches the pre-registered −0.02 floor: K5 does not qualify for a slot.** The fifth member's averaging damps
mim1 TKE by 0.0105 (K4's own step cost 0.0045); rel and mvpe improve slightly on both folds. One seed-4 fold member per
fold, so this is a single draw, but the rule was written before the read and stands. The in-sample five-member pack
(`k5_heads.sh`) is finished for the record only; the member-count axis rests at four unless a second, independently
pre-registered seed-4/5 sentinel pair is funded (2 × 2 fold runs ≈ 1 A100-h) and passes.

## In-sample read of the built pack (for the record): the interval gain is absent
`submissions/k5` (252,590,410 bytes, under the 256,000,000 cap; check_pack MATCH). Same windows as K4's read, GPU 3 shared:
rel 95.503 / tke 83.203 / mvpe 95.888 / sps **51.016** vs K4 95.495 / 83.171 / 95.882 / **51.020** — accuracy +0.003…+0.03,
sps −0.004: the fifth member adds no interval information the head can use, in-sample, and the packed call ran at
17–22 ms/sample on the shared card (graph capture of five members plus a busy GPU; not a board-comparable number, but a
warning that the capture budget is tighter at five). With the failed mim1 sentinel this closes the member-count axis
at four: two board points (3→4 = +0.038) and one held-out/in-sample pair (4→5 ≤ 0) behind it. Not uploaded.

## Re-checked with the gain (2026-09-14): a fifth member is dead even there
Five-member fold ensembles exist on mim1 and far_drop (B s0/s1 + K4B s2/s3/s4). Absolute cell-aggregated Q:
| | K4 + gain 0.5 | K5 + gain 0.5 | K4 + full gain | K5 + full gain |
|---|---|---|---|---|
| mim1 | 63.1649 | 63.1603 | 63.1726 | 63.1786 |
| far_drop | 62.1644 | 62.1678 | 62.1752 | 62.1782 |
| mean K5 − K4 | **−0.001** | | **+0.005** | |
Both inside the read's noise. The mechanism is visible and worth recording: K5's *plain* mean is more damped than K4's
(more averaging), so the gain restores more for it (mim1 +0.040 vs +0.019 at dose 0.5) — and lands in the same place.
**The gain makes ensemble size nearly irrelevant on the TKE channel, because it undoes exactly the damping that extra
members cause.** K5 stays closed, now for a second reason and with the gain in play.
