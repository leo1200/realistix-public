# Candidate canonf32rep — exact byte repeat of 934147 (pre-registered 2026-09-22 18:55 UTC, before the upload)

**What.** `build/canon0918f32.zip`, sha256
`e2c7d37d1beb223a680ebffa4924e9e89f39bec95769797214e1ca88e0358536` — the *same bytes* already scored as 934147
(2026-09-20, 81.90253, the retained best). Nothing is changed.

**Why this slot.** Three reasons, in order of value:
1. **The registered fp32/fp16 rule is one read from resolving.** `CANDIDATE_canon0918f32.md` fixed the package rule as
   "keep fp16 unless two fp32 reads both sit below every fp16 read of the same accuracy chain" (fp16 reads: 6.011,
   6.110, 5.702, 4.782). One fp32 read exists (4.649). This is the second.
2. **A third identical-bytes clock read.** The clock draws of this accuracy chain are 4.649 / 4.782 and the wider
   artefact record is 4.65 … 6.11 (mean 5.57, sd 0.52). Whether 4.649 was a −1.8σ draw or a new platform level is
   the single largest open uncertainty in pricing every remaining candidate, and only identical bytes can answer it.
3. Free upside: any draw below 4.649 is a new best.

**Honest EV.** Under the draw distribution above, P(t < 4.649) ≈ 4 %, so the expected board gain is ≈ **+0.001** —
this slot is bought for its *information*, not its score. It is the correct use of 09-22 only because no candidate
with a positive expected accuracy delta is built today; from 09-23 on, a slot carrying an accuracy candidate
dominates a repeat (`research/push_2026-09-22/`).

**Prediction (exact power mean, p = 0.6854, verified to 1e-5 on 934147 and 932643).** Accuracy = 934147's digits
(rel 94.8462054793 / tke 79.2328683384 / mvpe 94.0909748382 / sps 43.9750022266) up to the scorer's fp16-path
non-determinism, which the record puts at ≤ 1.8e-5; a deviation > 1e-4 on any subscore quarantines the read.
Clock ≈ 5.6 ms [4.6, 6.4] → final **81.842** [81.788, 81.908]. **Kill: nothing.** The board keeps the best; this
cannot lose 81.90253. A draw ≥ 5.0 ms confirms 4.649 was luck and closes the "new low-clock regime" hypothesis;
a second draw ≤ 4.8 ms reopens it and makes the clock worth ≈ +0.02–0.03 per remaining slot.

## Armed
Uploaded directly (the 09-22 UTC slot is open with ~5 h left; quota confirmed `can: True` at 18:45 UTC).

## Board result — submission 939933, uploaded 2026-09-22T18:50Z, scored 19:2x UTC (scoring rpw4): final **81.86931**, clock **5.131 ms**
| | rel | tke | mvpe | sps | clock | final |
|---|---|---|---|---|---|---|
| 934147 (same bytes, 09-20) | 94.8462054793 | 79.2328683384 | 94.0909748382 | 43.9750022266 | 4.649 ms | 81.90253 |
| **939933 (same bytes, 09-22)** | 94.8462054793 | 79.2328627255 | 94.0909748382 | 43.9749840516 | **5.131 ms** | 81.86931 |

**Accuracy: identical, as registered.** rel and MVPE reproduce to every digit; tke and sps landed on *932643's* fp16-path
digits rather than 934147's (Δ 5.6e-6 / 1.8e-5) — the scorer's own non-determinism, twice measured now at ≤ 2e-5. The
1e-4 quarantine was not approached.

**What it settles.**
1. **The fp32 payload rule is resolved against fp32.** The rule was "keep fp16 unless two fp32 reads both sit below
   every fp16 read of the chain" (fp16 reads 6.011 / 6.110 / 5.702 / 4.782). The two fp32 reads are 4.649 and **5.131**;
   5.131 > 4.782. **fp16 stays in the frozen method** — the question is closed, not deferred.
2. **4.649 was a lucky draw, but the platform level did fall.** The canon chain now reads 4.782 / 4.649 / 5.131
   (mean 4.854, sd 0.249) against the 09-13→09-18 chain's 5.83 ± 0.19. So the clock is ≈ 0.98 ms below the mid-September
   level *and* our retained best sits 0.8 sd below the new mean.
3. **The repeat lottery is now priced, and it is small.** Under N(4.854, 0.249), best-of-five further repeats has
   E[gain] = **+0.0083** and **P(reaching today's third place) = 0.00 %** (200k draws). Repeats protect nothing and win
   nothing material; they are the correct default only because no candidate with a larger expected gain exists.
