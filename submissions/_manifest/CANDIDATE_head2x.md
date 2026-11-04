# Candidate head2x — more teachers and a longer student (pre-registered 2026-09-24 05:35 UTC, before any run)

**Why this axis.** It is the only one in the record where measured gains transfer: two independent head-side
mechanisms went fold → board at 0.81× (pair) and 0.78× (R4), and on the registered fresh joint geometry `r3joint`
head mechanisms read *above* their bands (head fitting noise grows on harder cells; `research/push_2026-09-22/NOTES.md`
§8). The point forecast is bit-identical by construction, so only SPS can move.

**Arms (all on `r3joint`: its four fresh members, `--train-split train`, 15 held-out cells, 292 windows).**
* **T12-12k** — six more teachers (seeds 61 71 81 91 101 111, the teacher recipe byte for byte) → one student at
  12000 steps on all twelve. Everything else = the contract STUDENT literal.
* **T6-24k** — the six existing teachers → one student at 24000 steps.
* **T12-24k** — twelve teachers, 24000 steps.
* Reference: the measured `a2_12k` (6 teachers, 12000 steps), gained sps **46.9645**.

**Predictions (gain 0.5, official SPS, vs a2_12k).** Under 1/K head noise, h0 → k6 = +0.2055 implies G∞ ≈ 0.247 and
k12 − k6 ≈ +0.021 for the teacher average; the student carries ~all of it. T12-12k **+0.015 [−0.005, +0.030]**;
T6-24k **+0.008 [−0.010, +0.020]** (4k → 12k was +0.0336; returns diminish); T12-24k **+0.020 [0, +0.040]**.

**Gates (fixed now).** Adopt an arm into the contract only if **mean ΔSPS ≥ +0.010 with no cell < −0.030** (R4's
floors), rel/tke/mvpe identical to the digit (else quarantine). Kill: mean ≤ 0 or any cell < −0.030. Between:
documented, no change. If two arms pass, take the cheaper retrain unless the dearer one is ≥ +0.010 above it.
Contract cost if adopted: T12 adds 6 × ~8 A100-min of teachers; 24k adds ~25 A100-min of student. Inference cost,
size and latency unchanged (still one head).

**Honest scale.** +0.02 SPS ≈ +0.005 final. This does not close the 0.2 gap to first; it is the one lever with a
positive transfer record, and it is worth more on the harder private set than on the board.

## Read, 2026-09-24 ~07:10 UTC (`research/final_2026-09-24/measure_r3joint_head2x.json`, same instrument as R3/R4)
| variant (gain 0.5) | SPS | Δ vs a2_12k | worst cell Δ | cells < 0 | verdict |
|---|---|---|---|---|---|
| a2_12k (reference, 6 teachers, 12k) | 46.9645 | 0 | — | — | |
| k12 teacher average (12 heads, output mean) | 46.9613 | −0.0032 | −0.186 | 4 | reference only |
| **T12-12k** | 46.9712 | **+0.0067** | −0.005 | 4 | **BETWEEN** — below the +0.010 floor, no contract change |
| **T6-24k** | 46.9055 | **−0.0590** | −0.251 | 13 | **KILL** |

rel / tke / mvpe identical to the digit in every variant, as required. K6 → K12 moved the *teacher average* by +0.017
(46.9442 → 46.9613, the predicted +0.021 almost exactly) but the 12k student already sits above even the 12-teacher
average, and a student of twelve adds only +0.0067 over a student of six. **24k steps is unstable at lr 8e-4**: both
24k runs spike at step ~8000–8500 (loss 0.003 → 0.89 / 0.10; the cosine is still at 75 % of peak there, against 25 %
for the 12k schedule at the same step) and T6-24k never recovers the lost ground. The head axis is harvested at the
contract's 6 teachers × 12k steps; T12-24k is read below for the record.
| **T12-24k** | 46.9645 | **+0.0001** | −0.083 | — | **KILL** (a cell below −0.030) |

**Closed 2026-09-24:** all three registered arms fail their floor; no contract change. The 24k runs lose to their own
lr schedule, not to the target; a 24k student would need its own peak lr, which is a new screen, not this one.
