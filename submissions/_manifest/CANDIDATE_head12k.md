# Candidate head12k — the registered R4 contract change, on all data (pre-registered 2026-09-22 22:55 UTC, before the build finished)

**What.** The canonical four members and `k4a2`'s inference code, byte for byte, with **one literal changed**:
`STUDENT --steps 4000 → 12000` (the cosine horizon follows). Same six teachers (`canon0918_head_s{0,11,21,31,41,51}`),
same seed 7, same teacher-0 initialisation, same under-width ×3, same width 96, same architecture, same pack size,
same fp16 payload. **The point forecast is bit-identical, so only SPS can move** — the cleanest channel this project
has, and the one on which 926634 and 929146 both transferred cleanly.

**Why now.** R4 was registered on 09-18 as an add-on to R3 and has been owed since. Read today on the fresh
15-cell joint geometry (`research/push_2026-09-22/R4_RESULT.md`): **12k − 4k = +0.0336 SPS, all 15 cells positive,
worst +0.0126**, inside the registered band (+0.015 [−0.02, +0.04]) and above the +0.010 replacement floor. The old
three-fold estimate this replaces (+0.0169) understated it by 2×, in the same direction and for the same reason as
R3's pair-vs-student read: head *fitting noise* is larger on harder geometry, so averaging — and imitating the
average — pays more there. The 12k student also scores **+0.0203 above the six-head average it is fitted to**, where
the 4k student sits 0.0133 below it.

**Prediction (exact power mean, p = 0.6854, at a fixed clock).** rel / tke / mvpe **identical to every digit** to the
frozen artefact (quarantine the read at > 1e-4 on any of the three — the 1e-5 threshold has been falsified twice by
the scorer's own fp16-path noise). SPS: +0.027 at the head pair's measured 0.81× transfer, +0.006 at the marginal
increment's 0.17×. **Registered N = +0.004 [−0.002, +0.008]**, centre between the two precedents. Clock: unchanged
work — one head of the same architecture — so the draw is the platform's, ≈ 4.85 ± 0.25 from the canon chain.
**Board total ≈ 81.88 [81.82, 81.94].**

**Kill.** Any of rel / tke / mvpe moving > 1e-4 (the point forecast would not be what this candidate claims);
SPS ≤ 0 against 934147's 43.9750022266. A negative read retains the 4000-step student in the contract and closes R4.

**Build gates before the slot.** `check_pack.py` MATCH; `check_recipe.py` must **reject** this student against the
unchanged 4k contract literal (demonstrated, not weakened — the checker is doing its job) and pass only against a
research copy carrying the 12k literal; `tools/tests/mutate_recipe.py` still rejects all 13 mutants; extracted size
unchanged; paired latency within 0.05 ms of the frozen pack.

**Contract cost if it wins.** One literal, about +15–45 min on the organisers' rebuild, no new stochastic stage, no
architecture or size change. This is the cheapest contract change available and the only one with a fresh
private-like read behind it.

## Armed
Not armed. Proposed for **09-24**, after the build gates are recorded green. 09-23 remains the frozen-byte repeat.

## Build gates, 2026-09-22 23:05–23:25 UTC — all green
Zip `build/head12k.zip`, sha256 `c65a59e0c8700e39cb2fc72b0b0fbedea5140102a3e77d3b50073dcb6a07cda4`, 191,746,513 bytes; extracted 207.8 MB
of the 268 MB cap — **byte-for-byte the same size as the frozen pack**, as an unchanged architecture requires.

| gate | result |
|---|---|
| point forecast unchanged | **`prediction` bit-identical** to 934147 on 48 windows (max abs 0.0); `lower`/`upper` move (7.9e-3 / 4.0e-3 max) — exactly the claimed channel |
| `check_pack` | **MATCH** against the head checkpoint `canon0918_student12k` |
| `check_recipe` against the **unchanged 4k contract** | **correctly FAILS, exit 1**, naming precisely the two intended disagreements: `--steps: script 4000, shipped run 12000` and `the student checkpoint is step 12000, the script trains 4000`. The frozen pack still passes the same checker, **exit 0**. The checker was not weakened |
| `tools/tests/mutate_recipe.py` | **14/14 mutants behaved as required** (run because `check_pack.py` was modified today — the standing rule after metadata-only checks once passed a zeroed head) |
| paired latency (same process, interleaved, 3 cycles × 20 calls) | N=1 **+0.0215 ms**, N=8 **−0.0002 ms** — inside the registered 0.05 ms gate |

## Armed for 2026-09-23
Promoted from the 09-24 slot to **09-23**: every gate is green, and a slot carrying a candidate with positive
expected accuracy dominates a frozen-byte repeat, whose own expected gain is ≈ +0.0017. `VETO_2026-09-23` stands the
repeat down; `research/push_2026-09-22/arm_head12k.sh` waits for 00:00:30Z and verifies the sha256 before sending.
**If the read is negative, the 4000-step student is retained and R4 closes** — the contract literal is not changed on
a board result alone, and the fresh-geometry read (+0.0336 SPS, 15/15 cells) is the evidence that would have to be
weighed against it.

## Upload 1 — 940119, 2026-09-23T00:00:59Z: **FAILED on the platform, not on us**
`Failed | Transient platform storage error while downloading the 'submission' bundle; not a problem with your
submission. Please resubmit.` Quota re-checked immediately afterwards: **`can: True`** — the FAQ's rule that a
Failed upload does not consume the day holds, so the 09-23 slot is intact and no read was lost. The FAQ asks for a
resubmit about an hour later so the platform can recover; armed for **01:20 UTC**
(`research/push_2026-09-22/arm.sh`, same zip, sha256 re-verified before sending).

**Tooling consequence.** The arming scripts used until now treated "the platform returned an id" as success and
stopped polling at any terminal state. That is wrong in exactly this case: a slot is only really spent when a
submission reaches **Finished**. `arm.sh` supersedes them — it polls to a terminal state, and on `Failed` waits an
hour and resubmits until its deadline. With four slots left and the Development Phase closing 09-27, a silent lost
day near the end would have been expensive.

## Upload 2 — 940198, 2026-09-23T01:20:11Z, scored 01:25 UTC: **N = +0.00637, inside the registered band**
| | rel | tke | mvpe | sps | clock | final |
|---|---|---|---|---|---|---|
| 934147 (4000-step student, retained best) | 94.8462054793 | 79.2328683384 | 94.0909748382 | 43.9750022266 | 4.649 ms | 81.90253 |
| **940198 (12000-step student)** | 94.8462054793 | 79.2328645965 | 94.0909748382 | **44.0012119293** | 4.754 ms | 81.90147 |
| Δ | **0.0000000000** | −3.7e-6 | **0.0000000000** | **+0.0262097** | +0.105 | −0.00106 |

**The candidate's central claim is confirmed to every digit: rel and MVPE are unchanged, TKE moves 3.7e-6 (the
scorer's own fp16-path noise, twice measured at ≤ 1.8e-5), and the entire effect is +0.0262 SPS.** The point
forecast really is bit-identical, so this is the cleanest single-channel read the project has taken.

**Accuracy-only N at a fixed clock = +0.00637**, against a registered **+0.004 [−0.002, +0.008]** — inside the band,
above its centre. The observed final is −0.00106 *only* because the clock drew 4.754 against the incumbent's
favourable 4.649; at equal clocks this artefact is strictly better. The retained board best therefore stays
81.90253, and that is a statement about a platform draw, not about the method.

**The fresh-geometry instrument transferred at 0.78×** (fold +0.0336 SPS → board +0.0262), almost exactly the head
pair's measured 0.81× (fold +0.0998 → board +0.0807). Two independent head-side mechanisms have now transferred at
≈ 0.8× from held-out folds to the board, which is the strongest calibration this project has for the channel.

**Verdict: the registered R4 replacement rule is satisfied on both instruments** — fresh joint geometry +0.0336 SPS
with 15/15 cells positive, board +0.0262 SPS with the point forecast bit-identical, parity/size/latency gates green.
Nothing here is a judgement call: the kill conditions were fixed on 09-18 and none of them fired.

## Open issue — the `--replay-student` gate fails on the H100-built student (2026-09-23 02:10 UTC)
With the contract literal at 12000, `check_recipe --replay-student` on `submissions/head12k` reports
**80 of 80 student tensors differ from the replay**. Everything else in the same run is green: all four members and
the head are bitwise identical to their folded fp16 sources, the teacher digests and seeds match, and the data
manifest matches the recorded release.

**The likely cause is card class, not a defect.** `canon0918_student12k` was distilled on the **local H100**
(`CUDA_VISIBLE_DEVICES=3`), because that was the GPU free at 23:00; every canonical artefact in this project —
including the 4000-step student that replays bitwise 80/80 — was built on a **pq A100-PCIE-40GB**. The checker's own
message says as much ("same card class required; a different class gives a different model"), and `REPRODUCE.md`
records that a different class gives a different model with no tensor equal at the same seeds. The checkpoint format
does not record which GPU built it, which is why this was not caught before the pack was made.

**Being tested rather than assumed:** the 12000-step student is being rebuilt on a pq A100 (`s12k_a100`,
node10). If it replays bitwise there, the gate is card-class-bound as documented and the A100 build becomes the
contract artefact. If it does not, there is a real reproducibility defect in the 12k path and **the contract change
must be reverted** — a method the organisers cannot rebuild is worth less than nothing, whatever its SPS.

**What this does and does not affect.** The board read (940198, +0.0262 SPS, N = +0.00637) is evidence about the
*mechanism* — 12000 distillation steps beat 4000 — and that evidence stands; the fresh-geometry read (+0.0336 SPS,
15/15 cells) was measured on entirely separate artefacts and is untouched. What is in question is only whether
*those particular bytes* are reproducible, and the artefact that ships must be the one that replays.

### Replay diagnosis, 2026-09-23 04:30 UTC — the first hypothesis is refuted, and the test was not controlled
The 12000-step student was rebuilt on a pq A100 (`canon0918_student12k_a100`) and compared to the H100 build:
**0/80 tensors bit-identical, max relative elementwise difference 0.219** — the documented card-class behaviour,
so that much is confirmed. But the A100-built student **also fails** `--replay-student`: again 80 of 80 tensors
differ. **Card class alone does not explain the failure.**

Then the node IDs, which I should have checked before drawing any conclusion: the build ran on **node10** and the
replay on **node7**. Both are "A100" to `pq -t a100`, but they are different physical machines and nothing here
pins the A100 variant, driver or cuDNN version. So the second test was not a controlled card-class test either — it
compared two different nodes, exactly like the first.

**The control that decides it** (`ctl4k`, running): rebuild the **4000-step** student from the same canonical
teachers and compare with the shipped `canon0918_student`, which `STATE.md` records as replaying bitwise 80/80.
* If the 4k student reproduces across nodes and the 12k one does not, the fault is in the 12k horizon and the
  contract change is reverted.
* If the 4k student does **not** reproduce either, then cross-node bitwise reproduction is simply not a property
  this project has — it was only ever verified on one machine — and the standing claim in `STATE.md` and
  `REPRODUCE.md` is narrower than written. That would matter well beyond this candidate, because the Decision Phase
  rebuild happens on the organisers' hardware, not ours.
The second outcome would be the more important finding, and it is not the one I would prefer.
