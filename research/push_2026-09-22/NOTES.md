# Push round, 2026-09-22 — what the last five days are for

Joint Claude + Codex (gpt-6-astra, xhigh; brief `codex_brief.md`, report `codex/REPORT.md`, arithmetic
`codex/price_audit.py` → `codex/prices.json`). Owner's decision on the fork: **split the effort**.

## 1. The board moved against us, and the gap is not a thousandth
Re-read 18:45 UTC (`codex/live_board.json`): 1 roysegal 82.10103 · 2 **skabob 81.98770** (uploaded today) ·
3 np-user 81.96840 · 4 modu-lemon 81.91526 · **5 us 81.90253** · 6 JLShen 81.86335 · 7 PhysicsOracle 81.83962 ·
8 doomduke2 81.76869 · 9 andychang 81.75346 · 10 seantang 81.69702. We were rank 4 on 09-20.
**Gap to third 0.0659; cushion above tenth 0.2055.** Three of the top seven uploaded today.

## 2. What third place costs, priced exactly (power mean reproduces the API card to 4.9e-11)
Each row moves one displayed subscore, everything else fixed, to gain +0.065873:

| channel | now | needed | as a change in the underlying error |
|---|---:|---:|---|
| rel | 94.846205 | 94.984189 | **rel error −2.82 %** (−1.92 % if the SPS prefactor follows) |
| sps | 43.975002 | 44.246066 | **W̄ +0.616 %** |
| tke | 79.232868 | 79.885526 | tke error −3.93 % (−2.35 % coupled) |
| time | 92.604684 | 93.290071 | **4.649 → 3.771 ms** |
| mvpe | 94.090975 | 94.779795 | mvpe error −12.3 % (−6.47 % coupled) |

Exact marginals at 934147: rel 0.477444 · sps 0.243221 · tke 0.101048 · time 0.096210 · mvpe 0.095729;
d(final)/dt = **−0.070865 per board-ms**. (Corrections to earlier numbers: W̄ +1 % is +0.1068, not +0.111 — the old
figure kept the retired 0.253 sps coefficient.)

**The whole ledger of measured-positive, unshipped work is +0.013 optimistic and ≈ +0.005 honest.** FM3 +0.0044
(Codex's 0.5×-shrunk S2; +0.0134 at the fixed fold weight) · 12k interval student +0.0007 central (the marginal
precedent past the pair is 0.17×, not 0.81×) · learned variance target **−0.0069** once its TKE funding is removed
properly — the headline +0.012 was the wrong ranking quantity and the arm is rejected, not deferred ·
`k4a2portable` **−0.0245**, because it carries 929146's older weights and forfeits more accuracy than any plausible
timing gain buys. Nothing here reaches +0.066, and no new mechanism survived either pass.

## 3. The clock lottery is now measured, and it is small
Today's slot (939933) repeated 934147's exact bytes: **81.86931 at 5.131 ms**, accuracy identical (rel and MVPE to
every digit; tke/sps landed on 932643's fp16-path digits, Δ ≤ 1.8e-5 = the scorer's own noise).
* The canon chain now reads **4.782 / 4.649 / 5.131** (mean 4.854, sd 0.249) against the 09-13→09-18 chain's
  5.83 ± 0.19: the platform level fell ≈ 1 ms **and** our retained best sits 0.8 sd below the new mean.
* Best-of-five further repeats: **E[gain] +0.0083, P(reaching today's third) 0.00 %** (200k draws).
* **The registered fp32 rule is resolved against fp32** (two fp32 reads 4.649 and 5.131; 5.131 > the 4.782 fp16 min).
  fp16 stays in the frozen method. `CANDIDATE_canonf32rep.md`.

## 4. The fact that reframes the goal
`competition_info/rules.md`, the FAQ and the overview all say the same thing: **the development leaderboard only
picks the shortlist.** Top 10 → organisers retrain each method from scratch → **a private test set with unseen
regimes decides the top 3**. We are 5th with 0.2055 over tenth. And the organisers' own second email
(`competition_info/email2.md`) adds: *"In the final stage the evaluation is not just the static scoring formula; we
also look at how clean and novel the solution is and how well it generalizes."*
So board rank 3 is simultaneously **unreachable with anything measured** and **worth nothing for the prize**, while
the top-10 cushion is worth defending. That is the whole basis for the allocation below.

## 5. The allocation (owner: split the effort)
**Main line — the Decision Phase.**
* The **registered R3 fresh joint geometry** (`research/audit_2026-09-17/codex/PREREG_NEXT.md`), which PLAN still
  lists as owed. Fifteen real cells out, their simulation kept, 66 real left for training — all 15 identities
  asserted present. This is the one geometry the instrument park does not cover: `joint5` withholds only the four
  5-degree frontier cells, and `far_drop` keeps 24150_20 and scores no 5-degree cell, so **neither measures high Re
  and the 5-degree frontier at once** — which is exactly how the rules describe the private set. Added as
  `FOLDS["r3joint"]` in `launch_wave.py` (recipe check passed); four fresh members queued, ~1.8 A100-h.
  It answers the owed pair-vs-student contract clause (R3) and carries the 12k student read (R4).
* `watch_pages.py` polls the official pages every 15 min into `pages/` + `CHANGES.md`. The FAQ says the final
  package format lands "about 5 days before the end of the Development Phase" — i.e. today — and the evaluation
  example is promised before the shortlist is collected. Both are prize-critical and neither has appeared yet.

**Board line — one slot, 09-25, gated.** `CANDIDATE_fm3all.md`: the FM3 component-wise blend built on all data
(all-data flow → distilled sample-mean student → blend at the pre-registered w = 0.3, never refitted). Queued as one
chained A100 job. Registered at **+0.008 [−0.010, +0.020]** accuracy at a fixed clock, with an explicit clock risk of
0 to −0.044 that only the slot can resolve. Build gates are local and must all be green before it may claim a day.

**Slots.** 09-22 spent (939933, above). 09-23/24/26/27 default to the frozen bytes via
`arm_repeat.sh <UTC-date>`, which verifies the sha256, waits for 00:00:30Z, honours a `VETO_<date>` file so a
candidate can take the day, and stands down if the quota is already closed. 09-23 is armed.

## 6. Ideas killed today, for the record
* **A decorrelated donor's spread as an interval-head input.** The head already has the `extra_channels` hook and
  currently receives the *members'* spread, which FINDINGS measures at ≈ 0 because the members agree (ρ 0.94); a
  donor that disagrees informatively looked like the obvious fix, and W̄ +0.62 % is exactly third place. The archive
  had already run it: **"Full-residual flow spread into the head | 43.17 → 43.03, *despite* better corr"**
  (`docs/archive/FINDINGS_2026-09-10.md`). Better correlation with the error does not buy reward under an objective
  this asymmetric — the same shape as per-cell mean error (32.2), quantiles (28.0–35.6) and input variability
  (11.6–26.6) all scoring below a tuned constant band (36.0). Not reopened.
* **Treating each slot as a free +0.03 clock lottery.** The retained best already *contains* a favourable draw, so
  the lottery is conditional, not unconditional; Codex's correction is that the positive part belongs around the
  combined result, `V = E[(F(A,T) − B)+]`, not around the clock term alone. Priced in §3: +0.008 over five slots.

## 7. Build log, 2026-09-22 evening — what got made, and two things I got wrong
**Prize line.** Four fresh members on `r3joint` (1.8 A100-h, recipe check passed), then all six teachers
(seeds 0/11/21/31/41/51) on those members, `--train-split train` so the 15 cells never enter a head fit.
Held-out sps at the constant-band baseline reads 40.6 at coverage 0.941. The 4k and 12k interval students are
distilling now; `research/distil_2026-09-16/measure.py` then scores `h0` / `pair` / `a2` / `a2_12k` on the held-out
cells with official SPS, the gain and the guard — which answers the owed R3 pair-vs-student clause and R4 in one run.

**Board line.** The all-data FM3 pack is built and certified on construction (see `CANDIDATE_fm3all.md`): inert at
`flow_w = 0` to the bit, `check_pack` MATCH with the donor rebuilt from its hash-verified training checkpoint,
252,592,138 B, and TKE identical to three decimals. A held-out donor on `r3joint` is training for the accuracy read.

**Correction 1 — I registered an accuracy gate a build cannot pass.** "Release-set MVPE ratio ≤ 1.000" reads 1.0073
on `val_aoa`, and it had to: `val_aoa` is in-sample for an all-data donor *and* all-data members, so there is no
variance reduction left to trade there. An all-data artefact has no held-out cells by construction, so its local
gates can certify *construction* (inertness, parity, size, TKE invariance) and nothing else. The accuracy decision
moves to the fresh `r3joint` fold, where it always belonged.

**Correction 2 — a leakage guard that was right for every previous fold and wrong for this one.** The round-8 flow
trainer asserts the simulated training set shares no cell with the held-out real cells. Every earlier fold dropped
the simulation for its held-out cells, so the guard was free; the registered R3 geometry deliberately **keeps** it,
because the cells the private set is hypothesised to use are precisely those with simulation and no real
measurement. Dropping it would mimic the wrong thing. The guard is now conditional on `REALISTIX_SIM_KEEPS_CELLS`
and prints what it let through; **the assertion on real trajectories is untouched and absolute**.

**A scare that resolved.** The packed donor differed from the research `Student` by 6.9e-4 relative, which my own
1e-5 threshold called a structural mismatch. Localising it: submission.py's UNet reimplementation is **exact**
(0.0e0), and the whole residual is the modality fold — which a **shipped member reproduces at the same scale**
(3.6e-4 max / 1.4e-5 mean against the donor's 6.9e-4 / 1.4e-5). The donor is treated exactly like the four members
that already pass `check_pack` MATCH. `research/push_2026-09-22/fm3all/{localize,fold_ref}.py`.

**One frozen-tree change.** `tools/check_pack.py` now applies the donor blend to its fp32 reference, exactly as it
already does for `GAIN_DOSE`, rebuilding the donor from the recorded source checkpoint and refusing a digest
mismatch. Verified inert: the frozen pack still reads **MATCH**.

## 8. The night's results, and the one general finding
Three pre-registered reads, all on the **same** fresh 15-cell joint geometry (high Re *and* the 5-degree frontier
withheld together, simulation kept, 292 held-out windows) with fresh members and outer-training-only heads:

| registered item | predicted | measured | verdict |
|---|---|---|---|
| **R3** A2 student − pair | +0.01 [−0.06, +0.06] | **+0.0827 SPS**, 15/15 cells ≥ 0, worst +0.0294 | **PASS** — the student stays the contract head |
| **R4** 12k − 4k student | +0.015 [−0.02, +0.04] | **+0.0336 SPS**, 15/15 cells ≥ 0, worst +0.0126 | **PASS** the +0.010 replacement floor |
| **FM3** donor blend, student, w = 0.3 | continue ≥ +0.010 | **+0.0063 ΔQ** (TKE 1.00000, MVPE 0.990) | **KILL** — no slot, no flow in the package |

**The general finding: the two mechanism families transfer in opposite directions with geometry difficulty.**
Head-side mechanisms are worth *more* on the hard geometry than the easy folds said — the K = 6 increment over the
pair is +0.0961 here against +0.0225 on mim1/far_drop/angle05, and R3 and R4 both came in above their registered
bands. The flow-donor blend is worth *less*: +0.0063 here against +0.0136 there, with the student capturing 60 % of
its teacher instead of 87 %. The mechanism behind the head half is explicit — head **fitting noise** grows on harder
cells, so averaging heads and imitating that average both pay more there — and `FINDINGS.md` §2b already establishes
that the head axis is noise rather than policy. Since the private set is described as *harder* than the dev set, the
honest reading is that the shipped head method is under-priced by the public board and the donor is over-priced by
the easy folds. **Remaining contract risk belongs on the head, not the forecaster.**

Two corollaries worth keeping:
* The 12k student scores **+0.0203 above the six-head average it is fitted to** (the 4k student sits 0.0133 below
  it). Past some horizon, distillation stops being compression and becomes a better estimator than its target — so
  §2b's "head-noise headroom above the student ≈ +0.06 fold SPS" was measured against the wrong reference.
* The all-data FM3 build's MVPE 1.0073 really was an in-sample artefact: the same construction reads **0.990** on
  held-out cells. Ordering the held-out read before the slot was the right call, and the red gate is now retired by
  measurement rather than by argument.

## 9. Freeze-checklist item 5, done for head12k (2026-09-22 23:35 UTC)
`research/push_2026-09-22/standalone_check.sh` extracts a zip to a scratch directory with **no repo on
`PYTHONPATH` and no repo cwd**, then imports and calls `predict(x, {})` in a fresh process, the way the container
does. On `head12k.zip`: extracted **207,810,906 B** of the 268,435,456 cap, top level
`submission.py model.pth rpde_baselines _vendor load_baseline.py`, import 16.55 s (setup at import, which the FAQ
puts outside the timed region), first `predict` 0.013 s and repeated 0.010 s for 8 windows, all three arrays present,
finite, float32, and `lower <= prediction <= upper` everywhere. The script is reusable for whatever artefact
ultimately ships.
