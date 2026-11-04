# Candidate fm3all — the FM3 component-wise flow blend, built on all data (pre-registered 2026-09-22 21:00 UTC, before the build)

**Owner's decision 2026-09-22:** split the last five days — the Decision-Phase package and the registered R3 fresh
joint geometry are the main line; FM3 is built on spare A100s and gets **one slot, 09-25**, if and only if it clears
the gates below. The frozen method (934147 bytes) remains the deliverable unless the board confirms it.

**What.** The FM3 design read BETWEEN on 2026-09-21 (`CANDIDATE_fm3.md`: fold ΔQ +0.0136, every fold positive, TKE
invariant by construction, 87 % teacher capture at 1.78 ms/window) and was never built on all data or read on the
board. This candidate is that build:
1. an FM1 conditional flow trained on the **whole release** (81 real + 100 sim, every `REALISTIX_HELDOUT_*` emptied —
   note `train_final.sh` *unsets* them, which restores splits.py's research defaults, so "all data" needs them set
   empty; asserted 81/100 before launch), seed 0, 8000 steps, the round-8 recipe unchanged;
2. its K = 16 / S = 8 sample mean distilled into one width-96 student (the round-11 recipe unchanged: 4000 steps,
   stride-8 windows, EMA 0.995, fixed final EMA, no selection);
3. the shipped forecast blended with it **component-wise at the pre-registered w = 0.3** — the per-cell time-mean
   blended directly, the fluctuation *pattern* blended and renormalised to the shipped forecast's own per-cell
   variance, so **TKE is invariant by construction** and MVPE moves only through the time-mean.
**`w = 0.3` is fixed now and will not be refitted after any read.** Everything else in the pack — four canonical
members, the distilled interval head, gain dose 0.5, the fp16 payload, `k4a2`'s inference code — is untouched.

**Build gates (local, all before any slot is claimed).** Kill the candidate, do not weaken the gate:
* release-set TKE ratio within **1e-3** of 1 (the construction must hold; round 11 measured 3.5e-8 on folds);
* release-set MVPE ratio **≤ 1.000**;
* extracted pack **≤ 268,435,456 bytes** (arithmetic says 252,656,570 with the fp16 student — 15.8 MB of headroom,
  but the certificate is the measured export, not the arithmetic);
* standalone extraction, fresh-process, repeated-call and 180 s-wall checks pass; `check_pack.py` MATCH;
* paired same-process latency measured and recorded against the frozen pack. **Kill above +2.5 ms/window**, which is
  40 % beyond the student's measured 1.78 ms and therefore a fault detector, not an allowance.

**Prediction (exact power mean, p = 0.6854).** Accuracy-only N at a fixed clock: **+0.008 [−0.010, +0.020]**. The
band's low end is Codex's honestly-shrunk S2 estimate (+0.0044 at 0.5× transfer) stressed for the dose-1.0 precedent;
the high end is the measured fixed-w fold ΔF (+0.0134) transferring at 1×. rel is expected to carry it, MVPE to move
≤ ±0.004, **TKE to be invariant to 1e-3**, SPS to follow rel through the prefactor.
**Clock.** The student adds 1.78 local ms/window. The board slope on added *graph-captured* work is not identified:
K3 → K4 added a whole member (≈ +1 local ms) for board time **−0.001**, and the two-head pair (+1.05 local ms) read
*between* the single-head cards — both say ≈ 0; the P0/P1 probe line says 0.36 board-ms per local ms. The honest band
is therefore **0 to −0.044** of final, and at the pessimistic end it erases the accuracy gain. This is the candidate's
main risk and it can only be resolved by the slot.
**Registered total: 81.90 + (−0.05 … +0.02), centre +0.005.**

**Kill rules.** Board N ≤ 0 → FM3 closes for this competition and does not enter the package. An unexplained TKE move
> 1e-3 or an MVPE ratio > 1.002 quarantines the read (the construction would be falsified, not the idea). A clock
above 7.0 ms with positive accuracy is a *time* failure, not an accuracy one, and is recorded as such.

**Contract cost if it wins.** Two extra training stages for the organisers (a flow teacher at ~0.5 A100-h, then seeded
K16/S8 target generation and a 4000-step student at ~1 A100-h) and one extra model in the pack. Retrain variance for
those stages is **unmeasured**. That cost is why this needs a positive board read before it may enter the package,
and why a merely-neutral read is a rejection rather than a tie.

## Armed
Not armed. The 09-23 slot is held by the frozen-byte repeat (`research/push_2026-09-22/arm_repeat.sh`, veto-file
hook). FM3 may claim 09-25 only after every build gate above is recorded green.

## Build, 2026-09-22 21:00–23:30 UTC — three gates green, one red, and the red one is not decisive
Artefacts: all-data flow `fm3all/flow_all_s0/step8000.pt` (8000 steps, A100, 1225 s,
81 real + 100 sim asserted before launch) → all-data student `.../all/student/step4000.pt`
(sha256 `230335d1…8d01`, fixed final EMA, 4000 steps, 1364 s). Pack `submissions/fm3all/`, built by
`research/push_2026-09-22/fm3all/build_pack.py` (the donor folded and halved by `build_ckpt.py`'s own helpers).

| gate | registered | measured | |
|---|---|---|---|
| inertness at the default | bit-identical | **max abs 0.0 on all three arrays** over 48 windows, `flow_w = 0` | ✔ |
| donor reconstruction | — | submission.py's UNet is **exact** (0.0) against the research `Student`; the residual is the modality fold, and a **shipped member reads the same** (3.6e-4 max / 1.4e-5 mean vs the donor's 6.9e-4 / 1.4e-5) | ✔ |
| `check_pack` | MATCH | **MATCH** (max abs 2.96e-4 vs the frozen pack's 2.12e-4), with the donor **rebuilt from its hash-verified training checkpoint**, not read out of the pack | ✔ |
| extracted size | ≤ 268,435,456 B | **252,592,138 B** (15.8 MB spare) | ✔ |
| TKE ratio | within 1e-3 of 1 | **83.074 vs 83.074 — identical to three decimals**; the construction holds | ✔ |
| MVPE ratio | ≤ 1.000 | **1.0073** on `val_aoa` | ✘ |

**The red gate, honestly.** On `val_aoa` the blend reads rel 95.423 → 95.410, mvpe 95.840 → 95.811, sps 50.955 →
50.930, tke unchanged: flat to slightly negative on every channel the blend can move. **But `val_aoa` is in-sample for
an all-data donor and all-data members alike**, and an in-sample read cannot separate "the donor's time-mean is worse"
from "two models both fitted to these cells have no variance reduction left to trade" — the second is the expected
result and is consistent with the fold reads, not contrary to them. Writing "release-set MVPE ratio ≤ 1.000" into the
build gates was my error: it is an *accuracy* gate, and this build has no honest accuracy instrument, because an
all-data artefact has no held-out cells by construction. The construction gates (inertness, parity, size, TKE
invariance) are what a build can actually certify, and those are green.

**Consequence: the slot is not claimed on this evidence.** A held-out donor is being trained on the registered R3
fresh joint geometry (`fm3_r3joint`, queued 23:18 UTC) — high Re *and* the 5-degree frontier at once, the hardest and
most private-like geometry available. That gives FM3 a fourth fold read, on fresh members, against the case the rules
describe. **The 09-25 slot is contingent on that read: mean ΔQ ≥ +0.010 with MVPE ratio ≤ 1.000 and TKE invariant,
or the candidate closes.** No refit of `w` either way.

## Read on the fresh joint geometry, 2026-09-22 23:20 UTC — **KILL**, by the gate registered before the donor was trained
Donor trained with the R3 15 cells withheld (flow 8000 steps, student 4000 steps, sim kept as registered), scored on
the same 292 held-out windows and 15 cells as R3, equal-cell aggregation, the shipped construction:

| arm | ΔQ | rel ratio | tke ratio | mvpe ratio |
|---|---:|---:|---:|---:|
| teacher K16/S8, w = 0.2 | +0.0098 | 0.99733 | 1.00000 | 0.98632 |
| teacher K16/S8, w = 0.3 | +0.0105 | 0.99757 | 1.00000 | 0.98292 |
| teacher K16/S8, w = 0.5 | +0.0022 | 1.00175 | 1.00000 | 0.98298 |
| distilled student, w = 0.2 | +0.0064 | 0.99816 | 1.00000 | 0.99176 |
| **distilled student, w = 0.3 (what ships)** | **+0.0063** | 0.99846 | **1.00000** | 0.99036 |
| distilled student, w = 0.5 | −0.0013 | 1.00187 | 1.00000 | 0.99290 |

**Gate: mean ΔQ ≥ +0.010 with MVPE ≤ 1.000 and TKE invariant. Measured +0.0063. The gate fails; FM3 closes.**
Two of three conditions are met and met well — **TKE is invariant to five decimal places**, exactly as the
construction promises, and MVPE *improves* (0.990). The third is simply not there.

**The in-sample MVPE scare was an artefact, as suspected.** The all-data build read MVPE 1.0073 on `val_aoa`; on
genuinely held-out cells the same construction reads **0.990**. That vindicates the reasoning for ordering this read
before the slot, and retires the red gate as a measurement rather than leaving it hanging.

**What changed against the old folds, and why it matters more than the kill.** FM3 read +0.0136 on
mim1/far_drop/angle05 and **+0.0063** here; the student captures **60 %** of its teacher against **87 %** there.
So the flow-donor blend is worth *less* on the harder, more private-like geometry — while R3 and R4, both head-side,
are worth *more* there (pair-vs-student +0.0827 against a predicted +0.01; 12k +0.0336 against +0.0169). **The two
mechanism families transfer in opposite directions with geometry difficulty**, which is a sharper statement than
either read alone, and it is an argument for spending the remaining contract risk on the head rather than the
forecaster. Recorded in `research/push_2026-09-22/NOTES.md` §8.

**Disposition.** No slot on 09-25 for FM3, no flow model in the package, `w` not refitted. The artefacts stay for
the record (`submissions/fm3all/`, 252,592,138 B, all construction gates green) as the working demonstration that a
distilled flow donor can be blended at exactly invariant TKE — the transferable design, now with a fourth and
hardest fold behind its price.
