# Candidate: mvpe03 — members retrained with the MVPE loss term at 0.3 (2026-09-04)

**What it is.** The shipped recipe (`tools/train_final.sh`, three width-96 members, Muon,
EMA 0.998, `rel 8 + tke 1 + sps 1 + mse 1 + tmean 0.5 + spec 0.1`, step 6000; head at
4000/4000 with the spread input, as the script's `HEAD` literal says) with one change:
`--w-mvpe 0.3` (`W_MVPE=0.3`, an append-only knob added today). Trained on an A100
(node10, queue job `1788503616485-mvpe03`) — not the H100 the champion came off, so
the weights will differ from a matched-hardware retrain; the cross-hardware check
(REPRODUCE) put that at 0.004 of final. Packed with **cugraph16t's inference code**
(fp16 channels_last graphs + fused kernel + shipped Triton cache), so the timing side is
the 09-06 candidate's and only the members and head change.

**Evidence.** Four mimic folds × two seeds (2026-09-01 remeasure campaign): at dose 0.3
`d_mvpe +0.030 ± 0.006, 8/8`, rel and tke flat; dose 0.6 adds nothing (+0.034) and tke
starts to pay. The composed in-grid tail gate read −0.003 tke-zeroed (an sps cost at
ensemble level; reproduced to 0.000 after the tail-head fix). So this is a pure
mvpe-channel gain whose board value rides on the ~5× mvpe transfer coefficient — the
least controllable channel — against a small possible sps cost.

**Prediction, recorded before the score.** Against the best-of-the-week reference (the
09-06 read, or 914270 if the timing candidates fail): mvpe +0.03 local × ~5 = +0.15 board
× 0.103 = **+0.015**, sps ~0 to −0.02, time unchanged (same inference code, shape-identical
trunk), plus a retrain-noise term of ±0.015. **Central +0.01, band [−0.05, +0.05].**
Falsifier: a read below the band floor closes the mvpe term for good; a read inside the
band is uninformative on its own and the mvpe subscore is the number to read.

**Verification gates (automated after training, results appended below):** `check_pack`
MATCH against the three member runs; in-sample `run_and_score` on `val_aoa` finite with
rel_l2 > 95.0, tke > 82.0, mvpe > 95.0, sps > 50.0 (the champion's in-sample read is
95.474 / 83.272 / 95.815 / 51.141 — a retrain with a new loss term may move these by
~0.1); zip under the 256 MB cap. Armed for **2026-09-07T00:00:30Z** only if every gate
passes.

## Verification (automated, 2026-09-04 07:38Z) — gates passed, armed for 2026-09-07T00:00:30Z

Training job `1788503616485-mvpe03` finished rc=0 on node10 (A100). Pack members
`mvpe03_m0/m1/m2`, head with spread input. `check_pack`: **MATCH** (max abs 2.2e-4, mean
relative 2.9e-4 against the run checkpoints). Inference code and Triton cache copied
from `cugraph16t`. In-sample `run_and_score` on `val_aoa` (a build check, not a quality
estimate; the champion reads 95.474 / 83.272 / 95.815 / 51.141 on the same set):

| rel_l2 | tke | mvpe | sps | ms/sample |
|---|---|---|---|---|
| 95.483 | 83.223 | **95.902** | 51.008 | 0.94 |

mvpe +0.09 in-sample, tke −0.05, sps −0.13 — the expected signature of the term (mean
profile bought, a little fluctuation and coverage paid). Zip `build/mvpe03.zip`, 164.8 MB
extracted. Log `finish_mvpe03.log` next to the runs directory.


## Re-packed 2026-09-06 07:58Z as `mvpe03b` — inference code swapped to cugraph16's; prediction revised

916264 (cugraph16t) showed the container takes the Triton fallback and reads **slower** than cugraph16
(6.07 vs 5.34 ms/sample, accuracy identical to ten decimals; post-mortem in `CANDIDATE_cugraph16t.md`). mvpe03
was packed with that same `submission.py` (md5 `54dce1b0…`). To keep the slot's read clean, `submissions/mvpe03b/`
= mvpe03's `model.pth` (md5 `c4256c21…`, unchanged) + `submissions/cugraph16/submission.py` (md5 `249fd785…`,
the proven 5.34 ms path). Verified: local kit scores at chunk 1 identical to mvpe03's to the last digit
(rel 95.483 / tke 83.223 / mvpe 95.902 / sps 51.008), the zip smoke-tests standalone from a temp dir (setup 9 s,
graphs on, fused path absent by construction), 155 MB extracted. Zip `build/mvpe03b.zip`
(md5 below); the original `mvpe03.zip` is kept, unsubmitted. The armed process was replaced:
`submit_at.sh 2026-09-07T00:00:30Z build/mvpe03b.zip`, log `submit_mvpe03b.log`.

**Prediction, revised before the score.** Time should reproduce 915299's regime (5.3 ms → 92.1) — but 916264 read
6.07 on code whose timed path is byte-identical to cugraph16's, so day-to-day drift of ±0.5 in `time_score`
(≈ ±0.05 final) is live and the time channel cannot be predicted tighter than that today. Accuracy: the
pre-registered +0.01 [−0.05, +0.05] on the mvpe channel stands. **Central +0.01, band [−0.08, +0.06] against
81.75365**, with the time half of the band explicitly unresolved. Falsifier for the *time* question this slot now
also answers: `time_score` ≈ 91.6 again on cugraph16's own code → 916264's regression was platform drift, not
Triton, and the shipped 5.3 ms is not stable; ≈ 92.1 → the regression was Triton-linked and the fallback costs
~0.7 ms in the container.
201e9f79f01398de964e6ec8c35c4323  build/mvpe03b.zip


## Scored 2026-09-07 (917205, `mvpe03b`): **81.76275 = +0.0091 against 915299 — new best; prediction +0.01 [−0.08, +0.06] hit**

| channel | 915299 cugraph16 | 917205 mvpe03b | Δ | × weight |
|---|---|---|---|---|
| rel_l2 | 94.8238 | 94.8377 | +0.0138 | +0.0066 |
| tke | 79.0584 | 79.1766 | +0.1182 | +0.0125 |
| mvpe | 93.9106 | 93.9578 | +0.0472 | +0.0044 |
| sps | 43.7451 | 43.7253 | -0.0198 | -0.0050 |
| time | 92.1128 | 92.0179 | -0.0950 | -0.0090 |

Accuracy-only (rel + tke + mvpe) **+0.0235**, sps −0.005, time −0.009. The `w_mvpe 0.3`
members pay on all three accuracy channels — tke most (+0.118), then mvpe (+0.047) and rel (+0.014) — and the head
gives 0.02 of sps back; net +0.019 of accuracy-plus-sps against a −0.009 time read. Board record: the fourth positive of the
cugraph line (912717 → 914270 → 915299 → 917205).

**The time falsifier from the re-pack is answered.** cugraph16's inference code read **5.48 ms/sample (92.018)**
today against 5.34 yesterday on the same code and 6.07 the day before with the Triton import: the 916264 regression
was Triton-linked (≈ 0.6 ms of its 0.73), with ≈ 0.14 ms of day-to-day drift on top — inside the record's band.
Triton fusion stays closed; the cugraph16 inference path is the one to ship and to package for the Decision Phase.

**Board, 2026-09-07 ~01:30Z: rank 1 of 50** — 81.76275 against np-user's 81.76136 (gap +0.0014, i.e. one
time-noise band) and doomduke2's 81.74725; the #10 cutoff is 81.389 (margin 0.37). The lead is inside the board's
own reproducibility floor (0.015), so it is a tie for first that the next accurate read on either side will move.
