# Candidate: nospreadhead — the shipped artefact, head without the spread input (2026-09-01)

**What it is.** 904950's three members, untouched, bit-identical. One change:
the interval head retrained without the ensemble-spread input channels
(`--no-spread`; shipped flags otherwise: 4000/4000, lr 8e-4, tau 0.05,
train-split all, width 96). The spread input was the head's original design
premise; `spread_info.py` measured its capture fraction at 9% on ranking, and
removing it *helps*.

**Evidence, three instruments agreeing:**

| instrument | read |
|---|---|
| hk4 folds (4 mimic folds) | +0.042 sps local, 3/4 positive, +0.0106 final on the dR_bin channel |
| tail pair (hp_cur members, head-only swap) | **+0.161 sps**, accuracy channels bit-identical — **re-verified 2026-09-01 after the tail-holdout fix: +0.181**, see below |
| spread capture (spread_info) | 9% of ranking information — the input was nearly dead weight |

**Why it passes the tke-zero rule trivially:** the members are the shipped ones,
so rel/tke/mvpe cannot move — the candidate is funded entirely by the interval
channel, whose head-config transfer coefficient is the measured **1.22 ± 0.32**
(precedent: sps_bulk predicted +0.05, paid +0.057). 910411's board sps collapse
is exonerated as the member count: at three members the no-spread head is
positive on both instruments.

**Prediction, recorded before the score.** vs 904950 (81.597):
**+0.02 central, band [0.00, +0.06]** → board ≈ **81.62 [81.60, 81.66]**.
Time expected unchanged (head input channels drop 120→80; the timed trunk is
identical). Falsifier: a read below 81.597 means the interval-channel
coefficient does not cover input-feature removal; restore the spread input and
close the axis.

**Second in line** (not composed here): w_mvpe 0.3 — fold channel +0.030 ± 0.006
(8/8) with rel/tke flat, dose curve flat to 0.6 with tke starting to pay;
tail gate tke-zeroed −0.003. Its board case rides on the mvpe fold→board
amplification and needs its own slot after this one.

**Build verification (2026-09-01).** Pack members tensor-identical to the
904950 champion pack (`submissions/tke1em8ff/`, 164/164 per member); the
in-sample paired read against it: rel/tke/mvpe **identical to the last digit**,
sps +0.005. In-sample parity + out-of-sample gains is the expected signature of
the mechanism (in-sample the members agree, the spread input sits near zero and
the head learns to lean on it; out-of-sample it grows and misleads the widths).
Discovered en route: `submissions/all82_spread/model.pth` is still the
887158-era artefact — the champion pack lives in `submissions/tke1em8ff/`
(pointer recorded in train_final.sh).

## Re-verification after the tail-head leak (2026-09-01)

The tail rig that produced the +0.161 arm was found to have trained its interval
head on the very frames it then scored (`docs/archive/FINDINGS_2026-09-10.md` §0d). That mattered here more
than for most candidates, because the leak's suspected mechanism — a head with no
spread channel leaning on memorised per-window error — is *this candidate's own
change*, so the bias could have pointed straight at flattering it.

Both arms were rebuilt with `frames="head"`, same three members, head-only swap:

| head | pre-fix | fixed |
|---|---|---|
| with spread | 54.710 | 54.605 |
| no spread | 54.874 | 54.786 |
| **gap** | **+0.164** | **+0.181** |

**The evidence survives and is marginally stronger.** The leak inflated the *spread*
head slightly more (+0.105 against +0.088), so it was working against this candidate,
not for it. The suspected mechanism is also refuted: leak magnitude tracks member
count, not the spread input.

Scored through the packed artefact (same rig, both heads fixed):
`rel 96.423 / tke 81.397 / mvpe 96.905` **identical to the last digit** in both arms,
`sps 54.627 -> 54.807`, i.e. **+0.180**. The candidate is funded entirely by the
interval channel, exactly as designed.

**On the prediction.** Applying the head-config coefficient directly to +0.180 gives
board sps +0.220 [+0.162, +0.277] and final **+0.057 [+0.042, +0.072]** -- roughly
*triple* the recorded +0.02 central and above the top of the recorded band. The
recorded prediction is deliberately more conservative than the arithmetic, and the
reason is this candidate's own stated falsifier: the 1.22 +/- 0.32 coefficient was
measured for head *configuration* changes (width, step, constants) and this removes
an input *feature*, which may not be the same class. **Do not revise the prediction
upward on the strength of a larger local number** -- the discount was applied to the
coefficient's applicability, not to the measurement, and the measurement getting
bigger does not make the coefficient more likely to cover it. The band stays
**+0.02 [0.00, +0.06]**, now resting on a tail arm that has survived a deliberate
falsification attempt. The `hk4` fold arm
(+0.042 sps, 3/4) and the 9% capture diagnostic were never affected — neither runs
under a tail holdout.

## Armed 2026-09-01

`submit_at.sh` scheduled for **2026-09-02T00:00:30Z** (today's quota was already
spent). Log: `runs/submit_nospreadhead.log`. Zip:
`build/nospreadhead.zip`, 161.9 MB extracted, `submission.py` and `model.pth`
sha256-matched against `submissions/nospreadhead/`.

Pre-arming verification, all passed:

* the shipped head evaluates `val_aoa` only -- **no tail holdout was active** when it
  was trained, so the artefact is untouched by the 2026-09-01 tail-head bug, which
  affected the *evidence instrument* and not this pack;
* head recipe is the shipped one exactly bar the change under test: n_models 3,
  width 96, 4000/4000, lr 8e-4, tau 0.05, train-split all, `spread=False`;
* its members are `tke1em8ff_ens/m{1,2,3}` -- the champion's own ensemble;
* `check_pack` MATCH at 1.98e-4 against those members, identical to the champion's
  own parity figure;
* in-sample paired read vs 904950: rel/tke/mvpe identical to the last digit on both
  val_aoa and val_re, sps -0.030 / +0.015. Parity, as designed -- the case rests on
  the two out-of-sample instruments, not on these.

**Read it against the falsifier, not against hope.** A result below 81.597 means the
head-config coefficient does not cover input-feature removal; restore the spread
input and close the axis. A result inside [81.60, 81.66] confirms the coefficient
extends to input features, which is worth more than the points.

## Outcome — 911563, scored 2026-09-02

**81.55306 against the champion's 81.59694: −0.044, and the pre-registered
falsifier tripped** (prediction +0.02, band [0.00, +0.06]; "a read below 81.597
means the interval-channel coefficient does not cover input-feature removal").

| | 904950 (08-28) | 911563 (09-02) | Δ | × weight |
|---|---|---|---|---|
| rel_l2 | 94.82381 | 94.82381 | **+0.00000** | — |
| tke | 79.05878 | 79.05878 | **+0.00000** | — |
| mvpe | 93.91028 | 93.91028 | **+0.00000** | — |
| sps | 43.73919 | 43.74379 | **+0.00460** | +0.00116 |
| time | 90.50468 | 90.04014 | **−0.46454** | **−0.04399** |

**Three readings, in order of how much they should change future decisions.**

**1. The build was exactly what it claimed.** Three accuracy subscores identical to
the last digit. The members really were untouched, and this is the cleanest artefact
verification this project has produced — worth remembering as the signature to expect
from any head-only candidate.

**2. The change did nothing rather than harm.** sps moved **+0.005** where +0.079 was
needed for the predicted +0.02. The axis closes on *neutral*, so the honest summary is
that the spread input was dead weight and removing it is also dead weight: keep
whichever build is otherwise verified, and stop spending slots here.

**3. The calibration miss is the durable result, and it is large.** The tail pair read
**+0.181 sps** and the head-config transfer coefficient of **1.22 ± 0.32** — the
best-measured coefficient in the project, with `sps_bulk` as its precedent — predicted
board sps ≈ **+0.22**. Observed **+0.005**: an over-prediction of roughly **40×**. Two
of the three motivating instruments (hk4 folds +0.042 sps, the tail pair +0.181) agreed
in sign with each other and disagreed with the board by more than an order of
magnitude. **The head-config channel now has no usable transfer coefficient**, which
puts it where `docs/archive/FINDINGS_2026-09-10.md` §0b already put the accuracy total. Do not quote 1.22 again
without saying this happened.

**4. The −0.044 is not attributable to the artefact.** The timed trunk is
shape-identical and the head has *fewer* inputs (120→80), yet time fell 0.465. The
container ran **8.024 ms/sample on 08-28 and 8.919 on 09-02** — an 11% platform
slowdown — and §9a permits time comparisons day-adjacent only, which 08-28 → 09-02 is
not. Read the −0.044 as drift, and read the experiment as the sps line above.

Post-mortem consequence: 911563 is now the correct day-adjacent reference for
`CANDIDATE_initmove.md`, whose archive carries a **byte-identical `model.pth`**.
