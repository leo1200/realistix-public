# Candidate: coop25 — train the third member for the ensemble

Prepared September 9, 2026. The owner explicitly authorized autonomous follow-up
and submission of the most promising candidate within today's UTC slot. The live
API confirmed quota available at 13:16 UTC. **Submitted as 920181 at 13:34 UTC;
finished at 81.74507, -.02272 versus incumbent 918211. Not promoted.**

## Model and reproduction

Keep current submission 918211's full-data members0/1 fixed. Train member2 from
scratch (seed2) using `0.75 * loss(mean(m0,m1,m2), target) + 0.25 * loss(m2,target)`.
The loss is the current CompositeLoss plus SPS bulk; gradients update only m2.
Use the current Muon .02/.95, matrix decay .03, MVPE .3, EMA .998 recipe and the
8000-step cosine, retaining step6000. Train a fresh ordinary spread head around
this exact ensemble, seed0, width96, lr8e-4, tau.05, fixed4000/4000. All 81 usable
real and 100 simulated trajectories are used; no extra or augmented data.

There are still three width96 forecasters and one width96 interval head at
inference. Source is byte-identical to `submissions/wd03mv/submission.py`; its
historical header names the standard training script, while this manifest and
the research reproduction instructions specify the cooperative training change.
No new runtime operation or model parameter is introduced.

Full recipe, frozen-source hashes, commands, checkpoint ancestry, and reproduction:
[research report](../../research/audit_2026-09-09/REPORT.md),
reproduction instructions,
full-data manifest.

## Evidence available before the slot follow-up

Actual three-member means, paired at step6000 against the same two companions
and an independently trained third member of the same seed:

| Instrument | Result |
|---|---|
| Four mimic folds, third seed2, forecast score | +.00299 / +.02649 / +.02810 / +.01255; mean +.01753 |
| Same folds, separately fitted ordinary heads, head seed0 | SPS -.06396 / +.11222 / +.09078 / +.05582; mean +.04871 |
| Forecast plus learned intervals | -.01320 / +.05489 / +.05106 / +.02670; mean +.02986 |
| Third seed3 replication, mim1/mim3 | Forecast +.01682 / +.03020 |
| Head seed1 replication, fixed seed2 members | SPS +.02220 / +.08920; full local +.00859 / +.05066 |
| Different geometry: AoA15, Re17775/24150 | Forecast +.00136, effectively flat; learned heads added in slot follow-up |

Forecast weights are the archive's .4738 relative-L2, .1057 TKE, .0926 MVPE;
full local adds .2529 learned SPS. These are local comparisons, not predicted
leaderboard deltas. The four-fold component mean is rel score -.02053, TKE
+.25915, MVPE -.00145. Gains are substantially TKE-funded, with a small field
error cost. 918211's prior favorable local TKE result did not transfer to the
board. The mimic folds have been reused heavily. Seed replications share fixed
companions and folds; they are not independent new regimes.

Pure cooperative loss has a slightly larger selected four-fold point mean but
loses mim1 and costs more field accuracy. Weaker cooperation and a late-start
curriculum do not improve on the constant blend. The blend is the validated
fallback rather than the maximum screening score.

## Exact archive and deployment checks

- ZIP: `build/coop25.zip`, 150,015,781 bytes.
- ZIP SHA-256: `7493dc8b57aa51d8ca174000f2bed73be97a07581f1354c1e0655cc32e4d4398`.
- Model SHA-256: `7041e1af37eea1041c43e326c6499a0ae5164ef0aae9dd6f1bd08ee5057478fe`.
- Inference SHA-256: `0d47721eae007eacefbf4302eb904279e1dc1c218b810c226ea9b820542ae37c`.
- Only `submission.py` and `model.pth` are included, approximately 162.1 MB extracted.
- Member/head ages, recipe and companion hashes pass
  identity checks.
- Relative prediction/lower/upper differences from fp32 training modules are
  .0301% / .0511% / .0458%; finite ordered outputs, zero pressure and repeated
  calls pass pack parity.
- The exact ZIP passed isolated-interpreter verification on A100 with PyTorch
  2.2.2+cu121, batches 1/2/3/4/6/7/8/16/17/32/64/65 and a repeated batch1.
  All nine graph buckets captured, no repository module was imported, and the
  check took 8.79 seconds. A separate 187-window chunk1 scoring smoke took
  9.8 seconds including setup, below the stricter 180-second track limit.
  Standalone result.
  In-sample scores are rel95.432/TKE83.220/MVPE95.863/SPS51.107; these check
  deployment, not held-out quality. No local timing gain is claimed.

## Slot follow-up and selection

See the pre-registered follow-up.
All six additional fixed-step heads completed successfully. The different
geometry gives **+.025876 SPS**, positive on every partition (+.01688 / +.05861 /
+.01401), and **+.00790 full local** including its +.00136 point gain. This is a
modest additional positive result, not evidence for the larger mimic effect.

On the cooperative ensemble, corrected physical-unit prefactor training improves
SPS **+.052317 / +.017288** on mim1/mim3, mean **+.034802**. Fresh ordinary controls
reproduce the previous scores exactly. The mean falls below the predeclared .04
threshold for confirmation, so this objective is **not** substituted. The prior
normalized-floor tests on independent ensembles and these corrected cooperative
tests change both numerical treatment and ensemble: their difference cannot be
attributed specifically to the epsilon correction.

**Selection recorded 2026-09-09, before upload:** submit the exact `coop25.zip`
above, with the ordinary seed0 head. Across this audit and follow-up, all **88
training runs** completed (67 forecasters, 21 heads), **21.35 GPU job-hours**;
deployment/evaluation probes are additional and excluded from that count.

**Pre-submission expectation against 918211 (81.7677894393): +.01 final, with a
heuristic range [-.06, +.08].** This is an uncertainty allowance, not a calibrated
confidence interval. Illustrative component centers are relative-L2 -.01, TKE
+.05, MVPE approximately zero, SPS +.03, and timing unchanged, approximately
+.0085 under the archive's fitted delta weights. With TKE gains removed, the
four-fold component read is only about +.004 and the different-geometry read
about +.003. Do not turn the +.02986 local mean directly into a board prediction.
There is no timing improvement in the implementation; small timing changes on
the board are run variability. A loss worse than -.06 warrants closing this
recipe for promotion unless the decomposition attributes it to timing; an
in-band result is one transfer observation, not proof of the mechanism.

## Scored result — September 9, 2026

Submission **920181**, archive `coop25.zip`, accepted **13:34:12.181396 UTC**,
finished successfully. Today's quota is spent; no retry or subsequent upload is
scheduled. Best remains **918211, 81.76779, rank 3/50**.

| Subscore | 918211 | 920181 | Change | Estimated final contribution |
|---|---:|---:|---:|---:|
| Relative L2 | 94.846878 | 94.808440 | -.038439 | -.015525 |
| TKE | 78.912806 | 79.049634 | +.136828 | +.013053 |
| MVPE | 94.091739 | 94.019155 | -.072584 | -.007476 |
| SPS | 43.722892 | 43.703531 | -.019361 | -.005040 |
| Time | 92.174372 | 92.106384 | -.067988 | -.006792 |
| **Final** | **81.767789** | **81.745070** | **-.022719** | |

Contributions use historical fitted delta weights, not published exact weights;
their sum leaves a **-.000939** residual. The estimated non-time contribution is
**-.014988**. Timing contributes to the loss but does not explain it away.

The result is inside the pre-recorded [-.06,+.08] heuristic band, below its +.01
central expectation. TKE improved in the expected direction, so the ensemble
energy mechanism did transfer. The relative-L2 cost increased, MVPE moved more
negatively than the local near-null, and the modest local interval benefit did
not transfer. This is an unsuccessful promotion candidate and a useful
constraint on the next design: retain field/probe and interval quality while
addressing ensemble energy. A stronger cooperative dose is not justified by
this result. The corrected head objective was not tested on the board.

API receipt,
component decomposition,
pre-upload decision.
