**Recommend the existing, frozen k4a2 for September 17, then prioritise the private retraining contract.** The actual A2 artifact withstands independent verification. Its automatic build and provenance gates do not yet establish that a future retrain reproduces the method.

The requested 12k combination improves the fold results. Keeping 4k for September 17 is an operational-readiness decision, not evidence that 4k is statistically better.

The full report, scripts and receipts are in codex_audit/REPORT.md. No commits, shipped-code edits, uploads or scheduling changes were made.

**1. The artifact passes; several provenance mutations pass too.**

The strongest evidence is an independent all-data training replay: **all 80 FP32 student tensors reproduce byte for byte**. I also independently retrained one representative teacher, seed 21: all 80 tensors reproduce exactly. I did not retrain all six teachers or all four forecasters.

The six teacher files match the student’s recorded hashes, seeds **0, 11, 21, 31, 41, 51**, step 4000, `train_split=all`, and four-member hashes. Those member hashes match the actual files. The replay used 81 ordered release trajectories, all frames, and genuinely **unset** holdout variables. The student differs from teacher 0 in all 80 tensors.

The stem is **[96,161,3,3]**: 40 window channels, 40 forecast, one log-scale and 80 spread. Thus `head_width=96` and `head_spread=True` are correct. All member tensors reproduce the folding/casting operation exactly; all head tensors equal the student’s fp16 cast.

The archive:

- Contains only `submission.py` and `model.pth`, with valid CRCs and contents matching the directory.
- Extracts to **207,362,725 bytes**, leaving **61,072,731 bytes** below the cap.
- Has SHA256 `ec556824be90b6eef8be157621f18c6aecf7e2191227b21ee37e9759add25127`.
- Uses inference source byte-identical to k4g.

Standalone torch 2.2.2 tests captured all nine graph buckets. N=1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,63,64,65,129 produced finite, ordered outputs. All graph/eager comparisons passed the existing .002 tolerance; maximum difference was .00122. Injected capture/runtime failures fell back correctly. Intervening calls and changing another batch row caused no output change. No repository modules were imported.

For **all 187 release fixture windows evaluated individually**, the complete point-prediction arrays are identical across k4g, A2 and the archived pair. Independently retrieved 924146/926634 scorecards have identical rel/TKE/MVPE to ten digits.

However, **universal fresh-process bit identity is false**: some larger batches differ even between unchanged k4g runs. Observed differences reach 2.44e-4 in outputs and 1.34e-5 in release subscores. This does not implicate A2, but it limits the claim. Release data cannot reproduce hidden board predictions, and the API does not expose an archive checksum authenticating the local pair archive as the scored upload.

Independent three-cycle ABBA timing gave:

| Measurement | Result |
|---|---:|
| k4g median | 3.975014 ms |
| A2 median | 3.975237 ms |
| Mean paired difference | **+0.000363 ms** |
| Individual cycle differences | −.006332 / −.001522 / +.008943 ms |

There is no meaningful measured A2 overhead. Retire the absolute 4.25-ms gate.

The mutations expose the important boundary:

| Mutation | Result |
|---|---|
| Zeroed or seed-0 head substituted only in the pack | Tensor gate and `check_pack` reject |
| Member bit flipped, head key removed, incorrect spread metadata | Tensor/schema checks reject |
| Source student replaced by seed-0 weights with copied student metadata | **Build and `check_pack` pass** |
| Genuine step-2000 snapshot from the 4000-step replay | **Both pass** |
| Wrong-member student with honest member paths | Builder rejects |
| Wrong-member weights with copied good provenance | **Both pass** |
| Bad/missing teacher digests or duplicated teachers | Builder passes; bad-digest pack also passes `check_pack` |

Packing fidelity is not training provenance. The actual A2 is credible because of the replay; the gates alone would not catch several requested failure cases.

Other concrete defects:

- `build_a2.sh` lacks `set -e`; some failed checks can still be followed by `BUILD_A2_DONE`.
- The old standalone script prints several statistics without asserting success.
- `check_pack` guards its reference intervals against the plain mean, while shipped inference guards against the gained mean. Its discrepancy is not exclusively fp16 error.
- Historical fold students lack teacher-digest and sampling-environment stamps. Current hashes, member holdouts and partitions check out, but missing historical receipts cannot be reconstructed.
- Concurrent Arm-A/B scorers share `measure_<fold>.json` before renaming. Archived variant lists are correct; this is a race hazard, not demonstrated corruption.

Independent official scoring reproduced A2’s gains. Actual folded/fp16 packs also preserve the advantage:

| A2 minus pair | mim1 | far_drop | angle05 | Mean |
|---|---:|---:|---:|---:|
| Registered source evaluation | +.01580 | +.04534 | +.00628 | +.02247 |
| Independent packed evaluation | +.01804 | +.04350 | +.00472 | +.02209 |

**2. Initialization matters most at 4k; 12k improves the mean.**

I registered the additional controls before training. These are diagnostic comparisons on reused folds, not independently validated replacement candidates.

| Student: SPS gain over h0 | mim1 | far_drop | angle05 | Mean |
|---|---:|---:|---:|---:|
| Frozen A2, reread with controls | .09013 | .17069 | .10700 | **.12261** |
| Teacher-0 init, symmetric loss, 4k | .09216 | .15400 | .10171 | **.11595** |
| A2 with seed 17 | .09057 | .16912 | .10999 | **.12323** |
| Teacher-0 init, asymmetric ×3, 12k | .10584 | .18810 | .12457 | **.13950** |

The requested **A1+A2 combination works**: approximately +.01690 fold over A2, worth **+.00345 final** at the central transfer assumption. It improves 32/34 cell readings versus A2. But the worst angle cell becomes **−.15495 versus the pair**, another −.01295 relative to A2.

The combination captures about 93.6% of avg6’s mean gain. It deserves retention for a pre-registered fresh-geometry/retrain comparison. It has no all-data packed artifact or complete retrain certificate from this audit, so I would not automatically spend another public slot on it.

Parameter conclusions:

- **Teacher initialization:** load-bearing for the 4k budget. Fresh symmetric 4k gave about .0589; teacher-initialized symmetric 4k gives .1160. Teacher **0 specifically** remains unablated.
- **4000 steps:** adequate, not optimal. The combined 12k result falsifies a claim that this optimization axis is finished.
- **Under-width weight 3:** adds about .0066 fold, or .00135 final, beyond teacher initialization. It helps far_drop and angle05, slightly hurts mim1, and improves the worst angle loss. Useful, but 3 is not an identified optimum.
- **Seed 7:** seed 17 closely reproduces the mean. This weakens the lucky-seed concern; one replication does not establish full retrain variance.
- **Width 96:** retain. Narrower ordinary SPS heads performed poorly, but that does not prove narrower distilled students impossible. Current `--width` also changes teacher construction, so it is not an isolated student-width control.
- **Separate log-space losses:** preserve interval asymmetry and relative-scale fitting. They remain surrogate objectives, not an SPS optimum. Averaging linear distances before taking logs correctly targets the output ensemble. Regression also includes pixels excluded from the teachers’ masked SPS objective; that mismatch is untested.

K, steps, initialization, asymmetry and seed add no inference work when the final architecture stays fixed. They still affect training cost, generalization and reproducibility.

For **K**, independently scored fixed-prefix output averages at K=2/3/4/6 gained **.10011/.12486/.13522/.14905**. Under independent equal-variance errors, remaining variance is **1/K**; **1−1/K** is the fraction removed. Moving 6→12 removes another 1/12, only 10% more idealized gain than K=6. Correlation leaves a common error component, and SPS is nonlinear.

K12 was not trained here. Its idealized extra headroom does not justify six additional fits and another late search before the retrain contract works.

The .0265 gap to avg6 is a useful budget for **faithful compression**, worth approximately .0054 final. It is **not a strict upper bound on student SPS**: the 12k far_drop student actually exceeds avg6. After 12k, the remaining mean gap is about .00955 fold, or .00195 final.

I also tested naive teacher **weight averaging**. Its SPS changes versus h0 were **−4.959 / −5.495 / −3.052**, mean **−4.502**. Independently initialized features are not aligned; averaging parameters does not average functions. Further training might recover, but an intact teacher is a much better justified starting point.

Distilling **gained bounds** is not an established free improvement. The gain already enters the reported scoring; its interaction with A2-minus-pair is only about **2e-5 fold SPS**. The containment guard affects approximately .055–.125% of scored elements.

With plain prediction P, gained prediction G, and Δ=G−P, delivered bounds are `min(P−b,G)` and `max(P+a,G)`. Expressing them around G gives distances `max(b+Δ,0)` and `max(a−Δ,0)`, introducing zeros and changing the log target. Keeping P as center mostly teaches the existing guard. Refitting teachers against G is a new target policy. Do not bundle any of these changes into the frozen A2.

Detailed results are in sensitivity_summary.json.

**3. The gain is reproduced; its private transfer is substantially less certain.**

The transfer factor re-derives as:

`0.0806826064 / 0.1000692723 = 0.8062675`

That is one board intervention, not a transport law. Dividing the same board gain by individual fold gains produces .644–1.086, but those are not independent board observations. There is no defensible empirical confidence interval for transfer from this evidence.

Two interpretations of the requested sensitivity analysis differ materially:

| Transfer | Discount only A2’s increment above the observed pair | Discount A2’s whole gain above observed h0 |
|---|---:|---:|
| .5 | SPS **43.95265** | SPS **43.92200** |
| .80627 | **43.95953** | **43.95953** |
| 1.0 | **43.96389** | **43.98327** |

The first preserves the observed pair gain by assumption. The second crosses pair SPS at transfer **.6584**. At .5, the second loses about **.00491 final before latency**; removing one head could still compensate. Thus “A2 remains positive at every transfer” is not an unconditional conclusion.

The exchange-rate audit supports the scale but not the claimed precision:

- Independent regressions on archived 14/34 scorecards give SPS coefficients **.253000/.251478**, with leave-one-out RMS **.01152/.00962**.
- The exact k4g repeat gives time-score weight **.0964197**.
- With `S(t)=100/(1+sqrt(t/728.96))`, the derivative at 5.7018 ms is **−.654639 score/ms**.
- One **local** millisecond therefore costs **.012624/.018936/.022723 final** at assumed board/local slopes **.2/.3/.36**.

The .0189 figure is conditional on slope .3, not an independently established exchange rate. Same-chain board clocks of 5.6385/5.9662 ms bracket the pair’s 5.7018 ms; they do not identify that slope. At slope .3, break-even is approximately .0749 board SPS, or .0929 fold SPS at .806 transfer, per additional local millisecond.

The fold suite covers a missing-Re/block pattern, a removed high-Re frontier and an interior whole-angle exclusion. It does **not** cover fractional new angles, beyond-range angles, a new joint Re/AoA corner or complete fresh ensemble retraining.

There are **34 readings but only 28 distinct cells**, shared training data and repeated selection. Matching simulation cells are also removed in mimic/far_drop, unlike the hypothesized missing-real grid with simulation coverage. That mismatch is not necessarily conservative for this head comparison. The hidden 19-cell geometry remains a hypothesis.

The **7/34 negative source cells are a transport warning**, not a footnote. Packed inference produces **10/34 negatives**, with several additional signs near zero. The angle mean is only +.00628 source / +.00472 packed.

At 13950_5:

- avg6 itself loses approximately **.0463** to the pair.
- A2 loses another **.0959** to avg6.

Better distillation cannot remove the teacher-target component. An illustrative mixture giving an additional 13.7% weight to worst-cell behavior erases the overall source increment. That is sensitivity, not an estimated probability. Familiar-cell bootstrapping cannot establish coverage for unseen geometries.

The A2 result itself survives direct replay, official rescoring and packed evaluation. What does not survive is a strong claim of universal dominance or confidently predicted private gain.

I also correct my earlier report: I repeated the absolute latency gate, overstated unconditional bit identity and used the transfer coefficient too mechanically. The round-1 “approximation floor” was falsified. “Independent noise” and broad “optimal policy” claims exceeded the measurements. Rejecting A1 for its −.00088 angle result was an operational rule, not statistical proof of A2’s superiority. FINDINGS §2b still contains the stale clock claim despite the later correction.

**4. Register the existing A2, with a conditional clock forecast and a clearly defined kill rule.**

My September 17 prediction for the checksum-pinned artifact is:

| Quantity | Registered prediction |
|---|---:|
| SPS | **43.9595** |
| Final at incumbent’s exact clock | **81.8114** |
| Final assuming historical 5.456-ms clock | **approximately 81.827** |
| Subjective SPS planning range | **43.9095–44.0095** |

The range is judgmental, not a calibrated confidence interval.

Expected rel/TKE/MVPE are **94.8293790855 / 79.1320014641 / 94.1013126942**. A departure exceeding .005 warrants quarantine and investigation; numerical last-bit differences alone do not.

Keep **SPS ≤ 43.9414123258** as the conservative **promotion kill rule** for replacing the pair without relying on clock savings. It is appropriate for that claim. It is not a universal economic break-even or proof that distillation failed.

Below h0’s **43.8607297194**, kill this student configuration. Between h0 and pair, the student retains interval benefit and may remain an economic latency trade. Do not treat a lucky-clock final as evidence of better intervals, or a timing-only failure as evidence that positive SPS vanished. No automatic exact repeat.

The remaining plan:

- **09-17:** submit the existing verified A2 without parameter changes. Retain 926634 and k4g.
- **09-18:** implement the canonical retrain contract and mutation gates. Include 12k only in a pre-registered fresh-geometry/retrain comparison that does not delay this work. Reserve the slot for a concrete repair or complete clean-retrain confirmation.
- **09-19:** complete an empty-output-root retrain with old checkpoints inaccessible, followed by standalone packing and organizer instructions. Use the last slot only for a justified, verified artifact.
- **09-20:** freeze method, executable recipe, environment/data receipts, lineage, archive and chosen submission ID. Preserve the remaining development period for reproducibility and format fixes.

The immediate private-phase blocker is concrete. The actual A2 fails `check_recipe` in four places. Default `train_final.sh` trains one ordinary SPS head and copies dose-zero k4; `verify_best` still pins 923184. Changing `CHAMPION` alone does not implement distillation.

The required contract is:

**Four fresh forecasters → six fresh SPS teachers on those forecasters → one distilled student → discard teachers from inference.**

It must also preserve the forecasters’ **8000-step schedule with the selected 6000-step checkpoint**, rather than silently shortening the schedule.

I wrote the implementation specification and acceptance checks in RETRAIN_CONTRACT.md. Production implementation and a complete clean retrain remain outstanding because this audit was read-mostly. That gap deserves priority over another expected public gain of roughly .00345.