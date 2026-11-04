# Realistix audit and model research — 2026-09-09

**Completed including the authorized slot follow-up: 88 full training runs (67 forecasters, 21 heads), 21.35 GPU job-hours.** The strongest local result was cooperative training of the third ensemble member: **+.01753 accuracy-only** and **+.02986 including learned intervals**, averaged over four mimic folds. Its authorized September 9 submission, **920181**, scored **81.74507**, **-.02272** below 918211. **The incumbent remains best at 81.76779, rank 3/50.** The TKE mechanism transferred, but other accuracy and interval costs outweighed it; this candidate is not promoted.

## Submission result

`coop25.zip` was accepted at **2026-09-09 13:34:12 UTC** and finished successfully. Observed changes against 918211: relative-L2 score **-.03844**, TKE **+.13683**, MVPE **-.07258**, SPS **-.01936**, timing **-.06799**. Historical fitted delta weights attribute approximately **-.01499** to non-time components and **-.00679** to timing, with a -.00094 fit residual. Thus the loss cannot be explained by timing alone. The pre-recorded +.01 expectation with heuristic range [-.06,+.08] contained the result; that is not evidence that the point prediction was accurate.

Cooperation successfully addressed part of the ensemble's fluctuation-energy defect, while the local MVPE near-null and interval gain failed to transfer. Further work on this mechanism needs to preserve field/probe accuracy and interval quality; increasing cooperative pressure alone has no demonstrated case for promotion. The corrected prefactor head remains a small two-fold lead, not a validated replacement. Full score record and [candidate manifest](../../submissions/_manifest/CANDIDATE_coop25.md). Today's slot has been used; no additional upload is scheduled.

## Authorized submission follow-up

After the original 82-run round, the owner requested autonomous continuation and submission in the September 9 UTC slot. Six additional fixed4000 heads completed. Ordinary heads around the independent and cooperative ensembles on the different holdout geometry give **+.02588 SPS**, positive in every angle/Re partition, bringing its full local difference to **+.00790**. On cooperative mim1/mim3 ensembles, the corrected physical-unit prefactor objective gives **+.05232 / +.01729 SPS**, mean +.03480, below the predeclared .04 threshold for confirmation. Fresh ordinary controls reproduce the earlier SPS scores exactly. The ordinary head therefore remains in the candidate. These tests change both ensemble type and the numerical treatment relative to the original prefactor tests, so the difference is not an isolated epsilon-correction effect.

The exact `coop25.zip` passed an isolated-interpreter check on A100/PyTorch2.2.2, including irregular batches through 65, ordered finite intervals, repeated-call identity, and all nine CUDA graph buckets. No repository import is required. Prediction-stage smoke testing took 9.8 seconds including setup. These are deployment checks on released, in-sample windows, not hidden-set timing or quality claims. Follow-up plan, results, archive checks, and [candidate decision and pre-submission expectation](../../submissions/_manifest/CANDIDATE_coop25.md).

## Repository audit

The member model is a 22.65M-parameter residual U-Net: 20 input frames as 40 channels, two spatial downsamplings, a 32×64 output grid, GroupNorm/SiLU blocks, a last-frame residual, and modality FiLM folded away for inference. Three independent width-96 members are averaged and an asymmetric interval head predicts physical widths. Training uses released real+sim trajectories, input-only RMS normalization, Muon for matrices (including 2-D FiLM tables), AdamW for 1-D parameters, EMA, and several metric-specific losses.

The repository has a large empirical archive: this audit extracted **3,117 archived configuration records**. The inventory lists all exposed controls of both current trainers and maps the important implicit numerical/sampling/deployment constants. Counts are configurations, not independent trials; historical missing fields remain unknown.

The final inventory contains **112 forecaster controls and 22 interval-head controls**, including paths/provenance controls that should not be tuned against validation. No archived configuration record failed to load.

| Finding | Evidence and impact | Action |
|---|---|---|
| Default research recipe was obsolete | `kfold.BASE` still selected AdamW, TKE .5, no SPS and EMA .999 while describing itself as shipped | Added explicit `--recipe current`; archived commands retain their meaning and the selected recipe is printed |
| Reproduction checks did not target current best | Defaults in check_recipe/verify_best still point to 904950 (Aug 28), while the recorded latest is 918211 (Sep 8) | Check output now says which checkpoints it validates; explicit build overrides can be checked; current member/head verified separately below |
| Current head falsely failed recipe checking | `head_frames` is recorded provenance derived from the tail setting, not a parser flag | Derive/check its expected value instead of treating it as an unknown recipe option |
| Head checkpoints did not record their saved step | The forecaster records step; interval-head checkpoints only stored the requested schedule | Future heads now record step explicitly. For this frozen-source build, verify fixed_step=4000, completed history and the trainer's save rule |
| Simulation dose was confounded with coverage | `sim_frac` thins trajectories; GPU path ignored `sim_stride`; batch share was implicit | Added `sim_batch_frac` preserving all trajectories, trajectory-balanced sampling, and full-objective mixture importance weights. Unsupported GPU sim_stride now fails explicitly |
| Historical simulation loss scaling is partial | `w_sim_loss` scales CompositeLoss, while SPS and other outer terms remain pooled | Kept the compatibility behavior and documented it; new `sim_loss_frac` controls composite+SPS with guards against unsupported combinations |
| Gradient retention is a nonlinear memory filter | Existing Muon path clears matrices but retains AdamW grads, then clips the whole model each step | Preserved default; exposed clipping scale/grouping and Adam betas/epsilon. A 400-step probe measured median clip factor .435 and retained/fresh rest-gradient norm ratio .313; no step was unclipped |
| Resume checker rejected legitimate overrides | It compared the earlier BASE flag before the candidate's later occurrence | Last occurrence now wins; missing best.pt is not completion; recorded fold axes are checked |
| Fold helper failed on whole-angle/Re folds | It indexed HELDOUT_CELLS even for folds defined only by angle/Re, then cleared those axes | Apply all three axes and verify in fresh subprocesses |
| SPS training objective omits scorer factors | Bulk surrogate and default head reward omit per-window relative-L2/TKE/MVPE prefactors | Added opt-in full target-assisted forecaster reward and weighted head reward; verified against the official scorer, then tested actual learned heads |
| Checkpoint shapes cannot verify GN group count | 4/8/16 groups share identical parameter shapes but compute different functions | Reconstruct from saved args; packer refuses non-8-group members until inference supports them explicitly |
| Cooperative checkpoints require specific companions | A member optimized as part of a mean is not an interchangeable standalone predictor | Record companion paths, validate holdouts/ancestry, label ensemble metrics, and refuse packing with different companions |
| Missing and empty holdout environment values are different | Split defaults apply when an environment key is absent, while legacy argument metadata records absent keys as empty strings | Every held-out campaign sets all five split controls explicitly. Full-data reproduction unsets them as the production builder does and trains on all trajectories; its default validation partitions are only in-sample diagnostics. Legacy metadata alone cannot recover this distinction |

The audit did not silently change historical gradient clearing or replace production weights. The archived artifact verification passes for its seven pinned submissions. Local model-validation scores do not establish a hidden-board gain. In particular, 918211's +.005 recorded final-score change was funded by timing drift while the accuracy contribution was approximately -.008.

## Current recipe reproduction

The unchanged production builder needs its documented overrides for the Sep 8 model:

```bash
WD_MUON=0.03 W_MVPE=0.3 CHAMPION=wd03mv \
  NAME=my_fresh_build SUB=my_fresh_build bash tools/train_final.sh

python tools/check_recipe.py \
  --member runs/wd03mv_m0/best.pt \
  --head runs/wd03mv_head/best.pt \
  --member-args='--w-mvpe .3 --wd-muon .03'
```

`current_recipe_check.txt` records the check against actual saved checkpoint metadata. This is recipe agreement, not a fresh full retraining or a check of inference runtime. The production head uses spread; the temporary no-spread head from an earlier submission is not the current Sep 8 head.

## Experiment protocol

See PLAN.md for hypotheses, primary endpoints and continuation rules registered before the corresponding results. Every run uses a frozen source snapshot, full command and fold environment, recorded hardware, and a process exit status. The primary endpoint is step 6000 of an 8000-step cosine. Batch64's exposure-matched comparison uses step 3000 of 4000. No best.pt selection is used for the comparisons.

The first screen uses mim1 and mim3, seed 0, fresh current-recipe controls on A100. Scores below are local **accuracy-only weighted subscore deltas**, not predicted board points; weights .4738/.1057/.0926 are the archive convention. The constant-band SPS proxy is excluded. Two screening folds and one seed do not establish generalization or resolve tiny effects. The four mimic folds have already supported hundreds of prior decisions, so even a repeated positive requires an additional instrument.

Results are separated by the object being measured: RESULTS.md compares individual members; ENSEMBLES.md compares actual three-member means and learned intervals. Comparing cooperative histories directly with member histories would be invalid, and the ordinary fold scorers now reject that mistake.

## First screen results

All 24 runs completed successfully. No first-screen arm improves accuracy convincingly. Mean paired accuracy-only deltas over mim1/mim3 are: mix25 **-.0383**, mix75 **-.0396**, uniform trajectories -.0041, clip .25 -.0092, clip 4 -.0140, separate clipping -.0055, GN4 -.0023, GN16 -.0026, oracle SPS at 1.3 -.0079, Adam beta2 .95 -.0083, and TKE 1.3 -.0099. Tiny negatives are inconclusive, not proof that every setting on an axis is bad.

The full curves are important: extreme mixture and clipping changes can win early, regress sharply mid-run, and then recover. The precommitted fixed checkpoint prevents those transients from becoming selected “improvements.” RESULTS.md and summary.json contain completed-run component scores and paired differences; PARAMETERS.md / parameters.json provide the parameter inventory.

## Mechanism follow-ups

Changing the simulation sampling share to 25% while preserving the original objective mixture loses **-.00950**, much less than the uncorrected **-.03829**. Conversely, keeping the original 19/32 sampled simulation share but reducing its full objective mixture to 25% yields **+.00217**. That is too small to promote, but it separates changing the expected objective from changing sampling noise/exposure. The SPS importance weights preserve the element-weighted scoring convention, including different valid masks; they are not a claim that each domain contributes an exactly fixed fraction of every minibatch's scalar SPS loss.

Muon LR .015 gives -.00444 and .03 gives -.03942. Doubling batch size with matched total examples, doubled learning rates and squared Muon momentum/EMA retention gives -.03446. This does not establish that every batch64 schedule is bad; it rejects this specific exposure-matched scaling rule. The larger prefactor-weighted SPS surrogate gives +.00211 with opposite fold signs, and adding TKE compensation gives -.01262. Extending TKE alone to 1.5 gives -.01183. These follow-ups gave initially weak ideas a second mechanism-based trial; none qualifies as a reliable gain.

An exploratory stopping-age check also fails to rescue the first two rounds: allowing each arm its best *common* observed checkpoint still leaves every arm below baseline6000 except the same tiny simloss25/oracle26 results. The late GN and high-simulation curves improve relative to a baseline that is itself deteriorating; they do not exceed the baseline's primary endpoint. This check is exploratory and did not replace the registered comparisons.

## Cooperative training

Train two members normally, freeze their step6000 checkpoints, and train the third from scratch using

```
p = (p0 + p1 + p2) / 3
loss = 0.75 * current_loss(p, target) + 0.25 * current_loss(p2, target)
```

The current loss includes CompositeLoss and SPS bulk. Only p2 receives gradients. The target remains the released training target; this is not distillation. The architecture and number of inference passes stay unchanged. Cooperative/negative-correlation ensembles have prior literature; the contribution here is a tested adaptation to this repository's nonlinear scoring losses and deployment constraints, not a claim of inventing cooperative training. See the primary research references in PLAN.md.

| Constant blend, third-member seed2 | mim1 | mim2 | mim3 | mim4 | Mean |
|---|---:|---:|---:|---:|---:|
| Accuracy-only delta | +.00299 | +.02649 | +.02810 | +.01255 | **+.01753** |

Mean component changes are relative-L2 score **-.02053**, TKE **+.25915**, MVPE **-.00145**. Thus the gain is a tradeoff: preserving fluctuation energy after averaging at a small field-error cost. The third member alone can be worse: on mim1 it loses .0724 relative-L2 score and .1218 TKE, while the actual mean improves. This is precisely why member-only screening would miss the method.

There is a concrete reason to optimize this object. For temporally centered member fields `z_m`, squared energy obeys `E(mean(z_m)) = mean(E(z_m)) - mean(E(z_m - mean(z_m)))`. Averaging removes disagreement energy. Matching each member's TKE separately therefore does not match the mean's TKE. The observed tradeoff is consistent with targeting that ensemble defect directly; the TKE subscore also measures spatial error, so restoring a single global energy scalar would not be equivalent.

Initialization replication with third-member seed3 gives **+.01682 on mim1** and **+.03020 on mim3**. The blend is positive in all six paired member-seed/fold comparisons so far. These are not six independent regimes: the two screening folds recur, and both fixed companions stay unchanged between seeds.

Pure ensemble loss gives +.01968 over the four folds, but loses on mim1 and costs -.06834 relative-L2 score. Weaker cooperation (75% individual loss) gives only +.00329 on the screening folds. Multiplying pure cooperative gradients by 3 gives +.03117 there, dominated by mim3; it is not independently confirmed. The isolated full-data build uses the constant 25%-individual blend, not this larger selected screen result.

Per-cell diagnostics show that the blend reduces bulk reward by about .00015–.00019 on both screen folds. Its target-assisted SPS ceiling is flat on mim1 and +.1098 on mim3. Learned intervals must therefore be measured explicitly; a TKE gain is not sufficient. The curriculum follow-up first learns the ordinary field and then transitions to cooperation to test whether late specialization reduces that cost.

That curriculum completed at **+.00537 / +.01923** accuracy on mim1/mim3, mean +.01230 versus +.01555 for the constant blend. Bulk reward remained lower than the independent control on both folds (-.000178/-.000189). Against the constant blend, its oracle SPS changes +.0114/-.0437. The proposed repair did not improve its intended bulk mechanism consistently, so it is not substituted into the candidate.

## Learned interval results

| Constant blend with ordinary head, head seed0 | mim1 | mim2 | mim3 | mim4 | Mean |
|---|---:|---:|---:|---:|---:|
| Official SPS delta | -.06396 | +.11222 | +.09078 | +.05582 | **+.04871** |
| Forecast + learned-interval local delta | -.01320 | +.05489 | +.05106 | +.02670 | **+.02986** |

The full local metric uses the archive accuracy weights plus .2529 × learned SPS, with no runtime contribution. All heads are trained separately around their exact ensemble, on that ensemble's training split, with the same width96/spread/lr.0008/tau.05 settings and fixed4000 endpoint. No width-inflation factor or head seed is selected per fold. The negative mim1 result is part of the result, not discarded. See the comparison figure and ENSEMBLES.md.

Head-seed1 replication gives **+.02220 SPS / +.00859 full local on mim1** and **+.08920 SPS / +.05066 full local on mim3**. The mim3 benefit repeats closely; mim1 changes sign and is not a resolved gain. The candidate keeps the preselected head seed0. These replications hold member weights fixed and test head initialization only.

## Different holdout geometry

The preselected constant blend was also tested after withholding all AoA15 trajectories and Reynolds numbers17775/24150, using three newly trained ordinary controls. The same step6000/8000 schedule gives **+.00136 accuracy-only**, effectively flat. Component changes are relative L2 **-.00963**, TKE **+.05556**, MVPE **+.00048**. The mean angle-side delta is approximately +.00455 and the Re-side delta -.00184. TKE improves in all three evaluation partitions, but the benefit is much smaller than on the mimic folds and is partly consumed by field-error cost.

This is evidence of limited transfer, not a broad validation success. The geometry was new in this round; its underlying released trajectories were used historically. No learned-head gain is claimed on this geometry. Detailed partition differences are in ensemble_results.json.

Adding the official SPS prefactors to the head objective alone gives **-.02718 / -.00316 SPS** on mim1/mim3. Numerical alignment with a scoring formula is not enough to improve its finite-sample optimized model; the unweighted head remains in the candidate.

**Numerical edge case found in final review:** relative-error ratios are scale invariant only while epsilon floors are inactive. The frozen oracle/prefactor experiment sources apply denominator floors to normalized fields, which can disagree with official scoring for nearly steady, low-amplitude targets. The final implementation now converts to physical units before computing prefactors, and an additional near-steady-flow test matches the official scorer. The original round's experiments test the earlier surrogate; they do **not** settle the corrected epsilon-edge variant. The subsequent authorized slot follow-up trains the corrected head objective on cooperative ensembles, as reported above; the corrected forecaster oracle objective remains untrained. The selected cooperative candidate uses the original bulk loss and ordinary head and is unaffected. sps_epsilon_check.json records the synthetic diagnostic.

## Verification and artifacts

Eleven focused checks pass: numerical agreement of both new SPS objectives with official scoring, physical-unit epsilon behavior, finite gradients, importance-weight replication, exact mixture counts and coverage retention, trajectory/angle sampling, normalization reconstruction, fold environments, resume overrides, cooperative mean/gradient behavior, and pack rejection of changed companions. All 91 tool help pages render; `git diff --check` passes. Defaults of the historical training paths remain unchanged.

The packaged mim3 fold prototype is **162.1 MB**. Against the training modules in float32, mean relative differences are **.0302%** for prediction, **.0525%** for lower bounds and **.0471%** for upper bounds. Outputs have the required shapes, finite ordered intervals and zero pressure; repeated calls reproduce exactly. Nine CUDA graph batch sizes are available. This is numerical parity evidence on 16 windows, not a hidden-set SPS or runtime measurement. Exact paths and hash are in prototype_pack_check.json.

The **full-data candidate** is also built and verified:

- model.pth, **162.1 MB**, SHA-256 `7041e1af37eea1041c43e326c6499a0ae5164ef0aae9dd6f1bd08ee5057478fe`.
- Current full-data members0/1, a newly trained cooperative member2 at step6000, and a new ordinary spread head at step4000. Training manifest and [reproduction instructions](../../REPRODUCE.md).
- Forecast/lower/upper mean relative differences from the training modules: **.0301% / .0511% / .0458%**. Shapes, finiteness, interval ordering, zero pressure and repeated-call identity pass. Numerical check.
- Member/head checkpoint ages, recipe, companion hashes, and equality of inference source with `submissions/wd03mv/submission.py` pass. Identity check. The model sizes and number of forward passes are unchanged; runtime has not been measured on the leaderboard.

Bulky checkpoints, frozen sources and logs live under `research/audit_2026-09-09/`; manifests in this directory pin commands, splits, seeds and source hashes. The older `ensemble_reference_diagnostic_mim1.json` is an initial loading smoke test with a near-duplicate seed and is excluded from every comparison. Existing checkpoints and submissions have not been overwritten. The original round ended without an upload; the subsequent explicitly authorized slot follow-up submitted `coop25.zip` as **920181** on September 9 at 13:34 UTC.

## Recommendation

Keep the constant cooperative blend as a **candidate for a controlled external evaluation**, while retaining the current production artifact. It has a modest repeated local benefit and an unchanged inference architecture, but the gain is primarily TKE-driven, the broader geometry is flat, and mim1's learned-head result is unstable across seeds. Prior TKE gains in this repository have failed to transfer to the board. A hidden-board improvement has not been established.

The practical advances from this round are the trained candidate, ensemble-aware training and measurement, a complete control inventory, and repairs to recipe/fold/resume/pack verification. The sampling, clipping, normalization, learning-rate, batch-scaling and loss-surrogate branches received complete schedules and mechanism-based follow-ups; their losing and inconclusive results remain recorded. The epsilon-corrected optional forecaster SPS objective is mathematically tested but awaits training; its head counterpart was trained in the authorized slot follow-up and subsequently confirmed in the separate repair round.

All 82 scheduled training jobs completed with exit code 0; none remains running. status.json records the accounting. Its 20.70 GPU job-hours include loading and validation, exclude queue time, and exclude the separate gradient probe, evaluation-only jobs and packing checks.
