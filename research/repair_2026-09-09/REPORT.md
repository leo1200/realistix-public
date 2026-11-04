# Post-submission improvement round — September 9, 2026

Updated 2026-09-09T17:23:58+00:00. **98/98 scheduled training runs complete**, 18.21 completed GPU job-hours including diagnostics/retries. Exact commands and completion receipts are retained in the manifests.

The incumbent remains submission **918211, 81.767789**. September9 candidate **920181 scored81.745070**, a loss of.022719 despite better TKE. This round tests repairs to its field/probe tradeoff and interval calibration. No additional submission is scheduled. These local results are not a demonstrated board improvement.

See the predeclared plan, control definitions, machine-readable results, and screen gates. The evaluation folds have been reused; they are not an untouched test set.

## Decision

**Select meanfix as the next research candidate. It is built, verified and unsubmitted.** The incumbent remains918211; no upload is scheduled because September9’s slot was already used.

Its actual FP16 four-fold complete local gain is **+.031550**, with all four folds positive and leave-one-fold-out means+.027923 to+.036171. Relative-L2 score falls-.015724; TKE rises+.204288, MVPE changes only+.000024, and learned SPS rises+.068819. The complete contribution excluding TKE is+.009956. These reused local folds do not demonstrate a leaderboard improvement.

Balanced loss is a credible alternative with a slightly higher four-fold mean+.032804 and better relative-L2, but gives+.016138 on range edge and-.020169 on shifted angle/Re. Meanfix gives+.033073 and+.000288 respectively; the latter is effectively flat. Prefer its more consistent holdout results and FP32 mean/probe constraint after the prior submission’s MVPE regression. This is a judgment under uncertain transfer, not evidence that the small four-fold mean difference is statistically significant.

The constrained forecast improvement replicated with a second third-member seed. Its head-seed replication gains+.026118/+.040871 SPS. The final archive keeps exactly three ordinary inference members; its prediction/bound parity errors are below.05%, and standalone A100 checks pass. The initial absolute FP16 mean bound failed and was replaced only after the recorded rounding diagnosis; no model or head weights changed.

## The two surviving mechanisms

**Balanced loss** keeps cooperative training from scratch but raises relative-L2 weight8→12, MVPE .3→.6, and bulk SPS1→1.5. It retains 75% ensemble loss /25% individual loss, the original optimizer, fixed6000 of8000, and the ordinary interval head.

**Constrained correction** starts from the incumbent ordinary third member and freezes its entire backbone. It learns only an output convolution correction: `p2_new = p2 + d - mean_time(d)`. Thus the ensemble temporal mean and MVPE are preserved in exact arithmetic. AdamW fits34600 parameters for1000 cosine steps at1e-4, no decay or EMA, phase seed2909, with the current75/25 cooperative loss. Its fixed4000 interval head uses FP32 member predictions. Folding the correction into the original convolution leaves the inference architecture unchanged. Relative-L2 can still decline; this is measured, not guaranteed away.

## Actual deployment comparisons

These are fixed three-member models and learned interval heads scored through the incumbent standalone FP16 inference source. Both recipes use the same ordinary auto-trained control heads. This compares the complete training recipes at deployment precision.

| Recipe | Mimic folds | Δ rel | Δ TKE | Δ MVPE | Δ SPS | Complete local | Without TKE |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4 | +0.002472 | +0.135285 | +0.001333 | +0.068047 | +0.032804 | +0.018504 |
| output_coop | 4 | -0.015724 | +0.204288 | +0.000024 | +0.068819 | +0.031550 | +0.009956 |

Complete local means accuracy plus .2529 SPS; timing is excluded. Individual folds, leave-one-fold-out means and fitted board-weight sensitivity are in confirmation.json. These are local score-point differences, not predicted leaderboard gains.

| Recipe | Holdout | Complete local | Δ rel | Δ MVPE | Δ SPS |
|---|---|---:|---:|---:|---:|
| output_coop | far_keep | +0.033073 | -0.035782 | -0.000052 | +0.039911 |
| output_coop | shift | +0.000288 | -0.011819 | -0.000060 | -0.010947 |
| output_coop | mim2 | +0.042431 | -0.017887 | +0.000028 | +0.083889 |
| output_coop | mim4 | +0.039815 | -0.019598 | +0.000035 | +0.098599 |
| balanced | mim1 | +0.005852 | +0.014975 | -0.003005 | +0.038465 |
| balanced | mim3 | +0.036755 | +0.010618 | +0.005610 | +0.056121 |
| output_coop | mim1 | +0.017687 | -0.013126 | +0.000005 | +0.045212 |
| output_coop | mim3 | +0.026266 | -0.012287 | +0.000029 | +0.047577 |
| balanced | mim2 | +0.057595 | -0.013994 | -0.008817 | +0.103150 |
| balanced | mim4 | +0.031012 | -0.001710 | +0.011544 | +0.074452 |
| balanced | far_keep | +0.016138 | -0.016247 | +0.028852 | -0.028479 |
| balanced | shift | -0.020169 | +0.013334 | -0.001417 | -0.051316 |

## Forecasters

Changes below compare actual three-member ensembles at fixed6000 of8000. The constrained output stage instead uses fixed1000 of1000 and compares with its own initial FP32 ensemble. All other forecaster comparisons use the recorded same-fold ordinary third member and exact frozen companions. Velocity/probe is the local accuracy metric with TKE removed. Balanced loss passed this screening gate. The constrained fluctuation correction failed it and continued only after a pre-recorded complete-model route: its learned SPS gain also had to repay the field cost, with further folds and head-seed confirmation. This distinction is not erased by a positive total.

| Wave / arm | Completed folds | Δ rel score | Δ TKE | Δ MVPE | Weighted accuracy | Velocity/probe |
|---|---:|---:|---:|---:|---:|---:|
| architecture / coop25 | 2 | -0.011720 | +0.214824 | -0.009231 | +0.016299 | -0.006408 |
| architecture / decompose | 2 | -0.011334 | +0.197214 | -0.005298 | +0.014985 | -0.005861 |
| architecture / dmean25 | 2 | -0.011243 | +0.200507 | -0.013255 | +0.014639 | -0.006554 |
| balance_adapt / balanced | 2 | +0.010729 | +0.019841 | -0.003443 | +0.006862 | +0.004764 |
| balance_adapt / balanced13 | 2 | -0.003293 | +0.144081 | -0.009129 | +0.012824 | -0.002406 |
| balanced_confirm / balanced | 2 | -0.007945 | +0.239105 | +0.000906 | +0.021593 | -0.003681 |
| balanced_geometry / balanced | 2 | -0.001261 | +0.070451 | +0.014338 | +0.008177 | +0.000730 |
| balanced_seed3 / balanced | 2 | +0.012058 | +0.106162 | +0.001852 | +0.017106 | +0.005884 |
| edge_coop / coop25 | 1 | -0.056039 | +0.437019 | +0.022813 | +0.021754 | -0.024439 |
| mean_adapter / mean_coop | 2 | +0.000731 | +0.000000 | +0.002096 | +0.000540 | +0.000540 |
| mean_adapter / mean_coop01 | 2 | +0.000514 | +0.000000 | +0.001092 | +0.000345 | +0.000345 |
| mean_adapter / mean_own | 2 | +0.000658 | +0.000000 | +0.001886 | +0.000487 | +0.000487 |
| output_adapter / output_coop | 2 | -0.012707 | +0.153807 | +0.000000 | +0.010237 | -0.006021 |
| output_adapter / output_coop3 | 2 | -0.013168 | +0.164511 | +0.000000 | +0.011150 | -0.006239 |
| output_adapter / output_own | 2 | -0.004295 | +0.052240 | -0.000000 | +0.003487 | -0.002035 |
| output_confirm / output_coop | 2 | -0.018739 | +0.254778 | +0.000000 | +0.018052 | -0.008879 |
| output_geometry / output_coop | 2 | -0.023791 | +0.229847 | +0.000000 | +0.013023 | -0.011272 |
| output_seed3 / output_coop | 2 | -0.015462 | +0.203884 | -0.000000 | +0.014224 | -0.007326 |
| screen / balanced | 2 | +0.013055 | +0.027260 | +0.001211 | +0.009179 | +0.006297 |
| screen / coop25 | 2 | -0.010265 | +0.193289 | -0.000223 | +0.015547 | -0.004884 |
| screen / drop50 | 2 | -0.012146 | +0.198076 | -0.012887 | +0.013989 | -0.006948 |
| screen / mean_guard | 2 | -0.013055 | +0.185510 | -0.012288 | +0.012285 | -0.007323 |
| screen / meangrad3 | 2 | -0.003960 | +0.127798 | +0.000973 | +0.011722 | -0.001786 |
| screen / tke_route1 | 2 | +0.019593 | -0.091924 | -0.008821 | -0.001250 | +0.008466 |
| screen / tke_route3 | 2 | -0.017278 | +0.280287 | -0.010251 | +0.020491 | -0.009135 |

Accuracy weights are .4738 relative-L2 +.1057 TKE +.0926 MVPE. These are local comparison units, not the official final score. Hardware-matched cooperation controls and individual fold differences are retained in results.json. One completed fold is provisional, regardless of sign.

## Learned interval heads

| Wave / recipe | Fold | Head seed | Δ SPS versus paired ordinary head |
|---|---|---:|---:|
| adapter_heads / output_coop | mim1 | 0 | +0.047223 |
| adapter_heads / output_coop | mim3 | 0 | +0.033160 |
| balanced_confirm_heads / balanced | mim2 | 0 | +0.101207 |
| balanced_confirm_heads / balanced | mim4 | 0 | +0.074163 |
| balanced_geometry_heads / balanced | far_keep | 0 | -0.022346 |
| balanced_geometry_heads / balanced | shift | 0 | -0.060347 |
| balanced_heads / balanced | mim1 | 0 | +0.033837 |
| balanced_heads / balanced | mim3 | 0 | +0.053655 |
| head_confirm / prefactors | mim2 | 0 | -0.056163 |
| head_confirm / prefactors | mim4 | 0 | -0.007248 |
| nested_hybrid_mim1 / hybrid | mim1 | 0 | -0.060583 |
| nested_hybrid_mim3 / hybrid | mim3 | 0 | -0.049801 |
| nested_outerspread_mim1 / outerspread | mim1 | 0 | -0.008794 |
| nested_outerspread_mim3 / outerspread | mim3 | 0 | -0.027179 |
| output_confirm_heads / output_coop | mim2 | 0 | +0.070028 |
| output_confirm_heads / output_coop | mim4 | 0 | +0.087374 |
| output_geometry_heads / output_coop | far_keep | 0 | +0.103202 |
| output_geometry_heads / output_coop | shift | 0 | -0.031880 |
| output_headseed1 / output_coop | mim1 | 1 | +0.026118 |
| output_headseed1 / output_coop | mim3 | 1 | +0.040871 |

The corrected physical prefactors also had prior mim1/mim3 results of +.052317/+.017288. Their four-fold mean is **+.001548**, effectively flat. Nested heads use15 preselected calibration cells per outer fold and three inner forecasters excluding both these cells and all outer evaluation cells. The ordinary controls use the same windows, initial head weights and fixed4000 schedule. Evaluation always uses the outer ensemble.

Original nested calibration loses **-.060583/-.049801 SPS**, mean-.055192. The inner models increase calibration-cell MAE by4.1%/6.4% but spread by 15.4%/12.6%. Retaining outer spread while using held-out residuals reduces the losses to **-.008794/-.027179**, mean-.017986. Neither recipe qualifies for confirmation. This does not settle exhaustive cross-fitting over all cells.

Training precision alone was effectively flat: matched FP32 re-evaluation of the old auto-trained and new FP32-trained ordinary heads gives -.008557/+.013985 SPS, mean+.002714. Earlier comparisons using different evaluation precision do not isolate a training effect. The constrained candidate remains positive under matched FP32 evaluation. See precision_results.json.

## Diagnostics and implementation

- Whole-model parameter interpolation between same-seed ordinary/cooperative members fails at every interior dose on both folds. The .5 dose loses about.85/1.53 rel-score points and more TKE; no dose is selected.
- Nested calibration provenance rejects both direct and unsupported ancestral leakage. Loading inner models preserves RNG state; inner models are not packed.
- Cooperative mean-gradient routing preserves forward values and the TKE output derivative, but does not promise parameter-space noninterference.
- The existing decomposed output can be folded into the ordinary convolution. Unsupported output-path combinations are rejected before packing.
- The constrained fluctuation correction freezes all original parameters and fits34600 parameters with zero temporal mean. Its correction is folded into the existing output convolution, keeping three inference passes. The complementary1730-parameter mean-only correction preserves TKE but improves local accuracy by only about+.00054; composing the corrections does not remove the field cost. It is not selected.
- Base hashes, exact frozen backbones and output constraints are independently verified in adapter_constraints.json. Actual packed held-out scores appear above; full-data artifact receipts separately check numerical parity, interval ordering and standalone runtime.
- The first full-data package check failed an absolute temporal-mean change bound of2e-5 (observed4.27e-5 on16 windows). A separate187-window diagnosis found FP32 mean change at most1.49e-7 and essentially identical FP16 rounding error for incumbent and candidate (mean9.092e-6 versus 9.095e-6). Packed MVPE changes only+.000064665 points. The identical weights are reverified using this recorded diagnosis and a .001-point MVPE drift limit; the original failed bound is retained. This is a diagnostic-led revision of a numerical acceptance bound, not a claim that the original check passed. See meanfix_numerical_diagnosis.json.
- The first mim3 mean-gradient run met a GPU memory collision before training. Its log is preserved and an unchanged-source retry is recorded in the manifest. A later H200 comparison was restarted after severe GPU contention. The two first full-data launches failed before training because the launcher set TAIL to an empty string; corrected retries unset it as production does. No failed or incomplete schedule contributes to model comparisons.

Tests cover forward/gradient identities, score units, sampling, nested exclusions, RNG preservation and materialized output folding. Frozen-source manifests and per-job status/log/history files are the reproducibility record.





The literature connections and their limits are in PLAN.md. No novelty or coverage guarantee is claimed for the negative-correlation or hybrid calibration adaptations.
