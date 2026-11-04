**The audit changes two important conclusions.** The scoring formula is recoverable exactly, and CPU contention can make the captured gain path materially slower. Keep A2 as the preferred method to certify for the private retrain; prioritize the executable retraining contract and a K4 clock control. Fund only a tightly gated six-teacher/12k student follow-up. Do not bundle twelve teachers, reopen the forecaster screens, or treat the current clock explanation as settled.

The board snapshot remains roysegal 81.88925, us 81.80680: gap 0.08245, shortlist margin 0.23901. Those are snapshot facts, not a newly fetched leaderboard. [Source: ../claude/live_board_2026-09-17.json.] This audit used read-only API requests, CPU analyses and approximately eight minutes of GPU probing on idle H100 3. No training, uploads, commits, shipped-code edits or living-document edits were made. The working tree was already dirty. All new artifacts are in this directory. The independent ranking was written before opening Claude's ideas; its SHA256 is 03c36e5dbcda43ba92e2e99237db01999da1766d17751711a07c854582b9b5da. [Sources: independent_ranking.md, audit_receipt.json, runtime_budget.json, paired_runtime.json.]

**The history is incomplete, and final_score is not a weighted sum.** Claude's raw submission response contains the first API page only. Following `next` recovers 56 uploads, 41 scored, rather than 50/36. The omitted scored cards are 879338, 879743, 881057, 882102 and 882969; 879273 is an additional failed upload. The historical “34-card” fit and today's “36 cards” consequently refer to different populations, not successive nested samples. [Sources: all_submissions_api.json; ../claude/my_submissions_raw.json; ../../distil_2026-09-16/codex_audit/refit_weights.json.]

A generalized mean reproduces every available full-precision final score:

`F = (0.5 rel^p + 0.2 sps^p + 0.1 tke^p + 0.1 time^p + 0.1 mvpe^p)^(1/p), p = 0.6854`.

The rounded constants above give RMS 3.81e-11 and maximum error 8.72e-11 on all 41 cards. Refitting the five free parameters leave-one-out gives RMS 3.73e-11. The five older cards recovered after the initial discovery also reproduce. This is an empirical identification from our scorecards, not an assertion that the organizers published the formula. It is substantially stronger than another local linear refit. [Sources: power_mean_audit.py, power_mean_audit.json; discovery: power_fit_exploratory.json.]

| Model and card population | rel | sps | tke | time | mvpe | Fit RMS | LOO RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear, supplied 36, including P0/P1 | .41450 | .27619 | .10078 | .07884 | .16109 | .02131 | .02838 |
| Linear, supplied 36 minus P0/P1 | .47750 | .24514 | .10058 | .09654 | .09484 | .000401 | .000630 |
| Linear, complete 41 | .41900 | .26480 | .12873 | .07361 | .14357 | .05971 | .07690 |
| Linear, complete 41 minus P0/P1 | .46485 | .25131 | .10184 | .09160 | .10850 | .00596 | .00899 |
| Exact local derivatives at 926634 | .47730 | .24319 | .10105 | .09641 | .09569 | — | — |

Sources: score_audit.json and power_mean_audit.json. The locked coefficients themselves have RMS .18674 on the supplied 36. Including an intercept does not repair global linear misspecification: LOO RMS .02781 there. Excluding probes produces a useful operating-point approximation, but the exact formula explains why the approximation changes with the population. The outer power-mean coefficients are not the marginal exchange rates; use `dF/ds_j = w_j (F/s_j)^(1-p)`. There is no justification for another forecaster loss sweep merely because the SPS marginal changes from .2529 to .2432.

The old constant time-weight story also becomes clear. Identical-accuracy cards identify a local secant; the persistence probes inhabit a different operating point. Their different secant is not evidence that the organizers changed weights. The remaining empirical uncertainty is intervention transfer and runtime, not the algebra mapping known subscores to final.

**Several close decisions and trend attributions need correction.** Here N means the exact change after replacing the accuracy subscores while holding time_score at the comparator's value. For the endpoint decomposition I also computed symmetric Shapley contributions across the five channels; these sum exactly to observed final change.

| Comparison | Observed final change | Exact N | What follows |
|---|---:|---:|---|
| wd03mv vs mvpe03b, 918211–917205 | +.005035 | -.010041 | A time-funded public win; no demonstrated public accuracy gain from decay |
| K4 vs K3, 923184–918211 | +.037850 | +.038863 | A substantive accuracy gain, not just a latency repricing |
| Half-dose gain vs K4, 924146–923184 | -.014461 | +.009415 | Positive, but below the registered +.010 N threshold |
| Pair vs gained h0, 926634–924146 | +.015620 | +.019626 | Clean interval-estimator improvement |
| Pair plus gain vs K4, 926634–923184 | +.001158 | +.029043 | Clock removes approximately 96% of the gain, not 90% |
| A2 vs pair, 929146–926634 | -.018286 | +.000947 | Tiny positive interval increment; timing causes the final loss |

Source: power_mean_audit.json; original gates: submissions/_manifest/CANDIDATE_k4g.md, CANDIDATE_k4h2g05.md and CANDIDATE_k4a2.md. The gain passed the two approximate fits available at registration; do not rewrite that history. But it did not clear the intended economic threshold under the now-identified formula. Its exact break-even against 923184 is 5.41345 ms. The observed A2 breaks even against 926634 at 5.71682 ms. These are different from forecasts using an unobserved SPS improvement.

From 904950 to 926634 the final increased .209856. Exact channel attribution is rel +.002656, SPS +.049194, TKE +.007397, MVPE +.018278, time +.132330: approximately .07753 accuracy and .13233 time. Claude's audit's “+.045 accuracy” is arithmetically wrong. STATE's “+.008 of +.171” describes the earlier 918211 endpoint; it is stale as a summary through September 17. K4 contributed +.03886 non-time and should not disappear from the modelling narrative. [Source: power_mean_audit.json; contrasting claims: ../claude/AUDIT.md §2, STATE.md.]

K4's SPS gain does not isolate “spread information”: its mean forecast and a refitted head also changed. Likewise, weight decay's rel/MVPE benefit and TKE cost do not establish the generalization mechanism claimed by its in-sample explanation. The interval pair is cleaner because point predictions stay fixed. A2 establishes preservation of most of that interval benefit on this public set; it does not measure variance across complete private retrains.

**The prediction failures are principally transfer and clock failures, not unexplained score-fit noise.** The important documented misses and non-misses are:

| Registration | Forecast → observed | Audit reading |
|---|---|---|
| 903484, higher SPS loss | +.05 final → -.029 | Documented band miss: TKE cost transferred much more strongly; interval improvement less strongly |
| 908202, exact width-profile repeat | [81.586,81.640] → 81.58194 | Clock forecast missed; identical accuracy does not identify an architectural runtime penalty from two days alone |
| 910411, onelean | +.08 [0,+.15] → -.04664 | Tail TKE funding collapsed; SPS loss transferred more strongly; the tail-head evidence also required leakage repair |
| 911563, no-spread head | +.02 [0,+.06] → -.04388 | SPS was +.00460, not the +.22 implied by the old transfer coefficient; almost all final loss was time |
| 914270, CUDA graphs | +.25 [0,+.40] → +.05797 | Broad band hit, substantially optimistic central timing model |
| 916264, Triton | +.02 [0,+.05] → -.04552 | Timing falsifier fired; matching accuracy supports fallback, but no direct fallback telemetry was recorded |
| 924146, half-dose gain | Time-score [-.08,+.08] → -.24779 | Accuracy predictions broadly landed; time band missed; final remained inside its broad band |
| 926634, head pair | Head-only SPS +.065 [-.020,+.150] → +.080683 | Mechanism prediction passed; one point did not identify a GPU-to-board clock slope |
| 929146, A2 | SPS 43.9596 [43.9096,44.0096] → 43.9453; final [81.794,81.862] → 81.78851 | SPS band hit, final band missed; incremental SPS transfer was .173, not .81 |

Sources: corresponding candidate manifests; docs/archive/FINDINGS_2026-09-10.md, especially the 903484 post-mortem; docs/archive/log/2026-08-sessions.md for 908202; full precision scores in all_submissions_api.json. The decay and cooperative candidates also landed inside broad total-score bands while important channels missed. Do not convert a band-hit count into a validated transfer model. I could not reconcile the archive's changing headline counts into a complete timestamped forecast ledger and therefore do not report a new success fraction.

Other concrete stale statements: docs/MODEL.md still describes three members and the old footprint; FINDINGS §2 still repeats the old fourth-member closure; PLAN contains superseded slot schedules and “clock settled” language. The interval policy is optimal within the tested correction families, not a proved Bayes optimum. `SPS = 100 mean(R) mean(W)` is not generally exact: scoring.py averages their element-weighted product, and covariance/window masks matter. Finally, the outage interval labelled September 15 21:07 to September 16 approximately 07:00 is approximately 9h53, not 34 hours; the broader absence of successful uploads is a different observation. These are corrections to make later, not edits performed here. [Sources: named living documents, starting_kit/scoring.py:166, CANDIDATE_k4h2g05.md.]

**The clock has a measured code-by-CPU interaction.** I reproduce the cluster means: pre-gain 5.34101 ± .08196 ms, n=6; gain-chain 5.82942 ± .18661 ms, n=4. Their difference is .48841 ms. With the exact score map, replacing the early mean by the late mean costs approximately .03120 final for A2. Code and date are perfectly confounded in these cards, and the pre group includes an eager persistence probe. This is a descriptive separation, not a randomized test. [Sources: score_audit.json, power_mean_audit.json.]

The AST comparison finds changes only in `_matched_gain`, `_predict_eager` and `_post`. `_Graphed`, `_build_graphs`, `_pieces`, `predict` and `_predict_graphed` are unchanged; K4g and A2 source is byte-identical. Gain is inside the captured post graph. There are no added Python launches of the gain on each prediction. The three fp16 output arrays, their shape and copy count are unchanged: N=1 returns 737,280 bytes. The graph self-check still checks the smallest/largest buckets against eager at max-absolute tolerance .002; its capture budget is 40 seconds. An exception clears the graph dictionary, and runtime exceptions fall back eagerly. [Sources: code_path_receipt.json; submissions/k4g/submission.py; runtime_*.json.]

At N=1 `_pieces` gives one small bucket and four concurrent member replays plus the post replay. N=2/3/4 likewise use exact buckets; N=5 pads to 6; N=9 uses 8+1; N=15 uses 8+7 padded to 8; N=33 uses 32+1. No gain-specific batching branch exists. Importantly, the API logs reveal per-sample shape, not actual call batch size. N=1 remains a well-motivated working hypothesis, not observed harness telemetry. This audit timed N=1/4/16 and checked the decomposition separately. [Sources: runtime_probe.py, runtime_summary.json, api_logs/*.txt.]

I ran 24 fresh processes, balanced ABBA then BAAB, on idle H100 3 with torch 2.2.2+cu121. The CPU experiment pins to CPU 0; starvation adds one busy process on that same CPU. All processes captured all nine buckets and recorded zero runtime eager fallbacks. Mean N=1 timings, averaged across processes:

| CPU condition | K4 ms | K4g ms | Difference |
|---|---:|---:|---:|
| Normal affinity | 2.8179 | 2.8743 | +.0564 |
| Pinned, no competitor | 2.8691 | 2.8700 | +.0009 |
| Pinned, busy competitor | 4.2607 | 4.4914 | +.2307 |

The process-median difference in the last condition was +.8006 ms; it must not be substituted for the board's mean-time statistic. Because this was a new concern, I tightened the control: both modules resident in one process, balanced individual calls on release inputs, alternating quiet/contention periods. [Sources: runtime_*.json, runtime_summary.json, paired_runtime.py.]

| Same-process period | K4 mean ms | K4g mean ms | Difference |
|---|---:|---:|---:|
| Quiet before | 2.9178 | 2.9665 | +.0487 |
| Contention 1 | 4.6270 | 4.9981 | +.3711 |
| Quiet between | 2.8313 | 2.8752 | +.0439 |
| Contention 2 | 4.4203 | 5.0309 | +.6106 |
| Quiet after | 2.8467 | 2.8899 | +.0432 |

Source: paired_runtime.json. All nine graphs remained active; neither module fell back. N=1 graph/eager outputs matched exactly on the checked fixture. The smaller quiet penalty is real GPU work; its amplification under scheduling contention is consistent with synchronization/descheduling interactions, but I did not trace the scheduler and cannot identify the precise mechanism. Individual contention blocks vary, including small negative differences. This H100 experiment demonstrates possibility and repeatable period-level amplification; it does not prove the A800 ingestion environment has this condition. It directly refutes “captured gain means no plausible runtime cause.”

A full eager fallback was 16.33 ms at N=1 for K4 on this H100; earlier A100 measurements were 26–27 ms. Persistent fallback alone is consequently an implausible explanation for a half-millisecond shift at N=1, but intermittent fallback, different call sizes, CPU interference and platform changes cannot be excluded by that comparison. [Sources: runtime_normal_0_k4.json; CANDIDATE_k4g.md.]

**The API does not supply the host control previously claimed.** I fetched the detail endpoint and all eight exposed logs for each of twelve relevant cards. Every ingestion log reports A800-SXM4-80GB. The terminal `status_details` contains the scoring host; it is not the inference host. The authenticated owner response and its detail response expose neither execution start/finish timestamps nor ingestion hostname. The source serializer confirms these omissions. Thus repeated scoring hosts across the clusters do not rule out ingestion host assignment. The pair's rpw6 and A2's rpw2 ingestion identities are available from their manifests, not independently recovered in these terminal responses. [Sources: api_clock_summary.json; api_logs/; https://github.com/codalab/codabench/blob/develop/src/apps/api/serializers/submissions.py.]

| Card | Scoring host | Prediction-log to scoring-log object timestamp gap, seconds |
|---|---|---:|
| 915299 / 917205 / 918211 / 920181 | rpw1 / rpw7 / rpw3 / rpw1 | 151 / 157 / 166 / 200 |
| P0 / P1 | rpw0 / rpw3 | 45 / 47 |
| K4 / K4g / K4g repeat | rpw7 / rpw3 / rpw0 | 233 / 176 / 246 |
| Pair / A2 | rpw7 / rpw1 | 184 / 173 |

Source: api_clock_summary.json. These are log-object creation timestamp gaps, **not measured ingestion durations**. They may include setup, queues, serialization and logging policy. Their nonmonotonicity does not isolate the timed prediction path. True ingestion/scoring execution durations, CPU allocation, actual N and graph/fallback telemetry remain unavailable in the exposed records. I did not access hidden prediction data or evaluation files.

The September 18 exact K4 repeat is the right first public control. A fast result supports a gain-dependent effect; a slow one supports a common platform component. Neither single outcome logically settles mutually exclusive causes, because both can coexist. The decisive engineering control is same-worker, same-fixture, actual-harness-N, randomized paired K4/K4g timings with recorded CPU affinity/load and graph/fallback status. We now have that control locally under imposed CPU contention, but not on the organizer worker. Prefer mean timings and subsequent independent confirmation before repricing a code change.

At the pair's operating point, d(time_score)/d(ms)=-.654639 and d(final)/d(board-ms)=-.0631167. One local millisecond costs .01262/.01894/.02272 final **conditional** on board/local slopes .2/.3/.36. A noisy ordering with the two-head artifact between single-head artifacts does not prove zero slope. If the shift is shared platform drift, use a late-period centre provisionally: existing A2 at 5.8294 ms predicts 81.79974. If a gain-dependent penalty can be removed, the same A2 accuracy at 5.3410 ms predicts 81.83094. Removing the gain also removes approximately .0094 accuracy; that trade needs roughly .15 board-ms of savings. [Sources: power_mean_audit.json, slot_pricing.json.]

**The shipped-model audit finds a delivery gap, not evidence that A2's weights are wrong.** The earlier replay reproduced A2's 80 fp32 tensors and an independent teacher exactly, and packed held-out scoring reproduced its advantage. I did not repeat that expensive audit unnecessarily. I did rerun today's `check_recipe --pack submissions/k4a2/model.pth`: it exits 1 with four disagreements—fixed-step metadata, absent init-teacher, teachers and under-weight in the canonical recipe. The script still defaults to four members plus an ordinary head and copies K4's gain-free code; the pins still identify 923184. There are three different things: public best 926634, tested student 929146, canonical rebuild 923184. [Sources: check_recipe_A2.log; tools/train_final.sh; tools/verify_best.py; ../../distil_2026-09-16/codex_audit/SUMMARY.md.]

Do not waive the checker failure. Packing parity establishes fidelity to a supplied source checkpoint, not that the source completed the intended training. The prior audit's substituted-source, copied-provenance, duplicate-teacher and step-2000 mutants remain the relevant negative tests. A complete clean rebuild, source/data/launch receipts and fail-closed gates have more private-phase value than another public thousandth. [Source: ../../distil_2026-09-16/codex_audit/RETRAIN_CONTRACT.md.]

**My independent modelling ranking was short.** First, preserve A2's estimator-averaging benefit and test its transport across a fresh member draw and genuinely different exclusion geometry. Second, improve compression with the measured six-teacher/12k combination, conditionally. Third, diagnose the common teacher-error floor from disjoint seed groups before funding more teachers. No forecaster mechanism earned funding. The original priors, costs and kill rules are preserved in independent_ranking.md, rather than rewritten after seeing Claude's list.

The fixed-prefix K=2/3/4/6 gains are .10011/.12486/.13522/.14905 fold SPS. Fitting G(K)=Ginf(1-1/K) through the origin gives Ginf=.18393; it suggests only .01533 additional teacher SPS from K6 to K12. Its residuals are systematic, the prefixes share heads, and the baseline is a particular seed: it is a heuristic, not an identified ceiling. A2=.12255 and the 12k combination=.13950 show that optimization/compression remains measurable. The existing +.0169 over A2 is worth .00331 final at transfer .806, but only .00071 at today's marginal transfer .173. Neither transfer is a general law. [Sources: head_scaling_audit.json; ../../distil_2026-09-16/codex_audit/teacher_scaling.json and sensitivity_summary.json.]

The public pair gain supports averaging fitted functions. It does not separate initialization variation from minibatch-order/optimization variation. A cosine learning rate ending at zero does not erase stochastic trajectory dependence. Catastrophic weight soups show unaligned parameter representations, not exclusively “basin variance.” Six teachers share data, architecture and forecasters; distillation reduces conditional fitting variability, not necessarily variation from new regimes or fresh forecasters. The private retrain is one draw of all these sources. Avoid seed selection; preserve four ordinary forecasters, fixed teacher prefixes, fixed steps, and exact manifests.

The fresh geometry should combine a high-Re exclusion with the 5-degree frontier, while retaining matching simulation. `joint5` already exists, so calling that existing fold new would repeat the audit's own error. PREREG_NEXT.md specifies a new combined 15-cell exclusion, whose real trajectory identities I checked. It remains reused release data, not an independent private distribution. The existing 34 fold-cell readings represent 28 distinct cells; A2 loses to the pair on 10/34 in packed evaluation, and the known worst teacher target already loses before compression. Better distillation cannot remove teacher bias there. [Sources: research/improve_2026-09-10/claude/launch_wave.py; PREREG_NEXT.md; ../../distil_2026-09-16/codex_audit/SUMMARY.md.]

**Cross-examination and merged ranking.**

| Priority | Merged action | Public / private planning range in final points | Evidence, cost and kill |
|---|---|---|---|
| 1 operational | K4 repeat; conditional gain-free A2 timing control | Up to about +.03 timing recovery; removing gain costs about .0094 accuracy / runtime-specific | New CPU interaction makes Claude C1 stronger but its binary interpretation weaker. One control slot; research build only if indicated. Kill deployment change unless mean-time savings compensate measured accuracy loss. |
| 2, essential | Certify six-teacher/4k A2; compare pair on fresh members/geometry | Observed +.000947 non-time vs pair / subjective [-.015,+.015] | Claude C3 agreed, but private variance reduction is not quantified. Engineering 1–2 days; about 1.8 A100-h per fresh four-member set plus heads. Kill delivery on any provenance/recipe failure; hold private superiority if fresh-fold SPS <0 or new-cell loss exceeds .03. |
| 3, conditional | Six teachers, initialized asymmetric student, 12k | Centre +.001, [-.003,+.005] / [-.010,+.006] | Existing +.0169 fold SPS. At most 1 additional GPU-h with cached teachers. Require new joint-fold SPS gain >=.010, existing folds >=-.010, new cells >=-.030 versus A2. |
| Reserve, unfunded | K12 teachers | [-.005,+.004] / [-.010,+.006] | Claude C2 bundles an unrun K12 extension with a measured K6/12k result. The additional teacher gain is small and may not transfer. Diagnose seed-prefix/common error first. |
| Unfunded | Head refit against gained forecast | Claude's +.005 central is not established | Gain changes point prefactors/guard; with the plain bracket retained it does not create new target information. Compare matched teacher ensembles before compression, not one refit directly against A2. Existing gain interaction with A2-minus-pair is only about 2e-5 fold SPS. |
| Closed for this deadline | Forecaster screens and Re-gated mixing | No positive planning budget assigned | No new admissible source demonstrated; the proposed Re gate reuses the closed conditioning/mixing trade and does not identify whether an angle is unsupported. |

Ranges are my subjective planning judgments, not empirical intervals; direct evidence and cost sources are independent_ranking.md, PREREG_NEXT.md, head_scaling_audit.json, paired_runtime.json, and the cited prior audit. Claude's list is ../claude/IDEAS_claude.md.

I disagree especially with Claude C5's proposed universal TKE floor. Lag-one correlation and window length alone do not determine the conditional uncertainty of future sample variance. A predictable sinusoid can have high lag-one correlation and perfectly predictable finite-window variance. A stochastic process needs its conditional dynamics and spatial covariance specified before a relative-L2 floor follows. The target-assisted oracle does not solve that identification problem. Do not spend time turning an assumed 30–50% sampling-noise argument into a numerical ceiling or using it to declare the forecaster information-limited.

There is no compelling new forecaster information source in the allowed interface. In-window gradients, vorticity, inferred Re or convective coordinates are representations of existing measurements; simulation provides unpaired regime information already investigated through transfer, mixing, conditioning, frontier and covariance tests. External weights, new simulations and cross-window inference are outside the supplied contract. This is a funding judgment before the deadline, not proof that all possible inductive biases are exhausted.

**Prospective registrations and slots.** Exact executable-stage choices, predictions, bands, instruments, kill thresholds and a concrete new holdout are written in PREREG_NEXT.md. They are proposals for subsequent authorized work; none of these builds or uploads was executed here. The funded registrations are: R1 exact K4 control; R2 conditional gain-free A2; R3 clean retrain and fresh-geometry validation; R4 conditional K6/12k compression. No registration authorizes a parameter sweep after a miss.

| UTC date | Slot | Reading and fallback |
|---|---|---|
| September 18 | Byte-identical pinned 923184 | Predict 5.34 [5.1,5.6] ms under a gain-dependent effect, or 5.83 [5.4,6.25] under a shared drift component. Accuracy fixed; quarantine any subscore departure >.005. <=5.4 favors code dependence; >=5.6 favors shared drift; intermediate is inconclusive. |
| September 19 | If K4 is fast, verified gain-free A2 control; otherwise a complete fresh-retrain calibration | Gain-free A2 central SPS 43.9253 [43.9053,43.9453], central final 81.8216 at 5.34 ms; at central SPS it beats the holder below 5.5686 ms. If the retrain is not ready, an explicitly registered exact A2 repeat calibrates the late clock; it is not an automatic repeat granted by the old registration. |
| September 20 | Freeze-confirmation artifact from the accepted complete recipe | Prefer the independently built canonical artifact. Use K6/12k only after the fresh gate and complete pack certificate pass. If unfinished, spend the slot on a named existing control and freeze the simpler verified method; do not promote unfinished training. |

Source: PREREG_NEXT.md and slot_pricing.json. For an unchanged A2 calibration, SPS stays 43.9453065 with the other accuracy values fixed; a provisional 5.83 ms clock predicts 81.79970, and a 5.4–6.25 ms timing band maps to approximately 81.82709–81.77405. An exact repeat measures public runtime only. A fresh-build submission measures the combined retrain change and runtime; separate those channels rather than selecting whichever seed gets a lucky final. Best-retained scoring makes calibration useful even when none becomes a new best.

**Decision-Phase recommendation.** Freeze four freshly trained forecasters with the existing 8000-step schedule and step-6000 selection, six fresh SPS teachers on those exact members, and the fixed initialized asymmetric A2 student; discard teachers from inference. Certify this with the complete clean-room contract, immutable ordered-data/code/environment/launch receipts, tensor and completion gates, negative mutation tests, standalone inference and an honest new joint-geometry comparison. Prefer the student over the pair for its demonstrated public interval retention and lower GPU work, while retaining the pair as a method fallback if fresh validation fails. Keep dose .5 provisionally, but explicitly condition its final inclusion on the CPU/board control because its exact accuracy value is only +.009415. Do not silently add twelve teachers or 12k student training. Freeze method, portable command, seeds and checkpoint rule together; an organizer rebuilding today's default script would still get the wrong method.
