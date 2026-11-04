# Candidate k4h2g05 — two independently fitted interval heads, output-averaged, on the k4g point forecast (pre-registered 2026-09-15, joint Claude + Codex; NOT uploaded)

**What.** 923184's four members and the board-tested **dose-0.5** fluctuation gain, unchanged. The only change is the
interval: the *bounds* of two independently trained width-96 heads (seeds 0 and 11, same lr 8e-4 / τ 0.05 / step 4000,
both reading the same plain ensemble mean and spread) are averaged 50:50. No mixture-weight sweep, no seed search.
`research/slot_2026-09-15/codex/k4h2g05.zip`, sha256 `c743161a…b31bc39e7`, 233,629,780 bytes extracted (cap 268,435,456).

**Why it is not a closed axis.** Head *weight* soups, wider heads (144/192, failed 09-13), head inputs, head constants,
post-hoc widths and nested/OOF heads are all closed; none of them ensembles the head's *output*. The mechanism is the
one that already paid for the forecaster — averaging out fitting variability — applied to the estimator that owns
the lowest subscore. It was an unfunded row in Codex round 4 with no measured closure. It is not the withdrawn
cross-basin head mixture (that mixed training data; the organisers refit heads themselves, and a two-head protocol
is something they can refit).

**Held-out gate, registered before the fits** (`research/slot_2026-09-15/codex/HEAD_PAIR_PREREG.md`: mean actual SPS
≥ +0.08, every fold ≥ 0, no fold worse than either single head by 0.03). Six heads fitted on frozen current-recipe
four-member fold ensembles, outer-training data only:

| fold | ΔSPS vs head 0 | smallest cell |
|---|---|---|
| mim1 | +0.0739 | +0.0399 |
| far_drop | +0.1252 | +0.0940 |
| angle05 | +0.1003 | +0.0055 |
| **mean** | **+0.0998** | all 34 fold-cell readings positive |

**PASSES.** Attribution (Shapley, `headpair_decomposition.json`): ≈ 2/3 of the gain is interval *centring*, 1/3 width.
The folds overlap in regime; they are not 34 independent observations. In-sample on the packed weights: +0.0539 SPS
over k4g with every accuracy score identical. Board prediction shrunk to **+0.065 SPS [−0.020, +0.150]**.

**Why this artefact reads cleanly.** Verified on an idle A100 (`research/slot_2026-09-15/verify_pair.sh` log): the
prediction is **bit-identical to k4g** — time-mean max abs diff 0.0, temporal-variance ratio exactly 1.0 — so rel, TKE
and MVPE are 924146/925203's twice-measured board values to the digit and only two quantities move: **ΔSPS (the
mechanism) and Δtime (the clock)**. One slot therefore prices the head pair *and* the graphed-compute clock slope,
with nothing else confounding either.

**Prediction vs 923184 (judgement bands).** rel **94.82938** (−0.0234, fixed) · TKE **79.13200** (+0.1555, fixed) ·
MVPE **94.10131** (+0.000004, fixed) · SPS **43.926** (+0.085 [0, +0.170]) · time = the day's clock.
**N (non-time, SPS counted once) = +0.0268, working sd 0.0131**; central final ≈ **81.808**, band 81.760–81.852.

**The clock exposure, which is the whole risk.** Paired idle-A100 medians, three interleaved cycles, fresh process per
zip, 300 N=1 calls each: k4 4.175 · k4g 4.082 · k4g10 4.111 · **k4h2g05 5.157 ms** → the second head costs
**+1.05 ms locally** (Codex's independent read: +0.85). All nine graph buckets capture; no eager fallback in 50
mixed-size cases. What the board charges for that is the unresolved item in `FINDINGS.md` §5:

| board-ms per local GPU-ms | Δfinal from the clock | P(new best) |
|---|---|---|
| 0 | 0 | ≈ 0.75 |
| 0.36 (the P0→P1 slope) | ≈ −0.034 | ≈ 0.42 |
| 1.0 | ≈ −0.094 | ≈ 0.05 |

The same-chain evidence favours ≈ 0: P0 (our payload, almost no GPU work) read 5.34 ms and the incumbent (3.9 ms of
*graphed* GPU work) read 5.25 — a 3.9 ms lever arm with 0.09 ms of board response. The 0.36 slope comes from P1, whose
extra 3.2 ms was ungraphed synthetic work. k4h2g05's extra millisecond is graphed. Codex's 50/40/10 mixture over the
three slopes gives **P(new best) ≈ 0.55** and expected retained-best **+0.0122** (vs 0.50 / +0.0085 for k4g10).

**First place is not reachable tonight.** The leader is 81.85874; closing 0.0531 needs N ≥ 0.065 ≈ +0.21 SPS. P ≈ 3 %,
and eight clock reads cannot support that tail. The reachable band is rank 2–3.

**Decision rule (registered before the score).** Promote if the final improves AND N ≥ +0.010 under both fitted weight
sets AND the new head adds ≥ +0.030 SPS over 43.86073. **Kill** if the head adds ≤ 0 SPS, or N ≤ 0 under either fit.
**Quarantine** (suspect a build fault, do not interpret) if any of rel / TKE / MVPE differs from 924146's value by
more than 0.005 — they are algebraically fixed. A timing-only failure promises no repeat; there is no automatic seed,
mixture-weight or third-head search. A third head fits the cap but ordinary estimator-variance scaling predicts about
a third of this increment, and that is not a registration.

**The obvious follow-up if it pays:** distil the averaged bounds back into a single head. The gain is the removal of
fitting noise, which a single head fitted to the averaged target keeps, at one head's latency — removing the entire
clock exposure. ≈ 0.5 A100-h, needs its own held-out gate.

## Alternative on the shelf
`k4g10` (gain dose 1.0, `CANDIDATE_k4g10.md`, build sha256 `50c40daf…cf9cf5f60`, independently rebuilt byte-identical
by both passes): verified, 4.111 ms, no clock exposure, N = +0.012 (Codex, channel-wise transfer) to +0.015 (Claude,
in-sample ratios), P(new best) ≈ 0.50. Its incremental value over the already-scored dose 0.5 is only ≈ +0.002.

## Armed (owner's go, 2026-09-15 21:07 UTC)
`tools/submit_at.sh <now> research/slot_2026-09-15/codex/k4h2g05.zip` — log
`research/slot_2026-09-15/submit_k4h2g05.log`; zip sha256 `c743161a5a9abc50603c033f1945c91bde4a78bcf5fe23859162b32b51bc39e7`.
Submitted mid-day rather than at 00:00:30Z because the 09-15 slot was otherwise lost. The time-of-day caveat is
weaker than it first looked: **P0 (921738) was itself uploaded at 2026-09-10T21:13:35Z and drew 5.34 ms**, just under
the 5.46 ms mean of the eight reads, so the one mid-day point we have sits mid-distribution.

**Upload blocked by a platform fault, 21:07–21:25 UTC:** `POST /api/datasets/` (dataset reservation, step 1 of 4,
before any bytes move) returns HTTP 500. Not our artefact — the same call fails for a 191,640,125-byte reservation,
the size that uploaded successfully on 09-13. `GET` endpoints are healthy; the quota stayed `can: True` and no
submission was created. `research/slot_2026-09-15/retry_submit.sh` re-checks the quota before every attempt (so a
submission that succeeds but reports badly cannot be sent twice) and retries every two minutes until 23:50 UTC. Prediction, kill number and quarantine rule as above, recorded before the score.

## Board result — submission 926634, uploaded 2026-09-16T07:04:14Z, scored 07:09 UTC: final **81.80680, a new best (+0.00116)**
Ingestion rpw6, scoring rpw7. Uploaded on the 09-16 slot, not 09-15: Codabench's `POST /api/datasets/` write path was
down for ~34 hours (see below) and the 09-15 slot expired with the artefact ready.

| | rel | tke | mvpe | sps | time | final |
|---|---|---|---|---|---|---|
| 923184 (k4) | 94.853 | 78.977 | 94.1013087 | 43.841 | 92.164 (5.27 ms) | 81.80564 |
| 924146 (k4g) | 94.8293790855 | 79.1320014641 | 94.1013126942 | 43.8607297 | 91.916 (5.64) | 81.79118 |
| **926634 (k4h2g05)** | 94.8293790855 | 79.1320014641 | 94.1013126942 | **43.9414123258** | 91.8745 (5.70) | **81.80680** |
| Δ vs k4 | −0.023621 | +0.155001 | +0.000004 | **+0.100412** | −0.289486 | **+0.00116** |

**Against the registration.** rel predicted −0.02342 → −0.023621; tke +0.15550 → +0.155001; mvpe +0.000004 → +0.000004
— all three reproduce 924146/925203 **to ten digits**, exactly as the algebra required, so the quarantine rule did not
fire and the SPS and time channels are the only things that moved. sps +0.08502 [0, +0.170] → **+0.100412** ✓ (above
centre). **Head-pair-only SPS (vs k4g's 43.8607297): +0.0807**, against a +0.030 promote threshold and a shrunk
prediction of +0.065 — the held-out fold mean of +0.0998 transferred at **0.81×**, far better than the 0.63× the
dose-0.5 read suggested for this channel. N = **+0.0306** (locked) / **+0.0302** (34-card), against +0.010.

**Verdict: PROMOTE.** Final improved, N ≥ +0.010 under both fits, the head added ≥ +0.030 SPS. Output-averaging two
independently fitted heads is a real, transferred mechanism and is now the best artefact on the board.
`submissions/k4h2g05/` should become the shipped directory (see the promotion checklist below).

**What it did not buy: the rank.** Still #4. np-user 81.85874 · skabob 81.82371 · roysegal 81.81876 · **us 81.80680**.
The clock took 5.702 ms and −0.289 of time_score = −0.0275 of final, eating **90 % of the +0.0306 accuracy gain**.

**The clock slope, measured against my own recommendation.** I argued for this artefact partly on the reading that
graphed compute is nearly free on the container clock (P0 with no GPU work read 5.34; the incumbent with 3.9 ms of
graphed work read 5.25). **That reading is now the less likely one.** This read is the cleanest evidence available —
same chain, same scoring host, a known +1.05 ms local delta: rpw7 has previously returned 5.27 (923184) and 5.49
(917205); 926634 returned **5.702**. That is +0.21 to +0.43 board-ms for +1.05 local ms, i.e. a slope of roughly
**0.2–0.4**, consistent with P1's 0.36 and *not* with zero. One read against a clock with sd 0.243 is not conclusive,
but it is the best-controlled point we have and it points the other way from the argument I made.
**Consequence: the distillation follow-up is now the valuable one, not a nicety.** Fitting a single head to the
averaged bounds would keep the +0.081 SPS at one head's latency — worth ≈ +0.031 of final with no time price, i.e.
≈ 81.834 and rank 2. ≈ 0.5 A100-h plus its own held-out gate.

**Promotion checklist (not yet done — recipe changes are the owner's call):** move the pins to `submissions/k4h2g05/`;
teach `train_final.sh` to fit two heads (`HEAD_SEED` 0 and 11) and `build_ckpt`/`check_recipe`/`check_pack` to carry
and verify both; re-derive `tools/release_manifest.json`; re-run `verify_best`. The Decision-Phase retrain must fit
both heads on the organisers' own members, so the two-head protocol has to be described in the method, not shipped
as weights.

## Platform outage, 2026-09-15 21:07 → 2026-09-16 ~07:00 UTC (~34 h)
`POST /api/datasets/` returned HTTP 500 or hung. **Not our side**, established by measurement, not assertion: the
record *is* created each time (correct name and owner, valid key, `file_size` 0) and the failure is downstream where
Codabench mints the presigned upload URL, so our payload is accepted; `GET /api/my_profile/` returned 200 in 0.16 s
throughout, so the token is valid (a bad token gives 401/403, never a created record); it failed identically at
`file_size` 123 / 1000 / 191 MB / 216 MB; our stored footprint is 7.42 GB across 93 datasets while other users hold
single datasets up to 12 GB, so no storage cap; and **another team hit the identical zero-byte failure**
(`hemingshuaizhatian - submission`, 2026-09-16T01:59:07). No upload by anyone recorded non-zero bytes between
2026-09-14T00:00:54 and the recovery. **38 zero-byte orphan records remain on the account** (27 from 09-15, 11 from
09-16), inert (`in_use=[]`) and not yet cleaned.
**Our own failure inside that window:** the first retry loop died at 22:15 UTC when the home quota filled — the log
write failed and it crashed on a traceback nobody saw, so a recovery before 00:00 would have been missed anyway.
The hardened loop (`research/slot_2026-09-15/retry_submit.sh`: log on <data-disk>, `timeout` on every call,
quota re-checked before each attempt, tolerant status polling) caught the recovery within ten minutes.
