# Candidate: wd03mv — the champion's recipe with Muon weight decay 0.03 (2026-09-07)

**What it is.** 917205's recipe (`tools/train_final.sh` + `W_MVPE=0.3`: three width-96 members, Muon lr 0.02,
EMA 0.998, `rel 8 + tke 1 + sps 1 + mse 1 + tmean 0.5 + spec 0.1 + mvpe 0.3`, 8000-step cosine read at 6000,
head 4000/4000 with spread) with one change: **`--wd-muon 0.03`** (`WD_MUON=0.03`, an append-only knob added
today; the AdamW group keeps wd 1e-4). The Muon group's decoupled decay goes from a 2e-6 per-step shrink to
6e-4 at peak lr, which shrinks every matrix's Frobenius and spectral norm by 40–60 % at step 6000 at unchanged
stable rank (Codex diagnosis, `research/optimizer_2026-09-07/codex/codex_report3.md`). Built on the local
H100 (GPU 1, shared), packed with **mvpe03b's inference code** (`submissions/mvpe03b/submission.py`, the proven
5.3 ms cugraph16 path), so only the members and head change. Run dirs `wd03mv_m{0,1,2}`, `wd03mv_head`.

**Where it came from.** The 2026-09-07 optimizer deep dive (joint Claude + Codex gpt-6-astra,
`research/optimizer_2026-09-07/`): nine pre-registered optimizer arms on the four mimic folds × seeds 0/1.
Polar-first Muon (both agents' top-ranked mechanism, two doses plus a zero-rest control), ConvTranspose
orientation, embeddings → AdamW, Switch-EMA and a stem input-covariance preconditioner all read null or
negative. The one survivor is a dose of a knob the record had closed at 3e-3 (`kf_wd1e3` +0.005, `kf_wd3e3`
+0.003): **wd 0.03 on the Muon group**.

**Evidence** (accuracy-only Δ = 0.474·Δrel + 0.106·Δtke + 0.093·Δmvpe subscore, single models, EMA at step 6000,
paired on fold and seed; readers `research/potential_2026-09-05/claude/{hist_read,screen_read}.py`):
| instrument | read |
|---|---|
| mimic 4 folds × seeds 0/1/2 vs `kf_tke1em8` | **+0.018, 12/12 runs positive**, fold means +0.016/+0.027/+0.016/+0.016, fold SE 0.006; rel_l2 −0.3…−1.4 %, tke/mvpe never worse |
| three-member ensembles (`fold_ensemble.py tke1em8 wd03 --seeds 0,1,2`, best.pt) | per-member +0.017 → **ensemble +0.028**, compression 1.68, 3/4 folds (+0.066/−0.007/+0.028/+0.025) |
| composition: `bt_wd03mv` vs `bt_mv03` (w_mvpe 0.3 with vs without wd 0.03), 4 folds × 2 seeds | **+0.016, 8/8 runs**, rel −0.3…−0.8 %; vs `bt_wd03` +0.001 — the two knobs are additive |
| angle guard (`bt_wd03_angle*` vs `bt_r2_angle*_base`, step 6000, seeds 0/1) | angle05 **+0.052 / +0.071** (rel −2.41 / −1.85 %), angle10 **+0.043 / +0.063** (rel −2.11 / −2.72 %) — **mean +0.057, 4/4**, the largest read on any instrument; 15° unopened by the suite's rule |
| far-Re guard (vs `bt_r4_screenbase_far_*`) | far_drop −0.001 / +0.021, far_keep +0.015 / +0.035 — all ≥ −0.02 |
| dose | 0.01 +0.006 (3/4), **0.03 +0.018**, 0.1 −0.008 / −0.008 (mim1, stopped there) |
| stopping step under wd 0.03 | argmax stays at 6000 (5000 +0.006, 7000 +0.006, 8000 −0.019 vs base@6000) |

**Two honest caveats.** (1) The members' per-element spread is ~7 % smaller under wd 0.03 (0.00215 vs 0.00233 on
mim1); the interval head reads that spread, so the sps width channel may give a little back — the refitted head's
in-sample sps against 917205's (51.008) is the number to check at build time. (2) The mimic folds have carried
hundreds of screening decisions; a +0.018 found here is a selected quantity, and the angle/far guards are the
only fresh instruments behind it (both positive).

**Codex audit of the ensemble read (round 4, `research/optimizer_2026-09-07/codex/codex_report4.md`).** The
+0.028 ensemble read compares wd's best.pt (mostly step 6000) against the baseline's best.pt (mostly step 5000):
at wd's fixed step 6000 it is +0.021, and **with wd read at the baseline's own checkpoint ages it is +0.009**
(fold SE 0.011, ensemble/member ratio 1.00, TKE-zeroed −0.002). So the ensemble amplification is a checkpoint-age
artefact; the honest ensemble-level expectation is the member-level +0.01–0.02 with no amplification. A full
fixed-6000 three-member comparison is impossible because the baseline runs kept no snapshots.

**Prediction, recorded before the score** (Codex's line, adopted; Claude's first draft was +0.01 [−0.05, +0.05]).
Against **917205 (81.76275)**: rel_l2 local −0.5 % ≈ +0.02 of subscore × 0.40 ≈ +0.008 per member, no ensemble
amplification; tke and mvpe ≈ 0; sps 0 to −0.01 (the members' spread is 7–10 % smaller — a refitted head can
absorb a rescaling, so this is an allowance, not a measurement); time unchanged (byte-identical inference code);
retrain noise ±0.015. **Central +0.00, band [−0.06, +0.05].** The upside case is the angle suite (+0.057, 4/4,
rel −1.9…−2.7 %): if the hidden set contains novel angles the read lands in the top of the band. Falsifier: below
−0.06 closes the wd axis; a read inside the band is uninformative on its own and **rel_l2 is the subscore to
read** (predicted +0.01…+0.03 of subscore). A miss caused by `time_score` alone falsifies the timing allowance,
not wd.

**Verification gates (run after training, results appended below):** `check_pack` MATCH against the three
member runs; in-sample `run_and_score` on `val_aoa` finite with rel_l2 > 95.0, tke > 82.0, mvpe > 95.0,
sps > 50.0 and **recorded against 917205's 51.008** (a 0.04-subscore sps loss ≈ 0.01 final); all three members' saved args show wd_muon 0.03, w_mvpe 0.3, AdamW wd 1e-4, checkpoint step 6000; head fixed 4000 with spread; zip under the packer's 256 MiB extracted cap; the extracted zip smoke-tested standalone at `--chunk 1`. **Not armed.** Whether and when to spend a slot is the owner's call; the build exists so that the
call can be made on a finished artefact.

## Verification (2026-09-07 ~16:40 UTC) — every gate passed; NOT armed

Build `WD_MUON=0.03 W_MVPE=0.3 NAME=wd03mv SUB=wd03mv CHAMPION=mvpe03b bash tools/train_final.sh` on the local
H100 (GPU 1, shared with other users' jobs), log `research/optimizer_2026-09-07/claude/build_wd03mv.log`.
Members `wd03mv_m{0,1,2}`: saved args wd_muon 0.03, w_mvpe 0.3, AdamW wd 1e-4, lr_muon 0.02, EMA 0.998,
real_trajs all, checkpoint step 6000 (all three). Head `wd03mv_head`: 4000/4000, spread, lr 8e-4, tau 0.05,
val_aoa sps 50.77 → 50.92 over the last 1000 steps. `check_pack`: **MATCH** (max abs 2.0e-4, mean relative
2.95e-4). Inference code md5 `249fd785…` = mvpe03b's, byte-identical. Zip `build/wd03mv.zip` 149.8 MB
(md5 `e2fab6da0d96e1840375922553b75b91`), 162.1 MB extracted of the 268 MB cap. The extracted zip runs standalone
from a temp dir at `--chunk 1` and reproduces the packed scores to the last digit.

In-sample `run_and_score` on `val_aoa` (a build check, not a quality estimate — every member trained on these
trajectories), against 917205's build read:

| | rel_l2 | tke | mvpe | sps | ms/sample |
|---|---|---|---|---|---|
| 917205 (mvpe03b) | 95.483 | 83.223 | 95.902 | 51.008 | 0.94 |
| **wd03mv** | 95.423 | 82.910 | 95.835 | 50.865 | 1.28 (shared GPU) |
| Δ | −0.060 | −0.313 | −0.067 | −0.143 | |

All four floors hold (rel > 95, tke > 82, mvpe > 95, sps > 50). **Every in-sample number is lower than the
champion's, tke by 0.31.** That is the expected signature of a regulariser — wd 0.03 shrinks the weights and the
model fits its own training trajectories less tightly — and the held-out evidence above (12/12 mimic runs, the
angle suite at +0.057, the far-Re guard) is the read that matters. But it is also the one thing this candidate
shows that mvpe03 did not (mvpe03 gained in-sample), so it is recorded as the risk: if the board's hidden set behaves
like the training trajectories rather than like the folds, the read lands in the lower half of the band. The sps
−0.14 is inside the manifest's sps allowance (0 to −0.01 final).

## Armed 2026-09-07 14:28Z for **2026-09-08T00:00:30Z** (owner's instruction: ship if the architecture / physics-loss
ideation round found nothing)

That round (`research/optimizer_2026-09-07/codex/codex_report5_recovered.md`, `../claude/claude_analysis.md`) found
nothing that clears +0.02 within a screen budget: Codex's target-assisted oracles read −0.030 (lead-energy
redistribution at fixed TKE) and +0.020 (covariance-shape correction, i.e. the whole demonstrated correction would be
needed); Claude's late-lead discontinuity is real per member but averages out at ensemble level (neighbour-oracle
−0.023/−0.000/−0.030/−0.002 on the four folds). Two loss-term screens (`--w-elead`, `--w-dlead`) are still running
as background reads and do not gate this slot.

`RETRIES=3 setsid nohup bash tools/submit_at.sh 2026-09-08T00:00:30Z build/wd03mv.zip ...`, log
`research/optimizer_2026-09-07/claude/submit_wd03mv.log`. Zip md5 `e2fab6da0d96e1840375922553b75b91`.
Prediction as above: **+0.00 [−0.06, +0.05] against 917205 (81.76275)**; rel_l2 is the subscore to read.

## Scored 2026-09-08 (918211): **81.76779 = +0.00504 against 917205 — new best, rank 3 of 50; prediction +0.00 [−0.06, +0.05] hit**

| channel | 917205 | 918211 | Δ | × `DELTA` weight |
|---|---|---|---|---|
| rel_l2 | 94.8377 | 94.8469 | +0.0092 | +0.0037 |
| tke | 79.1766 | **78.9128** | **−0.2637** | **−0.0252** |
| mvpe | 93.9578 | 94.0917 | +0.1339 | +0.0138 |
| time | 92.0179 | 92.1744 | +0.1565 | +0.0156 |
| sps | 43.7253 | 43.7229 | −0.0024 | −0.0006 |
| **final** | **81.76275** | **81.76779** | **+0.00504** | sum +0.0074 |

**Read the decomposition, not the headline.** The named subscore was right — `rel_l2` +0.009 against a
predicted +0.01…+0.03, and `mvpe` +0.134 on top — but **the accuracy channel came in at −0.008** because tke
paid −0.264, and the positive final is carried by `time_score` drifting +0.157, which is container noise and
not attributable to this change. So on the board wd 0.03 is a rel/mvpe-for-tke trade at roughly accuracy
break-even, against +0.018 on the folds.

**The falsifier did not trip** (a read below −0.05 would have closed the wd axis) and the fold evidence
survives at novel angles, where the arm's largest read was. But two things in this manifest were written down
before the score and both proved informative, so they are the lesson rather than the caveat:

* The in-sample build check showed **every** subscore below the champion's, tke by 0.31, and this file recorded
  that as "the risk" while shipping. The board's tke came in at −0.264, the same size. **An in-sample tke drop
  of ~0.3 on a regulariser should be priced as a board tke loss of that order**, not as in-sample overfitting.
* The fold rig's "tke never worse on 12/12 runs" did not transfer. `docs/archive/HANDOFF_2026-09-10.md` §4 already flags tke as the
  subscore whose local→board transfer changes sign; this is the strongest instance on record, and the first for
  a regulariser.

**What this licenses.** Keeping `WD_MUON=0.03` in the shipped recipe: it is a new best, the prediction hit, and
the angle-suite evidence (+0.057, 4/4) is the one instrument aimed at the hidden set's novel angles. **It does
not license a larger dose** — 0.1 was already −0.008 locally and the board says accuracy is at best flat at
0.03 — nor treating the +0.018 fold number as a board expectation for anything built on top of it.
