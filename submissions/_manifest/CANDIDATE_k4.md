# Candidate k4 — fourth ordinary member + refitted head (pre-registered 2026-09-11, joint Claude + Codex round 5; NOT uploaded)

**What.** The exact incumbent members (wd03mv seeds 0/1/2) plus one ordinary seed-3 member trained with the identical
recipe on the same card class (H100, repaired `train_final.sh` argv, checkpoint 6000), and a width-96 spread head
refitted at lr 8e-4 / τ 0.05 / step 4000 on the four members — head seed 0 exported, seed 11 fitted as a nuisance
read, never selected between. Same inference source as the incumbent (wd03mv's). Extracted size ≈ 207 MB < 256 MB.

**Why now.** The archive closed 3→4 members at net −0.005 = +0.046 interval (a converted bound at an assumed 60 %
capture; the Muon-era realisation was 42 %) − 0.009 direct − 0.042 of *time* at 0.847 ms/member. P0/P1 (2026-09-10/11)
show kernel work nearly invisible to the container clock, so the time term is ≈ 0 ± 0.02 and the net is
+0.037 / +0.023 / −0.002 at capture 60 / 42 / 9 %. No capture fraction is measured for an ordinary fourth seed; the
board is the only instrument that prices the learned head.

**Prediction versus 918211** (Codex round 5 text): non-time Δ +0.02 [−0.02, +0.04]; total Δ +0.01 [−0.03, +0.04].
Interval improvement should outweigh any direct-accuracy loss. **Kill** if non-time Δ ≤ 0 under either weight fit,
or measured timing consumes the gain. A time-only win does not validate member count. This single board read does
not establish private-retrain improvement.

**Before any slot.** Sentinels: mim1 / far_drop direct ΔQ (4 vs 3 members, seeds 0/1/2/3) ≥ −0.02 each; extracted
size < 256,000,000 bytes; `check_recipe --pack` with `MEMBERS="0 1 2 3"`; `check_pack` (prediction + intervals) MATCH;
fresh-process setup; paired A100 N = 1 latency vs the incumbent (a +0.5 ms slowdown reprices, kill if it consumes the
projected gain). Slot per the joint table: 09-17, after the calibration slots; the owner decides.

## Build and local read (2026-09-11 evening)
Seed-3 member: `cleanclone/k4/wd03mv_m3` (incumbent argv, H100 GPU 3, checkpoint 6000). Heads on wd03mv m0/m1/m2 + m3
(`cleanclone/k4/head4_s0`, `head4_s11`); K=3 same-seed refit `head3_s0` = the incumbent's head bit for bit (same-class
determinism; its pack `k3ctl` scores identically to wd03mv). Packs: `submissions/k4` (head seed 0, exported),
`submissions/k4h11` (nuisance read). `model.pth` 207,325,674 bytes; `check_pack` prediction + intervals MATCH.
In-sample stride-40 val_aoa, same windows, GPU 3:
| | rel | tke | mvpe | sps | Δ non-time (locked weights) |
|---|---|---|---|---|---|
| wd03mv = k3ctl | 95.490 | 83.166 | 95.877 | 50.932 | 0 |
| k4 (head s0) | 95.495 | 83.171 | 95.882 | 51.020 | **+0.027** |
| k4h11 (head s11) | 95.495 | 83.171 | 95.882 | 51.005 | +0.023 |
Direct accuracy +0.005 on every channel (the fourth member's averaging), sps +0.088 / +0.073 — inside the pre-registered
non-time band (+0.02 [−0.02, +0.04]); in-sample, so it prices the head's capture on training windows only.
Sentinels (held-out ensemble means, step 6000, B s0/s1 + K4B s2 [+ s3]): mim1 −0.007, far_drop −0.013 — pass (≥ −0.02).
Paired A100 N=1 latency vs the incumbent: pq job `k4time` → `research/audit_2026-09-11/claude/readout/k4_timing_a100*.json`.
Zip `build/k4.zip`; standalone extraction check below. Slot: owner's decision (joint table: 09-17).

## Paired latency (pq A100-PCIE-40GB, median of 300 warm calls, same card, `readout/k4_timing_a100*.json`)
| N | wd03mv | k4 | Δ local | priced at slope 0.36 (P0/P1) | priced at slope 1 |
|---|---|---|---|---|---|
| 1 | 3.74 ms | 4.03 ms | +0.29 ms | ≈ −0.007 final | ≈ −0.02 final |
| 8 | 9.13 ms | 10.96 ms | +1.83 ms | ≈ −0.04 final | ≈ −0.12 final |
The container is believed to call predict() on 1–3 windows (FINDINGS §5); at that size the time cost does not consume the
projected +0.02 non-time gain under either slope. **Kill rule stands as pre-registered**: non-time board Δ ≤ 0 under
either weight fit, or the observed time loss ≥ the non-time gain. Prediction unchanged: non-time +0.02 [−0.02, +0.04],
total +0.01 [−0.03, +0.04]. Ready for a slot on the owner's go (`build/k4.zip`, 191.6 MB / 207.4 MB extracted).

## Armed (owner's go, 2026-09-11 21:00 UTC)
`tools/submit_at.sh 2026-09-12T00:00:30Z build/k4.zip` — log `research/audit_2026-09-11/claude/submit_k4.log`.
Zip sha256 c74edaef7872c978…, 191,639,453 bytes; inference source byte-identical to wd03mv's; model.pth 207,325,674 bytes.
Prediction (unchanged, recorded before the score): non-time Δ +0.02 [−0.02, +0.04], total Δ +0.01 [−0.03, +0.04] vs
918211 (81.76779; rel 94.847 / tke 78.913 / mvpe 94.092 / sps 43.723 / time 92.174). Kill: non-time Δ ≤ 0 under either
weight fit, or the time loss ≥ the non-time gain. On a pass: `MEMBERS="0 1 2 3"` becomes the frozen default, pins move.
Revised slot order after this: 09-13 exact 918211 repeat · 09-14 P0 repeat · 09-15 meanfix · 09-16 `final` ·
09-17 cc11 · 09-18 cc11h · 09-19 K4 repeat or K5 (252.6 MB) if positive, else incumbent.

## Board result — submission 923184, scored 2026-09-12 00:05 UTC: **final 81.80564 (+0.038 vs 918211) — new best; PASS**
| | rel | tke | mvpe | sps | time | final |
|---|---|---|---|---|---|---|
| 918211 | 94.847 | 78.913 | 94.092 | 43.723 | 92.174 | 81.768 |
| **923184 (k4)** | 94.853 | 78.977 | 94.101 | 43.841 | 92.164 | **81.806** |
| Δ | +0.006 | +0.064 | +0.009 | **+0.118** | −0.010 | +0.038 |
Non-time Δ **+0.040** under the locked weights and **+0.040** under DELTA (prediction +0.02 [−0.02, +0.04]: top edge);
time −0.001 of final (the +0.29 ms local cost was priced at −0.007; the clock barely saw it). Kill rule not triggered
under either fit. Every channel improved; the sps gain (+0.118 board vs +0.088 in-sample) says the head's capture of
the fourth member's spread transferred at ≥ 1×. Board after scoring: #1 81.824, #2 81.819, #3 81.807 (np-user), **#4 us
81.806** (0.002 behind #3, 0.018 behind #1). Pass rule applied: `MEMBERS="0 1 2 3"` is the frozen default, pins move to
923184 (`tools/train_final.sh`, `check_recipe`, `verify_best` after the commit), artefact archived read-only on both
copies. Next on this axis: K5 (seed 4, 252.6 MB extracted) — predicted ≈ half of K4's interval gain.
