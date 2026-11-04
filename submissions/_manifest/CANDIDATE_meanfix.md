# Candidate meanfix — September 9, 2026

**Ready and unsubmitted. Selected for the next authorized evaluation; no upload is scheduled.**
Incumbent remains918211 at81.7677894393. September9’s slot was already used by920181.

Archive: `build/meanfix.zip` (149,799,765 bytes; extracted 162,096,165).

- ZIP SHA256: `6c74ac97c43ea348d299cecf8e7b0a630bdeb5d4f3c0d682cade38800b4c96fe`
- Model SHA256: `5994f493d031d3023939d5f0932a0aa03d752900dd08629c1a046024deb06b47`
- Inference SHA256: `0d47721eae007eacefbf4302eb904279e1dc1c218b810c226ea9b820542ae37c` — byte-identical to wd03mv.

## Recipe and rationale

Start from the exact incumbent ordinary third member. Freeze its whole backbone
and original output convolution; learn34600 parameters for an output correction
with zero temporal mean. Fit1000 cosine steps in FP32 with AdamW lr1e-4, no decay
or EMA, phase seed2909, current composite+bulk loss, 75% ensemble /25% own loss.
The same two incumbent companions remain frozen. Use all81 real and100 simulated
trajectories. The correction folds into the existing convolution, preserving the
three-member inference architecture. FP32 time-mean velocity/MVPE is invariant;
relative-L2 and finite-precision outputs are not guaranteed unchanged.

Fit the ordinary width96 spread head at fixed4000, lr.0008, tau.05, seed0, lambda1,
using FP32 member predictions and all real trajectories. The adaptation ran on
TitanXP; the full head ran on H200. Full-data diagnostics did not select the recipe.
The adapter checkpoint’s explicit step1000 and completed history define its age;
its legacy inherited fixed_step field describes the pretrained member.

## Held-out evidence versus ordinary incumbent recipe

| Actual standalone FP16 comparison | meanfix | balanced alternative |
|---|---:|---:|
| Four mimic folds, complete local mean | +.031550 | +.032804 |
| Minimum mimic-fold gain | +.017687 | +.005852 |
| Range-edge holdout | +.033073 | +.016138 |
| Shifted angle/Re | +.000288 (flat) | -.020169 |

Meanfix four-fold component deltas: rel-.015724, TKE+.204288,
MVPE+.000024, learned SPS+.068819. Complete objective excluding TKE:+.009956.
Member seed3 forecast replication and head seed1 (+.026118/+.040871 SPS) support
the mechanism. All fold ages, seeds and recipes were fixed before confirmation.
These reused folds and locally weighted scores are not a board performance claim.
Meanfix is preferred for consistency and protection of the probe statistic;
balanced loss remains a viable research alternative with better field accuracy.

## Verification and limits

Actual FP16 prediction/bound relative mean errors versus FP32 are
.02864% / .04804% / .04284%. The exact ZIP imports in isolated Python with PyTorch2.2.2+cu121
on an A100 without repository imports. Batches1,2,3,4,6,7,8,16,17,32,64,65,1
pass shape/finiteness/interval-order and exact repeated-call checks. Warmup and
these batches take8.45s; another187 one-window calls take.736s. This is a deployment
smoke test, not a forecast of leaderboard timing.

The initial absolute mean-change bound2e-5 failed (4.27e-5 on16 windows). A separate
187-window diagnosis verifies FP32 mean change at most1.49e-7 and essentially the
same FP16 rounding error as the incumbent. Packed MVPE changes+.000064665 points.
The unchanged weights passed verification using that hash-pinned diagnosis and a
.001-point MVPE drift limit; the failed original bound remains in the receipt.

[Complete report](../../research/repair_2026-09-09/REPORT.md),
decision,
artifact receipt,
numerical diagnosis,
reproduction instructions.
