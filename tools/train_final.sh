#!/bin/bash
# The whole submission, from the official release to a packed zip, in one script.
#
# This is the artefact the Decision Phase needs: the top 10 are re-trained from
# scratch by the organizers before the prizes are decided, so the pipeline has to
# run end to end for someone who has never seen the repo. It reads only
# `training_data/train_real` and `training_data/train_sim` from the release.
#
#   bash tools/train_final.sh                    # full run, ~1 h on one H100
#   MEMBERS="0 1 2" GPU=3 bash tools/train_final.sh
#
# Every choice below was measured on held-out regimes; the reasoning is in
# docs/archive/HANDOFF_2026-09-10.md and the numbers that justify each one are in its sections 6 and 7.
set -euo pipefail
cd "$(dirname "$0")/.."

# GPU: a plain index or UUID for CUDA_VISIBLE_DEVICES. Under a scheduler that has
# already narrowed CUDA_VISIBLE_DEVICES (pq passes a UUID), the default used to be
# the literal 0, which re-widened the mask and trained on a card the scheduler had
# not granted. Inherit the mask when it is set; fall back to 0 only when it is not.
# `${VAR-0}` (no colon) keeps an explicitly EMPTY mask empty: a scheduler that granted
# no GPU must fail the CUDA preflight below, not silently train on card 0.
GPU=${GPU-${CUDA_VISIBLE_DEVICES-0}}
# Seeds. Four since 2026-09-12: the fourth ordinary member paid +0.038 on the board
# (923184 = 81.80564; sps +0.118, every accuracy channel up) once P0/P1 showed the
# container clock barely charges kernel work -- the old "three, net -0.005" closure was
# a time price. The head reads --n-models from this list (below).
MEMBERS=${MEMBERS:-"0 1 2 3"}
if [ "$(echo $MEMBERS | tr ' ' '\n' | sort | uniq -d | wc -l)" != "0" ]; then
  echo "MEMBERS='$MEMBERS' repeats a seed; identical members are not an ensemble" >&2; exit 1
fi
# Run directories (members, head). REALISTIX_RUNS is the portable knob; the literal
# is this host's data disk. Made absolute below, because the head's member symlinks
# are written from it and a relative target breaks when the head runs elsewhere.
R=${R:-${REALISTIX_RUNS:-$([ -d runs ] \
        && echo runs || echo "$PWD/runs")}}
mkdir -p "$R"; R=$(cd "$R" && pwd -P)
# Pack staging and the parity check's inputs. On this host both default to the data
# disk; elsewhere they land next to the runs, and the parity inputs are cut from the
# release itself (tools/build_eval_npz.py) so no historical cache is needed.
export REALISTIX_BUILD=${REALISTIX_BUILD:-$([ -d build ] \
        && echo build || echo "$R/../build")}
CHECK_PACK_EVAL=${CHECK_PACK_EVAL:-$([ -f eval/val_aoa/samples.npz ] \
        && echo eval/val_aoa || echo "$R/../eval/val_aoa")}
NAME=${NAME:-final}                  # prefix for the run directories under $R
# The submission directory whose submission.py is the shipped inference code (copied into the fresh build, with the
# fused-kernel cubins regenerated from the CUDA source embedded in it). Since 2026-10-01 `dec12kfinal5b`, the inference
# file of Codabench submission 947208, which the Decision-Phase package reproduces: four members,
# the dose-0.5 member-variance-matched fluctuation gain, the distilled student head, CUDA graphs captured at import, and
# fused GroupNorm/bias/glue CUDA kernels inside the graphs (943854 carried the GroupNorm/bias kernels; dec12kfinal4
# adds the post graph's head-input and output-assembly kernels, CANDIDATE_dec12kfinal4.md; dec12kfinal5 copies the
# prediction to the host while the interval head runs, CANDIDATE_dec12kfinal5.md; dec12kfinal5b adds cuDNN
# benchmark-mode algorithm selection at setup, CANDIDATE_dec12kfinalb.md).
# The rebuild writes its model.pth somewhere else, so the champion pack is never at risk. Defaults below ARE the
# shipped recipe; W_MVPE / WD_MUON still override for experiments.
CHAMPION=${CHAMPION:-dec12kfinal5b}
# Where this rebuild lands. It used to default to `all82_spread`, which is the
# 887158-era artefact and already has a model.pth -- so the documented one-command
# rebuild hit the overwrite guard below and exited 1 on any machine that had ever
# built anything. A fresh directory is what "reproduce the submission" means.
SUB=${SUB:-rebuild}
# Interpreter: this host's environment when it exists, otherwise whatever `python`
# is on PATH (the evaluation image has torch 2.2.2 there). Fail loudly if neither
# imports torch rather than an hour later in the first trainer.
PY=${PY:-python}
[ -x "$PY" ] || PY=$(command -v python || command -v python3 || true)
"$PY" -c "import torch, numpy, h5py" 2>/dev/null \
  || { echo "PY=$PY cannot import torch/numpy/h5py; set PY to a suitable interpreter" >&2; exit 1; }
# The recipe is bf16 on a CUDA card; without one the trainer would pick another
# precision and build a different model. Checked under the mask the build will use.
if [ "${PRINT_RECIPE:-0}" != "1" ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -c "import torch, sys; ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported(); \
print('gpu:     ' + (torch.cuda.get_device_name(0) if ok else 'NONE')); sys.exit(0 if ok else 1)" \
    || { echo "no CUDA device with bf16 under CUDA_VISIBLE_DEVICES='$GPU'; set GPU" >&2; exit 1; }
fi

# The research holdout variables are read once, at import of realistix.splits, and
# silently remove trajectories from training. A shell that has run a fold screen
# still carries them. The final build trains on the whole release, so they are
# cleared here rather than inherited (REALPDE_DATA and the knobs above are kept).
unset REALISTIX_HELDOUT_AOA REALISTIX_HELDOUT_RE REALISTIX_HELDOUT_CELLS \
      REALISTIX_HELDOUT_TAIL REALISTIX_SIM_KEEPS_CELLS REALISTIX_SIM_DIR \
      REALISTIX_ALLOW_MIXED_HOLDOUT REALPDE_DATA_OVERRIDE
export PYTHONDONTWRITEBYTECODE=1
# A git-only export must carry the data-identity check (2026-09-24 audit: both files were
# untracked, so a clean checkout failed here after the preflight).
for required in tools/data_manifest.py tools/release_manifest.json; do
  test -f "$required" || { echo "incomplete training release: $required missing" >&2; exit 1; }
done

# Fail before an hour of GPU time rather than after it. A fresh rebuild directory
# gets the champion's submission.py copied in; anything else has to already have one.
if [ ! -f "submissions/$SUB/submission.py" ] && [ "${PRINT_RECIPE:-0}" != "1" ]; then
  if [ -f "submissions/$CHAMPION/submission.py" ]; then
    mkdir -p "submissions/$SUB"
    cp "submissions/$CHAMPION/submission.py" "submissions/$SUB/submission.py"
    echo "copied submissions/$CHAMPION/submission.py -> submissions/$SUB/"
    # Fused-kernel inference sources carry their CUDA source; its cubins are build artefacts of that source,
    # regenerated here with torch's NVRTC (a no-op for a source without kernels). check_recipe verifies them.
    "$PY" tools/build_cubins.py "submissions/$SUB" || exit 1
  else
    echo "submissions/$SUB/submission.py not found, and no champion copy at " \
         "submissions/$CHAMPION/submission.py" >&2; exit 1
  fi
fi

# Refuse to overwrite a build that is already there. This script writes
# submissions/$SUB/model.pth, and the hazard is that a routine retraining run
# silently replaces an artefact holding a shortlist place, an hour after you stopped
# watching. Now it stops before spending the GPU time, and says both ways out.
# (SUB now defaults to a fresh `rebuild` directory rather than to the champion, so
# the documented one-command path no longer trips this guard on a used checkout.)
#
# The archived copies under archive/887158 are read-only and would survive this
# anyway; the guard exists so the working copy does too.
if [ -f "submissions/$SUB/model.pth" ] && [ "${FORCE:-0}" != "1" ] && [ "${PRINT_RECIPE:-0}" != "1" ]; then
  echo "submissions/$SUB/model.pth already exists ($(du -h "submissions/$SUB/model.pth" | cut -f1))." >&2
  echo "Refusing to overwrite it. Either build somewhere else:" >&2
  echo "    NAME=myrun SUB=myrun bash tools/train_final.sh   # after copying submission.py there" >&2
  echo "or say you meant it:" >&2
  echo "    FORCE=1 bash tools/train_final.sh" >&2
  exit 1
fi
DATA=${REALPDE_DATA:-data/RealPDE-Competition-Data/training_data}
for d in train_real train_sim; do
  test -d "$DATA/$d" || { echo "REALPDE_DATA/$d not found (REALPDE_DATA=$DATA)" >&2; exit 1; }
done
export REALPDE_DATA="$DATA"

# Report what will be *trained on*, not what is on disk. train_real ships 82 files
# but 7575_0 was withdrawn by the organisers on 2026-08-12 (it reuses 6300_0's raw
# measurements) and `splits.EXCLUDED` drops it inside `available()`, so the count
# that matters is 81. This printed the file count until 2026-08-21, and every
# document in the repository then said "all 82 trajectories".
USED=$(PYTHONPATH=src "$PY" -c "from realistix.splits import available; \
print('%d real, %d sim' % (len(available('real')), len(available('sim'))))")
echo "data:    $DATA  ($USED used; $(ls "$DATA/train_real"/*.h5 | wc -l) real files present)"
# The release is 82 real files (81 after the withdrawn duplicate) and 100 simulated.
# Anything else means a different data directory, a stale holdout or an incomplete
# download, and the model it would build is not the one the record describes.
if [ "$USED" != "81 real, 100 sim" ]; then
  echo "expected '81 real, 100 sim' from the release, got '$USED' -- refusing to build" >&2; exit 1
fi
# Identity, not a count: ordered names, sizes and digests of the release files as the
# trainers enumerate them (tools/release_manifest.json). check_recipe runs the same
# check itself, so PRINT_RECIPE skips the ~30 s of hashing here.
if [ "${PRINT_RECIPE:-0}" != "1" ]; then
  PYTHONPATH=src "$PY" tools/data_manifest.py --check tools/release_manifest.json || exit 1
fi
echo "runs:    $R/${NAME}_*"
echo "output:  submissions/$SUB/model.pth"

# Forecaster. Component attributions live in docs/MODEL.md, re-measured 2026-08-31
# under THIS recipe (FINDINGS 8a: attributions do not transfer between recipes any
# better than constants do). The current numbers: simulated windows +0.053 fold
# (~0.10 of final; the old +0.71 is superseded), Muon +0.070, EMA +0.047, width 96
# vs 64 +0.015, and the per-modality conditioning (--cond norm) measures INSIDE
# NOISE for removal -- its old +0.17 was absorbed by the loss re-derivation and
# w_sps/w_tke. It stays shipped because removal is not a gain, it folds away at
# export, and the verified build carries it.
#
# The stopping step is transferred, not selected: with --real-trajs all there is
# nothing left to validate on, so it has to come from a run that could. Which run
# matters -- at 58 trajectories the held-out optimum is 4000, but with AoA=20 in the
# training set it moves to ~6000 and flattens (78.11 / 78.12 / 78.09 at 5000 / 6000 /
# 7000). Under AdamW that made --fixed-step 6000 against an 8000-step cosine, because
# a cosine annealed to zero at the stopping point is *worse* than truncating the long
# one (76.78 against 76.94). Under Muon the schedule is taken at its end; see the
# --opt muon note below. Every snapshot is written out either way, so the choice can
# be revisited without re-training.
# The loss is re-derived from the scoring function rather than inherited. Fourteen
# submissions identify the weights as rel_l2 0.474, sps 0.253, tke 0.106, time 0.095,
# mvpe 0.093 (leave-one-out RMS 0.012; realistix.weights holds the fit), and the loss
# had carried plain MSE at 1.0 against the per-window *relative* L2 at 0.5 -- while
# the scorer never computes MSE at all. Sweeping the ratio: rel 1/2/4/8/16 gives
# +0.21/+0.32/+0.46/+0.61/+0.62 on val_aoa, peaking at 8. Dropping MSE entirely is
# much worse (+0.11), so it is doing real work where the relative L2's per-window
# denominator is small.
#
# --w-mvpe 0.0 is the one line here resting on a number that has since moved. MVPE
# was dropped on 2026-08-12 because the eight-point fit put its weight at 0.032; the
# fourteen-point refit puts it at 0.093, because 894861 was the first submission to
# move MVPE hard enough to identify the coefficient. The separate *sweep* of the
# term was negative on its own terms ("monotonically worse"), which is why this
# still says 0.0 -- but that sweep ran under the pre-correction objective and on the
# angle folds, so it is a re-test worth doing rather than a settled result.
#
# WIDTH/MULTS/BLOCKS are overridable but the defaults are the shipped ones, and stay that
# way: `--mults 1,2` (deepest level dropped) looked like a clear win on seven regime
# folds (+0.062, 7/7 positive) and lost 0.013 on the board as submission 891698,
# because the folds train on ~30% less data than this script does and a capacity
# reduction flatters itself there. Do not promote a width/depth/level change on
# k-fold evidence alone; see docs/archive/HANDOFF_2026-09-10.md §7.
#
# ANCHOR is overridable and defaults to the shipped `last`. `blend` learns a
# per-output-frame mix of the last observed frame and the input window's time-mean
# -- 20 scalars, no capacity change. It is **refuted**: screened at +0.028 on 3
# folds with 5/5 runs positive, then confirmed on 2026-08-18 at -0.004 +/- 0.011
# over 7 folds x 2 seeds, 3/7 positive. The split is the finding -- +0.014 on the
# folds it was screened against and -0.017 on the four it had never seen, i.e. the
# whole effect was selection. The flag stays because the measurement is worth more
# than the deletion; do not turn it on. See docs/archive/HANDOFF_2026-09-10.md section 0b.
#
# --opt muon is the 2026-08-23 optimizer find, shipped in 898763. Matrix-shaped
# parameters go on orthogonalized momentum (realistix.muon) and the rest stays on
# AdamW at --lr; dosing converged at lr 0.02 / momentum 0.95, and it is worth
# +0.038 of accuracy per member on the mimic folds and +0.064 at ensemble level --
# the largest ensemble-level local gain on record. Read the caveat in docs/archive/HANDOFF_2026-09-10.md
# before promoting anything else on that evidence: the board paid -0.016 of
# accuracy for it and +0.074 of final through sps and time instead.
#
# --fixed-step is the constant that has moved most, and each move tracked the
# training objective: AdamW's held-out optimum sat mid-schedule (6000), plain Muon's
# at the cosine's end (8000), under Muon + w_sps the argmax was 7000 on 8/8 held-out
# fold runs (shipped as 902375, board +0.070), and with the w_tke restoration it is
# back to 6000 (5000/6000 tie on the folds, tie-broken later; shipped as 904950).
# w_sps's amplitude damping accumulates late in the schedule, which is why the stop
# and any bulk-term dose have to be swept together rather than separately. After ANY
# recipe change, re-read the argmax off the fold histories before trusting this
# number. The loss weights (w_rel 6 is +0.006, 12 is -0.020) and batch 32 (64 is
# -0.164) carry over; every one of those was re-checked under the new optimizer.
#
# --w-sps 1.0 is the SPS-bulk loss term (losses.sps_bulk), shipped in 901812. The
# loss was re-derived from the scoring function for four subscores in 887158;
# sps_score was the one left out because it looked like the interval head's
# business. It is not: SPS factors into an accuracy prefactor times a width factor,
# and the CEILING on the width factor -- mean exp(-|e|/sigma) -- belongs to the
# forecaster. Dose 1.0 sits just under the measured damping gate (tools/
# error_anatomy.py: at doses >= 2 the term pays in fluctuation amplitude at k < 8,
# the same band the gain lives in; at 1.0 the amplitude is intact and the gain is
# phase and mean). Screened at dR_bin +0.0059 (4/4 mimic folds, sd 0.0012) on the
# interval channel -- the one instrument with a stable board transfer -- and the
# board paid +0.057 of final against a recorded prediction of +0.05.
# --w-tke 1.0 and --ema 0.998 are the 2026-08-28 finds (shipped together, +0.080
# board): w_sps shifted the operating point off the five-mechanism tke/rel
# frontier, and doubling the tke weight buys the amplitude back at half the rel
# cost ConFIG paid. Both were re-selections of constants last set under a recipe
# without w_sps -- the recipe-alignment discipline, and it overturned two earlier
# readings (EMA 0.998 measured -0.007 under plain Muon, w_tke 2.0 cost 0.28 under
# the pre-w_sps loss). The dose curve peaks at w_tke ~1.15 and that peak is worth
# +0.001, well below what the fold rig resolves: do not re-tune it.
TRAIN="--mode plus_sim --film --cond norm --width ${WIDTH:-96} --ema 0.998 --w-tmean 0.5 \
       --w-rel 8.0 --w-mvpe 0.3 --w-sps 1.0 --w-tke 1.0 --mults ${MULTS:-1,2,4} --blocks ${BLOCKS:-2} \
       --anchor ${ANCHOR:-last} \
       --opt muon --lr-muon 0.02 --muon-momentum 0.95 --wd-muon 0.03 \
       --steps 8000 --eval-every 1000 --real-trajs all --fixed-step 6000"

# Interval head. Trained on the SPS reward itself rather than on an error-prediction
# proxy, reading the ensemble's own per-element disagreement.
#
# Every constant here was re-selected on the mimic folds in 2026-08-22/23 and every
# one of them moved. The old "peaks after ~1000 steps and decays" was measured on
# val_aoa -- the harsh AoA=20 split -- before the hidden set's shape was known; on
# hidden-shaped folds held-out sps keeps rising past 1000 and a 4000-step schedule
# plateaus at 3000-4000. That alone was +0.30 local sps and paid +0.414 on the board
# (896716). The follow-up 16-configuration sweep then found the training lr rising
# monotonically from 1e-4 to 8e-4 (+0.312, collapsing at 1.6e-3) and the surrogate
# sharpness tau_frac peaking at 0.05 against the shipped 0.15 (+0.101); the joint
# confirmed at +0.374 over the 4000/4000 reference at three head seeds (sd 0.020)
# and paid +0.320 on the board (897732).
#
# Width stays 96. The width-192 joint reads +0.396 locally but puts 4x the head
# FLOPs into the timed region, and adding work has lost on the board every time.
HEAD="--n-models 3 --spread --width 96 --steps 4000 --eval-every 1000 \
      --lr 8e-4 --tau-frac 0.05 --train-split all --fixed-step 4000 --seed ${HEAD_SEED:-0}"

# Head METHOD (2026-09-17). The interval estimator that ships is not one fitted head but
# the average of several, compressed back into one:
#   `student` (default; the method validated on the board as 929146): TEACHER_SEEDS independent
#   heads are fitted with the HEAD recipe above (each `--seed` overriding HEAD_SEED; argparse takes
#   the last occurrence), their per-element bounds are averaged in linear units, and ONE head of
#   the same architecture is regressed onto that average (tools/distil_head.py, STUDENT literal:
#   teacher-0 initialisation, under-width penalised 3x, 4000 steps, seed 7). Teachers are training
#   artefacts; inference runs the student. Why: averaging two heads' OUTPUTS paid +0.0807 SPS on the
#   board (926634) at +1 ms; the student keeps the gain (+0.0846 board SPS over the single head,
#   929146) at one head's cost. FINDINGS §2b, CANDIDATE_k4a2.md, research/distil_2026-09-16/PREREG.md.
#   `single`: the seed-HEAD_SEED head alone (923184's method) -- the fallback if a fresh-geometry
#   read ever goes against the student.
HEAD_MODE=${HEAD_MODE:-student}
TEACHER_SEEDS=${TEACHER_SEEDS:-"0 11 21 31 41 51"}
# 2026-09-23: --steps 4000 -> 12000 (the registered R4 replacement; the cosine horizon is
# `CosineAnnealingLR(opt, a.steps)` in distil_head.py, so it follows the literal). Fixed on
# 09-18 in codex/PREREG_NEXT.md with its kill conditions, and none fired: on the fresh 15-cell
# joint geometry 12k - 4k = +0.0336 SPS with 15/15 cells positive (floor +0.010, no cell below
# -0.030), and on the board 940198 read +0.0262 SPS with rel and MVPE bit-identical to 934147
# and TKE moving 3.7e-6 -- accuracy-only N = +0.00637 against a registered +0.004 [-0.002,
# +0.008]. Parity, size and paired latency (+0.02 ms) were green before the slot. Nothing else
# moves: same six teachers, same seed 7, same teacher-0 init, same under-width 3, same width,
# same architecture, same pack size, same inference code. Retrain cost ~ +15-45 min.
# research/push_2026-09-22/R4_RESULT.md, submissions/_manifest/CANDIDATE_head12k.md.
STUDENT="--steps 12000 --batch 32 --lr 8e-4 --width 96 --seed 7 --train-split all \
         --init-teacher 0 --under-weight 3.0"
case "$HEAD_MODE" in student|single) ;; *) echo "HEAD_MODE must be student or single, got '$HEAD_MODE'" >&2; exit 1;; esac
if [ "$(echo $TEACHER_SEEDS | tr ' ' '\n' | sort | uniq -d | wc -l)" != "0" ]; then
  echo "TEACHER_SEEDS='$TEACHER_SEEDS' repeats a seed; identical teachers are not an ensemble" >&2; exit 1
fi
# Audit/test overrides, NOT part of the recipe: MEMBER_RUNS lists existing member checkpoint
# directories (one per seed in MEMBERS, same order) to link instead of training; TEACHER_RUNS lists
# existing teacher directories (one per TEACHER_SEEDS, same order) to reuse instead of fitting. They
# let the head/pack/check stages run in minutes against known checkpoints (the student replay is the
# positive control of the whole contract). A Decision-Phase rebuild sets neither.
MEMBER_RUNS=${MEMBER_RUNS:-}
TEACHER_RUNS=${TEACHER_RUNS:-}

# --- non-default builds -------------------------------------------------------
# Append-only: check_recipe proves the two literals above against the shipped
# artefact and never parses these. LEAN=1 is the 2026-09-01 lean loss (mse/tmean/
# spec to zero: individually dead, jointly +0.010 on 4/4 folds). A non-default
# member count re-reads --n-models off MEMBERS (argparse takes the last
# occurrence), and a single member adds --no-spread: the spread over one member
# is undefined, and it measured ~0 even with three (2026-08-31).
EXTRA_TRAIN=""
[ "${LEAN:-0}" = "1" ] && EXTRA_TRAIN="--w-mse 0.0 --w-tmean 0.0 --w-spec 0.0"
# W_MVPE=<w> re-adds the MVPE loss term (argparse takes the last occurrence of --w-mvpe).
# The 2026-09-01 fold read at 0.3: d_mvpe +0.030 +- 0.006 (8/8), rel/tke flat, priced +0.00-0.02.
[ -n "${W_MVPE:-}" ] && EXTRA_TRAIN="$EXTRA_TRAIN --w-mvpe $W_MVPE"
# WD_MUON=<wd> sets the Muon group's weight decay (the AdamW group keeps --wd 1e-4). The 2026-09-07
# optimizer deep dive (docs/log/2026-09-optimizer.md): 0.03 reads +0.018 per member on 12/12 mimic
# runs and +0.057 on the angle suite (4/4, rel_l2 -1.9..-2.7%); 0.01 is half that and 0.1
# over-regularises. The shipped 1e-4 shrinks the weights by only 2e-6 per step.
#
# Shipped as 918211 (2026-09-08) on top of W_MVPE=0.3, and read the result before turning it up:
# **+0.005 of final, but the accuracy channel was -0.008 of it.** rel_l2 +0.009 and mvpe +0.134 came
# in as predicted; tke paid **-0.264**, and the headline is positive only because time_score drifted
# +0.157 (container noise, not ours). The in-sample build check had already shown the signature --
# every subscore below the champion's, tke by 0.31 -- and the fold rig's "tke never worse" did not
# transfer. So this knob is a rel/mvpe-for-tke trade at the board, roughly break-even on accuracy,
# and a larger dose is more likely to cost than to pay. The fold ensemble read that looked like
# amplification (+0.028) was a checkpoint-age artefact: age-matched it is +0.009 with no
# amplification at all.
[ -n "${WD_MUON:-}" ] && EXTRA_TRAIN="$EXTRA_TRAIN --wd-muon $WD_MUON"
NMEM=$(echo $MEMBERS | wc -w)
EXTRA_HEAD=""
if [ "$NMEM" != "3" ]; then
  EXTRA_HEAD="--n-models $NMEM"
  [ "$NMEM" -lt 2 ] && EXTRA_HEAD="$EXTRA_HEAD --no-spread"
fi

# PRINT_RECIPE=1 prints the two resolved trainer argument lists (literal plus every
# override in effect) and exits. tools/check_recipe.py reads them from here, so the
# guard compares what this script would actually run, not a regex over its text.
if [ "${PRINT_RECIPE:-0}" = "1" ]; then
  echo "RESOLVED_TRAIN: $TRAIN $EXTRA_TRAIN"
  echo "RESOLVED_HEAD: $HEAD $EXTRA_HEAD"
  echo "RESOLVED_MEMBERS: $MEMBERS"
  echo "RESOLVED_HEAD_MODE: $HEAD_MODE"
  echo "RESOLVED_TEACHERS: $TEACHER_SEEDS"
  echo "RESOLVED_STUDENT: $STUDENT"
  exit 0
fi

# Fresh destinations only: a run directory that already exists would be resumed from
# or mixed into the ensemble glob below, and the head would silently read whatever
# best.pt it holds. FORCE=1 says you meant it.
if [ -z "$MEMBER_RUNS" ]; then
  for S in $MEMBERS; do
    if [ -e "$R/${NAME}_m${S}" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "$R/${NAME}_m${S} already exists; choose another NAME or FORCE=1" >&2; exit 1
    fi
  done
fi
TEACHER_DESTS=""
for T in $TEACHER_SEEDS; do TEACHER_DESTS="$TEACHER_DESTS $R/${NAME}_head_s${T}"; done
for D in "$R/${NAME}_ens" "$R/${NAME}_head" "$R/${NAME}_student" $TEACHER_DESTS; do
  if [ -e "$D" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "$D already exists; choose another NAME or FORCE=1" >&2; exit 1
  fi
done

if [ -z "$MEMBER_RUNS" ]; then
  for S in $MEMBERS; do
    echo "### forecaster seed $S"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/train_mixed.py $TRAIN $EXTRA_TRAIN \
        --seed "$S" --out "$R/${NAME}_m${S}"
  done
  MEMBER_DIRS=""; for S in $MEMBERS; do MEMBER_DIRS="$MEMBER_DIRS $R/${NAME}_m${S}"; done
else
  [ "$(echo $MEMBER_RUNS | wc -w)" = "$NMEM" ] \
    || { echo "MEMBER_RUNS names $(echo $MEMBER_RUNS | wc -w) directories for $NMEM seeds" >&2; exit 1; }
  for D in $MEMBER_RUNS; do [ -f "$D/best.pt" ] || { echo "$D/best.pt missing" >&2; exit 1; }; done
  echo "### forecasters: linking existing checkpoints (MEMBER_RUNS; audit/test path, not the recipe)"
  MEMBER_DIRS="$MEMBER_RUNS"
fi

echo "### ensemble directory"
mkdir -p "$R/${NAME}_ens"
i=0; for D in $MEMBER_DIRS; do i=$((i+1)); mkdir -p "$R/${NAME}_ens/m$i"
  ln -sf "$(cd "$D" && pwd -P)/best.pt" "$R/${NAME}_ens/m$i/best.pt"; done

if [ "$HEAD_MODE" = "single" ]; then
  echo "### interval head (single, seed ${HEAD_SEED:-0})"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/train_final_head.py $HEAD $EXTRA_HEAD \
      --runs "$R/${NAME}_ens/*" --out "$R/${NAME}_head"
  HEADCK="$R/${NAME}_head/best.pt"
else
  TEACHERS=""
  if [ -z "$TEACHER_RUNS" ]; then
    for T in $TEACHER_SEEDS; do
      echo "### interval-head teacher seed $T"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/train_final_head.py $HEAD $EXTRA_HEAD --seed "$T" \
          --runs "$R/${NAME}_ens/*" --out "$R/${NAME}_head_s${T}"
      TEACHERS="$TEACHERS,$R/${NAME}_head_s${T}/best.pt"
    done
  else
    [ "$(echo $TEACHER_RUNS | wc -w)" = "$(echo $TEACHER_SEEDS | wc -w)" ] \
      || { echo "TEACHER_RUNS count differs from TEACHER_SEEDS" >&2; exit 1; }
    echo "### interval-head teachers: reusing existing fits (TEACHER_RUNS; audit/test path, not the recipe)"
    for D in $TEACHER_RUNS; do [ -f "$D/best.pt" ] || { echo "$D/best.pt missing" >&2; exit 1; }
      TEACHERS="$TEACHERS,$(cd "$D" && pwd -P)/best.pt"; done
  fi
  TEACHERS=${TEACHERS#,}
  echo "### interval-head student (distil $(echo $TEACHER_SEEDS | wc -w) teachers into one head)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/distil_head.py $STUDENT \
      --teachers "$TEACHERS" --out "$R/${NAME}_student"
  HEADCK="$R/${NAME}_student/best.pt"
fi

echo "### package"
"$PY" tools/build_ckpt.py --head "$HEADCK" --n-models "$NMEM" \
    --out "submissions/$SUB/model.pth"
"$PY" tools/pack_submission.py "$SUB" --no-kit-deps

# Does the artefact compute what the runs computed? This was never run here, so the
# one check that would catch a packing defect was absent from the shipping path.
# It resolves members from the paths build_ckpt records, and exits non-zero on
# disagreement, so a broken pack stops the build instead of reaching the board.
echo "### pack parity"
# CHECK_PACK_EVAL names a Codabench-layout directory (tools/build_eval_npz.py). When it
# does not exist yet it is cut from the release's val_aoa trajectories here, so a
# fresh machine never depends on this host's evaluation cache.
if [ ! -f "$CHECK_PACK_EVAL/samples.npz" ]; then
  PYTHONPATH=src "$PY" tools/build_eval_npz.py val_aoa "$CHECK_PACK_EVAL" --stride 40
fi
CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/check_pack.py --sub "submissions/$SUB" --eval "$CHECK_PACK_EVAL"

# And is the pack the recipe? check_pack compares the mean forecast only; this
# re-derives every packed tensor from the run checkpoints, checks the head's
# fit-time member hashes, the pack schema and the data manifest, against the
# recipe as this very script resolved it. A build that fails here is not shipped.
echo "### recipe and pack integrity"
PYTHONPATH=src "$PY" tools/check_recipe.py --pack "submissions/$SUB/model.pth"
