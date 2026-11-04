#!/bin/bash
# Build a candidate the way it would ship, but on a split that is honestly held
# out, and score the packed artefact the way the leaderboard scores it.
#
# This exists because of 894861. From 882969 onward every shipped model trains on
# every usable trajectory and every frame, so no local split is out-of-sample for it and
# every number quoted for a candidate is in-sample fit. Distillation raised
# in-sample fit more than anything else tried (+0.18 rel_l2, +1.15 tke on data it
# trained on) and lost 0.159 on the board.
#
# An honest split alone is not enough -- the Re folds were honest and still said
# +0.034 for distillation. Three further properties are needed, and this script
# exists to enforce all four:
#
#   1. honest      REALISTIX_HELDOUT_TAIL withholds the last 20% of every
#                  trajectory; training never sees it, with a full window of gap.
#   2. full regime coverage   all 81 regimes stay in training, so the candidate is
#                  measured at the training-set size it will ship at. The regime
#                  folds train on 70-89% and the training-size law says effects
#                  move with that.
#   3. packed      the artefact is built and scored through submission.py, so
#                  sps_score comes from the learned asymmetric head rather than
#                  evaluate.subscores' constant 0.15-sigma band -- the proxy that
#                  supplied 60-75% of every fold headline.
#
#                  NOTE (2026-09-01): property 1 held for the forecasters and NOT for
#                  the head until this date. `train_final_head.py` built its sampler
#                  with the default frames="all", so the head trained on the same
#                  tail frames `val_ingrid` scored it on. Every sps_score and every
#                  composite (mean4) number this script produced before 2026-09-01 is
#                  therefore partly in-sample and has to be remeasured; the
#                  accuracy-only columns (rel_l2/tke/mvpe) are unaffected, because
#                  those come from the forecasters.
#   4. fixed step  the checkpoint is taken at a step fixed in advance, because the
#                  shipped model has nothing to select on. Selecting here would
#                  credit the candidate with a choice it cannot make.
#
#   NAME=base bash tools/honest_pipeline.sh
#   NAME=dist EXTRA="--distill-from /…/ig_tch/best.pt" STEPS=16000 bash tools/honest_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

R=${R:-runs}
PY=${PY:-python}
NAME=${NAME:?set NAME}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-8000}
FIXED=${FIXED:-$STEPS}
EXTRA=${EXTRA:-}
EVAL=${EVAL:-eval/ingrid}

export REALISTIX_HELDOUT_TAIL=${REALISTIX_HELDOUT_TAIL:-0.2}
export REALISTIX_HELDOUT_AOA=""
export REALISTIX_HELDOUT_RE=""

# ARCHIVAL DEFAULTS (2026-09-11): this literal is the August recipe (AdamW, EMA 0.999, no
# w_sps / w_tke 1.0 / w_mvpe / Muon wd) and the head below is the 2000/1000 head. Every
# number this script produced is on that recipe; the default stays so those numbers keep
# their meaning. RECIPE=current switches BASE to the incumbent (918211) forecaster recipe;
# the head literal is left as it is unless HEAD_STEPS/HEAD_FIXED are set. Neither the
# current recipe nor the 4000/4000 head has been run through this script yet.
BASE="--mode plus_sim --film --cond norm --width 96 --ema 0.999 --w-tmean 0.5 \
      --w-rel 8.0 --w-mvpe 0.0 --eval-every 1000 --snapshots"
if [ "${RECIPE:-august}" = "current" ]; then
  BASE="--mode plus_sim --film --cond norm --width 96 --ema 0.998 --w-tmean 0.5 \
        --w-rel 8.0 --w-mvpe 0.3 --w-sps 1.0 --w-tke 1.0 --mults 1,2,4 --blocks 2 --anchor last \
        --opt muon --lr-muon 0.02 --muon-momentum 0.95 --wd-muon 0.03 --eval-every 1000 --snapshots"
  echo "RECIPE=current: incumbent (918211) forecaster recipe"
else
  echo "RECIPE=august (archival default): AdamW / EMA 0.999 / no sps, tke 1.0, mvpe, Muon -- not the shipped recipe"
fi

echo "honest build '$NAME': tail holdout ${REALISTIX_HELDOUT_TAIL}, steps $STEPS, take $FIXED"
# Reuse is keyed on NAME alone, so changing EXTRA/STEPS/BASE while keeping the name
# silently mixes members from the old recipe into the new one. Record the expanded
# recipe next to the members and refuse a reuse that disagrees with it.
RECIPE="$BASE $EXTRA steps=$STEPS fixed=$FIXED seeds=$SEEDS tail=$REALISTIX_HELDOUT_TAIL"
stamp="$R/hp_${NAME}_recipe.txt"
if [ -f "$stamp" ] && [ "$(cat "$stamp")" != "$RECIPE" ]; then
  echo "ERROR: $R/hp_${NAME}_* was built from a different recipe:" >&2
  diff <(echo "$(cat "$stamp")") <(echo "$RECIPE") >&2 || true
  echo "Pick a new NAME, or delete the old artefacts to rebuild." >&2
  exit 1
fi
printf '%s' "$RECIPE" > "$stamp"

for S in $SEEDS; do
  out="$R/hp_${NAME}_m${S}"
  [ -f "$out/best.pt" ] && { echo "  have $out"; continue; }
  $PY tools/train_mixed.py $BASE $EXTRA --steps "$STEPS" --fixed-step "$FIXED" \
      --seed "$S" --out "$out"
done

mkdir -p "$R/hp_${NAME}_ens"
i=0; for S in $SEEDS; do i=$((i+1)); mkdir -p "$R/hp_${NAME}_ens/m$i"
  ln -sf "$R/hp_${NAME}_m${S}/best.pt" "$R/hp_${NAME}_ens/m$i/best.pt"; done

n=$(echo $SEEDS | wc -w)
# The spread over one member is NaN, and as of 2026-08-31 the spread input
# measures ~0 even with three (hk4 A/B) -- a single-member candidate trains its
# head without it rather than being impossible to build.
SPREAD="--spread"; [ "$n" -lt 2 ] && SPREAD=""
$PY tools/train_final_head.py --n-models "$n" $SPREAD --width 96 --steps "${HEAD_STEPS:-2000}" \
    --eval-every 250 --train-split train --fixed-step "${HEAD_FIXED:-1000}" \
    --runs "$R/hp_${NAME}_ens/*" --out "$R/hp_${NAME}_head"

$PY tools/check_head_frames.py "$R/hp_${NAME}_head/best.pt" || {
  echo "ERROR: the head about to be packed is not tail-clean (see above)." >&2; exit 1; }

sub="submissions/_honest_${NAME}"
mkdir -p "$sub"
# Copy a submission module matching the candidate when one exists (a width-profile
# candidate needs the fractional-mults parse), else the shipped one.
if [ -f "submissions/$NAME/submission.py" ]; then
  cp "submissions/$NAME/submission.py" "$sub/submission.py"
else
  cp submissions/tke1em8ff/submission.py "$sub/submission.py"
fi
$PY tools/build_ckpt.py --head "$R/hp_${NAME}_head/best.pt" --n-models "$n" \
    --out "$sub/model.pth"
echo
$PY tools/run_and_score.py "$sub/submission.py" "$EVAL"
