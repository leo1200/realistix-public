#!/bin/bash
# Turn a *packed-artefact* change into a paired (local, board) observation.
#
# `kfold.py` trains one model per fold, so it cannot express a change in ensemble
# size or in the interval head at all -- both are post-training. Those changes are a
# third of the submission history and include the only two board observations that
# move SPS hard. This builds the same change under the honest tail holdout, packs it
# through `submission.py`, and scores it the way the leaderboard does, so its
# `sps_score` comes from the learned asymmetric head and is comparable to the board's.
#
#   N=1               HEAD_W=96 bash tools/backtest_pack.sh   # 893273's probe
#   N=2 HEAD_SPREAD=1 HEAD_W=48 bash tools/backtest_pack.sh   # 886080's mistake
#   N=3               HEAD_W=96 bash tools/backtest_pack.sh   # the shipped shape
#
# Members are shared across calls (`ig_base_s0..2`), so only the first build pays for
# training. Compare two runs with `realistix.board.acc_delta(sub, ref, with_sps=True)`
# for the board side -- with_sps because this instrument's sps is legitimate.
#
# NOTE (2026-09-01): "this instrument's sps is legitimate" was false for every run
# before this date. `train_final_head.py` ignored the tail holdout when building its
# training sampler, so the head saw the tail that `val_ingrid` then scored. The
# sps-bearing back-tests recorded from this script (the 1-member and 2m+w48 rows in
# `predict_score.py`'s transfer table among them) need remeasuring before they can be
# quoted again.
set -euo pipefail
cd "$(dirname "$0")/.."

R=${R:-runs}
PY=${PY:-python}
EVAL=${EVAL:-eval/ingrid}
N=${N:?set N, the number of ensemble members}
HEAD_W=${HEAD_W:-96}
TAG=${TAG:-b${N}w${HEAD_W}}

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

for s in $(seq 0 $((N-1))); do
  [ -f "$R/ig_base_s${s}/best.pt" ] && { echo "### have ig_base_s${s}"; continue; }
  echo "### honest member $s"
  $PY tools/train_mixed.py $BASE --steps 8000 --fixed-step 8000 --seed "$s" \
      --out "$R/ig_base_s${s}"
done

ens="$R/hp_${TAG}_ens"; mkdir -p "$ens"
i=0; for s in $(seq 0 $((N-1))); do i=$((i+1)); mkdir -p "$ens/m$i"
  ln -sf "$R/ig_base_s${s}/best.pt" "$ens/m$i/best.pt"; done

# One member has nothing to disagree with, so the spread input goes -- which is what
# 893273 shipped, and is part of the change rather than a flaw in the comparison.
extra=""; [ "$N" -gt 1 ] && extra="--spread"
# Refuse to reuse a head that did not honour the tail holdout: it trained on the very
# tail this script then scores, so reusing one re-emits exactly the retracted number
# the fix exists to replace. Read the checkpoint's own record, not its mtime.
if [ -f "$R/hp_${TAG}_head/best.pt" ] && \
   ! $PY tools/check_head_frames.py "$R/hp_${TAG}_head/best.pt"; then
  echo "ERROR: $R/hp_${TAG}_head/best.pt is not tail-clean (see above)." >&2
  exit 1
fi
if [ ! -f "$R/hp_${TAG}_head/best.pt" ]; then
  echo "### ${N}-member head, width $HEAD_W"
  $PY tools/train_final_head.py --n-models "$N" $extra --width "$HEAD_W" --steps "${HEAD_STEPS:-2000}" \
      --eval-every 250 --train-split train --fixed-step "${HEAD_FIXED:-1000}" \
      --runs "$ens/*" --out "$R/hp_${TAG}_head"
fi

$PY tools/check_head_frames.py "$R/hp_${TAG}_head/best.pt" || {
  echo "ERROR: the head about to be packed is not tail-clean (see above)." >&2; exit 1; }

sub="submissions/_honest_${TAG}"
mkdir -p "$sub"; cp submissions/tke1em8ff/submission.py "$sub/submission.py"
$PY tools/build_ckpt.py --head "$R/hp_${TAG}_head/best.pt" --n-models "$N" \
    --out "$sub/model.pth"
echo "### scoring $TAG"
$PY tools/run_and_score.py "$sub/submission.py" "$EVAL" | tee "$R/hp_${TAG}_score.txt"
