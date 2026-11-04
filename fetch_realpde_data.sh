#!/usr/bin/env bash
#
# Reproduce the RealPDE competition data tree on another machine.
#
#   ./fetch_realpde_data.sh [DEST]
#
# Produces, under DEST (default ./RealPDE-Competition-Data):
#
#   training_data/train_sim/    100 .h5   simulated trajectories
#   training_data/train_real/    82 .h5   real PIV trajectories
#   example_data/3750_0.h5
#   baseline_checkpoints/{pack_ckpt_fp16.py,sim_pretrain/,sim_real_ft/}
#
# Safe to re-run: files whose size already matches are skipped, partial
# downloads resume, and extraction is skipped once the counts are right.
#
# Options:
#   --jobs N            parallel downloads (default 4)
#   --remove-tarballs   delete the two .tar.gz files after extraction
#   --refresh-manifest  re-fetch sizes from the HF API instead of the
#                       pinned table below (use if upstream republishes)
#   --verify-only       check an existing tree, download/extract nothing
#
# Needs: bash, curl, tar, gzip, coreutils. python3 only for --refresh-manifest.

set -euo pipefail

REPO="AI4Science-WestlakeU/RealPDE-Competition-Data"
BASE="https://huggingface.co/datasets/$REPO/resolve/main"
API="https://huggingface.co/api/datasets/$REPO?blobs=true"

JOBS=4
REMOVE_TARBALLS=0
REFRESH_MANIFEST=0
VERIFY_ONLY=0
DEST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs)             JOBS="$2"; shift 2 ;;
    --remove-tarballs)  REMOVE_TARBALLS=1; shift ;;
    --refresh-manifest) REFRESH_MANIFEST=1; shift ;;
    --verify-only)      VERIFY_ONLY=1; shift ;;
    -h|--help)          sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)                 echo "unknown option: $1" >&2; exit 2 ;;
    *)                  DEST="$1"; shift ;;
  esac
done
DEST="${DEST:-$PWD/RealPDE-Competition-Data}"

# Pinned manifest: exact byte sizes verified against the HF API on 2026-08-05.
# Downloads are checked against these, so a truncated or silently-changed file
# is caught rather than extracted.
MANIFEST=$(cat <<'EOF'
2504        .gitattributes
3564        README.md
2854        baseline_checkpoints/pack_ckpt_fp16.py
32154928    baseline_checkpoints/sim_pretrain/sim_cno.pth
402988026   baseline_checkpoints/sim_pretrain/sim_fno.pth
201396069   baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth
50401485    baseline_checkpoints/sim_pretrain/sim_transolver.pth
32154928    baseline_checkpoints/sim_real_ft/sim_real_cno.pth
402988026   baseline_checkpoints/sim_real_ft/sim_real_fno.pth
201396349   baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth
50401485    baseline_checkpoints/sim_real_ft/sim_real_transolver.pth
88807246    example_data/3750_0.h5
7393582393  train_real.tar.gz
8826593178  train_sim.tar.gz
EOF
)

# Expected trajectory counts after extraction.
declare -A EXPECT_COUNT=( [train_sim]=100 [train_real]=82 )

log()  { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for c in curl tar gzip; do
  command -v "$c" >/dev/null || fail "'$c' not found in PATH"
done

fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo -1; }

if [ "$REFRESH_MANIFEST" -eq 1 ]; then
  command -v python3 >/dev/null || fail "--refresh-manifest needs python3"
  log "Refreshing manifest from $API"
  MANIFEST=$(curl -fsSL "$API" | python3 -c '
import json,sys
for s in json.load(sys.stdin)["siblings"]:
    if s.get("size") is not None:
        print(s["size"], s["rfilename"])
')
  [ -n "$MANIFEST" ] || fail "could not fetch manifest"
fi

# ---------------------------------------------------------------- disk space
need_bytes=0
while read -r sz _; do [ -n "${sz:-}" ] && need_bytes=$((need_bytes + sz)); done <<<"$MANIFEST"
# tarballs are ~16 GB and expand to ~17 GB, so budget download + extraction
need_gb=$(( (need_bytes + 17000000000) / 1000000000 ))
mkdir -p "$DEST"
avail_gb=$(( $(df -Pk "$DEST" | awk 'NR==2{print $4}') / 1000000 ))
log "Destination : $DEST"
log "Space needed: ~${need_gb} GB (download + extracted), available: ${avail_gb} GB"
if [ "$VERIFY_ONLY" -eq 0 ] && [ "$avail_gb" -lt "$need_gb" ]; then
  fail "not enough free space at $DEST"
fi

# ----------------------------------------------------------------- download
# Fetch one file, resuming a partial and restarting an oversized one.
fetch_one() {
  local want="$1" rel="$2" path="$DEST/$2" have
  mkdir -p "$(dirname "$path")"
  have=$(fsize "$path")
  if [ "$have" -eq "$want" ]; then
    log "  skip (complete)  $rel"
    return 0
  fi
  if [ "$have" -gt "$want" ]; then
    log "  oversized, restarting  $rel"
    rm -f "$path"
  elif [ "$have" -gt 0 ]; then
    log "  resuming at $have  $rel"
  fi
  curl -fsSL --retry 5 --retry-delay 5 --retry-connrefused \
       -C - -o "$path" "$BASE/$rel" \
    || { log "  FAILED  $rel"; return 1; }
  have=$(fsize "$path")
  [ "$have" -eq "$want" ] || { log "  SIZE MISMATCH $rel (got $have, want $want)"; return 1; }
  log "  ok  $rel"
}

if [ "$VERIFY_ONLY" -eq 0 ]; then
  log ""
  log "Downloading 14 files with $JOBS parallel jobs..."
  running=0
  while read -r sz rel; do
    [ -n "${rel:-}" ] || continue
    fetch_one "$sz" "$rel" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done <<<"$MANIFEST"
  wait
fi

# ------------------------------------------------------------------- verify
log ""
log "Verifying sizes against manifest..."
bad=0
while read -r sz rel; do
  [ -n "${rel:-}" ] || continue
  got=$(fsize "$DEST/$rel")
  if [ "$got" -ne "$sz" ]; then
    printf '  BAD  %s (got %s, want %s)\n' "$rel" "$got" "$sz" >&2
    bad=$((bad + 1))
  fi
done <<<"$MANIFEST"
[ "$bad" -eq 0 ] || fail "$bad file(s) wrong size — re-run to resume"
log "  all 14 files byte-exact"

# ------------------------------------------------------------------ extract
# Filenames collide between the two splits (5025_0.h5 is in both), so each
# tarball keeps its own top-level directory under training_data/.
extract_one() {
  local split="$1" tgz="$DEST/$1.tar.gz" out="$DEST/training_data/$1"
  local want="${EXPECT_COUNT[$split]}" have=0
  [ -d "$out" ] && have=$(find "$out" -name '*.h5' -type f | wc -l)
  if [ "$have" -eq "$want" ]; then
    log "  skip (already $want files)  $split"
    return 0
  fi
  log "  gzip -t $split.tar.gz"
  gzip -t "$tgz" || { log "  CORRUPT $split.tar.gz"; return 1; }
  log "  extracting $split..."
  tar xzf "$tgz" -C "$DEST/training_data"
}

if [ "$VERIFY_ONLY" -eq 0 ]; then
  log ""
  log "Extracting into training_data/..."
  mkdir -p "$DEST/training_data"
  extract_one train_sim  &
  extract_one train_real &
  wait
fi

# ------------------------------------------------- verify extracted contents
log ""
log "Verifying extracted trajectory counts..."
for split in train_sim train_real; do
  want="${EXPECT_COUNT[$split]}"
  got=$(find "$DEST/training_data/$split" -name '*.h5' -type f 2>/dev/null | wc -l)
  [ "$got" -eq "$want" ] || fail "$split has $got .h5 files, expected $want"
  log "  $split: $got files"
done

if [ "$REMOVE_TARBALLS" -eq 1 ] && [ "$VERIFY_ONLY" -eq 0 ]; then
  log ""
  log "Removing tarballs (extraction verified)..."
  rm -f "$DEST/train_sim.tar.gz" "$DEST/train_real.tar.gz"
fi

log ""
log "Done. Tree at $DEST"
du -sh "$DEST/training_data" "$DEST/example_data" "$DEST/baseline_checkpoints" 2>/dev/null || true
cat <<'NOTES'

Notes on the data (these differ from the upstream dataset card):
  * There is NO measured_data/ group. Every .h5 stores u, v, t, x, y, aoa, re
    at the top level. One loader works for example, sim and real.
  * Real files have NO 'p' key at all — it is absent, not zero-filled.
    Sim files do have p, shape (1000, 64, 128) float32.
  * Real trajectories are ragged: 79 are T=868, plus one each T=607/492/282.
    Sim is uniformly T=1000.
  * Dtypes are inverted: real u/v float64 + t float32; sim u/v float32 + t float64.
  * Real FILENAMES DISAGREE with the 're' attribute in 78 of 82 files
    (10125_0.h5 has re=10142). Read Reynolds number from the 're' dataset,
    never from the filename. Sim filenames happen to match, which hides this.
  * Fields are 64x128; competition evaluation is 32x64 after 2x subsampling.

Data is CC BY-NC 4.0, non-commercial research/competition use.
NOTES
