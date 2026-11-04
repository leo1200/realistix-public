#!/bin/bash
# Resumable fetch of RealPDEBench foil (metadata, then real, then numerical); 4 parallel curls.
cd "$(dirname "$0")"
B=https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench/resolve/main
get() { p=$1; sz=$2; mkdir -p "$(dirname "$p")"
  [ -f "$p" ] && [ "$(stat -c%s "$p")" = "$sz" ] && return 0
  for i in 1 2 3 4 5; do curl -sfL -C - --retry 5 -o "$p" "$B/$p"; [ "$(stat -c%s "$p" 2>/dev/null)" = "$sz" ] && { echo "ok $p"; return 0; }; sleep 10; done
  echo "FAIL $p"; }
export -f get; export B
for grp in foil/[^h] foil/hf_dataset/[^rn] foil/hf_dataset/real/ foil/hf_dataset/numerical/; do
  grep "^$grp" filelist.txt | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'
done
echo DONE
