#!/usr/bin/env bash
# Like batch.sh, but each line starts once fewer than $2 lookahead.py runs (any launcher's) are
# alive, so lists launched at different times share one cap on the shared machine.
set -u
cd "$(dirname "$0")/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=.
L=research/roadless/atlas/lookahead/logs
PAT='^[^ ]*/python3 research/roadless/atlas/lookahead/lookahead.py run'
while read -r b s p; do
  while [ "$(pgrep -fc "$PAT")" -ge "$2" ]; do sleep 20; done
  ( uv run python research/roadless/atlas/lookahead/lookahead.py run "$b" "$s" "$p" > "$L/${b}__${s}__${p}.log" 2>&1; echo "$b $s exit $?" ) &
  sleep 20
done < <(grep -v '^#' "$1" | grep .)
wait
