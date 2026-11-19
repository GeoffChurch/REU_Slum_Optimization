#!/usr/bin/env bash
# Runs crossings.py on each "<world> <plan>" line of $1, a line starting once fewer than $2 of
# these runs are alive and fewer than $3 runs of any study (crossings.py or lookahead.py, any
# launcher's) are: the machine is shared. One BLAS thread each; logs in logs/.
set -u
cd "$(dirname "$0")/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
PY=${PY:-/home/gchurchill/src/reblock/.venv/bin/python}
L=research/roadless/atlas/routers/logs
MINE='^[^ ]*/python3? research/roadless/atlas/crossings/crossings.py run'
ANY='^[^ ]*/python3? research/roadless/atlas/(crossings/crossings|lookahead/lookahead).py run'
mkdir -p "$L"
while read -r w p; do
  while [ "$(pgrep -fc "$MINE")" -ge "$2" ] || [ "$(pgrep -fc "$ANY")" -ge "$3" ]; do
    sleep 20
  done
  ( "$PY" research/roadless/atlas/crossings/crossings.py run "$w" "$p" > "$L/${w//:/_}__$p.log" 2>&1
    echo "$w $p exit $?" ) &
  sleep 20
done < <(grep -v '^#' "$1" | grep .)
wait
