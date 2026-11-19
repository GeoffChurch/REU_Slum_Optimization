#!/usr/bin/env bash
# Runs crossings.py on each "<world> <plan>" line of a file, at most $2 at once (one BLAS thread
# each: the machine is shared), each logged to logs/<world>__<plan>.log.
set -u
cd "$(dirname "$0")/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=.
L=research/roadless/atlas/crossings/logs
grep -v '^#' "$1" | grep . | xargs -P "$2" -L 1 bash -c \
  'uv run python research/roadless/atlas/crossings/crossings.py run "$0" "$1" > "'$L'/${0//:/_}__$1.log" 2>&1; echo "$0 $1 exit $?"'
