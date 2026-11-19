#!/usr/bin/env bash
# Runs lookahead.py on each "<bid> <search> <plan>" line of a file, at most $2 at once (one BLAS
# thread each: the machine is shared), each logged to logs/<bid>__<search>__<plan>.log.
set -u
cd "$(dirname "$0")/../../../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=.
L=research/roadless/atlas/lookahead/logs
grep -v '^#' "$1" | grep . | xargs -P "$2" -L 1 bash -c \
  'uv run python research/roadless/atlas/lookahead/lookahead.py run "$0" "$1" "$2" > "'$L'/$0__$1__$2.log" 2>&1; echo "$0 $1 exit $?"'
