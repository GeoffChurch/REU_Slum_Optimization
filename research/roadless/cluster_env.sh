# The environment of a reblock task on the ai cluster (the prelude of cluster.py's blocks kind).
# CUDA comes from cupy's pip toolkit in the deps dir: the cluster's
# /usr/local/cuda is 11.8 and the nodes' drivers 12.8; the toolkit's headers and libraries are
# outside site-packages, so cupy is told where (CUDA_PATH, LD_LIBRARY_PATH).
D=$HOME/.cache/reblock-research/pydeps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONPATH=.:$D CUDA_PATH=$D/nvidia/cuda_runtime
LD_LIBRARY_PATH=$(ls -d "$D"/nvidia/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH
