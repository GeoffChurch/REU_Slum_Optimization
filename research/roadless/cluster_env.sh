# The environment of a reblock task on the ai cluster (the prelude of cluster.py's blocks kind).
# cupy finds the CUDA 12 toolkit it was installed with (the [ctk] wheels in the checkout's .venv),
# so nothing points it there; the cluster's /usr/local/cuda is 11.8 and the nodes' drivers 12.8.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONPATH=.
