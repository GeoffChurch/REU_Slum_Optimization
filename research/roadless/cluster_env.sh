# The environment of a reblock task on the ai cluster (the prelude of cluster.py's blocks kind).
# cupy finds the CUDA 12 toolkit it was installed with (the [ctk] wheels in the checkout's .venv),
# so nothing points it there; the cluster's /usr/local/cuda is 11.8 and the nodes' drivers 12.8.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONPATH=.
# no bytecode written into the per-commit checkout the run's tasks share (road_rescore.py
# requires it with the single-threaded settings)
export PYTHONDONTWRITEBYTECODE=1
# the complex's fixed meshes cached on the node's own disk (gap_clear.CACHE): a block's tasks
# share it, and the shared home's quota does not hold them
export GAP_CACHE="${TMPDIR:-/tmp}/gap_cache_$USER"
# reblock's derive cache (reblock.derive_graph: the road methods' proposals and their substrates)
# on the node's own disk too: the shared home is near its quota
export REBLOCK_CACHE_DIR="${TMPDIR:-/tmp}/reblock_derive_$USER"
