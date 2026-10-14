"""The GPU pilot (NOTES, "The complex as the runtime: one fixed mesh", the owner's GPU question): a
large block's systems on the fixed mesh factored by NVIDIA cuDSS (nvmath-python's DirectSolver,
SPD) on one cluster GPU, against SuperLU on the host in the runtime's order. cuDSS is not a
dependency of the project: the task runs this file again in an ephemeral environment (uv run
--with ..., uv's cache), nothing added to pyproject.toml or uv.lock.

    uv run python research/roadless/cluster.py submit blocks <run> --with <A_<id>.npz> <id> -- \\
        research/roadless/gap_cudss.py {id} 0.5

(`0.5`: the raster grid cluster.py sizes the task's card by, as every blocks run names it; unused
here.)

The systems (gap_rows/runtime's dumps, `A<name>` CSR, `b<name>`, `p<name>` the runtime's
elimination order): `x` a random 0/1 clearing at area D 0.05 (an exact score's), `e` the eps
world at 1e-4 (a SIMP iteration's). Per system: cuDSS's plan (reordering and symbolic
factorization), factorization, solve, and a refactorization and solve with the values changed
(the plan reused, as every state of one pattern would), each REPS times, the operands on the
host (nvmath moves them); SuperLU's factorization and solve on the host; the residuals. Rows
ROWS/<id>.parquet.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "gap_rows" / "runtime" / "cudss"
REPS = 3
WITH = ("nvmath-python[cu12]==1.0.0", "nvidia-cudss-cu12", "nvidia-nvjitlink-cu12", "scipy",
        "numpy", "pandas", "pyarrow")
INNER = "GAP_CUDSS_INNER"


def outer(bid: str) -> None:
    uv = Path.home() / ".local" / "bin" / "uv"
    # not `python -I`: uv lays the --with packages over its base environment through the
    # interpreter's path, which isolated mode drops (the pilot's cupy then found no CUDA wheels)
    cmd = [str(uv), "run", "--no-project", *(a for w in WITH for a in ("--with", w)),
           "python", str(Path(__file__).resolve()), bid]
    env = dict(os.environ, **{INNER: "1"})
    raise SystemExit(subprocess.call(cmd, env=env))


def _preload() -> None:
    """The CUDA libraries the wheels bring, loaded globally before cupy looks for them (on a
    node with no system copy, cuda.pathfinder missed the ephemeral environment's
    libnvJitLink)."""
    import ctypes
    import glob
    libs = sorted({lib for d in sys.path for lib in glob.glob(f"{d}/nvidia/*/lib/lib*.so*")})
    for _round in range(len(libs)):            # each round loads those whose dependencies are in
        left = []
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                left.append(lib)
        if len(left) in (0, len(libs)):
            break
        libs = left
    print(f"preloaded the CUDA wheels' libraries ({len(left)} not loaded: {left[:3]})",
          flush=True)


def inner(bid: str) -> None:
    _preload()
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from nvmath.sparse.advanced import (DirectSolver, DirectSolverMatrixType,
                                        DirectSolverOptions)
    from scipy.sparse.linalg import splu

    z = np.load(Path(os.environ["CLUSTER_SUBMIT_INPUTS"]) / f"A_{bid}.npz")
    rows = []
    for name in ("x", "e"):
        n = int(z[f"A{name}_n"])
        A = sp.csr_matrix((z[f"A{name}_data"], z[f"A{name}_indices"], z[f"A{name}_indptr"]),
                          shape=(n, n))
        b = z[f"b{name}"]
        p = z[f"p{name}"]
        # SuperLU on the host, the runtime's order
        t0 = time.time()
        lu = splu(A[p][:, p].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                  options=dict(SymmetricMode=True))
        t_slu_f = time.time() - t0
        t0 = time.time()
        x = np.empty(n)
        x[p] = lu.solve(b[p])
        t_slu_s = time.time() - t0
        r_slu = float(np.abs(b - A @ x).max() / np.abs(b).max())
        nnz_slu = int(lu.L.nnz + lu.U.nnz)
        del lu
        # cuDSS on the device (host operands: nvmath moves them; blocking, so each call's time
        # is its own)
        w = 1.0 + 1e-3 * np.cos(np.arange(A.nnz))
        A2 = sp.csr_matrix((A.data * w, A.indices, A.indptr), shape=A.shape)
        A2 = ((A2 + A2.T) * 0.5).tocsr()
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, blocking=True)
        for rep in range(REPS):
            t0 = time.time()
            with DirectSolver(A, b, options=opts, execution="cuda") as s:
                t_setup = time.time() - t0
                t0 = time.time()
                s.plan()
                t_plan = time.time() - t0
                t0 = time.time()
                s.factorize()
                t_fact = time.time() - t0
                t0 = time.time()
                xs = np.asarray(s.solve())
                t_solve = time.time() - t0
                r_gpu = float(np.abs(b - A @ xs).max() / np.abs(b).max())
                t0 = time.time()
                s.reset_operands(a=A2, b=b)
                t_reset = time.time() - t0
                t0 = time.time()
                s.factorize()
                t_refact = time.time() - t0
                t0 = time.time()
                xs2 = np.asarray(s.solve())
                t_resolve = time.time() - t0
                r_gpu2 = float(np.abs(b - A2 @ xs2).max() / np.abs(b).max())
            rows.append(dict(block=bid, system=name, n=n, nnz=int(A.nnz), rep=rep,
                             t_setup=t_setup, t_plan=t_plan, t_factorize=t_fact,
                             t_solve=t_solve, t_reset=t_reset, t_refactorize=t_refact,
                             t_resolve=t_resolve, resid=r_gpu, resid_refactorized=r_gpu2,
                             superlu_factorize=t_slu_f, superlu_solve=t_slu_s,
                             superlu_resid=r_slu, superlu_nnz=nnz_slu))
            print(rows[-1], flush=True)
    ROWS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ROWS / f"{bid}.parquet")


if __name__ == "__main__":
    if os.environ.get(INNER) == "1":
        inner(sys.argv[1])
    else:
        outer(sys.argv[1])
