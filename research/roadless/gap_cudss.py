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
device; SuperLU's factorization and solve on the host; the residuals. Rows
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
WITH = ("nvmath-python[cu12]==1.0.0", "nvidia-cudss-cu12", "cupy-cuda12x[ctk]", "scipy", "numpy",
        "pandas", "pyarrow")
INNER = "GAP_CUDSS_INNER"


def outer(bid: str) -> None:
    uv = Path.home() / ".local" / "bin" / "uv"
    cmd = [str(uv), "run", "--no-project", *(a for w in WITH for a in ("--with", w)),
           "python", "-I", str(Path(__file__).resolve()), bid]
    env = dict(os.environ, **{INNER: "1"})
    raise SystemExit(subprocess.call(cmd, env=env))


def inner(bid: str) -> None:
    import cupy as cp
    import cupyx.scipy.sparse as csp
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
        # cuDSS on the device
        Ad = csp.csr_matrix(A)
        bd = cp.asarray(b)
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD)
        for rep in range(REPS):
            cp.cuda.Device().synchronize()
            with DirectSolver(Ad, bd, options=opts) as s:
                t0 = time.time()
                s.plan()
                cp.cuda.Device().synchronize()
                t_plan = time.time() - t0
                t0 = time.time()
                s.factorize()
                cp.cuda.Device().synchronize()
                t_fact = time.time() - t0
                t0 = time.time()
                xd = s.solve()
                cp.cuda.Device().synchronize()
                t_solve = time.time() - t0
                r_gpu = float(cp.abs(bd - Ad @ xd).max() / cp.abs(bd).max())
                A2 = csp.csr_matrix((Ad.data * (1.0 + 1e-3 * cp.cos(cp.arange(Ad.nnz))),
                                     Ad.indices, Ad.indptr), shape=Ad.shape)
                A2 = (A2 + A2.T) * 0.5
                s.reset_operands(a=A2, b=bd)
                t0 = time.time()
                s.factorize()
                cp.cuda.Device().synchronize()
                t_refact = time.time() - t0
                t0 = time.time()
                xd2 = s.solve()
                cp.cuda.Device().synchronize()
                t_resolve = time.time() - t0
                r_gpu2 = float(cp.abs(bd - A2 @ xd2).max() / cp.abs(bd).max())
            rows.append(dict(block=bid, system=name, n=n, nnz=int(A.nnz), rep=rep,
                             t_plan=t_plan, t_factorize=t_fact, t_solve=t_solve,
                             t_refactorize=t_refact, t_resolve=t_resolve, resid=r_gpu,
                             resid_refactorized=r_gpu2, superlu_factorize=t_slu_f,
                             superlu_solve=t_slu_s, superlu_resid=r_slu, superlu_nnz=nnz_slu,
                             gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                             gpu_mem_gb=cp.get_default_memory_pool().total_bytes() / 2 ** 30))
            print(rows[-1], flush=True)
        del Ad, bd
        cp.get_default_memory_pool().free_all_blocks()
    ROWS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ROWS / f"{bid}.parquet")


if __name__ == "__main__":
    if os.environ.get(INNER) == "1":
        inner(sys.argv[1])
    else:
        outer(sys.argv[1])
