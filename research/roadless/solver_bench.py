"""CPU vs GPU AMG-CG on a block's baseline system: setup and solve time, and P agreement.

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/solver_bench.py <id> [hs,]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402


def main(bid: str, hs: list[float]) -> None:
    [b] = common.build_blocks([bid])
    polys = np.asarray(b.buildings.outlines)
    for h in hs:
        g = lifted.Grid.of(b.boundary, polys, list(b.streets.geometry), h)
        reach = lifted.grounded(g, g.ff0, lifted.Params(3.0, 8))
        f, _, _ = lifted.demand(g, polys, reach, np.ones(len(polys)))
        ref = None
        for name in ("cpu", "gpu"):
            p = lifted.Params(3.0, 8, solver=lifted.solver_of(name))
            t = time.time()
            sy = lifted.System(g, g.ff0, p)
            ts = time.time() - t
            bvec = sy.load(f, p)
            for rtol in (1e-9, 1e-5, 1e-3):
                sy.solve(bvec, rtol)                    # warm (GPU kernels compile once)
                t = time.time()
                u = sy.solve(bvec, rtol)
                P = float(bvec @ u)
                ref = P if ref is None else ref
                print(f"h {h:g} {name} unknowns {sy.n:,} setup {ts:6.1f}s  rtol {rtol:g}: solve "
                      f"{time.time() - t:6.2f}s  P rel diff vs cpu 1e-9 {abs(P - ref) / ref:.1e}",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [0.5, 1.0])
