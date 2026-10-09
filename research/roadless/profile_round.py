"""Profile one greedy round (rank, build, evaluate, the state's exact score) on one block:
where does the time go?

    PYTHONPATH=. uv run python research/roadless/profile_round.py <id> <h> <along> <solver> [picker]
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
import search  # noqa: E402
from clear import Clearing, sweep_of  # noqa: E402


def main(bid: str, h: float, along: str, solver: str, picker: str) -> None:
    b = common.build_blocks([bid])[0]
    scans = lifted.scans_of(solver)
    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    mesh = lifted.UniformMesh(h, offset=lifted.OFFSET)
    c = Clearing(b, mesh, lifted.Params(3.0, 8, along=lifted.along_of(along, scans),
                                     solver=lifted.solver_of(solver)),
                 population=common.POPULATIONS["area"])
    pr.disable()
    print(f"setup {time.time() - t0:.1f}s", flush=True)
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
    rnd = search.greedy_round(picker, sweep_of(solver))
    J = c.sc.J(c.sc.u0, 2.0)
    s = search.SearchState.start(c, power=2.0, score=search.Score(J, c.sc.P0),
                                  world=search.EXACT)
    rec = search.GreedyRows(block_id=bid, c=c, power=2.0, J0=J, P0=c.sc.P0,
                            t0=time.time())
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    search.Do(rnd)(s, rec)
    print(f"step {time.time() - t0:.1f}s ({len(s.order)} cleared)", flush=True)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4],
         sys.argv[5] if len(sys.argv) > 5 else "S0.01cat")
