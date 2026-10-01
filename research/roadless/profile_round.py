"""Profile one greedy round (tension + pick + exact solve) on one block: where does the time go?

    PYTHONPATH=. pixi run python research/roadless/profile_round.py <id> <h> <along> <solver> [picker]
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
from clear import Clearing, picker_of, sweep_of  # noqa: E402


def main(bid: str, h: float, along: str, solver: str, picker: str) -> None:
    b = common.build_blocks([bid])[0]
    scans = lifted.scans_of(solver)
    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    c = Clearing(b, h, lifted.Params(3.0, 8, along=lifted.along_of(along, scans),
                                     solver=lifted.solver_of(solver)),
                 population=common.POPULATIONS["area"])
    pr.disable()
    print(f"setup {time.time() - t0:.1f}s", flush=True)
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
    pk = picker_of(picker, sweep_of(solver))
    J = c.sc.J(c.sc.u0, 2.0)
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()
    t = c.tension(2.0)
    print(f"tension {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    out = pk.pick(c, t, J, 2.0)
    print(f"pick {time.time() - t1:.1f}s ({len(out.cleared)} cleared)", flush=True)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4],
         sys.argv[5] if len(sys.argv) > 5 else "S0.01cat")
