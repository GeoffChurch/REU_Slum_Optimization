"""The routers' own time (ears.Router): on a block's first state, per router and building cost
(ears.routers_at: Visibility at b0 only, Eikonal not at binf),
Paths.ears (every source's search, the best `pairs` drawn and costed) timed REPEAT times after a
compiling call, its least kept; beside it the step's exact rescoring (ears.offer less Paths.ears),
what a run's step costs besides its routing.

    T=research/roadless/atlas/routers/timing.py        # from the repo root, PYTHONPATH=.:src
    python $T <bid>...                                 # rows to results/timing.csv (appended)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import ears  # noqa: E402

REPEAT = 3
PLAN = "ears.paths.{r}.b{b:g}.f2000.e30.loop.nodt.gl0.25Croad.n2s0.1.M8.K4.D1"
BLDS = (0.0, 20.0, float("inf"))


def main(bids: list[str]) -> None:
    out = HERE / "results" / "timing.csv"
    out.parent.mkdir(exist_ok=True)
    for bid in bids:
        e = ears.world_of(bid, 0.25)
        g = e.g
        st = g.state(g.step >= 0)
        u, _ = g.solve(st)
        faces = e.faces()
        rows = []
        for b in BLDS:
            for r in (x.name for x in ears.routers_at(b)):
                plan = ears.Plan.of(PLAN.format(r=r, b=b))
                gain = plan.contrast.gains(g, st, u)
                e.set_u(u)
                plan.candidates.ears(e, faces, gain, cost=plan.cost, scale=0.0)
                ts = []
                for _ in range(REPEAT):
                    t0 = time.perf_counter()
                    plan.candidates.ears(e, faces, gain, cost=plan.cost, scale=0.0)
                    ts.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                ears.offer(e, plan, faces, st, u, scale=0.0, log=lambda s: None)
                step = time.perf_counter() - t0
                rows.append(dict(block=bid, b=b, router=r, raster=e.ra.shape[0] * e.ra.shape[1],
                                 paths_s=min(ts), step_s=step))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False,
                                  float_format="%.4g")


if __name__ == "__main__":
    if not sys.argv[1:]:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
