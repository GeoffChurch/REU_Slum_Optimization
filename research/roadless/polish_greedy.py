"""The greedy, then a polish. A greedy round (search.greedy_round) to the budget, its clearing
within the budget (every building of its steps in pick order while it fits: all of the
steps before the crossing one, part of that), then exchange refinement (search.polish) over
every building, ranked by the eps-world gradient at the 0/1 clearing (the tension), candidates
scored in the eps world at 1e-6 as SIMP's incumbent does (or, `E`, exactly:
search.PolishScoring), the result exactly.

First it frees (B's first stage, clear.free_first): the crossings that free the homes enclosed
at baseline, cheapest per unit of population, within the budget. The greedy starts from them
(its D counts them; its clearing is them and its picks while they fit the rest of the budget)
and the polish never moves them. Where no home is enclosed nothing is held and the run is as
before.

    PYTHONPATH=. uv run python research/roadless/polish_greedy.py <id> <p> <cpu|gpu> <picker> \\
        <tries> <width> <metric> <budget> <mesh> [x2] [E]

metric: a raster conductance or the gap complex's name (metrics.metric_of; the complex: cpu, mesh
0.5, which names no grid of its; cpu or gpu, its factorization's device). mesh: a common.mesh_of
token, the raster's grid of every solve and score. `x2`: two-for-one moves too (search.Swaps'
pairs); `E`: the candidates scored exactly. Rows in
polish_rows/<metric>/<picker>.P<tries>w<width>[x2][E]<the mesh's suffix>/D<budget>/<id>_p<p>.parquet
(h 0.5 unnamed, as in a SIMP plan): `perm` the polished clearing's Lens A, `greedy_perm` the
greedy's own clearing within the budget (a real clearing, not picker_compare's interpolation
between steps), `t` the free stage, the greedy and the polish, `t_greedy` the free stage and the
greedy, `mesh` its name, `free_D` the free stage's D, `free_cleared` its buildings, `enclosed`
the polished clearing's B (i) (the population share it leaves enclosed:
common.Scorer.enclosed_left), `device` the run's (seam.Metric.device).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import lifted  # noqa: E402
import search  # noqa: E402
import relax  # noqa: E402
import seam  # noqa: E402

RTOL = 1e-3          # the solves while polishing, as SIMP's base plan's (t0.001)
SCORE_EPS = 1e-6     # candidates' eps world, as SIMP's incumbent (k1s1e-06)


def rows_of(picker: str, tries: int, width: int, pairs: bool, scoring: search.PolishScoring,
            mesh: lifted.MeshSpec, along: str, budget: float) -> Path:
    return (HERE / "polish_rows" / along
            / f"{picker}.P{tries}w{width}{'x2' if pairs else ''}{scoring.suffix}{mesh.suffix}"
            / f"D{budget:g}")


def main(bid: str, power: float, metric: seam.Metric, picker_spec: str, tries: int, width: int,
         budget: float, mesh: lifted.MeshSpec, pairs: bool, *,
         scoring: search.PolishScoring) -> None:
    """`metric` the run's (metrics.metric_of: the raster's on `mesh`, or the complex's); its row
    (`run`'s) in rows_of."""
    out = (rows_of(picker_spec, tries, width, pairs, scoring, mesh, metric.name, budget)
           / f"{bid}_p{power:g}.parquet")
    row = run(bid, power, metric, picker_spec, tries, width, budget, mesh, pairs,
              scoring=scoring)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out)


def run(bid: str, power: float, metric: seam.Metric, picker_spec: str, tries: int, width: int,
        budget: float, mesh: lifted.MeshSpec, pairs: bool, *,
        scoring: search.PolishScoring) -> dict[str, object]:
    """The greedy and its polish on `bid` (module docstring): the run's row."""
    [b] = common.build_blocks([bid])
    c = metric.clearing(b, demand=common.PopulationDemand())
    rnd = search.greedy_round(picker_spec, metric.sweep)
    t0 = time.time()
    # B's first stage: the crossings that free the enclosed homes, the greedy's start (its D
    # counts them) and not movable in the polish; none enclosed, none held
    held = c.free_first(budget=budget, held=np.zeros(c.n, dtype=bool))
    c.removed[:] = held
    start = (search.Score(*c.exact(held, power)) if held.any()
             else search.Score(*c.baseline(power)))      # the baseline's, no solve
    s = search.SearchState.start(c, power=power, score=start, world=search.EXACT)
    search.greedy(rnd, budget)(s, search.Silent())
    t_greedy = time.time() - t0
    c.removed[:] = False
    r = ((relax.cut_to_budget(s.order, c.cost, budget - float(c.cost[held].sum())) > 0)
         | held).astype(float)
    c.removed[:] = r > 0
    rel = c.relaxation(power, q=1.0, rtol=RTOL, eps=clear.EPS)
    scorer = rel.metric_world(SCORE_EPS)
    greedy_perm = rel.perm(rel.exact(r))
    # the polish's own count: the greedy's scorings are not its tries
    s = search.SearchState.start(c, power=power, score=search.UNSCORED, world=None)
    s.movable[held] = False                                 # held stay held
    moves = search.polish(scoring.world(scorer), budget=budget, width=width, pairs=pairs,
                          pool=relax.PAIR_POOL, tries=tries)(s, search.Silent())
    rp, used = c.removed.astype(float), s.scorings
    perm = rel.perm(rel.exact(rp))
    t = time.time() - t0
    enclosed = c.enclosed_left(rp > 0)
    print(f"{bid} n={c.n} {picker_spec} at D {budget:g}: greedy {greedy_perm:.4f} -> polished "
          f"{perm:.4f} ({moves} moves, {used} scorings, enclosed {enclosed:.4f}); greedy "
          f"{t_greedy:.0f}s, total {t:.0f}s", flush=True)
    return dict(block=bid, n=c.n, picker=picker_spec, mesh=mesh.name, perm=perm,
                greedy_perm=greedy_perm, D=float(c.cost @ rp), greedy_D=float(c.cost @ r),
                moves=moves, scorings=used, t=t, t_greedy=t_greedy,
                cleared=np.flatnonzero(rp).tolist(), free_D=float(c.cost[held].sum()),
                free_cleared=np.flatnonzero(held).tolist(), enclosed=enclosed,
                device=metric.device)


if __name__ == "__main__":
    import metrics
    a = sys.argv[1:]
    flags = a[9:]
    if len(a) < 9 or flags not in ([], ["x2"], ["E"], ["x2", "E"]):
        raise SystemExit("usage: <id> <p> <cpu|gpu> <picker> <tries> <width> <metric> <budget> "
                         "<mesh> [x2] [E]")
    mesh = common.mesh_of(a[8])
    main(a[0], float(a[1]), metrics.metric_of(a[6], device=a[2], mesh=mesh), a[3], int(a[4]),
         int(a[5]), float(a[7]), mesh, "x2" in flags,
         scoring=search.polish_scoring_of("E" if "E" in flags else ""))
