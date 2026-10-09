"""The greedy, then a polish. A greedy round (search.greedy_round) to the budget, its clearing
within the budget (every building of its steps in pick order while it fits: all of the
steps before the crossing one, part of that), then exchange refinement (search.polish) over
every building, ranked by the eps-world gradient at the 0/1 clearing (the tension), candidates
scored in the eps world at 1e-6 as SIMP's incumbent does, the result exactly.

    PYTHONPATH=. uv run python research/roadless/polish_greedy.py <id> <p> <cpu|gpu> <picker> \\
        <tries> <width> <along> <budget> <mesh> [x2]

mesh: a common.mesh_of token, the grid of every solve and score. `x2`:
two-for-one moves too (search.Swaps' pairs). Rows in
polish_rows/<along>/<picker>.P<tries>w<width>[x2]<the mesh's suffix>/D<budget>/<id>_p<p>.parquet
(h 0.5 unnamed, as in a SIMP plan): `perm` the polished clearing's Lens A, `greedy_perm` the
greedy's own clearing within the budget (a real clearing, not picker_compare's interpolation
between steps), `t` the greedy and the polish, `t_greedy` the greedy alone, `mesh` its name.
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

RTOL = 1e-3          # the solves while polishing, as SIMP's base plan's (t0.001)
SCORE_EPS = 1e-6     # candidates' eps world, as SIMP's incumbent (k1s1e-06)


def rows_of(picker: str, tries: int, width: int, pairs: bool, mesh: lifted.MeshSpec, along: str,
            budget: float) -> Path:
    return (HERE / "polish_rows" / along
            / f"{picker}.P{tries}w{width}{'x2' if pairs else ''}{mesh.suffix}" / f"D{budget:g}")


def main(bid: str, power: float, device: str, picker_spec: str, tries: int, width: int,
         along: str, budget: float, mesh: lifted.MeshSpec, pairs: bool) -> None:
    out = (rows_of(picker_spec, tries, width, pairs, mesh, along, budget)
           / f"{bid}_p{power:g}.parquet")
    c = relax._clearing(bid, device, along, mesh)
    rnd = search.greedy_round(picker_spec, clear.sweep_of(device))
    t0 = time.time()
    s = search.SearchState.start(
        c, power=power, score=search.Score(c.sc.J(c.sc.u0, power), c.sc.P0), world=search.EXACT)
    search.greedy(rnd, budget)(s, search.Silent())
    t_greedy = time.time() - t0
    c.removed[:] = False
    r = relax.cut_to_budget(s.order, c.cost, budget)
    c.removed[:] = r > 0
    rel = relax.Relaxation(c, power, rtol=RTOL)
    scorer = relax.Relaxation(c, power, q=1.0, rtol=RTOL, eps=SCORE_EPS, params=c.p)
    greedy_perm = rel.perm(rel.exact(r))
    # the polish's own count: the greedy's scorings are not its tries
    s = search.SearchState.start(c, power=power, score=search.UNSCORED, world=None)
    moves = search.polish(scorer, budget=budget, width=width, pairs=pairs, pool=relax.PAIR_POOL,
                          tries=tries)(s, search.Silent())
    rp, used = c.removed.astype(float), s.scorings
    perm = rel.perm(rel.exact(rp))
    t = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(block=bid, n=c.n, picker=picker_spec, mesh=mesh.name, perm=perm,
                       greedy_perm=greedy_perm,
                       D=float(c.cost @ rp), greedy_D=float(c.cost @ r), moves=moves,
                       scorings=used, t=t, t_greedy=t_greedy,
                       cleared=np.flatnonzero(rp).tolist())]).to_parquet(out)
    print(f"{bid} n={c.n} {picker_spec} at D {budget:g}: greedy {greedy_perm:.4f} -> polished "
          f"{perm:.4f} ({moves} moves, {used} scorings); greedy {t_greedy:.0f}s, total {t:.0f}s",
          flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (9, 10) or (len(a) == 10 and a[9] != "x2"):
        raise SystemExit("usage: <id> <p> <cpu|gpu> <picker> <tries> <width> <along> <budget> "
                         "<mesh> [x2]")
    main(a[0], float(a[1]), a[2], a[3], int(a[4]), int(a[5]), a[6], float(a[7]),
         common.mesh_of(a[8]), len(a) == 10)
