"""The greedy, then a polish. A greedy picker (clear.py) to the budget, its clearing within the
budget (every building of its steps in pick order while it fits: all of the steps before the
crossing one, part of that), then exchange refinement (relax.exchange) over every building,
ranked by the eps-world gradient at the 0/1 clearing (the tension), candidates scored in the eps
world at 1e-6 as SIMP's incumbent does, the result exactly.

    PYTHONPATH=. uv run python research/roadless/polish_greedy.py <id> <p> <cpu|gpu> <picker> \\
        <tries> <width> <along> <budget> [x2]

`x2`: two-for-one moves too (relax.exchange's pairs). Rows in
polish_rows/<along>/<picker>.P<tries>w<width>[x2]/D<budget>/<id>_p<p>.parquet: `perm` the
polished clearing's Lens A, `greedy_perm` the greedy's own clearing within the budget (a real
clearing, not picker_compare's interpolation between steps), `t` the greedy and the polish,
`t_greedy` the greedy alone.
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
import relax  # noqa: E402

RTOL = 1e-3          # the solves while polishing, as SIMP's base plan's (t0.001)
SCORE_EPS = 1e-6     # candidates' eps world, as SIMP's incumbent (k1s1e-06)


def rows_of(picker: str, tries: int, width: int, pairs: bool, along: str, budget: float) -> Path:
    return (HERE / "polish_rows" / along / f"{picker}.P{tries}w{width}{'x2' if pairs else ''}"
            / f"D{budget:g}")


def main(bid: str, power: float, device: str, picker_spec: str, tries: int, width: int,
         along: str, budget: float, pairs: bool) -> None:
    out = rows_of(picker_spec, tries, width, pairs, along, budget) / f"{bid}_p{power:g}.parquet"
    c = relax._clearing(bid, device, along)
    picker = clear.picker_of(picker_spec, clear.sweep_of(device))
    t0 = time.time()
    order = [j for pk, _ in clear.grow(c, picker, power, budget, c.sc.J(c.sc.u0, power), False)
             for j in pk.cleared]
    t_greedy = time.time() - t0
    c.removed[:] = False
    r = relax.cut_to_budget(order, c.cost, budget)
    rel = relax.Relaxation(c, power, rtol=RTOL)
    scorer = relax.Relaxation(c, power, q=1.0, rtol=RTOL, eps=SCORE_EPS, params=c.p)
    greedy_perm = rel.perm(rel.exact(r))
    J, rp, moves, used = relax.exchange(
        scorer.value, lambda x: scorer.value_sgrad(x)[1], scorer.value(r), r, np.arange(c.n),
        c.cost, budget, tries, width, pairs)
    perm = rel.perm(rel.exact(rp))
    t = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(block=bid, n=c.n, picker=picker_spec, perm=perm, greedy_perm=greedy_perm,
                       D=float(c.cost @ rp), greedy_D=float(c.cost @ r), moves=moves,
                       scorings=used, t=t, t_greedy=t_greedy,
                       cleared=np.flatnonzero(rp).tolist())]).to_parquet(out)
    print(f"{bid} n={c.n} {picker_spec} at D {budget:g}: greedy {greedy_perm:.4f} -> polished "
          f"{perm:.4f} ({moves} moves, {used} scorings); greedy {t_greedy:.0f}s, total {t:.0f}s",
          flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 9 and a[8] != "x2":
        raise SystemExit(f"the optional last argument is x2, not {a[8]!r}")
    main(a[0], float(a[1]), a[2], a[3], int(a[4]), int(a[5]), a[6], float(a[7]), len(a) == 9)
