"""Add/remove subset search (plus-l take-away-r; wiki pages/methods/plus-l-take-away-r.md): the
greedy's add moves and first-order restore moves (a cleared building put back) under a
schedule. First schedule, grow then prune (the mycooc vocabulary recipe): the greedy clears up
to `grow` x d_max, then each round restores the cleared buildings whose clearing is worth least
per unit of population (Clearing.tension(restore=True): the greedy's finite-difference tension
toward everything put back; the derivative at the 0/1 clearing understates it, measured: Lens A
0.38 / 0.37 vs the greedy's 0.60 / 0.53 on two small blocks), re-solving between rounds, down
to nothing. Every state at or below d_max is scored exactly; restoring only ever shrinks the set,
so the states are nested and the rows come out in the greedy's format (every budget in one
run). A gate is judged with its companions cleared, the greedy's blind spot; the risk is
substitutes (each cheap to restore alone) restored in one batch, hence small batches.

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/search.py <ids,> <spec> <p> <cpu|gpu> [along] [d_max]

spec: GP<grow>x<greedy picker>r<restore share per round>[m<shortlist>], e.g.
GP3xS0.01catr0.005m8.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import RTOL_SCORE, Clearing, _exact, grow, picker_of, rows_dir, sweep_of  # noqa: E402
from relax import Relaxation  # noqa: E402

EPS_SCREEN = 1e-6               # the eps world the shortlist is scored in (the real one to ~1e-3)


class GrowPrune(NamedTuple):
    grow: float                 # the greedy clears up to grow x d_max
    picker: str                 # the greedy's picker
    restore: float              # population share restored per round
    shortlist: int              # first-order shortlist rescored per round (0: first order only)

    @property
    def name(self) -> str:
        return (f"GP{self.grow:g}x{self.picker}r{self.restore:g}"
                + (f"m{self.shortlist}" if self.shortlist else ""))


def plan_of(spec: str) -> GrowPrune:
    m = re.fullmatch(r"GP([0-9.]+)x(\S+?)r([0-9.]+)(?:m(\d+))?", spec)
    if m is None:
        raise ValueError(f"unknown search {spec!r}")
    return GrowPrune(float(m.group(1)), m.group(2), float(m.group(3)),
                     0 if m.group(4) is None else int(m.group(4)))


def restore_batch(c: Clearing, power: float, step: float, shortlist: int,
                  screen: Relaxation) -> np.ndarray:
    """The cleared buildings to put back this round: least loss of J per unit of population
    first, until the cleared share falls to the next multiple of `step` below. The loss is
    first order (the restore tension), or for its `shortlist` cheapest each restore's own loss
    in the eps world at EPS_SCREEN (the first-order ranking alone, rank correlation 0.6 -- 0.7
    with the exact loss, lost 0.22 Lens A on a small block where exact backward elimination
    beat the greedy)."""
    on = np.flatnonzero(c.removed)
    loss = -c.tension(power, restore=True).g[on] / c.cost[on]
    D = float(c.cost[on].sum())
    target = (np.ceil(D / step - 1e-9) - 1) * step
    order = on[np.argsort(loss, kind="stable")]
    k = int(np.searchsorted(np.cumsum(c.cost[order]), D - target - 1e-12)) + 1
    if shortlist:
        cand = order[:max(shortlist, 2 * k)]
        x = c.removed.astype(float)
        now = screen.value(x)
        own = np.empty(len(cand))
        for i, j in enumerate(cand):
            x[j] = 0.0
            own[i] = (screen.value(x) - now) / c.cost[j]
            x[j] = 1.0
        order = np.concatenate([cand[np.argsort(own, kind="stable")], order[len(cand):]])
        k = int(np.searchsorted(np.cumsum(c.cost[order]), D - target - 1e-12)) + 1
    return order[:k]


def grow_prune(b, plan: GrowPrune, power: float, along: str, device: str, d_max: float
               ) -> pd.DataFrame:
    p = lifted.Params(3.0, 8, along=lifted.along_of(along, lifted.scans_of(device)),
                      solver=lifted.solver_of(device))
    c = Clearing(b, 0.5, p, population=common.POPULATIONS["area"])
    J0, P0 = c.sc.J(c.sc.u0, power), c.sc.P0
    t0 = time.time()
    for _pk, _D in grow(c, picker_of(plan.picker, sweep_of(device)), power,
                        plan.grow * d_max, J0):
        pass
    print(f"  {b.block_id} grown to D {float(c.cost[c.removed].sum()):.3f} "
          f"{time.time() - t0:.0f}s", flush=True)
    screen = Relaxation(c, power, q=1.0, rtol=RTOL_SCORE, eps=EPS_SCREEN)
    states = []                                     # (D, J, P, cleared) at or below d_max
    while c.removed.any():
        c.removed[restore_batch(c, power, plan.restore, plan.shortlist, screen)] = False
        D = float(c.cost[c.removed].sum())
        if D <= d_max + 1e-12 and c.removed.any():
            J, P = _exact(c, [], power)
            states.append((D, J, P, c.removed.copy()))
    rows = [dict(block=b.block_id, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[],
                 P0=P0, t=0.0)]
    prev = np.zeros(c.n, dtype=bool)
    for D, J, P, r in reversed(states):
        rows.append(dict(block=b.block_id, n=c.n, step=len(rows), D=D,
                         perm=1 - (J / J0) ** (1 / power), perm1=1 - P / P0,
                         cleared=np.flatnonzero(r & ~prev).tolist(), P0=P0,
                         t=time.time() - t0))
        prev = r
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {plan.name}: "
          f"{len(rows) - 1} states; perm' {rows[-1]['perm']:.3f} at D {rows[-1]['D']:.3f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main(ids: list[str], plan: GrowPrune, power: float, device: str, along: str,
         d_max: float) -> None:
    out = rows_dir(plan.name, 0.5, "area", power, along)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for b in common.build_blocks(ids):
        f = out / f"{b.block_id}.parquet"
        if f.exists():
            continue
        try:
            rows = grow_prune(b, plan, power, along, device, d_max)
        except Exception as e:
            print(f"{b.block_id} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            failed.append(b.block_id)
            continue
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        rows.to_parquet(tmp)
        os.replace(tmp, f)
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main(sys.argv[1].split(","), plan_of(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
         sys.argv[5] if len(sys.argv) > 5 else "uni",
         float(sys.argv[6]) if len(sys.argv) > 6 else 0.15)
