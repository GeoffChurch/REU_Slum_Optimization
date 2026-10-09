"""Sequential stopping for screens: when a paired comparison has run enough blocks (BACKLOG,
"Sequential stopping"; NOTES, "Stopping by the value of more blocks").

The paired differences b - a arrive one block at a time, in a seeded random order, each with b's
and a's GPU time. The decision is to adopt b if its mean gain mu exceeds tau = rate x b's mean
extra GPU-seconds per block, `rate` the Lens A one GPU-second is worth (the owner's exchange
rate; NOTES brackets it by the variants adopted and closed). The posterior on mu is the Bayesian
bootstrap (mu = sum_i w_i d_i, w ~ Dirichlet(1)), no shape assumed: a parametric t pinned mu at
0 on such data. Deciding now loses min(E(tau - mu)+, E(mu - tau)+) on each block the decision
governs. That loss is linear in mu, so k more blocks are worth exactly its expected drop through
the posterior mean after them, under the bootstrap a Polya urn's (`value`). The screen stops
when no k is worth its cost, rate / future x b's GPU-seconds on the next k blocks (`future` the
block runs the decision governs; b's time on a block predicted as a's times b's time ratio so
far); at once when a block collapses (more than COLLAPSE below a: the sentinel's veto); and
never before MIN_BLOCKS, since the bootstrap is overconfident on few blocks. Replayed on finished
comparisons it says when each would have stopped, the GPU time it spent, and whether it decided
as all the blocks do.

    PYTHONPATH=. uv run python research/roadless/seqscreen.py <tuning|held> <budget> <rate> \\
        <future> <name>=<kind>,<along>,<what> <name>=... <b>/<a>...

(specs as simp_compare.py's). a's and b's times must come from the same kind of card.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simp_compare import LOADERS, blocks_of  # noqa: E402

DRAWS = 4000      # Polya-urn paths per value
MIN_BLOCKS = 15   # replayed on six held-out screens (NOTES): lower realized loss than 20 at 11 of
                  # 15 rates x futures, than 10 at 9; below 15 the bootstrap's 90% interval
                  # misses the 46-block mean 12 -- 21% of the time at 10 blocks
COLLAPSE = 0.1    # BACKLOG, "Screen before the full run": a block this far below a stops it
ORDER_SEED = 0
ORDERS = 20       # random block orders replayed per comparison


class Screen(NamedTuple):
    d: np.ndarray     # b - a per block, in the order run
    t: np.ndarray     # b's GPU-seconds per block
    ta: np.ndarray    # a's


class Stop(NamedTuple):
    n: int            # blocks run
    why: str          # "settled", "collapse" or "ran out"
    adopt: bool       # the decision then: mu's posterior mean above tau
    tau: float        # tau then


def value(d: np.ndarray, tau: float, k_max: int, rng: np.random.Generator) -> np.ndarray:
    """The value of 1 .. k_max more blocks: the expected drop in the decision's loss through the
    posterior mean after them, the next blocks drawn by a Polya urn on the bootstrap's atoms."""
    v, c = np.unique(d, return_counts=True)
    a = c.astype(float)
    w = rng.dirichlet(a, size=DRAWS)
    pick = (rng.random((DRAWS, k_max, 1)) > np.cumsum(w, axis=1)[:, None, :]).sum(axis=2)
    after = ((a @ v + np.cumsum(v[np.minimum(pick, len(v) - 1)], axis=1))
             / (a.sum() + np.arange(1, k_max + 1)))
    # the posterior mean is a martingale, so the drop is E(tau - m')+ on the side m is on
    # (E(m' - tau)+ below tau): one tail, less Monte Carlo error than the difference of two
    if a @ v / a.sum() >= tau:
        return np.maximum(tau - after, 0).mean(axis=0)
    return np.maximum(after - tau, 0).mean(axis=0)


def stop_at(s: Screen, *, rate: float, future: float, floor: int,
            rng: np.random.Generator) -> Stop:
    """Where the screen stops, why, and what it decides."""
    for n in range(1, len(s.d) + 1):
        tau = rate * float(np.mean(s.t[:n] - s.ta[:n]))
        if s.d[n - 1] < -COLLAPSE:
            return Stop(n, "collapse", False, tau)
        if n < floor or n == len(s.d):
            continue
        cost = s.t[:n].sum() / s.ta[:n].sum() * np.cumsum(s.ta[n:])
        if np.all(value(s.d[:n], tau, len(cost), rng) <= rate / future * cost):
            return Stop(n, "settled", float(s.d[:n].mean()) > tau, tau)
    return Stop(len(s.d), "ran out", float(s.d.mean()) > tau, tau)


def main(which: str, budget: float, rate: float, future: float, specs: list[str],
         pairs: list[str]) -> None:
    blocks = blocks_of(which)
    per = {}
    for spec in specs:
        name, rest = spec.split("=")
        kind, along, what = rest.split(",")
        per[name] = LOADERS[kind](along, what, budget).reindex(blocks).dropna()
    for pair in pairs:
        rng = np.random.default_rng(ORDER_SEED)     # each pair its own orders, whatever precedes it
        b, a = pair.split("/")
        both = per[a].index.intersection(per[b].index)
        s = Screen((per[b].perm[both] - per[a].perm[both]).to_numpy(),
                   per[b].t[both].to_numpy(), per[a].t[both].to_numpy())
        tau = rate * float(np.mean(s.t - s.ta))
        adopt = float(s.d.mean()) > tau
        stops, hours = [], []
        for o in range(ORDERS):
            i = rng.permutation(len(s.d))
            stop = stop_at(Screen(s.d[i], s.t[i], s.ta[i]), rate=rate, future=future,
                           floor=MIN_BLOCKS, rng=np.random.default_rng([ORDER_SEED, o]))
            stops.append(stop)
            hours.append(s.t[i][:stop.n].sum() / 3600)
        ns = np.array([x.n for x in stops])
        why = {w: sum(1 for x in stops if x.why == w) for w in sorted({x.why for x in stops})}
        print(f"{b} vs {a} [{len(s.d)}]: mu {s.d.mean():+.5f}, {np.mean(s.t - s.ta):+.1f} s a "
              f"block, tau {tau:.5f}: {'adopt' if adopt else 'keep a'}; stops after median "
              f"{int(np.median(ns))} blocks ({ns.min()}-{ns.max()}), {np.median(hours):.2f} "
              f"GPU-h, {why}, {sum(x.adopt != adopt for x in stops)} of {ORDERS} decided "
              f"otherwise")


if __name__ == "__main__":
    args = sys.argv[5:]
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
         [a for a in args if "=" in a], [a for a in args if "/" in a and "=" not in a])
