"""Sequential stopping for screens: how many blocks a paired comparison needs (BACKLOG,
"Sequential stopping").

The paired differences b - a arrive one block at a time, in a seeded random order. After each,
the posterior on their mean mu by the Bayesian bootstrap (mu = sum_i w_i d_i, w ~ Dirichlet(1)):
no shape assumed, because the differences are no shape -- most are exactly 0 (the variant left
the clearing alone) and the rest a skewed tail (a parametric t pinned mu at 0 on such data and
called a never-behind variant behind after 3 blocks). From MIN_BLOCKS on, a screen stops when
P(mu > tau) leaves [STOP_LO, STOP_HI] -- tau the smallest mean gain worth the variant's extra
time, the owner's value judgement (for a variant that cannot score below its base, tau 0 is
decided by the first positive block) -- or the 90% interval is narrower than `width`. Replayed
on finished comparisons it says how many blocks each would have needed.

    PYTHONPATH=. uv run python research/roadless/seqscreen.py <tuning|held> <budget> <tau> <width> \\
        <name>=<kind>,<along>,<what> <name>=... <b>/<a>...

(specs as simp_compare.py's)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simp_compare import LOADERS, blocks_of  # noqa: E402

DRAWS = 4000
MIN_BLOCKS = 20   # replayed on the held-out screens (tau 0.002): 10 mis-called the polish in 2 of
                  # 20 orders, 15 in 1, 20 in none -- and still stopped after ~20 of 46 blocks
STOP_LO, STOP_HI = 0.05, 0.95
ORDER_SEED = 0
ORDERS = 20                       # random block orders replayed per comparison


def posterior(d: np.ndarray, tau: float) -> tuple[float, float, float, float]:
    """(P(mu > tau), mean, 5%, 95%) of mu given paired differences d (Bayesian bootstrap)."""
    mu = np.random.default_rng(ORDER_SEED).dirichlet(np.ones(len(d)), size=DRAWS) @ d
    return (float(np.mean(mu > tau)), float(mu.mean()), float(np.quantile(mu, 0.05)),
            float(np.quantile(mu, 0.95)))


def stop_at(d: np.ndarray, tau: float, width: float) -> tuple[int, str]:
    """The block count at which the screen stops, and why."""
    for n in range(MIN_BLOCKS, len(d) + 1):
        p, _, lo, hi = posterior(d[:n], tau)
        if p > STOP_HI:
            return n, "worth"
        if p < STOP_LO:
            return n, "not worth"
        if hi - lo < width:
            return n, "settled"
    return len(d), "ran out"


def main(which: str, budget: float, tau: float, width: float, specs: list[str],
         pairs: list[str]) -> None:
    blocks = blocks_of(which)
    per = {}
    for spec in specs:
        name, rest = spec.split("=")
        kind, along, what = rest.split(",")
        per[name] = LOADERS[kind](along, what, budget).reindex(blocks).dropna()
    rng = np.random.default_rng(ORDER_SEED)
    for pair in pairs:
        b, a = pair.split("/")
        both = per[a].index.intersection(per[b].index)
        d = (per[b].perm[both] - per[a].perm[both]).to_numpy()
        p, m, lo, hi = posterior(d, tau)
        stops = [stop_at(d[rng.permutation(len(d))], tau, width) for _ in range(ORDERS)]
        ns = np.array([n for n, _ in stops])
        why = {w: sum(1 for _, x in stops if x == w) for _, w in stops}
        print(f"{b} vs {a} [{len(d)}]: all blocks P(mu>{tau:g}) {p:.3f}, mu {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; "
              f"stops after median {int(np.median(ns))} blocks (range {ns.min()}-{ns.max()}), {why}")


if __name__ == "__main__":
    args = sys.argv[5:]
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
         [a for a in args if "=" in a], [a for a in args if "/" in a and "=" not in a])
