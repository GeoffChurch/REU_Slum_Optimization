"""Sequential stopping for screens: when a paired comparison has run enough blocks (BACKLOG,
"Sequential stopping"; NOTES, "Stopping by the value of more blocks").

The paired differences b - a arrive one block at a time, in a seeded random order, each with b's
and a's GPU time. b is worth adopting if its mean gain mu exceeds tau = rate x b's mean extra
GPU-seconds per block, `rate` the Lens A one GPU-second is worth. The rate is not known, only
bracketed (RATES), so a screen's call is "adopt" or "keep" when it holds at every rate in the
bracket and "ask" when it turns on the rate. An "ask" is a question for the owner in units one
can judge (this much Lens A for this much more time?), and the answer narrows RATES: yes puts
the top at the screen's break-even rate mu / extra seconds, no the bottom.

The posterior on mu is the Bayesian bootstrap (mu = sum_i w_i d_i, w ~ Dirichlet(1)), no shape
assumed: a parametric t pinned mu at 0 on such data. Deciding now loses
min(E(tau - mu)+, E(mu - tau)+) on each block the decision governs. That loss is linear in mu,
so k more blocks are worth exactly its expected drop through the posterior mean after them,
under the bootstrap a Polya urn's (`value`). The screen stops when no k is worth its cost,
rate / FUTURE x b's GPU-seconds on the next k blocks (b's time on a block predicted as a's times
b's time ratio so far), both averaged over the bracket: the owner's rate is unknown here, so it
is taken as log-uniform across RATES (requiring it at every rate instead ran r2 against base to
a median 35 blocks against 21, for the same call); at once when a block collapses (more than
COLLAPSE below a: the sentinel's veto); and never before MIN_BLOCKS, since the bootstrap is
overconfident on few blocks. Replayed on finished comparisons it says when each would have
stopped, the GPU time it spent, and whether it called it as all the blocks do.

    PYTHONPATH=. uv run python research/roadless/seqscreen.py <tuning|held> <budget> \\
        <name>=<kind>,<along>,<what> <name>=... <b>/<a>...

Live (`next`), for a b being run with a's rows on every block: b's blocks go in the replay's
first order, and `next` prints the block to run after b's rows so far, or (exit STOPPED) the
stop and its call. A loop runs b until it stops:

    while id=$(PYTHONPATH=. uv run python research/roadless/seqscreen.py next held 0.05 \\
            fl=search,uni,FL3xS0.01catr0.005m8c100D0.05 cheap=polished,uni,S0.01cat.P64w8 \\
            fl/cheap); do <run b on $id>; done

(specs as simp_compare.py's). a's and b's times must come from the same kind of card.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simp_compare import LOADERS, blocks_of  # noqa: E402

DRAWS = 4000      # Polya-urn paths per value
MIN_BLOCKS = 15   # replayed on six held-out screens (NOTES): lower realized loss than 20 at 11 of
                  # 15 rates x futures, than 10 at 9; below 15 the bootstrap's 90% interval
                  # misses the 46-block mean 12 -- 21% of the time at 10 blocks
COLLAPSE = 0.1    # BACKLOG, "Screen before the full run": a block this far below a stops it
RATES = (2e-6, 1.9e-5)   # Lens A per GPU-second, bracketed by two calls on record (NOTES):
                         # .p256w8x2 adopted over .p64w8 at +0.00019 for 10.4 s a block (at most
                         # 1.9e-5), late sampling closed at +0.0001 for ~76 s (at least ~2e-6)
RATE_POINTS = 5   # rates across RATES (geometric) that the stop averages over
FUTURE = 820      # block runs a decision governs, the 82 large blocks about ten times over;
                  # replayed from 82 to 8200 it moved one screen's stop, 15 -> 24 blocks (NOTES)
ORDER_SEED = 0
ORDERS = 20       # random block orders replayed per comparison
STOPPED = 10      # `next`'s exit when the screen has stopped (a Python error exits 1)


class Screen(NamedTuple):
    d: np.ndarray     # b - a per block run, in the order run
    t: np.ndarray     # b's GPU-seconds per block run
    ta: np.ndarray    # a's on every block of the screen, in order: those run, then the rest


class Stop(NamedTuple):
    n: int            # blocks run
    why: str          # "settled", "collapse" or "ran out"
    call: str         # "adopt", "keep" or "ask" (`call`)


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


def call(mu: float, extra: float, rates: np.ndarray) -> str:
    """b over a at every rate ("adopt"), at none ("keep"), or depending on the rate ("ask"),
    for mean gain `mu` at `extra` GPU-seconds a block."""
    taus = rates * extra
    return "adopt" if mu > taus.max() else "keep" if mu < taus.min() else "ask"


def stop_at(s: Screen, *, rates: np.ndarray, future: float, floor: int,
            rng: np.random.Generator) -> Stop | None:
    """Where the screen stops, why, and its call; None while it runs on past the blocks run."""
    for n in range(1, len(s.d) + 1):
        if s.d[n - 1] < -COLLAPSE:
            return Stop(n, "collapse", "keep")
        extra = float(np.mean(s.t[:n] - s.ta[:n]))
        if n == len(s.ta):
            return Stop(n, "ran out", call(float(s.d[:n].mean()), extra, rates))
        if n < floor:
            continue
        cost = s.t[:n].sum() / s.ta[:n].sum() * np.cumsum(s.ta[n:])
        gain = np.mean([value(s.d[:n], r * extra, len(cost), rng) for r in rates], axis=0)
        if np.all(gain <= rates.mean() / future * cost):
            return Stop(n, "settled", call(float(s.d[:n].mean()), extra, rates))
    return None


def question(b: str, a: str, mu: float, mean_a: float, ratio: float, extra: float) -> str:
    """An "ask" put to the owner, and what each answer does to RATES."""
    return (f"  ask: is {mu:+.5f} Lens A ({100 * mu / mean_a:+.2f}% of {a}'s mean) worth "
            f"{ratio:.2f}x {a}'s GPU time? Yes puts RATES' top at {mu / extra:.1e}, no its bottom")


def specs_of(which: str, budget: float, specs: list[str]) -> dict[str, pd.DataFrame]:
    blocks = blocks_of(which)
    per = {}
    for spec in specs:
        name, rest = spec.split("=")
        kind, along, what = rest.split(",")
        per[name] = LOADERS[kind](along, what, budget).reindex(blocks).dropna()
    return per


def main(which: str, budget: float, specs: list[str], pairs: list[str]) -> None:
    rates = np.geomspace(*RATES, RATE_POINTS)
    per = specs_of(which, budget, specs)
    print(f"rate {RATES[0]:.1e} .. {RATES[1]:.1e} Lens A per GPU-second, future {FUTURE}")
    for pair in pairs:
        rng = np.random.default_rng(ORDER_SEED)     # each pair its own orders, whatever precedes it
        b, a = pair.split("/")
        both = per[a].index.intersection(per[b].index)
        s = Screen((per[b].perm[both] - per[a].perm[both]).to_numpy(),
                   per[b].t[both].to_numpy(), per[a].t[both].to_numpy())
        mu, extra = float(s.d.mean()), float(np.mean(s.t - s.ta))
        whole = call(mu, extra, rates)
        stops, hours = [], []
        for o in range(ORDERS):
            i = rng.permutation(len(s.d))
            stop = stop_at(Screen(s.d[i], s.t[i], s.ta[i]), rates=rates, future=FUTURE,
                           floor=MIN_BLOCKS, rng=np.random.default_rng([ORDER_SEED, o]))
            assert stop is not None     # every block run: it stops by the last
            stops.append(stop)
            hours.append(s.t[i][:stop.n].sum() / 3600)
        ns = np.array([x.n for x in stops])
        calls = {c: sum(1 for x in stops if x.call == c) for c in sorted({x.call for x in stops})}
        print(f"{b} vs {a} [{len(s.d)}]: mu {mu:+.5f}, {extra:+.1f} s a block: {whole}; stops "
              f"after median {int(np.median(ns))} blocks ({ns.min()}-{ns.max()}), "
              f"{np.median(hours):.2f} GPU-h, calls {calls}, "
              f"{sum(x.call != whole for x in stops)} of {ORDERS} not the whole screen's")
        if whole == "ask":
            print(question(b, a, mu, float(per[a].perm[both].mean()), s.t.sum() / s.ta.sum(),
                           extra))


def upcoming(which: str, budget: float, specs: list[str], pair: str) -> int:
    """The live screen: print the block b runs next (exit 0), or its stop (exit STOPPED)."""
    per = specs_of(which, budget, specs)
    b, a = pair.split("/")
    # the replay's first order over a's blocks: a finished screen replays to this very stop
    order = per[a].index[np.random.default_rng(ORDER_SEED).permutation(len(per[a]))]
    n = next((k for k, x in enumerate(order) if x not in per[b].index), len(order))
    run = order[:n]
    if len(stray := per[b].index.difference(run)):
        raise SystemExit(f"{b} has rows off the screen's order: {', '.join(stray)}")
    s = Screen((per[b].perm[run] - per[a].perm[run]).to_numpy(), per[b].t[run].to_numpy(),
               per[a].t[order].to_numpy())
    rates = np.geomspace(*RATES, RATE_POINTS)
    stop = stop_at(s, rates=rates, future=FUTURE, floor=MIN_BLOCKS,
                   rng=np.random.default_rng([ORDER_SEED, 0]))
    if stop is None:
        print(order[n])
        return 0
    mu, extra = float(s.d[:stop.n].mean()), float(np.mean(s.t[:stop.n] - s.ta[:stop.n]))
    print(f"{b} vs {a}: stopped after {stop.n} of {len(order)} blocks ({stop.why}): {stop.call}; "
          f"mu {mu:+.5f}, {extra:+.1f} s a block, {s.t[:stop.n].sum() / 3600:.2f} GPU-h",
          file=sys.stderr)
    if stop.call == "ask":
        print(question(b, a, mu, float(per[a].perm[run[:stop.n]].mean()),
                       s.t[:stop.n].sum() / s.ta[:stop.n].sum(), extra), file=sys.stderr)
    return STOPPED


if __name__ == "__main__":
    live = sys.argv[1] == "next"
    args = sys.argv[3 + live:]
    specs, pairs = [a for a in args if "=" in a], [a for a in args if "/" in a and "=" not in a]
    if live:
        (pair,) = pairs
        raise SystemExit(upcoming(sys.argv[2], float(sys.argv[3]), specs, pair))
    main(sys.argv[1], float(sys.argv[2]), specs, pairs)
