"""Guards for the less myopic optimizers (NOTES, "Walled-in homes, round 2: T, U and W, each with
its best optimizer"; walled.py's arms Fa -- Fd): each passes on its blocks and fails under each of
its faults (FAULTS: python walled_check.py <guard> <ids,|-> <fault>).

- `join-planted` (a synthetic strip, its street the east edge: a home in a pocket sealed by two
  buildings in a row, PLANTED's): the path moves' best chain at baseline is the two buildings, its
  estimate within PLANTED_RATIO of the exact change in P their clearing makes, and free first's
  crossing for the home is the same two. Faults `no-adjacency` (gap_joins.adjacency empty: single
  buildings only), `unsealed` (a sealed region's own resistance read as 0).
- `join-null` (a real block under metrics.HOMES): a join of buildings already cleared or held
  adds no move; the polish with a joiner offering only such joins ends where the polish without
  joins ends, its clearing and J bit for bit. Fault `kept-null` (search._joined keeping a join
  that adds nothing).
- `join-budget` (a real block under metrics.HOMES, at a random clearing on the budget): every
  crossing offered there fits by giving back the cleared buildings worth least (its D within the
  budget, the given-back the least by the ranking's -g / cost, none of them held or added), and
  scored exactly its J the model's afresh. Fault `no-giveback` (search._joined giving nothing
  back).
- `translucent` (a real block): SIMP under `.e0.05-0.0001` optimizes stage k of n at 0.05 (1e-4 /
  0.05)^(k / (n - 1)), the last at the plan's eps, every iterate in its stage's eps world. Fault
  `stage-eps-kept` (gap_clear.Relaxation.stage keeping its eps).
- `free-track` (a real block under metrics.HOMES): `.ft` keeps the better of its two tracks, each
  the run its free stage alone gives (`.f0`, and free first held: SIMP is deterministic), their
  scores its `tracks_perm`. Fault `first-only` (FreeTrack's tracks the first alone).
- `fw-bound` (a real block under metrics.HOMES, `.f0`): Frank-Wolfe's bound at FW_ITERS lies at or
  below the exact J of every clearing tried within the budget (its own rounding, random ones, free
  first's crossings). Fault `empty-step` (relax.knapsack's step nothing: the gap 0, the "bound" the
  start's J).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/walled_check.py
"""
from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_exits as gx  # noqa: E402
import gap_factor  # noqa: E402
import gap_joins  # noqa: E402
import metrics  # noqa: E402
import region_leaks  # noqa: E402
import relax  # noqa: E402
import search  # noqa: E402

if TYPE_CHECKING:
    from reblock.contracts import Block

POP = common.PopulationDemand()
BUDGET = relax.D_LENS
SEED = 20261024
PLANTED_RATIO = 3.0     # join-planted: the best chain's estimate within this factor of the exact
                        # change in P (the estimate's resistances are a sheet's squares, not the
                        # mesh's: a ranking, not a score)
FW_ITERS = 6            # fw-bound: Frank-Wolfe's iterations (a bound at every one)
FW_TOL = 1e-9           # fw-bound: a clearing's J below the bound by more than this (relative)


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _bare(name: str) -> gcl.ComplexMetric:
    """`name`'s metric without the region model's barriers and street boost (a synthetic block has
    no OSM features), its leak and placement kept."""
    m = gcl.complex_of(name, factorization=gap_factor.SuperLU())
    return dataclasses.replace(m, exits=dataclasses.replace(m.exits, barriers=gx.NoBarriers(),
                                                            boost=gx.NoBoost()))


# ------------------------------------------------------------------ join-planted

POCKET_HOME = (4.0, 6.0, 9.0, 11.0)     # the home in the west pocket
WALL_1 = (20.0, 0.0, 24.0, 18.0)       # the two buildings in a row sealing it, 0.3 m apart (under
WALL_2 = (24.3, 0.0, 28.0, 18.0)       # the shy distance's 0.5 m: one wall of two buildings)
EAST_HOMES = ((38.0, 3.0, 43.0, 8.0), (38.0, 10.0, 43.0, 15.0), (48.0, 6.0, 52.0, 12.0))


def _planted() -> Block:
    """A strip 60 x 18 m, its east edge the street; a home in a pocket west of a wall of two
    buildings in a row (the pocket's one way out), three homes east of it."""
    homes = [shapely.box(*POCKET_HOME), shapely.box(*WALL_1), shapely.box(*WALL_2),
             *(shapely.box(*h) for h in EAST_HOMES)]
    return region_leaks.synthetic_block("synthetic-planted-crossing", shapely.box(0, 0, 60, 18),
                                        shapely.LineString([(60.0, 0.0), (60.0, 18.0)]), homes,
                                        None)


def join_planted(_ids: list[str]) -> list[str]:
    b = _planted()
    m = _bare(metrics.HOMES).clearing(b, demand=POP)
    bad = []
    # the pocket's home and the wall's west building, whose own ground is the pocket's
    if not (np.array_equal(np.flatnonzero(m.walled0), [0, 1]) and not m.nodem.any()):
        return [f"the pocket's home and the wall's west building not the walled in with demand: "
                f"walled "
                f"{np.flatnonzero(m.walled0).tolist()}, without demand "
                f"{np.flatnonzero(m.nodem).tolist()} (the premise)"]
    none = np.zeros(m.n, dtype=bool)
    ch = gap_joins.chains(m, none, sources=m.n, width=4)
    x = none.copy()
    x[[1, 2]] = True
    dP = m.baseline(1.0)[1] - m.exact(x, 1.0)[1]
    crossings = m.freed(budget=1.0, held=none).crossings
    top = set(ch[0].buildings) if ch else set()
    ratio = ch[0].gain / dP if ch else np.nan
    path_ok = len(crossings) == 1 and set(crossings[0].path) == {1, 2}
    ok = top == {1, 2} and 1.0 / PLANTED_RATIO <= ratio <= PLANTED_RATIO and path_ok
    print("planted: the chains at baseline " + ", ".join(f"{list(c.buildings)} ({c.gain:.4g})"
                                                         for c in ch)
          + f"; the exact change of clearing [1, 2] {dP:.4g} (estimate x{ratio:.3f}); free "
          f"first's crossings {[cr.path for cr in crossings]} -> {'ok' if ok else 'FAILS'}",
          flush=True)
    if not ok:
        bad.append("planted")
    return bad


def _empty_adjacency(m: gcl.Model, one: np.ndarray, Z: Any) -> Any:
    import scipy.sparse as sp
    return sp.csr_matrix((m.n, m.n))


_REGIONS = gap_joins._regions


def _unsealed(m: gcl.Model, st: gcl.State) -> tuple[np.ndarray, np.ndarray]:
    lab, R = _REGIONS(m, st)
    return lab, np.zeros_like(R)


# ------------------------------------------------------------------ join-null, join-budget

class _Fixed:
    """A joiner offering the joins `fixed` (the guards')."""

    def __init__(self, fixed: list[search.Join], held: np.ndarray):
        self.fixed, self.held_ = fixed, held

    @property
    def held(self) -> np.ndarray:
        return self.held_

    def joins(self, s: search.SearchState) -> list[search.Join]:
        return list(self.fixed)


def _homes_model(bid: str) -> gcl.Model:
    [b] = common.build_blocks([bid])
    return metrics.metric_of(metrics.HOMES, device="cpu").clearing(b, demand=POP)


def _random(m: gcl.Model, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(m.n)
    x = np.zeros(m.n, dtype=bool)
    x[order[np.cumsum(m.cost[order]) <= BUDGET]] = True
    return x


def _polished(m: gcl.Model, x: np.ndarray, joins: tuple[search.Joiner, ...]
              ) -> tuple[float, np.ndarray]:
    rel = m.relaxation(1.0, q=1.0, eps=1e-4)
    scorer = rel.metric_world(1e-6)
    m.removed[:] = x
    s = search.SearchState.start(m, power=1.0, score=search.UNSCORED, world=None)
    search.polish(search.Exactly().world(scorer), budget=BUDGET, width=8, pairs=True,
                  pool=relax.PAIR_POOL, tries=32, joins=joins)(s, search.Silent())
    out = m.removed.copy()
    m.removed[:] = False
    return float(s.score.J), out


def join_null(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        m = _homes_model(bid)
        x = _random(m, np.random.default_rng(SEED))
        held = np.zeros(m.n, dtype=bool)
        held[np.flatnonzero(~x)[:3]] = True
        null = [search.Join(add=tuple(int(j) for j in np.flatnonzero(x)[:4]), est=0.0),
                search.Join(add=tuple(int(j) for j in np.flatnonzero(held)), est=0.0)]
        m.removed[:] = x
        s = search.SearchState.start(m, power=1.0, score=search.UNSCORED, world=None)
        g = np.zeros(m.n)
        tier = search._joined(s, search.Grad(g=g, r=x.astype(float)), _Fixed(null, held),
                              BUDGET)
        m.removed[:] = False
        none = len(tier.moves) == 0
        # the clearing's score, before and after a chain search has solved on the model
        Ja = m.exact(x, 1.0)[0]
        gap_joins.chains(m, x, sources=8, width=4)
        Jb = m.exact(x, 1.0)[0]
        J0, r0 = _polished(m, x, ())
        J1, r1 = _polished(m, x, (_Fixed(null, held),))
        same = J0 == J1 and np.array_equal(r0, r1) and Ja == Jb
        ok = none and same
        print(f"{bid}: null joins offered as moves {len(tier.moves)}; the clearing's J "
              f"{Ja:.12g} before a chain search, {Jb:.12g} after; the polish with the null joins "
              f"J {J1:.12g}, without {J0:.12g}, clearings alike {np.array_equal(r0, r1)} "
              f"-> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


_JOINED = search._joined


def _kept_null(s: search.SearchState, r: search.Grad, joiner: search.Joiner,
               budget: float) -> search.Batches:
    out = _JOINED(s, r, joiner, budget)
    extra = [search.Move(add=[], restore=[]) for j in joiner.joins(s)]
    return search.Batches([*out.moves, *extra], np.r_[out.est, np.zeros(len(extra))])


def join_budget(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        m = _homes_model(bid)
        none = np.zeros(m.n, dtype=bool)
        crossings = m.freed(budget=1.0, held=none).crossings
        if not crossings:
            bad.append(f"{bid}: no crossing (the premise)")
            continue
        # a clearing filled to the budget in a random order, none of the crossings' buildings
        # among it: every crossing then breaks the budget unless it gives back
        x = none.copy()
        avoid = np.zeros(m.n, dtype=bool)
        avoid[[b for cr in crossings for b in cr.path]] = True
        left = BUDGET
        for b in np.random.default_rng(SEED + 1).permutation(m.n):
            if not avoid[b] and m.cost[b] <= left:
                x[b] = True
                left -= float(m.cost[b])
        held = np.zeros(m.n, dtype=bool)
        rel = m.relaxation(1.0, q=1.0, eps=1e-4)
        g = rel.metric_world(1e-6).value_sgrad(x.astype(float))[1]
        m.removed[:] = x
        s = search.SearchState.start(m, power=1.0, score=search.UNSCORED, world=None)
        joiner = _Fixed([search.Join(add=tuple(cr.path), est=0.0) for cr in crossings], held)
        tier = search._joined(s, search.Grad(g=g, r=x.astype(float)), joiner, BUDGET)
        m.removed[:] = False
        worst, fits, least, exact = 0.0, True, True, 0.0
        key = -g / m.cost
        for mv in tier.moves:
            y = x.copy()
            y[mv.add] = True
            y[mv.restore] = False
            D = float(m.cost[y].sum())
            worst = max(worst, D)
            fits &= D <= BUDGET + 1e-12 and not set(mv.restore) & set(mv.add)
            kept = np.flatnonzero(x & ~np.isin(np.arange(m.n), mv.restore)
                                  & ~np.isin(np.arange(m.n), mv.add))
            if mv.restore and len(kept):
                least &= float(key[mv.restore].max()) <= float(key[kept].min()) + 1e-15
            J1 = m.exact(y, 1.0)[0]
            J2 = _homes_model(bid).exact(y, 1.0)[0] if mv is tier.moves[0] else J1
            exact = max(exact, abs(J1 / J2 - 1.0))
        gave = sum(1 for mv in tier.moves if mv.restore)
        ok = bool(tier.moves) and gave > 0 and fits and least and exact <= 1e-12
        print(f"{bid}: {len(crossings)} crossings, {len(tier.moves)} offered at a random "
              f"clearing on the budget (D {float(m.cost[x].sum()):.6f}), {gave} giving back: D "
              f"at most {worst:.6f} (budget {BUDGET:g}) {fits}, the least worth given back "
              f"{least}, J afresh within {exact:.1e} -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _no_giveback(s: search.SearchState, r: search.Grad, joiner: search.Joiner,
                 budget: float) -> search.Batches:
    return _JOINED(s, r, joiner, np.inf)


# ------------------------------------------------------------------ translucent

def translucent(ids: list[str]) -> list[str]:
    bad = []
    plan = relax.plan_of("fw0.q3.i10.e0.05-0.0001.k1s1e-06.p256w8x2E.f0")
    want = [0.05 * (1e-4 / 0.05) ** (k / 3) for k in range(4)]
    for bid in ids:
        m = _homes_model(bid)
        rel = m.relaxation(1.0, q=1.0, eps=plan.eps)
        seen: list[tuple[float, float]] = []

        def watch(x: np.ndarray, st: relax.Stage, rel: gcl.Relaxation = rel,
                  seen: list[tuple[float, float]] = seen) -> None:
            seen.append((st.eps, rel.eps))
        relax.simp(rel, relax.start_x(0, m.cost, BUDGET), BUDGET, plan.stages, 2, plan.update,
                   log=lambda s: None, watch=watch)
        stages = [st.eps for st in plan.stages]
        ok = (np.allclose(stages, want, rtol=1e-12) and all(a == b for a, b in seen)
              and rel.eps == plan.eps and sorted({a for a, _b in seen}) == sorted(stages))
        print(f"{bid}: the stages' eps {[f'{e:.3g}' for e in stages]}; every iterate in its "
              f"stage's world {all(a == b for a, b in seen)} ({len(seen)} iterates); after, "
              f"eps {rel.eps:g} -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _stage_eps_kept(self: gcl.Relaxation, st: relax.Stage) -> None:
    self.q, self.beta = st.q, st.beta


# ------------------------------------------------------------------ fw-bound

def fw_bound(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        m = _homes_model(bid)
        rel = m.relaxation(1.0, q=1.0, eps=1e-4)
        fw = relax.frank_wolfe(rel, BUDGET, FW_ITERS, log=lambda s: None)
        lb = float(fw["lb"])
        rng = np.random.default_rng(SEED + 2)
        xs = [relax.round_by_x(fw["x"], m.cost, BUDGET) > 0, *(_random(m, rng) for _ in range(4))]
        none = np.zeros(m.n, dtype=bool)
        xs.append(m.free_first(budget=BUDGET, held=none))
        Js = [m.exact(x, 1.0)[0] for x in xs]
        worst = min(J / lb - 1.0 for J in Js)
        ok = worst >= -FW_TOL and lb < rel.J0
        print(f"{bid}: the bound at {len(fw['hist'])} iterations J {lb:.6g} (J0 {rel.J0:.6g}); "
              f"{len(Js)} clearings within the budget, the lowest J {min(Js):.6g} ({worst:+.2e} "
              f"of the bound) -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _empty_step(grad: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    return np.zeros(len(grad))


# ------------------------------------------------------------------ free-track

TRACK_PLAN = "fw0.q3.i10.e0.0001.k1s1e-06.p64w8x2E"     # the guard's: a shorter polish


def free_track(ids: list[str]) -> list[str]:
    bad = []
    metric = metrics.metric_of(metrics.HOMES, device="cpu")
    for bid in ids:
        rows = {f: relax.one(bid, 1.0, metric, relax.plan_of(TRACK_PLAN + f), BUDGET, demand=POP)
                for f in (".ft", ".f0", "")}
        ft, none, first = rows[".ft"], rows[".f0"], rows[""]
        best = max((none, first), key=lambda r: float(r["simp_perm"]))
        ok = (ft["simp_perm"] == best["simp_perm"]
              and ft["simp_cleared"] == best["simp_cleared"]
              and ft["tracks_perm"] == [none["simp_perm"], first["simp_perm"]]
              and len(first["free_cleared"]) > 0)
        print(f"{bid}: `.ft` {ft['simp_perm']:.6f} (tracks {ft['tracks_perm']}, kept "
              f"{ft['track']}); `.f0` alone {none['simp_perm']:.6f}, free first held "
              f"{first['simp_perm']:.6f} ({len(first['free_cleared'])} held) "
              f"-> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _first_track_only(self: relax.FreeTrack, c: Any, *, budget: float,
                      held: np.ndarray) -> tuple[np.ndarray, ...]:
    return (held.copy(),)


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "join-planted": join_planted, "join-null": join_null, "join-budget": join_budget,
    "translucent": translucent, "fw-bound": fw_bound, "free-track": free_track}

FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "join-planted": {"no-adjacency": lambda: _patched(gap_joins, "adjacency", _empty_adjacency),
                     "unsealed": lambda: _patched(gap_joins, "_regions", _unsealed)},
    "join-null": {"kept-null": lambda: _patched(search, "_joined", _kept_null)},
    "join-budget": {"no-giveback": lambda: _patched(search, "_joined", _no_giveback)},
    "translucent": {"stage-eps-kept": lambda: _patched(gcl.Relaxation, "stage",
                                                       _stage_eps_kept)},
    "fw-bound": {"empty-step": lambda: _patched(relax, "knapsack", _empty_step)},
    "free-track": {"first-only": lambda: _patched(relax.FreeTrack, "tracks", _first_track_only)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    try:
        if fault is None:
            bad = GUARDS[guard](ids)
        else:
            with FAULTS[guard][fault]():
                bad = GUARDS[guard](ids)
    except (RuntimeError, ValueError) as e:
        bad = [f"raised {e!r}"]
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], [] if a[1] == "-" else a[1].split(","), a[2] if len(a) > 2 else None)
