"""Less greedy searches around Shock Reblock (owner 2026-11-19: "Is it possible to test a less
greedy version that jointly solves for k whole roads at a time? Or a floating search
middle-ground?"). Lookahead (depth-k rollouts, receding-horizon or k ears at once) was tried and
deleted as dominated (no better than greedy beyond noise at 2.5-7.5x the time; the wiki's
reblock/lookahead-does-not-beat-shock-reblocks-greedy, commit 52a0b2ec for its code and runs).

Shock Reblock (ears.py's `paths` plan) builds one ear a step: the best exact gain per cost among
the step's `exact` rescored candidates (ears.offer, ears.ratio). Every search here reuses that step
unchanged (ears.offer: the state's costed candidates and the splitting ones rescored by the
restricted per-face solve) and differs only in which ears it commits (a Search Strategy):

  greedy       Greedy: the best ears.ratio, ears.build's pick (the check `greedy` compares
               the two runs' ears).
  float<r>     Floating (SFFS): greedy adds; after each, conditional exclusion (the removable ear,
               not the one just added, whose removal leaves a state above the frontier of the states
               the run has held, interpolated at its D as the readout is, the largest margin first;
               repeated), then up to
               r swap rounds (one removable ear out, the best rescored ear at the state without it
               in, by P_out - exact, kept if the global P falls and D does not rise).

An ear is removable when, without it, every other ear still splits its face (removing that one
too lowers the face count: none is left a dead end).

Readout (summary.py): L_P at matched D along the Pareto frontier of the states a run held (for
greedy, its trajectory), linearly interpolated.

Rows (rows/<block>/<search>/): trace.parquet, one row per state the search held (its ears, P, L_P,
D, C, the wall time, the action), and replay/ (crossings.py's layout, so crossings/analyze.py reads
it): ears.build over the ears of the frontier state with the largest D under the plan's D, in
their add order, recorded by crossings.Recording.

    L=research/roadless/atlas/lookahead/lookahead.py     # from the repo root, PYTHONPATH=.
    uv run python $L run <bid> <search> <plan>
    uv run python $L check greedy <bid> <plan> [fault]   # greedy's ears are ears.build's, e.g.
        ears.paths.<router>.b20.f2000.e30.loop.nodt.gl0.25Croad.n2s0.1.M8.K4.D1 (b0 or binf for
        visibility)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
ROADLESS = HERE.parents[1]
sys.path.insert(0, str(ROADLESS))
sys.path.insert(0, str(ROADLESS / "atlas" / "crossings"))
import crossings as cx  # noqa: E402
import ears  # noqa: E402
import growth as gr  # noqa: E402

ROWS = HERE / "rows"
TIE_L = 1e-9           # an improvement in L_P must exceed this (floating-point ties are not moves)
CHECK_STEPS = 4        # check greedy: ears compared


# ------------------------------------------------------------------ the search space

@dataclass(kw_only=True)
class Node:
    """A state: its ears (ids into Space.table, in add order), faces, credit scale, P, D, C; its
    solve (st, u) until its step is offered, the step (ears.Offered) after."""
    ears: tuple[int, ...]
    faces: ears.Face
    scale: float
    P: float
    D: float
    C: float
    st: Any
    u: NDArray[np.float64] | None
    off: ears.Offered | None


class Space:
    """The block's EarWorld and the ears any search built (`table`, an ear's id its index; its
    triangles g.step's mark)."""

    def __init__(self, e: ears.EarWorld, plan: ears.Plan, *, log: Any) -> None:
        self.e, self.plan, self.log = e, plan, log
        self.table: list[ears.Ear] = []

    def set(self, ids: tuple[int, ...]) -> None:
        g = self.e.g
        g.step[:] = -1
        for i in ids:
            g.step[self.table[i].tris] = i

    def node(self, ids: tuple[int, ...], *, scale: float) -> Node:
        self.set(ids)
        g = self.e.g
        st = g.state(g.step >= 0)
        u, P = g.solve(st)
        return Node(ears=ids, faces=self.e.faces(), scale=scale, P=P, D=g.D(), C=g.C(), st=st,
                    u=u, off=None)

    def offered(self, n: Node) -> ears.Offered:
        if n.off is None:
            self.set(n.ears)
            n.off = ears.offer(self.e, self.plan, n.faces, n.st, n.u, scale=n.scale, log=self.log)
            n.st, n.u = None, None
        return n.off

    def child(self, n: Node, s: ears.Scored) -> Node:
        """n with the rescored ear s (one of n's step's) built."""
        self.table.append(s.ear)
        return self.node(n.ears + (len(self.table) - 1,),
                         scale=ears.scale_of(self.offered(n).scored, scale=n.scale))

    def peek(self, ids: tuple[int, ...]) -> tuple[float, float]:
        """(P, D) of the state `ids`, its solve not kept."""
        self.set(ids)
        g = self.e.g
        return g.solve(g.state(g.step >= 0))[1], g.D()

    def D_of(self, ids: tuple[int, ...]) -> float:
        self.set(ids)
        return self.e.g.D()

    def removable(self, ids: tuple[int, ...], r: int) -> bool:
        """Whether ids less r leaves every other ear splitting its face."""
        rest = tuple(i for i in ids if i != r)
        self.set(rest)
        g = self.e.g
        n0 = self.e.faces().n
        for j in rest:
            g.step[self.table[j].tris] = -1
            n = self.e.faces().n
            g.step[self.table[j].tris] = j
            if not n < n0:
                return False
        return True


class Held(NamedTuple):
    """A state a search held, in order: its ears, P, L_P, D, C, the wall time, the action."""
    ears: tuple[int, ...]
    P: float
    L_P: float
    D: float
    C: float
    t: float
    action: str


class Trace:
    """The held states, written to `path` at each (a long run killed keeps its trace)."""

    def __init__(self, sp: Space, *, path: Path) -> None:
        self.sp, self.path = sp, path
        self.t0 = time.time()
        self.held: list[Held] = []

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([h._asdict() | {"ears": list(h.ears)} for h in self.held])

    def hold(self, n: Node, action: str) -> None:
        P0 = self.sp.e.g.P0
        h = Held(ears=n.ears, P=n.P, L_P=1.0 - n.P / P0, D=n.D, C=n.C, t=time.time() - self.t0,
                 action=action)
        self.held.append(h)
        self.frame().to_parquet(self.path)
        self.sp.log(f"{action:>5} {len(n.ears):3d} ears: L_P {h.L_P:.4f}, D {h.D:.4f} "
                    f"({h.t:.0f}s)")

    def best_at(self, D: float) -> float:
        """L_P at D along the frontier of the held states (`frontier`, linearly interpolated, as
        the readout reads it; its last L_P past its end)."""
        fr = frontier(pd.DataFrame({"D": [h.D for h in self.held],
                                    "L_P": [h.L_P for h in self.held]}))
        return float(np.interp(D, fr.D.to_numpy(), fr.L_P.to_numpy()))


# ------------------------------------------------------------------ searches

class Search(Protocol):
    @property
    def name(self) -> str: ...

    def run(self, sp: Space, root: Node, tr: Trace, *, budget: float) -> None:
        """Holds states (tr.hold) from root until one reaches D `budget` or no ear is left."""
        ...


@dataclass(frozen=True, kw_only=True)
class Greedy:
    """ears.build's pick: each step the best ears.ratio among the rescored candidates."""

    @property
    def name(self) -> str:
        return "greedy"

    def run(self, sp: Space, root: Node, tr: Trace, *, budget: float) -> None:
        while root.D < budget:
            sc = sp.offered(root).scored
            if not sc:
                sp.log("no candidate ear splits a face: stopped")
                return
            root = sp.child(root, max(sc, key=lambda s: ears.ratio(s.ear)))
            tr.hold(root, "add")


@dataclass(frozen=True, kw_only=True)
class Floating:
    """SFFS (module docstring), `rounds` swap rounds after each add."""
    rounds: int
    max_moves: int             # a guard: more moves than this raises (no cycle is expected)

    @property
    def name(self) -> str:
        return f"float{self.rounds}"

    def run(self, sp: Space, root: Node, tr: Trace, *, budget: float) -> None:
        moves = 0
        while root.D < budget:
            sc = sp.offered(root).scored
            if not sc:
                sp.log("no candidate ear splits a face: stopped")
                return
            root = sp.child(root, max(sc, key=lambda s: ears.ratio(s.ear)))
            tr.hold(root, "add")
            last = root.ears[-1]
            while True:                               # conditional exclusion
                outs = [tuple(i for i in root.ears if i != r)
                        for r in root.ears if r != last and sp.removable(root.ears, r)]
                PD = [sp.peek(ids) for ids in outs]
                margin = [(1.0 - P / sp.e.g.P0) - tr.best_at(D) for P, D in PD]
                if not outs or max(margin) <= TIE_L:
                    break
                root = sp.node(outs[int(np.argmax(margin))], scale=root.scale)
                tr.hold(root, "drop")
                moves += 1
            for _ in range(self.rounds):              # swaps
                best: tuple[float, Node, ears.Scored] | None = None
                for r in root.ears:
                    if not sp.removable(root.ears, r):
                        continue
                    out = sp.node(tuple(i for i in root.ears if i != r), scale=root.scale)
                    for s in sp.offered(out).scored:
                        if np.array_equal(s.ear.path, sp.table[r].path):
                            continue
                        P = out.P - s.ear.exact
                        sp.table.append(s.ear)
                        D = sp.D_of(out.ears + (len(sp.table) - 1,))
                        sp.table.pop()
                        if (D <= root.D and (root.P - P) / sp.e.g.P0 > TIE_L
                                and (best is None or P < best[0])):
                            best = (P, out, s)
                if best is None:
                    break
                n = sp.child(best[1], best[2])
                if not ((root.P - n.P) / sp.e.g.P0 > TIE_L and n.D <= root.D):
                    sp.log(f"  swap refused on the global solve (P {n.P:.6e} vs {root.P:.6e})")
                    break
                root = n
                tr.hold(root, "swap")
                moves += 1
            if moves > self.max_moves:
                raise RuntimeError(f"{self.name}: {moves} moves, more than {self.max_moves}")


def search_of(spec: str) -> Search:
    if spec == "greedy":
        return Greedy()
    if spec.startswith("float"):
        return Floating(rounds=int(spec[5:]), max_moves=1000)
    raise SystemExit(f"{spec}: a search is greedy | float<rounds>")


# ------------------------------------------------------------------ replay (crossings' layout)

class Replay:
    """A Candidates offering the given ears one per call, in order (their stored triangles,
    costed and first-order-scored at the replay's state), none after."""

    def __init__(self, inner: ears.Candidates, table: list[ears.Ear], ids: tuple[int, ...]) -> None:
        self.inner, self.table, self.ids = inner, table, ids
        self.k = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def pairs(self) -> int:
        return self.inner.pairs

    def ears(self, e: ears.EarWorld, faces: ears.Face, gain: Any, *, cost: gr.Cost,
             scale: float) -> list[ears.Ear]:
        if self.k == len(self.ids):
            return []
        q = self.table[self.ids[self.k]]
        self.k += 1
        g, ra = e.g, e.ra
        dD, dC = g.costs_of(q.tris)
        c = float(cost.of(np.array([dD]), np.array([dC]), P0=g.P0)[0])
        face = int(faces.lab[ra.tri[q.path[0]]])
        return [q._replace(face=face, dD=dD, dC=dC, cost=c, first=float(gain[q.tris].sum()),
                           exact=np.nan)]


def replay(b: Any, plan: ears.Plan, table: list[ears.Ear], ids: tuple[int, ...], *,
           out: Path, world: str) -> None:
    """ears.build over `ids` on a fresh world of block `b`, recorded as crossings.run records."""
    t0 = time.time()
    rec = cx.Recording(Replay(plan.candidates, table, ids))
    rp = replace(plan, candidates=rec, D=1e9)
    metric = ears.metrics.metric_of(ears.metrics.RUNTIME, device="cpu")
    e = ears.EarWorld(gr.growth_of(b, metric, contrast=plan.contrast))
    ears.build(e, rp, out=out, log=lambda s: None)
    n = len(pd.read_parquet(out / "picks.parquet"))
    if n != len(ids):
        raise RuntimeError(f"replay built {n} of {len(ids)} ears")
    ra = e.ra
    pd.DataFrame(rec.cands).to_parquet(out / "cands.parquet")
    st = rec.states
    np.savez_compressed(out / "states.npz",
                        pf=np.stack([s["pf"] for s in st]), sink=np.stack([s["sink"] for s in st]),
                        road=np.stack([s["road"] for s in st]), u=np.stack([s["u"] for s in st]),
                        shape=np.asarray(ra.shape), lo=ra.lo, tri=ra.tri, bld=ra.bld)
    (out / "world.json").write_text(json.dumps(dict(
        world=world, plan=plan.name, block_id=b.block_id,
        boundary=shapely.to_wkt(b.boundary), n_buildings=len(b.building_geometries),
        coverage=float(b.building_geometries.area.sum() / b.boundary.area),
        t=time.time() - t0)))


# ------------------------------------------------------------------ run

def frontier(held: pd.DataFrame) -> pd.DataFrame:
    """The held states no other beats (less or equal D, more L_P), by D."""
    h = held.sort_values(["D", "L_P"], ascending=[True, False])
    return h[h.L_P > h.L_P.cummax().shift(fill_value=-np.inf)]


def run(bid: str, search: Search, plan: ears.Plan) -> None:
    t0 = time.time()
    b = cx.RealBlock(bid=bid).block()
    metric = ears.metrics.metric_of(ears.metrics.RUNTIME, device="cpu")
    e = ears.EarWorld(gr.growth_of(b, metric, contrast=plan.contrast))
    out = ROWS / bid / f"{search.name}.{plan.name}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{bid} {search.name} {plan.name}: {e.g.nt} triangles, raster {e.ra.shape} "
          f"({time.time() - t0:.0f}s)", flush=True)
    sp = Space(e, plan, log=lambda s: print(s, flush=True))
    tr = Trace(sp, path=out / "trace.parquet")
    root = sp.node((), scale=0.0)
    tr.hold(root, "start")
    search.run(sp, root, tr, budget=plan.D)
    held = tr.frame()
    print(f"search done: {len(tr.held)} states, {tr.held[-1].t:.0f}s", flush=True)
    fr = frontier(held)
    under = fr[fr.D < plan.D]
    ids = tuple(int(i) for i in under.iloc[-1].ears)
    replay(b, plan, sp.table, ids, out=out / "replay", world=f"block:{bid}:{search.name}")
    print(f"replayed {len(ids)} ears (D {under.iloc[-1].D:.4f}); done ({time.time() - t0:.0f}s)",
          flush=True)


def check_greedy(bid: str, plan: ears.Plan, fault: str | None) -> None:
    """greedy's first CHECK_STEPS ears (paths' pixels) are ears.build's on the plan (any router).
    Fault `second` (greedy taking each step's second-best rescored ear) fails it."""
    if fault not in (None, "second"):
        raise SystemExit("greedy's fault: second")
    e = ears.world_of(bid, plan.contrast.h)
    out = ROWS / "check" / bid / plan.name
    ears.build(e, plan, out=out, log=lambda s: None, steps=CHECK_STEPS)
    built = pd.read_parquet(out / "picks.parquet")
    e2 = ears.world_of(bid, plan.contrast.h)
    sp = Space(e2, plan, log=lambda s: None)
    n = sp.node((), scale=0.0)
    ok = True
    for k in range(CHECK_STEPS):
        sc = sorted(sp.offered(n).scored, key=lambda s: -ears.ratio(s.ear))
        s = sc[1] if fault == "second" and len(sc) > 1 else sc[0]
        xy = e2.ra.xy[s.ear.path]
        r = built.iloc[k]
        same = bool(np.allclose([xy[0, 0], xy[0, 1], xy[-1, 0], xy[-1, 1]],
                                [r.ax, r.ay, r.bx, r.by]) and np.isclose(s.ear.exact, r.exact))
        ok &= same
        print(f"  ear {k}: {'same' if same else 'DIFFERS'} (exact {s.ear.exact:.6e} vs "
              f"{r.exact:.6e})")
        n = sp.child(n, s)
    print(f"greedy {bid} {fault or 'none'}: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def main(argv: list[str]) -> None:
    if len(argv) == 4 and argv[0] == "run":
        run(argv[1], search_of(argv[2]), ears.Plan.of(argv[3]))
    elif len(argv) in (4, 5) and argv[:2] == ["check", "greedy"]:
        check_greedy(argv[2], ears.Plan.of(argv[3]), argv[4] if len(argv) == 5 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
