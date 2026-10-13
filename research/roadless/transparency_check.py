"""Guards of the road-transparency code (BACKLOG 5) that need the real blocks and stored rows; the
rest are unit tests (tests/test_transparency.py).

    S=research/roadless/transparency_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S opaque <out.csv> <procs> <ids,> <methods,>
    uv run python $S sets <ids,> <seeds>
    uv run python $S rule
    uv run python $S held
    uv run python $S claim
    uv run python $S hold-nothing <out.csv> <procs> <ids,>
    uv run python $S no-enclosed <out.csv> <procs> <ids,>
    uv run python $S free-roads <out.csv> <procs> <ids,>
    uv run python $S lead <stage> <out.csv> <procs> <ids,>

`opaque` (guard 1): each method as configured (compare_config's all_methods, common.arms()), with
`Opaque` for the price and `TransparentSubstrate(<its substrate>, Opaque())` for the route, run
fresh (the method's own `propose`, never the derive cache; clearance_looped's base tree too, handed
to the refiner as its `prior`), against item 4's stored network in road_rows/scratch/networks.
The same WKB, row for row (every coordinate exact), and the same widths; one row per job to
<out.csv>, with its wall and CPU seconds.

`sets` (guard 4), on both stored clearings (transparency_sets.SIMP_DEFAULT and CHEAP) of every block
in <ids,>, with the experiment's stratification (STRATIFICATION, equal area mass in 20 strata): the
cleared set's members are the stored row's positions, read again here, and its factors f there and
1 elsewhere; the matched random set has the clearing's count in every stratum (the strata computed
here, not by transparency_sets), the same members for the same seed and different members for
different seeds, and its `Draw` records its area over the clearing's and its Jaccard overlap with
the clearing as computed here; and every identity moves with the clearing's content, the factor,
the seed and the stratification. Then the area ratio's and the overlap's median, p90 and worst
over every (block, clearing, seed).

`rule`: the 13's choice rule (transparency_run.Choice.decide; NOTES, "Transparency for the road
methods: the 13 tuning blocks") on synthetic lens tables, no result read: resistance_lp's arms on
13 made-up blocks, each case one condition of the rule -- the largest qualifying candidate chosen,
with base, the best knob and the random sets in its form going on; behind base on B (i) on one
block, or more than 0.1 below it on (c), vetoes; a median (a) loss over 0.01 rules out; a knob
ahead of every candidate stops the method; a candidate below its own random sets (its form's, not
the other form's) is out; with every knob losing, a losing candidate above it is chosen (as the
rule is written).

`held`: the held-out blocks (transparency_run.HELD) derived again: item 4's 60 (its committed lens
rows' blocks, the same as its projection's blocks under 48 GB at h 0.5 and its networks'
directories), less simp_compare.TUNING, less the blocks without both stored clearings at h 0.5
(14401); and item 4's network of each plan method's preset on every one of them (check-base's).

`claim`: the held-out blocks' rules (transparency_run.Claim.judge and .verdict; NOTES,
"Transparency for the road methods: the held-out blocks") on synthetic lens tables over 20
made-up blocks, no result read, both methods of the plan file: every rule holding is a shift (the
interval rule 1's bootstrap over W, paired); the claim behind base on B (i) on one block, or its
gains straddling 0, fails rule 1; a knob behind base on B (i) on three blocks, where it is far
ahead on (c), is left out there and rule 2 holds against it on the other 17 (its interval brackets
the mean over them); a knob ahead of the claim fails rule 2 though the other knob is beaten; a
knob behind base everywhere leaves nothing to compare, and rule 2 does not hold against it; random
sets as good as the claim fail rule 3; a median (a) loss of 0.011 at count D 0.10 alone fails
rule 4.

The free-first stage (free_roads.FreeEnclosed; transparency_run's `free` stage), its guards on the
real blocks, each writing one row per job to <out.csv> and raising on any failure:

`hold-nothing` (guard 1): every arm of the held stage, as configured (its tree's or its own
`free_first` checked to be HoldNothing), proposed fresh here and set against the held stage's
network of record (computed on the cluster's nodes before free first existed): the same frame --
columns, WKB row for row, widths.

`no-enclosed` (guard 2), on blocks with no home enclosed at baseline: FREE's plan finds none and
holds nothing on each plan method's own graph, and every arm of the free stage (FREE on every arm)
proposed fresh is the held stage's network of record, frame for frame.

`free-roads` (guard 3), on blocks with homes enclosed at baseline: FREE's plan on each plan method's
graph and on its claim's routed one (TransparentSubstrate: the same nodes and edges), the same roads
on all; then the scorer itself (common.Scorer at the stage's mesh, on the CPU) finds the same
enclosed homes, each held road frees its home (and every home its step says it frees) once the
roads up to it are carved, what the roads leave enclosed is the plan's share, and their true
displacement under both weightings is the plan's and within its budget.

`lead <stage>` (guard 4), on a free stage's networks, ordered networks and prefix rows: every
network holds FREE's roads first (its marked leading rows, WKB row for row), its ordered network
leads with them, every prefix row at D LENS_D (count and area) holds them all, and on each of
those prefixes the scorer finds no home enclosed but those the plan left (unreachable, or over the
budget).

FAULT INJECTION, the free-first guards (each a patch in one process, no file edited):
`hold-nothing`: HoldNothing holding one road (its graph's first edge) fails it on all 17 arms of
6310. `no-enclosed`: the stage judging enclosure at a 0.1 m ring (the scorer's is lifted.RING_M,
1 m), under which it finds 2 homes enclosed on 6310 and 1 on 40144 where the scorer finds none,
fails it: it holds roads for them, and all 34 networks differ. `free-roads`: every held road one
segment short makes the stage itself raise, its road not freeing its home by the metric's test, on
all ten blocks. `lead`: the free-local stage's 51 networks on 41673, 21252 and 43708 re-cut with
the held roads where street_first_ordered puts them (common.canonical_order replaced by it) fail
it on 41: the held roads lead 10, every held road is inside the D 0.05 prefixes of 18, and the
D 0.05 prefixes of 31 leave homes enclosed that the held roads free (all of 21252's two in every
clearance_looped arm).

FAULT INJECTION, `rule`: the veto without B (i), and the random sets of every form for each
candidate, each fail it. `claim`: rule 2 over every block (the knob's B (i) ignored), rule 2
holding when any knob is beaten, rule 1 without B (i), and the kept blocks' weights not
renormalized each fail it. `sets` on six blocks: positions read off by one, draws that ignore the
strata, the seed ignored, a ClearedSet identity without the positions' digest, and AreaMass cutting
its strata by count each fail it. `opaque`: the brief's literal route formula (outside length from
the union, inside length per building), under which a stretch inside two overlapping opaque
footprints weighs more than open ground, fails it.
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

from reblock.contracts import Block, Method
from reblock.free_first import HoldNothing, held_count
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.cycle_native import CycleNativeReblocker
from reblock.methods.loop_closure import LoopClosureRefiner
from reblock.methods.resistance_lp import ResistanceLPReblocker
from reblock.methods.substrates import RoutingGraph
from reblock.permeability import with_width
from reblock.transparency import Opaque, TransparentSubstrate, factors_of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import free_roads as fr  # noqa: E402
import lifted  # noqa: E402
import transparency_sets as ts  # noqa: E402

NETS = HERE / "road_rows" / "scratch" / "networks"
SINGLE = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
# propose order only (no result depends on it): the costliest first
COST_GUESS = {"cycle_native": 2.5, "resistance_lp": 2.0, "clearance_looped": 1.0}
STRATIFICATION = ts.AreaMass(strata=20)     # the experiment's control (owner's ruling 2026-10-12)
J = TypeVar("J")                            # a pool's job
F_CHECK = 0.25              # any factor in (0, 1) does for these checks


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {SINGLE}")


# ------------------------------------------------------------------ opaque (guard 1)

Routed = CycleNativeReblocker | ResistanceLPReblocker | ClearanceReblocker


def _routed(m: Routed, name: str) -> Routed:
    if not isinstance(m.price, Opaque):
        raise AssertionError(f"{name}: configured with price {m.price!r}, not Opaque")
    return dataclasses.replace(
        m, substrate=TransparentSubstrate(base=m.substrate, transparency=Opaque()))


def opaque_arm(name: str) -> Routed | LoopClosureRefiner:
    """The configured arm `name`, its route made explicit: `TransparentSubstrate(substrate,
    Opaque())` (identical weights), its price checked to be `Opaque`."""
    m: Method = common.arms()[name]
    if isinstance(m, LoopClosureRefiner) and isinstance(m.base, ClearanceReblocker):
        return dataclasses.replace(m, base=_routed(m.base, name))
    if isinstance(m, CycleNativeReblocker | ResistanceLPReblocker):
        return _routed(m, name)
    raise TypeError(f"{name} is not one of the methods with a price")


def _roads(method: Method, b: Block) -> gpd.GeoDataFrame:
    """`method`'s roads on `b`, fresh: a refiner's base too, handed to it as its prior."""
    roads = (method.propose(b, prior=method.base.propose(b)).roads
             if isinstance(method, LoopClosureRefiner) else method.propose(b).roads)
    if roads is None:
        raise ValueError(f"{b.block_id}: no road frame")
    return roads


def _wkb(roads: gpd.GeoDataFrame) -> list[bytes]:
    return [bytes(w) for w in shapely.to_wkb(np.asarray(roads.geometry))]


def _opaque_one(job: tuple[str, str]) -> dict[str, object]:
    bid, name = job
    t_all = time.perf_counter()
    [b] = common.build_blocks([bid])
    method = opaque_arm(name)
    t0, c0 = time.perf_counter(), time.process_time()
    roads = _roads(method, b)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    stored = gpd.read_parquet(NETS / bid / f"{name}.parquet")
    a, g = list(stored.geometry), list(roads.geometry)
    same_n = len(a) == len(g)
    same = bool(same_n and _wkb(stored) == _wkb(roads)
                and np.array_equal(stored["width_m"].to_numpy(), roads["width_m"].to_numpy()))
    worst = (float(max(x.hausdorff_distance(y) for x, y in zip(a, g, strict=True)))
             if same_n and len(a) else np.nan)
    row = dict(block=bid, n=len(b.buildings), method=name, roads=len(g), stored_roads=len(a),
               same=same, max_hausdorff_m=worst, wall_s=wall, cpu_s=cpu,
               total_wall_s=time.perf_counter() - t_all)
    print(f"{time.strftime('%H:%M:%S')} {bid} {name}: {len(g)} roads (stored {len(a)}), "
          f"same {same}, {wall:.0f}s wall, {cpu:.0f}s CPU", flush=True)
    return row


def opaque(out: Path, procs: int, ids: list[str], methods: list[str]) -> None:
    _check_single_threaded()
    sizes = pd.read_parquet(HERE / "large82_area.parquet").set_index("block")["n"]
    jobs = sorted(((bid, name) for bid in ids for name in methods),
                  key=lambda j: -float(sizes[j[0]]) * COST_GUESS[j[1]])
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(min(procs, len(jobs)), maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(_opaque_one, jobs))
    df = pd.DataFrame(rows).sort_values(["method", "n"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"{len(df)} jobs, {procs} processes, {time.perf_counter() - t0:.0f}s wall; "
          f"CPU {df.cpu_s.sum():.0f}s; all identical: {bool(df.same.all())}")
    if not df.same.all():
        raise SystemExit("an opaque run differs from its stored network")


# ------------------------------------------------------------------ sets (guard 4)

def _stored_positions(rows: ts.StoredRows, bid: str) -> NDArray[np.int64]:
    """The row's positions, read independently of transparency_sets."""
    r = pd.read_parquet(HERE / rows.rows / f"{bid}_p2.parquet")
    return np.sort(np.asarray(r[rows.column].iloc[0], dtype=np.int64))


def _mass_strata(area: NDArray[np.float64], strata: int) -> NDArray[np.intp]:
    """Strata of equal total area, computed here rather than by transparency_sets: by area,
    smallest first, a building in the stratum where the running total passes its middle."""
    order = np.argsort(area, kind="stable")
    total = float(area[order].sum())
    out = np.empty(len(area), dtype=np.intp)
    running = 0.0
    for i in order:
        running += float(area[i])
        out[i] = int((running - area[i] / 2.0) / total * strata)
    return out


def sets(ids: list[str], seeds: list[int]) -> None:
    shares = []
    k = STRATIFICATION.strata
    for b in common.build_blocks(ids):
        area = np.asarray([g.area for g in b.buildings.outlines])
        stratum = _mass_strata(area, k)
        for kind, rows in (("simp_default", ts.SIMP_DEFAULT), ("cheap", ts.CHEAP)):
            if not (HERE / rows.rows / f"{b.block_id}_p2.parquet").exists():
                continue
            c = ts.Clearing.read(rows, b.block_id)
            want = _stored_positions(rows, b.block_id)
            cleared = ts.ClearedSet(clearing=c, f=F_CHECK)
            assert np.array_equal(cleared.members(b), want), (b.block_id, kind)
            f = factors_of(cleared, b)
            assert np.all(f[want] == F_CHECK) and np.all(np.delete(f, want) == 1.0)
            drawn = {}
            for seed in seeds:
                r = ts.MatchedRandomSet(clearing=c, stratification=STRATIFICATION, seed=seed,
                                        f=F_CHECK)
                d = r.draw(b)
                m = d.members
                again = ts.MatchedRandomSet(clearing=ts.Clearing.read(rows, b.block_id),
                                            stratification=STRATIFICATION, seed=seed,
                                            f=F_CHECK).members(b)
                assert np.array_equal(m, again), (b.block_id, kind, seed, "not reproducible")
                assert len(m) == len(want) == len(np.unique(m)), (b.block_id, kind, seed)
                assert np.array_equal(np.bincount(stratum[m], minlength=k),
                                      np.bincount(stratum[want], minlength=k))
                ratio = area[m].sum() / area[want].sum()
                jaccard = len(set(m.tolist()) & set(want.tolist())) / len(
                    set(m.tolist()) | set(want.tolist()))
                assert np.isclose(d.area_ratio, ratio, rtol=1e-12, atol=0.0)
                assert d.jaccard == jaccard
                f = factors_of(r, b)
                assert np.all(f[m] == F_CHECK) and np.all(np.delete(f, m) == 1.0)
                drawn[seed] = m
                shares.append(dict(block=b.block_id, kind=kind, seed=seed, count=len(m),
                                   ratio=ratio, jaccard=jaccard))
            for s1 in seeds:
                for s2 in seeds:
                    if s1 < s2 and np.array_equal(drawn[s1], drawn[s2]):
                        raise AssertionError(f"{b.block_id} {kind}: seeds {s1} and {s2} agree")
            # identities move with what changes the factors, and only with that
            moved = dataclasses.replace(c, cleared=(*c.cleared[:-1], c.cleared[-1] + 1))
            base = ts.MatchedRandomSet(clearing=c, stratification=STRATIFICATION, seed=seeds[0],
                                       f=F_CHECK)
            for other in (dataclasses.replace(base, clearing=moved),
                          dataclasses.replace(base, seed=seeds[0] + 1),
                          dataclasses.replace(base, f=F_CHECK / 2),
                          dataclasses.replace(base, stratification=ts.AreaMass(strata=k + 1)),
                          dataclasses.replace(base, stratification=ts.CountQuantiles(strata=k))):
                assert other.identity != base.identity and other.tag != base.tag
            assert ts.ClearedSet(clearing=moved, f=F_CHECK).identity != cleared.identity
            assert ts.ClearedSet(clearing=c, f=F_CHECK / 2).identity != cleared.identity
            reread = ts.Clearing.read(rows, b.block_id)
            assert ts.ClearedSet(clearing=reread, f=F_CHECK).identity == cleared.identity
        print(f"{b.block_id}: ok", flush=True)
    df = pd.DataFrame(shares)
    off = (df.ratio - 1.0).abs()
    print(f"{df.block.nunique()} blocks, {len(df)} random sets, {STRATIFICATION}: count = the "
          f"clearing's in every stratum on all")
    print(f"area ratio (random / clearing): median {df.ratio.median():.4f}, p10 "
          f"{df.ratio.quantile(0.1):.4f}, p90 {df.ratio.quantile(0.9):.4f}, "
          f"min {df.ratio.min():.4f}, max {df.ratio.max():.4f}; |ratio - 1| median "
          f"{off.median():.4f}, p90 "
          f"{off.quantile(0.9):.4f}, worst {off.max():.4f}")
    print(f"Jaccard overlap with the clearing: median {df.jaccard.median():.4f}, p90 "
          f"{df.jaccard.quantile(0.9):.4f}, worst {df.jaccard.max():.4f}")
    for what, row in (("area", df.loc[off.idxmax()]), ("overlap", df.loc[df.jaccard.idxmax()])):
        print(f"worst {what}: {row.block} {row.kind} seed {row.seed}: ratio {row.ratio:.4f}, "
              f"Jaccard {row.jaccard:.4f} ({row['count']} buildings)")
    per = df.groupby(["block", "kind"]).ratio.apply(lambda r: (r - 1.0).abs().max())
    print(f"(block, clearing) pairs whose worst seed is off by more than 0.25: "
          f"{int((per > 0.25).sum())} of {len(per)}, more than 0.10: {int((per > 0.10).sum())}")


def rule() -> None:
    """The 13's choice rule on synthetic lens tables (the module's docstring)."""
    import transparency_run as tr
    m = next(x for x in tr.TUNING_PLAN if x.method == "resistance_lp")
    blocks = tuple(f"synthetic_{i}" for i in range(13))
    simp = pd.Series(0.5, index=list(blocks))
    short = {a.name: a.change.label for a in m.arms}
    by = {v: k for k, v in short.items()}

    def lens_of(c: dict[str, NDArray[np.float64]],
                enc: dict[str, NDArray[np.float64]] | None = None,
                a: dict[str, NDArray[np.float64]] | None = None) -> dict[float, pd.DataFrame]:
        rows = [dict(block=b, method=n, c=c.get(short[n], c["base"])[i],
                     enc_c=(enc or {}).get(short[n], np.zeros(13))[i],
                     a=(a or {}).get(short[n], np.full(13, 0.8))[i])
                for n in short for i, b in enumerate(blocks)]
        return {d: pd.DataFrame(rows) for d in tr.GUARD_D}

    W = tr.boot_weights(len(blocks))
    base = np.full(13, 0.10)
    c = {"base": base, "chunks4": base + 0.002, "chunks16": base - 0.003,
         "simp-f0.1-price": base + 0.010, "simp-f0.1-price-route": base + 0.015,
         "cheap-f0.1-price": base + 0.008,
         **{f"rand{s}-f0.1-{w}": base + 0.001 for s in tr.SEEDS for w in ("price", "price-route")}}

    def decide(c: dict[str, NDArray[np.float64]], **kw: dict[str, NDArray[np.float64]]
               ) -> tuple[dict[str, dict[str, object]], str, str | None]:
        st, knob, chosen = tr.Choice.decide(m, lens_of(c, **kw), simp, W)
        return st, short[knob.name], None if chosen is None else short[chosen.name]

    st, knob, chosen = decide(c)
    r = st[by["simp-f0.1-price-route"]]
    assert knob == "chunks4" and chosen == "simp-f0.1-price-route", (knob, chosen)
    assert abs(float(r["mean_gain"]) - 0.015) < 1e-12 and r["blocks_up"] == 13
    assert float(r["lo"]) <= 0.015 <= float(r["hi"])
    assert abs(float(r["gap_closed"]) - 0.015 / 0.4) < 1e-12
    assert {short[n] for n, x in st.items() if x["goes_on"]} == {
        "base", "chunks4", "simp-f0.1-price-route",
        *[f"rand{s}-f0.1-price-route" for s in tr.SEEDS]}
    print("the largest qualifying candidate chosen; base, the best knob, its form's random sets on")
    st, _, chosen = decide(c, enc={"simp-f0.1-price-route": np.r_[0.001, np.zeros(12)]})
    assert st[by["simp-f0.1-price-route"]]["veto"] and chosen == "simp-f0.1-price", chosen
    print("behind base on B (i) on one block: vetoed, the next candidate chosen")
    low = np.r_[base[:12] + 0.03, base[12] - 0.11]
    st, _, chosen = decide({**c, "simp-f0.1-price-route": low})
    assert st[by["simp-f0.1-price-route"]]["veto"] and chosen == "simp-f0.1-price", chosen
    print("more than 0.1 below base on (c) on one block: vetoed")
    st, _, chosen = decide(c, a={"simp-f0.1-price-route": np.full(13, 0.8 - 0.011)})
    assert not st[by["simp-f0.1-price-route"]]["a_ok"] and chosen == "simp-f0.1-price", chosen
    print("a median (a) loss of 0.011: out")
    st, knob, chosen = decide({**c, "chunks16": base + 0.02})
    assert knob == "chunks16" and chosen is None and not any(x["goes_on"] for x in st.values())
    print("a knob ahead of every candidate: none chosen, the method stops")
    st, _, chosen = decide({**c, **{f"rand{s}-f0.1-price-route": base + 0.02 for s in tr.SEEDS}})
    assert not st[by["simp-f0.1-price-route"]]["above_own_random"]
    assert st[by["simp-f0.1-price"]]["above_own_random"] and chosen == "simp-f0.1-price", chosen
    print("below its own form's random sets: out (the other form's arm is not held to them)")
    losing = {**c, "chunks4": base - 0.002, "chunks16": base - 0.004,
              "simp-f0.1-price": base - 0.001, "simp-f0.1-price-route": base - 0.003,
              "cheap-f0.1-price": base - 0.005,
              **{f"rand{s}-f0.1-price": base - 0.0015 for s in tr.SEEDS}}
    st, _, chosen = decide(losing)
    assert chosen == "simp-f0.1-price" and float(st[by[chosen]]["mean_gain"]) < 0
    print("every knob losing: a losing candidate above the best knob and its random sets chosen")


def held() -> None:
    """The held-out blocks derived again, against transparency_run.HELD (the module's docstring)."""
    import road_rescore as rr
    import simp_compare
    import transparency_run as tr
    sixty = set(pd.read_csv(rr.summary_path(tr.ITEM4_ROWS, tr.MESH, "lens_D0.05")).block)
    proj = pd.read_parquet(tr.ITEM4_ROWS / "projection_h0.5_h0.5a5x8.parquet")
    fits = set(proj[proj.gb_uniform < 48.0].block)
    nets = {p.name for p in NETS.iterdir() if p.is_dir()}
    assert len(sixty) == 60 and sixty == fits == nets, (len(sixty), len(fits), len(nets))
    tuning = set(simp_compare.TUNING)
    assert len(tuning) == 13 and tuning <= sixty
    rest = sixty - tuning
    stored = (ts.SIMP_DEFAULT, ts.CHEAP)
    absent = sorted(b for b in rest
                    if not all((HERE / r.rows / f"{b}_p2.parquet").exists() for r in stored))
    held = tuple(sorted(rest - set(absent)))
    assert held == tr.HELD and len(set(tr.HELD)) == len(tr.HELD), (
        f"HELD differs: {sorted(set(held) ^ set(tr.HELD))}")
    print(f"item 4's {len(sixty)} blocks (its lens rows = its projection's fits = its networks), "
          f"less the 13: {len(rest)}; without both stored clearings at h 0.5: {absent}; held "
          f"out: {len(held)}, transparency_run.HELD")
    plan = tr.held_plan(tr.HELD_PLAN)
    for m in plan:
        missing = [b for b in held if not (NETS / b / f"{m.base.preset}.parquet").exists()]
        assert not missing, f"{m.method}: item 4 has no network on {missing}"
    print(f"the plan's {len(plan)} methods ({', '.join(m.method for m in plan)}): item 4's "
          f"network of each base's preset on all {len(held)}")


def claim() -> None:
    """The held-out blocks' rules on synthetic lens tables (the module's docstring)."""
    import transparency_run as tr
    plan = {m.method: m for m in tr.held_plan(tr.HELD_PLAN)}
    n = 20
    blocks = tuple(f"synthetic_{i:02d}" for i in range(n))
    simp = pd.Series(0.5, index=list(blocks))
    W = tr.boot_weights(n)
    base = np.full(n, 0.10)
    wiggle = 0.004 * np.sin(np.arange(n, dtype=float))     # so the intervals have width
    zero = np.zeros(n)
    Arr = NDArray[np.float64]

    def judge(m: tr.ClaimArms, c: dict[str, Arr], enc: dict[str, Arr] | None = None,
              a: dict[tuple[str, float], Arr] | None = None
              ) -> tuple[dict[tuple[int, str], tr.RuleRow], str]:
        short = {x.name: x.change.label for x in m.arms}
        lens = {d: pd.DataFrame([dict(block=b, method=name, c=c.get(lab, base)[i],
                                      enc_c=(enc or {}).get(lab, zero)[i],
                                      a=(a or {}).get((lab, d), np.full(n, 0.8))[i])
                                 for name, lab in short.items() for i, b in enumerate(blocks)])
                for d in tr.GUARD_D}
        tests, _ = tr.Claim.judge(m, lens, simp, W)
        return {(t.rule, t.against): t for t in tests}, tr.Claim.verdict(tests)

    lp, cl = plan["resistance_lp"], plan["clearance_looped"]
    good = {"simp-f0.1-price": base + 0.02 + wiggle, "chunks4": base + 0.005 + wiggle / 2,
            "chunks16": base + 0.004,
            **{f"rand{s}-f0.1-price": base + 0.001 + wiggle / 3 for s in tr.SEEDS}}
    t, v = judge(lp, good)
    r1 = t[(1, "base")]
    boot = W @ (0.02 + wiggle)
    assert v.startswith("a shift") and all(x.holds for x in t.values()), v
    assert abs(r1.value - float(np.mean(0.02 + wiggle))) < 1e-12 and r1.behind_i == 0
    assert abs(r1.lo - np.quantile(boot, 0.025)) < 1e-12 < r1.lo
    assert abs(r1.hi - np.quantile(boot, 0.975)) < 1e-12
    assert t[(2, "chunks4")].blocks == t[(2, "chunks16")].blocks == n
    print("every rule holding: a shift (rule 1's interval the bootstrap over W, paired)")
    t, v = judge(lp, good, enc={"simp-f0.1-price": np.r_[0.001, np.zeros(n - 1)]})
    assert not t[(1, "base")].holds and t[(1, "base")].behind_i == 1 and v == "no shift: rule 1 " \
        "fails", v
    print("the claim behind base on B (i) on one block: rule 1 fails")
    t, v = judge(lp, {**good, "simp-f0.1-price": base + 0.02 * (-1.0) ** np.arange(n)})
    assert not t[(1, "base")].holds and t[(1, "base")].lo < 0 < t[(1, "base")].hi, v
    print("gains straddling 0: rule 1 fails")
    walled = np.r_[np.ones(3), np.zeros(n - 3)].astype(bool)
    k4 = np.where(walled, base + 0.30, base + 0.005 + wiggle / 2)
    t, v = judge(lp, {**good, "chunks4": k4}, enc={"chunks4": np.where(walled, 0.001, 0.0)})
    r2 = t[(2, "chunks4")]
    mean_kept = float(np.mean((0.02 + wiggle - (k4 - base))[~walled]))
    assert r2.holds and r2.blocks == n - 3 and r2.behind_i == 3, r2
    assert abs(r2.value - mean_kept) < 1e-12 and r2.lo <= r2.value <= r2.hi, r2
    assert v.startswith("a shift"), v
    print("a knob behind base on B (i) on 3 blocks, far ahead on (c) there: left out, rule 2 "
          "holds over the other 17")
    t, v = judge(lp, {**good, "chunks16": base + 0.03 + wiggle})
    assert t[(2, "chunks4")].holds and not t[(2, "chunks16")].holds and v == "no shift: rule " \
        "2 fails", v
    print("a knob ahead of the claim: rule 2 fails, the other knob beaten notwithstanding")
    t, v = judge(lp, good, enc={"chunks16": np.full(n, 0.001)})
    assert t[(2, "chunks16")].blocks == 0 and not t[(2, "chunks16")].holds, t[(2, "chunks16")]
    print("a knob behind base on every block: nothing to compare, rule 2 does not hold")
    t, v = judge(lp, {**good, **{f"rand{s}-f0.1-price": good["simp-f0.1-price"]
                                 + 0.003 * (-1.0) ** np.arange(n) for s in tr.SEEDS}})
    r3 = t[(3, "random sets")]
    assert not r3.holds and r3.lo < 0 < r3.hi and v == "no shift: rule 3 fails", (r3, v)
    print("random sets as good as the claim: rule 3 fails")
    t, v = judge(lp, good, a={("simp-f0.1-price", 0.1): np.full(n, 0.8 - 0.011)})
    assert t[(4, "(a) at count D 0.05")].holds and not t[(4, "(a) at count D 0.1")].holds
    assert v == "no shift: rule 4 fails", v
    print("a median (a) loss of 0.011 at count D 0.10 alone: rule 4 fails")
    t, v = judge(cl, {"simp-f0.1-price-route": base + 0.02 + wiggle, "repulsion4": base + 0.003,
                      **{f"rand{s}-f0.1-price-route": base + wiggle / 3 for s in tr.SEEDS}})
    assert v.startswith("a shift") and [k for k in t if k[0] == 2] == [(2, "repulsion4")], v
    print("clearance_looped, one knob: every rule holding, a shift")


# ------------------------------------------------------------------ free first (BACKLOG 5)

def _same_frame(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> bool:
    """Two networks as frames: the same columns, the same WKB row for row (every coordinate
    exact), the same widths, the same held roads."""
    return bool(list(a.columns) == list(b.columns) and len(a) == len(b)
                and _wkb(a) == _wkb(b)
                and np.array_equal(a["width_m"].to_numpy(), b["width_m"].to_numpy())
                and held_count(a) == held_count(b))


def _against_held(job: tuple[str, str, str]) -> dict[str, object]:
    """One (stage, block, arm): the arm's network proposed fresh here, against the held stage's
    network of record (computed before free first existed, every arm holding nothing)."""
    import transparency_run as tr
    stage_name, bid, name = job
    stage, held_stage = tr._stage(stage_name), tr.held_stage()
    arm = stage.arm_of(name)
    [b] = common.build_blocks([bid])
    method = stage.method(arm, common.arms(), bid)
    first = tr._tree(method).free_first
    t0, c0 = time.perf_counter(), time.process_time()
    roads = _roads(method, b)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    theirs = gpd.read_parquet(held_stage.record_path(bid, name))
    same = _same_frame(roads, theirs)
    print(f"{time.strftime('%H:%M:%S')} {bid} {name} ({type(first).__name__}): {len(roads)} roads "
          f"(held stage {len(theirs)}), held {held_count(roads)}, same {same}, {wall:.0f}s",
          flush=True)
    return dict(stage=stage_name, block=bid, arm=name, first=type(first).__name__,
                hold_nothing=isinstance(first, HoldNothing), roads=len(roads),
                held_stage_roads=len(theirs), held=held_count(roads),
                columns=",".join(roads.columns),
                same=same, wall_s=wall, cpu_s=cpu)


def _pool_rows(fn: Callable[[J], dict[str, object]], jobs: list[J], procs: int) -> pd.DataFrame:
    with multiprocessing.get_context("fork").Pool(min(procs, len(jobs)), maxtasksperchild=1) as p:
        return pd.DataFrame(list(p.imap_unordered(fn, jobs)))


def _jobs(stage_name: str, ids: list[str]) -> list[tuple[str, str, str]]:
    """Every (block, arm) of the stage on `ids`, the costliest first (propose's order guess)."""
    import transparency_run as tr
    stage = tr._stage(stage_name)
    sizes = pd.read_parquet(HERE / "large82_area.parquet").set_index("block")["n"]
    jobs = [(float(sizes[bid]) * COST_GUESS[a.group], (stage_name, bid, a.name))
            for bid in stage.ids_of(",".join(ids)) for a in stage.arms]
    return [j for _, j in sorted(jobs, key=lambda c: -c[0])]


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df.to_string(index=False))


def hold_nothing(out: Path, procs: int, ids: list[str]) -> None:
    """Guard 1 (the module's docstring)."""
    _check_single_threaded()
    df = _pool_rows(_against_held, _jobs("held", ids), procs).sort_values(["block", "arm"])
    _write(df, out)
    bad = df[~(df.same.astype(bool) & df.hold_nothing.astype(bool))]
    print(f"{len(df)} networks on {df.block.nunique()} blocks, holding nothing: "
          f"{int(df.hold_nothing.sum())}; the held stage's network, frame for frame: "
          f"{int(df.same.sum())}")
    if len(bad):
        raise SystemExit(f"{len(bad)} networks are not the held stage's: {bad.arm.tolist()[:5]}")


def _method_graphs(bid: str, b: Block) -> dict[str, tuple[RoutingGraph, float]]:
    """Per plan method, its preset's routing graph and road width, and the claim's routed graph
    (TransparentSubstrate over the preset's: the same nodes and edges, other lengths)."""
    import transparency_run as tr
    presets = common.arms()
    out: dict[str, tuple[RoutingGraph, float]] = {}
    for m in tr.held_plan(tr.HELD_PLAN):
        for arm in (m.base, m.claim):
            tree = tr._tree(arm.method(presets, bid))
            out[arm.name] = (tree.substrate.build(b), tree.road_width_m)
    return out


def _plans(bid: str, b: Block) -> dict[str, fr.Plan]:
    """FREE's plan on every graph of `_method_graphs`; the held roads the same on all."""
    import transparency_run as tr
    plans = {name: tr.FREE.plan(b, graph, width_m=width)
             for name, (graph, width) in _method_graphs(bid, b).items()}
    first = next(iter(plans.values()))
    for name, p in plans.items():
        if (_wkb_roads(p) != _wkb_roads(first) or [s.home for s in p.steps]
                != [s.home for s in first.steps] or p.enclosed != first.enclosed):
            raise AssertionError(f"{bid}: {name}'s graph holds other roads than "
                                 f"{next(iter(plans))}'s")
    return plans


def _head(roads: gpd.GeoDataFrame, m: int) -> gpd.GeoDataFrame:
    """The first `m` roads, a frame."""
    return cast(gpd.GeoDataFrame, roads.iloc[:m])


def _wkb_roads(p: fr.Plan) -> list[bytes]:
    return [bytes(w) for w in shapely.to_wkb(np.asarray([s.road.geometry for s in p.steps]))]


def no_enclosed(out: Path, procs: int, ids: list[str]) -> None:
    """Guard 2 (the module's docstring). A block the scorer's placement finds a home enclosed on
    is no block for it; on the others the stage must hold nothing and change no network."""
    _check_single_threaded()
    import transparency_run as tr
    problems = []
    for b in common.build_blocks(ids):
        grid = tr.FREE.mesh.build(b, fr.REACH_P, common.AreaPopulation())
        if common.placement(grid, np.asarray(b.buildings.outlines), fr.REACH_P).enclosed.any():
            raise SystemExit(f"{b.block_id}: the scorer's placement finds homes enclosed at "
                             "baseline: not a block for this guard")
        plans = _plans(b.block_id, b)
        p = next(iter(plans.values()))
        print(f"{b.block_id}: no home enclosed at baseline; the stage finds {len(p.enclosed)} "
              f"and holds {len(p.steps)} roads on the {len(plans)} graphs ({', '.join(plans)})",
              flush=True)
        if p.enclosed or p.steps:
            problems.append(f"{b.block_id}: the stage holds {len(p.steps)} roads for "
                            f"{len(p.enclosed)} homes the scorer does not find enclosed")
    df = _pool_rows(_against_held, _jobs("free", ids), procs).sort_values(["block", "arm"])
    _write(df, out)
    bad = df[~df.same.astype(bool) | (df.held > 0)]
    print(f"{len(df)} networks with FREE on every arm, on {df.block.nunique()} blocks: the held "
          f"stage's network (holding nothing), frame for frame: {int(df.same.sum())}")
    if len(bad):
        problems.append(f"{len(bad)} networks differ from holding nothing: "
                        f"{bad.arm.tolist()[:5]}")
    if problems:
        raise SystemExit("\n".join(problems))


def _cpu_scorer(b: Block) -> common.Scorer:
    """The scorer road_rescore.py scores with (its mesh, physics and area population), its
    solves on the CPU: B (i)'s test is its `pl` and `free_of`."""
    import road_rescore as rr
    import transparency_run as tr
    p = lifted.Params(ell_m=rr.ELL_M, K=rr.K, along=lifted.along_of(rr.ALONG,
                                                                    lifted.scans_of("cpu")),
                      solver=lifted.solver_of("cpu"))
    return common.Scorer(b, tr.FREE.mesh, p, population=common.AreaPopulation(),
                         demand=common.PopulationDemand())


def _freed(sc: common.Scorer, roads: gpd.GeoDataFrame) -> NDArray[np.bool_]:
    """Per home: enclosed at baseline and freed once `roads` are carved, by the scorer (its
    free field, and B (i)'s test as Scorer.enclosed_left applies it)."""
    return common.reached(sc.pl, sc.grid, sc.free_of(roads), sc.p)


def _free_roads_one(bid: str) -> dict[str, object]:
    import transparency_run as tr
    [b] = common.build_blocks([bid])
    t0 = time.perf_counter()
    plans = _plans(bid, b)
    t_plan = (time.perf_counter() - t0) / len(plans)
    p = next(iter(plans.values()))
    width = next(iter(_method_graphs(bid, b).values()))[1]
    t0 = time.perf_counter()
    sc = _cpu_scorer(b)
    t_sc = time.perf_counter() - t0
    if tuple(int(h) for h in np.flatnonzero(sc.enclosed)) != p.enclosed:
        raise AssertionError(f"{bid}: the plan's enclosed homes {p.enclosed} are not the "
                             f"scorer's {np.flatnonzero(sc.enclosed).tolist()}")
    roads = [s.road.geometry for s in p.steps]

    def frame(k: int) -> gpd.GeoDataFrame:
        return with_width(gpd.GeoDataFrame(geometry=roads[:k], crs=b.crs), width)

    for k, s in enumerate(p.steps, start=1):
        freed = _freed(sc, frame(k))
        if not freed[s.home] or not all(freed[j] for j in s.freed):
            raise AssertionError(f"{bid}: road {k} (home {s.home}) does not free "
                                 f"{[j for j in s.freed if not freed[j]] or [s.home]}")
    left = sc.enclosed_left(sc.free_of(frame(len(roads))))
    polys = np.asarray(b.buildings.outlines)
    d_area = common.displacement(b, frame(len(roads)), common.AreaPopulation().weights(polys))
    d_count = common.displacement(b, frame(len(roads)), common.CountPopulation().weights(polys))
    if (abs(left - p.share_after) > 1e-12 or abs(d_area - p.D_area) > 1e-12
            or abs(d_count - p.D_count) > 1e-12 or max(d_area, d_count) > tr.FREE.budget):
        raise AssertionError(f"{bid}: left {left} against {p.share_after}, D {d_area} {d_count} "
                             f"against {p.D_area} {p.D_count}")
    return dict(block=bid, n=len(polys), enclosed=len(p.enclosed),
                homes=",".join(str(h) for h in p.enclosed), share_before=p.share_before,
                roads=len(p.steps), D_area=p.D_area, D_count=p.D_count,
                road_m=",".join(f"{r.length:.1f}" for r in roads),
                nodes=",".join(str(len(s.road.nodes)) for s in p.steps),
                held_for=",".join(str(s.home) for s in p.steps),
                freed=";".join(",".join(str(j) for j in s.freed) for s in p.steps),
                unreachable=",".join(map(str, p.unreachable)),
                over_budget=",".join(map(str, p.over_budget)), share_after=p.share_after,
                graphs=len(plans), t_plan_s=t_plan, t_scorer_s=t_sc)


def free_roads(out: Path, procs: int, ids: list[str]) -> None:
    """Guard 3 (the module's docstring)."""
    _check_single_threaded()
    df = _pool_rows(_free_roads_one, ids, procs).sort_values("n")
    _write(df, out)
    print(f"{len(df)} blocks: {int(df.enclosed.sum())} homes enclosed at baseline, "
          f"{int(df.roads.sum())} roads held, each freeing its home by the scorer's test; "
          f"left enclosed on {int((df.share_after > 0).sum())} blocks")


def _lead_one(job: tuple[str, str]) -> list[dict[str, object]]:
    import road_rescore as rr
    import transparency_run as tr
    stage_name, bid = job
    stage = tr._stage(stage_name)
    [b] = common.build_blocks([bid])
    p = next(iter(_plans(bid, b).values()))
    want = _wkb_roads(p)
    allowed = set(p.unreachable) | set(p.over_budget)
    sc = _cpu_scorer(b)
    rows = []
    for arm in stage.arms:
        net = gpd.read_parquet(stage.record_path(bid, arm.name))
        ordered = gpd.read_parquet(rr.ordered_path(stage.scratch, bid, arm.name))
        pre = pd.read_parquet(rr.prefix_dir(stage.rows) / f"{bid}__{arm.name}.parquet")
        n = held_count(net)
        at = tr._held_rows(ordered)         # where the ordered network has its held roads
        leads = bool(n == len(want) and _wkb(_head(net, n)) == want
                     and np.array_equal(at, np.arange(n)) and _wkb(_head(ordered, n)) == want)
        r: dict[str, object] = dict(block=bid, arm=arm.name, held=n, plan_roads=len(want),
                                    leads=leads, held_rows=",".join(map(str, at)))
        for pop in ("count", "area"):
            row = pre[(pre["pop"] == pop) & np.isclose(pre.budget, tr.LENS_D)].iloc[0]
            m = int(row.m)
            holds_all = bool(len(at) == len(want) and (at < m).all())
            left = set(np.flatnonzero(sc.enclosed & ~_freed(sc, _head(ordered, m))).tolist())
            r.update({f"m_{pop}": m, f"holds_all_{pop}": holds_all,
                      f"left_{pop}": ",".join(map(str, sorted(left))),
                      f"ok_{pop}": bool(holds_all and left <= allowed)})
        rows.append(r)
        print(f"{bid} {arm.name}: {n} held (plan {len(want)}), leads {leads}; at D {tr.LENS_D:g} "
              f"count prefix {r['m_count']} roads, area prefix {r['m_area']}, left enclosed "
              f"{r['left_count'] or '-'} | {r['left_area'] or '-'}", flush=True)
    return rows


def lead(stage_name: str, out: Path, procs: int, ids: list[str]) -> None:
    """Guard 4 (the module's docstring), on `stage_name`'s rows."""
    _check_single_threaded()
    import transparency_run as tr
    stage = tr._stage(stage_name)
    if not isinstance(stage.first, tr.FreeEveryArm):
        raise SystemExit(f"{stage_name}'s arms hold nothing first")
    with multiprocessing.get_context("fork").Pool(min(procs, len(ids)), maxtasksperchild=1) as pl:
        got = [r for rows in pl.imap_unordered(_lead_one, [(stage_name, bid) for bid in
                                                           stage.ids_of(",".join(ids))])
               for r in rows]
    df = pd.DataFrame(got).sort_values(["block", "arm"])
    _write(df, out)
    bad = df[~(df.leads.astype(bool) & df.ok_count.astype(bool) & df.ok_area.astype(bool))]
    print(f"{len(df)} networks on {df.block.nunique()} blocks: held roads leading "
          f"{int(df.leads.sum())}; every held road in the D {tr.LENS_D:g} prefixes and no home "
          f"left enclosed there but those the plan left: count {int(df.ok_count.sum())}, area "
          f"{int(df.ok_area.sum())}")
    if len(bad):
        raise SystemExit(f"{len(bad)} networks fail: {bad.arm.tolist()[:5]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "opaque":
        opaque(Path(a[1]), int(a[2]), a[3].split(","), a[4].split(","))
    elif a[0] == "sets":
        sets(a[1].split(","), [int(s) for s in a[2].split(",")])
    elif a[0] == "rule":
        rule()
    elif a[0] == "held":
        held()
    elif a[0] == "claim":
        claim()
    elif a[0] == "hold-nothing":
        hold_nothing(Path(a[1]), int(a[2]), a[3].split(","))
    elif a[0] == "no-enclosed":
        no_enclosed(Path(a[1]), int(a[2]), a[3].split(","))
    elif a[0] == "free-roads":
        free_roads(Path(a[1]), int(a[2]), a[3].split(","))
    elif a[0] == "lead":
        lead(a[1], Path(a[2]), int(a[3]), a[4].split(","))
    else:
        raise SystemExit(f"unknown command {a[0]!r}")
