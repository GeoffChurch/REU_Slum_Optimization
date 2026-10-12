"""Guards of the road-transparency code (BACKLOG 5) that need the real blocks and stored rows; the
rest are unit tests (tests/test_transparency.py).

    S=research/roadless/transparency_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S opaque <out.csv> <procs> <ids,> <methods,>
    uv run python $S sets <ids,> <seeds>

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

FAULT INJECTION, `sets` on six blocks: positions read off by one, draws that ignore the strata, the
seed ignored, a ClearedSet identity without the positions' digest, and AreaMass cutting its strata
by count each fail it. `opaque`: the brief's literal route formula (outside length from the union,
inside length per building), under which a stretch inside two overlapping opaque footprints weighs
more than open ground, fails it.
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

from reblock.contracts import Block, Method
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.cycle_native import CycleNativeReblocker
from reblock.methods.loop_closure import LoopClosureRefiner
from reblock.methods.resistance_lp import ResistanceLPReblocker
from reblock.transparency import Opaque, TransparentSubstrate, factors_of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import transparency_sets as ts  # noqa: E402

NETS = HERE / "road_rows" / "scratch" / "networks"
SINGLE = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
# propose order only (no result depends on it): the costliest first
COST_GUESS = {"cycle_native": 2.5, "resistance_lp": 2.0, "clearance_looped": 1.0}
STRATIFICATION = ts.AreaMass(strata=20)     # the experiment's control (owner's ruling 2026-10-12)
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


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "opaque":
        opaque(Path(a[1]), int(a[2]), a[3].split(","), a[4].split(","))
    elif a[0] == "sets":
        sets(a[1].split(","), [int(s) for s in a[2].split(",")])
    else:
        raise SystemExit(f"unknown command {a[0]!r}")
