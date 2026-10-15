"""The region model on the complex: exits as leaks to ground, tested on 5810@major (BACKLOG 4,
decision (4); NOTES, "Exits as leaks: the region model on the complex (5810@major)"; the brief
.superpowers/sdd/2026-10-13-region-leaks).

The configurations (CONFIGS, each a gap_exits.Exits, the metric's suffix naming it): the boundary
only with no barriers (`edge`: the raster run's setting, the per-block metric's name), the
boundary with the barriers (`bar`: the reference), with uniform leaks at lam 250, 500 and 1000 m
(`U250` ...), with building-boundary leaks at the same lam (`H250` ...), and the street boost as a
check (`S3`, `S3U500`), all with BARRIERS but `edge`. Each is run with the cheap preset
(polish_greedy.CHEAP: its runs named `<metric>@e0.05`) and SIMP's default (relax.SIMP_DEFAULT) at
area D BUDGET with free first, optimizing P under the area population.

    S=research/roadless/region_leaks.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S warm <id> <label>              # the label's fixed mesh into gap_clear.CACHE
    uv run python $S run <id> <label> <simp|cheap|prior> <cpu|gpu>   # a preset, or the prior
    uv run python $S members <procs>                # the member kblocks' baselines (per block)
    uv run python $S baseline <label>               # the region's baseline under a configuration
    uv run python $S score <label>                  # its runs' clearings: lenses, flows, maps
    uv run python $S cross <label> <labels,>        # other configurations' clearings under it
    uv run python $S report                         # the summaries (ROWS/summary_*.csv)

`run`'s row (relax.one's or polish_greedy.run's, with the label, the metric's name, the host, the
peak host memory, the device's peak memory on a GPU and the wall time) in
ROWS/runs/<label>/<preset>/<id>.parquet.

`members`: each member kblock of the region (regions.build's `member`) under the per-block metric
(its own street ring the exit) on a core: every home's escape time at baseline (the population's
demand), in the region's population units (AreaPopulation normalises each block to mean 1: a
kblock's potentials times its mean footprint area over the region's); ROWS/members/<kblock>.parquet.

`baseline`: the region's model under a configuration on a core: P0, P_{1/2},0, J_2,0, the homes
walled in, the leaks' total and the share of the injected current they drain at baseline (the
rest leaves by the boundary), every home's escape time, and for a leaky configuration the
self-absorption of ABSORB_N homes drawn at random (a home's unit injection solved alone: the share
of its current that leaks on the cells where its own building's injection lands, its own
building's share of that leak (HomeLeak: the self-trips), and the share that leaks anywhere); the
flow painted (gap_maps) at baseline on FRAME_H pixels. ROWS/baseline/<label>.parquet (and
_homes, _absorb), ROWS/images/<label>/baseline.npz.

`run ... prior`: the no-buildings prior (every footprint cleared on the fixed mesh, the atlas's
prior: the same homes' flow to the same exits) and its P, painted on PRIOR_H pixels (the maps'
frame). Every cell is then an unknown (16M on the region), past SuperLU's 32-bit indices on a core:
it runs on a GPU (cuDSS); ROWS/baseline/<label>_prior.parquet (its row, the frame among it) and
<label>_prior_image.parquet (the pixels, flattened: parquet, as a cluster run's rows come back).

`score`: each run of the configuration (both presets; under `edge` also the raster run's picks,
RASTER_RUNS) scored on its model: L_P, L_{P_1/2}, L_J2 and the population left walled in; its flow
painted; ROWS/score/<label>.parquet, ROWS/images/<label>/<run>.npz.

`cross`: the runs of other configurations (both presets) scored under this one's model: what a
clearing chosen under one exit model is worth under another (the street boost's check, its own
runs cut for the GPU budget: NOTES); ROWS/cross/<label>.parquet.

`report`: the barriers (summary_barriers), the baselines and self-absorption, the lens table, the
loop index (`loops`) of every painted state, the other configurations' clearings under each
(`cross`), the corridor maps (ROWS/maps/<label>_<preset>.png: the flow after against the
no-buildings prior, as the atlas's `rel` layer, the change in flow from the baseline, and a zoom
of it on 5810), the escape times of formal homes and deep informal ones (Homes: the formal grid
the member kblocks whose ring is mostly formal streets) beside the per-block model's, the cleared
sets (5810's share, the formal grid's, Jaccard between configurations, between the presets and
against the raster's picks); ROWS/summary_*.csv.

THE LOOP INDEX (`loops`). The corridor network at a threshold T is the open pixels whose flow is
at least T, and the exits X the pixels within EXIT_PX_M of the street line; with X contracted to
one node, the network's cycle rank (independent loops) is L = b0 - chi of the quotient, chi the
Euler characteristic of the pixels' 4-connectivity complex (pixels, adjacent pairs and full 2 x 2
blocks), chi(Y / X) = chi(Y) - chi(X) + 1. A tree of tendrils reaching in from the edge has L 0,
each corridor joining two points of the edge or closing a ring inside adds 1 (a grid of n blocks
inside an exit ring has n - 1: the `loop-index` guard's cases). Holes in the network under
HOLE_M2 (a pixel or two below T inside a lane) are filled first. T is a quantile LOOP_Q of the
baseline's flow over the open pixels (the top tenth of its area at 0.9), the same for every state
of a configuration; reported per km^2 of the region, for the network after a clearing and for the
corridors it induces (the pixels whose flow rose by at least T).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/region_leaks.py
"""
from __future__ import annotations

import dataclasses
import itertools
import multiprocessing
import os
import socket
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_exits as gx  # noqa: E402
import gap_maps  # noqa: E402
import metrics  # noqa: E402
import polish_greedy  # noqa: E402
import relax  # noqa: E402
import search  # noqa: E402

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from reblock.contracts import Block

REGION = "ZAF.9.3.1_1_5810@major"
SEED_KBLOCK = "ZAF.9.3.1_1_5810"
ROWS = HERE / "region_rows"
BUDGET = relax.D_LENS                   # area D 0.05
POWER = 1.0                             # P, the headline (decision (1))
POP = common.PopulationDemand()         # area population (decision (2))
PLAN = relax.plan_of(relax.SIMP_DEFAULT)
CHEAP = polish_greedy.CHEAP
# The barriers (gap_exits.OsmBarriers): a track's or a waterway's strip 8 m wide (the railway's
# two tracks, 3.7 -- 7.7 m apart on 5810@major, a median 4.25, merge into one corridor), a
# crossing's gap 3 m (a footway's: the region's one crossing of the railway inside it is an
# unpaved path)
BARRIERS = gx.OsmBarriers(line_m=8.0, cross_m=3.0)
LAMS = (250.0, 500.0, 1000.0)
BOOST = gx.StreetBoost(factor=3.0)


def _exits(*, barriers: gx.OsmBarriers | gx.NoBarriers, leak: gx.NoLeak | gx.UniformLeak
           | gx.HomeLeak, boost: gx.NoBoost | gx.StreetBoost) -> gx.Exits:
    return gx.Exits(barriers=barriers, leak=leak, boost=boost)


CONFIGS: dict[str, gx.Exits] = {
    "edge": gx.per_block(),
    "bar": _exits(barriers=BARRIERS, leak=gx.NoLeak(), boost=gx.NoBoost()),
    **{f"U{lam:g}": _exits(barriers=BARRIERS, leak=gx.UniformLeak(lam=lam), boost=gx.NoBoost())
       for lam in LAMS},
    **{f"H{lam:g}": _exits(barriers=BARRIERS, leak=gx.HomeLeak(lam=lam), boost=gx.NoBoost())
       for lam in LAMS},
    "S3": _exits(barriers=BARRIERS, leak=gx.NoLeak(), boost=BOOST),
    "S3U500": _exits(barriers=BARRIERS, leak=gx.UniformLeak(lam=500.0), boost=BOOST),
}
PRESETS = ("simp", "cheap")
# the raster run's picks on 5810@major (NOTES, "A region whose only exit is its edge"; J_2, on
# the composite raster): their rows, read as records
RASTER_RUNS = {
    "raster_cheap": HERE / "polish_rows/uni/S0.01cat.P64w8.a5x8/D0.05" / f"{REGION}_p2.parquet",
    "raster_simp": (HERE / "relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.a5x8/D0.05"
                    / f"{REGION}_p2.parquet")}
ABSORB_N = 300          # self-absorption: homes drawn
ABSORB_SEED = 20261015
FRAME_H = 1.0           # the painted flow's pixel (m)
PRIOR_H = 2.0           # the prior's and the maps' pixel (m: FRAME_H's 2 x 2, on the same edges)
FRAME_PAD = 2.0         # the frames' margin about the region (m: a multiple of both pixels)
STREET_CLASSES = ("residential", "living_street", "unclassified")   # a formal home's street
DRIVE_CLASSES = (*STREET_CLASSES, "service", "tertiary", "secondary", "primary", "trunk",
                 "motorway")                                # a deep home is far from all of them
FORMAL_M = 10.0         # a formal home: its footprint within this of a STREET_CLASSES way's centre
                        # line (m: a 13 m street reserve's half and a few metres' setback), in a
                        # kblock of the formal grid
GRID_SHARE = 0.5        # the formal grid: the member kblocks but SEED_KBLOCK whose outline lies
                        # within FORMAL_M of a STREET_CLASSES way along at least this share of
                        # its length (their streets the exits of the per-block model)
DEEP_M = 30.0           # a deep informal home: in SEED_KBLOCK, this far from every DRIVE_CLASSES
                        # way (m)
LOOP_Q = (0.8, 0.9, 0.95)   # the loop index's thresholds: quantiles of the baseline's flow
HOLE_M2 = 25.0          # ... holes in the network under this filled (m^2)
EXIT_PX_M = 2.0         # ... the exits: pixels within this of the street line (m)


def metric_of(label: str, preset: str, device: str) -> gcl.ComplexMetric:
    """The configuration's metric for a preset (the cheap preset's search its own: `@e0.05`)."""
    base = dataclasses.replace(metrics.metric_of(metrics.RUNTIME, device=device),
                               exits=CONFIGS[label])
    return {"simp": base, "cheap": CHEAP.on(base)}[preset]


# ------------------------------------------------------------------ the runs

def warm(bid: str, label: str) -> None:
    """The configuration's fixed mesh built into gap_clear.CACHE (a barrier changes the mesh)."""
    [b] = common.build_blocks([bid])
    metric = metric_of(label, "simp", "cpu")
    t0 = time.time()
    fx = metric.fixed(b, cache=gcl.CACHE, carving=gcl.Carving.none(gc.geometry(b)))
    print(f"{bid} {label}: the fixed mesh, {fx.fm.nt} triangles, {time.time() - t0:.0f}s (built "
          f"in {fx.t_build:.0f}s); peak {gap_calib.peak_gb():.1f} GB", flush=True)


def run_path(label: str, preset: str, bid: str) -> Path:
    return ROWS / "runs" / label / preset / f"{bid}.parquet"


def run(bid: str, label: str, preset: str, device: str) -> None:
    """A preset under a configuration (module docstring)."""
    metric = metric_of(label, preset, device)
    out = run_path(label, preset, bid)
    t0 = time.time()

    def go() -> dict[str, object]:
        if preset == "simp":
            return relax.one(bid, POWER, metric, PLAN, BUDGET, demand=POP)
        return polish_greedy.run(bid, POWER, metric, CHEAP.picker, CHEAP.tries, CHEAP.width,
                                 BUDGET, CHEAP.pairs, scoring=search.InEps())
    if device == "gpu":
        with gap_calib.GpuPeak() as peak:
            row = go()
        gpu_gb = peak.peak
    else:
        row, gpu_gb = go(), float("nan")
    row.update(label=label, preset=preset, metric=metric.name, host=socket.gethostname(),
               peak_rss_gb=gap_calib.peak_gb(), gpu_gb=gpu_gb, wall=time.time() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out)
    print(f"{bid} {label} {preset} on {device}: {metric.name}; peak {gap_calib.peak_gb():.1f} GB, "
          f"GPU {gpu_gb:.1f} GB, {row['wall']:.0f}s", flush=True)


def cleared_of(label: str, preset: str) -> NDArray[np.int64]:
    """A run's clearing (its row's buildings)."""
    r = pd.read_parquet(run_path(label, preset, REGION)).iloc[0]
    return np.asarray(r.simp_cleared if preset == "simp" else r.cleared, dtype=np.int64)


# ------------------------------------------------------------------ synthetic blocks

def synthetic_block(block_id: str, boundary: BaseGeometry, street: BaseGeometry,
                    homes: Sequence[BaseGeometry], osm: gx.Osm | None) -> Block:
    """A block in 19570's frame (its CRS, its corner the origin of these local coordinates): the
    polygon `boundary`, the street line `street`, the footprints `homes` and the OSM features
    `osm` (None: none, as a kblock), the rest 19570's (its building tier among it): the guards'
    and the loop index's known cases."""
    import geopandas as gpd
    [t] = common.build_blocks(["ZAF.9.3.1_1_19570"])
    x0, y0 = float(t.boundary.bounds[0]), float(t.boundary.bounds[1])
    to = np.array([x0, y0])

    def moved(g: BaseGeometry) -> BaseGeometry:
        return shapely.transform(g, lambda xy: xy + to)
    polys = [moved(h) for h in homes]
    bld = gpd.GeoDataFrame({"anchor": gpd.GeoSeries([p.representative_point() for p in polys],
                                                    crs=t.crs)},
                           geometry=polys, crs=t.crs)
    attrs: dict[str, object] = {}
    if osm is not None:
        attrs["osm"] = gx.Osm(**{k: v.set_geometry(v.geometry.translate(x0, y0))
                                 for k, v in osm._asdict().items()})
    return cast("Block", dataclasses.replace(
        t, block_id=block_id, boundary=moved(boundary),
        streets=gpd.GeoDataFrame(geometry=[moved(street)], crs=t.crs),
        parcels=gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[moved(boundary)], crs=t.crs),
        building_geometries=bld, source_content_hash=None, attrs=attrs))


# ------------------------------------------------------------------ the region's homes

class Homes(NamedTuple):
    """Per building of the region: its member kblock and row there, its footprint area, its
    distance to the nearest formal street way (STREET_CLASSES) and to the nearest drivable one
    (DRIVE_CLASSES), whether its kblock is of the formal grid (`grid`: GRID_SHARE), and the two
    groups of question 2: `formal` (in the formal grid, FORMAL_M of a formal street) and `deep`
    (in SEED_KBLOCK, at least DEEP_M from every drivable way)."""
    member: NDArray[np.object_]
    member_row: NDArray[np.int64]
    area: NDArray[np.float64]
    street_m: NDArray[np.float64]
    drive_m: NDArray[np.float64]
    grid: NDArray[np.bool_]
    formal: NDArray[np.bool_]
    deep: NDArray[np.bool_]


def formal_grid(b: Block) -> set[str]:
    """The member kblocks of the formal grid (GRID_SHARE)."""
    ways = gx.osm_in(b).ways
    streets = shapely.buffer(shapely.union_all(ways.geometry.to_numpy()[
        ways.highway.str.removesuffix("_link").isin(STREET_CLASSES).to_numpy()]), FORMAL_M)
    out = set()
    for m in common.member_blocks(sorted(set(b.building_geometries["member"]))):
        ring = shapely.boundary(cast("BaseGeometry", m.boundary))
        if (m.block_id != SEED_KBLOCK and float(shapely.length(shapely.intersection(
                ring, streets))) >= GRID_SHARE * float(shapely.length(ring))):
            out.add(m.block_id)
    return out


def homes_of(b: Block, grid: set[str]) -> Homes:
    bg = b.building_geometries
    polys = np.asarray(b.buildings.outlines)
    ways = gx.osm_in(b).ways
    g = ways.geometry.to_numpy()

    def dist(classes: tuple[str, ...]) -> NDArray[np.float64]:
        lines = g[ways.highway.str.removesuffix("_link").isin(classes).to_numpy()]
        _i, d = shapely.STRtree(lines).query_nearest(polys, return_distance=True,
                                                     all_matches=False)
        return np.asarray(d, dtype=np.float64)
    street_m, drive_m = dist(STREET_CLASSES), dist(DRIVE_CLASSES)
    member = bg["member"].to_numpy().astype(object)
    in_grid = np.isin(member.astype(str), sorted(grid))
    return Homes(member=member, member_row=bg["member_row"].to_numpy().astype(np.int64),
                 area=np.asarray(shapely.area(polys), dtype=np.float64), street_m=street_m,
                 drive_m=drive_m, grid=in_grid, formal=in_grid & (street_m <= FORMAL_M),
                 deep=(member == SEED_KBLOCK) & (drive_m >= DEEP_M))


# ------------------------------------------------------------------ the member kblocks

def _member_one(bid: str) -> str | None:
    out = ROWS / "members" / f"{bid}.parquet"
    try:
        got = common.member_blocks([bid])
        if [g.block_id for g in got] != [bid]:
            raise ValueError(f"the source yields {[g.block_id for g in got]} for {bid}")
        [b] = got
        m = metrics.metric_of(metrics.RUNTIME, device="cpu").clearing(b, demand=POP)
        area = np.asarray(shapely.area(np.asarray(b.buildings.outlines)), dtype=np.float64)
        pd.DataFrame({"member": bid, "member_row": np.arange(m.n), "u0": m.u_pop0,
                      "walled": m.walled0, "nodem": m.nodem, "area": area,
                      "mean_area": float(area.mean())}).to_parquet(out)
        print(f"{bid}: {m.n} homes, {int(m.walled0.sum())} walled in; peak "
              f"{gap_calib.peak_gb():.1f} GB", flush=True)
    except Exception as e:      # noqa: BLE001 -- reported, the pool goes on
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return bid
    return None


def members(procs: int) -> None:
    [b] = common.build_blocks([REGION])
    ids = sorted(set(b.building_geometries["member"]))
    (ROWS / "members").mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (ROWS / "members" / f"{i}.parquet").exists()]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        failed = [f for f in pool.imap_unordered(_member_one, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} members failed: {', '.join(failed)}")


# ------------------------------------------------------------------ the region's model on a core

class Painted(NamedTuple):
    """The frame the flows are painted on (gap_maps.Frame over the region's polygon at FRAME_H),
    the region's pixels (inside its polygon, barriers cut), and its exits' (EXIT_PX_M of the
    street line)."""
    frame: gap_maps.Frame
    inside: NDArray[np.bool_]
    exits: NDArray[np.bool_]


def painted_of(m: gcl.Model, h: float) -> Painted:
    fr = gap_maps.Frame.around(m.geo, h, FRAME_PAD)
    xs = fr.x0 + np.arange(fr.nx) * fr.h
    ys = fr.y0 + np.arange(fr.ny) * fr.h
    X, Y = np.meshgrid(xs, ys)
    region = cast("BaseGeometry", m.geo.boundary)
    shapely.prepare(region)
    inside = shapely.contains_xy(region, X, Y)
    near = shapely.buffer(cast("BaseGeometry", m.geo.streets), EXIT_PX_M)
    shapely.prepare(near)
    return Painted(frame=fr, inside=inside, exits=inside & shapely.contains_xy(near, X, Y))


def _model(label: str) -> tuple[Block, gcl.Model, float]:
    [b] = common.build_blocks([REGION])
    t0 = time.time()
    m = metric_of(label, "simp", "cpu").clearing(b, demand=POP)
    return b, m, time.time() - t0


def _paint_state(m: gcl.Model, k: NDArray[np.float64], pt: Painted
                 ) -> tuple[NDArray[np.float32], gap_maps.CellFlow, gcl.State]:
    st = m.state(k)
    fl = gap_maps.flow_of(m, st, m.pop)
    return gap_maps.flow_image(m, st, fl, pt.frame).astype(np.float32), fl, st


def _save_image(label: str, what: str, img: NDArray[np.float32], pt: Painted) -> None:
    d = ROWS / "images" / label
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{what}.npz", flow=img, inside=pt.inside, exits=pt.exits,
                        frame=np.array(pt.frame[:3]), shape=np.array(pt.frame[3:]))


def baseline(label: str) -> None:
    _b, m, t_model = _model(label)
    pt = painted_of(m, FRAME_H)
    t0 = time.time()
    img0, fl0, st0 = _paint_state(m, np.zeros(m.n), pt)
    _save_image(label, "baseline", img0, pt)
    leak = st0.sys.graph.leak
    row = dict(label=label, metric=m.exits.suffix, P0=m.base.P, Pq0=m.base.Pq, J2_0=m.base.J2,
               walled=int(m.walled0.sum()), walled_pop=float(m.cost[m.walled0].sum()),
               nodem=int(m.nodem.sum()), cells=m.c.n, unknowns0=m.cells0,
               leak_total=float(leak.sum()), leak_cells=int((leak > 0).sum()),
               leak_share=float(fl0.leak.sum() / fl0.drain.sum()),
               live_area=float(m.leak_site.area[m.leak_site.live].sum()), t_model=t_model,
               region_km2=float(shapely.area(cast("BaseGeometry", m.geo.boundary))) / 1e6)
    (ROWS / "baseline").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"u0": m.u_pop0, "walled": m.walled0, "nodem": m.nodem}).to_parquet(
        ROWS / "baseline" / f"{label}_homes.parquet")
    row.update(t_state=time.time() - t0)
    absorbed = ROWS / "baseline" / f"{label}_absorb.parquet"
    if not isinstance(m.exits.leak, gx.NoLeak) and not absorbed.exists():
        absorb(label, m, st0).to_parquet(absorbed)
    row.update(peak_gb=gap_calib.peak_gb())
    pd.DataFrame([row]).to_parquet(ROWS / "baseline" / f"{label}.parquet")
    print(f"{label}: {row}", flush=True)


def prior(bid: str, label: str, device: str) -> None:
    """The no-buildings prior (module docstring)."""
    [b] = common.build_blocks([bid])
    t0 = time.time()
    m = metric_of(label, "simp", device).clearing(b, demand=POP)
    pt = painted_of(m, PRIOR_H)
    img, fl, st = _paint_state(m, np.ones(m.n), pt)
    row = dict(label=label, P_prior=float(m._rhs(st, m.pop) @ fl.u), P0=m.base.P,
               leak_share_prior=float(fl.leak.sum() / fl.drain.sum()), t=time.time() - t0,
               device=device, host=socket.gethostname(), peak_gb=gap_calib.peak_gb(),
               x0=pt.frame.x0, y0=pt.frame.y0, h=pt.frame.h, ny=pt.frame.ny, nx=pt.frame.nx)
    (ROWS / "baseline").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"flow": img.ravel(), "inside": pt.inside.ravel()}).to_parquet(
        ROWS / "baseline" / f"{label}_prior_image.parquet")
    pd.DataFrame([row]).to_parquet(ROWS / "baseline" / f"{label}_prior.parquet")
    print(f"{label} prior: {row}", flush=True)


def absorb(label: str, m: gcl.Model, st0: gcl.State) -> pd.DataFrame:
    """Self-absorption (module docstring) at baseline, ABSORB_N homes with demand drawn at random:
    each home's unit injection solved alone."""
    assert st0.factor is not None
    leak = st0.sys.graph.leak
    Pc = st0.sys.Pc.tocsc()
    unk = st0.reach.unknown
    own_k = 0.0
    if isinstance(m.exits.leak, gx.HomeLeak):
        site = m.leak_site
        own_k = m.exits.leak.total(site) / float(np.where(site.live, site.inj, 0.0).sum())
    ok = np.flatnonzero(~m.nodem & ~m.walled0)
    pick = np.sort(np.random.default_rng(ABSORB_SEED).choice(ok, min(ABSORB_N, len(ok)),
                                                              replace=False))
    rows = []
    for h in pick:
        col = np.zeros(st0.sys.n)
        lo, hi = Pc.indptr[h], Pc.indptr[h + 1]
        col[Pc.indices[lo:hi]] = Pc.data[lo:hi]
        bvec = col * unk
        q = float(bvec.sum())
        u = st0.factor.solve(bvec)
        ring = bvec > 0
        rows.append(dict(home=int(h), injected=q, pop=float(m.pop[h]),
                         own_building=own_k * float(m.pop[h]) * float(bvec @ u) / q,
                         own_cells=float((leak[ring] * u[ring]).sum()) / q,
                         anywhere=float((leak * u).sum()) / q))
    print(f"{label}: self-absorption over {len(rows)} homes done", flush=True)
    return pd.DataFrame(rows)


def score(label: str) -> None:
    b, m, _t = _model(label)
    pt = painted_of(m, FRAME_H)
    runs: dict[str, NDArray[np.int64]] = {p: cleared_of(label, p) for p in PRESETS
                                          if run_path(label, p, REGION).exists()}
    if label == "edge":
        for key, path in RASTER_RUNS.items():
            r = pd.read_parquet(path).iloc[0]
            runs[key] = np.asarray(r.simp_cleared if "simp_cleared" in r.index else r.cleared,
                                   dtype=np.int64)
    rows = []
    for key, ids in runs.items():
        x = np.zeros(m.n)
        x[ids] = 1.0
        img, fl, st = _paint_state(m, x, pt)
        _save_image(label, key, img, pt)
        L = m.lenses_of(st)
        rows.append(dict(label=label, run=key, n=len(ids), D=float(m.cost[ids].sum()),
                         L_P=1.0 - L.P / m.base.P, L_Pq=1.0 - L.Pq / m.base.Pq,
                         L_J2=1.0 - (L.J2 / m.base.J2) ** 0.5, enclosed=L.enclosed,
                         n_enclosed=L.n_enclosed, leak_share=float(fl.leak.sum() / fl.drain.sum())))
        print(f"{label} {key}: {rows[-1]}", flush=True)
    (ROWS / "score").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ROWS / "score" / f"{label}.parquet")


def cross(label: str, sources: list[str]) -> None:
    """Other configurations' clearings under this one (module docstring)."""
    _b, m, _t = _model(label)
    rows = []
    for src in sources:
        for preset in PRESETS:
            if not run_path(src, preset, REGION).exists():
                continue
            ids = cleared_of(src, preset)
            x = np.zeros(m.n, dtype=bool)
            x[ids] = True
            L = m.lenses(x)
            rows.append(dict(label=label, source=src, preset=preset, n=len(ids),
                             L_P=1.0 - L.P / m.base.P, L_Pq=1.0 - L.Pq / m.base.Pq,
                             L_J2=1.0 - (L.J2 / m.base.J2) ** 0.5, enclosed=L.enclosed))
            print(f"{label} <- {src} {preset}: {rows[-1]}", flush=True)
    (ROWS / "cross").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ROWS / "cross" / f"{label}.parquet")


# ------------------------------------------------------------------ the loop index

def _chi4(Y: NDArray[np.bool_]) -> int:
    """The Euler characteristic of the pixels' 4-connectivity complex: pixels - adjacent pairs +
    full 2 x 2 blocks."""
    return int(Y.sum() - (Y[:, 1:] & Y[:, :-1]).sum() - (Y[1:, :] & Y[:-1, :]).sum()
               + (Y[1:, 1:] & Y[1:, :-1] & Y[:-1, 1:] & Y[:-1, :-1]).sum())


def loops(flow: NDArray[np.float32], inside: NDArray[np.bool_], exits: NDArray[np.bool_],
          threshold: float, h: float, *, contract: bool) -> int:
    """The corridor network's cycle rank with the exits contracted (module docstring): the open
    pixels inside the region whose flow is at least `threshold`, and the exits; holes under
    HOLE_M2 filled. `contract` False: the network's own cycle rank (the `loop-index` guard's
    fault)."""
    Y = (inside & (np.nan_to_num(flow, nan=-np.inf) >= threshold)) | exits
    holes, n = ndimage.label(inside & ~Y, structure=np.ones((3, 3), dtype=bool))
    small = np.flatnonzero(np.bincount(holes.ravel(), minlength=n + 1)[1:] * h * h < HOLE_M2) + 1
    Y |= np.isin(holes, small)
    lab, _n = ndimage.label(Y)
    if not contract:
        return int(_n - _chi4(Y))
    touch = len(np.setdiff1d(np.unique(lab[exits]), [0]))
    b0 = _n - touch + 1
    chi = _chi4(Y) - _chi4(exits) + 1
    return int(b0 - chi)


# ------------------------------------------------------------------ the report

def _image(label: str, what: str) -> NDArray[np.float32]:
    with np.load(ROWS / "images" / label / f"{what}.npz") as z:
        return np.asarray(z["flow"])


def _masks(label: str) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    with np.load(ROWS / "images" / label / "baseline.npz") as z:
        return np.asarray(z["inside"]), np.asarray(z["exits"])


def _states(label: str) -> list[str]:
    return [p.stem for p in sorted((ROWS / "images" / label).glob("*.npz"))]


def loop_rows() -> pd.DataFrame:
    rows = []
    for label in CONFIGS:
        if not (ROWS / "images" / label / "baseline.npz").exists():
            continue
        inside, exits = _masks(label)
        base = _image(label, "baseline")
        km2 = float(inside.sum()) * FRAME_H ** 2 / 1e6
        on = inside & np.isfinite(base) & (base > 0)
        for q in LOOP_Q:
            T = float(np.quantile(base[on], q))
            for what in _states(label):
                after = _image(label, what)
                L = loops(after, inside, exits, T, FRAME_H, contract=True)
                # the induced corridors: where the flow rose by at least T
                rise = (np.where(np.isfinite(after), after, 0.0)
                        - np.where(np.isfinite(base), base, 0.0)).astype(np.float32)
                Li = loops(rise, inside, exits, T, FRAME_H, contract=True)
                rows.append(dict(label=label, state=what, q=q, T=T, loops=L, per_km2=L / km2,
                                 induced=Li, induced_per_km2=Li / km2,
                                 corridor_km2=float((inside & (np.nan_to_num(after) >= T)).sum())
                                 * FRAME_H ** 2 / 1e6,
                                 induced_km2=float((inside & (rise >= T)).sum())
                                 * FRAME_H ** 2 / 1e6))
    return pd.DataFrame(rows)


def _coarse(img: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.float64]:
    """A FRAME_H image on PRIOR_H's pixels (the same edges): each the mean of its 2 x 2 pixels
    painted (NaN where none is), the frame's last row and column padded."""
    k = int(round(PRIOR_H / FRAME_H))
    ny, nx = shape
    out = np.full((ny * k, nx * k), np.nan)
    h, w = min(img.shape[0], ny * k), min(img.shape[1], nx * k)
    out[:h, :w] = img[:h, :w]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # a block with no pixel painted: NaN
        return np.asarray(np.nanmean(out.reshape(ny, k, nx, k), axis=(1, 3)))


def _jaccard(a: NDArray[np.int64], b: NDArray[np.int64]) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


def maps(label: str, homes: Homes, polys: NDArray[np.object_]) -> list[Path]:
    """The corridor maps of a configuration's two runs (module docstring)."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    prior_row = ROWS / "baseline" / f"{label}_prior.parquet"
    if not prior_row.exists():          # a configuration with no runs (the boost's: NOTES)
        return []
    r = pd.read_parquet(prior_row).iloc[0]
    x0, y0, h, ny, nx = float(r.x0), float(r.y0), float(r.h), int(r.ny), int(r.nx)
    px = pd.read_parquet(ROWS / "baseline" / f"{label}_prior_image.parquet")
    sp_ = px.flow.to_numpy().reshape(ny, nx)
    inside = px.inside.to_numpy().reshape(ny, nx)
    ext = (x0 - h / 2, x0 + (nx - 0.5) * h, y0 - h / 2, y0 + (ny - 0.5) * h)
    s0 = _coarse(_image(label, "baseline"), (ny, nx))
    sel = homes.member == SEED_KBLOCK
    out = []
    for preset in PRESETS:
        if not (ROWS / "images" / label / f"{preset}.npz").exists():
            continue
        s1 = _coarse(_image(label, preset), (ny, nx))
        ids = cleared_of(label, preset)
        ok = inside & np.isfinite(s1) & (s1 > 0) & np.isfinite(sp_) & (sp_ > 0)
        rel = np.full(s1.shape, np.nan)
        rel[ok] = (np.log(s1[ok] / s1[ok].sum()) - np.log(sp_[ok] / sp_[ok].sum()))
        on = inside & (np.isfinite(s1) | np.isfinite(s0))
        d = np.where(on, np.nan_to_num(s1) - np.nan_to_num(s0), np.nan)
        lim = float(np.nanquantile(np.abs(d), 0.995))
        fig, axes = plt.subplots(3, 1, figsize=(26, 30))
        panels = [(rel, "the flow after against the no-buildings prior: log of its share over "
                   "the prior's (0: as in the open field)",
                   dict(cmap="RdBu_r", vmin=-3.0, vmax=3.0)),
                  (d, "the change in flow from the baseline: red the corridors opened, blue the "
                   "flow drawn away", dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(
                       lim * 1e-2, vmin=-lim, vmax=lim))),
                  (d, "the same, on 5810", dict(cmap="RdBu_r", norm=mcolors.SymLogNorm(
                      lim * 1e-2, vmin=-lim, vmax=lim)))]
        for k, (ax, (v, title, kw)) in enumerate(zip(axes, panels, strict=True)):
            im = ax.imshow(v, origin="lower", extent=ext, interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, shrink=0.6)
            gpd.GeoSeries(list(polys)).plot(ax=ax, color="0.75", lw=0)
            gpd.GeoSeries(list(polys[ids])).plot(ax=ax, facecolor="none", edgecolor="k", lw=0.6)
            ax.set_aspect("equal")
            ax.set_title(f"{label}, {preset} ({len(ids)} cleared): {title}", fontsize=14)
            if k == 2:
                bx = shapely.bounds(shapely.union_all(polys[sel]))
                ax.set_xlim(bx[0] - 20, bx[2] + 20)
                ax.set_ylim(bx[1] - 20, bx[3] + 20)
        fig.tight_layout()
        path = ROWS / "maps" / f"{label}_{preset}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=55)
        plt.close(fig)
        out.append(path)
    return out


def barrier_rows(b: Block) -> pd.DataFrame:
    """The barriers on the region (BARRIERS.parts): what they cut, the crossings by kind, the
    region's parts with and without the crossings' gaps, the footprints they touch."""
    bp = BARRIERS.parts(b)
    face = cast("BaseGeometry", b.boundary)
    rows = [dict(what="strips in the face (m^2)", value=float(shapely.area(
                 shapely.intersection(bp.strips, face)))),
            dict(what="water areas in the face (m^2)", value=float(shapely.area(
                 shapely.intersection(bp.areas, face)))),
            dict(what="cut from the face (m^2)", value=float(shapely.area(
                 shapely.intersection(bp.barrier, face)))),
            dict(what="the gaps (m^2)", value=float(shapely.area(bp.gaps)))]
    for kind in ("bridge", "tunnel", "node"):
        rows.append(dict(what=f"crossings: {kind}", value=float(
            sum(k == kind for k, _g in bp.crossings))))
    for what, cut in (("with the gaps", bp.barrier),
                      ("without them", shapely.union(bp.strips, bp.areas))):
        parts = [p for p in shapely.get_parts(shapely.make_valid(shapely.difference(face, cut)))
                 if shapely.area(p) > 1.0]
        rows.append(dict(what=f"the face's parts {what}", value=float(len(parts))))
        rows.append(dict(what=f"the largest part {what} (m^2)", value=float(max(
            shapely.area(p) for p in parts))))
    polys = np.asarray(b.buildings.outlines)
    rows.append(dict(what="footprints touching a barrier", value=float(
        shapely.intersects(polys, bp.barrier).sum())))
    return pd.DataFrame(rows)


def report() -> None:
    [b] = common.build_blocks([REGION])
    barrier_rows(b).to_csv(ROWS / "summary_barriers.csv", index=False)
    homes = homes_of(b, formal_grid(b))
    geo = gc.geometry(b)
    polys = geo.polys
    pop = common.AreaPopulation().weights(np.asarray(b.buildings.outlines))
    cost = pop / pop.sum()
    # the runs: their rows, scores and lenses
    runs = []
    for label, preset in itertools.product(CONFIGS, PRESETS):
        p = run_path(label, preset, REGION)
        if not p.exists():
            continue
        r = pd.read_parquet(p).iloc[0]
        runs.append(dict(label=label, preset=preset, objective=float(
            r.simp_perm if preset == "simp" else r.perm), t=float(r.t), wall=float(r.wall),
            gpu_gb=float(r.gpu_gb), peak_rss_gb=float(r.peak_rss_gb), host=r.host,
            free_D=float(r.free_D), metric=r.metric))
    runs_df = pd.DataFrame(runs)
    scores = pd.concat([pd.read_parquet(p) for p in sorted((ROWS / "score").glob("*.parquet"))],
                       ignore_index=True)
    lens = runs_df.merge(scores.rename(columns={"run": "preset"}), on=["label", "preset"],
                         how="left")
    lens.to_csv(ROWS / "summary_lenses.csv", index=False)
    scores[scores.run.str.startswith("raster")].to_csv(ROWS / "summary_raster.csv", index=False)
    base = pd.concat([pd.read_parquet(ROWS / "baseline" / f"{label}.parquet")
                      for label in CONFIGS if (ROWS / "baseline" / f"{label}.parquet").exists()],
                     ignore_index=True)
    base.to_csv(ROWS / "summary_baseline.csv", index=False)
    # self-absorption
    ab = []
    for label in CONFIGS:
        p = ROWS / "baseline" / f"{label}_absorb.parquet"
        if p.exists():
            a = pd.read_parquet(p)
            w: pd.Series = a["pop"] / a["pop"].sum()
            for col in ("own_building", "own_cells", "anywhere"):
                ab.append(dict(label=label, share=col, median=float(a[col].median()),
                               mean=float(a[col].mean()), pop_mean=float((a[col] * w).sum()),
                               p90=float(a[col].quantile(0.9)), max=float(a[col].max()),
                               n=len(a)))
    pd.DataFrame(ab).to_csv(ROWS / "summary_absorb.csv", index=False)
    # the loop index
    loop_rows().to_csv(ROWS / "summary_loops.csv", index=False)
    # other configurations' clearings under each
    crossed = sorted((ROWS / "cross").glob("*.parquet"))
    if crossed:
        pd.concat([pd.read_parquet(p) for p in crossed], ignore_index=True).to_csv(
            ROWS / "summary_cross.csv", index=False)
    # formal and deep homes' escape times, beside the per-block model's
    region_mean = float(homes.area.mean())
    mem = pd.concat([pd.read_parquet(p) for p in sorted((ROWS / "members").glob("*.parquet"))],
                    ignore_index=True)
    mem["u0_region"] = mem.u0 * mem.mean_area / region_mean
    per_block = (pd.DataFrame({"member": homes.member.astype(str), "member_row": homes.member_row})
                 .merge(mem, on=["member", "member_row"], how="left"))
    esc = []
    groups = {"formal": homes.formal, "deep": homes.deep, "grid": homes.grid,
              "all": np.ones(len(homes.area), bool)}
    for label in ["per-block", *CONFIGS]:
        if label == "per-block":
            u = per_block.u0_region.to_numpy()
            walled = per_block.walled.to_numpy().astype(bool)
        else:
            p = ROWS / "baseline" / f"{label}_homes.parquet"
            if not p.exists():
                continue
            hb = pd.read_parquet(p)
            u, walled = hb.u0.to_numpy(), hb.walled.to_numpy().astype(bool)
        for g, sel in groups.items():
            ok = sel & np.isfinite(u) & ~walled
            esc.append(dict(label=label, group=g, n=int(ok.sum()), median=float(np.median(u[ok])),
                            p90=float(np.quantile(u[ok], 0.9)), mean=float(u[ok].mean())))
    pd.DataFrame(esc).to_csv(ROWS / "summary_escape.csv", index=False)
    # the cleared sets
    sets = {(label, preset): cleared_of(label, preset)
            for label, preset in itertools.product(CONFIGS, PRESETS)
            if run_path(label, preset, REGION).exists()}
    raster = {}
    for key, path in RASTER_RUNS.items():
        r = pd.read_parquet(path).iloc[0]
        raster[key.removeprefix("raster_")] = np.asarray(
            r.simp_cleared if "simp_cleared" in r.index else r.cleared, dtype=np.int64)
    where = []
    for (label, preset), ids in sets.items():
        in_seed = homes.member[ids] == SEED_KBLOCK
        where.append(dict(label=label, preset=preset, n=len(ids), D=float(cost[ids].sum()),
                          seed_n=int(in_seed.sum()), seed_D_share=float(
                              cost[ids][in_seed].sum() / cost[ids].sum()),
                          grid_n=int(homes.grid[ids].sum()), grid_D_share=float(
                              cost[ids][homes.grid[ids]].sum() / cost[ids].sum()),
                          formal_n=int(homes.formal[ids].sum()), formal_D_share=float(
                              cost[ids][homes.formal[ids]].sum() / cost[ids].sum()),
                          deep_n=int(homes.deep[ids].sum()),
                          jaccard_bar=_jaccard(ids, sets[("bar", preset)])
                          if ("bar", preset) in sets else np.nan,
                          jaccard_other_preset=_jaccard(
                              ids, sets[(label, "cheap" if preset == "simp" else "simp")])
                          if (label, "cheap" if preset == "simp" else "simp") in sets else np.nan,
                          jaccard_raster=_jaccard(ids, raster[preset])))
    pd.DataFrame(where).to_csv(ROWS / "summary_cleared.csv", index=False)
    jac = [dict(preset=p, a=a, b=c, jaccard=_jaccard(sets[(a, p)], sets[(c, p)]))
           for p in PRESETS for a, c in itertools.combinations(CONFIGS, 2)
           if (a, p) in sets and (c, p) in sets]
    pd.DataFrame(jac).to_csv(ROWS / "summary_jaccard.csv", index=False)
    for label in CONFIGS:
        for path in maps(label, homes, polys):
            print(path, flush=True)
    print(lens.to_string(), flush=True)


def _single_threaded() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit(f"{v}=1 is required (single-threaded workers)")


def main(argv: list[str]) -> None:
    _single_threaded()
    cmd = argv[0] if argv else ""
    if cmd == "warm" and len(argv) == 3:
        warm(common.ids_of(argv[1])[0], argv[2])
    elif cmd == "run" and len(argv) == 5 and argv[3] in PRESETS:
        run(common.ids_of(argv[1])[0], argv[2], argv[3], argv[4])
    elif cmd == "run" and len(argv) == 5 and argv[3] == "prior" and argv[2] in CONFIGS:
        prior(common.ids_of(argv[1])[0], argv[2], argv[4])
    elif cmd == "members" and len(argv) == 2:
        members(int(argv[1]))
    elif cmd in ("baseline", "score") and len(argv) == 2 and argv[1] in CONFIGS:
        {"baseline": baseline, "score": score}[cmd](argv[1])
    elif cmd == "cross" and len(argv) == 3 and argv[1] in CONFIGS:
        sources = argv[2].split(",")
        unknown = sorted(set(sources) - set(CONFIGS))
        if unknown:
            raise SystemExit(f"unknown configurations {unknown}: {sorted(CONFIGS)}")
        cross(argv[1], sources)
    elif cmd == "report" and len(argv) == 1:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
