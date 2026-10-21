"""The region model on the complex: exits as leaks to ground, on 5810@major (BACKLOG 4, decision
(4); NOTES, "Exits as leaks: the region model on the complex (5810@major)" and "The region's
leaks: destinations at buildings and on open ground (5810@major)"; the briefs
.superpowers/sdd/2026-10-13-region-leaks and 2026-10-15-region-mixed-leak).

The configurations (CONFIGS, each a Config: its gap_exits.Exits, the metric's suffix naming them,
and the median trip its leak's constants were chosen for): the boundary only with no barriers
(`edge`: the raster run's setting, the per-block metric's name), the boundary with the barriers
(`bar`: the reference), the region test's leaks, set then by a share s at buildings and a nominal
lam 250, 500 and 1000 m and now the two constants they are on 5810@major (region_test_leak: all
on open ground, s 0, `L250s0` ...; all at the footprints, s 1, `L250s1` ...); the constants
chosen once on the closed reference for a median trip of TRIP_M with SHARES of the leak at
buildings (`T500b90`, `T500b75`, `T500b100`: CALIBRATED); and the street boost as a check (`S3`,
`L500s0S3`, `T500b90S3`), all with BARRIERS and the edge open but `edge`. Each is run with the
cheap preset (polish_greedy.CHEAP: its runs named `<metric>@e0.05`) and SIMP's default
(relax.SIMP_DEFAULT) at area D BUDGET with free first, optimizing P under the area population.

    S=research/roadless/region_leaks.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S warm <id> <label>              # the label's fixed mesh into gap_clear.CACHE
    uv run python $S run <id> <label> <simp|cheap|prior> <cpu|gpu> [--replace]   # a preset
    uv run python $S members <procs>                # the member kblocks' baselines (per block)
    uv run python $S blocks <procs>                 # one metric everywhere: the 13, per block
    uv run python $S loo-blocks <procs>             # who each clearing helps: the 13's presets
    uv run python $S loo <id> <label> <budget s> <cpu|gpu>   # ... the region's SIMP clearing
    uv run python $S calibrate <share> <T> <start> <procs>   # the constants (closed reference)
    uv run python $S baseline <label>               # the region's baseline under a configuration
    uv run python $S score <label>                  # its runs' clearings: lenses, flows, maps
    uv run python $S cross <label> <labels,>        # other configurations' clearings under it
    uv run python $S report                         # the summaries (ROWS/summary_*.csv)

`run`'s row (relax.one's or polish_greedy.run's, with the label, the metric's name, the median
trip its constants were chosen for (`trip_m`, None for the region test's), the host, the peak host
memory, the device's peak memory on a GPU and the wall time) in
ROWS/runs/<label>/<preset>/<metric>/<id>.parquet.

`calibrate`: the leak's two constants (owner 2026-10-15: set once, the same on every block and
region) with `share` of the leak at buildings and a median trip T (THE TRIPS, below) on the closed
reference: 5810@major with the barriers and its edge closed (gap_exits.ClosedEdge: every trip ends
at a leak), at baseline; the constants scale together (pair_of), each to whole metres. Not a step
of any run: the constants chosen are frozen in CALIBRATED (`report` checks they are the
calibration's). ROWS/calib/T<T>b<percent>.parquet (each step's scale, constants, median, the
pooled share leaking inside, time; the inputs; the closed system's check, closed_check: every live
component has a leak) and _boot.parquet (the bootstrap's medians at the constants chosen).

`members`: each member kblock of the region (regions.build's `member`) under the per-block metric
(its own street ring the exit) on a core: every home's escape time at baseline (the population's
demand), in the region's population units (AreaPopulation normalises each block to mean 1: a
kblock's potentials times its mean footprint area over the region's); ROWS/members/<kblock>.parquet.

`blocks`: one metric everywhere (owner 2026-10-15), checked per block on TUNING, on a core, no
re-optimization: each block under the per-block metric (`plain`) and under it with the central
constants (`leaky`, block_exits: the street ring the edge): P0, P_{1/2},0, J_2,0; under the leak
the share of the current leaking at baseline, and self-absorption (absorb: a typical home's share
leaking inside, `anywhere`, and at its own building); the stored SIMP-default and cheap-preset
clearings (relax_rows, CHEAP_ROWS) scored under each: Lens A on each lens.
ROWS/blocks/<id>.parquet.

`loo-blocks`, `loo`: who each clearing helps (owner 2026-10-16: "Clearing N people's homes to
provide access to n<N people's homes seems unfair (on average at least)"; a measurement, no
objective changed). Each cleared building of a final clearing restored alone (leave-one-out from
it): the population of the other homes standing whose escape time rises by at least LOO_RISES (or
that the restoring walls in) is what its clearing serves (`helped`), against its own population
(`displaced`), with the rise in P (`gain`). On TUNING's stored SIMP-default and cheap-preset
clearings under the per-block metric, on cores; on the region's SIMP clearing under a
configuration, on a GPU within a wall budget (a seeded random order: a cut leaves a sample).
ROWS/loo/<where>_<preset>.parquet; summary_loo.csv.

`baseline`: the region's model under a configuration on a core: P0, P_{1/2},0, J_2,0, the homes
walled in, the leaks' total and the share of the injected current they drain at baseline (the
rest leaves by the boundary), every home's escape time, and for a leaky configuration the
self-absorption of ABSORB_N homes drawn at random (a home's unit injection solved alone,
home_solves: the share of its current that leaks on the cells where its own building's injection
lands, its own building's share of the buildings' leak (InternalLeak.per_foot: the self-trips),
and the share that leaks anywhere); the flow painted (gap_maps) at baseline on FRAME_H pixels.
ROWS/baseline/<label>.parquet (and _homes, _absorb), ROWS/images/<label>/baseline.npz.

`run ... prior`: the no-buildings prior (every footprint cleared on the fixed mesh, the atlas's
prior: the same homes' flow to the same exits) and its P, painted on PRIOR_H pixels (the maps'
frame). Every cell is then an unknown (16M on the region), past SuperLU's 32-bit indices on a core:
it runs on a GPU (cuDSS); ROWS/baseline/<label>_prior.parquet (its row, the frame among it) and
<label>_prior_image.parquet (the pixels, flattened: parquet, as a cluster run's rows come back).

`score`: each run of the configuration (both presets; under `edge` also the raster run's picks,
RASTER_RUNS) scored on its model: L_P, L_{P_1/2}, L_J2 and the population left walled in; its flow
painted; ROWS/score/<label>.parquet, ROWS/images/<label>/<run>.npz.

`cross`: the runs of other configurations (both presets) scored under this one's model: what a
clearing chosen under one exit model is worth under another (the street boost's check);
ROWS/cross/<label>.parquet, its rows of the sources named replaced and the others' kept.

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

THE TRIPS (`calibrate`; NOTES, "The region's leaks"). At baseline under a configuration's leak, a
home's trips are its current leaking inside: its unit injection solved alone (home_solves), the
current to ground through each cell's leak (gap_maps.leak_drain) a trip's end, the trip's length
the straight-line distance from the centroid of the home's injection points (weighted as it
injects) to the cell's centroid, its weight that current. The current leaving by the boundary
(the street faces and the ground cells) is no trip of the leak and is not counted. Trips pool
over TRIP_N homes drawn by population (with replacement, seeded: among the homes with demand not
walled in at baseline), each draw its home's trips per unit of its injection; the median trip is
the pooled distribution's (TRIP_BIN_M bins, linear within a bin). lam is found by bisection in
log lam over whole metres, after a bracketing search (secant steps in log-log), the same draws
at every step, until the median is within TRIP_TOL of T; the draws' uncertainty on the median is
a bootstrap over the draws (TRIP_BOOT resamples). The median rises with lam toward a limit as the
leak vanishes (read at LIMIT_LAM_M: the trips then end where the no-leak potential times the
leak's shape puts them, nearly all the current leaving by the boundary): a T past it is refused
(Unreachable), its steps recorded. Checked where the answer is known (gap_clear_check
`trip-calibration`): on open ground at s 0 the median is the screened point source's, 1.257 lam.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/region_leaks.py
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import multiprocessing
import os
import socket
import sys
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
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
BARRIERS = metrics.BARRIERS             # RUNTIME's since the consolidated pass
STREET_CLASSES = ("residential", "living_street", "unclassified")   # a formal home's street
DRIVE_CLASSES = metrics.DRIVE_CLASSES   # the drivable streets: a deep home is far from all of them
# The street boost (gap_exits.StreetBoost): 3x on every way with a highway tag on 5810@major, any
# surface (the region test's check, its classes spelt out, `_link` folded), and 10x on the paved
# drivable streets alone (owner 2026-10-16: "a 10x boost is I think reasonable"; footways, paths
# and tracks not; 2026-10-20: "only boost if it's explicitly tagged paved")
REGION_HIGHWAYS = ("footway", "motorway", "path", "residential", "secondary", "service",
                   "services", "tertiary", "track", "unclassified")
BOOST = gx.StreetBoost(factor=3.0, surfaces=gx.AnySurface(), classes=REGION_HIGHWAYS)
DRIVE_BOOST = metrics.BOOST             # 10x on DRIVE_CLASSES tagged paved: RUNTIME's
TRIP_M = 500.0          # the median trip the constants are chosen for, on the closed reference (m;
                        # owner 2026-10-15: "What if we just use T=500m?")
SHARES = (0.9, 0.75, 1.0)   # the share of the leak at buildings on the closed reference: the
                            # central pair's (owner 2026-10-15: "buildings should have greater
                            # leakage"), and the sensitivity pairs'
# The reference's open ground live at baseline and the footprint area injected there (m^2:
# gap_exits.LeakSite, the bar configuration's): the region test's (s, lam) as constants
REF_OPEN_M2 = 5944597.3272020295
REF_FOOT_M2 = 437058.2681828446
LAMS = (250.0, 500.0, 1000.0)    # the region test's nominal lam (m), at s 0 and 1
# The constants chosen on the closed reference (`calibrate`; ROWS/calib/, summary_calib.csv): per
# share at buildings, frozen. The central pair (90%): a median trip of 499.5 m there (4,000 draws,
# the bootstrap's sd 3.2 m)
CALIBRATED: dict[float, gx.InternalLeak] = {
    0.9: metrics.LEAK,                                      # RUNTIME's: open_m 6214, foot_m 562
    0.75: gx.InternalLeak(open_m=3608.0, foot_m=565.0, sites=gx.LiveSites()),   # 500.1 m
    1.0: gx.InternalLeak(open_m=math.inf, foot_m=552.0,
                         sites=gx.LiveSites()),                  # 492.2 m (sd 3.3 m)
}


class Config(NamedTuple):
    """A configuration: its exits, and the median trip (m) its leak's constants were chosen for on
    the closed reference (`trip_m`: TRIP_M; None for the boundary only and the region test's)."""
    exits: gx.Exits
    trip_m: float | None


def _bar(leak: gx.NoLeak | gx.InternalLeak, boost: gx.NoBoost | gx.StreetBoost) -> gx.Exits:
    return gx.Exits(barriers=BARRIERS, edge=gx.OpenEdge(), leak=leak, boost=boost)


def region_test_leak(s: float, lam: float) -> gx.InternalLeak:
    """The region test's leak at the share s and the nominal lam (s 0 or 1), as the constants it
    is on 5810@major with the barriers: its total sigma REF_OPEN_M2 / lam^2 all on open ground
    (open_m lam) or all at the footprints (foot_m lam sqrt(REF_FOOT_M2 / REF_OPEN_M2))."""
    if s == 0.0:
        return gx.InternalLeak(open_m=lam, foot_m=math.inf, sites=gx.LiveSites())
    if s == 1.0:
        return gx.InternalLeak(open_m=math.inf, foot_m=lam * math.sqrt(REF_FOOT_M2 / REF_OPEN_M2),
                               sites=gx.LiveSites())
    raise ValueError(f"the region test's shares were 0 and 1, not {s}")


def calibrated_label(share: float) -> str:
    """`T<TRIP_M>b<percent at buildings>`."""
    return f"T{TRIP_M:g}b{100.0 * share:g}"


def _configs() -> dict[str, Config]:
    out = {"edge": Config(exits=gx.per_block(), trip_m=None),
           "bar": Config(exits=_bar(gx.NoLeak(), gx.NoBoost()), trip_m=None)}
    for s in (0.0, 1.0):
        for lam in LAMS:
            out[f"L{lam:g}s{s:g}"] = Config(exits=_bar(region_test_leak(s, lam), gx.NoBoost()),
                                            trip_m=None)
    out["S3"] = Config(exits=_bar(gx.NoLeak(), BOOST), trip_m=None)
    out["L500s0S3"] = Config(exits=_bar(region_test_leak(0.0, 500.0), BOOST), trip_m=None)
    for share, leak in CALIBRATED.items():
        out[calibrated_label(share)] = Config(exits=_bar(leak, gx.NoBoost()), trip_m=TRIP_M)
    central = CALIBRATED[SHARES[0]]
    out[calibrated_label(SHARES[0]) + "S3"] = Config(exits=_bar(central, BOOST), trip_m=TRIP_M)
    out[calibrated_label(SHARES[0]) + "D10"] = Config(exits=_bar(central, DRIVE_BOOST),
                                                      trip_m=TRIP_M)
    return out


CONFIGS = _configs()
PRESETS = ("simp", "cheap")
# the raster run's picks on 5810@major (NOTES, "A region whose only exit is its edge"; J_2, on
# the composite raster): their rows, read as records
RASTER_RUNS = {
    "raster_cheap": HERE / "polish_rows/uni/S0.01cat.P64w8.a5x8/D0.05" / f"{REGION}_p2.parquet",
    "raster_simp": (HERE / "relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2.a5x8/D0.05"
                    / f"{REGION}_p2.parquet")}
ABSORB_N = 300          # self-absorption: homes drawn
ABSORB_SEED = 20261015
TRIP_N = 4000           # the trips: homes drawn by population (with replacement; on the closed
                        # reference the bootstrap's sd at 2,000 draws was 0.9% of the median, so
                        # about 0.64% here, a third of TRIP_TOL)
TRIP_SEED = 20261016
TRIP_BIN_M = 1.0        # ... a trip's length binned (m)
TRIP_TOL = 0.02         # ... the bisection stops at a median within this of T (relative)
TRIP_BOOT = 2000        # ... the bootstrap's resamples of the draws
TRIP_BOOT_SEED = 20261017
SECANT_MAX = 4.0        # ... a bracketing step changes lam by at most this factor
LAM_RES_M = 1.0         # ... lam calibrated to whole metres (its name: InternalLeak.name)
LIMIT_LAM_M = 1e6       # ... the median's limit as the leak vanishes, read at this lam (m: the
                        # leak 1e-6 of lam 1000 m's)
FRAME_H = 1.0           # the painted flow's pixel (m)
PRIOR_H = 2.0           # the prior's and the maps' pixel (m: FRAME_H's 2 x 2, on the same edges)
FRAME_PAD = 2.0         # the frames' margin about the region (m: a multiple of both pixels)
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


def per_block(device: str) -> gcl.ComplexMetric:
    """The per-block metric (each kblock's whole street ring its exit: gap_exits.per_block, no
    barriers, no leak, no boost) on RUNTIME's complex, the model the region is set beside
    (`members`, `blocks`): pinned here, not RUNTIME itself, which switched to the region model's
    exits at the consolidated pass."""
    return dataclasses.replace(metrics.metric_of(metrics.RUNTIME, device=device),
                               exits=gx.per_block())


def metric_of(label: str, preset: str, device: str) -> gcl.ComplexMetric:
    """The configuration's metric for a preset (the cheap preset's search its own: `@e0.05`)."""
    base = dataclasses.replace(metrics.metric_of(metrics.RUNTIME, device=device),
                               exits=CONFIGS[label].exits)
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
    """A run's row: under its label and preset, and its metric's full name (metric_of: every
    writer keyed by its metric, so no row of another code or RUNTIME reads as this one's)."""
    return ROWS / "runs" / label / preset / metric_of(label, preset, "cpu").name / f"{bid}.parquet"


def run(bid: str, label: str, preset: str, device: str, *, replace: bool) -> None:
    """A preset under a configuration (module docstring), its row written once
    (clearings.write_once: a row standing under its identity refuses the run before any work,
    unless `replace`)."""
    metric = metric_of(label, preset, device)
    out = run_path(label, preset, bid)
    clearings.check_new(out, replace=replace)
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
    row.update(label=label, preset=preset, metric=metric.name, trip_m=CONFIGS[label].trip_m,
               host=socket.gethostname(),
               peak_rss_gb=gap_calib.peak_gb(), gpu_gb=gpu_gb, wall=time.time() - t0)
    clearings.write_once(pd.DataFrame([row]), out, replace=replace)
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
        m = per_block("cpu").clearing(b, demand=POP)
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


# ------------------------------------------------------------------ one metric everywhere

# common.TUNING's stored preset clearings under the per-block metric (per_block), P at area D
# BUDGET: SIMP's default (relax_rows) and the cheap preset, the translucent search (the
# acceptance's per-device rows, phase 3: no polish_rows of its own)
CHEAP_ROWS = gap_calib.rows_of(per_block("cpu").name) / "preset" / "gpu" / "cheap@e0.05"


def block_exits() -> gx.Exits:
    """The per-block metric with the central constants alone (the per-block check of 2026-10-16,
    before kblocks carried their OSM features: the block's street ring its edge, open; no barriers,
    no boost). RUNTIME adds both since the consolidated pass (metrics.EXITS)."""
    return gx.Exits(barriers=gx.NoBarriers(), edge=gx.OpenEdge(), leak=CALIBRATED[SHARES[0]],
                    boost=gx.NoBoost())


def _stored_clearings(bid: str, n: int) -> dict[str, NDArray[np.bool_]]:
    plain = per_block("cpu")
    simp = pd.read_parquet(relax.rows_of(PLAN, plain.name, BUDGET) / f"{bid}_p1.parquet").iloc[0]
    cheap = pd.read_parquet(CHEAP_ROWS / f"{bid}.parquet").iloc[0]
    if cheap["metric"] != CHEAP.on(plain).name:
        raise ValueError(f"{bid}: the cheap preset's row is {cheap['metric']}'s")
    out = {}
    for key, ids in (("simp", simp["simp_cleared"]), ("cheap", cheap["cleared"])):
        x = np.zeros(n, dtype=bool)
        x[np.asarray(ids, dtype=np.int64)] = True
        out[key] = x
    return out


def _block_one(bid: str) -> str | None:
    out = ROWS / "blocks" / f"{bid}.parquet"
    try:
        [b] = common.build_blocks([bid])
        plain = per_block("cpu")
        rows = []
        for name, metric in (("plain", plain), ("leaky", dataclasses.replace(
                plain, exits=block_exits()))):
            t0 = time.time()
            m = metric.clearing(b, demand=POP)
            st0 = m.state(np.zeros(m.n))
            base = dict(block=bid, model=name, metric=metric.name, n=m.n,
                        walled=int(m.walled0.sum()), P0=m.base.P, Pq0=m.base.Pq, J2_0=m.base.J2)
            if name == "leaky":
                fl = gap_maps.flow_of(m, st0, m.pop)
                a = absorb(bid, m, st0)
                w: pd.Series = a["pop"] / a["pop"].sum()
                base.update(leak_share=float(fl.leak.sum() / fl.drain.sum()),
                            inside_median=float(a["anywhere"].median()),
                            inside_pop=float((a["anywhere"] * w).sum()),
                            own_median=float(a["own_building"].median()),
                            own_pop=float((a["own_building"] * w).sum()),
                            own_p90=float(a["own_building"].quantile(0.9)),
                            own_max=float(a["own_building"].max()), absorb_n=len(a))
            for key, x in _stored_clearings(bid, m.n).items():
                L = m.lenses(x)
                base.update({f"{key}_L_P": 1.0 - L.P / m.base.P,
                             f"{key}_L_Pq": 1.0 - L.Pq / m.base.Pq,
                             f"{key}_L_J2": 1.0 - (L.J2 / m.base.J2) ** 0.5,
                             f"{key}_D": float(m.cost[x].sum())})
            base.update(t=time.time() - t0)
            rows.append(base)
        pd.DataFrame(rows).to_parquet(out)
        print(f"{bid}: {rows}", flush=True)
    except Exception as e:      # noqa: BLE001 -- reported, the pool goes on
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return bid
    return None


def blocks(procs: int) -> None:
    """The per-block check (module docstring) on TUNING."""
    (ROWS / "blocks").mkdir(parents=True, exist_ok=True)
    todo = [i for i in common.TUNING if not (ROWS / "blocks" / f"{i}.parquet").exists()]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        failed = [f for f in pool.imap_unordered(_block_one, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


# ------------------------------------------------------------------ who each clearing helps

LOO_RISES = (0.05, 0.01)    # leave-one-out: a home is helped by a cleared building when its
                            # escape time rises by at least this (relative) with the building
                            # restored (owner 2026-10-16: "Clearing N people's homes to provide
                            # access to n<N people's homes seems unfair")
LOO_SEED = 20261018         # ... the order the cleared buildings are restored in (a budget cut
                            # leaves a random sample)


def leave_one_out(m: gcl.Model, x: NDArray[np.bool_], *, budget_s: float,
                  seed: int) -> pd.DataFrame:
    """Each building cleared in `x` restored alone (module docstring), in a seeded random order
    until `budget_s` of wall time: its population (`displaced`), the rise in P (`gain`), and per
    threshold of LOO_RISES the population of the other homes standing in `x` (not cleared, with
    demand, reached) whose escape time rises by at least it, or that the restoring walls in
    (`helped<rise>`, and their count). P and the escape times under the area population."""
    st = m.state(m.k_of(x))
    u = m.solve(st, m.pop)
    uh0 = m.home_u(st, u)
    P0 = float(m._rhs(st, m.pop) @ u)
    stay = ~x & ~m.nodem & m.live_homes_of(st)
    order = np.random.default_rng(seed).permutation(np.flatnonzero(x))
    rows = []
    t0 = time.time()
    for b in order:
        if time.time() - t0 > budget_s:
            break
        x1 = x.copy()
        x1[b] = False
        st1 = m.state(m.k_of(x1))
        u1 = m.solve(st1, m.pop)
        uh1 = m.home_u(st1, u1)
        walled = stay & ~m.live_homes_of(st1)
        row: dict[str, float | int] = dict(
            building=int(b), displaced=float(m.pop[b]),
            gain=float(m._rhs(st1, m.pop) @ u1) - P0, walled=int(walled.sum()))
        for r in LOO_RISES:
            helped = walled | (stay & ~walled & (uh1 >= (1.0 + r) * uh0))
            row[f"helped{r:g}"] = float(m.pop[helped].sum())
            row[f"n_helped{r:g}"] = int(helped.sum())
        rows.append(row)
    print(f"leave-one-out: {len(rows)} of {int(x.sum())} cleared buildings in "
          f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows).assign(cleared=int(x.sum()), P0=P0, budget_s=budget_s, seed=seed)


def loo_path(where: str, preset: str) -> Path:
    return ROWS / "loo" / f"{where}_{preset}.parquet"


def _loo_block(bid: str) -> str | None:
    try:
        [b] = common.build_blocks([bid])
        m = per_block("cpu").clearing(b, demand=POP)
        for preset, x in _stored_clearings(bid, m.n).items():
            leave_one_out(m, x, budget_s=math.inf, seed=LOO_SEED).assign(
                where=bid, preset=preset, metric=per_block("cpu").name).to_parquet(
                    loo_path(bid, preset))
        print(f"{bid}: leave-one-out done", flush=True)
    except Exception as e:      # noqa: BLE001 -- reported, the pool goes on
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return bid
    return None


def loo_blocks(procs: int) -> None:
    """leave_one_out on TUNING's stored presets' clearings, under the per-block metric."""
    (ROWS / "loo").mkdir(parents=True, exist_ok=True)
    todo = [i for i in common.TUNING if not loo_path(i, "cheap").exists()]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        failed = [f for f in pool.imap_unordered(_loo_block, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


def loo_region(bid: str, label: str, budget_s: float, device: str) -> None:
    """Leave-one-out on the region's SIMP clearing under a configuration, within `budget_s`."""
    [b] = common.build_blocks([bid])
    metric = metric_of(label, "simp", device)
    m = metric.clearing(b, demand=POP)
    x = np.zeros(m.n, dtype=bool)
    x[cleared_of(label, "simp")] = True
    out = loo_path(f"{bid}_{label}", "simp")
    out.parent.mkdir(parents=True, exist_ok=True)
    leave_one_out(m, x, budget_s=budget_s, seed=LOO_SEED).assign(
        where=f"{bid} {label}", preset="simp", metric=metric.name, device=device,
        host=socket.gethostname()).to_parquet(out)


def loo_rows() -> pd.DataFrame:
    """Per clearing measured (ROWS/loo/): the share of its cleared buildings, and of the displaced
    population, whose helped population is below their own, the median helped / displaced, per
    threshold; the buildings done of those cleared."""
    out = []
    for p in sorted((ROWS / "loo").glob("*.parquet")):
        d = pd.read_parquet(p)
        row: dict[str, object] = dict(where=d["where"].iloc[0], preset=d["preset"].iloc[0],
                                      done=len(d), cleared=int(d["cleared"].iloc[0]),
                                      displaced=float(d["displaced"].sum()),
                                      gain_median=float(d["gain"].median()))
        for r in LOO_RISES:
            h = d[f"helped{r:g}"]
            under = h < d["displaced"]
            row.update({f"under{r:g}": float(under.mean()),
                        f"under_pop{r:g}": float(d.loc[under, "displaced"].sum()
                                                 / d["displaced"].sum()),
                        f"ratio_median{r:g}": float((h / d["displaced"]).median()),
                        f"helped_total{r:g}": float(h.sum())})
        out.append(row)
    return pd.DataFrame(out)


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
    if isinstance(m.exits.leak, gx.InternalLeak) and not absorbed.exists():
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


class HomeSolve(NamedTuple):
    """A home's unit injection solved alone at a factored state (home_solves): the home, its
    injection per cell on the state's unknowns (`b`: its column of System.Pc there), that
    injection's total (`q`: 1 less what lands on ground cells and the open band, escaping at
    once) and the potential per cell (`u`)."""
    home: int
    b: NDArray[np.float64]
    q: float
    u: NDArray[np.float64]


def home_solves(st: gcl.State, homes: NDArray[np.int64]) -> Iterator[HomeSolve]:
    """Each of `homes`' unit injection solved alone at the factored state `st`, one solve a home:
    self-absorption's and the trips'."""
    assert st.factor is not None
    Pc = st.sys.Pc.tocsc()
    unk = st.reach.unknown
    for h in homes:
        col = np.zeros(st.sys.n)
        lo, hi = Pc.indptr[h], Pc.indptr[h + 1]
        col[Pc.indices[lo:hi]] = Pc.data[lo:hi]
        b = col * unk
        yield HomeSolve(home=int(h), b=b, q=float(b.sum()), u=st.factor.solve(b))


def absorb(label: str, m: gcl.Model, st0: gcl.State) -> pd.DataFrame:
    """Self-absorption (module docstring) at baseline, ABSORB_N homes with demand drawn at random:
    each home's unit injection solved alone."""
    leak = st0.sys.graph.leak
    leak_family = m.exits.leak
    if not isinstance(leak_family, gx.InternalLeak):
        raise ValueError(f"{label}: self-absorption needs an internal leak")
    own_k = leak_family.per_foot(m.leak_site)
    foot = np.asarray(shapely.area(m.polys_), dtype=np.float64)
    ok = np.flatnonzero(~m.nodem & ~m.walled0)
    pick = np.sort(np.random.default_rng(ABSORB_SEED).choice(ok, min(ABSORB_N, len(ok)),
                                                              replace=False))
    rows = []
    for hs in home_solves(st0, pick):
        ring = hs.b > 0
        rows.append(dict(home=hs.home, injected=hs.q, pop=float(m.pop[hs.home]),
                         own_building=own_k * float(foot[hs.home]) * float(hs.b @ hs.u) / hs.q,
                         own_cells=float((leak[ring] * hs.u[ring]).sum()) / hs.q,
                         anywhere=float((leak * hs.u).sum()) / hs.q))
    print(f"{label}: self-absorption over {len(rows)} homes done", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the trips

class TripSite(NamedTuple):
    """Where a model's trips start and end (module docstring): per cell its centroid (`cell_xy`:
    gap_maps.cell_geometry), per home the centroid of its injection points (`home_xy`, weighted
    as it injects: gap_flow.PointDemand.S), and the bins (`nbins` of TRIP_BIN_M, past the
    farthest cell from any home)."""
    cell_xy: NDArray[np.float64]
    home_xy: NDArray[np.float64]
    nbins: int


def trip_site(m: gcl.Model) -> TripSite:
    cell_xy = gap_maps.cell_geometry(m.fm, m.c).xy
    S = m.dem.S
    w = np.asarray(S.sum(axis=0)).ravel()
    home_xy = np.asarray(S.T @ m.dem.xy) / np.maximum(w, 1e-300)[:, None]
    span = float(np.hypot(*(cell_xy.max(axis=0) - cell_xy.min(axis=0))))
    return TripSite(cell_xy=cell_xy, home_xy=home_xy,
                    nbins=int(np.ceil(span / TRIP_BIN_M)) + 2)


def trip_origins(site: TripSite, homes: NDArray[np.int64]) -> NDArray[np.float64]:
    """Where each of `homes`' trips start: its injection's centroid."""
    return site.home_xy[homes]


def trip_rows(st: gcl.State, site: TripSite, homes: NDArray[np.int64]) -> NDArray[np.float64]:
    """Per home of `homes` its internal trips (module docstring) at the factored state `st`: the
    current leaking at each cell under its unit injection (gap_maps.leak_drain: the boundary's
    drainage apart), binned by the straight-line distance from its injection's centroid to the
    cell's centroid (homes x TRIP_BIN_M bins)."""
    out = np.zeros((len(homes), site.nbins))
    origins = trip_origins(site, homes)
    for i, hs in enumerate(home_solves(st, homes)):
        mass = gap_maps.leak_drain(st, hs.u)
        on = np.flatnonzero(mass)
        d = np.hypot(*(site.cell_xy[on] - origins[i]).T)
        out[i] = np.bincount((d / TRIP_BIN_M).astype(np.int64), weights=mass[on],
                             minlength=site.nbins)
    return out


class _TripJob(NamedTuple):
    st: gcl.State
    site: TripSite


_JOB: list[_TripJob] = []      # the forked workers' state and site (set before the pool forks)


def _trip_chunk(homes: NDArray[np.int64]) -> NDArray[np.float64]:
    [job] = _JOB
    return trip_rows(job.st, job.site, homes)


def trip_rows_of(st: gcl.State, site: TripSite, homes: NDArray[np.int64],
                 procs: int) -> NDArray[np.float64]:
    """trip_rows over `procs` processes forked after the factorization (its factor shared), in
    `homes`' order."""
    if procs == 1:
        return trip_rows(st, site, homes)
    _JOB[:] = [_TripJob(st=st, site=site)]
    try:
        with multiprocessing.get_context("fork").Pool(procs) as pool:
            parts = pool.map(_trip_chunk, np.array_split(homes, 4 * procs))
    finally:
        _JOB.clear()
    return np.concatenate(parts)


def trip_median(hist: NDArray[np.float64]) -> float:
    """The median of a pooled histogram of trips (TRIP_BIN_M bins), linear within its bin."""
    cum = np.cumsum(hist)
    half = 0.5 * float(cum[-1])
    k = int(np.searchsorted(cum, half))
    below = float(cum[k - 1]) if k else 0.0
    return (k + (half - below) / float(hist[k])) * TRIP_BIN_M


class TripSample(NamedTuple):
    """The homes the trips are solved for (`homes`, each once) and each one's weight in the pool
    (`w`: its draws, or its population when every home is taken)."""
    homes: NDArray[np.int64]
    w: NDArray[np.float64]


def trip_sample(m: gcl.Model, *, n: int, seed: int) -> TripSample:
    """`n` homes drawn by population with replacement (seeded), among those with demand not walled
    in at baseline: each draw weighs alike."""
    ok = np.flatnonzero(~m.nodem & ~m.walled0)
    draws = np.random.default_rng(seed).choice(ok, n, replace=True, p=m.pop[ok] / m.pop[ok].sum())
    homes, counts = np.unique(draws, return_counts=True)
    return TripSample(homes=homes.astype(np.int64), w=counts.astype(np.float64))


def every_home(m: gcl.Model) -> TripSample:
    """Every home with demand not walled in at baseline, weighted by its population."""
    ok = np.flatnonzero(~m.nodem & ~m.walled0).astype(np.int64)
    return TripSample(homes=ok, w=m.pop[ok])


class TripStep(NamedTuple):
    """One scale of the calibration: the scale, the leak's constants there (gap_exits
    InternalLeak's lengths, m), its pooled median trip (m), the share of the pool's current
    leaking inside, and the factorization's and solves' time (s)."""
    scale: float
    open_m: float
    foot_m: float
    median: float
    inside: float
    t: float


def trip_step(m: gcl.Model, *, scale: float, leak: gx.InternalLeak, sample: TripSample,
              site: TripSite, procs: int) -> tuple[TripStep, NDArray[np.float64]]:
    """The median trip under `leak` (at `scale`, recorded) on `m`'s exits otherwise (its barriers,
    edge and boost), and the sample's rows."""
    t0 = time.time()
    st = m.leaked_state(leak, np.zeros(m.n))
    rows = trip_rows_of(st, site, sample.homes, procs)
    pooled = sample.w @ rows
    return TripStep(scale=scale, open_m=leak.open_m, foot_m=leak.foot_m,
                    median=trip_median(pooled), inside=float(pooled.sum() / sample.w.sum()),
                    t=time.time() - t0), rows


class Unreachable(Exception):
    """No scale gives the target median trip: the median at `limit_scale` (the leak vanishing),
    its least upper bound, is below it. `steps` the calibration's (the limit's), `rows` the
    limit's sample rows."""

    def __init__(self, steps: list[TripStep], rows: NDArray[np.float64]):
        super().__init__(f"the median trip rises to {steps[-1].median:.1f} m as the scale grows")
        self.steps = steps
        self.rows = rows


def calibrate_lam(m: gcl.Model, *, leak_of: Callable[[float], gx.InternalLeak], target: float,
                  tol: float, lam0: float, resolution: float, limit_lam: float,
                  sample: TripSample, site: TripSite, procs: int
                  ) -> tuple[list[TripStep], NDArray[np.float64]]:
    """The scale (a multiple of `resolution`, m) at which the sample's median trip under
    `leak_of(scale)` (its lengths in proportion to the scale) is within `tol` (relative) of
    `target`: first the median's limit as the leak vanishes (at `limit_lam`: the trips then end
    where the no-leak potential times the leak's shape puts them), Unreachable if the target is
    past it; then a bracketing search from lam0 by secant steps in log-log and bisection in log
    scale (module docstring). Every step, and the rows at the last (the scale chosen). The median
    falls as the leak grows: it rises with the scale, toward the limit."""
    def snap(x: float) -> float:
        return float(max(round(x / resolution), 1) * resolution)

    def step_at(scale: float) -> tuple[TripStep, NDArray[np.float64]]:
        return trip_step(m, scale=scale, leak=leak_of(scale), sample=sample, site=site,
                         procs=procs)
    limit, limit_rows = step_at(limit_lam)
    steps: list[TripStep] = [limit]
    print(f"  the limit (scale {limit_lam:g}): median {limit.median:.1f} m, inside "
          f"{limit.inside:.2e} ({limit.t:.0f}s)", flush=True)
    if limit.median < target * (1.0 - tol):
        raise Unreachable(steps, limit_rows)
    lo: TripStep | None = None
    hi: TripStep | None = None
    lam = snap(lam0)
    while True:
        step, rows = step_at(lam)
        steps.append(step)
        print(f"  scale {lam:g} (open {step.open_m:g} m, footprints {step.foot_m:g} m): median "
              f"{step.median:.1f} m, inside {step.inside:.3f} ({step.t:.0f}s)", flush=True)
        if abs(step.median / target - 1.0) <= tol:
            return steps, rows
        if step.median < target:
            lo = step
        else:
            hi = step
        if lo is not None and hi is not None:
            if hi.scale - lo.scale <= resolution * 1.5:
                raise RuntimeError(f"no scale between {lo.scale:g} and {hi.scale:g} (at "
                                   f"{resolution:g} m) gives a median within {tol:g} of "
                                   f"{target:g}")
            lam = min(max(snap((lo.scale * hi.scale) ** 0.5), snap(lo.scale + resolution)),
                      snap(hi.scale - resolution))
            continue
        # not yet bracketed: a secant step in log-log toward the target (slope 1 at first), at
        # most a factor SECANT_MAX
        slope = 1.0
        if len(steps) > 2:      # two steps past the limit's
            a, b = steps[-2], steps[-1]
            slope = float(np.log(b.median / a.median) / np.log(b.scale / a.scale))
        ratio = (target / step.median) ** (1.0 / slope) if slope > 0.0 else SECANT_MAX
        ratio = min(max(ratio, 1.0 / SECANT_MAX), SECANT_MAX)
        if (ratio > 1.0) != (step.median < target):
            raise RuntimeError(f"the median trip does not rise with the scale: {steps}")
        lam = snap(lam * ratio)


def trip_boot(rows: NDArray[np.float64], sample: TripSample, *, n: int,
              seed: int) -> NDArray[np.float64]:
    """The median trip over `n` bootstrap resamples of the sample's draws (each home's draws
    `sample.w` taken as draws)."""
    draws = np.repeat(np.arange(len(sample.homes)), sample.w.astype(np.int64))
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k0 in range(0, n, 100):
        idx = rng.integers(0, len(draws), size=(min(100, n - k0), len(draws)))
        W = np.stack([np.bincount(draws[i], minlength=len(sample.homes)) for i in idx])
        for j, pooled in enumerate(W.astype(np.float64) @ rows):
            out[k0 + j] = trip_median(pooled)
    return out


def calib_path(share: float) -> Path:
    return ROWS / "calib" / f"{calibrated_label(share)}.parquet"


def pair_of(share: float, scale: float, *, open_m2: float, foot_m2: float) -> gx.InternalLeak:
    """The leak's constants with `share` of it at buildings on a ground of `open_m2` open
    (live at baseline) and `foot_m2` of footprint injected there (m^2), its total sigma open_m2 /
    scale^2: open_m = scale / sqrt(1 - share), foot_m = scale sqrt(foot_m2 / (share open_m2))
    (inf where the share is 0), each to whole metres (LAM_RES_M: the frozen constants)."""
    def whole(x: float) -> float:
        return x if x == math.inf else float(max(round(x / LAM_RES_M), 1) * LAM_RES_M)
    return gx.InternalLeak(
        open_m=whole(scale / math.sqrt(1.0 - share)) if share < 1.0 else math.inf,
        foot_m=whole(scale * math.sqrt(foot_m2 / (share * open_m2))) if share > 0.0 else math.inf,
        sites=gx.LiveSites())


class ClosedCheck(NamedTuple):
    """The closed reference at baseline (`closed_check`): its live cells' components over the
    conducting passages, those without a leak (0: well posed), the cells live and the homes walled
    in, against the edge open."""
    components: int
    unleaked: int
    live: int
    live_open: int
    walled: int
    walled_open: int


def closed_check(m: gcl.Model, st: gcl.State, open_walled: int) -> ClosedCheck:
    """Every live component of the closed reference's baseline state `st` has a leak (a component
    without one would hold its potential up to a constant: singular)."""
    c, r = st.sys.cells, st.reach
    unk = r.unknown
    on = (r.Tp > 0) & unk[c.ea] & unk[c.eb]
    g = sp.csr_matrix((np.ones(int(on.sum())), (c.ea[on], c.eb[on])), shape=(c.n, c.n))
    _k, lab = connected_components(g, directed=False)
    comps = np.unique(lab[unk])
    leaky = np.unique(lab[unk & (st.sys.graph.leak > 0)])
    return ClosedCheck(components=len(comps), unleaked=len(np.setdiff1d(comps, leaky)),
                       live=int(unk.sum()), live_open=int(m.leak_site.live.sum()),
                       walled=int(m.walled0.sum()), walled_open=open_walled)


def calibrate(share: float, target: float, start: float, procs: int) -> None:
    """The leak's constants with `share` of it at buildings for a median trip `target` on the
    closed reference (module docstring): 5810@major with the barriers and its edge closed; the
    search from the scale `start` (m: recorded)."""
    [b] = common.build_blocks([REGION])
    exits = gx.Exits(barriers=BARRIERS, edge=gx.ClosedEdge(),
                     leak=gx.InternalLeak(open_m=target, foot_m=target, sites=gx.LiveSites()),
                     boost=gx.NoBoost())
    metric = dataclasses.replace(metrics.metric_of(metrics.RUNTIME, device="cpu"), exits=exits)
    t0 = time.time()
    m = metric.clearing(b, demand=POP)
    open_walled = int(metric_of("bar", "simp", "cpu").clearing(b, demand=POP).walled0.sum())
    site = trip_site(m)
    sample = trip_sample(m, n=TRIP_N, seed=TRIP_SEED)
    ls = m.leak_site
    open_m2 = float(ls.area[ls.live].sum())
    foot_m2 = float(ls.foot[ls.live].sum())
    print(f"calibrate {100 * share:g}% at buildings, T {target:g} m, the edge closed: the model "
          f"in {time.time() - t0:.0f}s; open ground {open_m2:.6g} m^2, footprints {foot_m2:.6g} "
          f"m^2; {len(sample.homes)} homes of {TRIP_N} draws, {site.nbins} bins", flush=True)
    out = calib_path(share)
    out.parent.mkdir(parents=True, exist_ok=True)

    def leak_of(scale: float) -> gx.InternalLeak:
        return pair_of(share, scale, open_m2=open_m2, foot_m2=foot_m2)

    def record(steps: list[TripStep], rows: NDArray[np.float64], *, reached: bool) -> None:
        """The steps' rows with the calibration's inputs, the bootstrap's at the last (the
        constants chosen, or the limit), the closed system's check at the last."""
        boot = trip_boot(rows, sample, n=TRIP_BOOT, seed=TRIP_BOOT_SEED)
        last = leak_of(steps[-1].scale)
        chk = closed_check(m, m.leaked_state(last, np.zeros(m.n)), open_walled)
        g_open = ls.sigma / last.open_m ** 2
        g_foot = last.per_foot(ls)
        df = pd.DataFrame([st._asdict() for st in steps])
        df = df.assign(share=share, target=target, start=start, region=REGION,
                       barriers=BARRIERS.name,
                       edge="closed", open_m2=open_m2, foot_m2=foot_m2,
                       share_at=g_foot * foot_m2 / (g_foot * foot_m2 + g_open * open_m2),
                       tol=TRIP_TOL, n_draws=TRIP_N, n_homes=len(sample.homes), seed=TRIP_SEED,
                       limit_scale=LIMIT_LAM_M, limit=df.index == 0,
                       chosen=reached & (df.index == len(df) - 1),
                       boot_sd=float(boot.std(ddof=1)), boot_lo=float(np.quantile(boot, 0.025)),
                       boot_hi=float(np.quantile(boot, 0.975)), **chk._asdict(),
                       peak_gb=gap_calib.peak_gb())
        df.to_parquet(out)
        pd.DataFrame({"median": boot}).to_parquet(out.with_name(out.stem + "_boot.parquet"))
        print(df.to_string(), flush=True)
    try:
        steps, rows = calibrate_lam(m, leak_of=leak_of, target=target, tol=TRIP_TOL,
                                    lam0=start, resolution=LAM_RES_M, limit_lam=LIMIT_LAM_M,
                                    sample=sample, site=site, procs=procs)
    except Unreachable as e:
        record(e.steps, e.rows, reached=False)
        raise SystemExit(f"{100 * share:g}% at buildings: no scale gives a median trip of "
                         f"{target:g} m ({e})") from e
    record(steps, rows, reached=True)


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
    out = ROWS / "cross" / f"{label}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_parquet(out)
        keep = cast("pd.DataFrame", old.loc[~old["source"].isin(sources)])
        new = pd.concat([keep, new], ignore_index=True)
    new.to_parquet(out)


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


def calib_rows() -> pd.DataFrame:
    """Every calibration's steps (ROWS/calib/<label>.parquet, labelled by its file); each share's
    constants in CALIBRATED (the configurations') must be its calibration's chosen ones."""
    paths = sorted(p for p in (ROWS / "calib").glob("T*.parquet")
                   if not p.stem.endswith("_boot"))
    df = pd.concat([pd.read_parquet(p).assign(label=p.stem) for p in paths], ignore_index=True)
    for share, leak in CALIBRATED.items():
        label = calib_path(share).stem
        chosen = df.loc[(df["label"] == label) & df["chosen"], ["open_m", "foot_m"]]
        if [tuple(r) for r in chosen.to_numpy()] != [(leak.open_m, leak.foot_m)]:
            raise ValueError(f"{label}: CALIBRATED's {leak}, its calibration's "
                             f"{chosen.to_numpy().tolist()}")
    return df


def report() -> None:
    [b] = common.build_blocks([REGION])
    barrier_rows(b).to_csv(ROWS / "summary_barriers.csv", index=False)
    calib_rows().to_csv(ROWS / "summary_calib.csv", index=False)
    pd.concat([pd.read_parquet(p) for p in sorted((ROWS / "blocks").glob("*.parquet"))],
              ignore_index=True).to_csv(ROWS / "summary_blocks.csv", index=False)
    loo_rows().to_csv(ROWS / "summary_loo.csv", index=False)
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
            free_D=float(r.free_D), metric=r.metric, trip_m=r.trip_m))
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
    argv, again = clearings.replace_flag(argv)
    cmd = argv[0] if argv else ""
    if cmd == "warm" and len(argv) == 3:
        warm(common.ids_of(argv[1])[0], argv[2])
    elif cmd == "run" and len(argv) == 5 and argv[3] in PRESETS:
        run(common.ids_of(argv[1])[0], argv[2], argv[3], argv[4], replace=again)
    elif cmd == "run" and len(argv) == 5 and argv[3] == "prior" and argv[2] in CONFIGS:
        prior(common.ids_of(argv[1])[0], argv[2], argv[4])
    elif cmd == "members" and len(argv) == 2:
        members(int(argv[1]))
    elif cmd == "blocks" and len(argv) == 2:
        blocks(int(argv[1]))
    elif cmd == "loo-blocks" and len(argv) == 2:
        loo_blocks(int(argv[1]))
    elif cmd == "loo" and len(argv) == 5 and argv[2] in CONFIGS:
        loo_region(common.ids_of(argv[1])[0], argv[2], float(argv[3]), argv[4])
    elif cmd == "calibrate" and len(argv) == 5 and float(argv[1]) in SHARES:
        calibrate(float(argv[1]), float(argv[2]), float(argv[3]), int(argv[4]))
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
