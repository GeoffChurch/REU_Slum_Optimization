"""The consolidated pass: every block under the final metric (BACKLOG 4, the owner's decisions of
2026-10-13 -- 16; NOTES, "The consolidated pass: every block under the final metric"; the brief
.superpowers/sdd/2026-10-16-consolidated-pass).

RUNTIME switched to the one metric (metrics.EXITS: the barriers, the leaks' central constants, the
10x boost on the drivable streets, the edge open), every block carrying its OSM features
(common.build_blocks). This module is the pass's survey of what the switch touches and its parts'
effect on the baselines; the lineup's runs are the optimizers' own scripts (relax.py,
polish_greedy.py, search.py, road_rescore.py), keyed by the metric's name.

    S=research/roadless/consolidated.py    # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S features <ids,|@file|store> <procs>   # what meets each block: ROWS/features
    uv run python $S baselines <id> <parts,>     # a block's baselines under the switch's parts
    uv run python $S presets <id> <cpu|gpu> <default,hard,cheap,cheap005> [--replace]
    uv run python $S lenses <id> [--replace]      # every preset clearing's lenses, on a core
    uv run python $S reuse-networks <scratch>    # the lineup's networks nothing changes
    uv run python $S report                      # ROWS/summary_*.csv
    uv run python $S lineup-report               # ROWS/summary_lineup_*.csv
    uv run python $S screen <id> <variants,> [--replace]   # phase 5's screens on a block, on a card
    uv run python $S screen-lenses <id> [--replace]        # their clearings' lenses, on a core
    uv run python $S screen-report <preset> <wave>         # each screen by epistemetry, its axes

`features`: per block (the store's 303: clearings.Store's blocks), its OSM features (attrs["osm"])
as the metric reads them: the barriers' cut (gap_exits.OsmBarriers.parts within the block: its
area, the railway's and water's, the crossings), and the highway lines meeting it by class (a
drivable one the boost's: of DRIVE_CLASSES and tagged paved), each split into the length on its
edge (within EDGE_M of the street ring) and inside (deeper: a way the boost raises inside the
block, not a street of its ring). Rows ROWS/features/<id>.parquet.

`presets`: the lineup's presets on a block under RUNTIME, one after another on one device
(items_of: SIMP's default `.p256w8x2`, the hard-block preset `.p256w8x2.GS0.01cate1`, the cheap
preset, and the cheap preset at S0.005cat), each free first at area D 0.05 optimizing P, its rows
in relax_rows/ and polish_rows/ under the metric's name, and the run's one-card timing
(ROWS/onecard/<metric>/<device>/<id>.parquet).

`screen` (phase 5, pruning): the dominated variants on record that still exist on the complex
(SCREENS), each in the default's company, run on a block one after another in this process on one
card beside SIMP's default (re-run there: its time on the same card), each free first at area D
0.05 optimizing P, under the screens' metric (SCREENS_METRIC, a screen campaign's: phase 5's is
the previous boost's, which this code no longer builds, so `screen` refuses to extend it); rows
screen_rows/<metric>/<plan>/D0.05/<id>_p1.parquet, relax_rows' layout (exported to the store; a
variant not built for the block's free stage, `.m` where free first holds buildings, writes none).
`screen-lenses`: their clearings scored on a core as `lenses` scores the presets'.
`screen-report <preset> <wave>`: each variant screened against the default by epistemetry's
`advance` under the named preset (screen_presets(), resolved once, here: a Screen whose record is
the screens' identity), over the variant's order (common.HELD's blocks it is built for,
relax.built_for_holds: `.m` not where free first holds buildings), its rows a prefix of it: Lens A
of P its gain, its GPU-seconds on one card its cost, the default's there on the blocks run, then
the pass's on the rest, the cost model's covariate (screening.arrived_of). Each wave continues
from the screens' stored ledgers (ROWS/screen_decisions.json: the metric, the Screen's record and
each screen's state, Running or Stopped, with the blocks it was decided on; a ledger of another
metric or preset, or on blocks the rows no longer start with, is refused, and removing the file
starts the screens afresh). Beside each screen's stop and call, every axis the dominance rule
reads: each lens's paired difference (L_P, L_{P_1/2}, L_J2: the mean, a 95% Bayesian-bootstrap
interval, ahead / behind), B's order, the time on one card (the ratio's geometric mean and its
interval), and the break-even rate (the mean gain over the mean extra GPU-seconds, on the blocks
used: the rate an ask turns on); the next blocks of each screen still running (its step's, or
`<wave>` if more). (The GPU's memory in use, the rows' gpu_gb, counts the pool earlier runs in the
process left: not an axis.) ROWS/summary_screen.csv, the Screen's record in its `screen` column.

`baselines`: a block's baseline (P0, P_{1/2},0, J_2,0, the homes walled in, the unknowns, the
mesh's triangles and times) under each of PARTS, cumulative: `block` the per-block metric
(metrics.PER_BLOCK), `leak` with the central leak, `bar` with the barriers too, `runtime` with
the boost too (RUNTIME). Rows ROWS/baselines/<id>__<part>.parquet, each under the metric's name.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import multiprocessing
import socket
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from epistemetry.design.decide import KEEP
from epistemetry.design.screen import (Arrived, Bootstrap, ProportionalTo, Running, Screen,
                                       Stopped, ValueOfMoreUnits, VetoBelow, advance)
from epistemetry.stats.records import revive

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_exits as gx  # noqa: E402
import metrics  # noqa: E402
import screening  # noqa: E402

ROWS = HERE / "consolidated_rows"
EDGE_M = 3.0            # a way within this of the street ring runs on the block's edge (m: the
                        # boost's own reach, gap_exits.BOOST_M), deeper it is inside the block
POP = common.PopulationDemand()


# ------------------------------------------------------------------ the switch's parts

def part_exits() -> dict[str, gx.Exits]:
    """The switch's parts, cumulative: the per-block exits, the leak, the barriers, the boost."""
    e = metrics.EXITS
    no = gx.per_block()
    return {"block": no,
            "leak": dataclasses.replace(no, leak=e.leak),
            "bar": dataclasses.replace(no, leak=e.leak, barriers=e.barriers),
            "runtime": e}


def part_metric(part: str, device: str) -> gcl.ComplexMetric:
    m = dataclasses.replace(metrics.metric_of(metrics.PER_BLOCK, device=device),
                            exits=part_exits()[part])
    if part == "runtime" and m.name != metrics.RUNTIME:
        raise AssertionError(f"the parts' last is {m.name}, not RUNTIME {metrics.RUNTIME}")
    return m


# ------------------------------------------------------------------ features

def store_ids() -> list[str]:
    """The committed store's blocks (the 220 small, the 82 large, 5810@major)."""
    return sorted(clearings.Store.committed().prints())


def ids_arg(arg: str) -> list[str]:
    return store_ids() if arg == "store" else common.ids_of(arg)


def _features_one(bid: str) -> str | None:
    out = ROWS / "features" / f"{bid}.parquet"
    try:
        [b] = common.build_blocks([bid])
        osm = gx.osm_in(b)
        poly = b.boundary
        edge = shapely.buffer(shapely.union_all(b.streets.geometry.to_numpy()), EDGE_M)
        parts = metrics.BARRIERS.parts(b)
        cut = shapely.intersection(poly, parts.barrier)
        rail = osm.ways[osm.ways.railway.isin(gx.RAIL)]
        wet = osm.ways[osm.ways.waterway.isin(gx.WATERWAYS)]
        row: dict[str, object] = dict(
            block=bid, n=len(b.buildings.outlines), area_m2=float(poly.area),
            size=gap_calib.size_of(bid) if "@" not in bid else "region",
            barrier_m2=float(cut.area), barrier_share=float(cut.area / poly.area),
            rail_lines=len(rail), water_lines=len(wet),
            water_areas=int((osm.areas.natural.isin(gx.WATER_NATURAL)
                             | osm.areas.landuse.isin(gx.WATER_LANDUSE)
                             | osm.areas.waterway.isin(gx.WATER_WATERWAY)).sum()),
            crossings=len(parts.crossings))
        hw = osm.ways[osm.ways.highway != ""]
        cls = hw.highway.str.removesuffix("_link")
        geom = hw.geometry.to_numpy()
        inside = shapely.intersection(geom, poly)
        on_edge = shapely.length(shapely.intersection(inside, edge))
        deep = shapely.length(inside) - on_edge
        drive = cls.isin(metrics.DRIVE_CLASSES).to_numpy() & metrics.BOOST.surfaces.of(hw.surface)
        foot = cls.isin(("footway", "path", "pedestrian", "steps", "track", "cycleway")).to_numpy()
        row.update(drive_edge_m=float(on_edge[drive].sum()),
                   drive_inside_m=float(deep[drive].sum()),
                   foot_inside_m=float(deep[foot].sum()),
                   other_inside_m=float(deep[~drive & ~foot].sum()),
                   drive_classes=" ".join(sorted(set(cls[drive & (deep > 0)]))))
        out.parent.mkdir(parents=True, exist_ok=True)
        clearings.write_once(pd.DataFrame([row]), out, replace=True)
        print(f"{bid}: barrier {row['barrier_share']:.4f}, drivable inside "
              f"{row['drive_inside_m']:.0f} m, on the edge {row['drive_edge_m']:.0f} m", flush=True)
    except Exception as e:      # noqa: BLE001 -- reported, the pool goes on
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:400], flush=True)
        return bid
    return None


def features(ids: list[str], procs: int) -> None:
    todo = [i for i in ids if not (ROWS / "features" / f"{i}.parquet").exists()]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=8) as pool:
        failed = [f for f in pool.imap_unordered(_features_one, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


# ------------------------------------------------------------------ baselines

PARTS = ("block", "leak", "bar", "runtime")


def baseline_path(bid: str, part: str) -> Path:
    return ROWS / "baselines" / part_metric(part, "cpu").name / f"{bid}.parquet"


def baselines(bid: str, parts: list[str], *, replace: bool) -> None:
    """The block's baseline under each part (module docstring), one after another, each written
    once under its metric's name."""
    [b] = common.build_blocks([bid])
    for part in parts:
        out = baseline_path(bid, part)
        clearings.check_new(out, replace=replace)
        metric = part_metric(part, "cpu")
        t0 = time.time()
        fx = metric.fixed(b, cache=gcl.CACHE, carving=gcl.Carving.none(gc.geometry(b)))
        t_mesh = time.time() - t0
        m = metric.model(b, fx, demand=POP)
        row = dict(block=bid, part=part, metric=metric.name, n=m.n, P0=m.base.P, Pq0=m.base.Pq,
                   J2_0=m.base.J2, walled=int(m.walled0.sum()),
                   walled_share=float(m.cost[m.walled0].sum()), nodem=int(m.nodem.sum()),
                   unknowns=m.cells0, nt=fx.fm.nt, t_build=fx.t_build, t_mesh=t_mesh,
                   t_model=time.time() - t0 - t_mesh, host=socket.gethostname(),
                   peak_rss_gb=gap_calib.peak_gb())
        clearings.write_once(pd.DataFrame([row]), out, replace=replace)
        print(f"{bid} {part}: P0 {row['P0']:.6g} Pq0 {row['Pq0']:.6g} J2_0 {row['J2_0']:.6g} "
              f"walled {row['walled']}; {fx.fm.nt} triangles, mesh {t_mesh:.0f}s, model "
              f"{row['t_model']:.0f}s, peak {row['peak_rss_gb']:.1f} GB", flush=True)
        del m, fx


# ------------------------------------------------------------------ the presets (the lineup)

BUDGET = 0.05           # area D (Lens A's budget: relax.D_LENS)
POWER = 1.0             # P, the headline (decision (1)): what every preset optimizes
HARD = ".GS0.01cate1"   # the hard-block preset: the default plus the greedy's gated second track
CHEAP_005 = "S0.005cat"  # the measurement gap: the cheap preset with the finer batch's real
                         # clearing (BACKLOG, "Batch size": S0.005cat, its rows interpolated)


class Item(NamedTuple):
    """A preset of the lineup on one device: its rows' path and how it runs."""
    name: str
    path: Path
    run: Callable[[], dict[str, object]]


def items_of(bid: str, device: str) -> dict[str, Item]:
    """The lineup's presets on `bid` under RUNTIME, on `device` (the polish scored exactly on a
    core, in the eps world on a GPU: phase 2a, the same clearings), each a free-first run at area
    D BUDGET optimizing P, its rows where its optimizer keeps them (relax.rows_of,
    polish_greedy.rows_of: exported to the store), keyed by the metric's name."""
    import polish_greedy
    import relax
    import search
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    scoring = search.polish_scoring_of({"cpu": "E", "gpu": ""}[device])
    default = relax.plan_of(relax.SIMP_DEFAULT)._replace(polish_scoring=scoring)
    hard = relax.plan_of(default.name + HARD)
    c = polish_greedy.CHEAP

    def simp(plan: Any) -> Item:
        return Item(name=plan.name, path=relax.rows_of(plan, metric.name, BUDGET)
                    / f"{bid}_p{POWER:g}.parquet",
                    run=lambda: relax.one(bid, POWER, metric, plan, BUDGET, demand=POP))

    def cheap(picker: str) -> Item:
        on = c.on(metric)
        return Item(name=f"{picker}.P{c.tries}w{c.width}{scoring.suffix}@{on.name.split('@')[1]}",
                    path=polish_greedy.rows_of(picker, c.tries, c.width, c.pairs, scoring,
                                               on.name, BUDGET) / f"{bid}_p{POWER:g}.parquet",
                    run=lambda: polish_greedy.run(bid, POWER, on, picker, c.tries, c.width,
                                                  BUDGET, c.pairs, scoring=scoring))
    return {"default": simp(default), "hard": simp(hard), "cheap": cheap(c.picker),
            "cheap005": cheap(CHEAP_005)}


def onecard_path(bid: str, device: str) -> Path:
    return ROWS / "onecard" / metrics.RUNTIME / device / f"{bid}.parquet"


def presets(bid: str, device: str, names: list[str], *, replace: bool) -> None:
    """The lineup's presets `names` on `bid`, one after another in this process (on a GPU: one
    card), each row written once with the device, the host, the wall time and the peak memory;
    and the whole run's timing, its setup (the block built and the first mesh) included
    (`onecard`: the measurement gap, BACKLOG's "one-card timing of the whole lineup with
    setup")."""
    t_task = time.time()
    items = items_of(bid, device)
    unknown = sorted(set(names) - set(items))
    if unknown:
        raise SystemExit(f"unknown presets {unknown}: {sorted(items)}")
    for k in names:
        clearings.check_new(items[k].path, replace=replace)
    clearings.check_new(onecard_path(bid, device), replace=replace)
    timing: dict[str, object] = dict(block=bid, device=device, metric=metrics.RUNTIME,
                                     host=socket.gethostname(), presets=",".join(names))
    for k in names:
        item = items[k]
        t0 = time.time()
        if device == "gpu":
            with gap_calib.GpuPeak() as peak:
                row = item.run()
            gpu_gb = peak.peak
        else:
            row, gpu_gb = item.run(), float("nan")
        wall = time.time() - t0
        row.update(device=device, host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
                   gpu_gb=gpu_gb, wall=wall)
        clearings.write_once(pd.DataFrame([row]), item.path, replace=replace)
        timing.update({f"wall_{k}": wall, f"t_{k}": float(cast(float, row["t"])),
                       f"gpu_gb_{k}": gpu_gb})
        print(f"{bid} {k} ({item.name}) on {device}: {wall:.0f}s, peak "
              f"{gap_calib.peak_gb():.1f} GB, GPU {gpu_gb:.1f} GB", flush=True)
    timing.update(wall=time.time() - t_task, peak_rss_gb=gap_calib.peak_gb())
    clearings.write_once(pd.DataFrame([timing]), onecard_path(bid, device), replace=replace)


# ------------------------------------------------------------------ every clearing's lenses

def lenses_path(bid: str) -> Path:
    return ROWS / "lenses" / metrics.RUNTIME / f"{bid}.parquet"


def lenses(bid: str, *, replace: bool) -> None:
    """Every preset clearing of `bid` under RUNTIME (the lineup's rows on either device: items_of's
    paths), each scored on a core under RUNTIME on P, P_{1/2} and J_2 (Lens A at its D) with B (i),
    as road_rescore's rows are (`_lens_row`), the baseline's first: the presets' lenses and the
    price of roads' clearings. Rows ROWS/lenses/<metric>/<id>.parquet."""
    import road_rescore as rr
    out = lenses_path(bid)
    clearings.check_new(out, replace=replace)
    [b] = common.build_blocks([bid])
    m = metrics.metric_of(metrics.RUNTIME, device="cpu").clearing(b, demand=POP)
    rows = [dict(preset="baseline", device="", run="", n_cleared=0, D=0.0, stored_perm=np.nan,
                 **rr._lens_row(m, m.base))]
    for device in ("gpu", "cpu"):
        for k, item in items_of(bid, device).items():
            if not item.path.exists():
                continue
            r = pd.read_parquet(item.path).iloc[0]
            simp = "simp_cleared" in r.index
            ids = np.asarray(r["simp_cleared"] if simp else r["cleared"], dtype=np.int64)
            x = np.zeros(m.n, dtype=bool)
            x[ids] = True
            rows.append(dict(preset=k, device=device, run=item.name, n_cleared=int(x.sum()),
                             D=float(m.cost[x].sum()),
                             stored_perm=float(r["simp_perm"] if simp else r["perm"]),
                             **rr._lens_row(m, m.lenses(x))))
    clearings.write_once(pd.DataFrame(rows).assign(block=bid, metric=metrics.RUNTIME),
                         out, replace=replace)
    print(f"{bid}: {len(rows) - 1} clearings scored", flush=True)


# ------------------------------------------------------------------ phase 5: the screens

SCREENS = {   # variant -> the plan's suffix on SIMP's default ("": the default itself)
    "default": "",
    "r2l2": ".r2l2",        # late random roundings (BACKLOG, "Sturdier gate capture")
    "pP": ".pP",            # the polish over the undecided, then over every building
    "seed": ".gS0.01cat",   # the greedy's clearing offered to the incumbent first
    "m2": ".m2",            # a second SIMP start (BACKLOG, "Multiple starts")
    "mma": ".umma0.2",      # MMA's update (BACKLOG, "Damped OC or MMA")
    "doc": ".uoc0.3m0.2",   # damped OC (eta 0.3)
}

# The screens' rule and its numbers, set here at the edge (screening.py: the rule a Strategy, each
# preset an epistemetry Screen whose record is the run's identity), each with where it came from.
RATES_PER_GPU_SECOND = (2e-6, 1.9e-5)   # Lens A per GPU-second, bracketed by two calls on record
                        # (NOTES): .p256w8x2 adopted over .p64w8 at +0.00019 for 10.4 s a block (at
                        # most 1.9e-5), late sampling closed at +0.0001 for ~76 s (at least ~2e-6)
RATE_POINTS = 5         # rates across the bracket (geometric, equally weighted) the rule averages
FUTURE = 820.0          # block runs a decision governs, the 82 large blocks about ten times
                        # over; replayed from 82 to 8200 it moved one screen's stop, 15 -> 24
                        # blocks (NOTES)
SCREEN_FLOOR = 15       # replayed on six held-out screens (NOTES): lower realized loss than 20 at
                        # 11 of 15 rates x futures, than 10 at 9; below 15 the bootstrap's 90%
                        # interval misses the 46-block mean 12 -- 21% of the time at 10 blocks
SCREEN_STEP = 1         # blocks a screen runs past the floor before deciding again: one, as
                        # seqscreen re-decided after every block
COLLAPSE = 0.1          # BACKLOG, "Screen before the full run": a block this far below the default
                        # (Lens A) stops the variant at once, the default kept
SCREEN_OUTER = 4000     # outer draws of the next blocks a value is read from (seqscreen's DRAWS,
                        # its Polya-urn paths per value)
SCREEN_INNER = 200      # inner draws of the rest given them (the replays')
SCREEN_BLOCKS = 20      # batch-means blocks of a value's Monte Carlo standard error (the replays')
SCREEN_DRAWS = 4000     # draws a stop and a call read (seqscreen's DRAWS)
SCREEN_SEED = (0,)      # the live screens' draws, keyed by it and each draw's place (epistemetry
                        # `stream`)
SCREEN_MARGIN = 0.0     # Monte Carlo standard errors a value must clear its price by: none, as
                        # seqscreen had none
# The horizons a screen values: one Screen per preset, whatever the population. ValueOfMoreUnits
# caps each horizon at the blocks left, so a screen of u <= 46 blocks values 1..left exactly as the
# fourth replay's range(1, u + 1) did. A screen's order is common.HELD's blocks it is built for, so
# none is longer; the fixture's loader refuses a design that is (screening_check.designs).
SCREEN_HORIZONS = tuple(range(1, len(common.HELD) + 1))
# The metric the screens' rows, lenses and ledgers are under: a screen campaign's, since a ledger
# continued on another metric's rows would decide on other numbers. Phase 5's screens ran under the
# previous boost (every drivable way, whatever its surface: RUNTIME before 7e3af8dc), which this
# code no longer builds, so they are read, never extended. A screen campaign under another metric
# moves it and starts its ledger afresh (removing ROWS/screen_decisions.json).
SCREENS_METRIC = ("cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a+xB8c3Lo6214b562S10hliving_street-"
                  "motorway-primary-residential-secondary-service-tertiary-trunk-unclassified"
                  "va03a092b")
# The rule's rate points, a tuple of floats: a record holds a tuple, and an array (as seqscreen
# built its rates) would not record.
SCREEN_RATES = tuple(float(r) for r in np.geomspace(*RATES_PER_GPU_SECOND, RATE_POINTS))


def screen_presets() -> dict[str, Screen]:
    """The screens' presets, by the name a command line gives: each an epistemetry Screen over the
    bracket's RATE_POINTS geometric rates (SCREEN_RATES), equally weighted, over FUTURE uses, its
    stop's margin SCREEN_MARGIN Monte Carlo standard errors, its rule (screening.Rule) the one
    difference."""
    weights = (1.0 / RATE_POINTS,) * RATE_POINTS

    def screen_of(rule: screening.Rule) -> Screen:
        policy = ValueOfMoreUnits(
            belief=Bootstrap(), functionals=((1.0, 0.0), (0.0, 1.0)),  # gain, extra GPU-s
            target=rule.target_of(), horizons=SCREEN_HORIZONS,
            step=SCREEN_STEP, cost=ProportionalTo(covariate=screening.BASE_SECONDS),
            stop=rule.stop(),
            floor=SCREEN_FLOOR, outer=SCREEN_OUTER, inner=SCREEN_INNER, blocks=SCREEN_BLOCKS,
            draws=SCREEN_DRAWS)
        return Screen(policy=policy, drop=VetoBelow(threshold=-COLLAPSE, column=0, call=KEEP),
                      seed=SCREEN_SEED)

    return {
        # The default. Optimizes the realized loss: AdoptKeepAskByStake over the bracket's rates, a
        # wrong call losing |gain - rate x extra GPU-seconds| over FUTURE uses; in the fourth replay
        # its realized loss below seqscreen's at every rate, and below fewest-wrong's at the
        # bracket's bottom and middle (fewest-wrong's lower at its top). Gives up: more wrong calls
        # (67 against fewest-wrong's 37 in 1,400 distinct pairwise runs), for fewer asks (284
        # against 318).
        "realized-loss": screen_of(screening.RealizedLoss(rates=SCREEN_RATES, weights=weights,
                                                          uses=FUTURE, margin=SCREEN_MARGIN)),
        # Optimizes the wrong calls: AdoptKeepAsk over the bracket's ends, each arm priced by its
        # own stake; 37 wrong calls against 67 in 1,400 distinct pairwise runs of the fourth replay
        # (each design without the veto, C1m4/p64 with it too). Gives up: more asks (318 against
        # 284), and more realized loss at the bracket's bottom and middle.
        "fewest-wrong": screen_of(screening.ZeroOne(rates=SCREEN_RATES, weights=weights,
                                                    uses=FUTURE, margin=SCREEN_MARGIN)),
    }


def screen_plan(variant: str) -> Any:
    """SCREENS' plan for `variant`: the option spliced into SIMP's default where plan_of's
    grammar places it (the samples and the polish phases inside, the rest after)."""
    import relax
    base, suffix = relax.SIMP_DEFAULT, SCREENS[variant]
    if suffix in (".r2l2", ".pP"):
        head, polish = base.rsplit(".p", 1)
        spec = head + suffix + (".p" if suffix == ".r2l2" else "") + polish
    else:
        spec = base + suffix
    plan = relax.plan_of(spec)
    assert plan.name == spec, (plan.name, spec)
    return plan


def screen_path(variant: str, bid: str, *, metric: str) -> Path:
    """`variant`'s row on `bid` under `metric`: relax_rows' layout under screen_rows/ (SIMP's
    default re-run here would otherwise meet the pass's own row)."""
    plan = str(screen_plan(variant).name)
    return (HERE / "screen_rows" / metric / plan / f"D{BUDGET:g}"
            / f"{bid}_p{POWER:g}.parquet")


def screens_metric(metric: str, *, device: str) -> gcl.ComplexMetric:
    """The screens' metric built on `device`, refused where this code no longer builds it (a
    campaign's rows under a metric whose code has moved on are read, never extended)."""
    try:
        return metrics.metric_of(metric, device=device)
    except ValueError as e:
        raise SystemExit(f"the screens' metric {metric} does not build under this code ({e}): its "
                         f"screens are finished, read by screen-report and never extended; a "
                         f"screen campaign under RUNTIME moves SCREENS_METRIC and starts its "
                         f"ledger afresh") from e


def screen(bid: str, variants: list[str], *, replace: bool, metric: str) -> None:
    """Phase 5's screens on `bid` under `metric`: each of `variants` (SCREENS) in turn in this
    process, on one card, each row written once with the device, the host, the wall time and the
    peak memory (as `presets` writes the presets')."""
    import relax
    unknown = sorted(set(variants) - set(SCREENS))
    if unknown:
        raise SystemExit(f"unknown variants {unknown}: {sorted(SCREENS)}")
    m = screens_metric(metric, device="gpu")
    for v in variants:
        clearings.check_new(screen_path(v, bid, metric=metric), replace=replace)
    for v in variants:
        plan = screen_plan(v)
        t0 = time.time()
        with gap_calib.GpuPeak() as peak:
            try:
                row = relax.one(bid, POWER, m, plan, BUDGET, demand=POP)
            except NotImplementedError as e:    # `.m`: not built for the free stage's holds
                print(f"{bid} {v}: {e}", flush=True)
                continue
        wall = time.time() - t0
        row.update(device="gpu", host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
                   gpu_gb=peak.peak, wall=wall)
        clearings.write_once(pd.DataFrame([row]), screen_path(v, bid, metric=metric),
                             replace=replace)
        print(f"{bid} {v} ({plan.name}): {wall:.0f}s, GPU {peak.peak:.1f} GB", flush=True)


def screen_lenses_path(bid: str, *, metric: str) -> Path:
    return ROWS / "screen_lenses" / metric / f"{bid}.parquet"


def screen_lenses(bid: str, *, replace: bool, metric: str) -> None:
    """Every screen clearing of `bid` scored on a core under `metric`, as `lenses` scores the
    presets'. Rows ROWS/screen_lenses/<metric>/<id>.parquet."""
    import road_rescore as rr
    out = screen_lenses_path(bid, metric=metric)
    built = screens_metric(metric, device="cpu")
    clearings.check_new(out, replace=replace)
    [b] = common.build_blocks([bid])
    m = built.clearing(b, demand=POP)
    rows = []
    for v in SCREENS:
        if not screen_path(v, bid, metric=metric).exists():
            continue
        r = pd.read_parquet(screen_path(v, bid, metric=metric)).iloc[0]
        x = np.zeros(m.n, dtype=bool)
        x[np.asarray(r["simp_cleared"], dtype=np.int64)] = True
        rows.append(dict(variant=v, n_cleared=int(x.sum()), D=float(m.cost[x].sum()),
                         stored_perm=float(r["simp_perm"]), **rr._lens_row(m, m.lenses(x))))
    clearings.write_once(pd.DataFrame(rows).assign(block=bid, metric=metric),
                         out, replace=replace)
    print(f"{bid}: {len(rows)} screen clearings scored", flush=True)


def _bb_interval(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """The 95% interval of the mean of `d` under the Bayesian bootstrap (Rubin 1981): a reporting
    interval, not a decision (the screens decide through epistemetry)."""
    w = rng.dirichlet(np.ones(len(d)), size=4000)
    lo, hi = np.quantile(w @ d, [0.025, 0.975])
    return float(lo), float(hi)


class ScreenData(NamedTuple):
    """A variant's screen as its rows give it: its order (common.HELD's blocks it is built for), the
    blocks it has run (a prefix of the order), and those rows as epistemetry's Arrived
    (screening.arrived_of: its gain and GPU-seconds on the blocks run, the default's GPU-seconds on
    every block of the order)."""
    eligible: tuple[str, ...]
    run: tuple[str, ...]
    arrived: Arrived


def _rows_by_block(d: Path) -> pd.DataFrame:
    """Every row under `d`, by block; refused where there is none (no run under that metric)."""
    rows = _read(d)
    if rows.empty:
        raise SystemExit(f"{d}: no rows")
    return rows.set_index("block")


def _pass_rows(*, metric: str) -> pd.DataFrame:
    """SIMP's default in the pass under `metric` (relax_rows), by block: free first's clearing (does
    it hold buildings there?) and the GPU-seconds, a screen's forecast on the blocks it has not
    run."""
    import relax
    return _rows_by_block(relax.rows_of(relax.plan_of(relax.SIMP_DEFAULT), metric, BUDGET))


def screen_data(*, metric: str) -> dict[str, ScreenData]:
    """Every variant's screen but the default's, from the rows under `metric`, whatever the preset.
    Refused before any decision (SystemExit, naming it): a held-out block with no default time in
    the pass's rows (the cost model's covariate on a block not run, which epistemetry's Arrived
    would refuse without naming the block), and a variant's rows off its order or without the
    default's beside them."""
    import relax
    passed = _pass_rows(metric=metric)
    seconds = passed["t"].reindex(list(common.HELD)).to_numpy(dtype=float)
    untimed = [k for k, t in zip(common.HELD, seconds, strict=True) if not np.isfinite(t)]
    if untimed:
        raise SystemExit(f"the pass's rows under {metric} hold no default time on the held-out "
                         f"blocks {untimed}: a screen's cost model reads it on every block not run")
    holds = {k: len(passed.free_cleared[k]) > 0 for k in common.HELD}     # free first's
    rows = {v: _rows_by_block(screen_path(v, "x", metric=metric).parent) for v in SCREENS}
    a = rows["default"]
    out = {}
    for v, b in rows.items():
        if v == "default":
            continue
        plan = screen_plan(v)
        eligible = tuple(k for k in common.HELD if relax.built_for_holds(plan) or not holds[k])
        run = eligible[:len(b)]
        if set(b.index) != set(run):
            raise SystemExit(f"{v}: rows off its order: its {len(b)} blocks are not its order's "
                             f"first {len(b)} (off them: {sorted(set(b.index) - set(run))})")
        bare = sorted(set(run) - set(a.index))
        if bare:
            raise SystemExit(f"{v}: rows without the default's beside them, on {bare}")
        on, rest = list(run), list(eligible[len(run):])
        out[v] = ScreenData(eligible=eligible, run=run, arrived=screening.arrived_of(
            gain=(b.simp_perm[on] - a.simp_perm[on]).to_numpy(dtype=float),
            seconds=b.t[on].to_numpy(dtype=float),
            base_seconds=np.concatenate([a.t[on].to_numpy(dtype=float),
                                         passed.t[rest].to_numpy(dtype=float)])))
    return out


class Stored(NamedTuple):
    """A screen's state with the blocks it was decided on (its blocks run then, in its order): its
    decisions are the ledger the next wave continues from."""
    blocks: tuple[str, ...]
    state: Running | Stopped


DECISIONS = ROWS / "screen_decisions.json"
SUMMARY_SCREEN = ROWS / "summary_screen.csv"     # the screens' table (screen_table's)


def _decisions_json() -> Any:
    """DECISIONS parsed, None where there is none (no screen decided yet)."""
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else None


def read_decisions(screen: Screen, *, metric: str) -> dict[str, Stored]:
    """The screens' stored ledgers by variant (DECISIONS, the one JSON boundary, every key read by
    name): no file is no ledger yet. Refused (SystemExit) unless they were made under `metric` (on
    another metric's rows they decided on other numbers) and under `screen`, each record revives
    (a record epistemetry cannot read back, one of another pin, is refused saying that removing
    DECISIONS starts afresh), and each ledger is a variant's of SCREENS, a Running or Stopped state
    with its blocks."""
    j = _decisions_json()
    if j is None:
        return {}
    afresh = f"removing {DECISIONS.name} starts the screens afresh under the injected preset"

    def revived(record: Any, *, what: str) -> object:
        try:
            return revive(record)
        except (TypeError, ValueError) as e:     # what revive raises on a record it cannot read
            raise SystemExit(f"{DECISIONS.name}: {what} is no record epistemetry reads back "
                             f"({e}); {afresh}") from e

    if not (isinstance(j, dict) and set(j) == {"metric", "screen", "screens"}):
        raise SystemExit(f"{DECISIONS.name}: not an object of metric, screen and screens")
    if j["metric"] != metric:
        raise SystemExit(f"{DECISIONS.name}: ledgers under the metric {j['metric']}, not the "
                         f"screens' {metric}; {afresh}")
    if revived(j["screen"], what="the screen") != screen:
        raise SystemExit(f"{DECISIONS.name}: ledgers under the screen "
                         f"{json.dumps(j['screen'], sort_keys=True)}, not the injected preset's "
                         f"{json.dumps(screen.record(), sort_keys=True)}; {afresh}")
    out = {}
    for v, e in j["screens"].items():
        if v not in SCREENS or v == "default" or not (isinstance(e, dict)
                                                       and set(e) == {"blocks", "state"}):
            raise SystemExit(f"{DECISIONS.name}: {v!r} is not a variant's ledger of blocks and "
                             f"state (the variants: {[k for k in SCREENS if k != 'default']})")
        blocks, state = e["blocks"], revived(e["state"], what=f"{v}'s state")
        if not (isinstance(blocks, list) and all(isinstance(k, str) for k in blocks)
                and isinstance(state, Running | Stopped)):
            raise SystemExit(f"{DECISIONS.name}: {v}'s ledger is not a list of block ids and a "
                             f"Running or Stopped state")
        out[v] = Stored(blocks=tuple(blocks), state=state)
    return out


class ScreenTable(NamedTuple):
    """The screens decided: summary_screen.csv's rows (one a variant), each screen's state with the
    blocks it was decided on, and each running screen's next blocks."""
    table: pd.DataFrame
    states: dict[str, Stored]
    upcoming: dict[str, list[str]]


def screen_table(screen: Screen, *, data: Mapping[str, ScreenData],
                 ledgers: Mapping[str, Stored], wave: int, metric: str) -> ScreenTable:
    """Each variant of `data` screened by `screen` (epistemetry's advance), continuing from its
    ledger in `ledgers` where it has one (refused unless that ledger's blocks are a prefix of the
    variant's blocks run), with every axis the dominance rule reads, the lenses' rows under
    `metric`; a running screen's next blocks, its step's or `wave` if more. Computes, writes
    nothing."""
    unknown = sorted(set(ledgers) - set(data))
    if unknown:
        raise SystemExit(f"ledgers of {unknown}, which no screen's rows hold")
    for v, stored in ledgers.items():     # every ledger checked before any decision
        if data[v].run[:len(stored.blocks)] != stored.blocks:
            raise SystemExit(f"{v}: its ledger was decided on blocks its rows do not start with "
                             f"(stored {list(stored.blocks)}, run {list(data[v].run)}); removing "
                             f"{DECISIONS.name} starts the screens afresh")
    lens = _read(ROWS / "screen_lenses" / metric)
    la = lens[lens.variant == "default"].set_index("block")
    identity = json.dumps(screen.record(), sort_keys=True)
    out, states, upcoming = [], {}, {}
    for v, d in data.items():
        s = advance(screen, ledger=ledgers[v].state.decisions if v in ledgers else (),
                    arrived=d.arrived)
        states[v] = Stored(blocks=d.run, state=s)
        stop: str
        if isinstance(s, Stopped):
            n = s.arm_units[0]
            stop = screening.why_of(s, arm=0, units=len(d.eligible))
            call = screening.CALLS[s.calls[0][0]]
        else:
            n, stop, call = len(d.run), "", ""
            want = s.want[0]
            upcoming[v] = list(d.eligible[want[0]:want[0] + max(wave, len(want))])
        arm = d.arrived.arms[0]
        base = d.arrived.covariates[screening.BASE_SECONDS][:n]
        ratio = np.log(arm.costs[:n] / base)
        lo_t, hi_t = _bb_interval(ratio, np.random.default_rng(2))
        row: dict[str, object] = dict(
            variant=v, plan=screen_plan(v).name, eligible=len(d.eligible), run=len(d.run), used=n,
            stop=stop, call=call,
            # the rate an ask turns on: Lens A per extra GPU-second, on the blocks used
            break_even=float(arm.outcomes[:n, 0].mean() / arm.outcomes[:n, 1].mean()),
            time_ratio=float(arm.costs[:n].sum() / base.sum()),
            time_geo=float(np.exp(ratio.mean())), time_lo=float(np.exp(lo_t)),
            time_hi=float(np.exp(hi_t)), slower=int((ratio > 0).sum()))
        rng = np.random.default_rng(1)
        lb = lens[lens.variant == v].set_index("block")
        both = [k for k in d.run[:n] if k in la.index and k in lb.index]
        row["scored"] = len(both)
        for x in LENSES:
            diff = (lb[x][both] - la[x][both]).to_numpy(dtype=float)
            lo, hi = _bb_interval(diff, rng) if len(diff) else (np.nan, np.nan)
            row.update({f"{x}_mean": float(diff.mean()) if len(diff) else np.nan, f"{x}_lo": lo,
                        f"{x}_hi": hi, f"{x}_ahead": int((diff > 1e-9).sum()),
                        f"{x}_behind": int((diff < -1e-9).sum())})
        rb = [b_rank(np.array([la.enclosed[k], lb.enclosed[k]]),
                     np.array([la.L_P[k], lb.L_P[k]])) for k in both]
        row.update(B_ahead=sum(int(r[1] < r[0]) for r in rb),
                   B_behind=sum(int(r[1] > r[0]) for r in rb), screen=identity)
        out.append(row)
    return ScreenTable(table=pd.DataFrame(out), states=states, upcoming=upcoming)


def screen_report(screen: Screen, *, wave: int, metric: str) -> None:
    """Each screen decided by `screen` from its rows under `metric`, each continuing from its
    stored ledger (`screen_table`), and its axes; each running screen's next blocks.
    SUMMARY_SCREEN, and DECISIONS: the metric, the screen's record, and each screen's state with
    the blocks it was decided on."""
    pd.set_option("display.width", 220)
    st = screen_table(screen, data=screen_data(metric=metric),
                      ledgers=read_decisions(screen, metric=metric), wave=wave, metric=metric)
    # The ledgers' JSON built before either file is written: a record or a number JSON cannot hold
    # refuses here, leaving the table and the ledgers as they were, never one written without the
    # other.
    decisions = json.dumps(
        {"metric": metric, "screen": screen.record(),
         "screens": {v: {"blocks": list(s.blocks), "state": s.state.record()}
                     for v, s in st.states.items()}}, allow_nan=False, indent=1) + "\n"
    st.table.to_csv(SUMMARY_SCREEN, index=False)
    DECISIONS.write_text(decisions)
    print(st.table.drop(columns="screen").round(5).T.to_string())
    for v, ks in st.upcoming.items():
        print(f"{v} runs on: next {','.join(ks)}")


# ------------------------------------------------------------------ the road lineup's reuse

STORED_ROADS = HERE / "road_rows"           # item 4's networks (HoldNothing) and their rows
REUSED = ("clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
          "resistance_lp", "euclidean_grid")   # every method but the arterial (its cap changed)


def walled_under_runtime(bid: str) -> int:
    """The homes RUNTIME walls in on `bid` at baseline (the survey's `runtime` row)."""
    return int(pd.read_parquet(baseline_path(bid, "runtime")).walled.iloc[0])


def reuse_networks(scratch: Path) -> None:
    """The lineup's networks (road_rescore.lineup_arms) on the blocks of item 4's run where RUNTIME
    walls no home in: its free-first stage holds nothing there, and with nothing held a method
    runs as with HoldNothing (the hold-nothing guard; reblock's road methods unchanged since item
    4's run, 2026-10-10/11, but for the price and free-first plumbing, float for float under
    Opaque and HoldNothing), so each network of REUSED is item 4's, copied to the lineup's
    scratch with its canonical order, its prefixes' rows and its proposal row (`status` saying
    so). The arterial is proposed anew (its cap)."""
    import road_rescore as rr
    rows = rr.lineup_rows()
    done = 0
    for d in sorted((STORED_ROADS / "scratch" / "networks").iterdir()):
        bid = d.name
        if walled_under_runtime(bid):
            continue
        for name in REUSED:
            src = rr.net_path(STORED_ROADS / "scratch", bid, name, False)
            row = rr.proposal_row(STORED_ROADS, bid, name, False)
            out = rr.net_path(scratch, bid, name, False)
            if not (src.exists() and row.exists()):
                raise SystemExit(f"{bid} {name}: item 4's network or its row is missing")
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            rr.write_parquet(gpd.read_parquet(src), out)
            # its canonical order and prefixes too: a function of the network and the block
            rr.write_parquet(gpd.read_parquet(rr.ordered_path(STORED_ROADS / "scratch", bid, name)),
                             rr.ordered_path(scratch, bid, name))
            pre = rr.prefix_dir(rows) / f"{bid}__{name}.parquet"
            pre.parent.mkdir(parents=True, exist_ok=True)
            stored = rr.prefix_dir(STORED_ROADS) / f"{bid}__{name}.parquet"
            rr.write_parquet(pd.read_parquet(stored), pre)
            r = pd.read_parquet(row).assign(
                status="reused: item 4's HoldNothing network (RUNTIME walls no home in)")
            dest = rr.proposal_row(rows, bid, name, False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            rr.write_parquet(r, dest)
            done += 1
    print(f"{done} networks reused into {scratch}", flush=True)


# ------------------------------------------------------------------ report

def _read(d: Path) -> pd.DataFrame:
    fs = sorted(d.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def report() -> None:
    """ROWS/summary_features.csv (per block what meets it) and summary_switch.csv (per block each
    part's baseline and its move from the part before it), with the medians printed."""
    pd.set_option("display.width", 200)
    feat = _read(ROWS / "features")
    feat.sort_values("block").to_csv(ROWS / "summary_features.csv", index=False)
    feat["drive"] = feat.drive_edge_m + feat.drive_inside_m
    by = feat.groupby("size").agg(
        blocks=("block", "size"), barrier=("barrier_m2", lambda x: int((x > 0).sum())),
        rail=("rail_lines", lambda x: int((x > 0).sum())),
        water=("water_lines", lambda x: int((x > 0).sum())),
        water_areas=("water_areas", lambda x: int((x > 0).sum())),
        drive_any=("drive", lambda x: int((x > 0).sum())),
        drive_inside=("drive_inside_m", lambda x: int((x > 0).sum())),
        cut_median=("barrier_share", lambda x: float(x[x > 0].median()) if (x > 0).any() else 0.0),
        cut_max=("barrier_share", "max"))
    print("WHAT MEETS THE BLOCKS (blocks per size class)\n" + by.to_string())
    rows = []
    for part in PARTS:
        d = _read(ROWS / "baselines" / part_metric(part, "cpu").name)
        if len(d):
            rows.append(d)
    if not rows:
        return
    b = pd.concat(rows, ignore_index=True)
    w = b.pivot_table(index="block", columns="part", values=["P0", "Pq0", "J2_0", "walled"])
    out = pd.DataFrame(index=w.index)
    for k in ("P0", "Pq0", "J2_0"):
        for prev, part in itertools.pairwise(PARTS):
            if (k, part) in w and (k, prev) in w:
                out[f"{k}_{part}"] = w[(k, part)] / w[(k, prev)] - 1.0
        if (k, "runtime") in w and (k, "block") in w:
            out[f"{k}_all"] = w[(k, "runtime")] / w[(k, "block")] - 1.0
    for part in PARTS:
        if ("walled", part) in w:
            out[f"walled_{part}"] = w[("walled", part)]
    out = out.reset_index().merge(feat[["block", "size", "barrier_share", "drive_inside_m"]],
                                  on="block", how="left")
    out.to_csv(ROWS / "summary_switch.csv", index=False)
    cols = [str(c) for c in out.columns if str(c).endswith(("_leak", "_bar", "_runtime", "_all"))
            and not str(c).startswith("walled")]
    for name, sub in (("THE 13 TUNING BLOCKS", out[out.block.isin(common.TUNING)]),
                      ("THE STORE", out)):
        print(f"\n{name} ({len(sub)} blocks): each part's move of the baseline, relative to the "
              "part before it (median [min, max])")
        for c in cols:
            x = sub[c].dropna()
            if len(x):
                print(f"  {c:14s} {x.median():+.4f} [{x.min():+.4f}, {x.max():+.4f}] on {len(x)}")
        print("  by size: " + "; ".join(
            f"{sz!s} P0_all {g.P0_all.median():+.4f}" for sz, g in sub.groupby("size")
            if "P0_all" in g))


# ------------------------------------------------------------------ the lineup's report

LENSES = ("L_P", "L_Pq", "L_J2")         # P the headline, P_{1/2} the equity view, J_2 the old
PRESETS = ("default", "hard", "cheap", "cheap005")
# The arterial is research/arterial-speed's (its stop at the ramped destroyed-area D 0.20, the
# quality-first sampler), re-proposed on the lineup's blocks by that plan's Task 7; the old one's
# rows (300 roads, no stop) were deleted from the root when the measure moved it.
LINEUP_ROADS = ("clearance", "clearance_looped", "cycle_native",
                "cycle_native_betweenness_contrast", "resistance_lp", "euclidean_grid",
                "greedy_arterial_access_displacement")
OLD = STORED_ROADS / "cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a"      # item 4 on the complex
OLD_ROWS = OLD / "cpu" / "area_D0.02-0.05-0.1-0.15-0.2"


def b_rank(enc: Any, val: Any) -> Any:
    """Ranks in B's order (road_rescore._b_rank): the least population walled in first, then the
    larger lens; exact ties share a rank."""
    e, v = np.asarray(enc, dtype=float), np.asarray(val, dtype=float)
    ahead = (e[None, :] < e[:, None]) | ((e[None, :] == e[:, None]) & (v[None, :] > v[:, None]))
    return 1.0 + ahead.sum(axis=1)


def _roads_at(t: pd.DataFrame, budget: float) -> pd.DataFrame:
    """The networks' area prefixes at `budget` (the prefix reaching it, or the whole network)."""
    at = cast(pd.DataFrame, t[np.isclose(t.budget, budget)].copy())
    at["reach"] = at.D_area >= budget - 1e-9
    return at


def lineup_report() -> None:
    """The lineup under RUNTIME (the brief's item 14): ROWS/summary_lineup_*.csv, printed too."""
    from scipy.stats import kendalltau

    import road_rescore as rr
    pd.set_option("display.width", 220)
    feat = pd.read_csv(ROWS / "summary_features.csv").set_index("block")
    lens = _read(ROWS / "lenses" / metrics.RUNTIME)
    lens["size"] = lens.block.map(feat["size"])
    pre = lens[lens.preset.isin(PRESETS)]
    out = ROWS

    # the presets on every block: Lens A and B per lens, the order, the time
    rows = []
    for key, g in pre.groupby(["size", "preset"]):
        sz, k = cast(tuple[str, str], key)
        rows.append(dict(size=sz, preset=k, blocks=len(g), **{
            f"{x}_{f}": float(getattr(g[x], f)()) for x in LENSES for f in ("mean", "median")},
            enclosed_blocks=int((g.enclosed > 0).sum())))
    t1 = pd.DataFrame(rows)
    t1.to_csv(out / "summary_lineup_presets.csv", index=False)
    print("THE PRESETS (Lens A at area D 0.05 under RUNTIME, per lens)\n" + t1.round(4).to_string())
    pairs = []
    wide = {x: pre.pivot_table(index="block", columns="preset", values=x) for x in (*LENSES,
                                                                                  "enclosed")}
    for a, b in (("hard", "default"), ("cheap", "default"), ("cheap005", "cheap")):
        for x in LENSES:
            if a not in wide[x] or b not in wide[x]:
                continue
            d = (wide[x][a] - wide[x][b]).dropna()
            sizes = feat["size"].reindex(d.index)
            for name, dd in (("all", d), *((s_, d[sizes == s_])
                                          for s_ in sorted(sizes.unique()))):
                if len(dd):
                    pairs.append(dict(pair=f"{a}-{b}", lens=x, blocks=name, n=len(dd),
                                      mean=float(dd.mean()), median=float(dd.median()),
                                      ahead=int((dd > 1e-12).sum()),
                                      behind=int((dd < -1e-12).sum())))
    t2 = pd.DataFrame(pairs)
    t2.to_csv(out / "summary_lineup_pairs.csv", index=False)
    print("\nTHE PRESETS PAIRED (a - b per block)\n" + t2.round(4).to_string())
    order = []
    for bid, g in pre[pre.preset.isin(("default", "hard", "cheap"))].groupby("block"):
        for x in LENSES:
            rk = b_rank(g.enclosed, g[x])
            for k, r in zip(g.preset, rk, strict=True):
                order.append(dict(block=bid, lens=x, preset=k, rank=float(r)))
    to = pd.DataFrame(order)
    first = to[to["rank"] == 1].groupby(["lens", "preset"]).size().unstack(fill_value=0)
    print("\nFIRST IN B'S ORDER among default, hard, cheap (blocks; ties count for each)\n"
          + first.to_string())
    one = _read(ROWS / "onecard" / metrics.RUNTIME / "gpu")
    cpu = _read(ROWS / "onecard" / metrics.RUNTIME / "cpu")
    tim = pd.concat([one.assign(device="gpu"), cpu.assign(device="cpu")], ignore_index=True)
    tim["size"] = tim.block.map(feat["size"])
    tim.to_csv(out / "summary_lineup_onecard.csv", index=False)
    print("\nTIME PER BLOCK (s; the presets one after another with their setup, one device)\n"
          + tim.groupby(["size", "device"])[[str(c) for c in tim.columns
                                              if str(c).startswith("wall")]]
          .median().round(0).to_string())

    # the road methods under free first
    root = rr.lineup_rows()
    cx = _read(root / metrics.RUNTIME / "cpu" / "area_D0.02-0.05-0.1-0.15-0.2")
    if len(cx):
        assert set(cx.method) <= set(LINEUP_ROADS), set(cx.method)
    if len(cx):
        at = _roads_at(cx, 0.05)
        at["size"] = at.block.map(feat["size"])
        med = at[at.reach].groupby("method")[list(LENSES)].median()
        print("\nTHE ROAD METHODS at area D 0.05 (medians over the blocks they reach it on)\n"
              + med.round(4).to_string())
        price = []
        dflt = pre[pre.preset == "default"].set_index("block")
        for bid, g in at[at.reach].groupby("block"):
            if bid not in dflt.index:
                continue
            d = dflt.loc[bid]
            for x in LENSES:
                rk = b_rank(g.enclosed, g[x])
                best = g.iloc[int(np.argmin(rk))]
                di, dii = float(d.enclosed) - float(best.enclosed), float(d[x]) - float(best[x])
                price.append(dict(block=bid, lens=x, best=best.method, best_val=float(best[x]),
                                  best_enc=float(best.enclosed), simp_val=float(d[x]),
                                  simp_enc=float(d.enclosed), di=di, dii=dii,
                                  ahead=bool(di < 0 or (di == 0 and dii > 0))))
        tp = pd.DataFrame(price, columns=["block", "lens", "best", "best_val", "best_enc",
                                          "simp_val", "simp_enc", "di", "dii", "ahead"])
        tp.to_csv(out / "summary_lineup_price.csv", index=False)
        for x in LENSES if len(tp) else ():
            q = tp[tp.lens == x]
            tie = q[q.di == 0]
            print(f"  {x}: SIMP's default ahead of the best road on {int(q.ahead.sum())} of "
                  f"{len(q)}; where (i) ties ({len(tie)}) by a median {tie.dii.median():+.4f}; "
                  f"the best road: " + ", ".join(f"{k} {v}" for k, v in
                                                 q.best.value_counts().items()))
        tp["size"] = tp.block.map(feat["size"])
        spread: list[dict[str, object]] = []
        for key, q in tp[tp.di == 0].groupby(["lens", "size"]):
            x, sz = cast(tuple[str, str], key)
            share = q.best_val / q.simp_val
            spread.append(dict(lens=x, size=sz, ties=len(q), median=float(q.dii.median()),
                               p10=float(q.dii.quantile(0.1)), p90=float(q.dii.quantile(0.9)),
                               min=float(q.dii.min()), max=float(q.dii.max()),
                               share=float(share.median()), share_p10=float(share.quantile(0.1)),
                               share_p90=float(share.quantile(0.9))))
        if spread:
            print("\nWHERE (i) TIES: SIMP's default minus the best road, and the best road's "
                  "share of SIMP's lens (median, p10, p90)\n"
                  + pd.DataFrame(spread).round(4).to_string())
        # each method on the large and the oversized blocks: its rows, the blocks it reaches
        # D 0.05 on, its lenses there and against SIMP's default, its whole network's D, and the
        # wall of the networks proposed in this pass (the reused blocks' are item 4's)
        full_d = cx[cx.full.astype(bool)].pivot_table(index="block", columns="method",
                                                      values="D_area")
        prop = _read(root / "proposals")
        wall = prop[prop.status == "ok"].pivot_table(index="block", columns="method",
                                                     values="wall_s")
        full_d, wall = full_d.reindex(columns=LINEUP_ROADS), wall.reindex(columns=LINEUP_ROADS)
        roads: list[dict[str, object]] = []
        for mth, gm in at.groupby("method"):
            for sz in ("all", "large", "oversized"):
                gs = cast(pd.DataFrame, gm if sz == "all" else gm[gm["size"] == sz])
                r = gs[gs.reach].set_index("block")
                row: dict[str, object] = dict(
                    method=mth, size=sz, rows=len(gs), reach=len(r),
                    network_D=float(full_d[mth].reindex(gs.block).median()),
                    wall_h=float(wall[mth].reindex(gs.block).median() / 3600),
                    wall_h_max=float(wall[mth].reindex(gs.block).max() / 3600))
                for x in LENSES:
                    d = (r[x] - dflt[x].reindex(r.index)).dropna()
                    row[x] = float(r[x].median())
                    row[f"{x}_simp"] = float(d.median())
                    row[f"{x}_ahead"] = int((d > 0).sum())
                roads.append(row)
        tr = pd.DataFrame(roads)
        tr.to_csv(out / "summary_lineup_roads.csv", index=False)
        print("\nEACH METHOD at area D 0.05 by size: rows, blocks reaching D 0.05, the medians "
              "there, minus SIMP's default (_simp) and the blocks ahead of it (_ahead); the whole "
              "network's D and the proposals' wall in this pass (h)\n" + tr.round(4).to_string())
        big = feat.index[feat["size"].isin(("large", "oversized"))]
        have = set(zip(cx.block, cx.method, strict=True))
        miss = [(b, m) for b in big for m in LINEUP_ROADS if (b, m) not in have]
        print(f"\nNOT SCORED: {len(miss)} of {len(big) * len(LINEUP_ROADS)} (block, method)\n"
              + "\n".join(f"  {b} {m}" for b, m in miss))
        held = []
        for f in sorted((root / "scratch" / "networks").glob("*/*.parquet")):
            if f.name.endswith(".ordered.parquet"):
                continue
            net = gpd.read_parquet(f)
            held.append(dict(block=f.parent.name, method=f.stem,
                             held=int(net["held"].sum()) if "held" in net.columns else 0))
        th = pd.DataFrame(held)
        th.to_csv(out / "summary_lineup_held.csv", index=False)
        print("\nFREE FIRST: blocks where a method holds roads first\n"
              + th[th.held > 0].groupby("method").block.nunique().to_string())
        # the order's change: tau against item 4's order under the per-block metric
        old = pd.concat([pd.read_parquet(f) for f in sorted(OLD_ROWS.glob("*__*.parquet"))
                         if not f.name.endswith(("__plain.parquet", "__homes.parquet"))])
        old_at = _roads_at(old, 0.05)
        tau = []
        for bid, g in at[at.reach].groupby("block"):
            o = old_at[(old_at.block == bid) & old_at.reach].set_index("method")
            n = g.set_index("method")
            common_m = sorted(set(o.index) & set(n.index))
            if len(common_m) < 3:
                continue
            for x in LENSES:
                r_old = b_rank(o.loc[common_m, "enclosed"], o.loc[common_m, x])
                r_new = b_rank(n.loc[common_m, "enclosed"], n.loc[common_m, x])
                tau.append(dict(block=bid, lens=x, methods=len(common_m),
                                tau=float(kendalltau(r_old, r_new).statistic)))
        tt = pd.DataFrame(tau)
        tt.to_csv(out / "summary_lineup_tau.csv", index=False)
        print("\nTHE ORDER OF THE ROAD METHODS against item 4's (the per-block metric), Kendall "
              "tau per block\n" + tt.groupby("lens").tau.describe().round(3).to_string())


def main(argv: list[str]) -> None:
    argv, again = clearings.replace_flag(argv)
    cmd = argv[0] if argv else ""
    if cmd == "features" and len(argv) == 3:
        features(ids_arg(argv[1]), int(argv[2]))
    elif cmd == "presets" and len(argv) == 4 and argv[2] in ("cpu", "gpu"):
        presets(common.ids_of(argv[1])[0], argv[2], argv[3].split(","), replace=again)
    elif cmd == "lenses" and len(argv) == 2:
        lenses(common.ids_of(argv[1])[0], replace=again)
    elif cmd == "reuse-networks" and len(argv) == 2:
        reuse_networks(Path(argv[1]))
    elif cmd == "report" and len(argv) == 1:
        report()
    elif cmd == "lineup-report" and len(argv) == 1:
        lineup_report()
    elif cmd == "screen" and len(argv) == 3:
        screen(common.ids_of(argv[1])[0], argv[2].split(","), replace=again,
               metric=SCREENS_METRIC)
    elif cmd == "screen-lenses" and len(argv) == 2:
        screen_lenses(common.ids_of(argv[1])[0], replace=again, metric=SCREENS_METRIC)
    elif cmd == "screen-report" and len(argv) == 3 and argv[2].isdigit():
        named = screen_presets()     # the preset resolved once, here: a name with no default
        if argv[1] not in named:
            raise SystemExit(f"unknown preset {argv[1]!r}: the presets are {', '.join(named)}")
        screen_report(named[argv[1]], wave=int(argv[2]), metric=SCREENS_METRIC)
    elif cmd == "baselines" and len(argv) == 3:
        parts = argv[2].split(",")
        if not set(parts) <= set(PARTS):
            raise SystemExit(f"parts: {PARTS}")
        baselines(common.ids_of(argv[1])[0], parts, replace=again)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
