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

`features`: per block (the store's 303: clearings.Store's blocks), its OSM features (attrs["osm"])
as the metric reads them: the barriers' cut (gap_exits.OsmBarriers.parts within the block: its
area, the railway's and water's, the crossings), and the highway lines meeting it by class, each
split into the length on its edge (within EDGE_M of the street ring) and inside (deeper: a way
the boost raises inside the block, not a street of its ring). Rows ROWS/features/<id>.parquet.

`presets`: the lineup's presets on a block under RUNTIME, one after another on one device
(items_of: SIMP's default `.p256w8x2`, the hard-block preset `.p256w8x2.GS0.01cate1`, the cheap
preset, and the cheap preset at S0.005cat), each free first at area D 0.05 optimizing P, its rows
in relax_rows/ and polish_rows/ under the metric's name, and the run's one-card timing
(ROWS/onecard/<metric>/<device>/<id>.parquet).

`baselines`: a block's baseline (P0, P_{1/2},0, J_2,0, the homes walled in, the unknowns, the
mesh's triangles and times) under each of PARTS, cumulative: `block` the per-block metric
(metrics.PER_BLOCK), `leak` with the central leak, `bar` with the barriers too, `runtime` with
the boost too (RUNTIME). Rows ROWS/baselines/<id>__<part>.parquet, each under the metric's name.
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_exits as gx  # noqa: E402
import metrics  # noqa: E402

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
        drive = cls.isin(metrics.DRIVE_CLASSES).to_numpy()
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

TUNING = ("ZAF.9.3.1_1_1558", "ZAF.9.3.1_1_18895", "ZAF.9.3.1_1_20269", "ZAF.9.3.1_1_20543",
          "ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_23597", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_33717",
          "ZAF.9.3.1_1_38616", "ZAF.9.3.1_1_46841", "ZAF.9.3.1_1_5810", "ZAF.9.5.4_1_9712",
          "ZAF.9.5.4_1_9717")      # seqscreen.TUNING (not imported: its loaders read the rows)


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
        for prev, part in zip(PARTS, PARTS[1:]):
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
    for name, sub in (("THE 13 TUNING BLOCKS", out[out.block.isin(TUNING)]),
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
        tp = pd.DataFrame(price)
        tp.to_csv(out / "summary_lineup_price.csv", index=False)
        for x in LENSES:
            q = tp[tp.lens == x]
            tie = q[q.di == 0]
            print(f"  {x}: SIMP's default ahead of the best road on {int(q.ahead.sum())} of "
                  f"{len(q)}; where (i) ties ({len(tie)}) by a median {tie.dii.median():+.4f}; "
                  f"the best road: " + ", ".join(f"{k} {v}" for k, v in
                                                 q.best.value_counts().items()))
        vs = []
        for mth, gm in at[at.reach].groupby("method"):
            gi = gm.set_index("block")
            for x in LENSES:
                d = (gi[x] - dflt[x].reindex(gi.index)).dropna()
                vs.append(dict(method=mth, lens=x, n=len(d), median=float(d.median()),
                               ahead_of_simp=int((d > 0).sum())))
        print("\nEACH METHOD MINUS SIMP'S DEFAULT at D 0.05\n"
              + pd.DataFrame(vs).pivot_table(index="method", columns="lens",
                                             values="median").round(4).to_string())
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
    elif cmd == "baselines" and len(argv) == 3:
        parts = argv[2].split(",")
        if not set(parts) <= set(PARTS):
            raise SystemExit(f"parts: {PARTS}")
        baselines(common.ids_of(argv[1])[0], parts, replace=again)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
