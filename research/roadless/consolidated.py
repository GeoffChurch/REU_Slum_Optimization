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
    uv run python $S report                      # ROWS/summary_*.csv

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


def main(argv: list[str]) -> None:
    argv, again = clearings.replace_flag(argv)
    cmd = argv[0] if argv else ""
    if cmd == "features" and len(argv) == 3:
        features(ids_arg(argv[1]), int(argv[2]))
    elif cmd == "presets" and len(argv) == 4 and argv[2] in ("cpu", "gpu"):
        presets(common.ids_of(argv[1])[0], argv[2], argv[3].split(","), replace=again)
    elif cmd == "report" and len(argv) == 1:
        report()
    elif cmd == "baselines" and len(argv) == 3:
        parts = argv[2].split(",")
        if not set(parts) <= set(PARTS):
            raise SystemExit(f"parts: {PARTS}")
        baselines(common.ids_of(argv[1])[0], parts, replace=again)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
