"""The Loop Network's (cycle_native's) run time on the Demo's block, per road (owner 2026-11-04: the
Demo sets it beside Shock Reblock on 5810, both with a measured time). The lineup's proposal rows
time a whole run only (road_rescore.propose: wall_s), and the Demo's frames are canonical
prefixes of it (common.canonical_order: the held road, then street-first), not prefixes of its
build order; so this runs the lineup's arm (road_rescore.lineup_arms()["cycle_native"]: the
shipped preset, free_first the lineup's) once, fresh (no derive cache), in this one process,
single-threaded, and records when each road was committed: the method's PricedOverlap, which
takes each committed cycle's corridor once (the held roads' first), stamped at each `add`. The
roads it returns are checked, road for road, against the lineup's stored network, which the
Demo's frames are cut from.

    T=research/roadless/atlas/explore_proto/loop_time.py   # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        uv run python $T run        # ~25 min on 5810; writes TIMES (committed)

build.py's LineupTimed reads TIMES: a frame's time is when the run had built every road in it.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
sys.path.insert(0, str(HERE.parent / "roads5810"))
import common  # noqa: E402
import compute as cp  # noqa: E402
import geopandas as gpd  # noqa: E402
import road_rescore as rr  # noqa: E402
import shapely  # noqa: E402

import reblock.methods.cycle_native as cn  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
METHOD = "cycle_native"
TIMES = HERE / "loop_time_5810.json"      # committed: the measurement the Demo states
# The lineup the 5810 build-out page's Lineup family cuts its frames from (roads5810's compute.py
# LINEUP_NETWORKS: that run's rows root; road_rescore.lineup_rows() has since moved on with the
# arterial's identity, which leaves cycle_native's arm as it was).
STORED = cp.LINEUP_NETWORKS / BID / f"{METHOD}.parquet"


def cpu_model() -> str:
    """The CPU's model name, as /proc/cpuinfo gives it."""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("no model name in /proc/cpuinfo")


def stamped(commits: list[tuple[float, bytes]], roads: gpd.GeoDataFrame, *,
            hw: float) -> list[float]:
    """Per road of `roads` (the method's output, in its build order) the time of the commit that
    took it: the held roads all at the first commit, then each commit the next one or two roads (a
    cycle's out leg, and its way back where it has one), the pair taken where their corridor,
    buffered by `hw` as the method buffers them, is the commit's own."""
    geoms = list(roads.geometry)
    held = int(roads.held.sum())
    road_s: list[float] = []
    j = 0
    for t, wkb in commits:
        piece = shapely.from_wkb(wkb)
        if j == 0 and held > 0:
            n = held
        else:
            two = shapely.union_all([g.buffer(hw) for g in geoms[j:j + 2]])
            n = 2 if j + 2 <= len(geoms) and two.equals_exact(piece, 0.0) else 1
            if n == 1 and not shapely.union_all([geoms[j].buffer(hw)]).equals_exact(piece, 0.0):
                raise AssertionError(f"commit at {t:.1f}s: neither road {j} nor {j}, {j + 1}")
        road_s += [t] * n
        j += n
    if j != len(geoms):
        raise AssertionError(f"{len(commits)} commits cover {j} roads of {len(geoms)}")
    return road_s


def run() -> None:
    rr._check_single_threaded()
    method = rr.lineup_arms()[METHOD]
    [b] = common.build_blocks([BID])
    commits: list[tuple[float, bytes]] = []
    t0 = 0.0

    class Stamped(cn.PricedOverlap):     # type: ignore[misc,name-defined]
        """The method's own overlap, each `add` (one committed cycle's corridor) stamped."""

        def add(self, piece) -> None:  # type: ignore[no-untyped-def]
            super().add(piece)
            commits.append((time.perf_counter() - t0, shapely.to_wkb(piece)))

    cn.PricedOverlap = Stamped           # the module's name, read at each propose
    load0 = os.getloadavg()
    print(f"{time.strftime('%H:%M:%S')} {BID} n={len(b.buildings)} {METHOD}: load {load0}",
          flush=True)
    t0 = time.perf_counter()
    prop = method.propose(b)
    total = time.perf_counter() - t0
    load1 = os.getloadavg()
    roads = prop.roads
    raw = Path(os.environ.get("LOOP_TIME_RAW", "/dev/null"))
    if str(raw) != "/dev/null":                   # the raw stamps, should the checks below fail
        import pickle
        raw.write_bytes(pickle.dumps(dict(commits=commits, roads=roads, total=total)))
    stored = gpd.read_parquet(STORED)
    if len(stored) != len(roads) or not stored.geometry.reset_index(drop=True).geom_equals_exact(
            roads.geometry.reset_index(drop=True), tolerance=0.0).all():
        raise SystemExit(f"{METHOD}: {len(roads)} roads, not the lineup's stored {len(stored)}")
    road_s = stamped(commits, roads, hw=method.road_width_m / 2.0)
    out = dict(block=BID, method=METHOD, identity=repr(method.identity), seconds=total,
               road_s=road_s, n_roads=len(roads), host=socket.gethostname(),
               cpu=cpu_model(),
               threads={k: os.environ[k] for k in rr.SINGLE if k.endswith("THREADS")},
               load_before=load0, load_after=load1, nproc=os.cpu_count(),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    TIMES.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{time.strftime('%H:%M:%S')} {METHOD}: {len(roads)} roads in {total:.0f} s; load "
          f"{load1}; {TIMES}", flush=True)


def main(argv: list[str]) -> None:
    match argv:
        case ["run"]:
            run()
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
