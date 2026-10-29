"""A method's build-out on any block, on one budget grid (owner 2026-10-28 -- 29: the Demo page's
second block, Loop roads against topology under one slider), as the 5810 build-out page builds
its own: the block's plain baseline (compute.baseline), the method's frames at the grid's budgets
(per budget the shortest prefix of its build order whose D reaches it, as the lineup cuts every
method; its corridors carved into one mesh and solved exactly under RUNTIME: compute.frames_of's
rasters, lenses, each home's escape time, the sheet; and the whole network), and the per-budget
table. The block is the command line's, resolved once into a Site (block_site.py); nothing below
names one.

A method (Method, a Strategy, from the command line):
  grown <plan>   a growth or ears run (growth_rows/<block>/<plan>/), its steps in build order
                 (networks.Grown: each prefix's triangles unioned and smoothed); its seconds the
                 run's own to each step (steps.parquet's t);
  topology       the kept topology run (topology_run.py: atlas/blocks/<block>/topology_roads
                 .parquet) in the lineup's canonical order (common.canonical_order: drainage
                 first, each road after the roads it needs to reach the street); its seconds the
                 whole run's, at the whole network only;
  topology built the same network in its own build order (topology_order.py's watched re-run:
                 its proposal keeps none), its seconds when each prefix's last segment was
                 added.

The table (Site.kept/<key>_grid.json, committed): per grid budget the frame (its stop, D, road
pieces and km, L_P, the conductance added C, the seconds), or what stands there instead: `repeat`
(the frame is a lower budget's: the method's next prefix jumps past this budget too) or `stopped`
(the method's whole network falls short of the budget), and `over`, how far the frame's D passes
the budget (a method with no prefix near a budget: Loop roads' first ear is D 0.057); and the
whole network's own row. Beside
it the escape-time scale (the plain baseline's 99th percentile).

    B=research/roadless/atlas/buildout.py              # from the repo root, PYTHONPATH=.
    uv run python $B baseline <block>
    uv run python $B frames <block> grown <plan> | topology [built]
    uv run python $B table <block> grown <plan> | topology [built]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "roads5810"))
import networks as nw  # noqa: E402  (the build-out page's sources; puts block_site on the path)
from block_site import Site  # noqa: E402

cp, render, gr = nw.cp, nw.render, nw.gr
PERCENTILE = 99          # the escape-time view's scale: the plain baseline's homes' percentile


class Method(Protocol):
    """A method's build-out on a block (module docstring)."""

    @property
    def key(self) -> str: ...

    def network(self, b: Any) -> Any:
        """Its compute.Network on block b (prefixes in build order)."""
        ...

    def seconds(self, stop: int, *, size: int) -> float | None:
        """The run's seconds to prefix `stop` (None where the run has no time for it)."""
        ...

    @property
    def machine(self) -> str: ...


class GrownRun:
    """A growth or ears run's steps (networks.Grown)."""

    def __init__(self, site: Site, plan: str, *, grid: tuple[float, ...]) -> None:
        self.dir = gr.OUT / site.bid / plan
        if not (self.dir / "steps.parquet").exists():
            raise SystemExit(f"{self.dir}: no run")
        self.key = plan
        self.grid = grid
        self.steps = pd.read_parquet(self.dir / "steps.parquet").set_index("step").t
        self.machine = MACHINE

    def network(self, b: Any) -> Any:
        return nw.Grown(b, self.dir, targets=self.grid)

    def seconds(self, stop: int, *, size: int) -> float | None:
        return float(self.steps[stop])


class Topology:
    """The kept topology run in the lineup's canonical order (module docstring)."""
    key = "topology.a10.s0.w7.canonical"

    def __init__(self, site: Site) -> None:
        self.roads = gpd.read_parquet(site.kept / "topology_roads.parquet")
        self.info = json.loads((site.kept / "topology.json").read_text())
        self.machine = f"one process on {self.info['host']} ({self.info['cpu']})"

    def network(self, b: Any) -> Any:
        return cp.Roads(b, cp.common.canonical_order(b, self.roads))

    def seconds(self, stop: int, *, size: int) -> float | None:
        return float(self.info["wall_s"]) if stop == size else None


class TopologyBuilt:
    """The kept topology run in its own build order (topology_order.py's watched re-run, the same
    network: topology_built_roads.parquet, each segment in the order the greedy added it, with
    the second it was added)."""
    key = "topology.a10.s0.w7.built"

    def __init__(self, site: Site) -> None:
        self.roads = gpd.read_parquet(site.kept / "topology_built_roads.parquet")
        if not (np.diff(self.roads["index"].to_numpy()) == 1).all():
            raise SystemExit("topology_built_roads.parquet is not in build order")
        self.info = json.loads((site.kept / "topology_built.json").read_text())
        self.machine = f"one process on {self.info['host']} ({self.info['cpu']})"

    def network(self, b: Any) -> Any:
        return cp.Roads(b, self.roads)

    def seconds(self, stop: int, *, size: int) -> float | None:
        """When the prefix's last segment was added, from the run's start."""
        return float(self.roads.t.iloc[stop - 1])


def method_of(site: Site, argv: list[str], *, grid: tuple[float, ...]) -> Method:
    match argv:
        case ["grown", plan]:
            return GrownRun(site, plan, grid=grid)
        case ["topology"]:
            return Topology(site)
        case ["topology", "built"]:
            return TopologyBuilt(site)
    raise SystemExit(__doc__)


def baseline(site: Site) -> None:
    [b] = cp.common.build_blocks([site.bid])
    cp.baseline(b, data=site.data)


def frames(site: Site, method: Method, *, grid: tuple[float, ...]) -> None:
    [b] = cp.common.build_blocks([site.bid])
    t0 = time.time()
    net = method.network(b)
    site.networks.mkdir(parents=True, exist_ok=True)
    cp.frames_of(method.key, RUN, b, net, t0, frames=cp.Pass(targets=grid, full=True),
                 data=site.networks, meshes=site.networks / ".meshes")


def table(site: Site, method: Method, *, grid: tuple[float, ...]) -> None:
    """The per-budget table (module docstring) into Site.kept/<key>_grid.json."""
    info = nw.info_of(method.key, RUN, data=site.networks)
    C = nw.sheet.added(method.key, data=site.networks, runs=(RUN,))
    names = render.frame_names(info)
    by_m = {r["m"]: r for r in info["frames"]}
    size = max(by_m)
    want = {w["target"]: w["m"] for w in info["want"] if w["target"] is not None}

    def frame(m: int) -> dict[str, Any]:
        r = by_m[m]
        sec = method.seconds(m, size=size)
        return dict(stop=m, frame=names[m], D=round(r["D_area"], 4), roads=r["n_roads"],
                    km=round(r["road_m"] / 1000, 3), L_P=round(r["L_P"], 4),
                    C=round(C[names[m]]), seconds=None if sec is None else round(sec, 1))

    rows: list[dict[str, Any]] = []
    first: dict[int, float] = {}
    for t in grid:
        if t not in want:
            rows.append(dict(budget=t, status="stopped", D_whole=round(info["D_full"], 4)))
            continue
        m = want[t]
        status = "frame" if m not in first else "repeat"
        f = frame(m)
        rows.append(dict(budget=t, status=status, over=round(f["D"] - t, 4),
                         **({} if m not in first else dict(repeat_of=first[m])), **f))
        first.setdefault(m, t)
    u = np.load(site.data / "baseline_times.npz")["u"]
    base = json.loads((site.data / "baseline.json").read_text())
    doc = dict(block=site.bid, method=method.key, run=RUN, metric=base["metric"],
               machine=method.machine, grid=list(grid), rule=info["rule"],
               frames_seconds=round(float(info["t_total"]), 1),
               escape_scale=float(np.percentile(u[np.isfinite(u)], PERCENTILE)),
               escape_percentile=PERCENTILE, n_homes=int(np.isfinite(u).sum()),
               n_buildings=len(u), rows=rows, whole=frame(size))
    site.kept.mkdir(parents=True, exist_ok=True)
    (site.kept / f"{method.key}_grid.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{method.key}: {'budget':>6} {'status':>7} {'stop':>4} {'D':>7} {'km':>6} {'L_P':>6} "
          f"{'C':>7} {'s':>7}")
    for r in rows + [dict(budget=float("nan"), status="whole", **doc["whole"])]:
        if r["status"] == "stopped":
            print(f"{r['budget']:6.2f} stopped (whole network at D {r['D_whole']:.4f})")
            continue
        sec = "-" if r["seconds"] is None else f"{r['seconds']:.1f}"
        print(f"{r['budget']:6.2f} {r['status']:>7} {r['stop']:4d} {r['D']:7.4f} {r['km']:6.3f} "
              f"{r['L_P']:6.3f} {r['C']:7d} {sec:>7}")


GRID = tuple(round(0.01 * i, 2) for i in range(1, 16)) + tuple(round(0.05 * i, 2)
                                                                for i in range(4, 19))
RUN = "grid"             # the frames' run name (networks' <key>_<run>)
MACHINE = "one process on a workstation CPU (Intel Xeon w7-3455)"   # where grown runs ran


def main(argv: list[str]) -> None:
    match argv:
        case ["baseline", bid]:
            baseline(Site.of(bid))
        case ["frames", bid, *spec]:
            site = Site.of(bid)
            frames(site, method_of(site, spec, grid=GRID), grid=GRID)
        case ["table", bid, *spec]:
            site = Site.of(bid)
            table(site, method_of(site, spec, grid=GRID), grid=GRID)
        case _:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
