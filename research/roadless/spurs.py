"""Partition plus spurs (owner-approved 2026-10-28): from a partition SIMP cut, the ΔP greedy the
5810 page's comparison numbers come from (the flow-roads study's dp_greedy: a pool, a pick, a cost
and an objective, the page's the lineup's networks, `lazy:5`, `D` and `P`) adds roads until D
reaches the cut's own D plus `extra`.

The cut is a fixed network built before the first step: one more network in the pool's carving
(its stored ground, partition rev 4, as its corridor; its lines, so a pool road is open when it
touches the street, the cut or a built road), committed first; every lens against the bare block.
On the GPU (cuDSS, one plan for the pool's mesh, each candidate a refactorization) the exact
candidates are solved one after another in the process (dp_greedy.Serial); on the CPU by
dp_greedy.WORKERS forked processes. Each step's row in <build>/steps.json, the roads added in
order in built.parquet (built[0] the cut), the build in
spurs_builds/<RUNTIME>/<greedy>.x<extra>/<block>/<cut's plan>/ (Greedy.name: the pool's, the
pick's, the cost's and the objective's names, dp_greedy.name_of's).

`score`: the cut plus the shortest prefix of the added roads whose D, the build's, reaches the
cut's D plus each dD, on its own carved mesh with partition's measures (<build>/score.json), and
as the 5810 page reads an answer (networks.Lined, answer `spurs`) in
spurs_rows/<RUNTIME>/ps.<greedy>.D<the cut's budget plus dD>.<the cut's start>.x<dD>.v<partition's
rev>/, up to D PAGE_D. A dD the build does not reach is refused (build with a larger `extra`).

dp_greedy.py is not committed (.superpowers/sdd/2026-10-22-flow-roads/, with power_roads.py and
the site's flow5810/compute.py it loads, and the lineup's networks it pools,
road_rows/lineup-f8fa4e53/scratch/networks/): a cluster task gets them by `--unpack`, with the
cut's partition rows.

    S=research/roadless/spurs.py          # from the repo root, PYTHONPATH=.
    G="--pool lineup --pick lazy:5 --cost D --objective P"
    uv run python $S build <block> <cut's plan> gpu|cpu $G --extra 0.02
    uv run python $S score <block> <cut's plan> gpu|cpu $G --extra 0.02 --dD 0.01,0.02
    uv run python $S run <block> <cut's plan> gpu|cpu $G --extra 0.02 --dD 0.01,0.02   # both
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
FLOW = HERE.parents[1] / ".superpowers/sdd/2026-10-22-flow-roads"     # dp_greedy.py's home
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FLOW))
import common  # noqa: E402
import dp_greedy as dpg  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import metrics  # noqa: E402
import partition as pt  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402

BUILDS = HERE / "spurs_builds"
ROWS = HERE / "spurs_rows"
MESHES = gcl.CACHE / "spurs_dp"     # the pool's carved mesh (the node's own disk on the cluster)
PAGE_D = 0.15             # the 5810 page's slider ends here: no page rows beyond it


@dataclasses.dataclass(frozen=True, kw_only=True)
class Greedy:
    """The ΔP greedy, resolved once from the command line (dp_greedy's specs)."""
    pool: dpg.Pool
    pick: dpg.Chooser
    cost: dpg.Cost
    objective: dpg.Objective
    extra: float              # the build stops once D reaches the cut's own D plus this

    @property
    def name(self) -> str:
        return dpg.name_of(self.pool, self.pick, self.cost, self.objective).removeprefix(
            "dp_greedy_")

    def out(self, bid: str, plan: str) -> Path:
        return BUILDS / metrics.RUNTIME / f"{self.name}.x{self.extra:g}" / bid / plan


def cut_of(bid: str, plan: str) -> tuple[shapely.Geometry, NDArray[np.object_], float]:
    """(the cut's ground, its lines, its D) from the partition run's rows."""
    d = pt.OUT / metrics.RUNTIME / plan
    ground = gpd.read_parquet(d / f"{bid}_cut_ground.parquet").geometry.iloc[0]
    lines = gpd.read_parquet(d / f"{bid}_cut_lines.parquet").geometry.to_numpy()
    D = float(pd.read_parquet(d / f"{bid}.parquet").iloc[-1]["cut_D_exact"])
    return ground, lines, D


def build(bid: str, plan: str, device: str, *, greedy: Greedy) -> None:
    ground, lines, D = cut_of(bid, plan)
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    [b] = common.build_blocks([bid])
    pool = dpg.candidates(b, greedy.pool)
    cut = gpd.GeoDataFrame(dict(geometry=[shapely.MultiLineString(list(lines))],
                                width_m=[pt.ROAD_W], source=["cut"]), crs=b.crs)
    cand = gpd.GeoDataFrame(pd.concat([pool, cut], ignore_index=True), crs=b.crs)
    corr = dpg.corridors_of(pool) + [ground]
    solver: dpg.Solver = dpg.Serial() if device == "gpu" else dpg.Forked(workers=dpg.WORKERS)
    bd = dpg.Build(b, metric, cand, corridors=corr, pick=greedy.pick, cost=greedy.cost,
                   objective=greedy.objective, solver=solver, meshes=MESHES)
    print(f"{plan}: the cut's D {D:.4f}, to {D + greedy.extra:.4f}", flush=True)
    out = greedy.out(bid, plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    bd.run(out, base=[len(pool)], last=D + greedy.extra)


def score(bid: str, plan: str, device: str, *, greedy: Greedy, extras: list[float]) -> None:
    ground, lines, D = cut_of(bid, plan)
    d = greedy.out(bid, plan)
    steps = json.loads((d / "steps.json").read_text())
    built = gpd.read_parquet(d / "built.parquet")
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    [b] = common.build_blocks([bid])
    geo = gc.geometry(b)
    out = {}
    for x in extras:
        hit = [s for s in steps if s["D"] >= D + x]
        if not hit:
            raise SystemExit(f"{plan}: the build ends at D {steps[-1]['D']:.4f}, short of the "
                             f"cut's {D:.4f} plus {x:g} (build with --extra >= {x:g})")
        s = hit[0]
        add = built.iloc[1:s["n_roads"]]          # built[0] is the cut
        net = shapely.union(ground, road_corridor(add)) if len(add) else ground
        net = pt.gridded(geo, shapely.intersection(net, b.boundary))
        t0 = time.time()
        sc = pt.score_answer(b, metric, net, np.concatenate([lines, add.geometry.to_numpy()]))
        out[f"{x:g}"] = dict(step=s["step"], roads=len(add), D_build=s["D"], L_P_build=s["L_P"],
                             **{k: v for k, v in sc.items() if isinstance(v, (int, float))})
        print(f"{plan} +{x:g}: step {s['step']}, {len(add)} roads, D {sc['D_exact']:.4f} (the "
              f"build's {s['D']:.4f}), L_P {sc['L_P']:.4f} (the build's {s['L_P']:.4f}), C "
              f"{sc['C']:.0f}, dead ends {sc['dead_ends']} ({time.time() - t0:.0f}s)", flush=True)
        p = pt.Plan.of(plan.rsplit(".p", 1)[0])
        if p.D + x > PAGE_D + 1e-9:
            continue
        rd = (ROWS / metrics.RUNTIME
              / f"ps.{greedy.name}.D{round(p.D + x, 4):g}.{p.start.name}.x{x:g}.v{pt.REV}")
        rd.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(geometry=[net], crs=b.crs).to_parquet(rd / f"{bid}_spurs_ground.parquet")
        gpd.GeoDataFrame(geometry=list(np.concatenate([lines, add.geometry.to_numpy()])),
                         crs=b.crs).to_parquet(rd / f"{bid}_spurs_lines.parquet")
        pd.DataFrame([{f"spurs_{k}": v for k, v in out[f"{x:g}"].items()}]).assign(
            cut=plan, greedy=greedy.name, extra=greedy.extra).to_parquet(rd / f"{bid}.parquet")
    (d / "score.json").write_text(json.dumps(out, indent=1) + "\n")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("build", "score", "run"))
    ap.add_argument("block")
    ap.add_argument("plan", help="the cut's partition plan (partition_rows/<RUNTIME>/<plan>)")
    ap.add_argument("device", choices=("gpu", "cpu"))
    ap.add_argument("--pool", required=True, help="dp_greedy.pool_of's spec")
    ap.add_argument("--pick", required=True, help="dp_greedy.chooser_of's spec")
    ap.add_argument("--cost", required=True, help="dp_greedy.cost_of's spec")
    ap.add_argument("--objective", required=True, help="dp_greedy.objective_of's spec")
    ap.add_argument("--extra", required=True, type=float, help="the build's D past the cut's")
    ap.add_argument("--dD", help="score: the D past the cut's at which each answer is read, "
                                 "comma-separated")
    a = ap.parse_args(argv)
    greedy = Greedy(pool=dpg.pool_of(a.pool), pick=dpg.chooser_of(a.pick),
                    cost=dpg.cost_of(a.cost), objective=dpg.objective_of(a.objective),
                    extra=a.extra)
    if a.verb != "build" and a.dD is None:
        ap.error(f"{a.verb} takes --dD")
    if a.verb in ("build", "run"):
        build(a.block, a.plan, a.device, greedy=greedy)
    if a.verb in ("score", "run"):
        score(a.block, a.plan, a.device, greedy=greedy,
              extras=[float(x) for x in a.dD.split(",")])


if __name__ == "__main__":
    main(sys.argv[1:])
