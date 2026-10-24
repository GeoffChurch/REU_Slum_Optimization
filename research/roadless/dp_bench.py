"""ΔP greedy's exact step timed on a device (BACKLOG, "Try everything": cuDSS on a cluster card).

The flow-roads study's ΔP greedy (.superpowers/sdd/2026-10-22-flow-roads/dp_greedy.py, not
committed) solves each step's best candidates exactly, each a state of one mesh that carves its
whole pool of candidate roads: on 5810's pool (966 roads, 13.1M triangles) a candidate takes 11 s
on a core, 5.9 of them SuperLU's factorization. Here that step on a device: the pool
(dp_bench/<id>_pool.parquet, dp_greedy.pool's, shipped by `--unpack`) carved into one mesh, the
Thomson top-12 build replayed to its step 100 (dp_bench/<id>_step.json: the roads built before
it and the twelve candidates it solved, with their logged drops in P), the built state and each
candidate solved exactly (the atoms any of their roads carves, their corridors' union paved), each
drop against the logged one, and per solve the model's timing (gap_clear.Timing).
Rows: dp_bench/rows/<id>_<device>.parquet.

    uv run python research/roadless/dp_bench.py <id> cpu|gpu [--replace]
"""
from __future__ import annotations

import copy
import json
import socket
import sys
import time
from pathlib import Path
from typing import cast

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
import metrics  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402

DIR = HERE / "dp_bench"
MESHES = DIR / ".meshes"         # the pool's carved mesh, the node's own


def P_of(m: gcl.Model, nets: list[int]) -> tuple[float, dict[str, float]]:
    """P with the candidates `nets` built (one network of one stop that carves what any of them
    carves, its corridor their union: the model's carving swapped on a shallow copy), and the
    model's timing of it."""
    c = m.carving
    on = (c.stop[:, nets] >= 1).any(axis=1).astype(np.int64)[:, None]
    corridor = shapely.union_all([c.corridors[i][0] for i in nets])
    mc = copy.copy(m)
    mc.carving = c._replace(stop=on, stops=(1,), corridors=((corridor,),),
                            digest=f"merged{len(nets)}")
    mc.timing = gcl.Timing()
    st = mc.road_state(gcl.Prefix(net=0, stop=1))
    P = float(mc._rhs(st, mc.pop) @ mc.solve(st, mc.pop))
    return P, dict(mc.timing.t)


def run(bid: str, device: str, *, standing: clearings.Standing) -> None:
    out = DIR / "rows" / f"{bid}_{device}.parquet"
    if not standing.todo(out):
        return
    with _peak(device) as peak:
        t0 = time.time()
        metric = metrics.metric_of(metrics.RUNTIME, device=device)
        [b] = common.build_blocks([bid])
        cand = gpd.read_parquet(DIR / f"{bid}_pool.parquet")
        step = json.loads((DIR / f"{bid}_step.json").read_text())
        geo = gc.geometry(b)
        carving = gcl.Carving.of(geo, [[geo.local(road_corridor(cast(gpd.GeoDataFrame,
                                                                 cand.iloc[[i]])))]
                                       for i in range(len(cand))])
        MESHES.mkdir(parents=True, exist_ok=True)
        fx = metric.fixed(b, cache=MESHES, carving=carving)
        m = metric.model(b, fx, demand=common.PopulationDemand())
        setup = time.time() - t0
        print(f"{bid} {device}: {fx.fm.nt} triangles, setup {setup:.0f}s", flush=True)
        rows = []
        ts = time.time()
        P, tm = P_of(m, step["order"])
        rows.append(dict(i=-1, wall=time.time() - ts, P=P, dP=np.nan, dP_log=np.nan, **tm))
        for tr in step["tried"]:
            ts = time.time()
            Pi, tm = P_of(m, step["order"] + [tr["i"]])
            rows.append(dict(i=tr["i"], wall=time.time() - ts, P=Pi, dP=P - Pi,
                             dP_log=tr["dP"], **tm))
            print(f"{bid} {device} candidate {tr['i']}: {rows[-1]['wall']:.2f}s, dP "
                  f"{P - Pi:.4f} (log {tr['dP']:.4f})", flush=True)
    df = pd.DataFrame(rows).assign(block=bid, device=device, host=socket.gethostname(),
                                   triangles=fx.fm.nt, setup=setup, step=step["step"],
                                   peak_rss_gb=gap_calib.peak_gb(), gpu_gb=peak.peak)
    standing.write(df, out)


class _NoPeak:
    """No GPU: its peak is nan."""
    peak = np.nan

    def __enter__(self) -> _NoPeak:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _peak(device: str) -> gap_calib.GpuPeak | _NoPeak:
    return gap_calib.GpuPeak() if device == "gpu" else _NoPeak()


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 2 and argv[1] in ("cpu", "gpu"):
        run(argv[0], argv[1], standing=standing)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
