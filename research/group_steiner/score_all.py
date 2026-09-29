"""Every arm on the 220 blocks, at k = 0, 1, 2 access and under Lens A / Lens B.

Arms: the method-comparison lineup (+ clearance at depth 1), topology as shipped (a2 s0) and its best
of 30 runs (least road), and the group Steiner solver's optimal trees for k = 0, 1, 2 (gst.py).
Lineup and topology proposals are the derive-cached ones topo_obj.py produced.

    pixi run python score_all.py run <workers> [solver]   # 'solver' = only the solver arms
    pixi run python score_all.py report

Per arm and block: for each k, the least street-first prefix reaching peel depth k + 1
(`prefix_to_depth`), its road and displacement fraction; Lens A (10% displacement) permeability,
road and max depth; Lens B (P* 0.60) displacement and road.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOPO = HERE
GAPS = HERE
REPO = Path("/home/gchurchill/src/reblock")
sys.path.insert(0, str(GAPS))
sys.path.insert(0, str(TOPO))
ROWS = HERE / "score_rows"
# every solver tree directory is an arm: trees_k1 -> gst_k1, trees_k0_lam30 -> gst_k0_lam30
SOLVER_ARMS = tuple(sorted("gst_" + p.name[len("trees_"):] for p in HERE.glob("trees_*")
                           if p.is_dir()))

_BLOCKS: list = []
_ARMS: dict = {}
_ONLY_SOLVER = False


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def topology_runs() -> tuple[dict[str, str], set[tuple[str, str]]]:
    """(least-road successful run per block, every successful (block, run)) -- only blocks the
    topology study FINISHED, so no proposal here ever re-runs topology (uncapped) from cold."""
    t = pd.concat([pd.read_parquet(p) for p in (TOPO / "rows").glob("*.parquet")])
    t = t[t.arm.str.startswith("topology") & (t.error.fillna("") == "")]
    best = t.loc[t.groupby("block").total_m.idxmin()].set_index("block").arm.to_dict()
    return best, set(zip(t.block, t.arm, strict=True))


def roads_for(block, arm: str):
    from reblock.derivations import propose
    if arm in SOLVER_ARMS:
        p = HERE / f"trees_{arm[len('gst_'):]}" / f"{block.block_id}.parquet"
        return gpd.read_parquet(p) if p.exists() else None
    if arm == "topology_best":
        name = _ARMS["_best"].get(block.block_id)
        return None if name is None else propose(_ARMS["_topo"][name], block).roads
    if arm == "topology_shipped":
        if (block.block_id, "topology_a2_s0") not in _ARMS["_topo_ok"]:
            return None
        return propose(_ARMS["_topo"]["topology_a2_s0"], block).roads
    return propose(_ARMS[arm], block).roads


def score(block, roads, pcfg) -> dict:
    from reblock.budget import displacement, max_access_depth, prefix_to_depth
    from reblock.compare import lens_prefixes
    from reblock.derive.access import STREET_TOL, ParcelAdjacency
    from reblock.permeability import EgressContext, permeability
    n = len(block.buildings)
    adj = ParcelAdjacency.of(block, STREET_TOL)

    def length(g) -> float:
        return float(g.geometry.length.sum()) if g is not None and len(g) else 0.0

    row: dict = dict(total_m=length(roads), full_depth=max_access_depth(adj, roads),
                     depth_before=max_access_depth(adj, None))
    for k in (0, 1, 2):
        if roads is None or len(roads) == 0:
            ok = row["depth_before"] <= k + 1
            row.update({f"k{k}_reached": ok, f"k{k}_m": 0.0 if ok else np.nan,
                        f"k{k}_disp": 0.0 if ok else np.nan})
            continue
        pre, depth = prefix_to_depth(adj, roads, k + 1)
        ok = depth <= k + 1
        row.update({f"k{k}_reached": ok, f"k{k}_m": length(pre) if ok else np.nan,
                    f"k{k}_disp": displacement(block.buildings, pre) / n if ok else np.nan})
    ctx = EgressContext.of(block, pcfg.params)
    if roads is None or len(roads) == 0:
        row.update(a_P=permeability(ctx, roads), a_m=0.0, a_reached=False, b_reached=False)
        return row
    lp = lens_prefixes(ctx, roads, pcfg)
    a_d = displacement(block.buildings, lp.displacement) / n
    row.update(a_P=permeability(ctx, lp.displacement), a_m=length(lp.displacement), a_D=a_d,
               a_reached=bool(a_d >= pcfg.matched_displacement - 1e-9),
               a_depth=max_access_depth(adj, lp.displacement),
               b_D=displacement(block.buildings, lp.permeability) / n,
               b_m=length(lp.permeability), b_reached=bool(lp.reached),
               full_P=permeability(ctx, roads))
    return row


def one(i: int) -> None:
    from reblock.compare import load_permeability_config
    pcfg = load_permeability_config(REPO / "conf")
    block = _BLOCKS[i]
    arms = [a for a in _ARMS["_order"] if (a in SOLVER_ARMS) or not _ONLY_SOLVER]
    t0 = time.time()
    done = 0
    for arm in arms:
        out = ROWS / f"{block.block_id}__{arm}.parquet"
        if out.exists():
            continue
        try:
            roads = roads_for(block, arm)
        except Exception as e:                  # topology's disconnected-block crash
            roads, err = None, f"{type(e).__name__}: {e}"[:200]
        else:
            err = ""
        if roads is None and (arm in SOLVER_ARMS or arm.startswith("topology")) and not err:
            continue            # not solved yet, or topology unfinished / timed out on this block
        row = dict(block=block.block_id, n_parcels=len(block.parcels), arm=arm, error=err)
        if not err:
            row.update(score(block, roads, pcfg))
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        pd.DataFrame([row]).to_parquet(tmp)
        os.replace(tmp, out)
        done += 1
    log(f"{block.block_id} {len(block.parcels)} parcels: {done} arms {time.time() - t0:.0f}s")


def main() -> None:
    global _BLOCKS, _ARMS, _ONLY_SOLVER
    import topo_obj
    ROWS.mkdir(exist_ok=True)
    workers = int(sys.argv[2])
    _ONLY_SOLVER = len(sys.argv) > 3 and sys.argv[3] == "solver"
    _BLOCKS = topo_obj.blocks()
    reg = topo_obj.arms()
    _ARMS = {k: v for k, v in reg.items() if not k.startswith("topology")}
    _ARMS["_topo"] = {k: v for k, v in reg.items() if k.startswith("topology")}
    _ARMS["_best"], _ARMS["_topo_ok"] = topology_runs()
    _ARMS["_order"] = [*topo_obj.LINEUP, "clearance_d1", "topology_shipped", "topology_best",
                       *SOLVER_ARMS]
    log(f"{len(_BLOCKS)} blocks")
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(one, range(len(_BLOCKS))))
    log("done")


if __name__ == "__main__":
    main()
