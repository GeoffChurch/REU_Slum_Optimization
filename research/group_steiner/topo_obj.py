"""Every method scored on topology's OWN objective (Brelsford & Bettencourt 2015/2019):
the least new road that makes a block universally accessible -- every parcel fronting a road.

Per (block, arm): does the proposal reach universal access (max BFS access depth 1), and if so the
road length and displacement of the shortest street-first prefix that does (`prefix_to_depth`,
the canonical prefix every lens uses). Topology is run as shipped (alpha 2, seed 0) and as the
paper runs it: many samples at alpha 10 / 100, keep the shortest.

    NUMBA_NUM_THREADS=1 pixi run python topo_obj.py time <n>          # time topology on n blocks
    NUMBA_NUM_THREADS=1 pixi run python topo_obj.py run <workers> [limit]
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GAPS = HERE
REPO = Path("/home/gchurchill/src/reblock")
sys.path.insert(0, str(GAPS))
sys.path.insert(0, str(REPO))

ROWS = HERE / "rows"
LOG = HERE / "progress.log"
LINEUP = ["clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
          "resistance_lp", "euclidean_grid", "greedy_arterial_access_displacement"]
SEEDS = range(10)
ALPHAS = (2.0, 10.0, 100.0)

CAP_S = 900


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


signal.signal(signal.SIGALRM, _alarm)

_BLOCKS: list = []
_ARMS: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def blocks() -> list:
    import study220
    out = study220.build_blocks(study220.recipients())
    return sorted(out, key=lambda b: len(b.buildings))            # smallest first


def arms() -> dict:
    from hydra import compose, initialize_config_dir

    from reblock.methods.topology import TopologyMethod
    from reblock.presets import load_methods
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "conf")):
        cfg = compose(config_name="compare_config", overrides=["buildings=footprints"])
    reg = load_methods(cfg.all_methods)
    out = {}
    for name in LINEUP:
        m = reg[name]
        if name == "cycle_native_betweenness_contrast":
            sub = m.substrate
            m = dataclasses.replace(m, substrate=dataclasses.replace(
                sub, desire_source=dataclasses.replace(sub.desire_source, workers=1)))
        if name == "greedy_arterial_access_displacement":
            m = dataclasses.replace(m, workers=1)     # pool workers are daemonic: no nested pool
        out[name] = m
    # clearance ships at depth_target 2 (every parcel at most ONE parcel from a road), so it never
    # reaches universal access by design; at 1 it is the same deepest-first shortest-path greedy
    # as topology's strict variant, on the gap substrate instead of the parcel boundaries.
    out["clearance_d1"] = dataclasses.replace(reg["clearance"], depth_target=1)
    for a in ALPHAS:
        for s in SEEDS:
            out[f"topology_a{a:g}_s{s}"] = TopologyMethod(alpha=a, seed=s, road_width_m=7.0)
    return out


def score(block, roads) -> dict:
    from reblock.budget import displacement, max_access_depth, prefix_to_depth
    from reblock.derive.access import STREET_TOL, ParcelAdjacency
    adj = ParcelAdjacency.of(block, STREET_TOL)
    n = len(block.buildings)
    before = max_access_depth(adj, None)
    if roads is None or len(roads) == 0:
        return dict(depth_before=before, full_depth=before, total_m=0.0, reached=before <= 1,
                    access_m=0.0 if before <= 1 else np.nan,
                    access_disp=0.0 if before <= 1 else np.nan)
    prefix, depth = prefix_to_depth(adj, roads, 1)
    reached = depth <= 1
    return dict(depth_before=before, full_depth=max_access_depth(adj, roads),
                total_m=float(roads.geometry.length.sum()), reached=reached,
                access_m=float(prefix.geometry.length.sum()) if reached else np.nan,
                access_disp=displacement(block.buildings, prefix) / n if reached else np.nan)


def one(i: int) -> None:
    from reblock.derivations import propose
    block = _BLOCKS[i]
    out = ROWS / f"{block.block_id}.parquet"
    done = set(pd.read_parquet(out, columns=["arm"])["arm"]) if out.exists() else set()
    rows = []
    t0 = time.time()
    topo_dead = False
    for name, method in _ARMS.items():
        if name in done:
            continue
        if topo_dead and name.startswith("topology"):
            # one timeout marks the block as one topology cannot solve in CAP_S; its cost is
            # structural (all simple paths near deep parcels), not a per-seed accident
            rows.append(dict(block=block.block_id, n_bldg=len(block.buildings), arm=name,
                             secs=0.0, error="skipped: an earlier topology run timed out"))
            continue
        t = time.time()
        # topology's run time grows steeply with parcels (83 s at 257); cap each run so one block
        # cannot hold a worker for hours. A capped run is recorded as a timeout, not dropped.
        signal.alarm(CAP_S if name.startswith("topology") else 0)
        try:
            roads = propose(method, block).roads
            err = ""
        except _Timeout:
            roads, err = None, f"timeout {CAP_S}s"
            topo_dead = True
        except Exception as e:                      # topology's known crash modes, recorded
            roads, err = None, f"{type(e).__name__}: {e}"[:200]
        signal.alarm(0)
        row = dict(block=block.block_id, n_bldg=len(block.buildings), arm=name,
                   secs=time.time() - t, error=err)
        row.update(score(block, roads) if not err else {})
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        if done:
            df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, out)
    log(f"{block.block_id} {len(block.buildings)} bldg {len(rows)} arms {time.time() - t0:.0f}s")


def main() -> None:
    global _BLOCKS, _ARMS
    assert os.environ.get("NUMBA_NUM_THREADS") == "1"
    ROWS.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    _BLOCKS = blocks()
    if cmd == "time":
        from reblock.methods.topology import TopologyMethod
        n = int(sys.argv[2])
        pick = [_BLOCKS[int(q)] for q in np.linspace(0, len(_BLOCKS) - 1, n)]
        for b in pick:
            t = time.time()
            try:
                TopologyMethod(alpha=10.0, seed=0, road_width_m=7.0).propose(b)
                ok = "ok"
            except Exception as e:
                ok = f"{type(e).__name__}: {e}"[:120]
            print(b.block_id, len(b.buildings), f"{time.time() - t:.1f}s", ok, flush=True)
        return
    workers = int(sys.argv[2])
    if len(sys.argv) > 3:
        lim = int(sys.argv[3])
        _BLOCKS = [_BLOCKS[int(q)] for q in np.linspace(0, len(_BLOCKS) - 1, lim)]
    _ARMS = arms()
    log(f"{len(_BLOCKS)} blocks, {len(_ARMS)} arms")
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(one, range(len(_BLOCKS))))
    log("done")


if __name__ == "__main__":
    main()
