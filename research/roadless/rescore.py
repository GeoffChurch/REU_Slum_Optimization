"""Is the greedy exploiting its own discretization? Re-score its Lens A clearing and the road
lineup's Lens A prefixes under OTHER model settings (finer grid, shifted grid, other ell).

    PYTHONPATH=. uv run python research/roadless/rescore.py <n blocks> <workers>
"""
from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import cleared_through  # noqa: E402

SETTINGS = {                       # name: (h, ell, offset)
    "base": (0.5, 3.0, (0.3713, 0.1931)),
    "h0.25": (0.25, 3.0, (0.3713, 0.1931)),
    "shift": (0.5, 3.0, (0.62, 0.81)),
    "ell1": (0.5, 1.0, (0.3713, 0.1931)),
    "ell10": (0.5, 10.0, (0.3713, 0.1931)),
}
ARMS = ["resistance_lp", "cycle_native", "greedy_arterial_access_displacement"]
_BLOCKS: dict = {}
_ARMS: dict = {}


def one(bid: str) -> list[dict]:
    from reblock.budget import prefix_to_displacement, road_corridor
    from reblock.derivations import propose
    b = _BLOCKS[bid]
    g = pd.read_parquet(HERE / "clear_rows_M4_h0.5" / f"{bid}.parquet").sort_values("step")
    stop = g[g.D >= 0.10 - 1e-9].step.iloc[0]
    removed = cleared_through(g, int(stop))
    rows = []
    for name, (h, ell, off) in SETTINGS.items():
        t = time.time()
        sc = common.Scorer(b, h, lifted.Params(ell_m=ell, K=8), offset=off)
        gr = sc.grid
        geom = shapely.union_all(sc.polys[removed])
        op = (gr.isub & (~gr.bsub | gr.sub_of(geom))).mean(axis=-1)
        rows.append(dict(block=bid, setting=name, arm="clear_bldg",
                         perm=1 - sc.P_free(op) / sc.P0))
        for arm in ARMS:
            pre = prefix_to_displacement(b, propose(_ARMS[arm], b).roads, 0.10)
            op = (gr.isub & (~gr.bsub | gr.sub_of(road_corridor(pre)))).mean(axis=-1)
            rows.append(dict(block=bid, setting=name, arm=arm, perm=1 - sc.P_free(op) / sc.P0))
        print(f"{bid} {name} {time.time() - t:.0f}s", flush=True)
    return rows


def main(n: int, workers: int) -> None:
    global _BLOCKS, _ARMS
    done = sorted(p.stem for p in (HERE / "clear_rows_M4_h0.5").glob("*.parquet"))
    rng = np.random.default_rng(1)
    pick = sorted(rng.choice(done, size=min(n, len(done)), replace=False))
    blocks = common.build_blocks(common.recipients())
    _BLOCKS = {b.block_id: b for b in blocks if b.block_id in pick}
    _ARMS = common.arms()
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, pick) for r in rs]
    d = pd.DataFrame(rows)
    d.to_parquet(HERE / "rescore.parquet")
    piv = d.pivot_table(index=["block", "setting"], columns="arm", values="perm")
    piv["best_road"] = piv[ARMS].max(axis=1)
    piv["gap"] = piv.clear_bldg - piv.best_road
    print("\nclear_bldg minus the best road arm, per setting: median [min, max], share > 0")
    for s, g in piv.groupby(level="setting"):
        print(f"  {s:6s} greedy {g.clear_bldg.median():.3f}  best road {g.best_road.median():.3f}"
              f"  gap {g.gap.median():+.3f} [{g.gap.min():+.3f}, {g.gap.max():+.3f}]"
              f"  {np.mean(g.gap > 0):.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
