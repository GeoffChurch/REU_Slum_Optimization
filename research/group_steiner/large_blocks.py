"""The 36 large blocks of the betweenness frontier study (>= 1,000 buildings; 24 Cape Town, 12 Nairobi),
their lineup, and Lens A / Lens B scoring -- a compact rebuild of the lost research module, using the
repo's own config (`+example=explore`, footprints) so every lineup proposal is a derive-cache hit.

    PYTHONPATH=<repo> pixi run python large_blocks.py lineup <workers>    # score the lineup into large_score_rows/
"""
from __future__ import annotations

import dataclasses
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path("/home/gchurchill/src/reblock")
IDS = HERE.joinpath("large_ids.txt").read_text().split()
OUT = HERE / "large_score_rows"
LINEUP = ["clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
          "euclidean_grid", "greedy_arterial_access_displacement", "resistance_lp"]

_BLOCKS: dict = {}        # block_id -> (city, Block)
_ARMS: dict = {}          # city -> {arm: Method}
PCFG = None


def build() -> None:
    """Fill `_BLOCKS` and `_ARMS`. Blocks are built once per city at the footprint tier."""
    global PCFG
    from hydra import compose, initialize_config_dir

    from reblock.compare import load_permeability_config
    from reblock.presets import load_methods, load_stages
    PCFG = load_permeability_config(REPO / "conf")
    for city, prefix in (("capetown", "ZAF"), ("nairobi", "KEN")):
        ids = [i for i in IDS if i.startswith(prefix)]
        with initialize_config_dir(version_base=None, config_dir=str(REPO / "conf")):
            cfg = compose(config_name="compare_config",
                          overrides=["+example=explore", f"data={city}_full", "buildings=footprints"])
        blocks = list(load_stages(cfg).source.restricted(ids).region().blocks)
        assert {b.block_id for b in blocks} == set(ids), sorted(set(ids) - {b.block_id for b in blocks})
        for b in blocks:
            assert type(b.buildings).__name__ == "Footprints", type(b.buildings)
            _BLOCKS[b.block_id] = (city, b)
        reg = load_methods(cfg.all_methods)
        arms = {n: reg[n] for n in LINEUP}
        # pool workers cannot fork: the arterial's candidate pool and the betweenness all-pairs pass
        # must run serially here. `workers` is exempt from both cache keys, so proposals are unchanged.
        arms["greedy_arterial_access_displacement"] = dataclasses.replace(
            arms["greedy_arterial_access_displacement"], workers=1)
        m = arms["cycle_native_betweenness_contrast"]
        src = m.substrate.desire_source
        arms["cycle_native_betweenness_contrast"] = dataclasses.replace(
            m, substrate=dataclasses.replace(m.substrate,
                                             desire_source=dataclasses.replace(src, workers=1)))
        _ARMS[city] = arms
        print(f"{city}: built {len(blocks)}/{len(ids)} blocks", flush=True)


def by_size() -> list[str]:
    return sorted(_BLOCKS, key=lambda b: -len(_BLOCKS[b][1].buildings))


def score(bid: str, arm: str, roads, propose_s: float = 0.0) -> dict:
    """One row, as the lost large_study.one wrote it (Lens A at 10% displaced, Lens B at P* = 0.60)."""
    from reblock.compare import lens_prefixes
    from reblock.emit import pct_displaced
    from reblock.permeability import EgressContext, permeability
    city, block = _BLOCKS[bid]
    t0 = time.time()
    ctx = EgressContext.of(block, PCFG.params)
    lp = lens_prefixes(ctx, roads, PCFG)

    def length(g) -> float:
        return float(g.geometry.length.sum()) if g is not None and len(g) else 0.0

    a_d = pct_displaced(lp.displacement, block.buildings)
    return dict(city=city, block_id=bid, buildings=len(block.buildings), parcels=len(block.parcels),
                arm=arm, full_m=length(roads), full_D=pct_displaced(roads, block.buildings),
                full_P=permeability(ctx, roads), a_m=length(lp.displacement), a_D=a_d,
                a_P=permeability(ctx, lp.displacement),
                a_reached=bool(a_d >= PCFG.matched_displacement - 1e-9),
                b_m=length(lp.permeability), b_D=pct_displaced(lp.permeability, block.buildings),
                b_reached=bool(lp.reached), propose_s=propose_s, total_s=time.time() - t0 + propose_s)


def _write(row: dict) -> None:
    out = OUT / f"{row['block_id']}__{row['arm']}.parquet"
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame([row]).to_parquet(tmp)
    os.replace(tmp, out)


def _lineup_one(task: tuple[str, str]) -> None:
    from reblock.derivations import propose
    bid, arm = task
    if (OUT / f"{bid}__{arm}.parquet").exists():
        return
    city, block = _BLOCKS[bid]
    t0 = time.time()
    roads = propose(_ARMS[city][arm], block).roads
    _write(score(bid, arm, roads, time.time() - t0))
    print(f"{time.strftime('%H:%M:%S')} {bid} {arm}", flush=True)


if __name__ == "__main__":
    assert sys.argv[1] == "lineup"
    OUT.mkdir(exist_ok=True)
    build()
    order = by_size() if os.environ.get("LB_LARGEST_FIRST") else by_size()[::-1]
    tasks = [(b, a) for b in order for a in LINEUP]
    with multiprocessing.get_context("fork").Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
        list(pool.imap_unordered(_lineup_one, tasks))
    print("done", flush=True)
