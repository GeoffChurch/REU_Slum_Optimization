"""Benchmark the derive() L2 cache: cold (cleared) vs warm wall-time for a real
Cape Town multi-block reblock, plus the derivation cache's disk footprint.
Usage: PYTHONPATH=. pixi run python scripts/bench_cache.py
"""
from __future__ import annotations

import time
from pathlib import Path

from reblock import derive_graph
from reblock.data.kblock import KblockSource
from reblock.data.provision import ensure_city_data
from reblock.eval.kcomplexity import KComplexityEval
from reblock.methods.peel import PeelReblocker
from reblock.pipeline import PipelineSpec, run
from reblock.screen.identity import IdentityScreen

BLOCK_IDS = ["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_42413", "ZAF.9.3.1_1_21255"]


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _timed_run(blocks_path: Path, buildings_path: Path) -> float:
    spec = PipelineSpec(
        source=KblockSource(str(blocks_path), str(buildings_path),
                            region_id="capetown", block_ids=BLOCK_IDS),
        screen=IdentityScreen(BLOCK_IDS),
        method=PeelReblocker(),
        evals=[KComplexityEval()],
        max_blocks=len(BLOCK_IDS),
    )
    t0 = time.perf_counter()
    run(spec)
    return time.perf_counter() - t0


def main() -> None:
    blocks_path, buildings_path = ensure_city_data("capetown")
    cache_dir = Path(derive_graph._CACHE_DIR)

    derive_graph.memory.clear(warn=False)
    derive_graph.clear_l1()
    cold = _timed_run(blocks_path, buildings_path)
    cold_disk = _dir_bytes(cache_dir)

    warm = _timed_run(blocks_path, buildings_path)
    warm_disk = _dir_bytes(cache_dir)

    print(f"blocks: {len(BLOCK_IDS)}  method=peel")
    print(f"COLD (cache cleared): {cold:6.2f}s")
    print(f"WARM (cache hit):     {warm:6.2f}s   speedup {cold / warm:4.1f}x")
    print(f"cache disk: {cold_disk/1e6:6.2f} MB after cold, {warm_disk/1e6:6.2f} MB after warm")


if __name__ == "__main__":
    main()
