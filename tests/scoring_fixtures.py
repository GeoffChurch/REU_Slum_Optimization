"""Fixture builders shared by the scoring-equivalence harness (tests/test_scoring_equivalence.py)
and reused by later perf-refactor tasks. Reloads the 1808 sample block + its road sets (no
`propose()` calls) and pairs them with the reference (E, directness, curves, AUC) values pinned
in `tests/data/scoring/ref_values_1808.json`."""
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely import wkt

from reblock.contracts import Block
from reblock.data.kblock import KblockSource

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF: dict[str, dict[str, Any]] = json.loads(
    (_REPO_ROOT / "tests/data/scoring/ref_values_1808.json").read_text())


def _block_1808() -> Block:
    src = KblockSource(_REPO_ROOT / "tests/data/kblock/blocks_dji_sample.parquet",
                       _REPO_ROOT / "tests/data/kblock/buildings_dji_sample.parquet", "dji",
                       block_ids=["DJI.3_1_1808"])
    return next(iter(src.region().blocks))


def _roads(block: Block, key: str) -> gpd.GeoDataFrame | None:
    r = _REF[key]
    if "wkt" not in r:
        return None
    return gpd.GeoDataFrame(geometry=[wkt.loads(w) for w in r["wkt"]], crs=block.parcels.crs)


def sampled_fixtures() -> list[tuple[str, Block, gpd.GeoDataFrame | None, dict[str, Any]]]:
    b = _block_1808()
    return [(k, b, _roads(b, k), _REF[k]) for k in ("no_roads", "dijkstra", "arterial_buildable")]
