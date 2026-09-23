"""Whether this machine holds the city data a developer-local test reads, so it can SKIP rather
than download it. Shared through a non-test module, the precedent tests/scoring_fixtures.py set.

A test that loads a variant on the footprint tier (`conf/buildings/footprints.yaml`) needs more
than the city's block and point caches: `FootprintTiles` fetches the chosen blocks' polygon tile
on a miss -- ~320 MB, over the network -- which a test must never do.
"""
from __future__ import annotations

from pathlib import Path

from reblock.data.footprints import _FORMAT, DEFAULT_FOOTPRINT_CACHE

CACHE = Path.home() / ".cache" / "reblock"


def capetown_footprints_cached() -> bool:
    """The capetown_full block cache, plus at least one finished polygon tile at the CURRENT tile
    format -- an older format's tiles are re-fetched, not read. It cannot tell WHICH tile a region
    needs without loading the region; on a machine that has run any footprint example it is the
    pinned block's."""
    tiles = DEFAULT_FOOTPRINT_CACHE / f"v{_FORMAT}"
    return (CACHE / "blocks_capetown_full.parquet").exists() and any(
        tiles.glob("*/SOURCE_SHA256"))
