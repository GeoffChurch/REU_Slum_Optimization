"""FootprintTiles provisions polygon tiles for exactly the blocks asked for -- offline.

`fetch` is injected, so no test touches the network; the manifest is pre-written into the cache.
Each test guards a specific way per-tile provisioning could fail silently.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from reblock.data.footprints import FootprintTiles, ParquetBuildings
from reblock.derive_graph import source_hash

# Two adjacent S2-ish tiles; the blocks sit wholly inside the WEST one.
_WEST = box(18.0, -34.5, 18.5, -33.5)
_EAST = box(18.5, -34.5, 19.0, -33.5)
_URL = "https://x/polygons_s2_level_4_gzip/{}_buildings.csv.gz"


def _manifest(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "geometry": g.__geo_interface__,
              "properties": {"tile_url": _URL.format(tok)}}
             for tok, g in (("w1", _WEST), ("e1", _EAST))]
    (cache / "tiles.geojson").write_text(json.dumps({"type": "FeatureCollection",
                                                    "features": feats}))


def _square(x: float, y: float, s: float = 0.0001) -> Polygon:
    return box(x, y, x + s, y + s)


class _Fetch:
    """Writes a tiny polygon tile; records every URL it is asked for."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        rows = ["latitude,longitude,area_in_meters,confidence,geometry"]
        for k in range(6):
            x, y = 18.20 + 0.001 * k, -34.00
            conf = 0.9 if k < 5 else 0.1                 # the last one is below the floor
            rows.append(f'{y},{x},12.5,{conf},"{_square(x, y).wkt}"')
        with gzip.open(dest, "wt") as fh:
            fh.write("\n".join(rows) + "\n")


def _blocks() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"block_id": ["b"]}, geometry=[box(18.19, -34.01, 18.21, -33.99)],
                            crs=4326)


def test_fetches_only_the_tiles_covering_the_blocks(tmp_path: Path) -> None:
    """The whole point: a block costs the tile it sits in, not the corpus."""
    _manifest(tmp_path)
    fetch = _Fetch()
    FootprintTiles(cache_dir=tmp_path, fetch=fetch).for_blocks(_blocks())
    assert fetch.urls == [_URL.format("w1")]


def test_a_cached_tile_is_never_fetched_again(tmp_path: Path) -> None:
    _manifest(tmp_path)
    FootprintTiles(cache_dir=tmp_path, fetch=_Fetch()).for_blocks(_blocks())

    def forbidden(url: str, dest: Path) -> None:
        raise AssertionError(f"re-fetched a cached tile: {url}")
    got = FootprintTiles(cache_dir=tmp_path, fetch=forbidden).for_blocks(_blocks())
    assert len(got.buildings) == 5


def test_returns_polygons_above_the_confidence_floor_with_their_area(tmp_path: Path) -> None:
    _manifest(tmp_path)
    bld = FootprintTiles(cache_dir=tmp_path, fetch=_Fetch()).for_blocks(_blocks()).buildings
    assert set(bld.geometry.geom_type) == {"Polygon"}
    assert len(bld) == 5                                  # the 0.1-confidence row is dropped
    assert "area_in_meters" in bld.columns


def test_hands_back_the_source_sidecar_for_the_content_hash(tmp_path: Path) -> None:
    """The caller hashes these; they must identify the SOURCE download, and be small -- hashing
    the parquet parts instead would re-read hundreds of MB per region() just to build a key."""
    _manifest(tmp_path)
    used = FootprintTiles(cache_dir=tmp_path, fetch=_Fetch()).for_blocks(_blocks()).read_from
    assert [p.name for p in used] == ["SOURCE_SHA256"]
    assert len(used[0].read_text()) == 64


def test_a_crash_mid_parse_never_leaves_a_tile_that_looks_cached(tmp_path: Path) -> None:
    """Written to a temp dir and renamed into place only when complete. A half-written tile that
    passed the cache check would silently serve a fraction of the buildings forever."""
    _manifest(tmp_path)

    def dies(url: str, dest: Path) -> None:
        raise RuntimeError("network died")
    with pytest.raises(RuntimeError):
        FootprintTiles(cache_dir=tmp_path, fetch=dies).for_blocks(_blocks())
    assert not any(tmp_path.glob("v*/w1"))
    fetch = _Fetch()                                      # and the next call really fetches
    FootprintTiles(cache_dir=tmp_path, fetch=fetch).for_blocks(_blocks())
    assert fetch.urls == [_URL.format("w1")]


def test_the_default_strategy_keeps_every_existing_cache_key(tmp_path: Path) -> None:
    """ParquetBuildings must report exactly the file the old code hashed, so
    `source_hash(blocks, *read_from)` is byte-identical to the `source_hash(blocks, buildings)` it
    replaced -- which is why adding this strategy invalidates no existing result."""
    blocks_p = Path("tests/data/kblock/blocks_capetown_sample.parquet")
    bld_p = Path("tests/data/kblock/buildings_capetown_sample.parquet")
    read_from = ParquetBuildings(bld_p).for_blocks(gpd.read_parquet(blocks_p)).read_from
    assert source_hash(blocks_p, *read_from) == source_hash(blocks_p, bld_p)


def test_footprint_parcels_are_identical_to_point_parcels(tmp_path: Path) -> None:
    """The tier must change the building MODEL, never the parcels. On real data, anchoring on the
    polygon CENTROID instead of Open Buildings' PUBLISHED point changed all 263 parcels (2.1% of
    area) -- sub-millimetre shifts flip near-cocircular Voronoi topology. Here every published point
    is deliberately offset from its centroid. Watched failing with the centroid anchor restored.

    Row ORDER is not what this guards: `_voronoi_parcels` is order-independent, and this test was
    watched PASSING with the CSV-order restore removed. That is `test_building_rows_align_*`."""
    import numpy as np

    from reblock.buildings import Footprints, SpacingDiscs
    from reblock.data.kblock import KblockSource

    rng = np.random.default_rng(7)
    lon = 18.2 + rng.uniform(-0.004, 0.004, 40)
    lat = -34.0 + rng.uniform(-0.004, 0.004, 40)
    # published point sits OFF the polygon centroid, as it can in Open Buildings
    polys = [box(x - 0.00002 + 0.000007, y - 0.00002, x + 0.00002 + 0.000007, y + 0.00002)
             for x, y in zip(lon, lat, strict=True)]

    # ONE Open Buildings CSV, read by BOTH tiers through the same parser -- as in reality. Built
    # from exact floats instead, the point tier differs from the parsed polygon tier by 1 ULP on
    # ~1 in 4 coordinates (pandas' default parser is not round-trip exact), and this test fails
    # for a reason that cannot occur on real data.
    csv = "\n".join(["latitude,longitude,area_in_meters,confidence,geometry"] + [
        f'{y},{x},12.0,0.9,"{g.wkt}"' for x, y, g in zip(lon, lat, polys, strict=True)]) + "\n"

    def fetch(url: str, dest: Path) -> None:
        with gzip.open(dest, "wt") as fh:
            fh.write(csv)

    cache = tmp_path / "fp"
    _manifest(cache)
    blocks_p = tmp_path / "blocks.parquet"
    gpd.GeoDataFrame({"block_id": ["b"], "k_complexity": [0.0]},
                     geometry=[box(18.19, -34.01, 18.21, -33.99)], crs=4326).to_parquet(blocks_p)
    points_p = tmp_path / "points.parquet"
    import io

    import pandas as pd

    from scripts.fetch_kblock_fixtures import OB_FLOAT_PRECISION
    df = pd.read_csv(io.StringIO(csv), float_precision=OB_FLOAT_PRECISION)
    gpd.GeoDataFrame(df[["area_in_meters", "confidence"]],
                     geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                     crs=4326).to_parquet(points_p)

    pts = next(iter(KblockSource(blocks_p, points_p, building_tier=SpacingDiscs,
                                 min_buildings=4).region().blocks))
    fp = next(iter(KblockSource(blocks_p, points_p, building_tier=Footprints, min_buildings=4,
                                member_buildings=FootprintTiles(cache_dir=cache, fetch=fetch)
                                ).region().blocks))
    assert list(fp.parcels["parcel_id"]) == list(pts.parcels["parcel_id"])
    assert fp.parcels.geometry.equals(pts.parcels.geometry)
    assert set(fp.building_geometries.geometry.geom_type) == {"Polygon"}


def test_building_rows_align_with_the_point_tier(tmp_path: Path) -> None:
    """Row i of the footprint tier must be the same building as row i of the point tier, so a
    per-building comparison across tiers pairs the right buildings. Tiles are stored sorted by
    lat/lon for bbox reads; without restoring CSV order they come back in a different order --
    which is how a bogus 62 m "anchor shift" was once measured, row by row. The CSV order here is
    deliberately NOT lat/lon-sorted. Watched failing with the restore removed."""
    import io

    import numpy as np
    import pandas as pd

    from scripts.fetch_kblock_fixtures import OB_FLOAT_PRECISION

    rng = np.random.default_rng(11)
    lon = 18.2 + rng.uniform(-0.004, 0.004, 30)
    lat = -34.0 + rng.uniform(-0.004, 0.004, 30)
    csv = "\n".join(["latitude,longitude,area_in_meters,confidence,geometry"] + [
        f'{y},{x},12.0,0.9,"{_square(x, y).wkt}"' for x, y in zip(lon, lat, strict=True)]) + "\n"

    def fetch(url: str, dest: Path) -> None:
        with gzip.open(dest, "wt") as fh:
            fh.write(csv)

    _manifest(tmp_path)
    got = FootprintTiles(cache_dir=tmp_path, fetch=fetch).for_blocks(_blocks()).buildings
    ref = pd.read_csv(io.StringIO(csv), float_precision=OB_FLOAT_PRECISION)
    assert np.array_equal(got["longitude"].to_numpy(), ref["longitude"].to_numpy())
    assert np.array_equal(got["latitude"].to_numpy(), ref["latitude"].to_numpy())
