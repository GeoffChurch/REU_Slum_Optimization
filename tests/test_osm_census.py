"""Tests for scripts/osm_census.py's batch decode + driver logic.

Run via module form (`pixi run python -m pytest tests/test_osm_census.py`); `scripts` is a regular
package (see pyproject.toml's `pythonpath = ["."]` note) so `from scripts import osm_census`
resolves at collection time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Polygon

from reblock.data.osm_extract import FOOTPATH_TAGS, NEAR_MISS_TAGS
from scripts import osm_census


def _write_geoparquet_1_0_shaped(path: Path, gdf: gpd.GeoDataFrame) -> None:
    """Write `gdf` shaped exactly like the real ~/.cache/reblock/{ZAF,KEN}_geodata.parquet:
    geometry is a plain Arrow `binary` field (no per-field GeoArrow extension type), and the `geo`
    GeoParquet metadata describing its encoding/CRS lives ONLY in the file-level schema metadata.

    This local geopandas/pyarrow install writes BOTH the file-level `geo` JSON AND a per-field
    Arrow extension type (`ARROW:extension:name = geoarrow.wkb`) on `gdf.to_parquet(path)` --
    which `gpd.GeoDataFrame.from_arrow` decodes via the field-level extension type, not the
    file-level JSON, so a plain `to_parquet` round-trip does NOT reproduce the real files' shape
    (confirmed empirically). Stripping the field-level extension metadata after writing, while
    leaving the file-level `geo` metadata untouched, produces the true GeoParquet-1.0 shape.
    """
    gdf.to_parquet(path)
    table = pq.read_table(path)
    file_meta = table.schema.metadata
    stripped_fields = [
        pa.field(f.name, pa.binary()) if f.name == "geometry" else f for f in table.schema
    ]
    stripped_schema = pa.schema(stripped_fields, metadata=file_meta)
    stripped_table = pa.Table.from_arrays(table.columns, schema=stripped_schema)
    pq.write_table(stripped_table, path)


def _sample_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"block_id": ["a", "b"], "building_count": [10, 20], "k_complexity": [1, 2]},
        geometry=[
            Polygon([(18.50, -33.95), (18.51, -33.95), (18.51, -33.94), (18.50, -33.94)]),
            Polygon([(18.52, -33.95), (18.53, -33.95), (18.53, -33.94), (18.52, -33.94)]),
        ],
        crs=4326,
    )


def test_from_arrow_fails_on_geoparquet_1_0_shaped_batch(tmp_path: Path) -> None:
    """Confirms the failure mode the blocker describes: on a GeoParquet-1.0-shaped batch (real
    country files' shape), `gpd.GeoDataFrame.from_arrow` raises -- passing `geometry="geometry"`
    fails identically."""
    path = tmp_path / "shaped.parquet"
    _write_geoparquet_1_0_shaped(path, _sample_gdf())
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=10, columns=["block_id", "geometry"]))
    with pytest.raises(ValueError, match="No geometry column found"):
        gpd.GeoDataFrame.from_arrow(batch)
    with pytest.raises(ValueError, match="No geometry column found"):
        gpd.GeoDataFrame.from_arrow(batch, geometry="geometry")


def test_decode_batch_reads_real_geometry_from_geoparquet_1_0_shaped_batch(
    tmp_path: Path,
) -> None:
    """The regression guard for the blocker: `_decode_batch` must correctly recover real Polygon
    geometries from a batch shaped exactly like ~/.cache/reblock/ZAF_geodata.parquet, where
    `from_arrow` (see test above) cannot."""
    path = tmp_path / "shaped.parquet"
    _write_geoparquet_1_0_shaped(path, _sample_gdf())
    pf = pq.ParquetFile(path)
    batch = next(
        pf.iter_batches(
            batch_size=10, columns=["block_id", "building_count", "k_complexity", "geometry"]
        )
    )
    decoded = osm_census._decode_batch(batch)
    assert list(decoded["block_id"]) == ["a", "b"]
    assert decoded.crs is not None and decoded.crs.to_epsg() == 4326
    assert all(g.geom_type == "Polygon" for g in decoded.geometry)
    assert decoded.geometry.iloc[0].area == pytest.approx(0.0001, rel=0.05)


def test_read_footpath_and_near_miss_lines_reads_the_pbf_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading FOOTPATH_TAGS and NEAR_MISS_TAGS as two separate `read_pbf_lines` calls pays for
    the GDAL OSM driver's multi-GB temp SQLite build twice; must be exactly one call over the
    union, split afterward on the `highway` column."""
    calls: list[tuple[str, ...]] = []

    def fake_read_pbf_lines(pbf_path: Path, tags: tuple[str, ...]) -> gpd.GeoDataFrame:
        calls.append(tuple(tags))
        return gpd.GeoDataFrame(
            {"highway": ["path", "footway", "service", "residential"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 4,
            crs=4326,
        )

    monkeypatch.setattr(osm_census, "read_pbf_lines", fake_read_pbf_lines)
    footpaths, near_miss = osm_census._read_footpath_and_near_miss_lines(Path("x.osm.pbf"))

    assert len(calls) == 1
    assert set(calls[0]) == set(FOOTPATH_TAGS) | set(NEAR_MISS_TAGS)
    assert sorted(footpaths["highway"]) == ["footway", "path"]
    assert sorted(near_miss["highway"]) == ["residential", "service"]


def _prepare_fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_blocks: int) -> None:
    monkeypatch.setattr(osm_census, "CACHE", tmp_path)
    (tmp_path / "osm_pbf").mkdir(parents=True, exist_ok=True)
    (tmp_path / "osm_pbf" / osm_census.PBF["ZAF"]).write_bytes(b"fake-pbf-contents")
    blocks = gpd.GeoDataFrame(
        {
            "block_id": [f"b{i}" for i in range(n_blocks)],
            "building_count": [10] * n_blocks,
            "k_complexity": [1] * n_blocks,
        },
        geometry=[
            Polygon(
                [
                    (18.50 + i * 0.01, -33.95),
                    (18.51 + i * 0.01, -33.95),
                    (18.51 + i * 0.01, -33.94),
                    (18.50 + i * 0.01, -33.94),
                ]
            )
            for i in range(n_blocks)
        ],
        crs=4326,
    )
    blocks.to_parquet(tmp_path / "ZAF_geodata.parquet")
    empty = gpd.GeoDataFrame({"highway": []}, geometry=[], crs=4326)
    monkeypatch.setattr(osm_census, "read_pbf_lines", lambda *a, **k: empty)


def test_limit_is_exact_not_rounded_up_to_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--limit 3` with `--batch-size` bigger than 3 must process exactly 3 blocks, not the whole
    first batch."""
    _prepare_fake_cache(tmp_path, monkeypatch, n_blocks=10)
    monkeypatch.setattr(
        sys, "argv", ["osm_census", "--iso", "ZAF", "--limit", "3", "--batch-size", "50000"]
    )
    osm_census.main()
    out = pd.read_parquet(tmp_path / "osm_coverage_ZAF.parquet")
    assert len(out) == 3


def test_resumed_run_skips_already_censused_blocks_and_appends_new_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed run must resume from its own checkpointed output rather than starting over --
    running the same command twice (second time with a higher --limit) must not duplicate the
    blocks the first run already censused."""
    _prepare_fake_cache(tmp_path, monkeypatch, n_blocks=10)

    monkeypatch.setattr(
        sys, "argv", ["osm_census", "--iso", "ZAF", "--limit", "4", "--batch-size", "50000"]
    )
    osm_census.main()
    first = pd.read_parquet(tmp_path / "osm_coverage_ZAF.parquet")
    assert len(first) == 4
    assert set(first["block_id"]) == {"b0", "b1", "b2", "b3"}

    monkeypatch.setattr(
        sys, "argv", ["osm_census", "--iso", "ZAF", "--limit", "10", "--batch-size", "50000"]
    )
    osm_census.main()
    second = pd.read_parquet(tmp_path / "osm_coverage_ZAF.parquet")
    assert len(second) == 10
    assert set(second["block_id"]) == {f"b{i}" for i in range(10)}
    # No duplicate rows for blocks the first run already censused.
    assert second["block_id"].is_unique
