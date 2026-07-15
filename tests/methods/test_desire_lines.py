from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString

from reblock.methods.desire_lines import OSMDesireLines, _overpass_query, _parse_overpass_geom

UTM = CRS.from_epsg(32734)   # a South-Africa UTM zone (projected), for reprojection assertions


def test_overpass_query_uses_south_west_north_east_and_anchored_tags() -> None:
    # bbox is (min_lon, min_lat, max_lon, max_lat); Overpass wants (south,west,north,east) =
    # (min_lat, min_lon, max_lat, max_lon). Tags are anchored so "path" != "pathway".
    q = _overpass_query((18.735, -33.849, 18.755, -33.834), ["path", "footway"])
    assert "(-33.849,18.735,-33.834,18.755)" in q
    assert '["highway"~"^(path|footway)$"]' in q
    assert "out geom;" in q


def test_parse_overpass_geom_builds_linestrings_drops_short_and_reprojects() -> None:
    payload = {
        "elements": [
            {"type": "way", "id": 1, "geometry": [
                {"lat": -33.84, "lon": 18.74}, {"lat": -33.841, "lon": 18.741}]},
            {"type": "way", "id": 2, "geometry": [{"lat": -33.84, "lon": 18.74}]},  # 1 node: drop
            {"type": "node", "id": 3, "lat": -33.84, "lon": 18.74},           # not a way: skip
            {"type": "way", "id": 4, "geometry": None},           # ungeometried way: drop
        ]
    }
    gdf = _parse_overpass_geom(payload, UTM)
    assert len(gdf) == 1                                   # the single 2-node way
    assert gdf.crs == UTM                                  # reprojected off 4326
    assert gdf.geometry.iloc[0].geom_type == "LineString"


_BBOX = (18.735, -33.849, 18.755, -33.834)


def _write_geojson(path: Path, lines_lonlat: list[list[tuple[float, float]]]) -> None:
    gdf = gpd.GeoDataFrame(geometry=[LineString(c) for c in lines_lonlat],
                           crs=CRS.from_epsg(4326))
    gdf.to_file(path, driver="GeoJSON")


def test_osm_snapshot_is_loaded_without_fetching(tmp_path: Path) -> None:
    snap = tmp_path / "snap.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    src = OSMDesireLines(snapshot=str(snap))
    src._fetch = lambda query: pytest.fail("must not fetch when a snapshot is present")  # type: ignore[method-assign]
    gdf = src.desire_lines(_BBOX, UTM)
    assert len(gdf) == 1 and gdf.crs == UTM


def test_osm_cache_hit_is_loaded_without_fetching(tmp_path: Path) -> None:
    src = OSMDesireLines(cache_dir=str(tmp_path))
    # Pre-seed the cache at the exact key path the source will look for.
    cache_path = src._cache_path(_BBOX)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geojson(cache_path, [[(18.74, -33.84), (18.741, -33.841)]])
    src._fetch = lambda query: pytest.fail("must not fetch on a cache hit")  # type: ignore[method-assign]
    gdf = src.desire_lines(_BBOX, UTM)
    assert len(gdf) == 1


def test_osm_fetch_writes_cache_then_reuses_it(tmp_path: Path) -> None:
    calls = {"n": 0}
    payload = {"elements": [{"type": "way", "id": 1, "geometry": [
        {"lat": -33.84, "lon": 18.74}, {"lat": -33.841, "lon": 18.741}]}]}
    src = OSMDesireLines(cache_dir=str(tmp_path))
    src._fetch = lambda query: (calls.__setitem__("n", calls["n"] + 1), payload)[1]  # type: ignore[method-assign, func-returns-value]
    a = src.desire_lines(_BBOX, UTM)
    b = src.desire_lines(_BBOX, UTM)          # second call: cache hit, no second fetch
    assert len(a) == 1 and len(b) == 1 and calls["n"] == 1


def test_osm_identity_none_when_live_stable_with_snapshot(tmp_path: Path) -> None:
    assert OSMDesireLines().identity is None                       # live -> uncacheable
    snap = tmp_path / "snap.geojson"
    _write_geojson(snap, [[(18.74, -33.84), (18.741, -33.841)]])
    ident = OSMDesireLines(snapshot=str(snap)).identity
    assert ident is not None and isinstance(ident, tuple) and ident[0] == "osm"  # stable identity
