from pyproj import CRS

from reblock.methods.desire_lines import _overpass_query, _parse_overpass_geom

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
        ]
    }
    gdf = _parse_overpass_geom(payload, UTM)
    assert len(gdf) == 1                                   # the single 2-node way
    assert gdf.crs == UTM                                  # reprojected off 4326
    assert gdf.geometry.iloc[0].geom_type == "LineString"
