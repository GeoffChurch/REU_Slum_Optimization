from pathlib import Path

import geopandas as gpd
import pyogrio
import pytest
import yaml
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.data import osm_extract
from reblock.data.osm_extract import (
    FOOTPATH_TAGS,
    TOLERANCES,
    PbfDesireLines,
    assert_zone_fit,
    census_rows,
    interiority_row,
    read_pbf_lines,
    utm_zone_epsg,
)

CRS_M = CRS.from_epsg(32734)
BOUNDARY = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


def _lines(*geoms: LineString) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(geoms), crs=CRS_M)


def test_interiority_row_reports_count_and_length_at_every_tolerance() -> None:
    row = interiority_row(
        "b1", BOUNDARY, _lines(LineString([(10, 50), (90, 50)])), _lines(), CRS_M)
    assert row["block_id"] == "b1"
    for tol in TOLERANCES:
        assert row[f"n_interior_segments_{tol}"] == 1
        assert row[f"interior_length_m_{tol}"] == pytest.approx(80.0)


def test_interiority_row_count_gate_is_robust_where_length_is_not() -> None:
    """A path crossing the interior but touching the edge: length is trimmed by tolerance,
    the count is not. This is the spike's central finding and the reason both are reported."""
    row = interiority_row(
        "b2", BOUNDARY, _lines(LineString([(0, 50), (90, 50)])), _lines(), CRS_M)
    assert row["n_interior_segments_0.5"] == row["n_interior_segments_5.0"] == 1
    len_low: float = row["interior_length_m_0.5"]  # type: ignore[assignment]
    len_high: float = row["interior_length_m_5.0"]  # type: ignore[assignment]
    assert len_low > len_high


def test_interiority_row_keeps_near_miss_separate() -> None:
    row = interiority_row(
        "b3", BOUNDARY,
        _lines(LineString([(10, 50), (90, 50)])),
        _lines(LineString([(10, 20), (90, 20)])),
        CRS_M)
    assert row["n_interior_segments_0.5"] == 1
    assert row["n_near_miss_segments_0.5"] == 1
    assert row["interior_length_m_0.5"] == pytest.approx(80.0)


def test_interiority_row_uncovered_block_is_all_zero() -> None:
    row = interiority_row("b4", BOUNDARY, _lines(), _lines(), CRS_M)
    assert row["n_interior_segments_0.5"] == 0
    assert row["interior_length_m_0.5"] == 0.0


def test_config_tag_list_matches_python_definition() -> None:
    """conf/ and Python must not be able to drift: one list, one place."""
    shared = yaml.safe_load(Path("conf/desire_source/_footpath_tags.yaml").read_text())
    assert tuple(shared["footpath_tags"]) == FOOTPATH_TAGS

    osm_cfg = yaml.safe_load(Path("conf/desire_source/osm.yaml").read_text())
    assert osm_cfg["tags"] == "${footpath_tags}", (
        "osm.yaml must interpolate the shared list, not re-declare it")


def test_pbf_identity_is_stable_and_keys_on_content_and_tags(tmp_path: Path) -> None:
    """Unlike OSMDesireLines (identity None when live), a PBF source is cacheable -- which is
    what flips osm_footpaths from uncacheable to cacheable, so the identity must be content-keyed.
    """
    pbf = tmp_path / "x.osm.pbf"
    pbf.write_bytes(b"not-a-real-pbf-but-hashable")
    a = PbfDesireLines(pbf)
    b = PbfDesireLines(pbf)
    assert a.identity == b.identity
    assert a.identity is not None

    pbf2 = tmp_path / "y.osm.pbf"
    pbf2.write_bytes(b"different-content")
    assert PbfDesireLines(pbf2).identity != a.identity
    assert PbfDesireLines(pbf, tags=("footway",)).identity != a.identity


def test_pbf_desire_lines_equality_ignores_populated_caches(tmp_path: Path) -> None:
    """`_cache` is dataclass state (memoization), not identity: comparing two instances that have
    each populated their own `_cache` with a real DataFrame must not raise. Before `compare=False`
    on `_cache`, the dataclass-generated `__eq__` compared the two DataFrames with `==`, which
    pandas broadcasts elementwise, and the resulting DataFrame-of-bools has an ambiguous truth
    value -- `ValueError: The truth value of a DataFrame is ambiguous`."""
    pbf = tmp_path / "x.osm.pbf"
    pbf.write_bytes(b"not-a-real-pbf-but-hashable")
    a = PbfDesireLines(pbf)
    b = PbfDesireLines(pbf)
    a._cache = gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs=4326)
    b._cache = gpd.GeoDataFrame({"v": [2, 3]}, geometry=[Point(1, 1), Point(2, 2)], crs=4326)
    assert a == b  # equally configured (same path/tags) despite different populated caches


def test_pbf_identity_memoizes_digest_and_invalidates_on_content_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`identity` re-hashing the whole PBF on every access is measured at ~1s/block of pure
    hashing against a 3.31ms/block budget (see module docstring). A second access on an unchanged
    file must not re-read it; touching the content must still change the identity."""
    pbf = tmp_path / "x.osm.pbf"
    pbf.write_bytes(b"hello world")

    calls = 0
    real_sha256 = osm_extract._file_sha256

    def counting_sha256(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(path)

    monkeypatch.setattr(osm_extract, "_file_sha256", counting_sha256)

    src = PbfDesireLines(pbf)
    first = src.identity
    second = src.identity
    assert first == second
    assert calls == 1, "second access on an unchanged file must not re-hash"

    pbf.write_bytes(b"hello world, but longer now")  # different size -> stat signature changes
    third = src.identity
    assert third != first
    assert calls == 2, "a content change must trigger exactly one re-hash"


def test_read_pbf_lines_rejects_bare_str_tags() -> None:
    """A bare `str` IS a `Sequence[str]`, so `tags="path"` typechecks but would otherwise iterate
    character-by-character, silently building `highway IN ('p','a','t','h')` -- zero rows, no
    error. Reject it explicitly instead."""
    with pytest.raises(TypeError, match="bare str"):
        read_pbf_lines(Path("nonexistent.osm.pbf"), tags="path")


def test_read_pbf_lines_rejects_empty_tags() -> None:
    """`tags=()` would build the syntactically invalid `highway IN ()`, surfacing as an obscure
    OGR/SQL error from inside pyogrio; raise our own clear message instead."""
    with pytest.raises(ValueError, match="must not be empty"):
        read_pbf_lines(Path("nonexistent.osm.pbf"), tags=())


def test_read_pbf_lines_escapes_embedded_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tag containing an embedded single quote must not break out of the SQL string literal."""
    captured: dict[str, object] = {}

    def fake_read_dataframe(
        path: Path, *, layer: str, where: str, use_arrow: bool
    ) -> gpd.GeoDataFrame:
        captured["where"] = where
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    # Patch the `pyogrio` module object directly (not via `osm_extract.pyogrio`, which mypy
    # --strict's implicit-reexport check would flag): `read_pbf_lines` calls
    # `pyogrio.read_dataframe`, and modules are singletons in `sys.modules`, so this is the exact
    # same object `osm_extract.py`'s own `import pyogrio` resolved to.
    monkeypatch.setattr(pyogrio, "read_dataframe", fake_read_dataframe)
    read_pbf_lines(Path("nonexistent.osm.pbf"), tags=["o'brien's path"])
    assert captured["where"] == "highway IN ('o''brien''s path')"


def test_pbf_conforms_to_desire_line_source_protocol() -> None:
    """Structural conformance is enforced STATICALLY by this annotated binding -- mypy --strict
    fails if PbfDesireLines does not satisfy the Protocol. Do NOT rewrite this as
    `isinstance(..., DesireLineSource)`: DesireLineSource is a bare Protocol, not
    @runtime_checkable, so isinstance raises TypeError rather than returning False."""
    from reblock.methods.desire_lines import DesireLineSource

    source: DesireLineSource = PbfDesireLines(Path("nonexistent.osm.pbf"))
    assert callable(source.desire_lines)


def test_utm_zone_epsg_picks_hemisphere_and_zone() -> None:
    assert utm_zone_epsg(18.5, -33.9) == 32734      # Cape Town, zone 34 south
    assert utm_zone_epsg(36.8, -1.3) == 32737       # Nairobi, zone 37 south
    assert utm_zone_epsg(36.8, 1.3) == 32637        # just north of the equator


def test_assert_zone_fit_is_loud_about_a_forgotten_batch() -> None:
    """A single country-wide UTM does not crash -- it silently biases lengths by up to 3.5%.
    The assertion is what makes a missed batch loud instead of a quiet drift."""
    assert_zone_fit(18.5, 32734)                    # zone 34 central meridian is 21E
    with pytest.raises(ValueError, match="outside UTM zone"):
        assert_zone_fit(41.9, 32734)


def test_census_rows_emits_one_row_per_block() -> None:
    blocks = gpd.GeoDataFrame(
        {"block_id": ["a", "b"]},
        geometry=[
            Polygon([(18.50, -33.95), (18.51, -33.95), (18.51, -33.94), (18.50, -33.94)]),
            Polygon([(18.52, -33.95), (18.53, -33.95), (18.53, -33.94), (18.52, -33.94)]),
        ],
        crs=CRS.from_epsg(4326))
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS.from_epsg(4326))
    rows = census_rows(blocks, empty, empty, 32734)
    assert [r["block_id"] for r in rows] == ["a", "b"]
    assert all(r["n_interior_segments_0.5"] == 0 for r in rows)
    for r in rows:
        area: float = r["area_m2"]  # type: ignore[assignment]
        assert area > 0


def test_census_rows_matches_each_block_to_its_own_nearby_line_by_tree_position() -> None:
    """Regression guard for wrong-row selection. `census_rows` builds `STRtree(list(fp_m.geometry))`
    and then does `fp_m.iloc[fp_tree.query(geom)]`: the tree is keyed by POSITION in `fp_m`, so the
    query result must be consumed positionally (`.iloc`), not by label (`.loc`). A frame with a
    non-default, swapped index is exactly the case that tells the two apart -- under `.iloc` each
    block gets its own nearby line; under `.loc` (or any position/label misalignment) block "a"
    would instead receive block "b"'s line and vice versa, which -- since the two blocks and their
    lines are spatially disjoint -- clips to nothing, flipping every one of the counts and length
    comparisons asserted below rather than merely tweaking a number."""
    blocks = gpd.GeoDataFrame(
        {"block_id": ["a", "b"]},
        geometry=[
            Polygon([(18.50, -33.95), (18.51, -33.95), (18.51, -33.94), (18.50, -33.94)]),
            Polygon([(18.52, -33.95), (18.53, -33.95), (18.53, -33.94), (18.52, -33.94)]),
        ],
        crs=CRS.from_epsg(4326))
    # position 0 -> label 1 -> long line inside block "a"; position 1 -> label 0 -> short line
    # inside block "b". `.iloc[fp_tree.query(geom)]` (positional) gets this right; `.loc[...]`
    # would instead hand block "a" the SHORT line (label 0 -> position 1) and block "b" the LONG
    # one (label 1 -> position 0) -- geometries that don't even overlap that block's bbox, so the
    # clip drops them and the counts go to 0 instead of 1.
    footpaths = gpd.GeoDataFrame(
        geometry=[
            LineString([(18.502, -33.945), (18.508, -33.945)]),  # long, in block "a"
            LineString([(18.522, -33.945), (18.524, -33.945)]),  # short, in block "b"
        ],
        index=[1, 0],
        crs=CRS.from_epsg(4326))
    near_miss = gpd.GeoDataFrame(
        geometry=[
            LineString([(18.503, -33.947), (18.507, -33.947)]),  # short, in block "a"
            LineString([(18.521, -33.943), (18.529, -33.943)]),  # long, in block "b"
        ],
        index=[1, 0],
        crs=CRS.from_epsg(4326))

    rows = census_rows(blocks, footpaths, near_miss, 32734)
    row_a, row_b = rows
    assert row_a["block_id"] == "a"
    assert row_b["block_id"] == "b"
    assert row_a["n_interior_segments_0.5"] == 1
    assert row_b["n_interior_segments_0.5"] == 1
    assert row_a["n_near_miss_segments_0.5"] == 1
    assert row_b["n_near_miss_segments_0.5"] == 1

    fp_len_a: float = row_a["interior_length_m_0.5"]  # type: ignore[assignment]
    fp_len_b: float = row_b["interior_length_m_0.5"]  # type: ignore[assignment]
    nm_len_a: float = row_a["near_miss_length_m_0.5"]  # type: ignore[assignment]
    nm_len_b: float = row_b["near_miss_length_m_0.5"]  # type: ignore[assignment]
    # block "a" got the LONG footpath and the SHORT near-miss; block "b" the reverse. A
    # position/label mismatch swaps these pairings (or zeroes both), flipping one or both
    # inequalities -- so this is a real assertion on which line went where, not just presence.
    assert fp_len_a > fp_len_b
    assert nm_len_a < nm_len_b


@pytest.mark.network
def test_pbf_and_overpass_agree_on_a_pinned_bbox() -> None:
    """Two sources for the same data WILL disagree (Geofabrik extract timestamp vs live Overpass;
    GDAL `lines` layer vs Overpass `out geom`). Without this test, two sources is accommodation
    rather than a Strategy. Tolerance is loose because the snapshots differ in date, not content."""
    from reblock.methods.desire_lines import OSMDesireLines

    pbf = Path.home() / ".cache" / "reblock" / "osm_pbf" / "south-africa-latest.osm.pbf"
    if not pbf.exists():
        pytest.skip("run scripts/osm_census.py --fetch first")

    bbox = (18.55, -33.99, 18.58, -33.96)   # a Cape Flats window with dense footpath mapping
    crs = CRS.from_epsg(32734)
    a = PbfDesireLines(pbf).desire_lines(bbox, crs)
    b = OSMDesireLines(timeout_s=180.0).desire_lines(bbox, crs)
    assert a.geometry.length.sum() == pytest.approx(b.geometry.length.sum(), rel=0.25)
