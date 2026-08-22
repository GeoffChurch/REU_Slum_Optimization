"""The committed city tier: schema, column alignment, and precision/recall against the bake-off CSV.

The heavy test is ONE @pytest.mark.slow (see tests/test_region_grow_bundle.py's docstring for why).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.dts_keys import json_keys, ts_field_names

OUT = Path("examples/screen-map")
DTS = Path("web/src/screen_map.d.ts")
CSV_PATH = Path("examples/screen-bakeoff/screen_comparison.csv")
# The site's spine, read rather than typed -- `scripts/gen_screen_map.py`'s own `FOLLOW_SOURCE`.
# Naming the same artifact from both sides is what makes the guard below a comparison of two
# independently-produced bakes rather than a comparison of one bake against a literal here.
FOLLOW_SOURCE = Path("examples/perm-graph/bundle.json")
# Every artifact that names the followed block, and the field it names it in. The baker reads ONLY
# `FOLLOW_SOURCE`; the other three agree with it by hand today (`gen_region_grow.py`'s `SEED` is a
# typed literal), so nothing but the test below stops a re-bake of one of them from moving a stage
# off the spine while the marker keeps pointing at perm-graph's block.
SPINE_SOURCES = {FOLLOW_SOURCE: "block_id",
                 Path("examples/displacement-field/field.json"): "block_id",
                 Path("examples/method-comparison/frontier.json"): "block_id",
                 Path("examples/region-grow/hood.json"): "seed"}

pytestmark = pytest.mark.skipif(not (OUT / "capetown.json").exists(), reason="tier not baked")

# The bundle's `reblock.metric`-style metric names -> gen_screen_bakeoff.py's own display strings
# in screen_comparison.csv's `metric` column (that script's own METRICS list). Explicit and closed:
# the row lookup this replaces matched by `startswith(floor["metric"].split("_")[0])`, which is
# exactly the fragile runtime-string reach into a known-at-authoring-time set this project's own
# methodology forbids -- "depth_density_proxy" and "density_compactness" share the prefix "density"
# once split on "_", so that lookup was one rename away from picking the wrong row silently.
METRIC_CSV_NAME = {
    "depth_density_proxy": "depth_density proxy   √(nA)/P · n/A",
    "density": "density   n/A",
    "density_compactness": "density_compactness   n/P²",
    "depth_proxy": "depth proxy   √(nA)/P",
}


def _metric(name: str, n: float, a: float, p: float) -> float:
    """The four cheap screens (design §3.1), recomputed independently of `reblock.metric` and of
    `scripts/gen_screen_map.py`'s own `_score` -- two paths computing the same formula and agreeing
    is the guard; importing either would make this a test of the import, not the arithmetic."""
    if name == "depth_density_proxy":
        return math.sqrt(n * a) / p * (n / a)
    if name == "density":
        return n / a
    if name == "density_compactness":
        return n / p ** 2
    if name == "depth_proxy":
        return math.sqrt(n * a) / p
    raise ValueError(name)


@pytest.fixture(scope="session")
def capetown() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((OUT / "capetown.json").read_text(encoding="utf-8"))
    return result


@pytest.fixture(scope="session")
def nairobi() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((OUT / "nairobi.json").read_text(encoding="utf-8"))
    return result


def test_dts_declares_exactly_the_keys_both_bundles_carry(capetown: dict[str, Any],
                                                          nairobi: dict[str, Any]) -> None:
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    carried = json_keys(capetown) | json_keys(nairobi)
    assert carried - declared == set(), "carried but not declared"
    # `informal` is declared optional and carried only by Cape Town, so it is in `carried`.
    assert declared - carried == set(), "declared but not carried"


@pytest.mark.parametrize("city", ["capetown", "nairobi"])
def test_every_column_has_n_blocks_entries(city: str, request: pytest.FixtureRequest) -> None:
    """A truncated column would shorten the map without changing its shape -- no error, no blank
    canvas, just fewer blocks than the city has."""
    b = request.getfixturevalue(city)
    for column in ("block_id", "n", "area_m2", "perimeter_m", "rings"):
        assert len(b[column]) == b["n_blocks"], (city, column)


def test_capetown_carries_ground_truth_and_nairobi_does_not(capetown: dict[str, Any],
                                                             nairobi: dict[str, Any]) -> None:
    """Nairobi has no published informal layer (reblock.data.informal records the search). The
    field is ABSENT, not null -- a null column is a field that looks answerable and is not."""
    assert len(capetown["informal"]) == capetown["n_blocks"]
    assert set(capetown["informal"]) <= {0, 1}
    assert "informal" not in nairobi


def _crossings(ring: list[list[float]], x: float, y: float) -> int:
    """How many edges of one closed ring the ray from (`x`, `y`) towards +x crosses. `polygon_rings`
    emits shapely ring coordinates, whose last point repeats the first, so `ring[:-1]` against
    `ring[1:]` is every edge exactly once and the two are the same length."""
    return sum(
        1 for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:], strict=True)
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) / (y1 - y0) * (x1 - x0))


def test_the_followed_block_is_the_one_every_later_stage_uses(capetown: dict[str, Any]) -> None:
    """The site's spine, derived rather than typed -- and asserted across ALL FOUR stages, not just
    the one the baker reads. perm-graph, displacement-field and method-comparison each pin a
    `block_id` and region-grow carries a `seed`; the four agree only by hand, so without this a
    re-bake of any of the other three onto a different block would move that stage off the spine
    and leave both the marker and this file green.

    It also makes `FOLLOW_SOURCE`'s identity a checked fact rather than a coincidence: because all
    four are asserted equal, pointing the baker at any other one of them is provably the same
    bake."""
    pinned = {path: json.loads(path.read_text(encoding="utf-8"))[field]
              for path, field in SPINE_SOURCES.items()}
    assert len(set(pinned.values())) == 1, f"the site's stages pin different blocks: {pinned}"
    want = pinned[FOLLOW_SOURCE]
    follow = capetown["follow"]
    assert follow["block_id"] == want
    assert capetown["block_id"][follow["index"]] == want


def test_the_follow_marker_sits_inside_its_own_block(capetown: dict[str, Any]) -> None:
    """A marker outside its polygon would draw the ring in a neighbour's block -- silently, and at
    ~0.6 CSS px² per block on the widget's own map, invisibly wrong rather than obviously wrong.
    Hence
    `representative_point()` in the baker and not `centroid`: measured on this bundle, 1,491 of its
    16,451 blocks have a centroid that falls outside their own polygon.

    The bounding box pins the
    COORDINATE FRAME -- `follow.x`/`y` are origin-relative like every ring, so a world-CRS value
    would land ~6,200 km away, which is `origin` itself: 250 km of easting and 6,192 km of northing,
    the northing dominating by 25x.

    Two assertions, and the second SUBSUMES the first -- every point outside the bounding box is
    also outside the polygon, so the box adds no failure class of its own. It is kept for the
    diagnostic: a frame mistake fails it with both the offending number and the ring's own range in
    the message, where the crossing count would report only "outside". The crossing count is what
    catches the case the box cannot -- a point in a concavity or in an interior ring is inside the
    box and outside the block."""
    follow = capetown["follow"]
    rings = capetown["rings"][follow["index"]]
    xs = [p[0] for p in rings[0]]
    ys = [p[1] for p in rings[0]]
    assert min(xs) <= follow["x"] <= max(xs)
    assert min(ys) <= follow["y"] <= max(ys)
    crossings = sum(_crossings(r, follow["x"], follow["y"]) for r in rings)
    assert crossings % 2 == 1, f"marker is outside the block ({crossings} ray crossings)"


def test_nairobi_has_no_follow_key_at_all(nairobi: dict[str, Any]) -> None:
    """The followed block is in Cape Town. ABSENT, not null -- a null field is one that looks
    answerable and is not, exactly as `informal` is handled."""
    assert "follow" not in nairobi


def test_both_cities_carry_the_follow_colour(capetown: dict[str, Any],
                                             nairobi: dict[str, Any]) -> None:
    """The ENCODING is shared even where the marker is not: the widget reads its colour from
    whichever bundle is active, and a city switch must not leave it undefined."""
    for b in (capetown, nairobi):
        assert b["encoding"]["follow_color"].startswith("#")


def test_the_interior_rings_survived(capetown: dict[str, Any], nairobi: dict[str, Any]) -> None:
    """Measured: 6,990 Cape Town and 1,139 Nairobi blocks have a hole. Losing them changes no
    count any other test here checks."""
    assert sum(len(r) - 1 for r in capetown["rings"]) == 6990
    assert sum(len(r) - 1 for r in nairobi["rings"]) == 1139


def test_precision_and_recall_at_the_shipped_floor_match_the_bakeoff(
        capetown: dict[str, Any]) -> None:
    """Two independently computed paths agreeing. The CSV comes from gen_screen_bakeoff.py's own
    ranking; this recomputes from the bundle's raw n/A/P and ground-truth column. The numbers are
    READ from the CSV, never restated here -- a literal would make this a test of my typing.
    """
    rows = {r["metric"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))
            if r.get("floor")}
    assert rows, "the bake-off CSV must carry at least one shipped floor"

    n = capetown["n"]
    a = capetown["area_m2"]
    p = capetown["perimeter_m"]
    informal = capetown["informal"]
    total_informal = sum(informal)

    for floor in capetown["floors"]:
        if floor["precision"] is None:
            continue
        row = rows[METRIC_CSV_NAME[floor["metric"]]]
        scores = [_metric(floor["metric"], n[i], a[i], p[i]) for i in range(capetown["n_blocks"])]
        selected = [i for i, s in enumerate(scores) if s >= floor["value"]]
        hits = sum(informal[i] for i in selected)
        assert len(selected) == int(float(row["floor_n"])), floor["metric"]
        assert math.isclose(hits / len(selected), float(row["floor_prec"]), rel_tol=1e-6)
        assert math.isclose(hits / total_informal, float(row["floor_recall"]), rel_tol=1e-6)


@pytest.mark.slow
def test_bundle_matches_a_fresh_reload(capetown: dict[str, Any], nairobi: dict[str, Any]) -> None:
    """The staleness guard the tests above cannot provide: they only read the committed JSON and
    the bake-off CSV, so nothing here detects a stale capetown.json/nairobi.json if the kblock data
    or the ground-truth labelling changes upstream. This reloads both cities LIVE via
    `scripts.gen_screen_map.load_blocks` and diffs `block_id`/`n` against the committed columns,
    plus Cape Town's `informal` column against a fresh `reblock.data.informal.label_blocks` call.

    Deliberately does NOT re-simplify geometry or recompute `rings` byte-for-byte:
    `test_the_interior_rings_survived` and `test_every_column_has_n_blocks_entries` already cover
    geometry shape, and the `polygon_rings`-vs-`polygon_ring` fault injection (task-7-report.md)
    covers the encoding path -- redoing either here would only make this test slower for no added
    signal. Measured: ~1.2 s warm-cache for both cities' `block_id`/`n` -- there is no per-block
    solver in this generator, unlike RegionGrow's `DenseClusterRegionBuilder`.

    DEVELOPER-LOCAL BY DESIGN, same rationale as
    `tests/test_region_grow_bundle.py::test_bundle_is_what_production_builds_today` (`slow` is
    deliberately not deselected in CI, so the guard is opt-in via a cache check rather than opt-out
    via addopts -- a contributor with a warm cache gets the real guard for free, and CI gets neither
    the guard nor the download).

    Guards TWICE, on two different artifacts, rather than once: the two block parquets (needed for
    every assertion here), and -- only for the `informal` check -- the informal-structures
    shapefile too. `reblock.data.informal.settlement_extents` needs an 18 MB Edinburgh DataShare
    download when it is absent (`tests/test_informal_ground_truth.py`: "the network-touching path
    is exercised by the example generator, not the suite"), and a contributor can easily have the
    block parquets (from running almost any other example generator) without ever having fetched
    that. Skipping the WHOLE test over one missing file would throw away the free block_id/n guard
    for everyone in that position.
    """
    capetown_cache = Path.home() / ".cache" / "reblock" / "blocks_capetown_full.parquet"
    nairobi_cache = Path.home() / ".cache" / "reblock" / "blocks_nairobi_full.parquet"
    if not capetown_cache.exists() or not nairobi_cache.exists():
        pytest.skip("needs the capetown_full and nairobi_full caches; run "
                    "`pixi run python -m scripts.gen_screen_map`")

    from scripts.gen_screen_map import CITIES, load_blocks

    fresh_capetown = load_blocks("capetown", CITIES["capetown"])
    assert [str(x) for x in fresh_capetown["block_id"]] == capetown["block_id"], (
        "capetown: block_id is stale")
    assert [int(x) for x in fresh_capetown["building_count"]] == capetown["n"], (
        "capetown: n is stale")

    fresh_nairobi = load_blocks("nairobi", CITIES["nairobi"])
    assert [str(x) for x in fresh_nairobi["block_id"]] == nairobi["block_id"], (
        "nairobi: block_id is stale")
    assert [int(x) for x in fresh_nairobi["building_count"]] == nairobi["n"], (
        "nairobi: n is stale")

    shapefile = Path.home() / ".cache" / "reblock" / "coct_is" / "CoCT_IS_STRUCTURES_201802.shp"
    if not shapefile.exists():
        pytest.skip("needs the informal-structures shapefile (an 18 MB fetch no other test "
                    "performs); run `pixi run python -m scripts.gen_screen_bakeoff` once")

    from reblock.data.informal import label_blocks, settlement_extents

    extents = settlement_extents("capetown", epsg=CITIES["capetown"])
    _, label = label_blocks(fresh_capetown, extents)
    assert [int(x) for x in label] == capetown["informal"], "capetown: informal is stale"


def test_the_committed_readme_is_what_the_generator_writes() -> None:
    """The README's prose is a pure function of the two committed bundles, and this pins it there.

    Same guard `tests/test_displacement_field_bundle.py` puts on that directory's README, and it is
    what makes the followed block's on-screen size a COMPUTED number rather than a typed one: those
    px² figures used to be three literals in the generator's f-string, so a re-bake onto another
    block -- or onto another city extent -- would have left them standing and wrong, with nothing
    to fail. `_follow_px2` now derives them from the bundle's own rings and extent every bake; this
    is the line that catches the README not having been regenerated afterwards.

    `sizes` is rebuilt the way `main()` builds it, from the files themselves: `main()` measures the
    exact byte string it then writes, so each committed file's own bytes ARE that string, and
    gzipping them at the same level reproduces the same two numbers. No bake needed, and no source
    data -- only what is committed.
    """
    import gzip

    from scripts.gen_screen_map import CityBundle, readme_markdown

    bundles: dict[str, CityBundle] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for city in ("capetown", "nairobi"):
        raw = (OUT / f"{city}.json").read_bytes()
        bundles[city] = json.loads(raw)
        sizes[city] = (len(raw), len(gzip.compress(raw, compresslevel=9)))

    readme = OUT / "README.md"
    assert readme.read_text(encoding="utf-8") == readme_markdown(bundles, sizes), (
        "examples/screen-map/README.md is stale or hand-edited; regenerate it: "
        "pixi run python -m scripts.gen_screen_map")
