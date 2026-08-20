"""The committed field bundle, and whether it still describes what Python computes.

`examples/displacement-field/field.json` is baked once and committed, so nothing recomputes it on
the way to the browser. These tests are the only thing between a bad bake and a wrong picture --
and, because the widget draws from the bundle while the fallback PNG is drawn by matplotlib from
`reblock.render`'s own constants, they are also the only thing stopping the interactive figure and
its own fallback image from becoming two different pictures under one caption.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples/displacement-field/field.json"
PNG = ROOT / "examples/displacement-field/field.png"
DTS = ROOT / "web/src/field.d.ts"


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(BUNDLE.read_text(encoding="utf-8"))
    return loaded


def test_the_committed_dts_is_what_the_generator_writes() -> None:
    """Piece C left this open: nothing asserted the committed .d.ts equalled the generator's own
    template, so a hand edit was caught only for keys the recursive guard happened to walk."""
    from scripts.gen_displacement_field import DTS_TEMPLATE
    assert DTS.read_text(encoding="utf-8") == DTS_TEMPLATE, (
        "web/src/field.d.ts was hand-edited; regenerate it: "
        "pixi run python -m scripts.gen_displacement_field")


def test_every_declared_field_is_present_and_the_shapes_agree(bundle: dict[str, Any]) -> None:
    from reblock.permeability import PermeabilityParams
    b = bundle
    n = b["n_buildings"]
    assert n > 0
    for key in ("x", "y", "r"):
        assert len(b["buildings"][key]) == n, (
            f"buildings.{key} has {len(b['buildings'][key])} of {n}")
    assert len(b["roads"]) == 2, "two default roads (spec section 2)"
    assert all(len(r["coords"]) == 2 for r in b["roads"]), "the default roads are straight segments"
    assert b["width"]["floor_m"] == PermeabilityParams.min_road_width_m, (
        f"the slider floor is {b['width']['floor_m']} but permeability.py:205 raises below "
        f"{PermeabilityParams.min_road_width_m} -- the slider would offer a road the metric "
        f"refuses to score. Re-bake: pixi run python -m scripts.gen_displacement_field")
    assert b["width"]["default_m"] >= b["width"]["floor_m"]
    assert b["width"]["max_m"] > b["width"]["default_m"]
    assert len(b["reference"]) == 6, "six parity fixtures (spec section 6)"


def test_the_reference_fixtures_cover_the_cases_that_could_hide_a_bug(
        bundle: dict[str, Any]) -> None:
    """A fixture set that is five variations of the same road proves one thing five times."""
    cases = {c["name"]: c for c in bundle["reference"]}
    assert set(cases) == {"road1", "apart", "coincident", "widest", "in_a_gap", "outside"}, (
        sorted(cases))
    assert cases["outside"]["sum_c"] == 0.0, (
        "the outside-the-block fixture must be EXACTLY zero -- it is the only fixture that pins "
        "the clip at d = r rather than a tolerance")
    # Overlap is free, and the honest form of that is an EQUALITY, not an inequality: a road drawn
    # twice IS one road, because each road is buffered on its own and only then unioned. Measured:
    # both 32.0260. Any implementation that charges per-road instead of per-union breaks this
    # immediately, where "coincident < apart" would still pass.
    assert cases["coincident"]["sum_c"] == cases["road1"]["sum_c"], (
        "two coincident roads must cost EXACTLY what one costs")
    assert cases["apart"]["sum_c"] > cases["road1"]["sum_c"], "adding a disjoint road adds cost"
    # Width, isolated: same road, 20 m against 7 m. Measured 68.1452 against 32.0260.
    assert cases["widest"]["sum_c"] > cases["road1"]["sum_c"]
    # Position, isolated: same width, near-identical length (144.35 m against 143.67 m), through
    # the field's widest gap. Measured 21.8509 against 32.0260. NOT zero -- see _cases' docstring.
    assert 0.0 < cases["in_a_gap"]["sum_c"] < cases["road1"]["sum_c"]


@pytest.mark.slow
def test_the_bundle_and_the_closed_form_both_still_match_live_python(
        bundle: dict[str, Any]) -> None:
    """THE one block-loading test in this feature. It carries two jobs, because the load is the
    cost and everything else is microseconds:

    1. the committed bundle has not gone stale against the code that made it, and
    2. the closed-form identity holds on all EIGHT methods' real road sets -- 9 to 337 segments,
       including multi-part geometry that synthetic fixtures do not produce.

    Job 2 lived in Task 2 until it was measured: three tests sharing a module-scoped fixture became
    three concurrent block loads under `-n auto` (18 minutes, killed). Task 2 now pins the identity
    against shapely on synthetic geometry, which is fast and in one respect stronger (it can show
    the residual falling quadratically with `quad_segs`); this is where it meets real roads.

    Measured expectations, worst relative disagreement 4.36e-04, closed form higher in all eight
    (shapely's buffer is INSCRIBED, so shapely's distance is the one that is too large, and a
    larger d means a smaller c): clearance 102.3728/102.3888, clearance_looped 217.6588/217.6764,
    cycle_native 51.2719/51.2880, euclidean_grid 123.9661/123.9667,
    greedy_arterial_access_displacement 138.1986/138.2589, osm_footpaths 90.3466/90.3619,
    resistance_lp 52.5456/52.5624, topology 178.7863/178.8301.

    DEVELOPER-LOCAL, like every other block-loading test here: it needs ~/.cache/reblock's city
    data and CI must stay hermetic (tests/conftest.py).
    """
    blocks = Path.home() / ".cache" / "reblock" / "blocks_capetown_full.parquet"
    if not blocks.exists():
        pytest.skip("needs the capetown_full cache; run "
                    "`pixi run python -m scripts.gen_displacement_field`")

    import numpy as np
    from geopandas import GeoDataFrame, points_from_xy

    from reblock.budget import building_radii, displacement, displacement_from_distance
    from scripts._bundle_io import cm, sigfig
    from scripts._default_road import closed_form_distance, segments
    from scripts._example_block import load_example_block
    from scripts.gen_displacement_field import roads_from_case

    block, roads_by_method = load_example_block(None)
    radii = building_radii(block.building_points)
    ox, oy = float(bundle["origin"][0]), float(bundle["origin"][1])

    # Job 1a: the BLOCK has not moved under the bundle. Quantised on this side rather than
    # comparing raw metres, because the bundle only ever held the quantised numbers -- so an exact
    # equality is available here and a tolerance would be a choice, not a necessity.
    assert bundle["n_buildings"] == len(block.building_points)
    np.testing.assert_array_equal(bundle["buildings"]["x"],
                                  [cm(v - ox) for v in block.building_points.geometry.x])
    np.testing.assert_array_equal(bundle["buildings"]["y"],
                                  [cm(v - oy) for v in block.building_points.geometry.y])
    np.testing.assert_array_equal(bundle["buildings"]["r"], [sigfig(v) for v in radii])

    # Job 1b: every fixture's `sum_c`, recomputed from the bundle's OWN coordinates. Exact through
    # the same quantiser, not a tolerance: `sum_c` IS `sigfig(displacement(...))` of the quantised
    # road against the quantised field (`gen_displacement_field.quantised_field`), so no slack is
    # needed and any tolerance would have to be looser than the thing it is guarding. The store's
    # own worst case is 5e-06 relative -- half a step in the 6th significant digit, largest when
    # the mantissa is just above 1 -- which is wider than a last-digit typo in the committed file.
    bx = np.asarray(bundle["buildings"]["x"], dtype=float) + ox
    by = np.asarray(bundle["buildings"]["y"], dtype=float) + oy
    pts = GeoDataFrame(geometry=points_from_xy(bx, by), crs=block.crs)
    radii_b = np.asarray(bundle["buildings"]["r"], dtype=float)
    for case in bundle["reference"]:
        recomputed = displacement(pts, radii_b, roads_from_case(block, case, (ox, oy)))
        assert sigfig(recomputed) == case["sum_c"], (
            f"{case['name']}: the committed bundle says {case['sum_c']}, the code now computes "
            f"{recomputed}; regenerate: pixi run python -m scripts.gen_displacement_field")

    # Job 2: the closed form against shapely on the eight methods' REAL road sets, at raw
    # precision -- nothing here is quantised, because this is about the formula.
    raw_x = block.building_points.geometry.x.to_numpy(dtype=float)
    raw_y = block.building_points.geometry.y.to_numpy(dtype=float)
    assert len(roads_by_method) == 8, sorted(roads_by_method)
    for name, roads in sorted(roads_by_method.items()):
        truth = displacement(block.building_points, radii, roads)
        closed = displacement_from_distance(
            radii, closed_form_distance(raw_x, raw_y, segments(roads)))
        assert closed == pytest.approx(truth, rel=1e-3), name
        assert closed >= truth - 1e-9, (
            f"{name}: the closed form came out LOWER than shapely ({closed} < {truth}). Shapely's "
            "buffer is inscribed, so its distances are the large ones and its c the small ones -- "
            "a reversal means the formula changed, not the discretisation")


def test_coordinates_are_relative_to_the_origin_and_not_significant_figure_rounded(
        bundle: dict[str, Any]) -> None:
    """The coordinate-precision trap: 6 significant figures on a ~6,240,000 UTM northing quantises
    to 10 m, which dissolves the parcel geometry."""
    b = bundle
    assert len(b["origin"]) == 2
    assert abs(b["origin"][1]) > 1e6, "the origin should be the real UTM offset"
    ys = [y for ring in b["parcels"] for _, y in ring]
    assert max(abs(y) for y in ys) < 1e4, "coordinates are not relative to origin"
    # Centimetre precision means at least some coordinate has a non-zero second decimal.
    assert any(round(y, 2) != round(y, 1) for y in ys), (
        "every coordinate is decimetre-round: these look `sigfig`-rounded, not `cm`-rounded")
    # ...and NO coordinate carries more than centimetre precision. This is the half that catches
    # `sigfig` applied to an origin-relative coordinate: 6 significant figures on a ~200 m local
    # offset is 3-4 decimals, which is finer than `cm` rather than coarser, so the assertion above
    # would happily pass on it.
    for key in ("x", "y"):
        vals = list(b["buildings"][key])
        assert all(v == round(v, 2) for v in vals), (
            f"buildings.{key} carries sub-centimetre precision: quantised with `sigfig`, not `cm`")
    assert all(y == round(y, 2) for y in ys), (
        "a parcel coordinate carries sub-centimetre precision: quantised with `sigfig`, not `cm`")


def test_the_baked_colours_are_actually_in_the_committed_png(bundle: dict[str, Any]) -> None:
    """D1's pattern: the widget's colours and the fallback image's colours must be the same
    colours, not two lists kept in step by hand. A reader with JS off and a reader with JS on are
    looking at the same figure or the page is lying to one of them.

    A tolerance of 12 summed over three channels absorbs the corridor's alpha 0.25 and the disks'
    alpha = c, both of which composite the constant before it reaches a pixel. Shown to be doing
    real work rather than accepting anything: widening a baked colour by one hex step still passes,
    changing it to a different hue fails.
    """
    import numpy as np
    from PIL import Image

    with Image.open(PNG) as img:
        px = np.asarray(img.convert("RGB"), dtype=np.int32).reshape(-1, 3)
    for key in ("disk_color", "road_color", "boundary_color"):
        want = str(bundle["encoding"][key])
        rgb = np.array([int(want[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.int32)
        closest = int(np.abs(px - rgb).sum(axis=1).min())
        assert closest <= 12, (
            f"encoding.{key} = {want} appears nowhere in {PNG} (closest pixel is {closest} away "
            f"summed over three channels): the widget and its own fallback image are drawing "
            f"different colours")
