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

from tests.dts_keys import json_keys, ts_field_names

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


def test_the_dts_declares_exactly_the_bundle_keys(bundle: dict[str, Any]) -> None:
    """`DTS_TEMPLATE` and the `FieldBundle` TypedDict are two literals in one module with nothing
    else pinning them together, and generating a `.d.ts` at all is only worth doing if a renamed
    Python field becomes a TypeScript error rather than a blank panel.

    BIDIRECTIONAL, for the reason tests/test_web_bundle.py's twin gives: a declaration the artifact
    no longer carries is as much a regression as a key the declaration missed, because it leaves a
    dead field a widget author will write code against. Shares its two helpers with that twin
    (tests/dts_keys.py) rather than re-deriving the `.d.ts` parse, which is where its own I3 fix
    lives -- a second regex here would not inherit it.
    """
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    present = json_keys(bundle)
    missing = present - declared
    assert not missing, f"bundle keys missing from web/src/field.d.ts: {sorted(missing)}"
    extra = declared - present
    assert not extra, (
        f"web/src/field.d.ts declares keys the bundle does not have: {sorted(extra)}")


def test_the_encoding_matches_reblock_renders_live_constants(bundle: dict[str, Any]) -> None:
    """The point of baking the colours and weights: edit `_DISPLACED_PT` (or any of the others) in
    render.py and every PNG moves while the committed bundle keeps whatever was baked, forever --
    so the widget and its own fallback image drift apart with nothing failing. Fast: imports
    `reblock.render` directly, no block load.

    Every value with a render.py source is pinned to that source. `handle_radius_px` and `pad` have
    none -- they are the web figure's own affordances -- and are the only two omitted.

    `street_lw` is pinned to `_BOUNDARY_LW` because `_draw_boundary_and_streets` draws the outline
    and the street network in one pair of calls at one width. The first bake shipped 1.0 here
    against the PNG's 1.3, which is exactly the divergence this test exists to catch, found by
    writing it.
    """
    from reblock.render import (
        _BOUNDARY_COLOR,
        _BOUNDARY_LW,
        _CONTEXT_OUTLINE,
        _CORRIDOR_ALPHA,
        _DISK_OUTLINE_LW,
        _DISPLACED_PT,
        _PARCEL_LW,
        _ROAD_COLOR,
    )

    e = bundle["encoding"]
    assert e["parcel_color"] == _CONTEXT_OUTLINE
    assert e["parcel_lw"] == _PARCEL_LW
    assert e["boundary_color"] == _BOUNDARY_COLOR
    assert e["boundary_lw"] == _BOUNDARY_LW
    assert e["street_lw"] == _BOUNDARY_LW, (
        "streets and the block outline are drawn by ONE pair of calls in "
        "`_draw_boundary_and_streets`, so the widget must draw them at one width too")
    assert e["road_color"] == _ROAD_COLOR
    assert e["road_alpha"] == _CORRIDOR_ALPHA
    assert e["disk_color"] == _DISPLACED_PT
    assert e["disk_outline_lw"] == _DISK_OUTLINE_LW
    # The two with no render.py source are still pinned to SOMETHING, so a stray edit is visible.
    assert e["handle_radius_px"] > 0.0 and e["pad"] > 0.0
    assert set(e) == {"parcel_color", "parcel_lw", "boundary_color", "boundary_lw", "street_lw",
                      "road_color", "road_alpha", "disk_color", "disk_outline_lw",
                      "handle_radius_px", "pad"}, (
        f"encoding gained or lost a key: {sorted(e)}. A new one needs a line above, or it is "
        f"unchecked -- which is the state this test was written to end")


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
    # Position, isolated: same width, near-identical length (144.317 m against 143.664 m, both
    # measured off the committed coordinates), through the field's widest gap. Measured 21.8509
    # against 32.0260. NOT zero -- see `_cases`' docstring.
    assert 0.0 < cases["in_a_gap"]["sum_c"] < cases["road1"]["sum_c"]


def test_each_derivable_fixture_moves_exactly_one_variable(bundle: dict[str, Any]) -> None:
    """The property that gives the fixture set its meaning, and the only one a reviewer previously
    had to verify BY HAND.

    Four of the six fixtures are functions of `bundle.roads` alone, and each is supposed to differ
    from `road1` in exactly one respect. If `widest` ever differed in position as well as width it
    would isolate nothing -- and every other test in this file would still pass, because the parity
    tests only ever compare a fixture's `sum_c` to a recomputation from that same fixture's own
    coordinates. An unasserted invariant a human checked once is exactly the kind that rots.

    Also the only thing that checks `reference[].roads[].coords` against anything external. Job 1b
    recomputes `sum_c` from the bundle's own fixture coordinates, so a vertex corrupted somewhere no
    building is near moves no grazing distance, leaves `sum_c` bit-identical, and passes. Here the
    coordinates are compared to `bundle.roads` DIRECTLY -- lists against lists, not lengths or
    bearings or anything else derived, because a derived quantity is exactly what a compensating
    corruption survives.

    The remaining two (`in_a_gap`, `outside`) are functions of the block's geometry rather than of
    `bundle.roads`, so they are re-derived from the live block in the slow test's Job 1a instead.
    """
    cases = {c["name"]: c for c in bundle["reference"]}
    r0, r1 = bundle["roads"][0], bundle["roads"][1]
    floor, widest_m = bundle["width"]["floor_m"], bundle["width"]["max_m"]

    assert cases["road1"]["roads"] == [{"coords": r0["coords"], "width_m": floor}], (
        "road1 must be exactly road 1 of the default pair at the floor width -- it is the baseline "
        "every other fixture is read against, so if IT moves, all five comparisons change meaning")

    assert cases["apart"]["roads"] == [{"coords": r0["coords"], "width_m": floor},
                                       {"coords": r1["coords"], "width_m": floor}], (
        "apart must differ from road1 by the ADDITION OF ROAD 2 ONLY; its first road, or a width, "
        "also changed")

    coincident = cases["coincident"]["roads"]
    assert len(coincident) == 2, f"coincident must be two roads, not {len(coincident)}"
    assert coincident[0]["coords"] == coincident[1]["coords"], (
        "coincident's two roads are not coincident -- that is the whole fixture: it proves overlap "
        "is free by costing EXACTLY what one road costs")
    assert coincident == [{"coords": r0["coords"], "width_m": floor},
                          {"coords": r0["coords"], "width_m": floor}], (
        "coincident must differ from road1 by DUPLICATION ONLY; the road drawn twice is not "
        "road 1, or a width also changed")

    assert cases["widest"]["roads"] == [{"coords": r0["coords"], "width_m": widest_m}], (
        "widest must differ from road1 in WIDTH ONLY; its coordinates also changed, so it isolates "
        "nothing and the page's width claim rests on two variables moving at once")

    # ...and the two that are not derivable from `bundle.roads` are still each a single road at the
    # floor width, so "same width, different position" is at least locally true of `in_a_gap`.
    for name in ("in_a_gap", "outside"):
        assert [r["width_m"] for r in cases[name]["roads"]] == [floor], (
            f"{name} must be ONE road at the floor width -- it isolates position, so its width "
            f"must match road1's exactly")


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

    Job 1a covers every layer, not just the buildings, because it is the only guard that can see a
    single coordinate move; the fast precision guards cannot (see their own note).

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
    from scripts._bundle_io import line_coords, polygon_ring, sigfig
    from scripts._default_road import closed_form_distance, default_roads, segments
    from scripts._example_block import load_example_block
    from scripts.gen_displacement_field import (
        WIDTH_FLOOR_M,
        fixture_roads,
        quantised_field,
        road_specs,
        roads_from_case,
    )

    block, roads_by_method = load_example_block(None)
    radii = building_radii(block.building_points)
    ox, oy = float(bundle["origin"][0]), float(bundle["origin"][1])

    # Job 1a: EVERY layer the bundle carries, re-derived from the live block through the
    # generator's own expressions, compared exactly. Exact rather than tolerant because the bundle
    # only ever held the quantised numbers, so equality is available and a tolerance would be a
    # choice; and every layer rather than just the buildings because this is the only guard that
    # can see a SINGLE coordinate move -- the precision guards can tell `cm` from `sigfig`, but a
    # real coordinate is allowed to land on 44.10, so one hand-edited vertex is invisible to them.
    #
    # `default_roads` is re-run here rather than read back, which makes this the guard
    # scripts/_default_road.py:60 names: if the principal axis's sign normalisation were dropped,
    # road 2 lands on the other side of the centre and nothing else in the codebase notices.
    assert bundle["origin"] == [float(block.parcels.total_bounds[0]),
                                float(block.parcels.total_bounds[1])]
    assert bundle["block_id"] == block.block_id
    assert bundle["n_buildings"] == len(block.building_points)
    field = quantised_field(block, radii, ox, oy)
    assert bundle["buildings"] == field.stored
    assert bundle["parcels"] == [polygon_ring(g, ox, oy, what="parcel")
                                 for g in block.parcels.geometry]
    assert bundle["boundary"] == polygon_ring(block.boundary, ox, oy, what="boundary")
    assert bundle["streets"] == [line for g in block.streets.geometry
                                 for line in line_coords(g, ox, oy)]
    default = default_roads(block, WIDTH_FLOOR_M)
    assert bundle["roads"] == road_specs(default, ox, oy)

    # The SEVENTH carrier, and the only one nothing else can see. `in_a_gap` and `outside` are
    # derived from the block's geometry (the largest nearest-neighbour gap; a translation by twice
    # the diagonal), so `bundle.roads` cannot pin them the way it pins the other four -- and Job 1b
    # cannot either, because it recomputes `sum_c` from these very coordinates: a vertex corrupted
    # where no building sits moves no grazing distance and leaves `sum_c` bit-identical. Re-derived
    # through the generator's own `fixture_roads`, which is why that was split out of `_cases`.
    derived = {name: road_specs(rs, ox, oy)
               for name, rs in fixture_roads(block, field.points, field.radii, default)}
    assert [c["name"] for c in bundle["reference"]] == list(derived), (
        f"the committed fixtures are {[c['name'] for c in bundle['reference']]} but the generator "
        f"now produces {list(derived)}")
    for case in bundle["reference"]:
        assert case["roads"] == derived[case["name"]], (
            f"fixture {case['name']}'s road geometry is not what the generator now derives; "
            f"regenerate: pixi run python -m scripts.gen_displacement_field")

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


def _coordinates(b: dict[str, Any]) -> list[tuple[str, float]]:
    """Every coordinate the bundle carries, each labelled with where it came from.

    Enumerated by NAME, not discovered by walking every float: `buildings.r`, `sum_c` and
    `fraction` go through `sigfig`, not `cm`, so a blind walk would have to guess which rule
    applies and would fail on the three fields where the answer is "the other one". The schema is
    closed and known while this is being written, which is exactly when a name beats a probe.
    """
    out: list[tuple[str, float]] = []
    out += [("buildings.x", v) for v in b["buildings"]["x"]]
    out += [("buildings.y", v) for v in b["buildings"]["y"]]
    out += [("parcels", v) for ring in b["parcels"] for xy in ring for v in xy]
    out += [("boundary", v) for xy in b["boundary"] for v in xy]
    out += [("streets", v) for line in b["streets"] for xy in line for v in xy]
    out += [("roads[].coords", v) for r in b["roads"] for xy in r["coords"] for v in xy]
    out += [(f"reference[{c['name']}].roads[].coords", v)
            for c in b["reference"] for r in c["roads"] for xy in r["coords"] for v in xy]
    return out


def test_every_coordinate_carrier_is_covered_by_the_precision_guard(
        bundle: dict[str, Any]) -> None:
    """`_coordinates` is a hand-written enumeration, so it can go stale the moment the schema grows
    a layer -- and a guard that silently stops covering a field is worse than no guard. This pins
    its total against the shapes the bundle declares, so a new coordinate carrier makes this fail
    rather than quietly slip past the precision check."""
    b = bundle
    expected = (2 * b["n_buildings"]
                + sum(2 * len(ring) for ring in b["parcels"])
                + 2 * len(b["boundary"])
                + sum(2 * len(line) for line in b["streets"])
                + sum(2 * len(r["coords"]) for r in b["roads"])
                + sum(2 * len(r["coords"]) for c in b["reference"] for r in c["roads"]))
    assert len(_coordinates(b)) == expected, (
        f"_coordinates yields {len(_coordinates(b))} values but the bundle's own shapes account "
        f"for {expected}: a coordinate carrier was added to the schema and not to the guard")


def test_coordinates_are_relative_to_the_origin_and_not_significant_figure_rounded(
        bundle: dict[str, Any]) -> None:
    """The coordinate-precision trap: 6 significant figures on a ~6,240,000 UTM northing quantises
    to 10 m, which dissolves the parcel geometry."""
    b = bundle
    assert len(b["origin"]) == 2
    assert abs(b["origin"][1]) > 1e6, "the origin should be the real UTM offset"
    coords = _coordinates(b)
    assert max(abs(v) for _, v in coords) < 1e4, "coordinates are not relative to origin"

    # TOO FINE. This is the direction that catches `sigfig` applied to an origin-relative
    # coordinate: 6 significant figures on a ~200 m local offset is 3-4 decimals, which is finer
    # than `cm` -- so the coarseness check below passes on it happily and only an exactness check
    # sees it. Walked over every carrier by name (`_coordinates`) rather than a sampled few: an
    # over-precise `reference[].roads[].coords` is the same bug in a place nobody chose to look.
    fine = [(where, v) for where, v in coords if v != round(v, 2)]
    assert not fine, (
        f"{len(fine)} coordinate(s) carry sub-centimetre precision, e.g. {fine[:4]}: these were "
        f"quantised with something other than `cm`")

    # TOO COARSE, per carrier. A decimetre- (or metre-, or 10 m-) rounded value is still exactly
    # centimetre-round, so the check above cannot see it; what it cannot be is a whole LAYER of
    # values none of which uses its second decimal. Per carrier rather than pooled, because pooling
    # 3,700 parcel coordinates would drown a decimetre-rounded 118-vertex boundary.
    #
    # LIMIT, stated because it decided the shape of the guard: this cannot catch ONE coordinate
    # rounded to a decimetre. It is not a detectable event -- a real coordinate is allowed to land
    # on 44.10 -- and the thing that does catch it is the slow test's re-derivation of every layer
    # from the live block, which is exact.
    by_carrier: dict[str, list[float]] = {}
    for where, v in coords:
        by_carrier.setdefault(where, []).append(v)
    for where, vals in by_carrier.items():
        assert not all(v == round(v, 1) for v in vals), (
            f"every one of {where}'s {len(vals)} coordinates is decimetre-round: this layer looks "
            f"rounded coarser than `cm`, which dissolves geometry while still parsing")


def test_the_baked_colours_are_actually_in_the_committed_png(bundle: dict[str, Any]) -> None:
    """D1's pattern: the widget's colours and the fallback image's colours must be the same
    colours, not two lists kept in step by hand. A reader with JS off and a reader with JS on are
    looking at the same figure or the page is lying to one of them.

    An EXACT match is available here, and the mechanism is worth stating because a plausible wrong
    story would justify a much looser tolerance. `save_render` passes `transparent=True`, so the
    PNG is RGBA and the corridor's alpha 0.25 and each disk's alpha = c live in the alpha channel;
    `convert("RGB")` DROPS that channel rather than compositing it against white, so every
    constant's exact RGB survives into the pixels this test reads. Measured: all four colours sit
    at distance 0, and a one-hex-step change measures 1.

    The tolerance of 2 is therefore unused slack, kept only as headroom against a future matplotlib
    changing how it lays down a 0.4 px wireframe edge. It is deliberately tighter than a single hex
    step is wide in aggregate -- at 12 (the first draft's value, justified by the compositing story
    that turns out not to happen) roughly 3% of random colours would have passed.
    """
    import numpy as np
    from PIL import Image

    with Image.open(PNG) as img:
        assert img.mode == "RGBA", (
            f"{PNG} is {img.mode}, not RGBA: `save_render` saves with transparent=True, and this "
            f"test's exactness depends on convert('RGB') DROPPING an alpha channel rather than "
            f"compositing it")
        px = np.asarray(img.convert("RGB"), dtype=np.int32).reshape(-1, 3)
    for key in ("disk_color", "road_color", "boundary_color", "parcel_color"):
        want = str(bundle["encoding"][key])
        rgb = np.array([int(want[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.int32)
        closest = int(np.abs(px - rgb).sum(axis=1).min())
        assert closest <= 2, (
            f"encoding.{key} = {want} appears nowhere in {PNG} (closest pixel is {closest} away "
            f"summed over three channels): the widget and its own fallback image are drawing "
            f"different colours")
