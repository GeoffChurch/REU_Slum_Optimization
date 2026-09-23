"""The committed field bundle, and whether it still describes what Python computes.

`examples/displacement-field/field.json` is baked once and committed, so nothing recomputes it on
the way to the browser. These tests are the only thing between a bad bake and a wrong picture --
and, because the widget draws from the bundle while the fallback PNG is drawn by matplotlib from
`reblock.render`'s own constants, they are also the only thing stopping the interactive figure and
its own fallback image from becoming two different pictures under one caption.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.city_caches import capetown_footprints_cached
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
    from reblock.compare import load_permeability_config
    b = bundle
    n = b["n_buildings"]
    assert n > 0
    assert len(b["buildings"]) == n, f"buildings has {len(b['buildings'])} outlines of {n}"
    for i, building in enumerate(b["buildings"]):
        # Every ring CLOSED: the widget drops each ring's last vertex as the repeated first, so an
        # open ring would lose a real corner there and be clipped as a different shape.
        assert building and all(ring and len(ring) >= 4 and ring[0] == ring[-1]
                                for poly in building for ring in poly), (
            f"building {i}'s outline is not a list of polygons of closed rings")
    assert len(b["roads"]) == 2, "two default roads (spec section 2)"
    assert all(len(r["coords"]) == 2 for r in b["roads"]), "the default roads are straight segments"
    floor = load_permeability_config().params.min_road_width_m
    assert b["width"]["floor_m"] == floor, (
        f"the slider floor is {b['width']['floor_m']} but permeability.py:205 raises below "
        f"{floor} -- the slider would offer a road the metric "
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

    What this CANNOT pin, because it compares numbers and not pixels: the four stroke weights are
    one number read by two APIs with different units (matplotlib points against CSS pixels), so
    equal values here still draw ~1.27x heavier on the canvas at a 700 px figure. Measured and
    written up at `ENCODING` in scripts/gen_displacement_field.py; open under piece D.

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
        _DISPLACED_PT,
        _OUTLINE_LW,
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
    assert e["building_color"] == _DISPLACED_PT
    assert e["outline_lw"] == _OUTLINE_LW
    # POSITIVITY, and nothing more -- said plainly because the comment here used to claim "a stray
    # edit is visible", which is false: any other positive value passes, and nothing else in the
    # tree pins either number (`field-boot.test.ts` reads both back out of this same bundle). What
    # positivity is worth: zero or negative is the one wrong value that draws NOTHING and reports
    # nothing -- a zero `handle_radius_px` gives the drag a hit radius of zero, so the road stops
    # being draggable with no error anywhere, and a zero `pad` fits the drawing flush to the canvas
    # edge, clipping the outermost buildings.
    assert e["handle_radius_px"] > 0.0 and e["pad"] > 0.0
    assert set(e) == {"parcel_color", "parcel_lw", "boundary_color", "boundary_lw", "street_lw",
                      "road_color", "road_alpha", "building_color", "outline_lw",
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
        "compact support (a corridor reaching no outline costs nothing) rather than a tolerance")
    # Overlap is free, and the honest form of that is an EQUALITY, not an inequality: a road drawn
    # twice IS one road, because each road is buffered on its own and only then unioned. Measured:
    # both 91.9005. Any implementation that charges per-road instead of per-union breaks this
    # immediately, where "coincident < apart" would still pass.
    assert cases["coincident"]["sum_c"] == cases["road1"]["sum_c"], (
        "two coincident roads must cost EXACTLY what one costs")
    assert cases["apart"]["sum_c"] > cases["road1"]["sum_c"], "adding a disjoint road adds cost"
    # Width, isolated: same road, 20 m against 7 m. Measured 274.175 against 91.9005.
    assert cases["widest"]["sum_c"] > cases["road1"]["sum_c"]
    # Position, isolated PER METRE: same width, through the field's widest gap -- but a chord
    # through the gap is SHORTER (632 m against 980 m on the committed coordinates), so comparing
    # totals would credit the gap with what is only length. Measured 0.063 against 0.094
    # buildings/m. NOT zero -- see `_cases`' docstring.
    def per_metre(name: str) -> float:
        (road,) = cases[name]["roads"]
        (x0, y0), (x1, y1) = road["coords"]
        return float(cases[name]["sum_c"]) / math.hypot(x1 - x0, y1 - y0)
    assert cases["in_a_gap"]["sum_c"] > 0.0
    assert per_metre("in_a_gap") < per_metre("road1")


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
def test_the_bundle_still_matches_live_python(bundle: dict[str, Any]) -> None:
    """THE one block-loading test in this feature: the committed bundle has not gone stale against
    the code that made it.

    Job 1a covers every layer, not just the buildings, because it is the only guard that can see a
    single coordinate move; the fast precision guards cannot (see their own note). Job 1b re-prices
    every fixture from the bundle's own numbers.

    There is no second formula to hold against shapely here: the widget clips the same stored
    outlines against the polygon GEOS buffers each road to, and
    `web/test/displacement-model.test.ts` holds the TypeScript to Job 1b's own numbers.

    DEVELOPER-LOCAL, like every other block-loading test here: it needs ~/.cache/reblock's city
    data and the pinned block's footprint tile, and CI must stay hermetic (tests/conftest.py).
    """
    if not capetown_footprints_cached():
        pytest.skip("needs the capetown_full cache and the pinned block's footprint tile; run "
                    "`pixi run python -m scripts.gen_displacement_field`")

    import shapely
    from geopandas import GeoDataFrame

    from reblock.budget import displacement
    from reblock.buildings import ANCHOR_COL, Footprints
    from scripts._bundle_io import line_coords, polygon_rings, sigfig
    from scripts._default_road import default_roads
    from scripts._example_block import load_example_region
    from scripts.gen_displacement_field import (
        WIDTH_FLOOR_M,
        fixture_roads,
        outline_geometry,
        quantised_field,
        road_specs,
        roads_from_case,
    )

    # Needs the SHIPPED block's geometry and never touches a road, so it builds the region without
    # proposing: seconds instead of the ~2,900 a method run on the spine block costs. Not cacheable
    # between runs: `tests/conftest.py` gives every session a cold REBLOCK_CACHE_DIR by design.
    block = load_example_region()
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
    assert bundle["n_buildings"] == len(block.building_geometries)
    assert bundle["buildings"] == quantised_field(block, ox, oy).stored
    # `polygon_rings`, not `polygon_ring`: both fields carry EVERY ring since 2026-09-20. The
    # single-ring version raises on a hole rather than dropping one, which is how the spine block's
    # 1 holed parcel (of 6,619) and its holed BOUNDARY were found when the pin moved there.
    assert bundle["parcels"] == [ring for g in block.parcels.geometry
                                 for ring in polygon_rings(g, ox, oy, what="parcel")]
    assert bundle["boundary"] == polygon_rings(block.boundary, ox, oy, what="boundary")
    assert bundle["streets"] == [line for g in block.streets.geometry
                                 for line in line_coords(g, ox, oy)]
    default = default_roads(block, WIDTH_FLOOR_M)
    assert bundle["roads"] == road_specs(default, ox, oy)

    # The SEVENTH carrier, and the only one nothing else can see. `in_a_gap` and `outside` are
    # derived from the block's geometry (the widest nearest-neighbour gap; a translation by twice
    # the diagonal), so `bundle.roads` cannot pin them the way it pins the other four -- and Job 1b
    # cannot either, because it recomputes `sum_c` from these very coordinates: a vertex corrupted
    # where no building sits moves no overlap and leaves `sum_c` bit-identical. Re-derived through
    # the generator's own `fixture_roads`, which is why that was split out of `_cases`.
    derived = {name: road_specs(rs, ox, oy) for name, rs in fixture_roads(block, default)}
    assert [c["name"] for c in bundle["reference"]] == list(derived), (
        f"the committed fixtures are {[c['name'] for c in bundle['reference']]} but the generator "
        f"now produces {list(derived)}")
    for case in bundle["reference"]:
        assert case["roads"] == derived[case["name"]], (
            f"fixture {case['name']}'s road geometry is not what the generator now derives; "
            f"regenerate: pixi run python -m scripts.gen_displacement_field")

    # Job 1b: every fixture's `sum_c`, recomputed from the bundle's OWN outlines and roads. Exact
    # through the same quantiser, not a tolerance: `sum_c` IS `sigfig(displacement(...))` of the
    # quantised road against the quantised outlines (`gen_displacement_field.quantised_field`), so
    # no slack is needed and any tolerance would have to be looser than the thing it is guarding.
    stored = Footprints(GeoDataFrame(
        {ANCHOR_COL: shapely.points(block.buildings.xy)},
        geometry=[outline_geometry(o, ox, oy) for o in bundle["buildings"]], crs=block.crs))
    for case in bundle["reference"]:
        recomputed = displacement(stored, roads_from_case(block, case, (ox, oy)))
        assert sigfig(recomputed) == case["sum_c"], (
            f"{case['name']}: the committed bundle says {case['sum_c']}, the code now computes "
            f"{recomputed}; regenerate: pixi run python -m scripts.gen_displacement_field")


def _coordinates(b: dict[str, Any]) -> list[tuple[str, float]]:
    """Every coordinate the bundle carries, each labelled with where it came from.

    Enumerated by NAME, not discovered by walking every float: `sum_c` and `fraction` go through
    `sigfig`, not `cm`, so a blind walk would have to guess which rule applies and would fail on
    the fields where the answer is "the other one". The schema is
    closed and known while this is being written, which is exactly when a name beats a probe.
    """
    out: list[tuple[str, float]] = []
    out += [("buildings", v) for building in b["buildings"] for poly in building for ring in poly
            for xy in ring for v in xy]
    out += [("parcels", v) for ring in b["parcels"] for xy in ring for v in xy]
    # One level deeper than it used to be: `boundary` is a ring LIST now, so walking it as a
    # single ring hands `abs()` a coordinate pair and raises rather than checking anything.
    out += [("boundary", v) for ring in b["boundary"] for xy in ring for v in xy]
    out += [("streets", v) for line in b["streets"] for xy in line for v in xy]
    out += [("roads[].coords", v) for r in b["roads"] for xy in r["coords"] for v in xy]
    out += [(f"reference[{c['name']}].roads[].coords", v)
            for c in b["reference"] for r in c["roads"] for xy in r["coords"] for v in xy]
    return out


def _coordinate_shaped(node: Any, path: str = "") -> dict[str, list[float]]:
    """Every coordinate-SHAPED thing in the parsed artifact, discovered rather than named.

    The counterpart to `_coordinates` above, and deliberately the opposite discipline: that one
    enumerates by name because it has to know which quantiser each field went through; this one
    knows nothing about the schema and reports what the JSON's own shape says is geometry. The two
    are compared below, which is the only way the coverage question can be answered at all -- a
    guard whose expectation is a second copy of the list it checks cannot see a layer that is in
    NEITHER list, which is precisely the case it exists for.

    Two shapes count, and they are the only two the bundle's layers take:

    * a non-empty list whose every element is a list of exactly two numbers -- a ring, a polyline,
      or a road's `coords`;
    * a dict carrying `x` and `y` as equal-length numeric lists -- the struct-of-arrays form. No
      layer uses it today (`buildings` did, until it became outlines), and it stays because a
      future layer adding it would otherwise be invisible here. A sibling numeric list such as a
      radius is NOT matched: the rule keys on the pair of names, not on numeric-ness.

    A bare list of exactly two numbers counts too, so that a single point added as `[x, y]` cannot
    slip through as a third shape. `origin` is the one such value the bundle carries today, and the
    caller exempts it BY NAME with a reason -- it is absolute UTM, the one coordinate that must not
    be centimetre-quantised. A second bare pair therefore reddens instead of being absorbed.

    LIMIT, stated because it bounds what the caller can claim: a coordinate stored under a shape
    that is neither of these -- three interleaved arrays, a flat `[x0, y0, x1, y1, ...]`, a string
    of WKT -- is invisible here, and the bundle would have to grow one for that to matter.

    No FALSE POSITIVES on this artifact: every path it finds normalises to one of the seven
    carriers `_coordinates` names -- plus `origin`. Nothing else in the bundle is
    coordinate-shaped: `width` and `encoding` hold scalars, and so do `reference[].sum_c` and
    `fraction`.
    """
    def num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    found: dict[str, list[float]] = {}
    if isinstance(node, dict):
        rest = dict(node)
        x, y = rest.get("x"), rest.get("y")
        if (isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and len(x) > 0
                and all(num(v) for v in x) and all(num(v) for v in y)):
            found[f"{path}.x"] = [float(v) for v in x]
            found[f"{path}.y"] = [float(v) for v in y]
            del rest["x"], rest["y"]
        for key, value in rest.items():
            found |= _coordinate_shaped(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        if node and all(isinstance(v, list) and len(v) == 2 and all(num(c) for c in v)
                        for v in node):
            found[path] = [float(c) for xy in node for c in xy]
        elif len(node) == 2 and all(num(v) for v in node):
            found[path] = [float(v) for v in node]
        else:
            for i, value in enumerate(node):
                found |= _coordinate_shaped(value, f"{path}[{i}]")
    return found


def test_every_coordinate_carrier_is_covered_by_the_precision_guard(
        bundle: dict[str, Any]) -> None:
    """`_coordinates` is a hand-written enumeration, so it can go stale the moment the schema grows
    a layer -- and a guard that silently stops covering a field is worse than no guard.

    The expectation is derived from the ARTIFACT, not from a second copy of the enumeration. It
    used to be the latter: a hand-written sum, term for term over the same closed list
    `_coordinates` walks, which meant a carrier added to the bundle and to neither list passed --
    exactly the case this test's name promises to catch, and confirmed by fault injection that it
    did. What was actually being checked was that two adjacent hand-written lists had been edited
    together.

    Compared as a MULTISET of values rather than as a set of paths, because the two sides label
    differently on purpose: `_coordinates` groups by layer (`parcels`, one bucket for all 263
    rings) since that is the granularity its coarseness check needs, while the walk reports each
    ring at its own JSON path. Values are exact -- both sides read the same parsed floats -- so
    equality is available and any surplus is a carrier nobody quantises, checks, or draws.
    """
    found = _coordinate_shaped(bundle)
    # `origin` is the ONE coordinate `_coordinates` must not walk: it is the absolute UTM offset
    # every other coordinate is measured from, so it is neither origin-relative nor `cm`-quantised
    # and the precision guard would reject it correctly. Exempted here by name, with that reason,
    # rather than by making the shape rule blind to bare pairs -- a SECOND bare pair is a new
    # carrier and must redden.
    assert "origin" in found, (
        f"the bundle no longer carries `origin` as a bare coordinate pair (found "
        f"{sorted(found)}); the exemption below is now describing nothing")
    del found["origin"]

    carried = Counter(v for values in found.values() for v in values)
    walked = Counter(v for _, v in _coordinates(bundle))
    surplus, missing = carried - walked, walked - carried
    unguarded = sorted(p for p, values in found.items() if any(surplus[v] for v in values))
    assert carried == walked, (
        f"the bundle carries {sum(carried.values())} coordinate-shaped values but `_coordinates` "
        f"walks {sum(walked.values())}. Unwalked carriers: {unguarded[:6]}"
        + (f" (+{len(unguarded) - 6} more)" if len(unguarded) > 6 else "")
        + (f"; {sum(missing.values())} value(s) `_coordinates` names are not coordinate-shaped in "
           f"the artifact" if missing else "")
        + ". A coordinate carrier was added to the schema and not to `_coordinates`, so it is "
          "skipping the precision guard below.")


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
    PNG is RGBA and the corridor's alpha 0.25 and each building's alpha = c live in the alpha
    channel; `convert("RGB")` DROPS that channel rather than compositing it against white, so every
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
    for key in ("building_color", "road_color", "boundary_color", "parcel_color"):
        want = str(bundle["encoding"][key])
        rgb = np.array([int(want[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.int32)
        closest = int(np.abs(px - rgb).sum(axis=1).min())
        assert closest <= 2, (
            f"encoding.{key} = {want} appears nowhere in {PNG} (closest pixel is {closest} away "
            f"summed over three channels): the widget and its own fallback image are drawing "
            f"different colours")


def test_the_committed_readme_is_what_the_generator_writes() -> None:
    """Same guard as the `.d.ts` one above, for the same reason and against a documented specimen:
    `examples/nairobi/README.md` claims 89 blocks while every `meta.json` beside it says 43. This
    README's every fact is read out of the bundle, so a hand edit -- or a re-bake nobody
    regenerated the prose for -- fails here rather than shipping a wrong number in a directory
    nobody re-reads. Recomputed from the COMMITTED `field.json`, so it needs no bake to run."""
    from scripts.gen_displacement_field import readme_markdown
    readme = ROOT / "examples/displacement-field/README.md"
    # Loaded here rather than through the `bundle` fixture: the fixture is declared
    # `dict[str, Any]`, and `readme_markdown` takes the `FieldBundle` TypedDict, so the fixture
    # would need a `cast` that asserts exactly what this test is checking.
    loaded = json.loads(BUNDLE.read_text(encoding="utf-8"))
    assert readme.read_text(encoding="utf-8") == readme_markdown(loaded), (
        "examples/displacement-field/README.md is stale or hand-edited; regenerate it: "
        "pixi run python -m scripts.gen_displacement_field")
