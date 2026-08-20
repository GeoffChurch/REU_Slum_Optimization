"""Bake the Displacement page's field figure and its widget bundle.

One PNG (the widget's BOOT state -- road 1 alone at 7 m, since road 2 defaults off) plus a
self-contained JSON payload and its generated .d.ts.

Self-contained rather than reading examples/perm-graph/bundle.json, even though both widgets sit on
the same block and that bundle already holds the parcels, boundary, streets and origin: sharing one
payload couples two widgets so that retuning one silently changes the other. The generator CODE is
shared instead (scripts/_bundle_io.py, scripts/_default_road.py, scripts/_example_block.py), which
makes data drift impossible while leaving each widget its own artifact.

Run:  pixi run python -m scripts.gen_displacement_field
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import numpy as np
from geopandas import GeoDataFrame, points_from_xy
from numpy.typing import NDArray
from shapely.affinity import translate
from shapely.geometry import LineString

from reblock.budget import building_radii, displacement
from reblock.contracts import Block
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, PermeabilityParams
from reblock.render import (
    _BOUNDARY_COLOR,
    _BOUNDARY_LW,
    _CONTEXT_OUTLINE,
    _CORRIDOR_ALPHA,
    _DISK_OUTLINE_LW,
    _DISPLACED_PT,
    _PARCEL_LW,
    _ROAD_COLOR,
    render_field,
    save_render,
)
from scripts._bundle_io import cm, line_coords, polygon_ring, sigfig
from scripts._default_road import chord, default_roads
from scripts._example_block import PINNED_METHOD, load_example_block

log = logging.getLogger(__name__)

OUT = Path("examples/displacement-field")
DTS = Path("web/src/field.d.ts")

# The slider's floor is not "7.0" but the value permeability.py:205 RAISES below, and the default is
# the width a method emits when nothing else specifies one -- both read from their declarations
# rather than retyped. A road narrower than the floor is one the pipeline rejects, so a slider that
# could produce one would offer the reader a configuration the metric refuses to score.
#
# NOTE these are the CODE's defaults (`PermeabilityParams`' dataclass field and
# `permeability.DEFAULT_ROAD_WIDTH_M`), NOT conf/permeability.yaml:31 -- which happens to set the
# same 7.0. The dataclass is the right binding because it is what the validator at :205 compares
# against when no config overrides it, so the slider and the validator cannot disagree. Editing the
# yaml does NOT move the slider; re-basing the dataclass default does, and fails
# tests/test_displacement_field_bundle.py until the bundle is re-baked.
WIDTH_FLOOR_M = PermeabilityParams.min_road_width_m
WIDTH_DEFAULT_M = DEFAULT_ROAD_WIDTH_M
WIDTH_MAX_M = 20.0
WIDTH_STEP_M = 0.5


class Encoding(TypedDict):
    """What the widget draws with. Every value that the PNG also draws with is read from the
    `reblock.render` constant `render_field` itself uses, so the two cannot drift: a reader with JS
    off and a reader with JS on must see the same figure. The last two have no PNG equivalent --
    they are the web figure's own affordances, and `render_field`'s framing is not pinned here at
    all (see the widget's own sizing)."""
    parcel_color: str
    parcel_lw: float
    boundary_color: str
    boundary_lw: float
    street_lw: float
    road_color: str
    road_alpha: float
    disk_color: str
    disk_outline_lw: float
    handle_radius_px: float
    pad: float


# `street_lw` is `_BOUNDARY_LW`, not a separate number: `_draw_boundary_and_streets` draws the
# outline and the street network in ONE pair of calls at one width, so a widget drawing streets
# thinner than the PNG does is drawing a different figure. It is kept as its own key because the
# canvas draws the two layers separately and a future divergence should be expressible.
ENCODING: Encoding = Encoding(
    parcel_color=_CONTEXT_OUTLINE,
    parcel_lw=_PARCEL_LW,
    boundary_color=_BOUNDARY_COLOR,
    boundary_lw=_BOUNDARY_LW,
    street_lw=_BOUNDARY_LW,
    road_color=_ROAD_COLOR,
    road_alpha=_CORRIDOR_ALPHA,
    disk_color=_DISPLACED_PT,
    disk_outline_lw=_DISK_OUTLINE_LW,
    handle_radius_px=7.0,
    pad=0.04,
)


class RoadSpec(TypedDict):
    """One road as the bundle stores it: origin-relative centimetres, plus its own width."""
    coords: list[list[float]]
    width_m: float


class ReferenceCase(TypedDict):
    """One parity fixture: a road set, and Python's own answer for THAT road set."""
    name: str
    roads: list[RoadSpec]
    sum_c: float
    fraction: float


class Buildings(TypedDict):
    """Disk centres (origin-relative, `cm`) and radii (`sigfig`), in building order."""
    x: list[float]
    y: list[float]
    r: list[float]


class Width(TypedDict):
    """The slider's own range."""
    floor_m: float
    max_m: float
    step_m: float
    default_m: float


class FieldBundle(TypedDict):
    """The whole artifact. A TypedDict rather than a bare dict for the same reason `ReferenceCase`
    is one: it is a closed key set at a JSON boundary, and `web/src/field.d.ts` declares exactly
    these names -- tests/test_displacement_field_bundle.py pins the two together in both
    directions, so a rename here is a TypeScript error rather than a blank panel."""
    block_id: str
    n_buildings: int
    origin: list[float]
    buildings: Buildings
    parcels: list[list[list[float]]]
    boundary: list[list[float]]
    streets: list[list[list[float]]]
    roads: list[RoadSpec]
    width: Width
    encoding: Encoding
    reference: list[ReferenceCase]


def roads_from_case(block: Block, case: ReferenceCase,
                    origin: tuple[float, float]) -> GeoDataFrame:
    """Rebuild a reference fixture's road set from the bundle's own numbers.

    Exists so tests/test_displacement_field_bundle.py can recompute a fixture against live code
    without re-deriving how a case is defined -- the bundle is the single description of it.

    A fixture's coordinates are ORIGIN-RELATIVE, like every other coordinate in the bundle, so the
    origin has to come back in here.

    `ReferenceCase` is a `TypedDict`, not a frozen dataclass: this value arrives through
    `json.loads`, and the directives' own carve-out is exactly that -- dict-ness forced by an
    external interface. A dataclass here would mean every caller converting at the boundary for a
    checker benefit a TypedDict already gives.
    """
    return roads_from_specs(block, case["roads"], origin)


def roads_from_specs(block: Block, specs: list[RoadSpec],
                     origin: tuple[float, float]) -> GeoDataFrame:
    """The road set a list of bundle road entries describes, back in the block's own CRS.

    Split out of `roads_from_case` because the BAKE needs it too, and for the reason the whole
    quantise-then-measure order below exists: a fixture's `sum_c` has to be the cost of the road
    the bundle actually describes, not of the unrounded road it was derived from.
    """
    ox, oy = origin
    return GeoDataFrame(
        {"width_m": [float(r["width_m"]) for r in specs]},
        geometry=[LineString([(x + ox, y + oy) for x, y in r["coords"]]) for r in specs],
        crs=block.crs)


def road_specs(roads: GeoDataFrame, ox: float, oy: float) -> list[RoadSpec]:
    """A road set as the bundle stores it.

    Raises on multipart geometry: the format gives a road ONE coordinate list, so a
    MultiLineString has no representation here and would have to be silently split or dropped.
    Reachable -- `Block.streets` and several methods' road sets are routinely multipart; the two
    default roads are not.
    """
    out: list[RoadSpec] = []
    for geom, width_m in zip(roads.geometry, roads["width_m"], strict=True):
        if not isinstance(geom, LineString):
            raise ValueError(
                f"a road is a {geom.geom_type}, not a LineString -- the bundle gives each road a "
                f"single coordinate list, so multipart geometry has no representation in it")
        out.append(RoadSpec(coords=[[cm(x - ox), cm(y - oy)] for x, y in geom.coords],
                            width_m=float(width_m)))
    return out


class QuantisedField(NamedTuple):
    """The building field exactly as the bundle carries it, in the two forms the bake needs.

    `stored` is what gets written -- origin-relative, `cm` for the centres and `sigfig` for the
    radii. `points`/`radii` are those SAME numbers put back in the block's CRS, for measuring
    fixtures against. One derivation, two shapes: `main` must not re-quantise the centres for the
    payload, or the file and the measurement stop being guaranteed to describe each other.
    """
    stored: Buildings
    points: GeoDataFrame
    radii: NDArray[np.float64]


def quantised_field(block: Block, radii: NDArray[np.float64], ox: float,
                    oy: float) -> QuantisedField:
    """The building field as the bundle carries it -- see `QuantisedField`.

    Every fixture is measured against THIS field rather than the raw one, and that is not a
    rounding nicety. The browser reads `buildings.x/y/r` and a fixture's `coords`, and the parity
    test compares its answer to `sum_c` at 1e-3 relative -- a tolerance justified by shapely's
    inscribed-buffer residual (4.4e-04) and nothing else. Centimetre rounding moves a road by up
    to 7 mm, which moves `c` for a grazed building by 7mm/r ~ 3e-03 and the sum by ~1e-02
    absolute, ~2e-04 relative: the same order as the residual the tolerance exists for. Measuring
    the raw geometry would spend the parity budget on the quantiser and leave the formula
    unchecked.
    """
    pts = block.building_points
    stored = Buildings(x=[cm(v - ox) for v in pts.geometry.x],
                       y=[cm(v - oy) for v in pts.geometry.y],
                       r=[sigfig(v) for v in radii])
    bx = np.asarray(stored["x"], dtype=np.float64) + ox
    by = np.asarray(stored["y"], dtype=np.float64) + oy
    return QuantisedField(stored=stored,
                          points=GeoDataFrame(geometry=points_from_xy(bx, by), crs=block.crs),
                          radii=np.asarray(stored["r"], dtype=np.float64))


def _set(block: Block, geoms: list[LineString], width_m: float) -> GeoDataFrame:
    """A road set at one width. Every fixture is built through here so no fixture can accidentally
    carry a width the slider could not produce."""
    if not (WIDTH_FLOOR_M <= width_m <= WIDTH_MAX_M):
        raise ValueError(f"{width_m} m is outside the slider's own range")
    return GeoDataFrame({"width_m": [float(width_m)] * len(geoms)},
                        geometry=list(geoms), crs=block.crs)


def _cases(block: Block, pts: GeoDataFrame, radii: NDArray[np.float64], roads: GeoDataFrame,
           ox: float, oy: float) -> list[ReferenceCase]:
    """The six parity fixtures. Chosen so no two of them could pass for the same reason:

    road1      -- road 1 alone at the floor width. The BASELINE, and the PNG's own number.
    apart      -- both default roads, corridors disjoint
    coincident -- road 2 moved onto road 1: costs EXACTLY what road1 costs
    widest     -- road 1 alone at WIDTH_MAX_M: isolates width
    in_a_gap   -- road 1's direction through the field's widest gap: isolates position
    outside    -- road 1 translated clear of the block: exactly zero

    Each isolates ONE of the page's claims against `road1`, which is why the baseline is in the set:
    without it, "in_a_gap is cheaper than apart" compares one road against two and demonstrates
    nothing about gaps.

    Measured on the pinned block: road1 32.0260, apart 47.8436, coincident 32.0260, widest 68.1452,
    in_a_gap 21.8509, outside 0.0. On the RAW, unquantised geometry the same six are 32.0260,
    47.8488, 32.0260, 68.1581, 21.8465, 0.0 -- the 1.1e-04 to 2.0e-04 relative gap between the two
    columns is the quantiser, and measuring the quantised side is the point (see `quantised_field`).

    `in_a_gap` is NOT free, and do not chase a zero: the widest gap here has radius 6.95 m against a
    2.19 m median, so a 7 m road down its middle still leaves its two neighbours at d = 3.45 m
    against r = 6.95, i.e. c ~ 0.5 each. A chord across a block cannot stay outside every disk.
    `outside` is the fixture that pins the clip at d = r.

    `pts`/`radii` are the QUANTISED field (see `quantised_field`), and every road set is round-
    tripped through `road_specs` before it is measured, so each `sum_c` is the cost of the road the
    bundle describes rather than of the unrounded road it came from.
    """
    out: list[ReferenceCase] = []
    for name, rs in fixture_roads(block, pts, radii, roads):
        specs = road_specs(rs, ox, oy)
        total = displacement(pts, radii, roads_from_specs(block, specs, (ox, oy)))
        out.append(ReferenceCase(name=name, roads=specs, sum_c=sigfig(total),
                                 fraction=sigfig(total / len(pts))))

    # The one fixture with an EXACT expectation, checked on the value that will actually be
    # committed rather than on the float behind it. A fixture that is merely nearly outside the
    # block proves nothing about the clip at d = r, so this fails the bake instead of shipping.
    outside = next(c for c in out if c["name"] == "outside")
    if outside["sum_c"] != 0.0:
        raise AssertionError(
            f"the 'outside' fixture displaces {outside['sum_c']}, so it is not outside the block "
            f"-- it would prove nothing about the clip at d = r. Fix the translation, do not bake "
            f"it.")
    return out


def fixture_roads(block: Block, pts: GeoDataFrame, radii: NDArray[np.float64],
                  roads: GeoDataFrame) -> list[tuple[str, GeoDataFrame]]:
    """The six fixtures' road sets, by name, in the block's own CRS and before any quantisation.

    Split out of `_cases` so the DERIVATION is callable on its own. Four of the six are functions
    of `bundle.roads` alone and are pinned in a fast test with no block load; the other two --
    `in_a_gap` and `outside` -- are functions of the block's geometry, so the only honest check is
    to re-derive them from the live block, which the one slow test now does. Before this existed,
    a corrupted vertex in a fixture road was inert: if no building sits near it, no grazing distance
    moves, `sum_c` stays bit-identical, and every test passed.

    The four derivable ones are stated here so the invariant is readable in one place -- each moves
    exactly ONE variable against `road1`:

        road1      == roads[0]                       at the floor width   (the baseline)
        apart      == [roads[0], roads[1]]           at the floor width   (a second road)
        coincident == [roads[0], roads[0]]           at the floor width   (overlap only)
        widest     == roads[0]                       at WIDTH_MAX_M       (width only)
    """
    bp = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    r1 = cast(LineString, roads.geometry.iloc[0])
    r2 = cast(LineString, roads.geometry.iloc[1])
    hull = block.parcels.union_all()
    diag = float(np.hypot(*(np.asarray(block.parcels.total_bounds[2:])
                            - np.asarray(block.parcels.total_bounds[:2]))))

    # The widest gap in the field IS the largest nearest-neighbour distance: that is what a large
    # radius means. Put the road through the midpoint of that pair, along road 1's direction.
    widest = int(np.argmax(radii))
    others = np.delete(np.arange(len(bp)), widest)
    partner = int(others[np.argmin(np.hypot(*(bp[others] - bp[widest]).T))])
    gap_mid = (bp[widest] + bp[partner]) / 2.0
    r1_dir = np.asarray(r1.coords[-1], dtype=np.float64) - np.asarray(r1.coords[0],
                                                                     dtype=np.float64)
    r1_dir = r1_dir / np.hypot(*r1_dir)
    # `chord` rather than a second intersect-and-take-the-longest-piece here: a concave block cuts
    # the line into several parts and only the longest is a road, which is exactly the rule
    # `default_roads` already had to state.
    in_gap = chord(hull, gap_mid, r1_dir)
    # The normal, so the translation clears the block along its own shortest axis; 2x the diagonal
    # is more than the block's extent in any direction.
    away = np.array([-r1_dir[1], r1_dir[0]]) * (2.0 * diag)

    named: list[tuple[str, GeoDataFrame]] = [
        ("road1", _set(block, [r1], WIDTH_FLOOR_M)),
        ("apart", _set(block, [r1, r2], WIDTH_FLOOR_M)),
        ("coincident", _set(block, [r1, r1], WIDTH_FLOOR_M)),
        ("widest", _set(block, [r1], WIDTH_MAX_M)),
        ("in_a_gap", _set(block, [in_gap], WIDTH_FLOOR_M)),
        ("outside", _set(block, [translate(r1, xoff=away[0], yoff=away[1])], WIDTH_FLOOR_M)),
    ]
    return named


DTS_TEMPLATE = '''// GENERATED by scripts/gen_displacement_field.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_displacement_field
// This file is what makes a renamed Python field a TypeScript error instead of a blank panel.
export interface Encoding {
  parcel_color: string;
  parcel_lw: number;
  boundary_color: string;
  boundary_lw: number;
  street_lw: number;
  road_color: string;
  road_alpha: number;
  disk_color: string;
  disk_outline_lw: number;
  handle_radius_px: number;
  pad: number;
}
export interface Width {
  /** The slider's floor IS PermeabilityParams.min_road_width_m: the pipeline raises below it. */
  floor_m: number;
  max_m: number;
  step_m: number;
  default_m: number;
}
export interface Road {
  coords: [number, number][];
  width_m: number;
}
/** A road configuration plus Python's own sum of c_i for it -- the parity fixtures. `sum_c` is
 * measured on THESE coordinates and on `buildings` as quantised below, not on the unrounded
 * geometry they came from, so the parity tolerance measures the formula and not the quantiser. */
export interface ReferenceCase {
  name: string;
  roads: Road[];
  sum_c: number;
  fraction: number;
}
export interface FieldBundle {
  block_id: string;
  n_buildings: number;
  /** UTM easting/northing subtracted from every coordinate below; all geometry is local metres. */
  origin: [number, number];
  /** Disk centres (relative to `origin`) and radii, in metres, in building order. */
  buildings: { x: number[]; y: number[]; r: number[] };
  parcels: [number, number][][];
  /** Block exterior ring, relative to `origin` -- the ring the fallback PNG draws. */
  boundary: [number, number][];
  /** Existing street network, relative to `origin`; one entry per disjoint line (a block's
   * streets are not always a single connected LineString). Fallback-parity, same as `boundary`. */
  streets: [number, number][][];
  /** Road 1 and road 2 (the default-road rule). Road 2 boots OFF; the toggle adds it. */
  roads: Road[];
  width: Width;
  encoding: Encoding;
  reference: ReferenceCase[];
}
'''


def readme_markdown(bundle: FieldBundle) -> str:
    """This directory's README, written from the bundle it documents.

    GENERATED, not handwritten, for one reason: every fact worth putting in it -- the block, the
    building count, the slider's three widths, and the six fixtures' costs -- is already in
    `field.json`, and a handwritten copy of a number is a copy that rots. This repo has the
    specimen: `examples/nairobi/README.md` claims 89 blocks while every `meta.json` beside it says
    43. `tests/test_displacement_field_bundle.py` pins the committed file to this function's
    output, the same way it pins `web/src/field.d.ts` to `DTS_TEMPLATE`, so the two cannot drift
    even if nobody re-bakes.

    What each fixture ISOLATES is deliberately NOT restated here -- `_cases`'s docstring is the one
    place that reasoning lives, and a second copy of it in prose would be the very drift this
    function exists to prevent, one abstraction level up.
    """
    w = bundle["width"]

    def row(c: ReferenceCase) -> str:
        widths = ", ".join(f"{r['width_m']:g} m" for r in c["roads"])
        return (f"| `{c['name']}` | {len(c['roads'])} | {widths} | "
                f"{c['sum_c']:g} | {c['fraction']:.1%} |")

    rows = "\n".join(row(c) for c in bundle["reference"])
    return f"""<!-- GENERATED by scripts/gen_displacement_field.py -- do not edit. Regenerate:
     pixi run python -m scripts.gen_displacement_field -->

# The displacement field

The figure set for the site's [Displacement](../../docs/_partials/displacement.md) section: the
model drawn literally — every building a disk of its own radius `rᵢ`, the road corridor beneath
them, each disk shaded by the share `cᵢ` of it the corridor takes.

![the displacement model on the pinned block](field.png)

`field.png` is the **boot state** of the site's interactive `DisplacementField` figure, and the
picture a reader with JavaScript off is left with: road 1 alone, at the floor width of
{w['floor_m']:g} m. `field.json` is the payload that figure fetches — the same block's buildings,
parcels, boundary, streets and two default roads, plus the drawing encoding, so the widget and the
PNG above are one picture and not two. The bake also writes `web/src/field.d.ts`, which is what
makes a renamed field a TypeScript error rather than a blank panel.

**Provenance.** Block `{bundle['block_id']}`, {bundle['n_buildings']:,} buildings — the block
`conf/example/method_comparison.yaml` pins, so this is the same block every method page's
before/after uses, and the same one [`../perm-graph/`](../perm-graph/) draws. The two roads are
derived from the block by rule (`default_roads`), not taken from any method's output. Corridor
width is the reader's to change, in {w['step_m']:g} m steps: floor {w['floor_m']:g} m, default
{w['default_m']:g} m, maximum {w['max_m']:g} m.

**Reference cases.** `field.json` carries {len(bundle['reference'])} road configurations, each with
the displacement Python computes for it. They are what `web/test/displacement-model.test.ts` holds
the TypeScript model to, what `tests/test_displacement_field_bundle.py` re-derives against live
Python, and where the site caption's numbers come from. Each moves exactly one variable against the
`road1` baseline — `_cases` in `scripts/gen_displacement_field.py` says which, and why no two of
them could pass for the same reason.

| case | roads | width | Σcᵢ | of {bundle['n_buildings']:,} buildings |
|---|---|---|---|---|
{rows}

Not one of the flagships in [`../README.md`](../README.md): those are walkthroughs that reproduce a
result from the CLI, and this is a figure set for one page.

Regenerate: `pixi run python -m scripts.gen_displacement_field`
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    # The roads are DISCARDED: this bake derives its own two by rule. `PINNED_METHOD` is passed
    # only because the loader has no block-only mode, and solving one method is minutes cheaper
    # than solving all eight for a block we then use on its own.
    block, _ = load_example_block(PINNED_METHOD)
    radii = building_radii(block.building_points)
    roads = default_roads(block, WIDTH_FLOOR_M)

    # Everything geometric is emitted RELATIVE to this, in metres. PARCEL bounds, not the building
    # points' or the graph nodes' -- the parcels are the outermost thing any of these figures draws,
    # so a canvas fitted to them cannot clip a vertex.
    ox = float(block.parcels.total_bounds[0])
    oy = float(block.parcels.total_bounds[1])

    # The BOOT state, and road 1 only: road 2 defaults off, so the fallback image has to show what
    # the widget shows before the reader touches anything.
    save_render(render_field(block, cast(GeoDataFrame, roads.iloc[[0]]), radii), OUT / "field.png")
    log.info("wrote %s", OUT / "field.png")

    field = quantised_field(block, radii, ox, oy)
    cases = _cases(block, field.points, field.radii, roads, ox, oy)

    # No `is not None` guard: `Block.streets` is a required, non-Optional field whose geometry
    # column `__post_init__` checks, so an empty frame is the only reachable "no streets" case and
    # it yields an empty list on its own.
    street_coords: list[list[list[float]]] = []
    for g in block.streets.geometry:
        street_coords.extend(line_coords(g, ox, oy))

    bundle = FieldBundle(
        block_id=block.block_id,
        n_buildings=len(block.building_points),
        origin=[ox, oy],
        buildings=field.stored,
        parcels=[polygon_ring(g, ox, oy, what=f"block {block.block_id!r}'s parcel")
                 for g in block.parcels.geometry],
        # `block.boundary`, not the parcel union: this layer exists for fallback parity, and
        # `_draw_boundary_and_streets` (render.py) is what the committed PNG draws it with.
        boundary=polygon_ring(block.boundary, ox, oy,
                              what=f"block {block.block_id!r}'s boundary"),
        streets=street_coords,
        roads=road_specs(roads, ox, oy),
        width=Width(floor_m=WIDTH_FLOOR_M, max_m=WIDTH_MAX_M, step_m=WIDTH_STEP_M,
                    default_m=WIDTH_DEFAULT_M),
        encoding=ENCODING,
        reference=cases,
    )
    (OUT / "field.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT / "field.json",
             (OUT / "field.json").stat().st_size / 1024.0)

    DTS.parent.mkdir(parents=True, exist_ok=True)
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")
    log.info("wrote %s", DTS)

    (OUT / "README.md").write_text(readme_markdown(bundle), encoding="utf-8")
    log.info("wrote %s", OUT / "README.md")

    for c in cases:
        log.info("%-11s sum_c=%-10s fraction=%s  (%d road%s)", c["name"], c["sum_c"],
                 c["fraction"], len(c["roads"]), "" if len(c["roads"]) == 1 else "s")


if __name__ == "__main__":
    main()
