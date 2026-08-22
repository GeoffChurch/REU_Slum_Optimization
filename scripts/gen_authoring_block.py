"""Bake examples/authoring/ -- the block the draw-your-own-road widget will rebuild in the browser.

DEBT: every consumer named in this file is later in this piece and absent at this commit -- Task 2's
`web/src/py/solve.py`, Task 4's `web/src/widgets/draw-road.ts`, Task 5's
`web/test/pyodide-parity.test.ts`. They are named because they are what the bundle's shape is FOR,
and written in the future tense because none of them exists yet.

The widget will run `reblock.permeability` itself, under Pyodide, on a road the reader drew. So this
bundle is not a picture: it is the INPUT to a solve, and the browser rebuilds a real `Block` from
it before calling the real solver. What it carries beyond the geometry is the road-INVARIANT half
of the egress graph (design §1.6) -- parcel centroids, the footpath edge list, which parcels front
the street, and the no-roads baseline -- because none of that moves when a road is drawn, so
recomputing it on every edit would be waste.

Coordinates are FULL float64 and absolute, which is what separates this bundle from every other
one on the site. `scripts/_bundle_io.py`'s `cm`/`sigfig`/`polygon_rings` quantise by design -- they
serve bundles that get drawn -- and here quantisation is not a rounding error, it is the answer
changing. MEASURED on this bundle's own reference roads: rounding every coordinate to centimetres
and re-solving (baseline included) moves `crossing` by 1.6611e-03 and `spur` by 1.9550e-03. Design
§1.4's 4.71e-05 is the same effect measured on the clearance method's road set, so it is the
smaller of the two figures and not the one to size a runtime-parity tolerance against. Task 5's
parity test will compare the browser's answer against `reference` below to decide whether the WASM
runtime agrees with CPython; a tolerance wide enough to absorb either figure would no longer be
measuring that.

The block is ZAF.9.3.1_1_40972, the same one PermGraph, Frontier, DisplacementField and RegionGrow
pin and ScreenMap follows (`gen_screen_map.py` derives its `follow` from perm-graph's bundle rather
than naming a block).

Outputs, into examples/authoring/:

    block.json   the bundle (schema: web/src/authoring.d.ts, generated here)
    README.md    generated

Reproduce with `pixi run python -m scripts.gen_authoring_block`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import geopandas as gpd
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from reblock.compare import load_permeability_config
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.mesh import parcel_radii
from reblock.permeability import WIDTH_COL, PermeabilityParams, solve_egress
from scripts._example_block import PINNED_METHOD, load_example_block

log = logging.getLogger(__name__)

OUT = Path("examples/authoring")
DTS = Path("web/src/authoring.d.ts")

# Fixed roads whose CPython answers are baked for the parity test. Two, deliberately: one crossing
# the block and one short spur, so a runtime that agreed on a trivial case and not a real one
# cannot pass. Both are literals rather than derived from the block, so a re-bake that moved the
# geometry would change the ANSWER for a fixed road instead of quietly moving the road too.
# Coordinates are absolute EPSG:32734 metres, the same frame the bundle's geometry is in.
REFERENCE_ROADS: dict[str, list[tuple[float, float]]] = {
    "crossing": [(276425.0, 6237640.0), (276495.0, 6237645.0), (276565.0, 6237695.0)],
    "spur": [(276497.0, 6237600.0), (276497.0, 6237630.0)],
}


class ReferenceCase(TypedDict):
    """One fixed road and `solve_egress`' own `1 - p/p0` for it -- `web/src/authoring.d.ts`'s
    `AuthoringReference`."""
    name: str
    road: list[list[float]]
    permeability: float


class NodesDict(TypedDict):
    """Per-parcel node columns, in parcel order -- `web/src/authoring.d.ts`'s `nodes`."""
    cx: list[float]
    cy: list[float]
    ground: list[bool]


class EdgesDict(TypedDict):
    """Per-edge columns, in `Mesh` order -- `web/src/authoring.d.ts`'s `edges`."""
    rows: list[int]
    cols: list[int]
    footpath_g: list[float]


class BaselineDict(TypedDict):
    """`solve_egress(block, None, params)`'s own answer -- `web/src/authoring.d.ts`'s `baseline`."""
    p0: float
    potential: list[float]


class AuthoringBundle(TypedDict):
    """The whole artifact -- `web/src/authoring.d.ts`'s `AuthoringBlock`, generated from this
    file."""
    block_id: str
    crs_epsg: int
    boundary: list[list[list[float]]]
    parcel_id: list[str]
    parcels: list[list[list[list[float]]]]
    streets: list[list[list[float]]]
    building_points: list[list[float]]
    nodes: NodesDict
    edges: EdgesDict
    baseline: BaselineDict
    reference: list[ReferenceCase]


DTS_TEMPLATE = """// GENERATED by scripts/gen_authoring_block.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_authoring_block
// This file is what makes a renamed Python field a TypeScript error instead of a blank panel.
/** One fixed road and CPython's own permeability for it. DEBT: baked so that Task 5's
 * `web/test/pyodide-parity.test.ts` will need no Python process at test time -- the same device
 * `field.json` and `hood.json` already use. That test does not exist yet. */
export interface AuthoringReference {
  name: string;
  road: [number, number][];
  permeability: number;
}
export interface AuthoringBlock {
  block_id: string;
  /** Projected, and asserted so by the baker: `Block.__post_init__` rejects a geographic CRS,
   * and a browser-side failure there would surface as a Pyodide traceback in a figcaption. */
  crs_epsg: number;
  /** FULL float64, NOT the cm-rounded form the render bundles ship. MEASURED: rounding this
   * bundle's geometry to centimetres and re-solving moves `crossing` by 1.6611e-03 and `spur` by
   * 1.9550e-03; design §1.4's 4.71e-05 is the same effect on the clearance method's road set, and
   * is the smaller figure. Either is more than a runtime-parity guard can absorb.
   * Exterior ring first, then interiors -- the same shape every `parcels` entry has. */
  boundary: [number, number][][];
  parcel_id: string[];
  parcels: [number, number][][][];
  streets: [number, number][][];
  /** One per parcel, but NOT in parcel order -- `parcel_radii` resolves the correspondence by
   * containment, and this bundle preserves whatever order the source had rather than inventing
   * one the Python does not rely on. */
  building_points: [number, number][];
  reference: AuthoringReference[];

  /** The road-INVARIANT half of the graph (design §1.6), baked once so the runtime returns two
   * arrays rather than a mesh. Parcel centroids and the footpath edge list do not move when a
   * road is drawn; only the conductances on those edges do. */
  nodes: { cx: number[]; cy: number[]; ground: boolean[] };
  edges: { rows: number[]; cols: number[]; footpath_g: number[] };
  /** `solve_egress(block, None)`'s own answer: the no-roads baseline permeability divides
   * against, and the potentials the "before" picture shows. Road-invariant, so computing it in
   * the browser on every edit would be waste. */
  baseline: { p0: number; potential: number[] };
}
"""


def _rings(geom: BaseGeometry, *, what: str) -> list[list[list[float]]]:
    """Every ring of a simple Polygon -- exterior first, then interiors -- at FULL float64.

    The unrounded twin of `_bundle_io.polygon_rings`, and separate from it rather than a precision
    flag on it: that function's whole reason to exist is that the drawn bundles all quantise the
    same way, and a caller-chosen precision would put the trap it was extracted to remove back
    inside it. Nothing else is shared either -- there is no origin subtraction here (design §2:
    the origin trick shortens cm-rounded strings, and at full precision it only adds a subtraction
    the browser would have to undo before handing the numbers to shapely).

    isinstance, not geom_type, so this line IS the runtime guard mypy can also verify: it narrows
    `geom` to Polygon, which is what makes `.exterior`/`.interiors` type-check instead of resolving
    through BaseGeometry, the union GeoSeries iteration yields. `Block.boundary` is typed
    `Polygon | MultiPolygon`, so the MultiPolygon case is reachable by type; it raises because a
    bundle that gave one polygon one ring list would have to drop geometry to fit it.
    """
    if not isinstance(geom, Polygon):
        raise ValueError(
            f"{what} is a {geom.geom_type}, not a Polygon -- this bundle gives one polygon one "
            f"list of rings; report this instead of silently dropping geometry")
    return [[[x, y] for x, y in ring.coords] for ring in [geom.exterior, *geom.interiors]]


def _lines(geom: BaseGeometry, *, what: str) -> list[list[list[float]]]:
    """A street's coordinates, one list per component, at FULL float64 -- the unrounded twin of
    `_bundle_io.line_coords` (see `_rings` for why it is a separate function rather than a flag).

    A block's streets are not always one connected LineString, so a MultiLineString explodes into
    several entries; `footpath_mesh` unions the street geometry before measuring ground membership,
    which is the same union either way.
    """
    if isinstance(geom, LineString):
        lines: list[LineString] = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        raise ValueError(
            f"{what} is a {geom.geom_type}, not line geometry -- report this instead of silently "
            f"dropping it")
    return [[[x, y] for x, y in line.coords] for line in lines]


def _road_frame(road: list[list[float]], params: PermeabilityParams, crs: CRS) -> GeoDataFrame:
    """One drawn road in the shape `solve_egress` takes: a geometry column plus the mandatory
    `width_m`, which `buildable_widths` refuses a road set without. `min_road_width_m` is the
    narrowest road the metric will score at all, so it is what a reader's freehand line gets --
    the widget offers no width control (design §7: everything but the road is fixed input)."""
    line = LineString([(x, y) for x, y in road])
    return gpd.GeoDataFrame({"geometry": [line], WIDTH_COL: [params.min_road_width_m]},
                            geometry="geometry", crs=crs)


def _block_from_bundle(bundle: AuthoringBundle) -> Block:
    """Rebuild a Block from the JSON about to be written, to prove the JSON is sufficient.

    Task 2 EXTRACTS this into `web/src/py/solve.py` and this module imports it back, so the
    browser and the baker share one reconstruction. Written here first on purpose: the baker
    cannot depend on a module that is tested against the bundle the baker has not written yet.
    """
    crs = CRS.from_epsg(bundle["crs_epsg"])
    boundary_rings = bundle["boundary"]
    parcels = gpd.GeoDataFrame(
        {"parcel_id": bundle["parcel_id"],
         "geometry": [Polygon(rings[0], rings[1:]) for rings in bundle["parcels"]]},
        geometry="geometry", crs=crs)
    streets = gpd.GeoDataFrame(
        {"geometry": [LineString(coords) for coords in bundle["streets"]]},
        geometry="geometry", crs=crs)
    points = gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in bundle["building_points"]]},
        geometry="geometry", crs=crs)
    return Block(block_id=bundle["block_id"], crs=crs,
                 boundary=Polygon(boundary_rings[0], boundary_rings[1:]),
                 parcels=parcels, streets=streets, building_points=points)


def _short(value: object) -> str:
    """`repr`, truncated -- a 263-entry potential column makes a useless error message."""
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _assert_round_trip(bundle: AuthoringBundle, params: PermeabilityParams) -> None:
    """A bundle that cannot reproduce its own reference answers is not shippable, so this is a
    bake-time exit rather than a test -- there is nothing to commit if it fails.

    Checks BOTH halves of what the browser will do with this JSON. The baked columns
    (`nodes`, `edges`, `baseline`) are compared against what a block rebuilt from the JSON produces,
    because nothing else ever reads them back -- the solve recomputes the mesh rather than
    consuming the baked one, so a wrong `footpath_g` column would survive a reference-road check
    untouched. The reference roads then check the solve itself.

    Exact equality, not a tolerance: the reconstruction is fed the same float64 coordinates the
    source block carried, so the two solves are the same arithmetic on the same bits. A difference
    of any size means the reconstruction did not get everything the solve reads out of the JSON --
    which is what this distinguishes, and no more: it cannot say WHICH of a lost coordinate, a
    changed ring winding or a dtype shift caused it.

    Deliberately NOT threaded with the source block's `adj`/`radii` the way `main` threads them:
    those are derived from `block.parcels.geometry` and `block.building_points`, so handing them
    over would skip re-deriving exactly the things the JSON has to carry.
    """
    rebuilt = _block_from_bundle(bundle)
    rebuilt_baseline = solve_egress(rebuilt, None, params)
    mesh = rebuilt_baseline.mesh
    columns: list[tuple[str, object, object]] = [
        ("nodes.cx", bundle["nodes"]["cx"], mesh.cx.tolist()),
        ("nodes.cy", bundle["nodes"]["cy"], mesh.cy.tolist()),
        ("nodes.ground", bundle["nodes"]["ground"], [bool(g) for g in mesh.ground]),
        ("edges.rows", bundle["edges"]["rows"], mesh.rows.tolist()),
        ("edges.cols", bundle["edges"]["cols"], mesh.cols.tolist()),
        ("edges.footpath_g", bundle["edges"]["footpath_g"], mesh.footpath_g.tolist()),
        ("baseline.p0", bundle["baseline"]["p0"], rebuilt_baseline.p),
        ("baseline.potential", bundle["baseline"]["potential"],
         rebuilt_baseline.potential.tolist()),
    ]
    for label, baked, got in columns:
        if baked != got:
            raise SystemExit(
                f"round trip failed for {label}: the block rebuilt from this JSON does not "
                f"reproduce the baked column -- {_short(baked)} != {_short(got)}")

    p0 = bundle["baseline"]["p0"]
    for case in bundle["reference"]:
        sol = solve_egress(rebuilt, _road_frame(case["road"], params, rebuilt.crs), params)
        got_p = 1.0 - sol.p / p0
        if got_p != case["permeability"]:
            raise SystemExit(
                f"round trip failed for {case['name']}: {got_p!r} != {case['permeability']!r}")


def readme_markdown(bundle: AuthoringBundle) -> str:
    """This directory's README, written from the bundle it documents -- see
    `gen_displacement_field.readme_markdown`'s docstring for why generated beats handwritten:
    every fact worth stating here is already in `block.json`, and a handwritten copy of a number
    is a copy that rots."""
    rows = "\n".join(
        f"| `{c['name']}` | {len(c['road'])} | {c['permeability']:.10f} |"
        for c in bundle["reference"])
    n_streets = len(bundle["streets"])
    counts = (f"{len(bundle['parcels']):,} parcels, {n_streets:,} street "
              f"line{'' if n_streets == 1 else 's'} and "
              f"{len(bundle['building_points']):,} building points")
    graph = (f"{len(bundle['nodes']['cx']):,} parcel centroids, "
             f"{sum(bundle['nodes']['ground']):,} of them street-fronting")
    return f"""<!-- GENERATED by scripts/gen_authoring_block.py -- do not edit. Regenerate:
     pixi run python -m scripts.gen_authoring_block -->

# The authoring block

The block the site's DrawRoad widget will rebuild in the browser. It carries no picture: the widget
boots Pyodide, installs the `reblock` wheel, rebuilds a real `Block` from `block.json` and calls
`reblock.permeability`'s own solver on whatever road the reader drew. Nothing here is a
re-implementation of the metric, which is the entire reason this stage boots a Python runtime.

*The widget, the runtime seam and the parity test are later tasks of this piece and are not on
disk yet; this bundle is baked first because they are all written against it.*

**The block** is `{bundle['block_id']}` in EPSG:{bundle['crs_epsg']} -- the same block every other
stage of the site follows. It carries {counts}.

**Full float64, absolute coordinates.** Every other committed bundle here ships cm-rounded,
origin-relative metres, which is right for drawing and wrong for solving. Measured on the reference
roads below: rounding this block's geometry to centimetres and re-solving moves `crossing` by
1.6611e-03 and `spur` by 1.9550e-03. (Design §1.4's 4.71e-05 is the same effect measured on the
clearance method's road set -- a different, smaller number for a different road set.) The parity
test will compare the browser's answer against the baked CPython answers below to decide whether
the WASM runtime agrees with CPython, so the two sides have to be reading the same numbers.

**The baked graph.** A drawn road changes per-edge conductance and per-parcel potential and nothing
else -- the mesh takes no roads and grounding comes from the street -- so the road-invariant half is
baked once rather than recomputed on every edit: {graph},
{len(bundle['edges']['rows']):,} footpath edges, and a no-roads baseline of
p0 = {bundle['baseline']['p0']:.6f}.

**Reference roads.** Fixed polylines with `solve_egress`' own `1 - p/p0` for each. Before writing
anything the baker rebuilds the block from the JSON it is about to write, checks every baked column
(`nodes`, `edges`, `baseline`) against what that reconstruction produces, and re-solves these
roads -- so a bundle that has lost precision never reaches the disk.

| road | vertices | permeability |
|---|---|---|
{rows}

Regenerate: `pixi run python -m scripts.gen_authoring_block`
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)

    # The dataclass defaults, NOT `load_permeability_config()`: the wheel micropip installs in the
    # browser is built from `src/reblock` alone (pyproject's hatch `packages`), so `conf/` does not
    # travel with it and the browser's `PermeabilityParams()` is the default set. Baking against
    # the configured set instead would make the parity test compare two different metrics. The
    # check below turns a yaml edit into a bake failure rather than leaving this widget silently
    # scoring roads by different parameters from every other figure on the site; note that
    # `load_permeability_config` reads five of the six fields (it does not read
    # `min_road_width_m`), so that one field compares default against default.
    params = PermeabilityParams()
    configured = load_permeability_config().params
    if params != configured:
        raise SystemExit(
            f"conf/permeability.yaml no longer matches PermeabilityParams' defaults: {configured} "
            f"vs {params}. The browser constructs the defaults -- conf/ is not in the wheel -- so "
            f"this bundle would score roads differently from the rest of the site. Either move the "
            f"defaults with the yaml, or teach the bundle to carry the params.")

    block, _ = load_example_block(PINNED_METHOD)
    epsg = block.crs.to_epsg()
    if epsg is None:
        raise SystemExit(f"{block.crs} has no EPSG code -- the browser rebuilds the CRS from one")
    log.info("loaded %s: %d parcels, %d streets, %d building points", block.block_id,
             len(block.parcels), len(block.streets), len(block.building_points))

    # Parcel adjacency and the per-parcel footprint radii are functions of `block` alone and do
    # not move when a road is added, so they are built ONCE and threaded through every solve on
    # this block -- which is what `solve_egress`' `adj`/`radii` parameters exist for.
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    radii = parcel_radii(block, params)

    # One baseline solve, and the mesh it was ASSEMBLED FROM. `EgressSolution` carries that
    # assembly precisely so nothing has to build a second `footpath_mesh` beside it that would be
    # free to disagree (see its docstring); every road-invariant column below reads off this one.
    baseline = solve_egress(block, None, params, adj=adj, radii=radii)
    mesh = baseline.mesh
    log.info("mesh: %d nodes, %d edges, %d grounded; p0 = %r", mesh.n, len(mesh.rows),
             int(mesh.ground.sum()), baseline.p)

    points = block.building_points.geometry

    reference: list[ReferenceCase] = []
    for name, road in REFERENCE_ROADS.items():
        coords = [[x, y] for x, y in road]
        sol = solve_egress(block, _road_frame(coords, params, block.crs), params,
                           adj=adj, radii=radii)
        reference.append(ReferenceCase(name=name, road=coords,
                                       permeability=1.0 - sol.p / baseline.p))
        log.info("reference %s: permeability %r", name, reference[-1]["permeability"])

    bundle = AuthoringBundle(
        block_id=block.block_id,
        crs_epsg=epsg,
        boundary=_rings(block.boundary, what=f"block {block.block_id!r}'s boundary"),
        # The source column is integer-valued on this block; the identifiers are carried as
        # strings so the declared type does not follow the source's dtype. `Block.__post_init__`
        # requires the column and the egress solve never reads it -- the round-trip assertion is
        # what holds that to account.
        parcel_id=[str(v) for v in block.parcels["parcel_id"]],
        parcels=[_rings(g, what=f"block {block.block_id!r}'s parcel {i}")
                 for i, g in enumerate(block.parcels.geometry)],
        streets=[coords for g in block.streets.geometry
                 for coords in _lines(g, what=f"block {block.block_id!r}'s street")],
        # `.x`/`.y` off the GeoSeries, the same accessor `mesh.parcel_radii` reads these points
        # through, rather than per-geometry attributes: iterating a GeoSeries yields
        # `BaseGeometry`, which has no coordinates until it is narrowed.
        building_points=[[float(x), float(y)]
                         for x, y in zip(points.x, points.y, strict=True)],
        nodes=NodesDict(cx=mesh.cx.tolist(), cy=mesh.cy.tolist(),
                        ground=[bool(g) for g in mesh.ground]),
        edges=EdgesDict(rows=mesh.rows.tolist(), cols=mesh.cols.tolist(),
                        footpath_g=mesh.footpath_g.tolist()),
        baseline=BaselineDict(p0=baseline.p, potential=baseline.potential.tolist()),
        reference=reference,
    )

    _assert_round_trip(bundle, params)
    log.info("round trip: the baked columns and %d reference roads all reproduce exactly from a "
             "block rebuilt from this JSON", len(reference))

    (OUT / "block.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT / "block.json",
             (OUT / "block.json").stat().st_size / 1024.0)

    DTS.parent.mkdir(parents=True, exist_ok=True)
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")
    log.info("wrote %s", DTS)

    (OUT / "README.md").write_text(readme_markdown(bundle), encoding="utf-8")
    log.info("wrote %s", OUT / "README.md")


if __name__ == "__main__":
    main()
