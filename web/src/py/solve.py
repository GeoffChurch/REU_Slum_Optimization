"""The Python that runs inside Pyodide.

Lives under `web/src/` because it ships to the browser like everything else here, and is plain
CPython so `tests/test_solve_py.py` can test it without a runtime. Its imports below --
`geopandas`, `pyproj`, `shapely`, and `reblock.contracts`/`reblock.permeability` themselves -- are
exactly what `reblock.permeability` already pulls in transitively (via `reblock.contracts`, which
it imports `Block` from). Nothing here adds a package to the browser's install.

DEBT: `web/test/pyodide-parity.test.ts` does not exist at this commit -- Task 5 of this piece adds
it. It will run this same file under the Pyodide runtime and check its answers against
`tests/test_solve_py.py`'s CPython ones, which is the only thing that could catch an import added
here: this file has no Python test of its own install cost.

It calls `solve_egress`, NOT `permeability`: `permeability` runs two solves because it recomputes
the no-roads baseline every call, and that baseline is road-invariant (design §1.6) and already
baked into `examples/authoring/block.json` as `baseline.p0`. One solve per edit is what makes
re-solve-while-dragging affordable.
"""
from __future__ import annotations

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.permeability import WIDTH_COL, PermeabilityParams, solve_egress


def block_from_bundle(bundle: dict) -> Block:
    """Rebuild a `Block` from an `examples/authoring/block.json`-shaped bundle.

    Extracted from `scripts/gen_authoring_block.py`'s own reconstruction, which now loads this
    function back (by path -- see that script's loader) so the baker and this runtime share one
    reconstruction instead of two that could drift apart.
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


def solve(block: Block, road: list[list[float]], p0: float) -> dict:
    """One `solve_egress` call for a reader-drawn `road`, normalized against the caller-supplied
    `p0` (the bundle's baked no-roads baseline).

    Refused here if `road` has fewer than two points, ahead of the widget's own check: the
    widget's check exists so the reader never gets this far, this one is the contract a caller
    cannot skip. `PermeabilityParams()` is constructed with no arguments -- the wheel `micropip`
    installs ships `src/reblock` alone, so `conf/` cannot reach the browser, and the baked
    `examples/authoring/block.json` was itself baked against these same defaults.
    """
    if len(road) < 2:
        raise ValueError("a road needs at least two points")
    line = LineString([(x, y) for x, y in road])
    params = PermeabilityParams()
    roads = gpd.GeoDataFrame(
        {"geometry": [line], WIDTH_COL: [params.min_road_width_m]},
        geometry="geometry", crs=block.crs)
    sol = solve_egress(block, roads, params)
    return {
        "permeability": 1.0 - sol.p / p0,
        "roadMetres": line.length,
        "potential": sol.potential.tolist(),
        "conductance": sol.conductance.tolist(),
    }
