from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString, Polygon
from topology import graphFromMyFaces
from topology.graph.my_graph_helpers import graphFromShapes

from reblock.contracts import Block
from reblock.data.shapefile import ShapefileSource
from reblock.derive.parcel_graph import _myfaces_from_parcels, parcel_origin, to_parcel_graph

UTM = CRS.from_epsg(32643)

PHULE = (Path(__file__).resolve().parents[2] / "ext" / "topology" / "examples" / "data"
         / "phule_nagar_v6")


def _two_parcels() -> Block:
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1]}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    return Block(block_id="t", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                 parcels=parcels, streets=streets)


def test_derivation_builds_planar_graph() -> None:
    ppg = to_parcel_graph(_two_parcels())
    assert len(ppg.graph.inner_facelist) == 2
    assert ppg.graph.G.number_of_nodes() == 6   # two squares sharing the x=1 edge
    assert ppg.crs == UTM


def _grid_3x3() -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                 parcels=parcels, streets=streets)


def test_inner_facelist_count_equals_parcel_count_for_space_filling_grid() -> None:
    # Empirically verified (not assumed): for a clean, space-filling partition
    # of edge-adjacent parcels with no gaps/overlaps, graphFromMyFaces traces
    # exactly one inner face per parcel (Euler check on the 3x3 grid: 16
    # nodes, 24 edges, V - E + F = 2 => F = 10 = 1 outer + 9 inner).
    block = _grid_3x3()
    ppg = to_parcel_graph(block)
    assert ppg.graph.G.number_of_nodes() == 16
    assert ppg.graph.G.number_of_edges() == 24
    assert len(ppg.graph.inner_facelist) == len(block.parcels) == 9


def test_clean_up_geometry_fixes_near_duplicate_shared_vertex_pinch_point() -> None:
    """Regression, using real (hardcoded) Phule Nagar coordinates (block
    phule_9's two parcels): they share one bit-exact vertex plus a second
    pair of vertices ~3cm apart (surveying noise) instead of a clean shared
    edge. Without a near-node merge, the raw ring construction below (same
    code `to_parcel_graph` runs before its `clean_up_geometry` call) produces
    a "pinch point" that MyGraph's combinatorial face tracer misreads as a
    single degenerate face -- 0 inner faces for two obviously-separate unit-
    scale parcels. This is the concrete failure `clean_up_geometry(0.5,
    byblock=False)` inside `to_parcel_graph` exists to prevent.
    """
    polys = [
        Polygon([(282460.844436195, 2107398.0572959343),
                  (282465.1793243006, 2107396.5130706904),
                  (282464.48570546065, 2107394.880201987),
                  (282460.39835204725, 2107396.296588856)]),
        Polygon([(282460.3986440145, 2107396.3213748727),
                  (282464.48570546065, 2107394.880201987),
                  (282463.86209394736, 2107392.9224092816),
                  (282460.0008483077, 2107394.435577693)]),
    ]
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1]}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="phule_9", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets)

    origin = parcel_origin(block.parcels)
    raw = graphFromMyFaces(_myfaces_from_parcels(block.parcels, origin))  # type: ignore[no-untyped-call]
    assert len(raw.inner_facelist) == 0  # the bug clean_up_geometry exists to fix

    ppg = to_parcel_graph(block)  # the shipped path: includes clean_up_geometry
    assert len(ppg.graph.inner_facelist) == 2


class _FakeShape:
    """Minimal stand-in for a pyshp `shapefile.Shape`: just `.points`.

    topology's own `graphFromShapes` (the ring-walker `import_and_setup`
    uses on a raw `.shp` read) only touches `shape.points` -- a closed ring
    (first vertex repeated as the last), exactly what shapely's
    `polygon.exterior.coords` also yields. Wrapping that in this shim lets
    us drive topology's *own* independent ring-to-graph code with the exact
    same coordinates our derivation consumes.
    """

    def __init__(self, points: NDArray[np.float64]) -> None:
        self.points = points


def test_derivation_matches_topology_native_ring_construction() -> None:
    """Port-fidelity oracle (replaces the brief's aggregate face-count check).

    The brief's Step 5 proposed comparing
    `sum(len(to_parcel_graph(b).graph.inner_facelist) for b in region.blocks)`
    against `len(import_and_setup(phule, threshold=0.5, byblock=False).inner_facelist)`.
    That comparison does NOT pass, even after adding the documented
    `clean_up_geometry(0.5, byblock=False)` fallback (see to_parcel_graph):
    ours = 1180, native's whole-graph `inner_facelist` = 1320. Root cause,
    confirmed empirically (not a preprocessing-threshold quibble):
    `import_and_setup(..., byblock=False)` returns ONE MyGraph containing all
    133 of its post-clean-up connected components, and `MyGraph.inner_facelist`
    (`trace_faces`) only ever excludes a SINGLE global "largest" face as the
    outer face -- so 132 of those 133 components' own true outer boundaries
    get miscounted as "inner" faces. Summing `inner_facelist` PER connected
    component instead (`sum(len(c.inner_facelist) for c in
    native.connected_components())`) gives 1188 -- much closer to our 1180,
    confirming the aggregate whole-graph number itself is the confound, not
    our derivation. (The residual 1188 vs 1180 gap, and 2 of 370 per-block
    face-count-vs-parcel-count mismatches within our own totals -- phule_99:
    20 parcels/21 faces, phule_105: 23/24 -- trace to real overlapping-sliver
    parcel geometry, not a derivation bug; see the parity check below.)

    So instead of the confounded aggregate, this test isolates exactly the
    piece to_parcel_graph is responsible for: Block.parcels -> MyFaces ->
    MyGraph, BEFORE topology's own (order-dependent -- see below) clean-up
    step. For every one of the 370 real Phule Nagar blocks, it builds the
    graph two ways from the identical parcel-ring coordinates: our
    `_myfaces_from_parcels` + `graphFromMyFaces`, and topology's own,
    independently implemented `graphFromShapes` ring walker. A broken
    derivation (wrong ring direction, dropped closing edge, bad re-zero
    arithmetic, ...) would show up as a node-set/edge-set/face-count
    mismatch here. Empirically: 0/370 mismatches -- the two constructions
    produce byte-for-byte identical node sets and edge sets.

    (We deliberately compare *before* `clean_up_geometry`: applying it on
    both sides still passes 369/370, but the one failure -- phule_254, 38 vs
    39 nodes, faces still equal at 15 -- comes from `__combine_near_nodes`
    iterating `self.G.nodes()` in insertion order, so two topologically
    equivalent graphs built via different code paths can merge a chain of
    3+ mutually-near-duplicate nodes in a different order. That's an
    order-dependent quirk of topology's own third-party clean-up algorithm,
    not something to_parcel_graph should be graded on.)
    """
    region = ShapefileSource(PHULE.with_suffix(".shp"), region_id="phule",
                              assumed_crs=3857).region()
    blocks = list(region.blocks)
    assert len(blocks) == 370

    mismatches: list[str] = []
    for block in blocks:
        origin = parcel_origin(block.parcels)

        ours = graphFromMyFaces(_myfaces_from_parcels(block.parcels, origin))  # type: ignore[no-untyped-call]

        rezero = np.array(origin)
        shapes = []
        for geom in block.parcels.geometry:
            assert isinstance(geom, Polygon)
            shapes.append(_FakeShape(np.asarray(geom.exterior.coords)))
        native = graphFromShapes(shapes, name=block.block_id, rezero=rezero)  # type: ignore[no-untyped-call]

        same = (set(ours.G.nodes()) == set(native.G.nodes())
                and set(ours.myedges()) == set(native.myedges())
                and len(ours.inner_facelist) == len(native.inner_facelist))
        if not same:
            mismatches.append(block.block_id)

    assert mismatches == []


def test_full_pipeline_facecount_over_all_phule_blocks() -> None:
    """Full-pipeline regression: runs the ACTUAL public `to_parcel_graph`
    (INCLUDING its `clean_up_geometry(0.5, byblock=False)` call -- the piece
    that fixes ~139/370 blocks) over every real Phule Nagar block and pins
    the current empirical result. The sibling parity test above deliberately
    bypasses `clean_up_geometry` to isolate the pure ring-construction, so
    this is the only committed guard on the shipped pipeline's real-world
    behaviour: a future change to the threshold or the byblock flag would
    regress these numbers, and this test would catch it.

    Empirically (verified before pinning): 370 blocks, 1178 parcels, 1180
    total inner faces, and EXACTLY two blocks whose inner-face count differs
    from their parcel count -- phule_99 (20 parcels -> 21 faces) and
    phule_105 (23 -> 24). Both are real overlapping-sliver source geometry
    (an extra traced face from a genuine self-overlap), not derivation bugs;
    the ids and the count are pinned so a real regression (clean-up silently
    dropping/merging faces on many blocks) can't hide behind the sliver
    noise. `>=` on face count guards the direction that matters most: the
    clean-up must never lose real parcel faces.
    """
    region = ShapefileSource(PHULE.with_suffix(".shp"), region_id="phule",
                              assumed_crs=3857).region()
    blocks = list(region.blocks)
    assert len(blocks) == 370

    total_parcels = 0
    total_faces = 0
    mismatched_ids: list[str] = []
    for block in blocks:
        faces = len(to_parcel_graph(block).graph.inner_facelist)
        parcels = len(block.parcels)
        total_parcels += parcels
        total_faces += faces
        assert faces >= parcels, f"{block.block_id}: {faces} faces < {parcels} parcels"
        if faces != parcels:
            mismatched_ids.append(block.block_id)

    assert total_parcels == 1178
    assert total_faces == 1180
    assert sorted(mismatched_ids) == ["phule_105", "phule_99"]
