"""Derivation: Block -> topology's planar parcel graph (a MyGraph view)."""
from __future__ import annotations

from dataclasses import dataclass

from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon
from topology import MyEdge, MyFace, MyGraph, MyNode, graphFromMyFaces

from reblock.contracts import Block


@dataclass(frozen=True)
class PlanarParcelGraph:
    graph: MyGraph
    origin: tuple[float, float]
    crs: CRS


def _myfaces_from_parcels(parcels: GeoDataFrame, origin: tuple[float, float]) -> list[MyFace]:
    """Ring (exterior coords, re-zeroed) -> MyFace, one per parcel.

    Factored out of `to_parcel_graph` so the port-fidelity oracle
    (tests/derive/test_parcel_graph.py) can build the *same* pre-clean-up
    graph we build here and diff it node-for-node/edge-for-edge against
    topology's own `graphFromShapes` ring walker, on identical input.
    """
    faces: list[MyFace] = []
    for geom in parcels.geometry:
        if not isinstance(geom, Polygon):
            raise ValueError(f"parcel geometry must be a Polygon, got: {type(geom).__name__}")
        ring = list(geom.exterior.coords)[:-1]
        nodes = [MyNode((x - origin[0], y - origin[1])) for x, y in ring]
        edges = [MyEdge((nodes[i], nodes[(i + 1) % len(nodes)])) for i in range(len(nodes))]
        faces.append(MyFace(edges))
    return faces


def to_parcel_graph(block: Block) -> PlanarParcelGraph:
    minx, miny, _, _ = block.parcels.total_bounds
    origin = (float(minx), float(miny))

    graph = graphFromMyFaces(_myfaces_from_parcels(block.parcels, origin))
    # Real parcel geometry has near-coincident (but not bit-identical) shared
    # vertices and slivers of overlap between adjacent parcels (surveying /
    # digitization noise) -- see topology's own import_and_setup, which always
    # runs clean_up_geometry before tracing faces. Without this, MyGraph's
    # combinatorial face tracer can misread a shared-vertex pinch point (two
    # parcels touching at one exact vertex plus a pair of near-duplicate
    # vertices a few cm apart) as a single degenerate face instead of two
    # separate parcel faces. byblock=False so nodes can merge even when two
    # parcels of the same Block don't share an exact vertex at all (and so
    # are separate connected components in the raw, unmerged graph).
    graph = graph.clean_up_geometry(0.5, byblock=False)
    return PlanarParcelGraph(graph=graph, origin=origin, crs=block.crs)
