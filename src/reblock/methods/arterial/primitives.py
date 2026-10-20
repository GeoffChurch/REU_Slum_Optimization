"""Geometry primitives shared by the arterial engines: target sampling, candidate-chord
enumeration, the parcel-boundary snap graph's precomputed data, and the merge/explode/planarize
helpers that turn a list of `LineString`s into a road `GeoDataFrame`. No engine loop and no scoring
live here -- see `engines.py` and `scoring.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import unary_union

from reblock.derive.access import ParcelAdjacency, one_past_deepest, parcel_access_layers
from reblock.methods.boundary_graph import _rnd
from reblock.permeability import with_width


def _xy(c: tuple[float, ...]) -> tuple[float, float]:
    """First two components of a coordinate tuple (shapely yields 3-tuples for Z-aware
    geometry; every geometry here is 2-D, so drop anything past x, y)."""
    return (c[0], c[1])


def _deep_targets(adjacency: ParcelAdjacency, roads: GeoDataFrame | None,
                  k: int) -> list[tuple[float, float]]:
    """Representative points of the k deepest-access parcels (spur targets), _rnd-snapped."""
    block = adjacency.block
    depths = parcel_access_layers(adjacency, roads, unreached=one_past_deepest)
    order = depths.sort_values(ascending=False, kind="stable")
    id_to_pos = {pid: i for i, pid in enumerate(block.parcels["parcel_id"])}
    geoms = list(block.parcels.geometry)
    out: list[tuple[float, float]] = []
    for pid in list(order.index)[:k]:
        rep = geoms[id_to_pos[pid]].representative_point()
        out.append(_rnd(_xy(rep.coords[0])))
    return out


def _candidate_chords(anchors: list[tuple[float, float]],
                       targets: list[tuple[float, float]]) -> list[LineString]:
    """Through-roads (anchor pairs) + spurs (anchor -> deep target). De-duplicated, sorted."""
    seen: set[frozenset[tuple[float, float]]] = set()
    chords: list[LineString] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            key = frozenset((a, b))
            if a != b and key not in seen:
                seen.add(key)
                pair: list[tuple[float, float]] = sorted((a, b))
                chords.append(LineString(pair))
        for t in targets:
            key = frozenset((a, t))
            if a != t and key not in seen:
                seen.add(key)
                spur: list[tuple[float, float]] = sorted((a, t))
                chords.append(LineString(spur))
    return sorted(chords, key=lambda ls: ls.wkt)


@dataclass
class _SnapGraph:
    """`_boundary_graph(block.parcels)` plus everything `_snap` needs precomputed ONCE per block:
    the nearest-node lookup tree and, for every graph edge, its midpoint (as a shapely `Point`,
    for the vectorized ufunc below) and length. Building this once per block -- instead of per
    candidate chord -- lifts the per-edge Python-loop `mid.distance(chord)` (the profiled `_snap`
    hot cost) out of the greedy's ~thousands-of-candidates inner loop."""
    g: nx.Graph
    node_tree: STRtree
    nodes: list[tuple[float, float]]
    edges: list[tuple[tuple[float, float], tuple[float, float]]]
    mid_points: np.ndarray               # shapely Point per edge, aligned with `edges`
    lengths: np.ndarray                  # edge length (== g[u][v]["weight"]), aligned with `edges`


def _snap_graph(g: nx.Graph) -> _SnapGraph:
    nodes = list(g.nodes)
    node_tree = STRtree([Point(nd) for nd in nodes])
    edges = list(g.edges())
    xs = np.array([(u[0] + v[0]) / 2 for u, v in edges], dtype=float)
    ys = np.array([(u[1] + v[1]) / 2 for u, v in edges], dtype=float)
    lengths = np.array([g[u][v]["weight"] for u, v in edges], dtype=float)
    return _SnapGraph(g, node_tree, nodes, edges, shapely.points(xs, ys), lengths)


def _merge(lines: list[LineString]) -> BaseGeometry | None:
    """`unary_union(lines)`, or `None` for an empty list (the incremental-`_planarize` base)."""
    return unary_union(lines) if lines else None


def _explode(merged: BaseGeometry | None, crs: CRS, width_m: float) -> GeoDataFrame:
    """Explode a (possibly `None`) merged/noded geometry into a one-row-per-LineString
    GeoDataFrame, every row stamped with `width_m` -- road width is mandatory, and these
    frames are scored by `displacement` exactly like an emitted proposal's."""
    parts: list[BaseGeometry] = (
        [] if merged is None
        else (list(merged.geoms) if isinstance(merged, BaseMultipartGeometry) else [merged]))
    rows = [ln for ln in parts if "LineString" in ln.geom_type and ln.length > 0]
    return with_width(gpd.GeoDataFrame({"geometry": rows}, geometry="geometry", crs=crs),
                      width_m)


def _planarize(lines: list[LineString], crs: CRS, width_m: float) -> GeoDataFrame:
    """unary_union the lines (nodes crossings), explode to LineStrings, one row each."""
    return _explode(_merge(lines), crs, width_m)
