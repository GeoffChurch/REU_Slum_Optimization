"""Put transported donor linework onto the recipient's own routing substrate.

A transported vertex lands wherever the GW map sends it, buildings included. Both snaps pull every
vertex onto the nearest node of the recipient's substrate graph (the chord substrate
`ClearanceReblocker` routes on, whose nodes sit in the gaps between parcels) and collapse
consecutive repeats. They differ only in how consecutive nodes are then joined:

- `NearestNodeSnap` joins them with a straight segment, which crosses whatever lies between two
  nodes that are not substrate-adjacent. It is the pair matrix's transplant.
- `RoutedSnap` joins them by the shortest substrate path, so every hop is a real gap edge. It is
  the single-donor transplant's (`donor_transplant.DonorTransplantReblocker`).

On the one pair both were scored on, routing came out slightly worse (0.666 vs 0.682 permeability
at 46.9% displacement either way, docs/superpowers/notes/2026-07-23-ot-road-transplant.md §3).
That is n=1, and each snap backs a published benchmark, so both stay. Routing on a
clearance-repulsion edge cost instead of length was tried on the same pair and lost at every
repulsion from -6 to +6 (displacement flat, permeability falling as the paths lengthened), so it
is not offered.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, runtime_checkable

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from reblock.contracts import Block
from reblock.methods.substrates import RoutingGraph, Substrate
from reblock.transplant.gw import Arr
from reblock.transplant.transport import line_parts

log = logging.getLogger(__name__)


@runtime_checkable
class GapSnap(Protocol):
    def snap(self, lines: gpd.GeoDataFrame, block: Block) -> gpd.GeoDataFrame:
        """`lines` (in `block`'s CRS) moved onto `block`'s substrate; a line left with fewer
        than two distinct nodes is dropped."""
        ...


def _xy(p: Arr) -> tuple[float, float]:
    return (float(p[0]), float(p[1]))


@dataclass(frozen=True)
class NearestNodeSnap:
    substrate: Substrate

    def snap(self, lines: gpd.GeoDataFrame, block: Block) -> gpd.GeoDataFrame:
        graph = self.substrate.build(block)
        tree = cKDTree(graph.pts)
        out: list[LineString] = []
        for geom in lines.geometry:
            for part in line_parts(geom):
                _, idx = tree.query(np.asarray(part.coords, dtype=np.float64))
                snapped = graph.pts[idx]
                cleaned = [_xy(snapped[0])]
                for pt in snapped[1:]:
                    if _xy(pt) != cleaned[-1]:
                        cleaned.append(_xy(pt))
                if len(cleaned) >= 2:
                    out.append(LineString(cleaned))
        return gpd.GeoDataFrame(geometry=out, crs=block.crs)


def _length_graph(graph: RoutingGraph) -> nx.Graph:
    """The substrate as an undirected graph weighted by each edge's own length -- walking
    distance along the gap, not crow-flies between snap points."""
    g = nx.Graph()
    g.add_nodes_from(range(len(graph.pts)))
    mask = graph.rows < graph.cols                  # one direction per undirected edge
    for u, v, d in zip(graph.rows[mask], graph.cols[mask], graph.edist[mask], strict=True):
        g.add_edge(int(u), int(v), weight=float(d))
    return g


@dataclass(frozen=True)
class RoutedSnap:
    substrate: Substrate

    def snap(self, lines: gpd.GeoDataFrame, block: Block) -> gpd.GeoDataFrame:
        """A line whose consecutive nodes fall in different substrate components cannot be
        routed and is dropped whole; a single block's tessellation is connected, so each drop is
        logged rather than expected."""
        graph = self.substrate.build(block)
        tree = cKDTree(graph.pts)
        nxg = _length_graph(graph)
        out: list[LineString] = []
        dropped = 0
        for geom in lines.geometry:
            for part in line_parts(geom):
                coords = np.asarray(part.coords, dtype=np.float64)
                if len(coords) < 2:
                    continue
                _, idx = tree.query(coords)
                nodes = [int(idx[0])]
                for i in idx[1:]:
                    if int(i) != nodes[-1]:
                        nodes.append(int(i))
                if len(nodes) < 2:
                    continue
                path_xy = [_xy(graph.pts[nodes[0]])]
                routed = True
                for u, v in pairwise(nodes):
                    try:
                        path = nx.shortest_path(nxg, u, v, weight="weight")
                    except nx.NetworkXNoPath:
                        routed = False
                        dropped += 1
                        break
                    for node in path[1:]:
                        pt = _xy(graph.pts[node])
                        if pt != path_xy[-1]:
                            path_xy.append(pt)
                if routed and len(path_xy) >= 2:
                    out.append(LineString(path_xy))
        if dropped:
            log.warning("RoutedSnap: dropped %d line(s) spanning disconnected substrate nodes",
                        dropped)
        return gpd.GeoDataFrame(geometry=out, crs=block.crs)
