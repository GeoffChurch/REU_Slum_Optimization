"""Access-depth via a BFS parcel peel (kblock's k-complexity definition).

Layer 1 = touches a street; layer L = L-1 parcels from the nearest street.
Runs on parcels directly (STRtree adjacency + BFS) -- no topology graph, so
it is robust to the weak-dual degeneracies of `reblock.eval.kcomplexity`
(a single-file corridor scoring k=1 regardless of length; k silently
capping at 8) and is native per-parcel by `parcel_id`.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import networkx as nx
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from shapely import STRtree, union_all
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from reblock.contracts import Block
from reblock.derive.adjacency import parcel_adjacency

# The `Block.streets` seam tolerance: half a metre, plus one absolute epsilon for the float64 the
# comparison is made in. `to_parcel_graph`'s `clean_up_geometry(0.5, byblock=False)` step merges
# parcel vertices within half a unit, so anything gating that seam -- the peel's parcel-adjacency
# and street-seeding here, and `TopologyMethod`'s street-edge matching (which imports this same
# constant) -- must tolerate the same drift, or noisy real boundary geometry silently
# under-matches. One constant so they never disagree. Being an epsilon MORE tolerant than the merge
# step is harmless; being less would not be.
#
# ## Why the epsilon, and why absolute
#
# A distance is a difference of coordinates, so its error floor is set by COORDINATE magnitude, not
# by the tolerance being tested. UTM tops out at the 1e7 m false northing and float64 carries
# ~2.2e-16 relative precision, so a coordinate anywhere in any UTM zone arrives with ~2.2e-9 m of
# representational noise before a single geometry operation runs. Measured on Cape Town's
# EPSG:32734: max |coordinate| 6,237,723 m, one ULP 9.3e-10 m, and observed deviations at this very
# threshold of 5.0e-10 m -- sub-ULP debris of the coordinate system, not a flaw in any calculation.
#
# A RELATIVE tolerance is the wrong instrument here and dangerously so: 1e-9 relative to 0.5 m is
# 5e-10 m, which IS the observed noise, so the parameterisation invites a number that looks
# conservative and does not cover the case. Anything tighter covers none of it.
#
# 1e-8 m is 4.5x the 2.2e-9 m bound and 7.7 orders of magnitude below the half metre it guards: it
# cannot change a decision anyone means to make, only the ones nobody meant to make.
#
# ## Why it lives in the constant rather than at each comparison
#
# Several consumers hand `tol` straight to GEOS -- `STRtree.query(predicate="dwithin",
# distance=tol)`, `snap(a, b, tol)` -- where there is no comparison of ours to intervene in. One
# value that already carries the epsilon is the only form every consumer can honour, and it leaves
# no second almost-identical constant for a call site to pick wrongly.
#
# This was not hypothetical: `euclidean_grid` trims to `street_buffer: 0.5`, exactly this
# tolerance, so it parked every grid road on the knife edge and `_road_net` and
# `street_connectivity` disagreed about three of nine of them.
# See notes/2026-09-14-road-net-is-not-planarized.md.
STREET_TOL = 0.5 + 1e-8


@dataclass(frozen=True, eq=False)
class ParcelAdjacency:
    """A block's parcel adjacency at one tolerance: built ONCE and handed to everything that walks
    it -- the peel (`parcel_access_layers`), the permeability mesh (`reblock.permeability.
    EgressContext`), the arterial's per-candidate scoring.

    It carries its block, so nothing takes a block and its neighbour sets side by side, and no
    caller can pair one block with another block's adjacency.

    `neighbours` is a field rather than something derived on access so that a caller holding the
    adjacency from elsewhere can supply it -- the browser rebuilds it from a baked bundle, because
    computing it under Pyodide kills the interpreter (`web/src/py/solve.py`). `of` is the ordinary
    way in. Either way the length is checked against the block, which is what catches another
    block's adjacency; it cannot catch another tolerance's, so `tol` is carried beside it and every
    consumer reads the tolerance from here rather than taking its own.

    Frozen and compared by identity: two adjacencies are the same only if they are the same object,
    and nothing hashes or compares them. The neighbour sets are exactly `parcel_adjacency`'s --
    never re-built into another container, because set iteration order decides edge order in the
    permeability mesh and therefore the summation order of its Laplacian.
    """
    block: Block
    tol: float
    neighbours: list[set[int]]

    def __post_init__(self) -> None:
        if len(self.neighbours) != len(self.block.parcels):
            raise ValueError(
                f"adjacency has {len(self.neighbours)} neighbour sets but block "
                f"{self.block.block_id!r} has {len(self.block.parcels)} parcels -- it was built "
                f"for a different block")

    @classmethod
    def of(cls, block: Block, tol: float) -> ParcelAdjacency:
        """`block`'s parcel adjacency at `tol` (`parcel_adjacency`)."""
        return cls(block, tol, parcel_adjacency(list(block.parcels.geometry), tol))


UnreachedDepth = Callable[[int, int], int]
"""The depth `parcel_access_layers` reports for a parcel with NO path to a street, from
`(deepest reached layer, parcel count)`. A required argument, because the two behaviours below
answer different questions and neither is a safe default for the other."""


def one_past_deepest(deepest: int, n_parcels: int) -> int:
    """One layer past the deepest reached parcel: sorts an unreachable parcel last, honestly,
    instead of silently capping it like the weak-dual `k` did. What a displayed or reported depth
    wants. It MOVES as roads are added -- the deepest reached layer shrinks -- so it is wrong for
    anything compared across road prefixes."""
    del n_parcels
    return deepest + 1


def past_every_parcel(deepest: int, n_parcels: int) -> int:
    """`n_parcels + 1`, deeper than any true in-block depth can be. Prefix-STABLE: an access
    burden compared across road prefixes needs an unreachable parcel to cost the same at every
    prefix, and `clearance._relax_depth` needs every placeholder to sit above any real depth."""
    del deepest
    return n_parcels + 1


class StreetConnectivity(NamedTuple):
    seed_geom: BaseGeometry | None      # streets + road segments that reach a street
    n_components: int                   # touch-components among the road segments
    connected_frac: float               # road length in street-connected components / total


def _line_parts(roads: GeoDataFrame | None) -> list[BaseGeometry]:
    """Positive-length line parts of `roads` (explode Multi*; drop non-lines)."""
    if roads is None or roads.empty:
        return []
    parts: list[BaseGeometry] = []
    for g in roads.geometry:
        if g is None or g.is_empty:
            continue
        geoms: list[BaseGeometry] = list(g.geoms) if isinstance(g, BaseMultipartGeometry) else [g]
        parts.extend(p for p in geoms if "LineString" in p.geom_type and p.length > 0)
    return parts


def street_connectivity(
    streets: GeoDataFrame, roads: GeoDataFrame | None, tol: float
) -> StreetConnectivity:
    """Seed geometry = streets plus only those road segments whose touch-component
    reaches a street (variant a). Floating interior roads grant no access."""
    street_geom = union_all(list(streets.geometry)) if len(streets) else None
    segs = _line_parts(roads)
    if not segs:
        return StreetConnectivity(street_geom, 0, 0.0)
    tree = STRtree(segs)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(segs)))
    for i, g in enumerate(segs):
        for j in tree.query(g, predicate="dwithin", distance=tol):
            jj = int(j)
            if i < jj:
                graph.add_edge(i, jj)
    comps = list(nx.connected_components(graph))
    live: list[int] = []
    for comp in comps:
        if street_geom is not None and any(segs[i].distance(street_geom) <= tol for i in comp):
            live.extend(comp)
    total = sum(segs[i].length for i in range(len(segs)))
    live_len = sum(segs[i].length for i in live)
    frac = live_len / total if total > 0 else 0.0
    parts = ([street_geom] if street_geom is not None else []) + [segs[i] for i in live]
    seed = union_all(parts) if parts else None
    return StreetConnectivity(seed, len(comps), frac)


def parcel_access_layers(
    adjacency: ParcelAdjacency, roads: GeoDataFrame | None, *, unreached: UnreachedDepth,
) -> pd.Series:
    """BFS-peel access depth per parcel of `adjacency.block`: 1 = touches a street, L = L-1
    parcels deep.

    Seeds the frontier with parcels within `adjacency.tol` of the street network
    (`block.streets`, plus any additional `roads`), then peels outward one
    parcel-adjacency hop at a time. A parcel with no path to any street
    (disconnected from the rest of the block) gets the depth `unreached` assigns it --
    `one_past_deepest` or `past_every_parcel`; see each for which question it answers.

    Returned `pd.Series` is indexed by `parcel_id` (not position), so it
    survives reordering of `block.parcels`.
    """
    block, tol, adj = adjacency.block, adjacency.tol, adjacency.neighbours
    parcels = block.parcels
    ids = list(parcels["parcel_id"])
    geoms = list(parcels.geometry)

    street = street_connectivity(block.streets, roads, tol).seed_geom

    layer = [0] * len(geoms)
    # VECTORIZED frontier seed. This was a Python loop calling `g.distance(street)` once per
    # parcel, which is one GEOS round trip per parcel per call -- and this function is called once
    # per CANDIDATE inside the arterial greedy's inner loop, so it dominated: profiled at 278,833
    # scalar `shapely.measurement` calls (2.54 s of 28.8 s) on a 50-parcel block scoring 5,228
    # candidates. `shapely.dwithin` is the same predicate (`distance <= tol`) evaluated over the
    # whole array in one call. Bit-identical, verified against the scalar form over every
    # parcel/prefix pair on real blocks -- the same swap `permeability.edge_conductances` records
    # as worth 4.4x at region scale.
    if street is None or not geoms:
        frontier: deque[int] = deque()
    else:
        near = shapely.dwithin(np.asarray(geoms, dtype=object), street, tol)
        frontier = deque(int(i) for i in np.flatnonzero(near))
    seen = set(frontier)
    for i in frontier:
        layer[i] = 1

    while frontier:
        i = frontier.popleft()
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                layer[j] = layer[i] + 1
                frontier.append(j)

    stranded = [i for i, depth in enumerate(layer) if depth == 0]
    if stranded:
        far = unreached(max(layer), len(layer))
        for i in stranded:
            layer[i] = far

    return pd.Series(layer, index=pd.Index(ids, name="parcel_id"), dtype="int64")
