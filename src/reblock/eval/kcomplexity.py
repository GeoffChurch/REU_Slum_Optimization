"""KComplexityEval: Δ of topology's k-complexity from inserting proposed roads."""
from __future__ import annotations

from geopandas import GeoDataFrame
from shapely.geometry import LineString
from topology import k_complexity

from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.parcel_graph import to_parcel_graph

_EndpointKey = frozenset[tuple[float, float]]


def _endpoint_keys(lines: GeoDataFrame, origin: tuple[float, float]) -> set[_EndpointKey]:
    keys: set[_EndpointKey] = set()
    for geom in lines.geometry:
        if isinstance(geom, LineString):
            pts = [(round(x - origin[0], 2), round(y - origin[1], 2)) for x, y in geom.coords]
            for a, b in zip(pts, pts[1:], strict=False):
                keys.add(frozenset((a, b)))
    return keys


def _k(block: Block, extra_roads: GeoDataFrame | None) -> int:
    # Slice 1: Block.streets == the block boundary, so topology's native
    # define_roads() (outer-face detection) marks the initial streets robustly.
    # Proposed interior roads are 2-point method edges matched by exact endpoints.
    ppg = to_parcel_graph(block)
    ppg.graph.define_roads()
    if extra_roads is not None and not extra_roads.empty:
        keys = _endpoint_keys(extra_roads, ppg.origin)
        for edge in ppg.graph.myedges():
            a, b = edge.nodes
            if frozenset(((a.x, a.y), (b.x, b.y))) in keys:
                edge.road = True
    return k_complexity(ppg.graph)


class KComplexityEval:
    def score(self, block: Block, proposal: Proposal) -> Metrics:
        k_before = _k(block, None)
        k_after = _k(block, proposal.roads)
        added = (float(proposal.roads.geometry.length.sum())
                 if proposal.roads is not None and not proposal.roads.empty else 0.0)
        return Metrics(block_id=block.block_id, method=proposal.method, eval="kcomplexity",
                       values={"k_before": float(k_before), "k_after": float(k_after),
                               "delta_k": float(k_before - k_after),
                               "added_road_length_m": added})
