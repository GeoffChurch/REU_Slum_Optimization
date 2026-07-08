"""StructureEval: the orthogonal network-quality metric basis for a (block, proposal),
emitted as a Metrics. Reachability/equity via geometric access, directness via circuity,
throughput via max-flow, redundancy/permeability/crossings via the noded graph,
cross-block continuity via the interior boundaries. Smoothness (Axis I) is deferred to
the arc-emitting Phase-1 slice.
"""
from __future__ import annotations

from shapely.geometry import MultiLineString

from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.geometric_access import geometric_access_distances
from reblock.derive.network_metrics import (
    boundary_redundant_road_fraction,
    circuity,
    cross_block_trunk_length_m,
    crossing_counts,
    degree_fractions,
    meshedness,
    n_cross_block_streets,
    node_network,
    throughput_ratio,
)


class StructureEval:
    def score(self, block: Block, proposal: Proposal) -> Metrics:
        roads = proposal.roads
        interior = block.attrs.get("interior_boundaries")
        if not isinstance(interior, MultiLineString):
            interior = MultiLineString([])

        graph = node_network(roads, block.streets)
        geo = geometric_access_distances(block, roads)
        n_parcels = max(len(block.parcels), 1)
        road_len = (float(roads.geometry.length.sum())
                    if roads is not None and not roads.empty else 0.0)

        values = {
            # A reachability
            "geometric_access_max_m": float(geo.max()) if len(geo) else 0.0,
            # B equity
            "geometric_access_p95_m": float(geo.quantile(0.95)) if len(geo) else 0.0,
            # C directness
            "circuity": circuity(block, roads),
            # D throughput
            "throughput_ratio": throughput_ratio(graph, block),
            # E redundancy
            "meshedness": meshedness(graph),
            # G cost
            "added_road_length_per_parcel": road_len / n_parcels,
            # H cross-block
            "n_cross_block_streets": float(n_cross_block_streets(roads, interior)),
            "cross_block_trunk_length_m": cross_block_trunk_length_m(roads, interior),
            "boundary_redundant_road_fraction": boundary_redundant_road_fraction(roads, interior),
        }
        values.update({k: float(v) for k, v in degree_fractions(graph).items()})  # F permeability
        values.update({k: float(v) for k, v in crossing_counts(graph).items()})  # F crossings/T
        return Metrics(
            block_id=block.block_id, method=proposal.method, eval="structure", values=values)
