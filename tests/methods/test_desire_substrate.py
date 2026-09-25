"""DesireWarpedSubstrate: the base graph, routed in a metric shortened along desire lines."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from reblock.contracts import Block
from reblock.derive_graph import closure_hash
from reblock.methods.demand_greedy import demand_edge_weights
from reblock.methods.desire_lines import DesireField, NoDesire, mapped_field
from reblock.methods.desire_substrate import DesireWarpedSubstrate
from reblock.methods.substrates import ChordSubstrate, GridSubstrate, RoutingGraph
from tests.methods.test_demand_greedy import _line_at, _slab


class _Lines:
    """A desire source of one mapped line at height `y`."""

    def __init__(self, y: float, identity: Hashable = ("test", "lines")) -> None:
        self.y, self.identity = y, identity

    def desire_field(self, block: Block) -> DesireField:
        return mapped_field(_line_at(block, self.y))


WARPED = DesireWarpedSubstrate(base=GridSubstrate(res=0.5), desire_source=_Lines(3.0),
                               buffer_m=0.3, eps=0.1, gamma=1.0)


def _path_ys(graph: RoutingGraph, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """The y of every node on the shortest path between the nodes nearest `a` and `b`."""
    n = len(graph.pts)
    csr = csr_matrix((graph.edist, (graph.rows, graph.cols)), shape=(n, n))
    i = int(np.argmin(np.hypot(*(graph.pts - a).T)))
    j = int(np.argmin(np.hypot(*(graph.pts - b).T)))
    _, pred = dijkstra(csr, indices=i, return_predecessors=True)
    path = [j]
    while path[-1] != i:
        path.append(int(pred[path[-1]]))
    return graph.pts[path, 1]


def test_only_the_edge_lengths_change() -> None:
    block = _slab(6, 5)
    base, warped = WARPED.base.build(block), WARPED.build(block)
    np.testing.assert_array_equal(warped.pts, base.pts)
    np.testing.assert_array_equal(warped.rows, base.rows)
    np.testing.assert_array_equal(warped.cols, base.cols)
    assert warped.net_tol == base.net_tol
    np.testing.assert_array_equal(
        warped.edist, demand_edge_weights(base, WARPED.desire_source.desire_field(block),
                                          buffer_m=0.3, eps=0.1, gamma=1.0))


def test_a_route_bends_toward_the_desire_line() -> None:
    """Across the block low down, the Euclidean shortest path stays low; warped toward a line
    high up, it climbs to use it.

    FAULT INJECTION: returning the base graph unwarped from `build` fails this.
    """
    block = _slab(6, 5)
    a, b = (0.5, 0.5), (5.5, 0.5)
    assert _path_ys(WARPED.base.build(block), a, b).max() < 1.5
    assert _path_ys(WARPED.build(block), a, b).max() >= 2.5


def test_no_desire_scales_every_edge_alike() -> None:
    """With no field every edge is outside every corridor, so every edge is divided by the same
    eps^gamma and every shortest path is unchanged."""
    block = _slab(6, 5)
    flat = replace(WARPED, desire_source=NoDesire())
    np.testing.assert_allclose(flat.build(block).edist, WARPED.base.build(block).edist / 0.1)


def test_identity_carries_its_code_and_every_setting() -> None:
    ident = WARPED.identity
    assert isinstance(ident, tuple) and ident[0] == closure_hash("reblock.methods.desire_substrate")
    for changed in (replace(WARPED, gamma=2.0), replace(WARPED, eps=0.2),
                    replace(WARPED, buffer_m=1.0), replace(WARPED, base=ChordSubstrate()),
                    replace(WARPED, desire_source=_Lines(3.0, identity=("test", "other")))):
        assert changed.identity != ident


def test_a_live_desire_source_makes_it_uncacheable() -> None:
    assert replace(WARPED, desire_source=_Lines(3.0, identity=None)).identity is None


def test_the_tag_names_the_base_and_separates_desire_sources() -> None:
    other = replace(WARPED, desire_source=_Lines(3.0, identity=("test", "other")))
    assert WARPED.tag.startswith("grid+desire:") and other.tag != WARPED.tag


@pytest.mark.parametrize("field,value", [("buffer_m", -1.0), ("eps", 0.0), ("gamma", 0.0)])
def test_rejects_settings_with_no_meaning(field: str, value: float) -> None:
    kwargs: dict[str, float] = {"buffer_m": 0.3, "eps": 0.1, "gamma": 1.0, field: value}
    with pytest.raises(ValueError, match="DesireWarpedSubstrate needs"):
        DesireWarpedSubstrate(base=GridSubstrate(res=0.5), desire_source=_Lines(3.0), **kwargs)


def test_empty_block_lines_are_an_empty_field() -> None:
    """A desire source with no lines leaves the metric flat, like NoDesire."""
    block = _slab(6, 5)

    class _Empty:
        identity = ("test", "empty")

        def desire_field(self, block: Block) -> DesireField:
            return mapped_field(gpd.GeoDataFrame(geometry=[], crs=block.crs))

    np.testing.assert_allclose(replace(WARPED, desire_source=_Empty()).build(block).edist,
                               WARPED.base.build(block).edist / 0.1)
