"""Transparency in the road methods: what the price changes in each and what it must not, and that
the price and the route each move a network toward the buildings they discount.

The block is irregular on purpose: 80 Voronoi parcels with square footprints of mixed size that
straddle the parcel edges, so that every route crosses some building and a discount decides which.
On a regular lattice with the footprints inside their parcels the parcel edges are free, and a
discount mostly decides nothing.

Fault-injected where a docstring says so: the fault was applied, the test seen to fail, the code
restored.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace

import geopandas as gpd
import numpy as np
import pytest
import shapely
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon, box

from reblock.budget import road_corridor
from reblock.buildings import ANCHOR_COL, Footprints
from reblock.contracts import Block, Method
from reblock.methods.clearance import ClearanceReblocker, _edge_weights, _sigmoid
from reblock.methods.cycle_native import CycleNativeReblocker
from reblock.methods.resistance_lp import ResistanceLPReblocker
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from reblock.transparency import Opaque, Transparency, TransparentSubstrate, price_tag
from tests.methods.test_clearance import CLEARANCE
from tests.methods.test_cycle_native import CYCLE
from tests.permeability_fixtures import SHIPPED
from tests.test_loop_closure import LOOPS

UTM = CRS.from_epsg(32734)
SIDE_M = 90.0


@dataclass(frozen=True)
class Uniform:
    """Every building at one factor."""

    f: float

    @property
    def identity(self) -> Hashable:
        return ("uniform", self.f)

    @property
    def tag(self) -> str:
        return price_tag(self.identity)

    def factors(self, block: Block) -> NDArray[np.float64]:
        return np.full(len(block.buildings), self.f)


@dataclass(frozen=True)
class Chosen:
    """`members` at factor `f`, every other building 1."""

    members: tuple[int, ...]
    f: float

    @property
    def identity(self) -> Hashable:
        return ("chosen", self.members, self.f)

    @property
    def tag(self) -> str:
        return price_tag(self.identity)

    def factors(self, block: Block) -> NDArray[np.float64]:
        out = np.ones(len(block.buildings))
        out[list(self.members)] = self.f
        return out


def _block() -> Block:
    """80 Voronoi parcels on a 90 m square, streeted along the bottom, each with a square footprint
    of side 4.2 - 8.4 m around its anchor (so footprints straddle parcel edges and overlap)."""
    rng = np.random.default_rng(0)
    anchors = rng.uniform([0.0, 0.0], [SIDE_M, SIDE_M], size=(80, 2))
    bound = box(0.0, 0.0, SIDE_M, SIDE_M)
    cells = [c.intersection(bound) for c in shapely.get_parts(
        shapely.voronoi_polygons(shapely.multipoints(anchors), extend_to=bound))]
    parcels: list[Polygon] = [next(c for c in cells if c.contains(Point(p))) for p in anchors]
    sides = rng.uniform(4.2, 8.4, size=len(anchors))
    outlines = [box(x - s / 2, y - s / 2, x + s / 2, y + s / 2)
                for (x, y), s in zip(anchors, sides, strict=True)]
    return Block(
        block_id="voronoi", crs=UTM, boundary=bound,
        parcels=gpd.GeoDataFrame({"parcel_id": [str(i) for i in range(len(anchors))]},
                                 geometry=parcels, crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[LineString([(-5.0, 0.0), (SIDE_M + 5.0, 0.0)])],
                                 crs=UTM),
        building_geometries=gpd.GeoDataFrame({ANCHOR_COL: [Point(p) for p in anchors]},
                                             geometry=outlines, crs=UTM),
        source_content_hash=None, building_tier=Footprints)


def _corridor(block: Block) -> tuple[int, ...]:
    """The buildings anchored in the middle fifth of the block, street to back: a corridor, as the
    clearings the experiment discounts are corridors and pockets rather than scatter."""
    x = block.buildings.xy[:, 0]
    return tuple(int(i) for i in np.flatnonzero((x > 0.4 * SIDE_M) & (x < 0.6 * SIDE_M)))


def _same_roads(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> bool:
    """The same roads in the same order, coordinate for coordinate."""
    return bool(np.array_equal(shapely.to_wkb(np.asarray(a.geometry)),
                               shapely.to_wkb(np.asarray(b.geometry))))


# ------------------------------------------------------------- cycle_native: a scale-free price

CAP = 0.20


def test_cycle_natives_cap_binds_on_the_block() -> None:
    """What lets the next test fail: at CAP the displacement cap is what stops the method, so a cap
    that read the priced (smaller) displacement would build on."""
    block = _block()
    at_cap = replace(CYCLE, max_displacement=CAP).propose(block).roads
    above = replace(CYCLE, max_displacement=1.5 * CAP).propose(block).roads
    assert at_cap is not None and above is not None and 0 < len(at_cap) < len(above)


@pytest.mark.parametrize("f", [0.5, 0.3, 0.1])
def test_a_uniform_price_is_a_no_op_for_cycle_native(f: float) -> None:
    """Every building at one factor scales every candidate's priced cost alike, so the ranking --
    gain per unit of priced displacement -- is unchanged, and the cap is in TRUE displacement: the
    network is the opaque one, road for road.

    FAULT INJECTION: letting the cap read priced displacement (`spent = taken.priced_total() / n_b`
    and `d = spent + inc.priced / n_b` in `CycleNativeReblocker.propose`) fails this for every f."""
    block = _block()
    opaque = replace(CYCLE, max_displacement=CAP).propose(block).roads
    priced = replace(CYCLE, max_displacement=CAP, price=Uniform(f)).propose(block).roads
    assert opaque is not None and priced is not None and len(opaque) > 0
    assert _same_roads(opaque, priced)


# ------------------------------------------- the price and the route reach what each method builds

STRONG = 0.05

RoadMethod = CycleNativeReblocker | ResistanceLPReblocker | ClearanceReblocker

METHODS: dict[str, RoadMethod] = {
    "cycle_native": replace(CYCLE, max_displacement=CAP),
    "resistance_lp": ResistanceLPReblocker(substrate=ChordSubstrate(), max_displacement=CAP,
                                           max_road_m=1e6, chunks=8, params=SHIPPED,
                                           road_width_m=DEFAULT_ROAD_WIDTH_M, price=Opaque()),
    "clearance": CLEARANCE,
}


def _routed(method: RoadMethod, transparency: Chosen) -> RoadMethod:
    return replace(method, substrate=TransparentSubstrate(base=method.substrate,
                                                          transparency=transparency))


def _share_displaced(block: Block, roads: gpd.GeoDataFrame, members: tuple[int, ...]) -> float:
    """The share of the network's TRUE displacement that falls on `members`."""
    c = block.buildings.displacement(road_corridor(roads))
    return float(c[list(members)].sum() / c.sum())


def _share_crossed(block: Block, roads: gpd.GeoDataFrame, members: tuple[int, ...]) -> float:
    """Of the road centrelines' length inside any footprint, the share inside `members`'."""
    lines = shapely.union_all(np.asarray(roads.geometry))
    outlines = np.asarray(block.buildings.outlines)
    inside_any = shapely.intersection(lines, shapely.union_all(outlines)).length
    return float(shapely.intersection(lines, shapely.union_all(outlines[list(members)])).length
                 / inside_any)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_a_price_discount_moves_displacement_onto_the_discounted(name: str) -> None:
    """At a strong discount on a corridor of buildings, more of the network's true displacement
    falls on the corridor: the price reaches what each method chooses. Measured: cycle_native
    0.21 -> 0.64, resistance_lp 0.29 -> 0.47, clearance 0.06 -> 0.26.

    FAULT INJECTION: ranking by the true increment (`inc.true` for `inc.priced` in cycle_native's
    cost), restoring resistance_lp's plain budget (coefficient 1.0 in `solve_coverage_lp`'s budget
    row and `.true` in `_round`), or dropping the factor from clearance's building term
    (`factors[nearest]` -> ones in `_node_clearance`) fails that method's case."""
    block = _block()
    members = _corridor(block)
    method = METHODS[name]
    opaque = method.propose(block).roads
    priced = replace(method, price=Chosen(members, STRONG)).propose(block).roads
    assert opaque is not None and priced is not None
    assert _share_displaced(block, priced, members) > _share_displaced(block, opaque, members) + 0.1


@pytest.mark.parametrize("name", ["clearance", "cycle_native"])
def test_a_route_discount_moves_displacement_onto_the_discounted(name: str) -> None:
    """The same through routing alone, price opaque: routes run through the cheap corridor and
    displace it. Measured: cycle_native 0.21 -> 0.50, clearance 0.06 -> 0.32.

    FAULT INJECTION: `TransparentSubstrate.build` returning the base graph unchanged fails both."""
    block = _block()
    members = _corridor(block)
    method = METHODS[name]
    opaque = method.propose(block).roads
    routed = _routed(method, Chosen(members, STRONG)).propose(block).roads
    assert opaque is not None and routed is not None
    assert _share_displaced(block, routed, members) > _share_displaced(block, opaque, members) + 0.1


def test_a_route_discount_moves_resistance_lps_roads_into_the_discounted() -> None:
    """resistance_lp routes its candidates on the substrate too: under the route alone its roads
    run through the corridor's footprints more (their share of road length inside footprints 0.21
    -> 0.37). Its share of displacement on the corridor FALLS (0.29 -> 0.15): its budget prices true
    displacement, so a candidate routed through buildings costs it more and it builds elsewhere.
    The route moves it only with the price (0.29 -> 0.50 with both).

    FAULT INJECTION: `TransparentSubstrate.build` returning the base graph unchanged fails this."""
    block = _block()
    members = _corridor(block)
    method = METHODS["resistance_lp"]
    opaque = method.propose(block).roads
    routed = _routed(method, Chosen(members, STRONG)).propose(block).roads
    assert opaque is not None and routed is not None
    assert _share_crossed(block, routed, members) > _share_crossed(block, opaque, members) + 0.1


# ------------------------------------------------------------- clearance: the building term


def test_clearances_building_term_is_scaled_by_the_nearest_buildings_factor() -> None:
    """Node cost (1 - t) + t * f / clearance, f that of the building the clearance is measured to:
    an edge near building 1 alone (f 0.25) keeps its length term and quarters its building term,
    and building 0's factor does not reach it.

    FAULT INJECTION: an off-by-one in the nearest building's index -- `factors[nearest - 1]` in
    `_node_clearance` -- fails this."""
    pts = np.array([[20.0, 0.0], [30.0, 0.0]])
    rows, cols = np.array([0, 1]), np.array([1, 0])
    edist = np.array([10.0, 10.0])
    buildings = np.array([[-50.0, 0.0], [25.0, 2.0]])     # building 0 far, building 1 beside it
    radii = np.zeros(2)
    t = _sigmoid(1.0)

    def weight(f0: float, f1: float) -> float:
        w = _edge_weights(pts, rows, cols, edist, buildings, radii, t,
                          factors=np.array([f0, f1]))
        assert w[0] == w[1]
        return float(w[0])

    length_term = 10.0 * (1.0 - t)
    assert (weight(1.0, 0.25) - length_term
            == pytest.approx(0.25 * (weight(1.0, 1.0) - length_term)))
    assert weight(0.25, 1.0) == weight(1.0, 1.0)


# ------------------------------------------------------------- the label carries the price

# Today's labels, which an opaque run keeps: they name render files.
LABELS = {
    "clearance": "clearance:chord_diag:r0:d2:mr400",
    "clearance_looped": "loop_closure:clearance:chord_diag:r0:d1:mr400:bf0.12:ml400",
    "cycle_native": "cycle_native:chord_diag:d0.2:mc400",
    "resistance_lp": "resistance_lp:chord_diag:d0.2:k8",
}


def _arm(name: str, price: Transparency) -> Method:
    if name == "clearance_looped":       # the loop stage has no price: its base tree does
        return replace(LOOPS, base=replace(CLEARANCE, depth_target=1, price=price))
    return replace(METHODS[name], price=price)


@pytest.mark.parametrize("name", sorted(LABELS))
def test_a_priced_run_is_labelled_apart_from_the_opaque_one(name: str) -> None:
    """A label names render files, so it carries what changes the road set: an opaque run keeps
    today's label, and a priced run's differs by the price's tag.

    FAULT INJECTION: dropping `{self.price.tag}` from cycle_native's, resistance_lp's or
    clearance's `proposal_id` fails that method's case; clearance's fails clearance_looped's too,
    whose label carries its base's."""
    block = _block()
    price = Chosen(_corridor(block), STRONG)
    assert _arm(name, Opaque()).propose(block).proposal_id == LABELS[name]
    priced = _arm(name, price).propose(block).proposal_id
    assert priced != LABELS[name] and price.tag in priced
