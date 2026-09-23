"""The arterial's objective and cost are injected strategies, not strings the engines switch on."""
import dataclasses
from itertools import product
from pathlib import Path

import geopandas as gpd
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from shapely.geometry import LineString, Point

from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import (
    Access,
    ArterialCost,
    ArterialObjective,
    Directness,
    Displacement,
    Efficiency,
    GreedyArterialReblocker,
    Length,
    Repulsion,
    SnapToBoundary,
    engines,
)
from reblock.methods.arterial.primitives import _snap_graph
from reblock.methods.arterial.scoring import step_state
from reblock.methods.boundary_graph import _boundary_graph
from tests.methods.test_arterial import UTM, _grid_block, _grid_block_with_points

OBJECTIVES: tuple[ArterialObjective, ...] = (Access(), Efficiency(), Directness())
COSTS: tuple[ArterialCost, ...] = (Length(), Displacement(), Repulsion())


def test_objectives_and_costs_satisfy_their_protocols() -> None:
    assert all(isinstance(o, ArterialObjective) for o in OBJECTIVES)
    assert all(isinstance(c, ArterialCost) for c in COSTS)
    assert not isinstance("directness", ArterialObjective)
    assert not isinstance("length", ArterialCost)


def test_every_objective_and_cost_pair_has_its_own_cache_key() -> None:
    """Nine configurations, nine identities and nine proposal ids. A strategy that answered with
    another's identity would share that one's cached proposal and eval depths."""
    methods = [GreedyArterialReblocker(objective=o, cost=c, n_anchors=4, max_roads=1)
               for o, c in product(OBJECTIVES, COSTS)]
    assert len({m.identity for m in methods}) == 9
    block = dataclasses.replace(_grid_block(3), source_content_hash="nine-keys")
    assert len({m.propose(block).proposal_id for m in methods}) == 9


def _configured_arterials() -> dict[str, GreedyArterialReblocker]:
    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
        out = {name: instantiate(entry) for name, entry in cfg.all_methods.items()
               if entry["_target_"] == "reblock.methods.arterial.GreedyArterialReblocker"}
        for name in ("greedy_arterial", "greedy_arterial_repulsion",
                     "greedy_arterial_displacement"):
            method_cfg = compose(config_name="config",
                                 overrides=["shapefile=x", f"method={name}"])
            out[f"method={name}"] = instantiate(method_cfg.method)
    return out


def test_every_configured_arterial_injects_strategy_instances() -> None:
    """A string left in a config would reach the reblocker as a `str` and fail only when a proposal
    is first asked for -- hours into a run. Every configured arterial is checked here instead."""
    arterials = _configured_arterials()
    assert len(arterials) >= 9
    for name, m in arterials.items():
        assert isinstance(m, GreedyArterialReblocker), name
        assert isinstance(m.objective, ArterialObjective), (name, m.objective)
        assert isinstance(m.cost, ArterialCost), (name, m.cost)


def test_displacement_prices_a_road_against_the_committed_corridor() -> None:
    """The per-step state the displacement cost closes over is the committed corridor: a candidate
    running along an already-committed road displaces almost nobody NEW, while the same candidate
    over an empty network displaces everyone it touches."""
    pts = gpd.GeoDataFrame(geometry=[Point(i + 0.5, j + 0.5) for i in range(8) for j in range(3)],
                           crs=UTM)
    block = _grid_block_with_points(pts)
    road = LineString([(2.0, 0.0), (2.0, 3.0)])
    half_width_m = 1.0
    empty = Displacement().at_step(block, engines._step_network([], block, half_width_m))
    after = Displacement().at_step(block, engines._step_network([road], block, half_width_m))
    alone = empty.of(road)
    assert alone > 0.0
    assert after.of(road) == pytest.approx(0.0, abs=1e-9)
    # a parallel road half a metre over shares most of the committed corridor
    shifted = LineString([(2.5, 0.0), (2.5, 3.0)])
    assert after.of(shifted) < empty.of(shifted)


def test_step_state_is_frozen() -> None:
    """Forked workers inherit `_STEP_STATE` copy-on-write and only ever read it; frozen makes that
    real rather than a convention."""
    block = _grid_block(3)
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    net = engines._step_network([], block, 1.0)
    st = step_state(block, sg=_snap_graph(_boundary_graph(block.parcels)),
                    realizer=SnapToBoundary(), objective=Directness().for_block(block, adj),
                    cost=Length(), committed=net)
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.cost = Repulsion().at_step(block, net)    # type: ignore[misc]
