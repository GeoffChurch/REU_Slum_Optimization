"""scripts/consensus_matrix.py: every arm is a configured Method, and every arm is scored through
the one shared truncation -- never on its whole network, never on a truncation of its own."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import geopandas as gpd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import scripts.consensus_matrix as study
from reblock.budget import prefix_to_displacement
from reblock.buildings import SpacingDiscs
from reblock.compare import LensPrefixes, lens_prefixes, load_permeability_config
from reblock.contracts import Block
from reblock.data.pools import PoolFootpaths, ScreenedPool
from reblock.emit import pct_displaced
from reblock.eval.agreement import buffered_iou, directional_chamfer
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.methods.osm_footpaths import OsmFootpathsReblocker
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, EgressContext, permeability, with_width
from reblock.presets import load_method
from reblock.transplant.consensus import ConsensusDesireSource
from reblock.transplant.donor_transplant import DonorTransplantReblocker

UTM = CRS.from_epsg(32734)
PCFG = load_permeability_config()


def _slab(w: int, h: int) -> Block:
    """`w` x `h` 10 m parcels, one building each, street frontage along the bottom edge only."""
    polys = [Polygon([(10 * i, 10 * j), (10 * i + 10, 10 * j), (10 * i + 10, 10 * j + 10),
                      (10 * i, 10 * j + 10)]) for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    return Block(block_id="r", crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10 * w, 0)])], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(
                     geometry=[Point(p.centroid.x + 2, p.centroid.y - 1) for p in polys],
                     crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


BLOCK = _slab(6, 5)


def _arms() -> dict[str, gpd.GeoDataFrame]:
    """Three arms of at least four roads each, so no truncation below can equal a whole network."""
    own = with_width(gpd.GeoDataFrame(geometry=[
        LineString([(25, 0), (25, 15)]), LineString([(25, 15), (25, 35)]),
        LineString([(25, 35), (5, 35)]), LineString([(25, 35), (45, 35)])], crs=UTM),
        DEFAULT_ROAD_WIDTH_M)
    arms = {"own": own}
    for name, repulsion in (("clearance", 0.0), ("repelled", 3.0)):
        roads = ClearanceReblocker(substrate=ChordSubstrate(), repulsion=repulsion,
                                   depth_target=1, max_roads=400,
                                   road_width_m=DEFAULT_ROAD_WIDTH_M).propose(BLOCK).roads
        arms[name] = cast(gpd.GeoDataFrame, roads)
    assert all(len(r) >= 4 for r in arms.values())
    return arms


def _length(roads: gpd.GeoDataFrame) -> float:
    return float(roads.geometry.length.sum()) if len(roads) else 0.0


def test_every_arm_is_scored_through_the_one_truncation_and_nothing_else(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """THE guard. `truncate` is replaced by one that hands back sentinel prefixes no real lens
    would pick; every number in every row must then be the sentinel's, and every arm must have
    passed through it exactly once.

    FAULT INJECTION, each fails this:
    - scoring one arm's Lens A permeability on its whole network (`permeability(ctx, arm_roads)`
      for `arm == "repelled"`) -- that arm's `a_permeability` is not the sentinel's;
    - truncating one arm with a function of its own (a `roads.iloc[:k]` length match for
      `arm == "clearance"`, bypassing `truncate`) -- the spy never sees that arm.
    """
    arms = _arms()
    seen: list[str] = []

    def sentinel(ctx: EgressContext, roads: gpd.GeoDataFrame, target_displacement: float,
                 pcfg: object) -> study.Truncations:
        del ctx, pcfg
        seen.append(next(name for name, r in arms.items() if r is roads))
        assert target_displacement == pct_displaced(arms["own"], BLOCK.buildings)
        return study.Truncations(
            lenses=LensPrefixes(displacement=cast(gpd.GeoDataFrame, roads.iloc[:1]),
                                permeability=cast(gpd.GeoDataFrame, roads.iloc[:2]),
                                reached=False),
            prediction=cast(gpd.GeoDataFrame, roads.iloc[:3]))

    monkeypatch.setattr(study, "truncate", sentinel)
    rows = study.score_recipient(BLOCK, arms, {n: {} for n in arms}, "own", PCFG)
    assert sorted(seen) == sorted(arms)

    ctx = EgressContext.of(BLOCK, PCFG.params)
    truth = arms["own"]
    for row in rows:
        roads = arms[row["arm"]]
        a, b, p = (cast(gpd.GeoDataFrame, roads.iloc[:m]) for m in (1, 2, 3))
        assert (row["a_road_m"], row["a_displacement"], row["a_permeability"]) == (
            _length(a), pct_displaced(a, BLOCK.buildings), permeability(ctx, a)), row["arm"]
        assert (row["b_road_m"], row["b_displacement"], row["b_permeability"],
                row["b_reached"]) == (_length(b), pct_displaced(b, BLOCK.buildings),
                                      permeability(ctx, b), False), row["arm"]
        assert (row["p_road_m"], row["p_displacement"], row["p_permeability"]) == (
            _length(p), pct_displaced(p, BLOCK.buildings), permeability(ctx, p)), row["arm"]
        assert (row["iou_3m"], row["iou_10m"]) == (
            buffered_iou(p, truth, r=3.0), buffered_iou(p, truth, r=10.0)), row["arm"]
        assert (row["chamfer_precision_m"], row["chamfer_recall_m"]) == directional_chamfer(
            p, truth), row["arm"]


def test_the_one_truncation_is_the_lenses_compare_budgets_reports_at() -> None:
    """`truncate` is the shipped lens code and Lens A's own function, not a copy of either
    (`tests/test_compare_budgets.py` holds `compare_budgets` to the same `lens_prefixes`).

    FAULT INJECTION: truncating the prediction with the old study's "longest construction-order
    prefix within" instead of `prefix_to_displacement` fails the prediction comparison.
    """
    ctx = EgressContext.of(BLOCK, PCFG.params)
    for roads in _arms().values():
        target = pct_displaced(_arms()["own"], BLOCK.buildings)
        got = study.truncate(ctx, roads, target, PCFG)
        want = lens_prefixes(ctx, roads, PCFG)
        for mine, theirs in ((got.lenses.displacement, want.displacement),
                             (got.lenses.permeability, want.permeability),
                             (got.prediction, prefix_to_displacement(BLOCK, roads, target))):
            assert [g.wkb for g in mine.geometry] == [g.wkb for g in theirs.geometry]
        assert got.lenses.reached == want.reached


def test_a_row_is_scored_end_to_end() -> None:
    """Every column, on today's metric. The reference is scored as an arm too, through the same
    truncation, against its own whole network."""
    arms = _arms()
    rows = study.score_recipient(BLOCK, arms, {n: {"k": 1} for n in arms}, "own", PCFG)
    assert [r["arm"] for r in rows] == list(arms)
    for row in rows:
        assert set(row) == set(study.StudyRow.__annotations__)
        assert row["p_displacement"] >= row["p_target_displacement"] or (
            row["p_road_m"] == _length(arms[row["arm"]]))       # reached, or ran out of road
        assert 0.0 <= row["iou_10m"] <= 1.0 and row["params"] == '{"k": 1}'
    own = next(r for r in rows if r["arm"] == "own")
    assert own["iou_3m"] > 0.9


def _study(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        return compose(config_name="consensus_matrix", overrides=["out=x.parquet", *overrides])


def test_the_pairing_and_the_sweep_are_configuration() -> None:
    """Held-out and leaky are two donor presets on one method; a k rung is the same arm with
    `donors.k` appended. No arm is a code branch."""
    cfg = _study(["k_sweep.arms=[consensus_held_out,single_held_out]", "k_sweep.ks=[1,8]"])
    arms = study.arm_overrides(cfg)
    assert arms["consensus_held_out"] == ["donor_pool=capetown", "method=consensus",
                                          "donors=held_out"]
    assert arms["consensus_leaky"][-1] == "donors=leaky"
    assert arms["single_held_out_k8"] == [*arms["single_held_out"], "donors.k=8"]
    assert {"consensus_held_out_k1", "consensus_held_out_k8", "single_held_out_k1"} <= set(arms)


def test_every_arm_loads_as_a_configured_method(offline_city_cache: Path) -> None:
    """Built through the presets, lazily: nothing here materializes the pool."""
    loaded = study.load_study(_study(["k_sweep.arms=[consensus_held_out]", "k_sweep.ks=[3]"]))
    assert isinstance(loaded.pool, ScreenedPool)
    assert loaded.reference == "own"
    consensus = loaded.arms["consensus_held_out_k3"]
    assert isinstance(consensus, DemandGreedyReblocker)
    assert isinstance(consensus.desire_source, ConsensusDesireSource)
    assert consensus.desire_source.donors.k == 3
    assert (consensus.eps, consensus.depth_target) == (0.05, 1)
    assert isinstance(loaded.arms["single_leaky"], DonorTransplantReblocker)
    assert isinstance(loaded.arms["clearance"], ClearanceReblocker)
    own = loaded.arms["own"]
    assert isinstance(own, OsmFootpathsReblocker) and isinstance(own.source, PoolFootpaths)
    assert isinstance(own.source.pool, ScreenedPool)
    assert own.source.pool.spec.census_dir == loaded.pool.spec.census_dir == offline_city_cache


def test_the_reference_must_be_run(offline_city_cache: Path) -> None:
    with pytest.raises(ValueError, match="must be run"):
        study.load_study(_study(["run=[clearance]"]))


def test_the_consensus_preset_is_the_published_operating_point(
        offline_city_cache: Path) -> None:
    """eps 0.05, depth 1: the consensus benchmarks' extraction, not demand_greedy's own preset."""
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="config", overrides=["method=consensus"])
    method = load_method(cfg.method)
    assert method == DemandGreedyReblocker(
        desire_source=cast(DemandGreedyReblocker, method).desire_source,
        substrate=ChordSubstrate(), buffer_m=3.0, eps=0.05, gamma=1.0, depth_target=1,
        max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M)
