"""The research pools: which blocks land in which role, what the Cape Town pool is composed of, and
how donor material is fetched and classified."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.data.counts import OpenBuildingsCount
from reblock.data.kblock import KblockSource
from reblock.data.pools import (
    PBF_BY_ISO,
    DonorSkip,
    PoolSpec,
    ScreenedPool,
    _source_key,
    evenly_spaced,
    fetch_donor_lines,
    iso_of,
    load_pools,
)
from reblock.metric import DENSITY_COMPACTNESS_FLOOR, AbsoluteGate
from reblock.presets import Stages, load_research, load_stages
from reblock.region import IdentityRegionBuilder
from reblock.screen.dense_compact import DenseCompactScreen
from reblock.screen.identity import IdentityScreen

ROOT = Path(__file__).resolve().parents[2]
CT_BLOCKS = ROOT / "tests" / "data" / "kblock" / "blocks_capetown_sample.parquet"
CT_BLD = ROOT / "tests" / "data" / "kblock" / "buildings_capetown_sample.parquet"
UTM = CRS.from_epsg(32734)

# Stored building counts in the fixture: 3934 = 124, 4310 = 94, 4341 = 203 (all inside the
# [60, 300] compute band), 3968 = 388 (above it). Real parcels after the point join: 211, 147, 276.
IN_BAND_SCREENED = "ZAF.9.1.2_1_3934"
BOTH_ROLES = "ZAF.9.1.4_1_4310"
DONOR_ONLY = "ZAF.9.1.4_1_4341"
ABOVE_BAND = "ZAF.9.1.2_1_3968"


def _census(tmp_path: Path) -> Path:
    pd.DataFrame({
        "block_id": [BOTH_ROLES, DONOR_ONLY, ABOVE_BAND, "ZAF.9.3.1_1_1435", "ZAF.9.3.1_1_1455"],
        "census_failed": [False, False, False, True, False],
        "interior_length_m_0.5": [150.0, 100.0, 500.0, 900.0, 99.0],
    }).to_parquet(tmp_path / "osm_coverage_ZAF.parquet")
    return tmp_path


def _spec(tmp_path: Path) -> PoolSpec:
    source = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", min_buildings=10,
                          block_ids=None, building_tier=SpacingDiscs, member_buildings=None)
    screen = IdentityScreen([IN_BAND_SCREENED, BOTH_ROLES, ABOVE_BAND])
    return PoolSpec(stages=Stages(source=source, screen=screen,
                                  region_builder=IdentityRegionBuilder()),
                    census_dir=_census(tmp_path), min_building_count=60, max_building_count=300,
                    min_parcels=50, min_interior_m=100.0)


def test_roles_are_screened_recipients_and_census_donors_inside_the_band(
        tmp_path: Path) -> None:
    """A block can hold either role or both. Out of band, failed in the census, or short of
    interior footpath, it holds neither -- whatever else is true of it.

    FAULT INJECTION: drawing donors from the screen's flags (the pre-decoupling pool) drops
    `DONOR_ONLY` and fails the donor assertion.
    """
    pools = load_pools(_spec(tmp_path))
    ids = [b.block_id for b in pools.blocks]
    assert ids == sorted([IN_BAND_SCREENED, BOTH_ROLES, DONOR_ONLY])
    assert {ids[i] for i in pools.recipients} == {IN_BAND_SCREENED, BOTH_ROLES}
    assert {ids[i] for i in pools.donors} == {BOTH_ROLES, DONOR_ONLY}
    assert list(pools.blocks_gdf["block_id"]) == ids
    assert pools.blocks_gdf.crs == pools.blocks[0].crs


def test_a_block_short_of_real_parcels_is_dropped_from_both_roles(tmp_path: Path) -> None:
    """The stored count is a proxy for the building-point join (4310 stores 94 buildings and
    tessellates into 147 parcels); the parcel floor is on the join."""
    pools = load_pools(replace(_spec(tmp_path), min_parcels=150))
    assert {b.block_id for b in pools.blocks} == {IN_BAND_SCREENED, DONOR_ONLY}


def test_a_screen_that_selects_nothing_of_its_own_flags_everything(tmp_path: Path) -> None:
    """`Screen.select` returning None means "all blocks": every in-band block is a recipient,
    including one the list screen above never flagged."""
    spec = _spec(tmp_path)
    pools = load_pools(replace(spec, stages=replace(spec.stages, screen=IdentityScreen(None)),
                               min_parcels=250))
    assert DONOR_ONLY in {pools.blocks[i].block_id for i in pools.recipients}


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(config_name=config_name, overrides=overrides)


def _pool(*overrides: str) -> ScreenedPool:
    return load_research(_compose("config", list(overrides)).donor_pool, ScreenedPool)


def test_the_capetown_pool_is_the_shipped_density_compactness_screen(
        offline_city_cache: Path) -> None:
    """The pool's stages are spelled in conf/donor_pool/, because the pool composes nothing; they
    must be exactly what the shipped groups build -- the stages this pool was once composed from,
    `compare_config` under these overrides. A hand-built `Gate(kind="absolute", ...)` is how the
    pool broke silently once, when `Gate` became a Protocol.

    FAULT INJECTION: `min_buildings: 10` in the pool's screen (conf/donor_pool/_screened.yaml), or
    `min_buildings: 10` in its source (capetown.yaml), each fails the comparison below.
    """
    spec = _pool("donor_pool=capetown").spec
    shipped = load_stages(_compose("compare_config", [
        "data=capetown_full", "data.min_buildings=30", "buildings=spacing",
        "screen=dense_compact", "screen.min_buildings=30", "metric=density_compactness",
        "building_count=open_buildings", "proxy_keep_n=1000", "region_builder=identity"]))
    screen, want = spec.stages.screen, shipped.screen
    assert isinstance(screen, DenseCompactScreen) and isinstance(want, DenseCompactScreen)
    assert screen.metric == want.metric and screen.metric.name == "density_compactness"
    assert screen.gate == want.gate == AbsoluteGate(value=DENSITY_COMPACTNESS_FLOOR)
    assert (screen.proxy_keep_n, screen.min_buildings) == (
        want.proxy_keep_n, want.min_buildings) == (1000, 30)
    assert screen.counts == want.counts and isinstance(screen.counts, OpenBuildingsCount)
    assert spec.stages.region_builder == shipped.region_builder == IdentityRegionBuilder()
    source, want_source = spec.stages.source, shipped.source
    assert isinstance(source, KblockSource) and isinstance(want_source, KblockSource)
    assert _source_key(source) == _source_key(want_source)
    assert (source.region_id, source.min_buildings) == ("capetown", 30)
    assert (spec.census_dir, spec.min_building_count, spec.max_building_count, spec.min_parcels,
            spec.min_interior_m) == (offline_city_cache, 60, 300, 50, 100.0)


def test_the_zone_pool_is_the_capetown_screen_over_one_shortlist_zone(
        offline_city_cache: Path) -> None:
    """Same screen, region builder, census and bounds; only the source differs."""
    zone = _pool("donor_pool=shortlist_zone", "donor_pool.spec.stages.source.epsg=32735").spec
    capetown = _pool("donor_pool=capetown").spec
    zs, cs = zone.stages.screen, capetown.stages.screen
    assert isinstance(zs, DenseCompactScreen) and isinstance(cs, DenseCompactScreen)
    assert (zs.metric, zs.gate, zs.proxy_keep_n, zs.min_buildings, zs.counts) == (
        cs.metric, cs.gate, cs.proxy_keep_n, cs.min_buildings, cs.counts)
    assert replace(zone, stages=capetown.stages) == capetown
    source = zone.stages.source
    assert isinstance(source, KblockSource)
    assert source.blocks_path == offline_city_cache / "blocks_shortlist_z32735.parquet"
    assert (source.region_id, source.min_buildings) == ("shortlist-z32735", 30)


def test_a_pool_composes_no_config() -> None:
    """A pool is handed built stages. Composing inside it is what broke every donor-driven method
    under `@hydra.main` ("GlobalHydra is already initialized"), so the research modules may not
    import Hydra's composition at all.

    FAULT INJECTION: re-adding `from hydra import compose, initialize_config_dir` to
    `reblock/data/pools.py` fails this.
    """
    research = [ROOT / "src" / "reblock" / "data" / "pools.py",
                *(ROOT / "src" / "reblock" / "transplant").glob("*.py")]
    composing = []
    for path in research:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module in ("hydra", "hydra.initialize",
                                                                   "hydra.compose"):
                composing.append(path.name)
    assert composing == []


def test_evenly_spaced_keeps_both_extremes() -> None:
    key = [5.0, 1.0, 9.0, 3.0, 7.0]
    assert evenly_spaced([0, 1, 2, 3, 4], key, 3) == [1, 0, 2]
    assert evenly_spaced([0, 1], key, 5) == [1, 0]


def _ids(*block_ids: str) -> list[Block]:
    """Block stand-ins carrying only what `iso_of` reads."""
    return cast("list[Block]", [SimpleNamespace(block_id=b) for b in block_ids])


def test_iso_of_picks_the_extract_from_the_block_ids() -> None:
    """A PBF covers exactly its own extract, so a Kenyan pool pointed at the South Africa file
    does not error -- every donor comes back with no interior footpaths. The first Nairobi run
    reported `empty_interior: 90` and zero pairs, which reads as a fact about Nairobi and is
    contradicted by the census (56 of those blocks carry >=100 m each). Re-run against the Kenya
    extract it produced 500 pairs and 7 skips. Derive the extract from the data, never by hand."""
    assert iso_of(_ids("ZAF.9.3.1_1_44882", "ZAF.9.1_1_1")) == "ZAF"
    assert iso_of(_ids("KEN.1.1_1_100")) == "KEN"
    assert PBF_BY_ISO["KEN"] != PBF_BY_ISO["ZAF"]

    with pytest.raises(ValueError, match="spans multiple countries"):
        iso_of(_ids("ZAF.9.3.1_1_44882", "KEN.1.1_1_100"))
    with pytest.raises(ValueError, match="no Geofabrik extract"):
        iso_of(_ids("BRA.1_1_1"))


def _square() -> Block:
    parcel = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    return Block(block_id="sq", crs=UTM, boundary=parcel,
                 parcels=gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[parcel], crs=UTM),
                 streets=gpd.GeoDataFrame(geometry=[parcel.exterior], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(geometry=[], crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


class _Flaky:
    """A FootpathSource that fails `failures` times before returning `lines`."""

    identity = None

    def __init__(self, lines: gpd.GeoDataFrame, failures: int) -> None:
        self._lines, self.failures, self.calls = lines, failures, 0

    def footpaths(self, bbox: tuple[float, float, float, float],
                  crs: CRS) -> gpd.GeoDataFrame:
        del bbox
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("flaky")
        return self._lines.to_crs(crs)


def test_fetch_classifies_what_it_gets() -> None:
    """Interior material comes back widthless -- an alignment, not a road -- and each way of
    getting none is named, so a run can count it rather than silently thin its pool."""
    block = _square()
    inside = gpd.GeoDataFrame(geometry=[LineString([(10, 0), (10, 15)])], crs=UTM)
    along = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs=UTM)

    got = fetch_donor_lines(_Flaky(inside, failures=2), block, base_backoff_s=0.0)
    assert isinstance(got, gpd.GeoDataFrame) and len(got) == 1
    assert "width_m" not in got.columns
    assert fetch_donor_lines(_Flaky(along, 0), block, base_backoff_s=0.0) is (
        DonorSkip.EMPTY_INTERIOR)
    assert fetch_donor_lines(_Flaky(inside, failures=4), block, max_tries=4,
                             base_backoff_s=0.0) is DonorSkip.FETCH_FAILED
