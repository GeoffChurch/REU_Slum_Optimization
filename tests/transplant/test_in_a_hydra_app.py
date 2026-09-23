"""A donor-driven method is an ordinary Method: built from the run config the way `reblock.run` and
`reblock.compare` build theirs, and proposing while their Hydra is initialized.

The pool used to materialize by composing Hydra itself, so inside any `@hydra.main` app it raised
"GlobalHydra is already initialized". Here the config is composed once, the method built from it by
the typed loader, and the pool materialized and proposed from INSIDE the still-open Hydra context
-- a pool that composed would raise exactly as it did in the apps.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from shapely.geometry import LineString

import reblock.data.pools as pools_module
from reblock.contracts import Method
from reblock.derivations import propose
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.presets import load_method
from reblock.transplant.consensus import ConsensusDesireSource
from reblock.transplant.donor_transplant import DonorTransplantReblocker
from reblock.transplant.donors import Donors
from tests.transplant.pool_fixtures import Footpaths

ROOT = Path(__file__).resolve().parents[2]
KBLOCK = ROOT / "tests" / "data" / "kblock"
RECIPIENT = "ZAF.9.1.2_1_3934"         # 124 stored buildings, 211 parcels after the join
DONOR = "ZAF.9.1.4_1_4341"             # 203 stored buildings, 276 parcels


@pytest.fixture
def sample_city(offline_city_cache: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The Cape Town sample as the full-city cache, a census naming one donor, and a footpath
    source holding one path through that donor -- everything the capetown preset reads."""
    shutil.copy(KBLOCK / "blocks_capetown_sample.parquet",
                offline_city_cache / "blocks_capetown_full.parquet")
    shutil.copy(KBLOCK / "buildings_capetown_sample.parquet",
                offline_city_cache / "buildings_capetown_full.parquet")
    pd.DataFrame({"block_id": [DONOR], "census_failed": [False],
                  "interior_length_m_0.5": [150.0]}).to_parquet(
        offline_city_cache / "osm_coverage_ZAF.parquet")
    blocks = gpd.read_parquet(offline_city_cache / "blocks_capetown_full.parquet")
    donor = blocks[blocks["block_id"] == DONOR].to_crs(blocks.estimate_utm_crs())
    p = donor.geometry.iloc[0].representative_point()
    path = LineString([(p.x - 15.0, p.y), (p.x + 15.0, p.y)])
    monkeypatch.setattr(pools_module, "pbf_footpaths", lambda iso: Footpaths([path]))
    yield offline_city_cache


def _donors(method: Method) -> Donors:
    if isinstance(method, DonorTransplantReblocker):
        return method.donors
    assert isinstance(method, DemandGreedyReblocker)
    assert isinstance(method.desire_source, ConsensusDesireSource)
    return method.desire_source.donors


@pytest.mark.parametrize("method", ["consensus", "donor_transplant"])
def test_a_donor_method_proposes_inside_an_initialized_hydra(sample_city: Path,
                                                             method: str) -> None:
    """FAULT INJECTION: composing inside `pools._materialize` (an `initialize_config_dir` +
    `compose`, as the pool once did) raises "GlobalHydra is already initialized" here."""
    overrides = [f"method={method}", "donor_pool=capetown", "donors=leaky", "donors.k=1",
                 "donors.min_donors=1", "~donor_pool.spec.stages.screen",
                 f"+donor_pool.spec.stages.screen={{_target_: "
                 f"reblock.screen.identity.IdentityScreen, block_ids: [{RECIPIENT}]}}"]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        built = load_method(compose(config_name="config", overrides=overrides).method)
        blocks = {b.block_id: b for b in _donors(built).pool.pools().blocks}
        assert set(blocks) == {RECIPIENT, DONOR}
        proposal = propose(built, blocks[RECIPIENT])
    roads = proposal.roads
    assert roads is not None and len(roads) > 0
    assert proposal.identity is not None, "a configured pool keeps the proposal cacheable"
