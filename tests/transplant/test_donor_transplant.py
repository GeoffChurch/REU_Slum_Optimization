"""The single-donor transplant as a Method: the closest donor's footpaths, routed and stamped."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import geopandas as gpd
from hydra import compose, initialize_config_dir

from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width
from reblock.presets import load_method
from reblock.transplant.donor_transplant import DonorTransplantReblocker
from reblock.transplant.donors import Donors
from reblock.transplant.snap import RoutedSnap
from tests.transplant.pool_fixtures import SIGNATURE, TRANSPORT, pool, slab

RECIPIENT = slab(6, 5, "r", x=0.0, source_hash="t")
DONORS = (slab(6, 6, "a", x=3000.0, source_hash="t"), slab(7, 5, "b", x=4000.0, source_hash="t"),
          slab(5, 6, "c", x=5000.0, source_hash="t"))
BLOCKS = (RECIPIENT, *DONORS)
METHOD = DonorTransplantReblocker(
    donors=Donors(pool=pool(BLOCKS, with_paths=BLOCKS, content=("test", "transplant")),
                  exclusion_radius_m=2000.0, k=3, min_donors=1, signature=SIGNATURE,
                  transport=TRANSPORT),
    snap=RoutedSnap(ChordSubstrate()), road_width_m=DEFAULT_ROAD_WIDTH_M)


def test_the_closest_donor_is_routed_onto_the_recipient_and_stamped() -> None:
    """Of the fitted donors, the smallest GW distance -- not the nearest signature, which ranks
    them -- and nothing of the others."""
    fits = METHOD.donors.fits(RECIPIENT)
    best = min(fits, key=lambda f: f.gw_dist)
    proposal = METHOD.propose(RECIPIENT)
    expected = with_width(RoutedSnap(ChordSubstrate()).snap(best.transported, RECIPIENT),
                          DEFAULT_ROAD_WIDTH_M)
    roads = cast(gpd.GeoDataFrame, proposal.roads)
    assert len(roads) and [g.wkb for g in roads.geometry] == [g.wkb for g in expected.geometry]
    assert (roads["width_m"] == DEFAULT_ROAD_WIDTH_M).all()
    assert proposal.params["donor"] == best.donor.block_id
    assert proposal.params["donors"] == 3


def test_two_donor_draws_on_one_block_do_not_share_a_proposal_id() -> None:
    """The eval cache keys on (block, proposal_id), and held-out and leaky draw different donors."""
    leaky = replace(METHOD, donors=replace(METHOD.donors, exclusion_radius_m=0.0))
    a, b = METHOD.propose(RECIPIENT), leaky.propose(RECIPIENT)
    assert a.identity is not None and a.proposal_id != b.proposal_id


def test_the_preset_is_the_benchmarks_single_donor_arm(offline_city_cache: Path) -> None:
    """Routed along the recipient's own gaps (not straight between snapped nodes), as streets."""
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="config", overrides=["method=donor_transplant"])
    method = load_method(cfg.method)
    assert isinstance(method, DonorTransplantReblocker)
    assert method.snap == RoutedSnap(ChordSubstrate())
    assert method.road_width_m == DEFAULT_ROAD_WIDTH_M
