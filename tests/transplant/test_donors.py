"""Donor selection: who a recipient draws, in what order, and that each (recipient, donor) fit is
paid for once."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from pyproj import CRS

import reblock.transplant.donors as donors_module
from reblock.contracts import Block
from reblock.data.pools import CapetownPool
from reblock.presets import load_research
from reblock.transplant.donors import Donors, TooFewDonors
from reblock.transplant.gw import Arr
from reblock.transplant.operating_points import SIGNATURE as PUBLISHED_SIGNATURE
from reblock.transplant.operating_points import TRANSPORT as PUBLISHED_TRANSPORT
from reblock.transplant.signature import signature, signature_distance
from reblock.transplant.transport import Transport, TransportParams, fit_transport, parcel_xy
from tests.transplant.pool_fixtures import SIGNATURE, TRANSPORT, pool, slab

RECIPIENT = slab(6, 4, "r", x=0.0)
TOUCHING = slab(4, 4, "touching", x=60.0)          # shares the recipient's east edge
NEAR = slab(5, 4, "near", x=500.0)
FAR_A = slab(6, 5, "far_a", x=3000.0)
FAR_B = slab(4, 6, "far_b", x=4000.0)
FAR_C = slab(7, 3, "far_c", x=5000.0)
BLOCKS = (RECIPIENT, TOUCHING, NEAR, FAR_A, FAR_B, FAR_C)


def _donors(radius: float, k: int, *, min_donors: int = 1,
            with_paths: Sequence[Block] = BLOCKS) -> Donors:
    return Donors(pool=pool(BLOCKS, with_paths=with_paths), exclusion_radius_m=radius, k=k,
                  min_donors=min_donors, signature=SIGNATURE, transport=TRANSPORT)


def _ids(blocks: Sequence[Block]) -> list[str]:
    return [b.block_id for b in blocks]


def test_the_recipient_and_everything_within_the_radius_are_ineligible() -> None:
    """Strictly beyond the radius, so even the leaky arm (0 m) excludes a touching neighbour --
    `data.settlements.exclusion_holdout`'s rule."""
    assert _ids(_donors(2000.0, 15).eligible(RECIPIENT)) == ["far_a", "far_b", "far_c"]
    assert _ids(_donors(0.0, 15).eligible(RECIPIENT)) == ["far_a", "far_b", "far_c", "near"]


def test_donors_come_nearest_signature_first_and_only_with_footpaths() -> None:
    """FAULT INJECTION: dropping the footpath check lets `far_b`, which carries none, in."""
    donors = _donors(0.0, 15, with_paths=(FAR_A, FAR_C, NEAR))
    r_sig = signature(parcel_xy(RECIPIENT), SIGNATURE)
    expected = sorted([FAR_A, FAR_C, NEAR], key=lambda b: signature_distance(
        signature(parcel_xy(b), SIGNATURE), r_sig))
    fits = donors.fits(RECIPIENT)
    assert _ids([f.donor for f in fits]) == _ids(expected)
    assert all(f.transported.crs == RECIPIENT.crs and len(f.transported) for f in fits)


def test_too_few_donors_is_refused_at_every_k() -> None:
    """A k=1 rung must admit exactly the recipients the k=15 arm does, or a sweep compares
    different populations rung to rung.

    FAULT INJECTION: stopping the scan at k (not max(k, min_donors)) refuses the k=1 rung a
    recipient with three eligible donors.
    """
    for k in (1, 15):
        with pytest.raises(TooFewDonors, match="2 eligible donors"):
            _donors(2000.0, k, min_donors=3, with_paths=(FAR_A, FAR_B)).fits(RECIPIENT)
    three = (FAR_A, FAR_B, FAR_C)
    assert len(_donors(2000.0, 1, min_donors=3, with_paths=three).fits(RECIPIENT)) == 1
    assert len(_donors(2000.0, 15, min_donors=3, with_paths=three).fits(RECIPIENT)) == 3


def test_the_rungs_nest_and_share_every_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit does not depend on k: the k=2 rung is the k=3 rung's first two donors, and costs no
    new GW solve -- the derivation cache holds each (recipient, donor) fit.

    FAULT INJECTION: calling `_fit_impl` directly instead of through `derive` makes it 5 solves.
    """
    calls = 0

    def counting(donor_xy: Arr, recipient_xy: Arr, params: TransportParams) -> Transport:
        nonlocal calls
        calls += 1
        return fit_transport(donor_xy, recipient_xy, params)

    monkeypatch.setattr(donors_module, "fit_transport", counting)
    blocks = [replace(b, source_content_hash="nest") for b in BLOCKS]
    shared = pool(blocks, with_paths=blocks, content=("test", "nest"))
    three = Donors(pool=shared, exclusion_radius_m=0.0, k=3, min_donors=1, signature=SIGNATURE,
                   transport=TRANSPORT).fits(blocks[0])
    two = Donors(pool=shared, exclusion_radius_m=0.0, k=2, min_donors=1, signature=SIGNATURE,
                 transport=TRANSPORT).fits(blocks[0])
    assert calls == 3
    assert [(f.donor.block_id, f.gw_dist) for f in two] == [
        (f.donor.block_id, f.gw_dist) for f in three[:2]]


def test_a_recipient_in_another_crs_is_refused() -> None:
    elsewhere = replace(RECIPIENT, crs=CRS.from_epsg(32735))
    with pytest.raises(ValueError, match="would not compare"):
        _donors(0.0, 15).eligible(elsewhere)


def test_the_donor_presets_run_at_the_published_operating_point() -> None:
    """`conf/donors/` spells the transport and signature `scripts/pair_matrix.py` reads from
    `operating_points`; the held-out and leaky draws differ in their radius and nothing else."""
    built = {}
    for preset in ("held_out", "leaky"):
        with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
            cfg = compose(config_name="config", overrides=[f"donors={preset}"])
        built[preset] = load_research(cfg.donors, Donors)
    held_out, leaky = built["held_out"], built["leaky"]
    assert held_out.transport == PUBLISHED_TRANSPORT
    assert held_out.signature == PUBLISHED_SIGNATURE
    assert (held_out.exclusion_radius_m, leaky.exclusion_radius_m) == (2000.0, 0.0)
    assert replace(leaky, exclusion_radius_m=2000.0) == held_out
    assert held_out.pool == CapetownPool(config_dir=Path("conf"))
