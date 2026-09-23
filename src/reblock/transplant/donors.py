"""Donor selection and fitting: the half every donor-driven method shares.

For a recipient, `Donors.fits`

1. takes the pool's donors strictly farther than `exclusion_radius_m` from it -- 2 km is the
   held-out arm, 0 the leaky one, and the recipient itself never qualifies;
2. ranks them by shape signature (`signature`), nearest first, and keeps the first `k` whose own
   OSM footpaths survive the interior filter (`osm_footpaths.block_footpaths`);
3. fits entropic unbalanced GW from each one's parcel cloud to the recipient's and carries its
   footpaths through the fitted map (`transport`).

The footpath read and the fit are the expensive steps, and both go through the derivation cache
(`derive_graph.derive`), keyed on the blocks' identities, the footpath source's and the transport
parameters. A fit does not depend on `k`, so a k-sweep pays for each (recipient, donor) pair once,
and its rungs nest: k=3's donors are k=15's first three. The ranking's donor signatures go through
it too, as one table per pool and signature parameters, which every draw from that pool shares.

The two donor methods are what is built on these fits: a weighted consensus of all of them
(`consensus.ConsensusDesireSource`, routed by `demand_greedy`) and the closest one alone
(`donor_transplant.DonorTransplantReblocker`).
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
from pyproj import CRS

from reblock.contracts import Block
from reblock.data.pools import DonorPool
from reblock.derive_graph import config_identity, derive
from reblock.methods.osm_footpaths import FootpathSource, block_footpaths
from reblock.transplant.gw import Arr
from reblock.transplant.signature import SignatureParams, signature, signature_distance
from reblock.transplant.transport import (
    TransportParams,
    fit_transport,
    parcel_xy,
    transport_lines,
)


class TooFewDonors(ValueError):
    """The recipient has fewer eligible donors carrying footpaths than `Donors.min_donors`, so no
    donor-driven method proposes for it."""


@dataclass(frozen=True, eq=False)
class DonorFit:
    """One donor's own footpaths carried into the recipient's frame (pre-snap), and how far apart
    the two blocks' shapes are."""

    donor: Block
    footpaths: gpd.GeoDataFrame         # the donor's interior footpaths, donor CRS, widthless
    footpaths_identity: Hashable | None  # their cache address; None when uncacheable
    transported: gpd.GeoDataFrame       # the same lines in the recipient's CRS
    gw_dist: float


@dataclass(frozen=True, eq=False)
class _FootpathsOf:
    """Identified carrier: one block's own interior footpaths, as `source` maps them."""

    source: FootpathSource
    block: Block

    @property
    def identity(self) -> Hashable | None:
        source, block = self.source.identity, self.block.identity
        return None if source is None or block is None else ("footpaths_of", source, block)


def _footpaths_impl(inp: _FootpathsOf) -> gpd.GeoDataFrame:
    return block_footpaths(inp.source, inp.block)


@dataclass(frozen=True, eq=False)
class _Transplant:
    transported: gpd.GeoDataFrame
    gw_dist: float


@dataclass(frozen=True, eq=False)
class _FitInput:
    """Identified carrier: one (recipient, donor) GW fit and the donor's footpaths carried
    through it."""

    recipient: Block
    donor: Block
    footpaths: gpd.GeoDataFrame
    footpaths_identity: Hashable | None
    transport: TransportParams

    @property
    def identity(self) -> Hashable | None:
        recipient, transport = self.recipient.identity, config_identity(self.transport)
        if recipient is None or self.footpaths_identity is None or transport is None:
            return None
        return ("donor_fit", recipient, self.footpaths_identity, transport)


def _fit_impl(inp: _FitInput) -> _Transplant:
    fit = fit_transport(parcel_xy(inp.donor), parcel_xy(inp.recipient), inp.transport)
    return _Transplant(transported=transport_lines(inp.footpaths, fit, crs=inp.recipient.crs),
                       gw_dist=fit.gw_dist)


@dataclass(frozen=True, eq=False)
class _PoolSignatures:
    """Identified carrier: every donor's shape signature in one pool, under one parameter set."""

    pool: DonorPool
    params: SignatureParams

    @property
    def identity(self) -> Hashable | None:
        pool, params = self.pool.identity, config_identity(self.params)
        return None if pool is None or params is None else ("donor_signatures", pool, params)


def _signatures_impl(inp: _PoolSignatures) -> dict[str, Arr]:
    pools = inp.pool.pools()
    return {pools.blocks[j].block_id: signature(parcel_xy(pools.blocks[j]), inp.params)
            for j in pools.donors}


@dataclass(frozen=True)
class Donors:
    """Which donors a recipient draws, and how each is fitted to it."""

    pool: DonorPool
    exclusion_radius_m: float   # a donor must lie strictly beyond this from the recipient
    k: int                      # donors fitted, nearest signature first
    # A recipient with fewer eligible donors carrying footpaths gets no proposal, at ANY k: the
    # scan runs to max(k, min_donors), so a k=1 rung admits exactly the recipients k=15 does.
    min_donors: int
    signature: SignatureParams
    transport: TransportParams

    def __post_init__(self) -> None:
        if not self.exclusion_radius_m >= 0.0:
            raise ValueError(f"exclusion_radius_m must be >= 0, got {self.exclusion_radius_m}")
        if self.k < 1 or self.min_donors < 1:
            raise ValueError(f"k and min_donors must be >= 1, got k={self.k}, "
                             f"min_donors={self.min_donors}")

    @cached_property
    def _signatures(self) -> dict[str, Arr]:
        # Keyed on the pool and the parameters, not on this Donors: a study's held-out and leaky
        # arms, and every k rung, draw from one pool and share its one computation.
        return derive(_signatures_impl, _PoolSignatures(pool=self.pool, params=self.signature))

    def eligible(self, recipient: Block) -> list[Block]:
        """Every pool donor beyond the exclusion radius, in pool order."""
        pools = self.pool.pools()
        if CRS(recipient.crs) != CRS(pools.blocks_gdf.crs):
            raise ValueError(f"recipient {recipient.block_id} is in {recipient.crs}, the pool in "
                             f"{pools.blocks_gdf.crs}: distances and signatures would not compare")
        dist = pools.blocks_gdf.geometry.distance(recipient.boundary)
        return [pools.blocks[j] for j in pools.donors
                if pools.blocks[j].block_id != recipient.block_id
                and float(dist.iloc[j]) > self.exclusion_radius_m]

    def fits(self, recipient: Block) -> list[DonorFit]:
        """The recipient's `k` nearest eligible donors that carry footpaths, nearest signature
        first, each fitted. Raises `TooFewDonors` below `min_donors`."""
        r_sig = signature(parcel_xy(recipient), self.signature)
        ranked = sorted(self.eligible(recipient),
                        key=lambda b: signature_distance(self._signatures[b.block_id], r_sig))
        source = self.pool.footpaths()
        picked: list[_FootpathsOf] = []
        for donor in ranked:
            if len(picked) >= max(self.k, self.min_donors):
                break
            carrier = _FootpathsOf(source=source, block=donor)
            if not derive(_footpaths_impl, carrier).empty:
                picked.append(carrier)
        if len(picked) < self.min_donors:
            raise TooFewDonors(f"{recipient.block_id}: {len(picked)} eligible donors carry "
                               f"footpaths beyond {self.exclusion_radius_m:g} m; need "
                               f"{self.min_donors}")
        return [self._fit(recipient, carrier) for carrier in picked[:self.k]]

    def _fit(self, recipient: Block, carrier: _FootpathsOf) -> DonorFit:
        lines = derive(_footpaths_impl, carrier)
        fit = derive(_fit_impl, _FitInput(recipient=recipient, donor=carrier.block,
                                          footpaths=lines, footpaths_identity=carrier.identity,
                                          transport=self.transport))
        return DonorFit(donor=carrier.block, footpaths=lines,
                        footpaths_identity=carrier.identity, transported=fit.transported,
                        gw_dist=fit.gw_dist)
