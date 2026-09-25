"""DonorTransplantReblocker: the closest donor's real footpaths, carried onto the recipient.

The single-donor arm of the consensus benchmark, as a Method: of the recipient's fitted donors
(`donors.Donors`), the one with the smallest GW distance, its transported footpaths put on the
recipient's substrate by `snap` and stamped as streets.

Kept, not retired. The 2026-07-23 arc measured it Pareto-dominated and the 2026-07-28 k-sweep found
the consensus extraction beat it in every block -- but under the retired commute metric and a
length-matching rule that favoured the consensus (notes/2026-07-28-consensus-k-sweep-and-
displacement.md). Neither holds any more, so it is not measured dominated, and it runs through the
same lenses as every other arm (`scripts/consensus_matrix.py`) until it is.
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass

from reblock.contracts import Block, Proposal
from reblock.derive_graph import closure_hash, config_identity
from reblock.permeability import with_width
from reblock.transplant.donors import Donors
from reblock.transplant.snap import GapSnap


@dataclass(frozen=True)
class DonorTransplantReblocker:
    donors: Donors
    snap: GapSnap           # how the transplant reaches the recipient's substrate
    road_width_m: float     # stamped on every transplanted road

    @property
    def identity(self) -> Hashable | None:
        config = config_identity(self)
        return None if config is None else (closure_hash(__name__), config)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the transplant is block-only
        fits = self.donors.fits(block)
        best = min(fits, key=lambda f: f.gw_dist)
        roads = with_width(self.snap.snap(best.transported, block), self.road_width_m)
        identity = self.identity
        pid = ("donor_transplant" if identity is None else
               f"donor_transplant:{hashlib.sha256(str(identity).encode()).hexdigest()[:8]}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None, roads=roads, proposal_id=pid,
            method="donor_transplant",
            params={"donor": best.donor.block_id, "gw_dist": best.gw_dist, "donors": len(fits),
                    "road_width_m": self.road_width_m})
