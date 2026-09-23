"""The parameter sets the transplant benchmarks run at: one each, named, shared by every script.

Every published OT result (docs/superpowers/notes/2026-07-2[3-8]-*) was measured at these values,
so a script that wants to be comparable passes these and a script that varies one says so where it
does. Python rather than conf/ because no Hydra entry point reads them.
"""
from __future__ import annotations

from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from reblock.transplant.consensus import ConsensusParams
from reblock.transplant.gw import GWParams
from reblock.transplant.signature import SignatureParams
from reblock.transplant.transport import TransportParams

# eps = 0.01 in POT's convention (see `GWParams`). The 2026-07-23 ablation found eps = 0.05 already
# collapses a transported network to ~3% of its length, and the 2026-07-27 eps ladder found the
# pair-matrix estimate converges by here and dies only at 4x this. tau = 1.0 lets the coupling
# concentrate on well-matched anchors. idw_k = 8 anchors per warped vertex.
TRANSPORT = TransportParams(gw=GWParams(eps=0.01, tau=1.0, outer_iters=30, inner_iters=100),
                            idw_k=8)

# 50 points so a block's signature is always a real 50-parcel subsample -- which is why the pools
# floor blocks at 50 parcels -- averaged over 5 draws.
SIGNATURE = SignatureParams(n_sub=50, n_boot=5, seed=0)

# The 2026-07-23 extraction as the consensus benchmarks ran it: a 3 m corridor (the metric's old
# corridor_m), a 0.05 floor -- 21x cheaper inside a full-agreement corridor than outside any --
# and a complete tree (depth_target=1). `demand_greedy`'s own preset differs on the last two
# (eps 0.1, depth 2); these are the consensus benchmarks' values, not a re-tune.
CONSENSUS = ConsensusParams(substrate=ChordSubstrate(), buffer_m=3.0, eps=0.05, gamma=1.0,
                            depth_target=1, max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M)
