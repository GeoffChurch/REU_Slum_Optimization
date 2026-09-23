"""The GW transport and shape signature the transplant benchmarks run at: one each, named.

Every published OT result (docs/superpowers/notes/2026-07-2[3-8]-*) was measured at these values.
`scripts/pair_matrix.py` reads them here; the donor presets every donor-driven method is configured
with (`conf/donors/_selection.yaml`) spell the same numbers, and
`tests/transplant/test_donors.py::test_the_donor_presets_run_at_the_published_operating_point`
fails if the two drift.
"""
from __future__ import annotations

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
