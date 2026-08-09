"""AccessBurdenEval: the access DEFICIT a proposal removes, as sum of squared depths.

`KComplexityEval` already reports max peel depth (`k_before`/`k_after`/`delta_k`). Max depth is the
literature's block complexity and it is the right STANDARD, but it is a terrible RANKER on this
corpus: blocks start at k = 3-5, so there are only a few integers to move through, and methods tie
constantly. Measured at 7 m and a 10% displacement budget, four of five methods reported k0 = 2.0
while their burdens spanned 0.89-1.46
(`notes/2026-08-08-c3-the-access-curve-crosses-and-burden-is-the-statistic.md`).

So this reports the distribution instead of its maximum:

    burden = sum_i (depth_i - 1)^2 / n_parcels

**Zero-indexed, and that is the whole point.** `parcel_access_layers` returns 1 for a parcel that
fronts a street, so the shipped `budget.access_burden` (sum of depth^2, retired as a reported metric
in the 2026-07-22 permeability consolidation) scored a PERFECT block at `n`, not 0. Subtracting one
first makes it a genuine deficit measure: **0 exactly when every parcel fronts a street**, which is
`k = 1`, universal street access, the objective the Brelsford/Bettencourt line treats as the
definition of reblocking.

`burden_reduction = 1 - after/before` is the normalized headline, deliberately the same shape as
`permeability` (`1 - P(roads)/P(no_roads)`) so the two axes read alike. It is monotone
non-decreasing in the road set for the same reason: adding roads can only shrink depths, verified on
163/163 (block, method) prefix series in
`notes/2026-08-08-c5-the-access-objective-was-never-wired-up-and-it-wins.md`.

## Why this axis and not just permeability

They diverge, which is what makes both worth reporting. `euclidean_grid` is mid-pack under
permeability and LAST on access at every budget and road width; `resistance_lp`, a permeability
optimizer, is the best access method at 7 m. An axis that only ever agreed would be redundant.

## What it does NOT do

No prefix sweep and no budget. This scores the proposal it is handed, exactly like every other Eval.
Curves and matched-displacement lenses live in `scripts/compare_budgets.py`, which is where
permeability's lenses live too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reblock.contracts import Block, Metrics, Proposal
from reblock.derivations import access_after, access_before


def burden(depths: pd.Series) -> float:
    """`sum (depth - 1)^2 / n` -- 0 iff every parcel fronts a street. See the module docstring for
    why the shift is load-bearing (the un-shifted form scores a perfect block at n, not 0)."""
    if len(depths) == 0:
        return 0.0
    d0 = depths.to_numpy(dtype=float) - 1.0
    return float((d0 ** 2).sum() / len(d0))


class AccessBurdenEval:
    """Emits the zero-indexed access burden before/after a proposal, its normalized reduction, and
    the two companion shape statistics (deepest parcel, share still off-street)."""

    def score(self, block: Block, proposal: Proposal) -> Metrics:
        pre = access_before(block)
        post = access_after(block, proposal)
        b_pre, b_post = burden(pre), burden(post)
        # 1 - after/before, mirroring permeability. A block already at universal access has nothing
        # to reduce: report 0.0 rather than a division by zero, since no road can improve on it.
        reduction = (1.0 - b_post / b_pre) if b_pre > 0.0 else 0.0
        after = post.to_numpy(dtype=float)
        return Metrics(
            block_id=block.block_id, method=proposal.method, eval="access_burden",
            values={
                "burden_before": b_pre,
                "burden_after": b_post,
                "burden_reduction": reduction,
                # k - 1, so 0 == universal street access, matching the burden's zero point
                "k0_after": float(np.max(after) - 1.0) if len(after) else 0.0,
                "share_deficient_after": float((after > 1.0).mean()) if len(after) else 0.0,
            },
            fields={"access_before": pre, "access_after": post})
