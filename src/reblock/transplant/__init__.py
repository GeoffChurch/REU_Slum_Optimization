"""OT road transplant: carry one block's linework onto another through a shape correspondence.

A research substrate, not a reblocker. The 2026-07-23 arc closed single-donor transplant as a
reblocking method and the 2026-07-28 k-sweep showed the consensus gain is the extraction, not the
averaging (docs/superpowers/notes/2026-07-23-ot-road-transplant.md,
2026-07-28-consensus-k-sweep-and-displacement.md). What ships here is what the benchmark scripts
(`scripts/pair_matrix.py`, `scripts/consensus_matrix.py`, `scripts/consensus_sweep.py`) need to
reproduce and extend those results:

- `gw`        entropic unbalanced Gromov-Wasserstein between two distance matrices
- `transport` parcel-centroid anchors, the fitted donor -> recipient map, and line warping
- `snap`      putting warped linework onto the recipient's own routing substrate
- `signature` a cheap GW-consistent shape descriptor for shortlisting donors
- `consensus` a weighted demand field over several donors, extracted on the recipient's substrate
- `operating_points` the named parameter sets every published result was measured at

No derivation or method module imports this package, so nothing here is in any derivation's code
closure and editing it never invalidates the cache.
"""
