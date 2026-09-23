"""OT road transplant: carry one block's linework onto another through a shape correspondence.

The 2026-07-23 arc closed single-donor transplant as a reblocking method and the 2026-07-28 k-sweep
showed the consensus gain is the extraction, not the averaging -- both under the retired metric
(docs/superpowers/notes/2026-07-23-ot-road-transplant.md,
2026-07-28-consensus-k-sweep-and-displacement.md). What ships here is the two donor-driven methods,
configured like any other (`conf/method/consensus.yaml`, `conf/method/donor_transplant.yaml`), and
what the benchmark scripts (`scripts/pair_matrix.py`, `scripts/consensus_matrix.py`) need:

- `gw`        entropic unbalanced Gromov-Wasserstein between two distance matrices
- `transport` parcel-centroid anchors, the fitted donor -> recipient map, and line warping
- `snap`      putting warped linework onto the recipient's own routing substrate
- `signature` a cheap GW-consistent shape descriptor for shortlisting donors
- `donors`    which donors a recipient draws from a pool, each fitted (cached per pair)
- `consensus` the donors as a weighted desire field, which `demand_greedy` routes toward
- `donor_transplant` the closest donor alone, as a Method
- `operating_points` the transport and signature every published result was measured at

No shipped module imports this package, so nothing here is in any shipped derivation's import
closure; it reaches one only by configuration, and then carries its own code hash in its identity
(`tests/transplant/test_isolation.py`).
"""
