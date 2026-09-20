# What the browser can actually solve, measured

2026-09-20. Prompted by asking whether the `Explore` page could follow the 6,619-parcel Cape Town
block (`ZAF.9.3.1_1_5810`) instead of the 263-parcel one it pins (`ZAF.9.3.1_1_40972`).

## The numbers

| | 263 parcels | 6,619 parcels |
|---|---|---|
| native `solve_egress(block, road, params)` | 22.1 ms | **590 ms** |
| Pyodide `runtime.solve(road)` | **52 ms** (median of 5) | ~1.4 s *(extrapolated)* |
| authoring bundle on disk | 132 KB | ~3.3 MB *(extrapolated)* |
| cold Pyodide boot | 10.9 s | unchanged — boot is parcel-independent |

**Pyodide costs 2.35× native** on this workload (52 ms against 22.1 ms, same call, same block).
The solve is near-linear in parcels: 25× the parcels for 26.7× the time, which is what a sparse
egress solve should do. Both extrapolations apply those two measured factors; neither is measured
at 6,619 directly, and baking an authoring bundle for that block is what would remove the
assumption.

**Read these as an order of magnitude.** One machine, Node rather than a browser, and the
2.35× factor was measured only at 263 parcels — WASM memory pressure could move it at 26× the
working set. The decision does not hinge on the precision: even at double the estimate, ~2.8 s is
usable for an action the reader deliberately triggers.

## Why the first measurement was wrong, which is the reusable part

The obvious comparison — `permeability(block, None)` natively, 92 ms at 263 parcels — made Pyodide
look **faster than native**, 52 ms against 92 ms. It is not. Two mistakes in one number:

* `roads=None` is a different code path from the one the widget calls. `solve.py` calls
  `solve_egress(block, roads, params)` with a real road.
* The block the widget solves is rebuilt by `block_from_bundle` from a bundle that already
  carries the road-INVARIANT half of the egress graph — parcel centroids, the footpath edge list,
  which parcels front the street. `gen_authoring_block`'s docstring says so explicitly, and it is
  the whole reason the bundle has that shape. A block built from `KblockSource` carries none of
  it, so the native call was paying for parcel adjacency that the browser never recomputes.

A cross-runtime ratio is only meaningful when both sides run the same call on the same inputs, and
"the same call" here was not obvious from either side's signature.

## What it means for the swap

Feasible. ~1.4 s per solve for an action the reader triggers deliberately — the widget solves on
commit (Enter or double-click), not on drag, and already carries two throttles plus an in-flight
guard, so a slower solve degrades to lag rather than breaking.

The bundle is the part that cannot be shrunk the usual way: it carries FULL float64 absolute
coordinates by design, because quantising to centimetres moves the answer by 1.66e-03 while the
parity tolerance is 1e-15. ~3.3 MB is large for a page asset but not out of band here —
`examples/screen-map/capetown.json` is 6.2 MB and already ships.

The pin is one line, `PINNED_VARIANT` in `scripts/_example_block.py`, shared by PermGraph,
Frontier, DisplacementField and RegionGrow, with ScreenMap deriving its `follow` from perm-graph's
bundle. So the swap moves all five stages together, which is what "five stages, one block" means —
it is not a per-widget choice.

*Not measured, and it is the one that would decide it:* the bundle baked at 6,619 parcels, solved
under Pyodide. Both figures above for that column are products of two measured factors, not
observations.
