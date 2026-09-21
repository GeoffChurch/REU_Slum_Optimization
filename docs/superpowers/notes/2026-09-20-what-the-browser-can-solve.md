# What the browser can actually solve, measured

2026-09-20. Prompted by asking whether the `Explore` page could follow the 6,619-parcel Cape Town
block (`ZAF.9.3.1_1_5810`) instead of the 263-parcel one it pins (`ZAF.9.3.1_1_40972`).

## The numbers

**Superseded 2026-09-20, same day.** The table below was an extrapolation and it was wrong in the
way that mattered: it predicted ~1.4 s for the 6,619-parcel solve, and the real answer was that the
solve *did not complete at all*. What it actually did, under Pyodide, was kill the interpreter --
`NoGilError: Attempted to use PyProxy when Python GIL not held`, raised from inside shapely's
`intersection`, down `solve_egress -> footpath_mesh -> parcel_adjacency`.

| | 263 parcels | 6,619 parcels |
|---|---|---|
| native `solve_egress`, adjacency recomputed | 22.1 ms | 593.9 ms |
| native `solve_egress`, adjacency **passed** | — | **192.9 ms** |
| Pyodide `runtime.solve` | 52 ms (measured) | measured only as a **fatal error**, since fixed |
| authoring bundle on disk | 132 KB | 3.2 MB (predicted 3.3) |
| cold Pyodide boot | 10.9 s | unchanged — boot is parcel-independent |

The cause was not scale. `block_from_bundle` rebuilt the `Block` from geometry alone and `solve()`
called `solve_egress` without `adj`, so every solve recomputed the parcel adjacency -- an STRtree
query plus a shapely `snap().intersection()` per candidate pair -- while the bundle had been
carrying that exact adjacency all along, baked so the browser would not have to. 19,443 edges in
the bundle; 19,443 from `parcel_adjacency`. Nothing read them.

Passing it is **68% of the solve** on this block and a 3x speedup on the small one too. The bundle
size prediction was the one thing the extrapolation got right.

**What to take from this:** the bundle-size column extrapolated cleanly because bytes scale with
parcels. The timing column did not, because it was extrapolating a code path that should never
have run, on a runtime whose failure mode at scale is a crash rather than a slowdown. A ratio
measured at one size predicts the next size only if the algorithm is the same at both, and
"recompute the adjacency every call" stops being an algorithm and starts being a bug somewhere
between 263 and 6,619.

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
