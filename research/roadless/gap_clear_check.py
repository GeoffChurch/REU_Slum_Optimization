"""Guards of the complex as the runtime (gap_clear.py's fixed mesh; NOTES, "The complex as the
runtime: one fixed mesh"), each with its fault injection (a patch in one process, no file edited):
a guard passes on the code as it is and must fail under its fault. Every guard meshes afresh (a
cache of its own, in a temporary directory), so a fault in the meshing is seen.

    S=research/roadless/gap_clear_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S baseline <ids,> [fault]    # faults: interior, exits
    uv run python $S weights <ids,> [fault]     # fault: no-conform
    uv run python $S reach <ids,> [fault]       # fault: threshold
    uv run python $S gradient <ids,> [fault]    # fault: tie
    uv run python $S free <ids,> [fault]        # fault: drop-crossing
    uv run python $S identity - [fault]         # fault: collide
    uv run python $S convex <ids,> [fault]      # fault: multiply
    uv run python $S pattern <ids,> [fault]     # fault: drop-identity
    uv run python $S constants - [fault]        # fault: omit
    uv run python $S device <ids,> [fault]      # fault: perturb (on a GPU: cluster.py --complex)
    uv run python $S roads <ids,> [fault]       # fault: one-face-closed
    uv run python $S flux <ids,> [fault]        # fault: flip-one
    uv run python $S device-roads <ids,> [fault]   # fault: perturb (on a GPU)
    uv run python $S leak-assembly <ids,> [fault]  # faults: drop-one, not-live
    uv run python $S leak-fold <ids,> [fault]      # fault: shares-swapped
    uv run python $S enclosure <ids,> [fault]      # fault: every-cell
    uv run python $S leak-gradient <ids,> [fault]  # fault: adjoint-unleaked
    uv run python $S leak-scale - [fault]          # fault: per-cell
    uv run python $S barriers <-|ids,> [fault]     # faults: not-inserted, gap-closed,
                                                   # features-dropped, unshifted
    uv run python $S exits-identity - [fault]      # faults: in-code, suffix-dropped
    uv run python $S loop-index - [fault]          # fault: uncontracted
    uv run python $S trip-calibration - [fault]    # faults: boundary-trips, wrong-home
    uv run python $S closed-edge - [fault]         # fault: edge-open
    uv run python $S street-boost <-|ids,> [fault]  # faults: classes-dropped, surfaces-dropped
    uv run python $S road-boost <-|ids,> [fault]    # faults: unboosted-road, ways-dropped
    uv run python $S no-features <ids,> [fault]    # fault: neighbour-features
    uv run python $S underflow - [fault]           # fault: unfloored
    uv run python $S faces <ids,> [fault]          # fault: unsorted
    uv run python $S lazy-baseline <ids,> [fault]  # faults: atoms, eager
    uv run python $S both-ways <guards,> <ids,|->  # each guard, then under each of its faults

`baseline`: the fixed mesh (the runtime's metric, METRIC) against the complex rebuilt
(gap_compare.ComplexModel, gap_flow's mesh under the shy distance at 0.5): P0, P_{1/2},0 and
J_2,0 within BASE_TOL (relative), the homes with no demand alike, each home's escape time within
HOME_TOL (relative), and the ONE_K most populous buildings each cleared alone, L_P within ONE_TOL.
Faults: `interior` (the most populous building's interior faces dropped from the mesh: its
clearing opens no ground), `exits` (the street line's vertices, moved by the linework's snap, read
at the unsnapped tolerance: the exits lost; the defect this guard caught while it was built, P0
22% high on 20269).
`weights`: no passage weight is negative or clipped (every face boundary two triangles share has
its centres in order), and at each of WEIGHT_STATES' states every conductance is non-negative and
every home with demand has u0 > 0. Fault `no-conform`: the conformity's splits skipped (a mesh
with face boundaries not Delaunay: refused).
`reach`: at the baseline, the ONE_K single clearings, REACH_RANDOM random clearings at area D
BUDGET and free first's state, the homes the flow model reaches (Model.live_homes_of) equal those
the complex grown by w / 2 reaches at width 0 (gap_complex's test), home for home. Fault
`threshold`: the flow model's reach of a home off by one comparison (a ring with no triangle
reaching the street counted as reaching it: gap_flow.Ring.live).
`gradient`: the model's gradient (power 1 and 2, q 1 and 3, eps GRAD_EPS) at a random x in [0.1,
0.4] and at SIMP's start (x = D everywhere: every triangle within w / 2 of two footprints a tie)
against differences on GRAD_BUILDINGS buildings (at the start, those in the most two-way ties and
no wider one, `_paired`): per building the closest over GRAD_STEPS of the central difference
extrapolated from h and h / 2 (`against_differences`), within GRAD_TOL of the largest, under each
of GRAD_FORMS. Run on 19570, 38136 and 20269 (a large block: its solves' roundoff, divided by the
step, is what the steps are chosen against). Fault `tie`: a tie's gradient broken to one building
(gap_flow.Terms.vjp).
`free`: free first on the complex (Model.free_first, budget 1) frees every walled-in home, and the
flow model's own reach agrees home for home (Model.free_first raises where it does not). Fault
`drop-crossing`: one crossing dropped from free first's answer (its claim kept).
`identity`: the complex's names spell back (gap_clear.complex_of), and a complex metric whose name
would be a raster row directory's (every directory under relax_rows/ and polish_rows/, and the
conductances' names: the retired raster's records) raises; phase 2b's ported scripts (road_rescore
cx, the presets' rows, pweight) write under the runtime's name and no raster's, and a stored run
under a raster name is a record, read by no code (clearings_check.metric_of_run refuses it,
clearings_export.this_code passes it over), under the complex's it reads as the complex's. Fault
`collide`: gap_clear.check_disjoint disabled (a raster-named complex metric accepted, and the
ported scripts' rows under it written into the raster's directories).
`convex` (the study's, from gap_flow_check.py): along CONVEX_LINES random segments of the
clearing box, every passage's factor concave and P (q 1, eps GRAD_EPS) convex, under each of
GRAD_FORMS. Fault `multiply`: a point's terms multiplied (transmittances in series: equal at every
0/1 state, not concave).
`pattern` (phase 2a: assembly on one fixed pattern): at the baseline, a random 0/1 clearing, a
fractional state, the eps world and free first's state, the operator on the fixed pattern
(gap_clear.Graph) against the operator assembled afresh over the state's unknowns from the
passages (`_operator`, phase 1's assembly): on the unknowns equal to PATTERN_TOL, every row off
them an identity row (diagonal 1, its off-diagonals explicit zeros), every value on the pattern's
structure; and P from the model's solve against a direct solve of the other to PATTERN_P_TOL. The baseline holds
walled-in cells (the footprints' interiors and the zone strips, and on 20269 and 23597 walled-in
homes' ground). Fault `drop-identity`: one row off the unknowns left with a diagonal 0 (its
identity dropped).
`constants` (phase 2a: the run's identity): every constant of the complex's code
(gap_clear.code_constants' modules: public UPPER_CASE names, the exempt ones apart) changed one at
a time changes the code's digest and so the metric's name, and the name carries the digest. Fault
`omit`: the digest computed without one constant (MESH_REV).
`device` (phase 2a: the GPU path): cuDSS (gap_factor.CuDSS) against SuperLU on the same states
of one fixed mesh: the baseline's lenses and escape times, a random clearing's lenses, its exact
J_2, SIMP's value and gradient (q 3, eps GRAD_EPS, J_1 and J_2) at its start and at a random x,
the tension (adding and putting back) at the random clearing and the polish's eps-world value,
each within DEVICE_TOL of its largest magnitude. Run on a GPU node. Fault `perturb`: one
refactorization (the random clearing's) handed its values with the diagonal raised by 1e-6
(relative).
`roads` (phase 2b: road re-scoring on the complex): a road network's prefixes (ROAD_NETWORK's area
prefixes, road_rescore's rows) cut into one fixed mesh (gap_clear.Carving), each of ROAD_STOPS a
state on it, against the complex rebuilt with the prefix carved (gap_compare.ComplexModel, the
corridor the road rule frees): L_P, L_{P_1/2} and L_J2 within ROAD_TOL, the homes reached
equal home for home. Fault `one-face-closed`: at every prefix the largest atom it carves left
standing (a road face left closed).
`flux` (phase 2b: the corridor pictures, gap_maps): at the baseline and FLUX_RANDOM random
clearings, the passages' and street faces' fluxes (gap_maps.flow_of) conserve current at every
open cell, the net outflow of an unknown cell equal to its injection within FLUX_TOL of the
largest current through a cell, and the drainage (street faces and ground cells) equal to the
injection in all. Fault `flip-one`: the largest passage flux's sign flipped.
`device-roads` (phase 2b: the road lineup ran on cores): `device` on a carved mesh, cuDSS against
SuperLU at the `roads` guard's prefixes: each prefix's P, P_{1/2} and J_2 and every home's escape
time, within DEVICE_TOL. Run on a GPU node. Fault `perturb`: the first prefix's refactorization
(each plan's second, after the baseline's) as in `device`.

The region model's exits (gap_exits; NOTES, "Exits as leaks: the region model on the complex
(5810@major)" and "The region's leaks: destinations at buildings and on open ground"), the
internal leak at LEAK_LAM on the runtime's metric at each building share of LEAK_SHARES (all on
open ground, all at buildings, half and half):
`leak-assembly`: with leaks, at the baseline, a random 0/1 clearing, the most populous building
cleared alone, a fractional state and the eps world, the operator on the fixed pattern against
`_operator`'s assembly afresh plus a leak computed independently from its definition (each cell's
triangles' area, the cells live at baseline found by a search of their own, the injection per cell
from the demand's points): equal to PATTERN_TOL on the unknowns, the structure the pattern's, and P
against a direct solve to PATTERN_P_TOL. Faults: `drop-one` (the largest leak dropped),
`not-live` (a leak on the largest cell not live at baseline).
`leak-fold` (the two families folded into one): at FOLD_SHARES, the leak per cell on the model's
own cells against the region test's two formulas (`_uniform_was`, `_home_was`: UniformLeak's and
HomeLeak's per_cell before the fold, verbatim) mixed as s says, bit for bit at s 0 and 1 and to
PATTERN_TOL between, and against the definition's independent oracle (as `leak-assembly`'s) to
PATTERN_TOL; the baseline's operator against `_operator`'s plus the oracle to PATTERN_TOL. Fault
`shares-swapped`: the open ground's share 1 - s applied to the buildings (and s to open ground).
`enclosure`: under each share the homes walled in at baseline are exactly those walled in without
leaks (on blocks with walled-in homes). Fault `every-cell`: the leak on every cell.
`leak-gradient`: the `gradient` guard's states and powers on the runtime's form under each share.
Fault `adjoint-unleaked`: J_2's adjoint solved on the system without its leaks.
`leak-scale`: on SCALE_BLOCK (open ground SCALE_L wide, its edge the street, one small home at its
centre: the mesh graded from R near the outlines to R 2^levels in the open), the leak at SCALE_LAM
at each building share s of SCALE_SHARES: the home's potential over SCALE_R against the screened
Laplacian's point source, Q' K0(r / lam') / (2 pi sigma), its shape (the ratio's spread over the
distance bins) and its level within SCALE_TOL. lam' = lam / sqrt(1 - s), the open ground's share
alone screening it; Q' the home's current less what its own building's destination drains (s 0:
all of it). Fault `per-cell`: the same conductance on every cell (the leak's mean), not one per
area.
`barriers`: on BARRIER_BLOCK (a street on its south edge only, a railway across it with a crossing
node's path and a footbridge, a water channel north of the railway parting the north in two), every
open triangle inside a barrier lies in a crossing's gap, no home is walled in, and each crossing
removed walls in the homes it alone served, exactly (the path the north-west's, the footbridge the
north-east's), and the crossing node with its path missing from the features still opens the
north-west (its gap across the track); and on each real block named (a railway or water meets
it, read from its own features afresh: common.own_osm), RUNTIME's fixed mesh has no triangle inside
the cut where the leak-only metric's has some, the mesh's cache key is not the uncut block's, and
RUNTIME's name carries the barriers, its atoms are the footprints in the cut mesh's own frame,
and the complex walls in the homes the flow model does (Model.free_first's check). Faults:
`not-inserted` (the barriers not cut), `gap-closed` (the crossings' gaps not cut),
`features-dropped` (every block built with no OSM features: the cut absent), `unshifted` (the
atoms left in the uncut block's frame: on 3934, whose railway cut moves its origin, the
footprints misplaced and the complex's walled-in homes not the flow model's).
`exits-identity`: the per-block metric's name is PER_BLOCK and its code digests 28 constants to the
same digest; RUNTIME is the per-block metric with metrics.EXITS, read back, and the region's
T500b90D10; every configuration of region_leaks (and more shares and lam with the boost) under
each preset's search spells a name that parses back to it, and no two share a name; every stored
region run's name (the committed region_rows/summary_lenses.csv, and the runs' rows where they
are) and every baseline's suffix (summary_baseline.csv) parses back and is its label's
configuration's. Faults: `in-code` (the exits' module in the code's walk), `suffix-dropped` (the
leak's part left out of the suffix).
`loop-index` (region_leaks.loops on cases with a known answer): blocks whose open ground is only
streets 6 m wide, painted from the flow at FRAME_H, every open pixel with flow in the network: a
grid of 3 x 3 blocks inside an exit ring (8 loops with the exits one node), a ladder of 5 bars off
a street on one side with a back lane (5), the same bars reaching the far wall, a comb of dead ends
(0). Fault `uncontracted`: the exits not contracted (the grid's own cycle rank, 9).
`trip-calibration` (region_leaks' trips and calibration, on open ground at s 0): on SCALE_BLOCK,
region_leaks.calibrate_lam to a median trip of TRIP_T (to TRIP_GUARD_TOL, lam to TRIP_GUARD_RES)
gives a lam whose screened point source's median, 1.2572 lam, is within region_leaks.TRIP_TOL of
TRIP_T, and it refuses a target past the median's limit as the leak vanishes; on TRIP_BLOCK (the
same ground, a second home TRIP_EDGE_M from the street edge, the boundary taking 44% of its
current) the pooled median trip at TRIP_LAM is within TRIP_TOL of the method of images' (each
home's point source less its first images in the square's four edges, integrated on a polar grid,
the trips inside the square only). Faults: `boundary-trips` (the street faces' drainage counted
as trips), `wrong-home` (each home's trips measured from the next home's injection).
`closed-edge`: on TRIP_BLOCK with its edge closed (gap_exits.ClosedEdge), every live component has
a leak and the cells live are the open edge's, the street drains nothing and the leaks the
injection, and the pooled median trip is the method of images' with the images added. Fault
`edge-open`: ClosedEdge.mesh leaving the edge open.
`street-boost`: on STREET_BOOST_BLOCK (an asphalt residential way, an asphalt footway, a paved
tertiary link, an unpaved residential way and an untagged tertiary), the boost on paved residential
and tertiary at 10x boosts every triangle within BOOST_M of the asphalt residential way and the
link (`_link` folded) and none near the footway, the unpaved way or the untagged one, and
multiplies a passage between boosted triangles by 10 and the others by 1; and on each real block
named (a paved drivable way, one not tagged paved and a footway inside it), RUNTIME's boost raises
every triangle near a paved drivable way, none near only a footway or path or a drivable way not
tagged paved, a passage between boosted triangles 10x and the others 1x. Faults
`classes-dropped`: every way with a highway tag boosted; `surfaces-dropped`: a drivable way
boosted whatever its surface.
`road-boost` (owner 2026-10-20: "proposed roads should get the same 10x boost as existing drivable
streets"): on STREET_BOOST_BLOCK under that block's boost and under no boost, and on each real block
named under RUNTIME, each carved by the faces guard's two nested stops, every road prefix's state
conducts the exits' road factor times the unboosted mesh on the passages between two triangles in
its corridor (by centroid), the exits' mesh outside it, and reaches on its own mesh; a clearing's
state is on the exits' mesh. Faults `unboosted-road`: a prefix on the exits' mesh;
`ways-dropped`: the corridor boosted on the unboosted mesh.
`no-features` (RUNTIME's switch, NOTES "The consolidated pass"): on each block named, carrying its
own features less every drivable way and barrier feature (no real block lacks a drivable street
on its ring), RUNTIME's baseline (P0, P_{1/2},0, J_2,0, every home's escape time, the homes walled
in) is the leak-only metric's to NO_FEATURES_TOL. Fault `neighbour-features`: the block carries a
neighbouring kblock's features (the one sharing the most drivable street with it).

`underflow` (the refined solve's stopping test, gap_factor._residual; NOTES, "Before the
consolidated pass: three fixes"): on UNDERFLOW_BLOCK (22 homes at one end of a strip 4 km long,
under the source's building floor: built through common.member_blocks), whose potential falls
through the subnormals to 0 along the strip, the model builds (its baseline's solves refined),
the potential under the population's demand has subnormal cells by a direct solve (the premise:
the underflow front is there), and the refined solve equals scipy's spsolve within UNDERFLOW_TOL
of the largest. Fault `unfloored`: the backward error with each term read at its own size
(_residual before the fix, verbatim): the baseline's solve stalls at a backward error of 1.

`faces` and `lazy-baseline` (NOTES, "Mesh speedups before the lineup's rerun"), each on a block
plain and carved by CROSS_HALF_M roads (`_cross_carving`: one along the block's middle, then that
and one across it). `faces`: gap_clear._faces on the atoms, its per-face footprints grouped by one
sort (gap_clear._by_face), equals the same with the per-face scan it replaced (`_scan_was`,
verbatim): the faces, the noded lines, and every face's cover and zone, element for element.
Fault `unsorted`: each face's footprints in the tree query's order, not ascending.
`lazy-baseline`: a carved mesh (RUNTIME's, meshed afresh) builds no baseline complex
(gap_complex.build is never called) and its file holds none; built on use, from its file or not,
it equals the plain mesh's array for array (the plain one's stored, and read back). Faults:
`atoms` (the carved mesh's baseline built on the carving's atoms, not the buildings), `eager` (a
carved mesh's baseline built with it).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import clearings_check  # noqa: E402
import clearings_export  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_factor  # noqa: E402
import gap_compare as gcmp  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_exits as gx  # noqa: E402
import gap_flow as gf  # noqa: E402
import gap_maps  # noqa: E402
import metrics  # noqa: E402
import pweight  # noqa: E402
import region_leaks  # noqa: E402
import road_rescore as rr  # noqa: E402
import search  # noqa: E402

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from reblock.contracts import Block

NAME = metrics.PER_BLOCK          # the complex's core: the per-block metric (no exits; RUNTIME's
                                  # exits are the leak, barrier and boost guards')
METRIC = gcl.complex_of(NAME, factorization=gap_factor.SuperLU())
BUDGET = 0.05
BASE_TOL = 2e-3          # the baseline against the rebuild: the rebuild's own lattice, moved,
                         # moves P0 by up to 1.9e-3 and J_2,0 by 4.6e-3 (19570)
HOME_TOL = 5e-2          # ... each home's escape time (the rebuild's lattice moved: up to 9.8%)
ONE_K = 3
ONE_TOL = 2e-3           # a single clearing's L_P against the rebuild's
WEIGHT_STATES = 3        # random 0/1 clearings at BUDGET, besides the baseline, the eps world and
                         # a fractional state
REACH_RANDOM = 2
GRAD_FORMS = ("shy0.5", "close0.125", "ramp0.5")
GRAD_EPS = 1e-4          # SIMP base's eps world
GRAD_BUILDINGS = 6
GRAD_Q = (1.0, 3.0)
GRAD_STEPS = (1e-3, 1e-4)    # against_differences' steps h (each with h / 2), the closest taken:
                             # above the solves' roundoff over the step (on 20269, 1e-4 of the
                             # gradient at 1e-5 and 1e-3 at 1e-6, measured), the extrapolation
                             # cancelling a tie's O(h) term (the start's: 2e-4 at 1e-4 on 38136)
GRAD_TOL = 1e-4
CONVEX_LINES = 4
CONVEX_POINTS = 7
CONVEX_TOL = 1e-9
SEED = 20261014
POP = common.PopulationDemand()
PATTERN_TOL = 1e-12      # the fixed pattern's operator against the one assembled afresh ...
PATTERN_P_TOL = 1e-9     # ... and P from its refined solve against an unrefined direct solve of
                         # the other (1e-12 -- 2e-11 apart on 19570 and 20269)
DEVICE_TOL = 1e-8        # cuDSS against SuperLU, of the largest magnitude (both refined to the
                         # floating-point floor: 1e-12 -- 1e-10 apart on 19570 and 20269)
PERTURB = 1e-6           # fault `perturb`: the diagonal's relative rise
ROAD_ROWS = HERE / "road_rows"      # item 4's road networks and their prefixes (road_rescore)
ROAD_NETWORK = "resistance_lp"      # the network the guard carves: the lineup's leader
ROAD_STOPS = (1, 3, 5)               # its area prefixes checked (D 0.02, 0.10, 0.20)
ROAD_TOL = ONE_TOL                   # a prefix's lenses against the rebuild's, as a clearing's
FLUX_RANDOM = 2
FLUX_TOL = 1e-9          # a cell's imbalance, of the largest current through a cell (the solves
                         # refined to the backward-error floor: 2e-11 -- 5e-11 on 19570
                         # and 20269)
LEAK_LAM = 50.0          # the leak guards' lam (m): a small block's scale (19570 is 215 m long)
LEAK_PAIRS = ((LEAK_LAM, math.inf), (math.inf, 20.0), (100.0, 25.0))   # ... the leak's lengths
                         # (open_m, foot_m): open ground only, footprints only, both
FOLD_SHARES = (0.0, 1.0, 0.75)   # leak-fold: the region test's shares at LEAK_LAM, as constants
                                 # (0.75: not symmetric under the swap)
PER_BLOCK = "cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a"   # the per-block metric's name: every
                                                          # stored per-block row's, unchanged
PER_BLOCK_CONSTANTS = 28
SCALE_L = 800.0          # leak-scale: the open square's side (m)
SCALE_LAM = 60.0         # ... the uniform leak's lam (the square's half-side over 6.7 lam: its
                         # Dirichlet edge moves the potential within SCALE_R by under 1e-4)
SCALE_R = (10.0, 200.0)  # ... the distances read (m): beyond the home's ring, short of the edge
SCALE_BINS = 12
SCALE_TOL = 0.03         # ... the ratio's spread over the bins, and its level against the formula
SCALE_PAIRS = ((SCALE_LAM, math.inf), (SCALE_LAM * math.sqrt(2.0), 2.0))   # ... the leak's
                         # lengths: open ground only, and with the home's own footprint leaking
CLOSED_EDGE_M = 50.0     # closed-edge: TRIP_BLOCK's second home this far from the closed edge
TRIP_T = 63.0            # trip-calibration: the median trip calibrated to on SCALE_BLOCK (m: lam
                         # near 50, where the square's edge takes 0.1% of the current)
TRIP_GUARD_TOL = 1e-3    # ... the bisection's tolerance there (relative), and lam's resolution (m)
TRIP_GUARD_RES = 0.01
TRIP_EDGE_M = 50.0       # ... TRIP_BLOCK's second home: its centre this far from the south edge
TRIP_LAM = 60.0          # ... the lam the two homes' pooled trips are read at
TRIP_ORACLE_DR = 0.05    # ... the oracle's polar grid: radial step (m) and angles
TRIP_ORACLE_NT = 1440
BARRIER_EDGE_M = 0.05   # barriers on a real block: the cut less this (m) holds no triangle (its
                        # edges are the mesh's own, noded at LINE_SNAP_M)
NO_FEATURES_TOL = 1e-12  # no-features: RUNTIME's baseline against the leak-only metric's (relative)
UNDERFLOW_BLOCK = "ZAF.9.1.4_1_7283"    # underflow: its potential underflows along a 4 km strip
UNDERFLOW_TOL = 1e-9     # ... its refined solve against spsolve, of the largest (9.6e-13 apart)
CROSS_HALF_M = 3.0       # faces, lazy-baseline: the carving's roads' half-width (m; a 6 m road)


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


@contextmanager
def _patched_item(owner: list[bool], i: int, value: bool) -> Iterator[None]:
    old = owner[i]
    owner[i] = value
    try:
        yield
    finally:
        owner[i] = old


def _model(b, metric: gcl.ComplexMetric = METRIC) -> gcl.Model:
    """The block's model meshed afresh (a temporary cache: a fault in the meshing is seen)."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = metric.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
    return metric.model(b, fx, demand=POP)


def _random(m: gcl.Model, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(m.n)
    x = np.zeros(m.n, dtype=bool)
    x[order[np.cumsum(m.cost[order]) <= BUDGET]] = True
    return x


def _largest(m: gcl.Model, k: int) -> list[int]:
    return [int(j) for j in np.argsort(-m.pop, kind="stable")[:k]]


# ------------------------------------------------------------------ baseline

def baseline(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        try:
            m = _model(b)
        except Exception as e:      # noqa: BLE001 -- a fault may break the mesh: reported
            print(f"{bid}: the model fails to build ({e!r}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        cm = gcmp.ComplexModel(b, METRIC.fs, METRIC.form)
        rel = {"P0": m.base.P / cm.P0 - 1.0, "Pq0": m.base.Pq / cm.Pq0 - 1.0,
               "J2_0": m.base.J2 / cm.J2_0 - 1.0}
        same_walled = bool(np.array_equal(m.nodem, cm.walled))
        live = ~m.nodem
        home = float(np.max(np.abs(m.u_pop0[live] / cm.u0[live] - 1.0))) if live.any() else 0.0
        ones = {}
        for j in _largest(m, ONE_K):
            x = np.zeros(m.n, dtype=bool)
            x[j] = True
            L = m.lenses(x)
            r, _u = cm.score(~x, None)
            ones[j] = (1.0 - L.P / m.base.P) - r["L_P"]
        worst_one = max(abs(v) for v in ones.values())
        ok = (max(abs(v) for v in rel.values()) <= BASE_TOL and same_walled
              and home <= HOME_TOL and worst_one <= ONE_TOL)
        print(f"{bid}: baseline against the rebuild "
              + ", ".join(f"{k} {v:+.1e}" for k, v in rel.items())
              + f"; no-demand homes alike {same_walled}; homes' u0 within {home:.1e}; single "
              f"clearings' L_P " + ", ".join(f"{j} {v:+.1e}" for j, v in ones.items())
              + f" -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


_FACES = gcl._faces


def _faces_less_interior(geo, spec, grow):
    """gap_clear._faces less the faces the most populous building covers (fault `interior`)."""
    faces, cover, zone, noded = _FACES(geo, spec, grow)
    j = int(np.argmax(common.POPULATIONS["area"].weights(geo.polys)))
    keep = np.array([j not in cv for cv in cover])
    return faces[keep], cover[keep], zone[keep], noded


# ------------------------------------------------------------------ weights

def weights(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        try:
            m = _model(b)
        except Exception as e:      # noqa: BLE001
            print(f"{bid}: the mesh is refused ({e!r}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        fm, c = m.fm, m.c
        L = np.hypot(*(fm.xy[fm.pu] - fm.xy[fm.pv]).T)
        neg = int((fm.T < 0).sum())
        cross = fm.face[fm.pa] != fm.face[fm.pb]
        cross_inf = int((cross & np.isinf(fm.T)).sum())
        rng = np.random.default_rng(SEED)
        states = {"baseline": np.zeros(m.n), "eps": np.full(m.n, GRAD_EPS),
                  "fractional": rng.uniform(0.0, 1.0, m.n)}
        for k in range(WEIGHT_STATES):
            states[f"random{k}"] = _random(m, rng).astype(float)
        worst = np.inf
        for _name, k_ in states.items():
            st = m.factors(k_)
            worst = min(worst, float(st.reach.Tp.min()), float(st.reach.Gf.min()))
        u0_ok = bool((m.u_pop0[~m.nodem] > 0).all())
        ok = neg == 0 and cross_inf == 0 and worst >= 0 and u0_ok
        print(f"{bid}: {len(L)} passages, {neg} negative weights, {cross_inf} across faces "
              f"with one circumcentre, {fm.splits} conformity splits; the least conductance over "
              f"{len(states)} states {worst:.2e}; u0 > 0 for every home with demand {u0_ok} -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _no_conform(a):
    """gap_clear._conform without its splits (fault `no-conform`)."""
    return gcl.Conformed(arrays=a, unflat=0, flat_left=0, splits=0, rounds=0)


# ------------------------------------------------------------------ reach

def reach(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        try:
            m = _model(b)
        except Exception as e:      # noqa: BLE001 -- a fault may break the model: reported
            print(f"{bid}: the model fails to build ({e!r}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        rng = np.random.default_rng(SEED + 1)
        states = {"baseline": np.zeros(m.n, dtype=bool)}
        for j in _largest(m, ONE_K):
            x = np.zeros(m.n, dtype=bool)
            x[j] = True
            states[f"single {j}"] = x
        for k in range(REACH_RANDOM):
            states[f"random{k}"] = _random(m, rng)
        fr = gc.free_first(m.geo, m.spec, w=m.form.filtration_w, grow=m.form.grow, budget=1.0,
                           cost=m.cost, pop=m.pop, held0=np.zeros(m.n, dtype=bool), cx0=None)
        states["free first"] = fr.held
        parts = []
        for name, x in states.items():
            flow = m.live_homes_of(m.factors(m.k_of(x)))
            cxl = m.complex_live(x)
            d = np.flatnonzero(flow ^ cxl)
            parts.append(f"{name}: {int((~flow).sum())} walled in, {len(d)} apart")
            if len(d):
                bad.append(f"{bid} {name} {d[:5].tolist()}")
        print(f"{bid}: " + "; ".join(parts), flush=True)
    return bad


def _ring_off_by_one(self, live_tri):
    """gap_flow.Ring.live with its test `> 0` read `>= 0` (fault `threshold`): every home's ring
    reaches the street."""
    return (self.tri @ live_tri.astype(np.float64) >= 0) | self.band


# ------------------------------------------------------------------ gradient

def gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for token in GRAD_FORMS:
            m = _model(b, _form_metric(token))
            for where, (x0, pick) in _grad_states(m).items():
                for power in (1.0, 2.0):
                    for q in GRAD_Q:
                        err = against_differences(m.relaxation(power, q=q, eps=GRAD_EPS), x0,
                                                  pick)
                        ok = err <= GRAD_TOL
                        print(f"{bid} {token} {where} J_{power:g} q {q:g}: gradient against "
                              f"differences {err:.1e} of the largest -> "
                              f"{'ok' if ok else 'FAILS'}", flush=True)
                        if not ok:
                            bad.append(f"{bid} {token} {where} p{power:g} q{q:g}")
    return bad


def _grad_states(m: gcl.Model) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The gradient guards' states, each with the buildings checked: a random x in [0.1, 0.4]
    (GRAD_BUILDINGS random buildings) and SIMP's start, x = BUDGET everywhere (`_paired`'s)."""
    rng = np.random.default_rng(SEED)
    return {"random": (rng.uniform(0.1, 0.4, m.n),
                       rng.choice(m.n, GRAD_BUILDINGS, replace=False)),
            "start": (np.full(m.n, BUDGET), _paired(m.terms, m.n))}


def against_differences(rel: gcl.Relaxation, x0: np.ndarray, pick: np.ndarray) -> float:
    """The adjoint gradient of `rel` at x0 against differences on the buildings `pick`: per
    building the closest, over the steps h of GRAD_STEPS, of the central difference extrapolated
    from h and h / 2, 2 D(h / 2) - D(h); the largest difference, of the largest |gradient| among
    them. D(h) reaches the derivative at O(h^2) where J is smooth, but at a tie of the minimum
    (SIMP's start) it reaches the mean of the two one-sided derivatives, which is the tie's
    shared gradient, only at O(h) (the two sides curve apart): the extrapolation cancels that
    term, so a step large enough for the solves' roundoff serves both (NOTES, "Before the
    consolidated pass: three fixes")."""
    _J, g = rel.value_grad(x0)
    err = []
    for j in pick:
        best = np.inf
        for h in GRAD_STEPS:
            d = []
            for step in (h, h / 2.0):
                e = np.zeros(len(x0))
                e[j] = step
                d.append((rel.value(x0 + e) - rel.value(x0 - e)) / (2.0 * step))
            best = min(best, abs(2.0 * d[1] - d[0] - g[j]))
        err.append(best)
    return float(np.max(err) / np.abs(g[pick]).max())


def _paired(terms: gf.Terms, n: int) -> np.ndarray:
    """At SIMP's start (every k alike) the GRAD_BUILDINGS buildings in the most two-way ties (a
    point within w / 2 of it and one other footprint, phi 0) and in no wider one: there the
    central difference of the minimum is the shared gradient's 1/2 (of three tied, the central
    difference is still 1/2, the shared gradient 1/3: both supergradients)."""
    zero = terms.phi == 0.0
    row, b = terms.row[zero], terms.b[zero]
    size = np.bincount(row, minlength=terms.n * terms.G)[row]
    two = np.bincount(b[size == 2], minlength=n)
    wider = np.bincount(b[size > 2], minlength=n) > 0
    return np.argsort(-np.where(wider, -1, two), kind="stable")[:GRAD_BUILDINGS]


def _vjp_first(self, k, g):
    """gap_flow.Terms.vjp with a tie's gradient given wholly to its first building (fault
    `tie`)."""
    v, mins = self._mins(k)
    at = v <= mins[self.row] + gf.TIE_TOL
    idx = np.flatnonzero(at)
    first = np.ones(len(idx), dtype=bool)
    first[1:] = self.row[idx[1:]] != self.row[idx[:-1]]
    idx = idx[first]
    wt = (1.0 - self.phi[idx]) / self.G
    return np.bincount(self.b[idx], weights=g[self.row[idx] // self.G] * wt, minlength=len(k))


# ------------------------------------------------------------------ free first

def free(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        m = _model(b)
        try:
            held = m.free_first(budget=1.0, held=np.zeros(m.n, dtype=bool))
        except RuntimeError as e:
            print(f"{bid}: free first and the flow model part ({e}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        left = m.walled0 & ~m.live_homes_of(m.factors(m.k_of(held)))
        ok = not left.any()
        print(f"{bid}: {int(m.walled0.sum())} walled in, free first holds "
              f"{np.flatnonzero(held).tolist()}, the flow model leaves "
              f"{np.flatnonzero(left).tolist()} walled in -> {'ok' if ok else 'FAILS'}",
              flush=True)
        if not ok:
            bad.append(bid)
    return bad


_FREE_FIRST = gc.free_first


def _drop_crossing(geo, spec, *, w, grow, budget, cost, pop, held0, cx0):
    """gap_complex.free_first with its last crossing's buildings dropped from the held set, its
    claim (`left`) kept (fault `drop-crossing`)."""
    fr = _FREE_FIRST(geo, spec, w=w, grow=grow, budget=budget, cost=cost, pop=pop, held0=held0,
                     cx0=cx0)
    if not fr.crossings:
        return fr
    held = fr.held.copy()
    held[fr.crossings[-1][1]] = False
    return fr._replace(held=held)


# ------------------------------------------------------------------ identity

def _raster_names() -> list[str]:
    names = {"uni+apwfs", "uni+b1+apwfs", "uni+d0.5+apwfs", "ss100k2n2r30+apwfs", "uni"}
    for d in ("relax_rows", "polish_rows"):
        if (HERE / d).is_dir():
            names |= {p.name for p in (HERE / d).iterdir() if p.is_dir()
                      and not p.name.startswith("cx+")}
    return sorted(names)


def identity(_ids: list[str]) -> list[str]:
    bad = []
    for name in (NAME, f"cx+close0.125+s0.125r1b1+R0.5g4x2q4+v{gcl.CODE}",
                 f"cx+ramp0.5s0.5+s0.125r1b1+R0.25q4+lumped+v{gcl.CODE}"):
        try:
            back = gcl.complex_of(name, factorization=gap_factor.SuperLU()).name
        except ValueError as e:
            back = repr(e)
        if back != name:
            bad.append(f"{name} spells {back}")
    refused = 0
    names = _raster_names()
    for raster in names:
        class Colliding(gcl.ComplexMetric):
            @property
            def name(self) -> str:
                return raster
        try:
            colliding = Colliding(spec=METRIC.spec, fs=METRIC.fs, form=METRIC.form,
                                  lumped=False, search_eps=METRIC.search_eps,
                                  exits=METRIC.exits, factorization=METRIC.factorization)
        except ValueError:
            refused += 1
            continue
        bad.append(f"a complex metric named {raster!r} was accepted")
        # what the ported scripts would then write: rows under the raster's directory
        for where, path in _ported_paths(colliding).items():
            if raster in path.parts:
                bad.append(f"{where} would write under the raster's {raster!r}: {path}")
    # the ported scripts' rows under the runtime's metric: its name, never a raster's
    for where, path in _ported_paths(METRIC).items():
        if (not any(p.split("@")[0] == METRIC.name for p in path.parts)
                or set(path.parts) & set(names)):
            bad.append(f"{where} writes under {path}")
    # reading: a stored run under a raster name is a record, never read as the complex's
    # (clearings_check.metric_of_run refuses it, clearings_export.this_code passes it over); the
    # complex's reads as the complex's
    template = next(i for i in clearings.Store.committed().runs().values()
                    if i.along == "uni" and i.mesh == "h0.5")
    read_ok = 0
    for raster in names:
        metric_name, _search = clearings_export.split_along(raster)
        try:
            got = clearings_check.metric_of_run(dataclasses.replace(template, along=metric_name))
        except ValueError:
            got = None
        if got is not None or clearings_export.this_code(raster):
            bad.append(f"a run under {raster!r} reads as the complex's")
        else:
            read_ok += 1
    got = clearings_check.metric_of_run(dataclasses.replace(template, along=NAME, mesh=""))
    if not (isinstance(got, gcl.ComplexMetric) and got.name == NAME):
        bad.append(f"a run under {NAME} reads as {got.name}")
    print(f"identity: names spelled back; {refused} of {len(names)} raster names refused as a "
          f"complex metric's ({', '.join(names[:6])}, ...); the ported scripts' rows under "
          f"{NAME} ({', '.join(_ported_paths(METRIC))}); {read_ok} of {len(names)} raster runs "
          f"records, read by no code -> {'ok' if not bad else 'FAILS'}", flush=True)
    return bad


def _ported_paths(metric: gcl.ComplexMetric) -> dict[str, Path]:
    """Where each script ported in phase 2b writes its rows under `metric`."""
    presets = rr.cx_presets(metric, search.Exactly())
    return {"road_rescore cx": rr.cx_dir(ROAD_ROWS, metric), "SIMP's default": presets.simp,
            "the cheap preset": presets.cheap, "pweight": pweight.rows_of(metric.name)}


# ------------------------------------------------------------------ convexity (the study's)

def convex(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for token in GRAD_FORMS:
            m = _model(b, _form_metric(token))
            rel = m.relaxation(1.0, q=1.0, eps=GRAD_EPS)
            rng = np.random.default_rng(SEED + 2)
            worst_k, worst_P = -np.inf, np.inf
            for _ in range(CONVEX_LINES):
                a, z = rng.uniform(0.0, 1.0, m.n), rng.uniform(0.0, 1.0, m.n)
                ks, Ps = [], []
                for t in np.linspace(0.0, 1.0, CONVEX_POINTS):
                    x = a + t * (z - a)
                    ks.append(m.factors(rel.k_of(x)).kp)
                    Ps.append(rel.value(x))
                K2 = np.diff(np.array(ks), 2, axis=0)
                P2 = np.diff(np.array(Ps), 2)
                worst_k = max(worst_k, float(K2.max()))
                worst_P = min(worst_P, float((P2 / np.max(Ps)).min()))
            ok = worst_k <= CONVEX_TOL and worst_P >= -CONVEX_TOL
            print(f"{bid} {token}: on {CONVEX_LINES} lines the factors' largest second difference "
                  f"{worst_k:.1e}, P's smallest {worst_P:.1e} of P -> {'ok' if ok else 'FAILS'}",
                  flush=True)
            if not ok:
                bad.append(f"{bid} {token}")
    return bad


def _multiply(self, k):
    """gap_flow.Terms.value with a point's terms multiplied (transmittances in series), not their
    minimum: equal at every 0/1 state, not concave (fault `multiply`)."""
    v = self.phi + k[self.b] * (1.0 - self.phi)
    prod = np.ones(self.n * self.G)
    if len(self.row):
        np.multiply.at(prod, self.row, v)
    return prod.reshape(self.n, self.G).mean(axis=1)


def _form_metric(token: str) -> gcl.ComplexMetric:
    """The runtime's metric under the width form `token` (gap_compare.form_of)."""
    return gcl.complex_of(f"cx+{gcmp.form_of(token).name}+s0.125r1b1+R0.5g4x2q4+v{gcl.CODE}",
                          factorization=gap_factor.SuperLU())


# ------------------------------------------------------------------ the fixed pattern

def _operator(st: gcl.State) -> tuple[sp.csr_matrix, np.ndarray]:
    """(the state's operator over its unknowns, assembled afresh from the passages as phase 1
    did: a passage between two unknowns off the diagonal, every conducting one on its unknowns'
    diagonals, the street faces' too; the unknown cells, in order)."""
    c, r = st.sys.cells, st.reach
    cells = np.flatnonzero(r.unknown)
    unk = -np.ones(c.n, dtype=np.int64)
    unk[cells] = np.arange(len(cells))
    N = len(cells)
    ua, ub = unk[c.ea], unk[c.eb]
    diag = (np.bincount(ua[ua >= 0], weights=r.Tp[ua >= 0], minlength=N)
            + np.bincount(ub[ub >= 0], weights=r.Tp[ub >= 0], minlength=N))
    uf = unk[c.fc]
    diag += np.bincount(uf[uf >= 0], weights=r.Gf[uf >= 0], minlength=N)
    both = (ua >= 0) & (ub >= 0) & (r.Tp > 0)
    rows = np.concatenate([ua[both], ub[both], np.arange(N)])
    cols = np.concatenate([ub[both], ua[both], np.arange(N)])
    vals = np.concatenate([-r.Tp[both], -r.Tp[both], diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N)), cells


def pattern(ids: list[str]) -> list[str]:
    from scipy.sparse.linalg import spsolve
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        m = METRIC.clearing(b, demand=POP)
        rng = np.random.default_rng(SEED + 3)
        fr = m.free_first(budget=1.0, held=np.zeros(m.n, dtype=bool))
        states = {"baseline": m.state(np.zeros(m.n)),
                  "random": m.state(_random(m, rng).astype(float)),
                  "fractional": m.state(rng.uniform(0.0, 1.0, m.n)),
                  "eps": m.state(np.full(m.n, GRAD_EPS)),
                  "free first": m.state(fr.astype(float))}
        parts = []
        for name, st in states.items():
            A = st.A
            assert A is not None
            p = st.sys.graph.pattern
            same_structure = (np.array_equal(A.indptr, p.indptr)
                              and np.array_equal(A.indices, p.indices))
            ref, cells = _operator(st)
            sub = A[cells][:, cells]
            err = float(abs(sub - ref).max() / abs(ref).max()) if ref.nnz else 0.0
            off = ~st.reach.unknown
            rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
            on_off = off[rows]
            diag = rows == A.indices
            ident = bool(np.all(A.data[on_off & diag] == 1.0)
                         and np.all(A.data[on_off & ~diag] == 0.0)
                         and np.all(A.data[off[A.indices] & ~diag] == 0.0))
            rhs = m._rhs(st, m.pop)
            u = m.solve(st, m.pop)
            P_fixed = float(rhs @ u)
            P_ref = float(rhs[cells] @ spsolve(ref.tocsc(), rhs[cells]))
            dP = abs(P_fixed / P_ref - 1.0)
            ok = same_structure and err <= PATTERN_TOL and ident and dP <= PATTERN_P_TOL
            parts.append(f"{name}: {int(off.sum())} of {len(off)} cells off the unknowns, "
                         f"operator {err:.1e}, identity rows {ident}, P {dP:.1e}"
                         + ("" if ok else " FAILS"))
            if not ok:
                bad.append(f"{bid} {name}")
        print(f"{bid}: " + "; ".join(parts), flush=True)
    return bad


_VALUES = gcl.Graph.values


def _drop_identity(self: gcl.Graph, r: gcl.Reach) -> np.ndarray:
    """gap_clear.Graph.values with the first row off the unknowns given a diagonal 0 (fault
    `drop-identity`)."""
    data = _VALUES(self, r)
    off = np.flatnonzero(~r.unknown)
    if len(off):
        i = off[0]
        p = self.pattern
        lo, hi = p.indptr[i], p.indptr[i + 1]
        data[lo + int(np.flatnonzero(p.indices[lo:hi] == i)[0])] = 0.0
    return data


# ------------------------------------------------------------------ the code's identity

_DIGESTED = gcl.code_constants


def _perturbed(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value * 2 + 1
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return (*value, 0)
    if isinstance(value, np.ndarray):
        return value + 1
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        f = dataclasses.fields(value)[0]
        return dataclasses.replace(value, **{f.name: _perturbed(getattr(value, f.name))})
    raise TypeError(f"no perturbation for {type(value).__name__}")


def constants(_ids: list[str]) -> list[str]:
    bad = []
    base = gcl.code_digest()
    if not NAME.endswith(f"+v{base}") or gcl.CODE != base:
        bad.append(f"the metric's name {NAME} does not carry the code's digest {base}")
    mods = gcl._identity_modules()
    seen = 0
    for mod in mods:
        short = mod.__name__.rsplit(".", 1)[-1]
        for name, value in list(vars(mod).items()):
            key = f"{short}.{name}"
            if name.startswith("_") or not name.isupper() or key in gcl._EXEMPT:
                continue
            seen += 1
            with _patched(mod, name, _perturbed(value)):
                d = gcl.code_digest()
            if d == base:
                bad.append(f"{key} changed, the digest did not")
    for key in gcl._EXTERNAL:
        mod_name, name = key.split(".")
        mod = sys.modules[mod_name]
        seen += 1
        with _patched(mod, name, _perturbed(getattr(mod, name))):
            if gcl.code_digest() == base:
                bad.append(f"{key} changed, the digest did not")
    print(f"constants: {seen} constants, each changed alone; the digest {base} changed for "
          f"{seen - len(bad)} -> {'ok' if not bad else 'FAILS: ' + ', '.join(bad[:5])}",
          flush=True)
    return bad


def _omit_mesh_rev() -> dict[str, str]:
    """gap_clear.code_constants without MESH_REV (fault `omit`)."""
    out = _DIGESTED()
    del out["gap_clear.MESH_REV"]
    return out


# ------------------------------------------------------------------ the device

def device(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        fx = METRIC.fixed(b, cache=gcl.CACHE, carving=gcl.Carving.none(gc.geometry(b)))
        ms = {dev: dataclasses.replace(METRIC, factorization=gap_factor.factorization_of(dev))
              .model(b, fx, demand=POP) for dev in ("cpu", "gpu")}
        out: dict[str, dict[str, np.ndarray]] = {}
        for dev, m in ms.items():
            rng = np.random.default_rng(SEED + 4)
            x = _random(m, rng)
            r: dict[str, np.ndarray] = {}
            r["baseline"] = np.array([m.base.P, m.base.Pq, m.base.J2])
            r["baseline homes"] = m.base.u
            L = m.lenses(x)
            r["random"] = np.array([L.P, L.Pq, L.J2])
            r["random homes"] = L.u
            r["random J_2"] = np.array(m.exact(x, 2.0))
            for power in (1.0, 2.0):
                rel = m.relaxation(power, q=3.0, eps=GRAD_EPS)
                for where, xf in (("start", np.full(m.n, BUDGET)),
                                  ("random x", np.random.default_rng(SEED + 5).uniform(
                                      0.1, 0.4, m.n))):
                    J, g = rel.value_sgrad(xf)
                    r[f"SIMP J_{power:g} {where}"] = np.concatenate([[J], g])
            m.removed[:] = x
            for restore in (False, True):
                g = m.tension(1.0, restore=restore).g
                r[f"tension {'restore' if restore else 'add'}"] = np.where(np.isfinite(g), g,
                                                                           0.0)
            m.removed[:] = False
            r["polish eps world"] = np.array([m.relaxation(1.0, q=1.0, eps=1e-6)
                                              .value(x.astype(float))])
            out[dev] = r
        bad += _devices_apart(bid, out)
    return bad


def _devices_apart(bid: str, out: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Each quantity cuDSS gave (out["gpu"]) against SuperLU's (out["cpu"]), within DEVICE_TOL
    of its largest magnitude; what parts."""
    bad = []
    worst = 0.0
    parts = []
    for key in out["cpu"]:
        a, z = out["cpu"][key], out["gpu"][key]
        fin = np.isfinite(a)
        if not np.array_equal(fin, np.isfinite(z)):
            bad.append(f"{bid} {key}: the devices part on which values are finite")
            continue
        err = float(np.abs(a[fin] - z[fin]).max() / max(np.abs(a[fin]).max(), 1e-300))
        worst = max(worst, err)
        parts.append(f"{key} {err:.1e}")
        if err > DEVICE_TOL:
            bad.append(f"{bid} {key} {err:.1e}")
    print(f"{bid}: cuDSS against SuperLU, of the largest: " + ", ".join(parts)
          + f"; worst {worst:.1e} -> {'ok' if worst <= DEVICE_TOL else 'FAILS'}", flush=True)
    return bad


def device_roads(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        case = _road_case(bid)
        out: dict[str, dict[str, np.ndarray]] = {}
        for dev in ("cpu", "gpu"):
            m = dataclasses.replace(METRIC, factorization=gap_factor.factorization_of(dev)).model(
                case.block, case.fixed, demand=POP)
            r: dict[str, np.ndarray] = {}
            for k in ROAD_STOPS:
                L = m.road_lenses(gcl.Prefix(net=0, stop=k))
                r[f"stop {k}"] = np.array([L.P, L.Pq, L.J2])
                r[f"stop {k} homes"] = L.u
            out[dev] = r
        bad += _devices_apart(bid, out)
    return bad


_CUDSS_FACTOR = gap_factor._CuDSSPlan.factor


def _perturb_one() -> contextlib.AbstractContextManager[None]:
    """The second factorization on each cuDSS plan (after the baseline's: the random
    clearing's) handed its values with the diagonal raised by PERTURB (fault `perturb`)."""
    calls: dict[int, int] = {}

    def factor(self: gap_factor._CuDSSPlan, data: np.ndarray, unknown: np.ndarray
               ) -> gap_factor.Factor:
        calls[id(self)] = calls.get(id(self), 0) + 1
        if calls[id(self)] == 2:
            p = self.pattern
            rows = np.repeat(np.arange(p.n), np.diff(p.indptr))
            data = np.where(rows == p.indices, data * (1.0 + PERTURB), data)
        return _CUDSS_FACTOR(self, data, unknown)
    return _patched(gap_factor._CuDSSPlan, "factor", factor)


# ------------------------------------------------------------------ the refined solve

def underflow(_ids: list[str]) -> list[str]:
    from scipy.sparse.linalg import spsolve
    bid = UNDERFLOW_BLOCK
    got = common.member_blocks([bid])       # under the source's building floor: ids_of refuses it
    if [g.block_id for g in got] != [bid]:
        raise ValueError(f"the source yields {[g.block_id for g in got]} for {bid}")
    [b] = got
    try:
        m = _model(b)
    except RuntimeError as e:
        print(f"{bid}: the baseline's solve raises ({e}) -> FAILS", flush=True)
        return [f"{bid}: {e}"]
    st = m.state(np.zeros(m.n))
    assert st.A is not None
    unk = st.reach.unknown
    u = m.solve(st, m.pop)[unk]
    direct = np.asarray(spsolve(st.A[unk][:, unk].tocsc(), m._rhs(st, m.pop)[unk]))
    err = float(np.abs(u - direct).max() / np.abs(direct).max())
    sub = int(((direct != 0.0) & (np.abs(direct) < np.finfo(np.float64).tiny)).sum())
    bad = []
    if not sub:
        bad.append(f"{bid}: no subnormal potential: not a block for this guard")
    if err > UNDERFLOW_TOL:
        bad.append(f"{bid}: the refined solve {err:.1e} of the largest off spsolve's")
    print(f"{bid}: {m.n} homes, {int(unk.sum())} unknowns, {sub} of them subnormal by spsolve; "
          f"the model's refined solve against it {err:.1e} of the largest -> "
          f"{'ok' if not bad else 'FAILS'}", flush=True)
    return bad


@njit
def _residual_unfloored(indptr, indices, data, b, x):  # type: ignore[no-untyped-def]
    """gap_factor._residual before the underflow fix, verbatim (fault `unfloored`): each term of
    the denominator at its own size, a row whose denominator is 0 passed over."""
    n = len(b)
    r = np.empty(n)
    berr = 0.0
    for i in range(n):
        s = b[i]
        d = abs(b[i])
        for p in range(indptr[i], indptr[i + 1]):
            v = data[p] * x[indices[p]]
            s -= v
            d += abs(v)
        r[i] = s
        if d > 0.0:
            e = abs(s) / d
            if e > berr:
                berr = e
    return berr, r


# ------------------------------------------------------------------ roads

class RoadCase(NamedTuple):
    """A block's ROAD_NETWORK area prefixes cut into its fixed mesh (the road guards')."""
    block: Block
    stops: list[int]            # each prefix's road count
    corridors: list[BaseGeometry]   # each prefix's corridor, in the block's local frame
    fixed: gcl.Fixed


def _road_case(bid: str) -> RoadCase:
    [b] = common.build_blocks([bid])
    geo = gc.geometry(b)
    pre = pd.read_parquet(rr.prefix_dir(ROAD_ROWS) / f"{bid}__{ROAD_NETWORK}.parquet")
    stops = sorted({int(mk) for mk in pre[pre["pop"] == rr.CX_POP].m})
    ordered = gpd.read_parquet(rr.ordered_path(ROAD_ROWS / "scratch", bid, ROAD_NETWORK))
    cors = [geo.local(rr.corridor_of(ordered, mk)) for mk in stops]
    with tempfile.TemporaryDirectory() as tmp:
        fx = METRIC.fixed(b, cache=Path(tmp), carving=gcl.Carving.of(geo, [cors]))
    return RoadCase(block=b, stops=stops, corridors=cors, fixed=fx)


def roads(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        case = _road_case(bid)
        b, stops, cors, carving = case.block, case.stops, case.corridors, case.fixed.carving
        m = METRIC.model(b, case.fixed, demand=POP)
        cm = gcmp.ComplexModel(b, METRIC.fs, METRIC.form)
        parts = []
        for k in ROAD_STOPS:
            L = m.road_lenses(gcl.Prefix(net=0, stop=k))
            r, _u, live = cm.scored(np.ones(m.n, dtype=bool), cors[k - 1])
            d = {"L_P": (1.0 - L.P / m.base.P) - r["L_P"],
                 "L_Pq": (1.0 - L.Pq / m.base.Pq) - r["L_Pq0.5"],
                 "L_J2": (1.0 - (L.J2 / m.base.J2) ** 0.5) - r["L_J2"]}
            apart = np.flatnonzero(L.live ^ live)
            ok = max(abs(v) for v in d.values()) <= ROAD_TOL and not len(apart)
            parts.append(f"stop {k} ({stops[k - 1]} roads): " + ", ".join(
                f"{key} {v:+.1e}" for key, v in d.items()) + f", {len(apart)} homes apart"
                + ("" if ok else " FAILS"))
            if not ok:
                bad.append(f"{bid} stop {k}")
        print(f"{bid} {ROAD_NETWORK}, {carving.n} atoms ({int((carving.stop > 0).sum())} carved): "
              + "; ".join(parts), flush=True)
    return bad


_CARVED = gcl.Carving.carved


def _one_face_closed(self: gcl.Carving, prefix: gcl.Prefix) -> np.ndarray:
    """gap_clear.Carving.carved with the largest atom the prefix carves left standing (fault
    `one-face-closed`)."""
    out = _CARVED(self, prefix)
    if out.any():
        on = np.flatnonzero(out)
        out = out.copy()
        out[on[int(np.argmax(shapely.area(self.polys[on])))]] = False
    return out


# ------------------------------------------------------------------ the pictures' fluxes

def flux(ids: list[str]) -> list[str]:
    """The `flux` guard (module docstring), and the same at a random clearing under the uniform
    leak (its drainage the street's and the leaks')."""
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        m = _model(b)
        ml = _model(b, _leaky(gx.InternalLeak(open_m=LEAK_LAM, foot_m=math.inf)))
        rng = np.random.default_rng(SEED + 6)
        states = {"baseline": (m, np.zeros(m.n))}
        for k in range(FLUX_RANDOM):
            states[f"random{k}"] = (m, _random(m, rng).astype(float))
        states["leaky random"] = (ml, _random(ml, rng).astype(float))
        parts = []
        for name, (m, k_) in states.items():
            st = m.state(k_)
            fl = gap_maps.flow_of(m, st, m.pop)
            bal = gap_maps.conservation(st, fl)
            worst = float(np.abs(bal).max())
            total = float(fl.drain.sum() / fl.inj.sum() - 1.0)
            ok = worst <= FLUX_TOL and abs(total) <= FLUX_TOL
            parts.append(f"{name}: {int(st.reach.live.sum())} open cells, worst imbalance "
                         f"{worst:.1e}, drainage / injection - 1 {total:+.1e}"
                         + ("" if ok else " FAILS"))
            if not ok:
                bad.append(f"{bid} {name}")
        print(f"{bid}: " + "; ".join(parts), flush=True)
    return bad


def _flip_one() -> contextlib.AbstractContextManager[None]:
    """gap_maps._passage_flux with the largest flux's sign flipped (fault `flip-one`)."""
    flux_of = gap_maps._passage_flux

    def flipped(Tp: np.ndarray, u: np.ndarray, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
        F = flux_of(Tp, u, ea, eb).copy()
        i = int(np.argmax(np.abs(F)))
        F[i] = -F[i]
        return F
    return _patched(gap_maps, "_passage_flux", flipped)


# ------------------------------------------------------------------ the exits: leaks

LEAKS = tuple(gx.InternalLeak(open_m=o, foot_m=f) for o, f in LEAK_PAIRS)


def _leaky(leak: gx.NoLeak | gx.InternalLeak, *,
           edge: gx.OpenEdge | gx.ClosedEdge = gx.OpenEdge()) -> gcl.ComplexMetric:
    """The runtime's metric with `leak` (no barriers, no boost: a kblock has no OSM features),
    its edge `edge` (the street ring open, as the per-block metric's, unless a guard closes it)."""
    return dataclasses.replace(METRIC, exits=gx.Exits(barriers=gx.NoBarriers(), edge=edge,
                                                      leak=leak, boost=gx.NoBoost()))


def _tri_area(fm: gcl.FullMesh) -> np.ndarray:
    """Per triangle its area (m^2)."""
    a, b, c = (fm.xy[fm.tri[:, k]] for k in range(3))
    return np.asarray(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])))


def _leak_oracle(m: gcl.Model, leak: gx.InternalLeak) -> np.ndarray:
    """The leak per cell from its definition, independently of gap_exits and the model's Graph:
    each cell's triangles' area, the cells live at baseline by a search of their own (scipy's
    components over the passages conducting at baseline and a street node joined to the cells
    with a conducting street face or a ground cell), the injection per cell from the demand's
    points with each building's footprint area; sigma / open_m^2 per m^2 of open ground,
    sigma / foot_m^2 per m^2 of footprint injected."""
    from scipy.sparse.csgraph import connected_components
    fm, c = m.fm, m.c
    area = np.bincount(c.cell, weights=_tri_area(fm), minlength=c.n)
    r0 = m.factors(np.zeros(m.n)).reach
    on = r0.Tp > 0
    street = c.n
    ground = np.unique(c.gcell[m.factors(np.zeros(m.n)).kt[fm.fa[c.pg]] > 0])
    touch = np.concatenate([c.fc[r0.Gf > 0], ground])
    rows = np.concatenate([c.ea[on], touch])
    cols = np.concatenate([c.eb[on], np.full(len(touch), street)])
    _k, lab = connected_components(sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                                 shape=(c.n + 1, c.n + 1)), directed=False)
    is_ground = np.zeros(c.n, dtype=bool)
    is_ground[ground] = True
    live = (lab[:c.n] == lab[street]) & ~is_ground
    sigma = m.fs.sigma
    pts = np.flatnonzero(m.fx.ptri >= 0)
    foot_area = np.asarray(shapely.area(np.asarray(m.block.buildings.outlines)))
    foot = np.bincount(c.cell[m.fx.ptri[pts]], weights=np.asarray(m.dem.S @ foot_area)[pts],
                       minlength=c.n)
    return (np.where(live, sigma * area / leak.open_m ** 2, 0.0)
            + np.where(live, sigma * foot / leak.foot_m ** 2, 0.0))


def _uniform_was(site: gx.LeakSite, lam: float) -> np.ndarray:
    """UniformLeak.per_cell of the region test (verbatim)."""
    return np.where(site.live, site.sigma * site.area / lam ** 2, 0.0)


def _home_was(site: gx.LeakSite, lam: float, pop_inj: np.ndarray) -> np.ndarray:
    """HomeLeak.per_cell of the region test (verbatim: its total UniformLeak's, placed by the
    population's injection `pop_inj`, System.Pc @ pop)."""
    total = float(site.sigma * site.area[site.live].sum() / lam ** 2)
    inj = np.where(site.live, pop_inj, 0.0)
    return total * inj / float(inj.sum())


def leak_fold(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        with tempfile.TemporaryDirectory() as tmp:
            fx = METRIC.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
        bare = METRIC.model(b, fx, demand=POP)
        site = bare.leak_site
        open_m2 = float(site.area[site.live].sum())
        foot_m2 = float(site.foot[site.live].sum())
        parts = []
        for s in FOLD_SHARES:
            # the region test's (s, LEAK_LAM) as the two constants on this block (unrounded)
            leak = gx.InternalLeak(
                open_m=LEAK_LAM / math.sqrt(1.0 - s) if s < 1.0 else math.inf,
                foot_m=LEAK_LAM * math.sqrt(foot_m2 / (s * open_m2)) if s > 0.0 else math.inf)
            m = _leaky(leak).model(b, fx, demand=POP)
            got = m.sys.graph.leak
            was = ((1.0 - s) * _uniform_was(m.leak_site, LEAK_LAM)
                   + s * _home_was(m.leak_site, LEAK_LAM, np.asarray(m.sys.Pc @ m.pop)))
            d_was = float(np.abs(got - was).max() / np.abs(was).max())
            oracle = _leak_oracle(m, leak)
            d_oracle = float(np.abs(got - oracle).max() / np.abs(oracle).max())
            st = m.state(np.zeros(m.n))
            A = st.A
            assert A is not None
            ref, cells = _operator(st)
            ref = (ref + sp.diags(oracle[cells])).tocsr()
            d_op = float(abs(A[cells][:, cells] - ref).max() / abs(ref).max())
            ok = ((d_was == 0.0 if s == 0.0 else d_was <= PATTERN_TOL)
                  and d_oracle <= PATTERN_TOL and d_op <= PATTERN_TOL)
            parts.append(f"s {s:g} ({leak.name}): against the region test's formulas {d_was:.1e}"
                         + (" (bit for bit)" if d_was == 0.0 else "")
                         + f", the oracle {d_oracle:.1e}, the operator {d_op:.1e}"
                         + ("" if ok else " FAILS"))
            if not ok:
                bad.append(f"{bid} s {s:g}")
        print(f"{bid} leak-fold at lam {LEAK_LAM:g}: " + "; ".join(parts), flush=True)
    return bad


_INTERNAL_PER_CELL = gx.InternalLeak.per_cell


def _lengths_swapped(self: gx.InternalLeak, site: gx.LeakSite) -> np.ndarray:
    """InternalLeak.per_cell with the open ground's length applied to the footprints and the
    footprints' to open ground (fault `lengths-swapped`)."""
    return _INTERNAL_PER_CELL(gx.InternalLeak(open_m=self.foot_m, foot_m=self.open_m), site)


def leak_assembly(ids: list[str]) -> list[str]:
    from scipy.sparse.linalg import spsolve
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for leak in LEAKS:
            m = _model(b, _leaky(leak))
            oracle = _leak_oracle(m, leak)
            rng = np.random.default_rng(SEED + 7)
            one = np.zeros(m.n)
            one[_largest(m, 1)] = 1.0
            states = {"baseline": np.zeros(m.n), "random": _random(m, rng).astype(float),
                      "the most populous cleared": one,
                      "fractional": rng.uniform(0.0, 1.0, m.n), "eps": np.full(m.n, GRAD_EPS)}
            parts = []
            for name, k_ in states.items():
                st = m.state(k_)
                A = st.A
                assert A is not None
                p = st.sys.graph.pattern
                same_structure = (np.array_equal(A.indptr, p.indptr)
                                  and np.array_equal(A.indices, p.indices))
                ref, cells = _operator(st)
                ref = (ref + sp.diags(oracle[cells])).tocsr()
                sub = A[cells][:, cells]
                err = float(abs(sub - ref).max() / abs(ref).max()) if ref.nnz else 0.0
                rhs = m._rhs(st, m.pop)
                P_fixed = float(rhs @ m.solve(st, m.pop))
                P_ref = float(rhs[cells] @ spsolve(ref.tocsc(), rhs[cells]))
                dP = abs(P_fixed / P_ref - 1.0)
                ok = same_structure and err <= PATTERN_TOL and dP <= PATTERN_P_TOL
                parts.append(f"{name}: operator {err:.1e}, P {dP:.1e}, the pattern's "
                             f"{same_structure}" + ("" if ok else " FAILS"))
                if not ok:
                    bad.append(f"{bid} {leak.name} {name}")
            print(f"{bid} {leak.name} (leak {float(oracle.sum()):.4g} on "
                  f"{int((oracle > 0).sum())} cells): " + "; ".join(parts), flush=True)
    return bad


@contextmanager
def _leak_fault(change: Callable[[np.ndarray, gx.LeakSite], np.ndarray]) -> Iterator[None]:
    """The internal leak's per_cell with `change` applied to its answer."""
    def per_cell(self: gx.InternalLeak, site: gx.LeakSite) -> np.ndarray:
        return change(_INTERNAL_PER_CELL(self, site), site)
    with _patched(gx.InternalLeak, "per_cell", per_cell):
        yield


def _drop_one(leak: np.ndarray, site: gx.LeakSite) -> np.ndarray:
    """The largest leak dropped (fault `drop-one`)."""
    out = leak.copy()
    out[int(np.argmax(out))] = 0.0
    return out


def _not_live(leak: np.ndarray, site: gx.LeakSite) -> np.ndarray:
    """A leak on the largest cell not live at baseline (fault `not-live`)."""
    out = leak.copy()
    j = int(np.argmax(np.where(site.live, -np.inf, site.area)))
    out[j] = float(leak.max())
    return out


def _every_cell(leak: np.ndarray, site: gx.LeakSite) -> np.ndarray:
    """The leak on every cell, live at baseline or not (fault `every-cell`)."""
    return site.sigma * site.area / LEAK_LAM ** 2


def enclosure(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        with tempfile.TemporaryDirectory() as tmp:
            fx = METRIC.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
        plain = METRIC.model(b, fx, demand=POP)
        parts = [f"{int(plain.walled0.sum())} walled in without leaks"]
        for leak in LEAKS:
            m = _leaky(leak).model(b, fx, demand=POP)
            d = np.flatnonzero(m.walled0 ^ plain.walled0)
            parts.append(f"{leak.name}: {int(m.walled0.sum())} walled in, {len(d)} apart")
            if len(d):
                bad.append(f"{bid} {leak.name} {d[:5].tolist()}")
        print(f"{bid}: " + "; ".join(parts) + (" -> FAILS" if any(bid in x for x in bad)
                                                 else " -> ok"), flush=True)
    return bad


def leak_gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for leak in LEAKS:
            m = _model(b, _leaky(leak))
            for where, (x0, pick) in _grad_states(m).items():
                for power in (1.0, 2.0):
                    for q in GRAD_Q:
                        err = against_differences(m.relaxation(power, q=q, eps=GRAD_EPS), x0,
                                                  pick)
                        ok = err <= GRAD_TOL
                        print(f"{bid} {leak.name} {where} J_{power:g} q {q:g}: gradient against "
                              f"differences {err:.1e} of the largest -> "
                              f"{'ok' if ok else 'FAILS'}", flush=True)
                        if not ok:
                            bad.append(f"{bid} {leak.name} {where} p{power:g} q{q:g}")
    return bad


def _sgrad_unleaked(self: gcl.Relaxation, x: np.ndarray) -> tuple[float, np.ndarray]:
    """gap_clear.Relaxation.value_sgrad with J_p's adjoint (p != 1) solved on the system without
    its leaks (fault `adjoint-unleaked`)."""
    m = self.m
    st = m.state(self.k_of(x))
    J, u, uh = self._J(st)
    p = self.power
    if p == 1.0:
        lam = u
    else:
        bare = m.sys.with_graph(m.sys.graph.with_leak(np.zeros(m.sys.n)))
        st2 = m._state(bare, self.k_of(x), prefix=gcl._NO_ROADS)
        lam = m.solve(st2, np.where(m.nodem, 0.0, m.inj * p * np.nan_to_num(uh) ** (p - 1.0)))
    g = m.grad(st, u, lam) * (1.0 - self.eps)
    if self.q != 1.0:
        g = g * self.q * np.maximum(self.proj(x), 1e-300) ** (self.q - 1.0)
    return J, g


# ------------------------------------------------------------------ synthetic blocks
# (region_leaks.synthetic_block: in 19570's frame, local metres)

def _square(x: float, y: float, side: float) -> BaseGeometry:
    return shapely.box(x, y, x + side, y + side)


def _open_block(block_id: str, homes: list[BaseGeometry]) -> Block:
    """SCALE_L of open ground, its whole edge the street, with `homes`."""
    L = SCALE_L
    return region_leaks.synthetic_block(
        block_id, shapely.box(0.0, 0.0, L, L),
        shapely.LineString(shapely.box(0.0, 0.0, L, L).exterior.coords), homes, None)


def _scale_block() -> Block:
    """SCALE_BLOCK: one 2 m home at the open ground's centre."""
    c = SCALE_L / 2.0
    return _open_block("synthetic-open", [_square(c - 1.0, c - 1.0, 2.0)])


def leak_scale(_ids: list[str]) -> list[str]:
    from scipy.special import k0
    b = _scale_block()
    with tempfile.TemporaryDirectory() as tmp:
        fx = METRIC.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
    bad = []
    for open_m, foot_m in SCALE_PAIRS:
        leak = gx.InternalLeak(open_m=open_m, foot_m=foot_m)
        m = _leaky(leak).model(b, fx, demand=POP)
        st = m.state(np.zeros(m.n))
        u = m.solve(st, m.pop)
        cg = gap_maps.cell_geometry(m.fm, m.c)
        src = np.array(m.geo.polys[0].centroid.coords[0])
        r = np.hypot(*(cg.xy - src).T)
        on = st.reach.unknown & (r >= SCALE_R[0]) & (r <= SCALE_R[1])
        edges = np.geomspace(SCALE_R[0], SCALE_R[1], SCALE_BINS + 1)
        which = np.digitize(r[on], edges) - 1
        lam_open = open_m
        ratio = u[on] / k0(r[on] / lam_open)
        per_bin = np.array([np.median(ratio[which == k]) for k in range(SCALE_BINS)])
        rhs = m._rhs(st, m.pop)
        # the current the home's own footprint's destination drains (none without a footprint
        # leak), the rest spreading on the open ground
        own = leak.per_foot(m.leak_site) * np.where(m.leak_site.live, m.leak_site.foot, 0.0) * u
        Q = float(rhs.sum() - own.sum())
        expect = Q / (2.0 * np.pi * m.fs.sigma)
        spread = float(per_bin.max() / per_bin.min() - 1.0)
        level = float(np.median(per_bin) / expect - 1.0)
        sizes = cg.area[on]
        ok = spread <= SCALE_TOL and abs(level) <= SCALE_TOL
        print(f"leak-scale {leak.name}: open ground {SCALE_L:g} m, {int(on.sum())} cells at "
              f"{SCALE_R[0]:g} -- "
              f"{SCALE_R[1]:g} m (areas {sizes.min():.2g} -- {sizes.max():.2g} m^2); u / K0(r / "
              f"lam') per bin {np.round(per_bin / expect, 4).tolist()} of Q' / (2 pi sigma) = "
              f"{expect:.4g} (Q' {Q / float(rhs.sum()):.4g} of the home's current): spread "
              f"{spread:.1e}, level {level:+.1e} -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"{leak.name}: spread {spread:.1e}, level {level:+.1e}")
    return bad


def _per_cell(leak: np.ndarray, site: gx.LeakSite) -> np.ndarray:
    """The same conductance on every live cell, the leak's mean (fault `per-cell`)."""
    return np.where(site.live, float(leak[site.live].mean()), 0.0)


# the barriers guard's block (local metres): a street on the south edge only, a railway across
# at y 40 (BARRIERS' strip), a crossing node's path at x 30 and a footbridge at x 90, a water
# channel x 58 -- 62 from the railway to the north edge; homes south, north-west and north-east
_BAR_HOMES = {"south": [(10.0, 10.0), (50.0, 10.0), (100.0, 10.0), (70.0, 25.0)],
              "north-west": [(10.0, 60.0), (40.0, 55.0)],
              "north-east": [(75.0, 60.0), (100.0, 55.0)]}


def _barrier_block(*, node: bool, bridge: bool, path: bool = True) -> Block:
    """The barriers guard's block with its crossing node (`node`) and its footbridge (`bridge`) or
    without them (the path and the footway without their crossing tags); `path` False: the node's
    path missing from the features (a crossing tagged on the track alone: its gap across it)."""
    keep = [True, path, True]
    ways = gpd.GeoDataFrame(
        {"highway": ["", "footway", "footway"], "railway": ["rail", "", ""],
         "waterway": ["", "", ""], "surface": ["", "", ""], "bridge": [False, False, bridge],
         "tunnel": [False, False, False]},
        geometry=[shapely.LineString([(-10, 40), (130, 40)]),
                  shapely.LineString([(30, 20), (30, 60)]),
                  shapely.LineString([(90, 30), (90, 50)])])
    ways = cast(gpd.GeoDataFrame, ways[keep].reset_index(drop=True))
    nodes = gpd.GeoDataFrame({"railway": ["crossing"] if node else []},
                             geometry=[shapely.Point(30, 40)] if node else [])
    areas = gpd.GeoDataFrame({"natural": ["water"], "landuse": [""], "waterway": [""]},
                             geometry=[shapely.box(58, 43, 62, 81)])
    homes = [_square(x, y, 4.0) for group in _BAR_HOMES.values() for x, y in group]
    return region_leaks.synthetic_block("synthetic-barriers", shapely.box(0, 0, 120, 80),
                      shapely.LineString([(0, 0), (120, 0)]), homes,
                      gx.Osm(ways=ways, nodes=nodes, areas=areas))


def barriers(ids: list[str]) -> list[str]:
    metric = dataclasses.replace(METRIC, exits=gx.Exits(barriers=region_leaks.BARRIERS,
                                                        edge=gx.OpenEdge(), leak=gx.NoLeak(),
                                                        boost=gx.NoBoost()))
    groups = np.repeat(list(_BAR_HOMES), [len(v) for v in _BAR_HOMES.values()])
    bad = []
    parts = []
    for node, bridge, path, expect in ((True, True, True, ()), (False, True, True, ("north-west",)),
                                       (True, False, True, ("north-east",)),
                                       (False, False, True, ("north-west", "north-east")),
                                       (True, True, False, ())):
        b = _barrier_block(node=node, bridge=bridge, path=path)
        bp = region_leaks.BARRIERS.parts(b)
        m = _model(b, metric)
        walled = sorted({str(g) for g in groups[m.walled0]})
        want = sorted(set(expect))
        ok = walled == want
        if node and bridge and path:
            # every open triangle inside a barrier (the strips and areas before the gaps) lies in a
            # crossing's gap
            geo = m.geo
            whole = geo.local(shapely.union(bp.strips, bp.areas))
            gaps = geo.local(bp.gaps).buffer(1e-6)
            cen = m.fm.xy[m.fm.tri].mean(axis=1)
            pts = shapely.points(cen[:, 0], cen[:, 1])
            open_ = m.fm.open_tri
            inside = open_ & shapely.contains(whole, pts)
            stray = int((inside & ~shapely.contains(gaps, pts)).sum())
            ok = ok and stray == 0 and int(inside.sum()) > 0
            parts.append(f"both crossings: {int(inside.sum())} open triangles inside a barrier, "
                         f"{stray} outside the gaps")
        parts.append(f"{'path' if node else 'no path'}"
                     f"{'' if path else ' (its node alone, the way missing)'}, "
                     f"{'bridge' if bridge else 'no bridge'}: walled in {walled or 'none'} "
                     f"(expected {want or 'none'})" + ("" if ok else " FAILS"))
        if not ok:
            bad.append(f"node {node} bridge {bridge} path {path}")
    print("barriers: " + "; ".join(parts), flush=True)
    for b in common.build_blocks(ids):
        bad += _barriers_on_block(b)
    return bad


def _runtime() -> gcl.ComplexMetric:
    """RUNTIME on this code (metrics.EXITS on the per-block metric)."""
    return dataclasses.replace(METRIC, exits=metrics.EXITS)


def _leak_only() -> gcl.ComplexMetric:
    """The per-block metric with RUNTIME's leak alone (no barriers, no boost)."""
    return dataclasses.replace(METRIC, exits=dataclasses.replace(gx.per_block(),
                                                                 leak=metrics.EXITS.leak))


def _runtime_barriers() -> gx.OsmBarriers:
    return metrics.BARRIERS


def _centroids(fm: gcl.FullMesh) -> np.ndarray:
    cen = fm.xy[fm.tri].mean(axis=1)
    return np.asarray(shapely.points(cen[:, 0], cen[:, 1]))


def _barriers_on_block(b: Block) -> list[str]:
    """A real block a railway or water meets (the premise, read from its own features afresh,
    common.own_osm): under RUNTIME its fixed mesh has no triangle inside the cut (the barrier
    within the block, less BARRIER_EDGE_M) where the leak-only metric's mesh has some, the mesh's
    cache key is not the uncut block's, and the metric's name carries the barriers' part."""
    bid = b.block_id
    bar = _runtime_barriers()
    own = dataclasses.replace(b, attrs={**b.attrs, "osm": common.own_osm(b)})
    want = shapely.intersection(b.boundary, bar.parts(own).barrier)
    if want.area <= 0.0:
        return [f"{bid}: no railway or water meets it (the premise)"]
    counts = {}
    for key, metric in (("runtime", _runtime()), ("leak-only", _leak_only())):
        with tempfile.TemporaryDirectory() as tmp:
            fx = metric.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
        geo = gc.geometry(metric.exits.barriers.block(b))
        core = geo.local(shapely.buffer(want, -BARRIER_EDGE_M))
        counts[key] = int(shapely.contains(core, _centroids(fx.fm)).sum())
    runtime = _runtime()
    keys = {k: gcl.Fixed.key(blk, runtime.spec, runtime.fs, runtime.form)
            for k, blk in (("cut", bar.block(b)), ("uncut", b))}
    # the atoms are the footprints in the cut mesh's own frame (a cut at the block's edge moves
    # its origin), and the complex's walled-in homes are the flow model's (Model.free_first)
    m = _model(b, runtime)
    moved = gc.origin_of(b) != gc.origin_of(bar.block(b))
    atoms = bool(np.all(shapely.equals_exact(m.fx.carving.polys, m.geo.polys, 1e-6)))
    try:
        m.free_first(budget=BUDGET, held=np.zeros(m.n, dtype=bool))
        agree = True
    except RuntimeError as e:
        agree = False
        print(f"barriers on {bid}: {e}", flush=True)
    ok = (counts["runtime"] == 0 < counts["leak-only"] and keys["cut"] != keys["uncut"]
          and f"+x{bar.name}" in runtime.name and atoms and agree)
    print(f"barriers on {bid}: the cut {want.area:.0f} m^2 ({want.area / b.boundary.area:.4f} of "
          f"the block); triangles inside it {counts['runtime']} under RUNTIME, "
          f"{counts['leak-only']} under the leak alone; mesh keys "
          f"{'differ' if keys['cut'] != keys['uncut'] else 'EQUAL'}; the name carries "
          f"{bar.name}: {f'+x{bar.name}' in runtime.name}; the frame "
          f"{'moved' if moved else 'kept'}, the atoms on the footprints {atoms}, the walled-in "
          f"homes the flow model's {agree} -> {'ok' if ok else 'FAILS'}", flush=True)
    return [] if ok else [f"{bid}: {counts}, keys {keys}, atoms {atoms}, walled alike {agree}"]


def _features_dropped() -> contextlib.AbstractContextManager[None]:
    """Fault `features-dropped`: every block built with no OSM features (empty frames)."""
    import regions
    with_osm = regions.with_osm

    def empty(b: Block, extract: regions.Extract) -> Block:
        got = with_osm(b, extract)
        osm = gx.osm_in(got)
        return dataclasses.replace(got, attrs={**got.attrs, "osm": gx.Osm(
            ways=cast(gpd.GeoDataFrame, osm.ways.iloc[:0]),
            nodes=cast(gpd.GeoDataFrame, osm.nodes.iloc[:0]),
            areas=cast(gpd.GeoDataFrame, osm.areas.iloc[:0]))})
    return _patched(regions, "with_osm", empty)


def _no_feature_osm(b: Block) -> gx.Osm:
    """`b`'s own features (common.own_osm) less every drivable way (metrics.DRIVE_CLASSES, `_link`
    folded) and every feature a barrier is made of: the block as it would read if no barrier and
    no drivable way met it (every real block has a drivable street on its ring: the survey,
    consolidated_rows/features), its footways, paths and the rest kept."""
    osm = common.own_osm(b)
    w = osm.ways
    drive = w.highway.str.removesuffix("_link").isin(metrics.DRIVE_CLASSES)
    barrier = w.railway.isin(gx.RAIL) | w.waterway.isin(gx.WATERWAYS)
    a = osm.areas
    water = (a.natural.isin(gx.WATER_NATURAL) | a.landuse.isin(gx.WATER_LANDUSE)
             | a.waterway.isin(gx.WATER_WATERWAY))
    return gx.Osm(ways=cast(gpd.GeoDataFrame, w[~drive & ~barrier].reset_index(drop=True)),
                  nodes=cast(gpd.GeoDataFrame, osm.nodes[~osm.nodes.railway.isin(
                      gx.CROSSING_NODES)].reset_index(drop=True)),
                  areas=cast(gpd.GeoDataFrame, a[~water].reset_index(drop=True)))


def _attached_osm(b: Block) -> gx.Osm:
    """The features `no-features` attaches to its block (fault `neighbour-features`: another
    block's)."""
    return _no_feature_osm(b)


def _neighbour_osm(b: Block) -> gx.Osm:
    """The features meeting the kblock that touches `b` and shares the most of the boost's ways
    with it (a drivable way tagged paved within BOOST_M of its polygon): a neighbouring block's
    (fault `neighbour-features`)."""
    import regions
    src = common._source()
    allb = gpd.read_parquet(src.blocks_path, columns=["block_id", "geometry"])
    allb = allb.to_crs(b.crs)
    near = allb[(allb.block_id.astype(str) != b.block_id)
                & (shapely.distance(allb.geometry.to_numpy(), b.boundary) < 0.5)]
    extract = regions.extract_of(src.blocks_path, b.block_id.split(".", 1)[0])
    best, shared = None, -1.0
    for bid, poly in zip(near.block_id.astype(str), near.geometry, strict=True):
        osm = extract.meeting(poly)
        boosted = (osm.ways.highway.str.removesuffix("_link").isin(metrics.BOOST.classes).to_numpy()
                   & metrics.BOOST.surfaces.of(osm.ways.surface))
        w = osm.ways[boosted]
        reach = shapely.intersection(w.geometry.to_numpy(), b.boundary.buffer(gx.BOOST_M))
        got = float(shapely.length(reach).sum())
        if got > shared:
            best, shared = osm, got
    if best is None:
        raise ValueError(f"{b.block_id}: no kblock touches it")
    return best


def no_features(ids: list[str]) -> list[str]:
    """On each block, with the features no barrier or boost reads (`_no_feature_osm`: the premise,
    checked on them), RUNTIME's baseline (P0, P_{1/2},0, J_2,0, every home's escape time, the homes
    walled in) equals the leak-only metric's to NO_FEATURES_TOL: the barriers and the boost change
    nothing where they meet nothing. Fault `neighbour-features`: the block carries the features of
    a neighbouring kblock (its paved drivable streets along the shared ring)."""
    bad = []
    for b in common.build_blocks(ids):
        bid = b.block_id
        premise = _no_feature_osm(b)
        pb = dataclasses.replace(b, attrs={**b.attrs, "osm": premise})
        bar = _runtime_barriers()
        if (shapely.intersects(b.boundary, bar.parts(pb).barrier)
                or premise.ways.highway.str.removesuffix("_link").isin(
                    metrics.DRIVE_CLASSES).any()):
            bad.append(f"{bid}: the premise (no barrier, no drivable way) does not hold")
            continue
        bare = dataclasses.replace(b, attrs={**b.attrs, "osm": _attached_osm(b)})
        rt, lo = _model(bare, _runtime()), _model(b, _leak_only())
        rel = {k: abs(getattr(rt.base, k) / getattr(lo.base, k) - 1.0) for k in ("P", "Pq", "J2")}
        live = np.isfinite(lo.u_pop0) & (lo.u_pop0 > 0)
        homes = float(np.max(np.abs(rt.u_pop0[live] / lo.u_pop0[live] - 1.0))) if live.any() \
            else 0.0
        same_walled = bool(np.array_equal(rt.walled0, lo.walled0))
        worst = max(*rel.values(), homes)
        ok = worst <= NO_FEATURES_TOL and same_walled
        print(f"no-features on {bid} ({len(premise.ways)} ways kept, none drivable, no barrier): "
              f"RUNTIME against the leak alone, P0 {rel['P']:.1e}, Pq0 {rel['Pq']:.1e}, J2_0 "
              f"{rel['J2']:.1e}, homes {homes:.1e}, walled in alike {same_walled} -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"{bid}: {worst:.1e} apart")
    return bad


_BARRIER_PARTS = gx.OsmBarriers.parts


def _gap_closed(self: gx.OsmBarriers, block: Block) -> gx.BarrierParts:
    """OsmBarriers.parts with the crossings' gaps not cut (fault `gap-closed`)."""
    p = _BARRIER_PARTS(self, block)
    return p._replace(barrier=shapely.union(p.strips, p.areas))


# ------------------------------------------------------------------ the street boost

def _boost_block() -> Block:
    """STREET_BOOST_BLOCK: 120 x 80 m, its south edge the street; an asphalt residential way
    across it (y 40), an asphalt footway (y 65), a paved tertiary link (x 20, y 5 -- 30), an
    unpaved residential way (y 25, x 30 -- 110) and an untagged tertiary (x 112, y 5 -- 30); homes
    clear of them."""
    n = 5
    ways = gpd.GeoDataFrame(
        {"highway": ["residential", "footway", "tertiary_link", "residential", "tertiary"],
         "railway": [""] * n, "waterway": [""] * n,
         "surface": ["asphalt", "asphalt", "paved", "unpaved", ""],
         "bridge": [False] * n, "tunnel": [False] * n},
        geometry=[shapely.LineString([(-10, 40), (130, 40)]),
                  shapely.LineString([(10, 65), (110, 65)]),
                  shapely.LineString([(20, 5), (20, 30)]),
                  shapely.LineString([(30, 25), (110, 25)]),
                  shapely.LineString([(112, 5), (112, 30)])])
    empty = gpd.GeoDataFrame({"railway": []}, geometry=[])
    areas = gpd.GeoDataFrame({"natural": [], "landuse": [], "waterway": []}, geometry=[])
    homes = [_square(x, y, 4.0) for x, y in ((60.0, 15.0), (95.0, 50.0), (40.0, 72.0))]
    return region_leaks.synthetic_block("synthetic-boost", shapely.box(0, 0, 120, 80),
                                        shapely.LineString([(0, 0), (120, 0)]), homes,
                                        gx.Osm(ways=ways, nodes=empty, areas=areas))


def street_boost(ids: list[str]) -> list[str]:
    boost = gx.StreetBoost(factor=10.0, surfaces=gx.PAVED, classes=("residential", "tertiary"))
    b = _boost_block()
    m = _model(b, dataclasses.replace(METRIC, exits=gx.Exits(
        barriers=gx.NoBarriers(), edge=gx.OpenEdge(), leak=gx.NoLeak(), boost=boost)))
    fm0 = m.fx.fm
    cen = fm0.xy[fm0.tri].mean(axis=1)
    pts = shapely.points(cen[:, 0], cen[:, 1])
    geo = m.geo
    ways = gx.osm_in(b).ways
    lines = {f"{h} {u}": geo.local(g) for h, u, g in zip(ways.highway, ways.surface,
                                                         ways.geometry, strict=True)}
    near = {k: np.asarray(shapely.dwithin(g, pts, gx.BOOST_M - 0.5)) for k, g in lines.items()}
    far = ~np.asarray(shapely.dwithin(lines["residential asphalt"], pts, gx.BOOST_M + 0.5))
    on = boost.on_way(fm0, geo, b)
    # the passages' weights: 10x between two boosted triangles, as they were between two others
    both = on[fm0.pa] & on[fm0.pb]
    none = ~on[fm0.pa] & ~on[fm0.pb]
    fin = np.isfinite(fm0.T) & (fm0.T > 0)
    ratio_on = m.fm.T[both & fin] / fm0.T[both & fin]
    ratio_off = m.fm.T[none & fin] / fm0.T[none & fin]
    def count(k: str, mask: np.ndarray) -> tuple[int, int]:
        return int((near[k] & mask & on).sum()), int((near[k] & mask).sum())
    every = np.ones(len(on), dtype=bool)
    res, link = count("residential asphalt", every), count("tertiary_link paved", every)
    foot = count("footway asphalt", far)
    unpaved, untagged = count("residential unpaved", far), count("tertiary ", every)
    ok = (res[0] == res[1] > 0 and link[0] == link[1] > 0 and foot[0] == 0 < foot[1]
          and unpaved[0] == 0 < unpaved[1] and untagged[0] == 0 < untagged[1]
          and bool(np.allclose(ratio_on, 10.0, rtol=1e-12))
          and bool(np.allclose(ratio_off, 1.0, rtol=1e-12)))
    print(f"street-boost ({boost.name}): boosted triangles on the asphalt residential way "
          f"{res[0]} of {res[1]}, on the paved tertiary link {link[0]} of {link[1]}, on the "
          f"asphalt footway {foot[0]} of {foot[1]}, on the unpaved residential way {unpaved[0]} "
          f"of {unpaved[1]}, on the untagged tertiary {untagged[0]} of {untagged[1]}; passages "
          f"between boosted triangles x{float(ratio_on.min()):.12g} -- "
          f"{float(ratio_on.max()):.12g}, between others x{float(ratio_off.min()):.12g} -- "
          f"{float(ratio_off.max()):.12g} -> {'ok' if ok else 'FAILS'}", flush=True)
    out = [] if ok else [f"residential {res}, link {link}, footway {foot}, unpaved {unpaved}, "
                         f"untagged {untagged}"]
    for b in common.build_blocks(ids):
        out += _boost_on_block(b)
    return out


FOOT_CLASSES = ("footway", "path")      # the real block's ways the boost must leave alone


def _boost_on_block(b: Block) -> list[str]:
    """A real block with a paved drivable way, an unpaved or untagged one and a footway inside it
    (the premise, from its own features read afresh): under RUNTIME every triangle within
    BOOST_M - 0.5 m of a paved drivable way is boosted, none within BOOST_M - 0.5 m of a footway or
    path, or of a drivable way not tagged paved, and beyond BOOST_M + 0.5 m of every paved
    drivable way, and a passage between two boosted triangles conducts 10x, between two others as
    before."""
    bid = b.block_id
    runtime = _runtime()
    boost = metrics.BOOST
    m = _model(b, runtime)
    fm0, geo = m.fx.fm, m.geo
    pts = _centroids(fm0)
    ways = common.own_osm(b).ways
    cls = ways.highway.str.removesuffix("_link")
    paved = ways.surface.isin(gx.PAVED.values)
    drive = geo.local(shapely.union_all(
        ways.geometry[cls.isin(boost.classes) & paved].to_numpy()))
    rough = geo.local(shapely.union_all(
        ways.geometry[cls.isin(boost.classes) & ~paved].to_numpy()))
    foot = geo.local(shapely.union_all(ways.geometry[cls.isin(FOOT_CLASSES)].to_numpy()))
    near_drive = np.asarray(shapely.dwithin(drive, pts, gx.BOOST_M - 0.5))
    far_drive = ~np.asarray(shapely.dwithin(drive, pts, gx.BOOST_M + 0.5))
    near_foot = np.asarray(shapely.dwithin(foot, pts, gx.BOOST_M - 0.5)) & far_drive
    near_rough = np.asarray(shapely.dwithin(rough, pts, gx.BOOST_M - 0.5)) & far_drive
    on = boost.on_way(fm0, geo, b)
    both = on[fm0.pa] & on[fm0.pb]
    none = ~on[fm0.pa] & ~on[fm0.pb]
    fin = np.isfinite(fm0.T) & (fm0.T > 0)
    ratio_on = m.fm.T[both & fin] / fm0.T[both & fin]
    ratio_off = m.fm.T[none & fin] / fm0.T[none & fin]
    d = int((near_drive & on).sum()), int(near_drive.sum())
    f = int((near_foot & on).sum()), int(near_foot.sum())
    r = int((near_rough & on).sum()), int(near_rough.sum())
    ok = (d[0] == d[1] > 0 and f[0] == 0 < f[1] and r[0] == 0 < r[1] and len(ratio_on) > 0
          and bool(np.allclose(ratio_on, boost.factor, rtol=1e-12))
          and bool(np.allclose(ratio_off, 1.0, rtol=1e-12)))
    print(f"street-boost on {bid}: boosted triangles near a paved drivable way {d[0]} of {d[1]}, "
          f"near a footway or path only {f[0]} of {f[1]}, near a drivable way not tagged paved "
          f"only {r[0]} of {r[1]}; passages between boosted triangles "
          f"x{float(ratio_on.min()) if len(ratio_on) else float('nan'):.12g}, between others "
          f"x{float(ratio_off.max()):.12g} -> {'ok' if ok else 'FAILS'}", flush=True)
    return [] if ok else [f"{bid}: paved drivable {d}, footway {f}, not paved {r}"]


_ON_WAY = gx.StreetBoost.on_way


def _every_class(self: gx.StreetBoost, fm: gcl.FullMesh, geo: gc.Geometry,
                 block: Block) -> np.ndarray:
    """StreetBoost.on_way with its class filter dropped: every way with a highway tag (fault
    `classes-dropped`)."""
    every = sorted(set(gx.osm_in(block).ways.highway.str.removesuffix("_link")) - {""})
    return np.asarray(_ON_WAY(dataclasses.replace(self, classes=tuple(every)), fm, geo, block))


def _any_surface(self: gx.StreetBoost, fm: gcl.FullMesh, geo: gc.Geometry,
                 block: Block) -> np.ndarray:
    """StreetBoost.on_way with its surface filter dropped: a way of its classes boosted whatever
    its surface tag (fault `surfaces-dropped`)."""
    return np.asarray(_ON_WAY(dataclasses.replace(self, surfaces=gx.AnySurface()), fm, geo,
                              block))


# ------------------------------------------------------------------ the road boost

def road_boost(ids: list[str]) -> list[str]:
    boost = gx.StreetBoost(factor=10.0, surfaces=gx.PAVED, classes=("residential", "tertiary"))
    b = _boost_block()
    out = _road_boost_on(b, dataclasses.replace(METRIC, exits=gx.Exits(
        barriers=gx.NoBarriers(), edge=gx.OpenEdge(), leak=gx.NoLeak(), boost=boost)))
    out += _road_boost_on(b, METRIC)
    for b in common.build_blocks(ids):
        out += _road_boost_on(b, _runtime())
    return out


def _road_boost_on(b: Block, metric: gcl.ComplexMetric) -> list[str]:
    """`b` carved by _cross_carving, on `metric`: at each road prefix a passage between two
    triangles whose centroids lie in its corridor conducts the exits' road factor times the
    unboosted mesh's, one between two triangles outside it as the exits' mesh (the ways' boost
    kept there), and the state's reach is its mesh's; a clearing's state is on the exits' mesh."""
    bid = b.block_id
    with tempfile.TemporaryDirectory() as tmp:
        fx = metric.fixed(b, cache=Path(tmp), carving=_cross_carving(gc.geometry(b)))
    m = metric.model(b, fx, demand=POP)
    road = metric.exits.boost.road
    fm0 = m.fx.fm
    cen = fm0.xy[fm0.tri].mean(axis=1)
    fin = np.isfinite(fm0.T) & (fm0.T > 0)
    ok = m.factors(np.zeros(m.n)).fm is m.fm
    parts = []
    for stop in range(1, m.carving.stops[0] + 1):
        st = m.road_factors(gcl.Prefix(net=0, stop=stop))
        inside = shapely.contains_xy(m.carving.corridors[0][stop - 1], cen[:, 0], cen[:, 1])
        both_in = inside[fm0.pa] & inside[fm0.pb] & fin
        both_out = ~inside[fm0.pa] & ~inside[fm0.pb]
        ratio = st.fm.T[both_in] / fm0.T[both_in]
        same_out = bool(np.array_equal(st.fm.T[both_out], m.fm.T[both_out]))
        tp = bool(np.array_equal(st.reach.Tp, m.sys.graph.reach(st.fm, st.kt, st.kp).Tp))
        ok = (ok and int(both_in.sum()) > 0 and bool(np.allclose(ratio, road, rtol=1e-12))
              and same_out and tp)
        parts.append(f"stop {stop}: {int(both_in.sum())} passages inside its corridor "
                     f"x{float(ratio.min()):.12g} -- {float(ratio.max()):.12g} (the road's "
                     f"x{road:g}), outside {'as' if same_out else 'APART FROM'} the exits' mesh, "
                     f"its reach {'its mesh' if tp else 'NOT its mesh'}")
    print(f"road-boost on {bid} ({metric.exits.boost.name or 'no boost'}): " + "; ".join(parts)
          + f" -> {'ok' if ok else 'FAILS'}", flush=True)
    return [] if ok else [f"{bid} ({metric.exits.boost.name or 'no boost'})"]


def _road_unboosted(self: gcl.Model, prefix: gcl.Prefix) -> gcl.FullMesh:
    """Model.mesh_of leaving a road prefix's corridor unboosted (fault `unboosted-road`)."""
    return self.fm


def _road_without_ways(self: gcl.Model, prefix: gcl.Prefix) -> gcl.FullMesh:
    """Model.mesh_of boosting a road prefix's corridor on the unboosted mesh, the ways' boost
    dropped (fault `ways-dropped`)."""
    if prefix.stop == 0:
        return self.fm
    on = self.carving.paved(prefix, self.cen_)
    return self.exits.edge.mesh(gx.boosted(self.fx.fm,
                                           np.where(on, self.exits.boost.road, 1.0)))


# ------------------------------------------------------------------ the exits' identity

def exits_identity(_ids: list[str]) -> list[str]:
    bad = []
    digest = gcl.code_digest()
    n_const = len(gcl.code_constants())
    if NAME != PER_BLOCK or f"+v{digest}" not in PER_BLOCK or n_const != PER_BLOCK_CONSTANTS:
        bad.append(f"the per-block metric is {NAME} with {n_const} constants digesting to "
                   f"{digest}, not {PER_BLOCK} with {PER_BLOCK_CONSTANTS}")
    try:
        runtime = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())
        rt_ok = (metrics.RUNTIME == PER_BLOCK + metrics.EXITS.suffix
                 and runtime.name == metrics.RUNTIME and runtime.exits == metrics.EXITS
                 and region_leaks.metric_of("T500b90D10", "simp", "cpu").name == metrics.RUNTIME)
    except ValueError as e:
        rt_ok = False
        print(f"RUNTIME: {e}", flush=True)
    if not rt_ok:
        bad.append(f"RUNTIME {metrics.RUNTIME} is not the per-block metric with metrics.EXITS "
                   "read back, or not the region's T500b90D10")
    configs = {label: c.exits for label, c in region_leaks.CONFIGS.items()}
    for leak in (gx.InternalLeak(open_m=math.inf, foot_m=135.57463805510008),
                 gx.InternalLeak(open_m=250.0, foot_m=math.inf),
                 gx.InternalLeak(open_m=1581.0, foot_m=148.0),
                 gx.InternalLeak(open_m=0.1 + 0.2, foot_m=3e-7)):
        configs[f"{leak.name}S3"] = gx.Exits(barriers=region_leaks.BARRIERS, edge=gx.OpenEdge(),
                                            leak=leak, boost=region_leaks.BOOST)
        configs[f"{leak.name} closed"] = gx.Exits(barriers=region_leaks.BARRIERS,
                                                  edge=gx.ClosedEdge(), leak=leak,
                                                  boost=gx.NoBoost())
    configs["Lo500 alone"] = gx.Exits(barriers=gx.NoBarriers(), edge=gx.OpenEdge(),
                                      leak=gx.InternalLeak(open_m=500.0, foot_m=math.inf),
                                      boost=gx.NoBoost())
    names: dict[str, str] = {}
    for label, exits in configs.items():
        for preset in region_leaks.PRESETS:
            metric = region_leaks.CHEAP.on(dataclasses.replace(METRIC, exits=exits)) \
                if preset == "cheap" else dataclasses.replace(METRIC, exits=exits)
            name = metric.name
            try:
                back = gcl.complex_of(name, factorization=gap_factor.SuperLU())
                same = back.exits == exits and back.search_eps == metric.search_eps
            except ValueError as e:
                same = False
                print(f"{label} {preset}: {e}", flush=True)
            if not same:
                bad.append(f"{label} {preset}: {name} does not parse back")
            if name in names:
                bad.append(f"{label} {preset} and {names[name]} share {name}")
            names[name] = f"{label} {preset}"
    stored = _stored_names()
    for where, label, preset, name in stored:
        try:
            want = region_leaks.metric_of(label, preset, "cpu").name
            spelt = gcl.complex_of(name, factorization=gap_factor.SuperLU()).name
        except (KeyError, ValueError) as e:
            want, spelt = "", f"{e!r}"
        if not name == spelt == want:
            bad.append(f"{where}: {label} {preset}'s {name} is not its configuration's "
                       f"{want or '(none)'} ({spelt})")
    print(f"exits-identity: the per-block metric {NAME} ({n_const} constants, digest {digest}); "
          f"{len(names)} names of {2 * len(configs)} configurations x presets, e.g. "
          f"{metric.name}; {len(stored)} stored names of the region's runs and baselines -> "
          f"{'ok' if not bad else 'FAILS: ' + '; '.join(bad[:4])}", flush=True)
    return bad


class _Stored(NamedTuple):
    where: str
    label: str
    preset: str
    name: str


def _stored_names() -> list[_Stored]:
    """The region's stored names: its runs' metrics by label and preset (the committed
    summary_lenses.csv, and the runs' rows where they are, each under its directory's label and
    preset, which its row must name too), and its baselines' (summary_baseline.csv: the runtime's
    name and the configuration's suffix, SIMP's metric)."""
    rows = region_leaks.ROWS
    out = []
    lens = pd.read_csv(rows / "summary_lenses.csv")
    for label, preset, name in zip(lens["label"], lens["preset"], lens["metric"], strict=True):
        out.append(_Stored("summary_lenses.csv", str(label), str(preset), str(name)))
    for path in sorted((rows / "runs").glob(f"*/*/*/{region_leaks.REGION}.parquet")):
        r = pd.read_parquet(path).iloc[0]
        label, preset = path.parent.parent.parent.name, path.parent.parent.name
        if path.parent.name != r["metric"]:
            out.append(_Stored(str(path.relative_to(rows)), f"{label}?", preset,
                               f"its directory {path.parent.name}, its row {r['metric']}"))
            continue
        named = f"{label} {preset}" if (r["label"], r["preset"]) == (label, preset) \
            else f"its row's {r['label']} {r['preset']}"
        out.append(_Stored(str(path.relative_to(rows)) + ("" if named == f"{label} {preset}"
                                                          else f" ({named})"),
                           label if named == f"{label} {preset}" else f"{label}?", preset,
                           str(r["metric"])))
    base = pd.read_csv(rows / "summary_baseline.csv", keep_default_na=False)
    for label, suffix in zip(base["label"], base["metric"], strict=True):
        out.append(_Stored("summary_baseline.csv", str(label), "simp", NAME + str(suffix)))
    return out


# ------------------------------------------------------------------ the trips' calibration

K0_MEDIAN = 1.2571513906775686   # the median of (r / lam^2) K0(r / lam): x K1(x) = 1 / 2


def _image_rows(srcs: np.ndarray, box: tuple[float, float, float, float], lam: float,
                nbins: int, *, sign: float) -> np.ndarray:
    """Per source (local metres) its trips on open ground `box`: the screened point source with
    its first images in the four edges, `sign` -1 (the edge held at 0: an open edge) or +1 (no
    current through it: a closed edge), its leak sigma u / lam^2 integrated on a polar grid about
    it (TRIP_ORACLE_DR, TRIP_ORACLE_NT; the ground outside the box apart), binned as region_leaks
    bins a home's trips, per unit current."""
    from scipy.special import k0
    x0, y0, x1, y1 = box
    r = (np.arange(int(nbins * region_leaks.TRIP_BIN_M / TRIP_ORACLE_DR)) + 0.5) * TRIP_ORACLE_DR
    th = (np.arange(TRIP_ORACLE_NT) + 0.5) * 2.0 * np.pi / TRIP_ORACLE_NT
    R, TH = np.meshgrid(r, th, indexing="ij")
    out = []
    for sx, sy in srcs:
        X, Y = sx + R * np.cos(TH), sy + R * np.sin(TH)
        u = k0(R / lam)
        for ix, iy in ((2 * x0 - sx, sy), (2 * x1 - sx, sy), (sx, 2 * y0 - sy), (sx, 2 * y1 - sy)):
            u = u + sign * k0(np.hypot(X - ix, Y - iy) / lam)
        inside = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
        ring = (np.where(inside, u, 0.0).sum(axis=1) * r * (2.0 * np.pi / TRIP_ORACLE_NT)
                * TRIP_ORACLE_DR / (2.0 * np.pi * lam ** 2))
        out.append(np.bincount((r / region_leaks.TRIP_BIN_M).astype(np.int64), weights=ring,
                               minlength=nbins)[:nbins])
    return np.array(out)


def trip_calibration(_ids: list[str]) -> list[str]:
    bad = []
    def open_only(lam: float) -> gx.InternalLeak:
        return gx.InternalLeak(open_m=lam, foot_m=math.inf)
    lam0 = TRIP_T / K0_MEDIAN
    # the calibration on one home at the centre: the screened point source's median
    m = _model(_scale_block(), _leaky(open_only(lam0)))
    site = region_leaks.trip_site(m)
    steps, _rows = region_leaks.calibrate_lam(
        m, leak_of=open_only, target=TRIP_T, tol=TRIP_GUARD_TOL, lam0=lam0 * 0.8,
        resolution=TRIP_GUARD_RES, limit_lam=region_leaks.LIMIT_LAM_M,
        sample=region_leaks.every_home(m), site=site, procs=1)
    lam = steps[-1].scale
    analytic = K0_MEDIAN * lam / TRIP_T - 1.0
    ok = abs(analytic) <= region_leaks.TRIP_TOL
    print(f"trip-calibration: one home, T {TRIP_T:g} m: lam {lam:g} m in {len(steps) - 1} steps "
          f"(the limit {steps[0].median:.1f} m) "
          f"(the median {steps[-1].median:.3f} m, {steps[-1].inside:.4f} of the current inside); "
          f"{K0_MEDIAN:.4f} lam is T {analytic:+.1e} -> {'ok' if ok else 'FAILS'}", flush=True)
    if not ok:
        bad.append(f"one home: 1.257 lam is T {analytic:+.1e}")
    # a target past the limit (the leak vanishing) is refused
    past = 1.5 * steps[0].median
    try:
        region_leaks.calibrate_lam(
            m, leak_of=open_only, target=past, tol=TRIP_GUARD_TOL, lam0=lam0,
            resolution=TRIP_GUARD_RES, limit_lam=region_leaks.LIMIT_LAM_M,
            sample=region_leaks.every_home(m), site=site, procs=1)
        bad.append(f"one home: T {past:.1f} m, past the limit, was calibrated")
    except region_leaks.Unreachable as e:
        print(f"trip-calibration: one home, T {past:.1f} m: refused ({e})", flush=True)
    # two homes, one near the street edge: the pooled trips against the method of images
    b = _two_homes(TRIP_EDGE_M)
    m = _model(b, _leaky(open_only(TRIP_LAM)))
    site = region_leaks.trip_site(m)
    sample = region_leaks.every_home(m)
    step, rows = region_leaks.trip_step(m, scale=TRIP_LAM, leak=open_only(TRIP_LAM),
                                        sample=sample, site=site, procs=1)
    x0, y0, x1, y1 = (float(v) for v in shapely.bounds(cast("BaseGeometry", m.geo.boundary)))
    oracle = _image_rows(site.home_xy[sample.homes], (x0, y0, x1, y1), TRIP_LAM, site.nbins,
                         sign=-1.0)
    want = region_leaks.trip_median(sample.w @ oracle)
    err = step.median / want - 1.0
    ok = abs(err) <= region_leaks.TRIP_TOL
    print(f"trip-calibration: two homes ({TRIP_EDGE_M:g} m from the edge, lam {TRIP_LAM:g} m): "
          f"pooled median {step.median:.2f} m against the images' {want:.2f} m ({err:+.1e}); "
          f"inside {np.round(rows.sum(axis=1), 4).tolist()} against "
          f"{np.round(oracle.sum(axis=1), 4).tolist()}; per home "
          f"{[round(region_leaks.trip_median(r), 2) for r in rows]} against "
          f"{[round(region_leaks.trip_median(r), 2) for r in oracle]} -> "
          f"{'ok' if ok else 'FAILS'}", flush=True)
    if not ok:
        bad.append(f"two homes: the median {err:+.1e} off the images'")
    return bad


def _two_homes(edge_m: float) -> Block:
    """SCALE_L of open ground, its edge the street: a home at the centre and one `edge_m` from the
    south edge (TRIP_BLOCK)."""
    c = SCALE_L / 2.0
    return _open_block("synthetic-open-two", [_square(c - 1.0, c - 1.0, 2.0),
                                              _square(c - 1.0, edge_m - 1.0, 2.0)])


def closed_edge(_ids: list[str]) -> list[str]:
    bad = []
    b = _two_homes(CLOSED_EDGE_M)
    leak = gx.InternalLeak(open_m=TRIP_LAM, foot_m=math.inf)
    with tempfile.TemporaryDirectory() as tmp:
        fx = METRIC.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
    m_open = _leaky(leak).model(b, fx, demand=POP)
    m = _leaky(leak, edge=gx.ClosedEdge()).model(b, fx, demand=POP)
    st = m.state(np.zeros(m.n))
    # (i) well posed: every live component has a leak, and the cells live are the open edge's
    chk = region_leaks.closed_check(m, st, int(m_open.walled0.sum()))
    ok = chk.unleaked == 0 and chk.live == chk.live_open and chk.walled == chk.walled_open
    # (ii) the drainage is the leak's alone
    fl = gap_maps.flow_of(m, st, m.pop)
    street = float(np.abs(fl.Ff).sum() / fl.inj.sum())
    leaked = float(fl.leak.sum() / fl.inj.sum() - 1.0)
    ok = ok and street <= FLUX_TOL and abs(leaked) <= FLUX_TOL
    # (iii) the trips against the method of images, the images added (no current through the edge)
    site = region_leaks.trip_site(m)
    sample = region_leaks.every_home(m)
    step, rows = region_leaks.trip_step(m, scale=TRIP_LAM, leak=leak, sample=sample, site=site,
                                        procs=1)
    x0, y0, x1, y1 = (float(v) for v in shapely.bounds(cast("BaseGeometry", m.geo.boundary)))
    oracle = _image_rows(site.home_xy[sample.homes], (x0, y0, x1, y1), TRIP_LAM, site.nbins,
                         sign=1.0)
    want = region_leaks.trip_median(sample.w @ oracle)
    err = step.median / want - 1.0
    inside = rows.sum(axis=1)
    ok = ok and abs(err) <= region_leaks.TRIP_TOL and bool(np.all(np.abs(inside - 1.0) <= 1e-6))
    print(f"closed-edge: components {chk.components}, unleaked {chk.unleaked}; cells live "
          f"{chk.live} (open {chk.live_open}), homes walled in {chk.walled} (open "
          f"{chk.walled_open}); the street's drainage {street:.1e} of the injection, the leaks' "
          f"{leaked:+.1e}; two homes ({CLOSED_EDGE_M:g} m from the edge, lam {TRIP_LAM:g} m): "
          f"pooled median {step.median:.2f} m against the images' {want:.2f} m ({err:+.1e}), "
          f"inside {np.round(inside, 6).tolist()}; per home "
          f"{[round(region_leaks.trip_median(r), 2) for r in rows]} against "
          f"{[round(region_leaks.trip_median(r), 2) for r in oracle]} -> "
          f"{'ok' if ok else 'FAILS'}", flush=True)
    if not ok:
        bad.append(f"closed edge: unleaked {chk.unleaked}, street {street:.1e}, leaks "
                   f"{leaked:+.1e}, median {err:+.1e}")
    return bad


_LEAK_DRAIN = gap_maps.leak_drain
_TRIP_ORIGINS = region_leaks.trip_origins


def _with_street(st: gcl.State, u: np.ndarray) -> np.ndarray:
    """gap_maps.leak_drain with the street faces' drainage added (fault `boundary-trips`)."""
    c = st.sys.cells
    return np.asarray(_LEAK_DRAIN(st, u) + np.bincount(c.fc, weights=st.reach.Gf * u[c.fc],
                                                        minlength=c.n))


def _next_home(site: region_leaks.TripSite, homes: np.ndarray) -> np.ndarray:
    """region_leaks.trip_origins, each home's trips measured from the next home's injection
    (fault `wrong-home`)."""
    return np.roll(_TRIP_ORIGINS(site, homes), 1, axis=0)


# ------------------------------------------------------------------ the loop index

def _index_block(kind: str) -> tuple[Block, int]:
    """The loop-index guard's block of `kind` and its known loop count (module docstring)."""
    if kind == "grid":
        homes = [shapely.box(6 + 26 * i, 6 + 26 * j, 26 + 26 * i, 26 + 26 * j)
                 for i in range(3) for j in range(3)]
        box = shapely.box(0, 0, 84, 84)
        return region_leaks.synthetic_block("synthetic-grid", box,
                                            shapely.LineString(box.exterior.coords), homes,
                                            None), 8
    top = 56.0 if kind == "ladder" else 62.0
    homes = [shapely.box(6 + 16 * k, 6, 16 + 16 * k, top) for k in range(5)]
    return region_leaks.synthetic_block(f"synthetic-{kind}", shapely.box(0, 0, 84, 62),
                                        shapely.LineString([(0, 0), (84, 0)]), homes,
                                        None), (5 if kind == "ladder" else 0)


def loop_index(_ids: list[str]) -> list[str]:
    bad = []
    parts = []
    for kind in ("grid", "ladder", "comb"):
        b, want = _index_block(kind)
        m = _model(b)
        pt = region_leaks.painted_of(m, region_leaks.FRAME_H)
        img, _fl, _st = region_leaks._paint_state(m, np.zeros(m.n), pt)
        got = region_leaks.loops(img, pt.inside, pt.exits, float(np.finfo(np.float32).tiny),
                                 region_leaks.FRAME_H, contract=_CONTRACT[0])
        parts.append(f"{kind} {got} (known {want})")
        if got != want:
            bad.append(f"{kind} {got} != {want}")
    print("loop-index: " + ", ".join(parts) + f" -> {'ok' if not bad else 'FAILS'}", flush=True)
    return bad


_CONTRACT = [True]       # the guard's `contract` (fault `uncontracted`: False)


_IDENTITY_MODULES = gcl._identity_modules
_SUFFIX = gx.Exits.suffix


def _suffix_without_leak(self: gx.Exits) -> str:
    """Exits.suffix with the leak's part left out (fault `suffix-dropped`)."""
    parts = self.barriers.name + self.edge.name + self.boost.name
    return f"+x{parts}v{gx.EXITS_CODE}" if parts else ""


# ------------------------------------------------------------------ the mesh's speedups

def _cross_carving(geo: gc.Geometry) -> gcl.Carving:
    """`geo`'s footprints cut by two nested stops: a road along the block's middle (its bounds'
    longer side, through its centroid), then that and one across it, each CROSS_HALF_M wide on
    either side."""
    boundary = cast("BaseGeometry", geo.boundary)
    x0, y0, x1, y1 = shapely.bounds(boundary)
    c = shapely.centroid(boundary)
    along = shapely.LineString([(x0, c.y), (x1, c.y)] if x1 - x0 >= y1 - y0
                               else [(c.x, y0), (c.x, y1)])
    across = shapely.LineString([(c.x, y0), (c.x, y1)] if x1 - x0 >= y1 - y0
                                else [(x0, c.y), (x1, c.y)])
    first = shapely.buffer(along, CROSS_HALF_M)
    return gcl.Carving.of(geo, [[first, shapely.union(first, shapely.buffer(across,
                                                                             CROSS_HALF_M))]])


def _scan_was(qi: np.ndarray, qb: np.ndarray, n: int) -> list[tuple[int, ...]]:
    """gap_clear._faces's grouping before gap_clear._by_face, verbatim: a scan per face."""
    return [tuple(sorted(qb[qi == k].tolist())) for k in range(n)]


def _unsorted(qi: np.ndarray, qb: np.ndarray, n: int) -> list[tuple[int, ...]]:
    """gap_clear._by_face with each face's footprints in the query's order (fault `unsorted`)."""
    o = np.argsort(qi, kind="stable")
    qi, qb = qi[o], qb[o]
    cuts = np.searchsorted(qi, np.arange(n + 1))
    return [tuple(qb[cuts[k]:cuts[k + 1]].tolist()) for k in range(n)]


def faces(ids: list[str]) -> list[str]:
    bad = []
    for b in common.build_blocks(ids):
        geo = gc.geometry(b)
        for label, carving in (("plain", gcl.Carving.none(geo)), ("carved", _cross_carving(geo))):
            atoms = geo._replace(polys=carving.polys)
            got = gcl._faces(atoms, METRIC.spec, METRIC.form.grow)
            with _patched(gcl, "_by_face", _scan_was):
                was = gcl._faces(atoms, METRIC.spec, METRIC.form.grow)
            same_faces = (len(got[0]) == len(was[0])
                          and bool(shapely.equals_exact(got[0], was[0], tolerance=0.0).all()))
            same_noded = bool(shapely.equals_exact(got[3], was[3], tolerance=0.0))
            cover = sum(x != y for x, y in zip(got[1].tolist(), was[1].tolist(), strict=True))
            zone = sum(x != y for x, y in zip(got[2].tolist(), was[2].tolist(), strict=True))
            ok = same_faces and same_noded and cover == 0 and zone == 0
            print(f"{b.block_id} {label}: {carving.n} atoms, {len(got[0])} faces "
                  f"({sum(map(len, was[1].tolist()))} cover and {sum(map(len, was[2].tolist()))} "
                  f"zone pairs); faces {'equal' if same_faces else 'APART'}, noded lines "
                  f"{'equal' if same_noded else 'APART'}, {cover} covers and {zone} zones apart "
                  f"-> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{b.block_id} {label}")
    return bad


def _arrays_apart(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    """The names whose arrays differ (in value, dtype or shape), or that one side lacks."""
    return sorted(k for k in a.keys() | b.keys()
                  if k not in a or k not in b or a[k].dtype != b[k].dtype
                  or not np.array_equal(a[k], b[k]))


def lazy_baseline(ids: list[str]) -> list[str]:
    bad = []
    metric = _runtime()
    real = gc.build
    builds = {"n": 0}

    def counted(*a: Any, **k: Any) -> gc.Complex:
        builds["n"] += 1
        return real(*a, **k)
    for b in common.build_blocks(ids):
        bid = b.block_id
        geo = gc.geometry(b)
        carving = _cross_carving(geo)
        builds["n"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            with _patched(gc, "build", counted):
                fx = metric.fixed(b, cache=Path(tmp), carving=carving)
                again = metric.fixed(b, cache=Path(tmp), carving=carving)    # read back
            [path] = Path(tmp).glob("*.npz")
            kept = [n for n in np.load(path).files if n.startswith("cx_")]
        meshed = builds["n"]
        with tempfile.TemporaryDirectory() as tmp:
            plain = metric.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(geo))
            plain_again = metric.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(geo))
            [path] = Path(tmp).glob("*.npz")
            plain_kept = [n for n in np.load(path).files if n.startswith("cx_")]
        want = gc.complex_arrays(plain.cx0.complex())
        apart = {"plain read back": _arrays_apart(gc.complex_arrays(plain_again.cx0.complex()),
                                                  want),
                 "carved": _arrays_apart(gc.complex_arrays(fx.cx0.complex()), want),
                 "carved read back": _arrays_apart(gc.complex_arrays(again.cx0.complex()), want)}
        ok = (meshed == 0 and not kept and bool(plain_kept)
              and not any(apart.values()))
        print(f"{bid}: the carved mesh ({carving.n} atoms) built {meshed} complexes and keeps "
              f"{len(kept)} baseline arrays (the plain mesh's file {len(plain_kept)}); against "
              "the plain mesh's baseline, " + ", ".join(
                  f"{k} {len(v)} arrays apart{' (' + ', '.join(v) + ')' if v else ''}"
                  for k, v in apart.items()) + f" -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


@contextmanager
def _baseline_on_atoms() -> Iterator[None]:
    """A carved mesh's baseline built on the last carving's atoms (gap_clear.Carving.of's), not
    the buildings (fault `atoms`)."""
    last: list[gcl.Carving] = []
    of, init = gcl.Carving.of, gcl.LazyBaseline.__init__

    def carving_of(geo: gc.Geometry, networks: list[list[object]]) -> gcl.Carving:
        last.append(of(geo, networks))
        return last[-1]

    def on_atoms(self: gcl.LazyBaseline, geo: gc.Geometry, spec: gc.Spec, grow: float) -> None:
        init(self, geo._replace(polys=last[-1].polys), spec, grow)
    with _patched(gcl.Carving, "of", staticmethod(carving_of)), \
            _patched(gcl.LazyBaseline, "__init__", on_atoms):
        yield


def _eager(self: gcl.LazyBaseline, geo: gc.Geometry, spec: gc.Spec, grow: float) -> None:
    """gap_clear.LazyBaseline built when made (fault `eager`)."""
    _LAZY_INIT(self, geo, spec, grow)
    _ = self.arrays


_LAZY_INIT = gcl.LazyBaseline.__init__


# ------------------------------------------------------------------ the run

GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "baseline": baseline, "weights": weights, "reach": reach, "gradient": gradient,
    "free": free, "identity": identity, "convex": convex, "pattern": pattern,
    "constants": constants, "device": device, "roads": roads, "flux": flux,
    "device-roads": device_roads, "leak-assembly": leak_assembly, "leak-fold": leak_fold,
    "enclosure": enclosure, "leak-gradient": leak_gradient, "leak-scale": leak_scale,
    "barriers": barriers, "exits-identity": exits_identity, "loop-index": loop_index,
    "trip-calibration": trip_calibration, "closed-edge": closed_edge,
    "street-boost": street_boost, "road-boost": road_boost, "underflow": underflow,
    "no-features": no_features,
    "faces": faces, "lazy-baseline": lazy_baseline}


def _fault_exits():
    """The band line's faces read at the unsnapped tolerance (gap_complex.TOL)."""
    return _patched(gcl, "BAND_TOL_M", gc.TOL)


FAULTS: dict[str, dict[str, Callable]] = {
    "baseline": {"interior": lambda: _patched(gcl, "_faces", _faces_less_interior),
                 "exits": _fault_exits},
    "weights": {"no-conform": lambda: _patched(gcl, "_conform", _no_conform)},
    "reach": {"threshold": lambda: _patched(gf.Ring, "live", _ring_off_by_one)},
    "gradient": {"tie": lambda: _patched(gf.Terms, "vjp", _vjp_first)},
    "free": {"drop-crossing": lambda: _patched(gc, "free_first", _drop_crossing)},
    "identity": {"collide": lambda: _patched(gcl, "check_disjoint", lambda name: None)},
    "convex": {"multiply": lambda: _patched(gf.Terms, "value", _multiply)},
    "pattern": {"drop-identity": lambda: _patched(gcl.Graph, "values", _drop_identity)},
    "constants": {"omit": lambda: _patched(gcl, "code_constants", _omit_mesh_rev)},
    "device": {"perturb": _perturb_one},
    "device-roads": {"perturb": _perturb_one},
    "roads": {"one-face-closed": lambda: _patched(gcl.Carving, "carved", _one_face_closed)},
    "flux": {"flip-one": _flip_one},
    "leak-assembly": {"drop-one": lambda: _leak_fault(_drop_one),
                      "not-live": lambda: _leak_fault(_not_live)},
    "leak-fold": {"lengths-swapped": lambda: _patched(gx.InternalLeak, "per_cell",
                                                      _lengths_swapped)},
    "enclosure": {"every-cell": lambda: _leak_fault(_every_cell)},
    "leak-gradient": {"adjoint-unleaked": lambda: _patched(gcl.Relaxation, "value_sgrad",
                                                           _sgrad_unleaked)},
    "leak-scale": {"per-cell": lambda: _leak_fault(_per_cell)},
    "barriers": {"not-inserted": lambda: _patched(gx.OsmBarriers, "block",
                                                  lambda self, block: block),
                 "gap-closed": lambda: _patched(gx.OsmBarriers, "parts", _gap_closed),
                 "features-dropped": _features_dropped,
                 "unshifted": lambda: _patched(gcl.Carving, "shifted", lambda self, by: self)},
    "no-features": {"neighbour-features": lambda: _patched(sys.modules[__name__], "_attached_osm",
                                                           _neighbour_osm)},
    "exits-identity": {"in-code": lambda: _patched(gcl, "_identity_modules",
                                                   lambda: (*_IDENTITY_MODULES(), gx)),
                       "suffix-dropped": lambda: _patched(gx.Exits, "suffix",
                                                          property(_suffix_without_leak))},
    "loop-index": {"uncontracted": lambda: _patched_item(_CONTRACT, 0, False)},
    "closed-edge": {"edge-open": lambda: _patched(gx.ClosedEdge, "mesh", lambda self, fm: fm)},
    "street-boost": {"classes-dropped": lambda: _patched(gx.StreetBoost, "on_way", _every_class),
                     "surfaces-dropped": lambda: _patched(gx.StreetBoost, "on_way",
                                                          _any_surface)},
    "road-boost": {"unboosted-road": lambda: _patched(gcl.Model, "mesh_of", _road_unboosted),
                   "ways-dropped": lambda: _patched(gcl.Model, "mesh_of", _road_without_ways)},
    "underflow": {"unfloored": lambda: _patched(gap_factor, "_residual", _residual_unfloored)},
    "faces": {"unsorted": lambda: _patched(gcl, "_by_face", _unsorted)},
    "lazy-baseline": {"atoms": _baseline_on_atoms,
                      "eager": lambda: _patched(gcl.LazyBaseline, "__init__", _eager)},
    "trip-calibration": {"boundary-trips": lambda: _patched(gap_maps, "leak_drain",
                                                            _with_street),
                         "wrong-home": lambda: _patched(region_leaks, "trip_origins",
                                                        _next_home)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(guards: list[str], ids: list[str]) -> None:
    """Each guard as it is, then under each of its faults (FAULTS), one after another in this
    process: one cluster task runs the GPU guards so (`device` and `device-roads` on a block, both
    ways, in one submit). Raises naming every run not as `main` expects."""
    wrong = []
    for guard in guards:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, ids, fault)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(guards)} on {', '.join(ids)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "both-ways" and set(a[1].split(",")) <= set(GUARDS):
        both_ways(a[1].split(","), [] if a[2] == "-" else common.ids_of(a[2]))
        raise SystemExit(0)
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], [] if a[1] == "-" else common.ids_of(a[1]), a[2] if len(a) == 3 else None)
