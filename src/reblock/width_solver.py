"""Choose a per-road width and direction for a network someone else drew.

## NOT WIRED IN. Measured at region scale and it did not pay.

Nothing imports this outside its own tests. It is kept in the tree only so a variant can be tried
without rewriting it -- if it is still unused when someone next reads this, delete it; an option
nobody selects is exactly the wart the repo's no-legacy rule bans.

Against the lens's own prefix truncation on the 4,615-parcel density_compactness region:

    promote from `don't build`   better on  6/12   median -0.0134
    demote  from two-way         better on  3/12   median -0.0266

Cost was never the problem (1.6 s median, 30 s worst) -- the batched `linearized_gain` scoring
works. The problem is that the lattice's MIDDLE elements are dominated in both search directions:
promoting, one-way offers ~0.5x the gain for 0.57x the cost, so two-way always wins gain-per-metre;
demoting, dropping a road forfeits everything but saves everything and wins the ratio. One-way was
chosen 0-17 times out of 8-3,522 pieces. With the middle unused this is just road SELECTION, and it
loses to `budget.street_first_ordered` at every budget.

The block-scale result that motivated it (flip beats truncate, 54/72, p=0.0020) does not translate
because that probe FORCED the topology to stay and only allowed narrowing. Given the freedom to drop
a road instead, the optimizer drops.

**What would have to change to make this worth wiring in:** a benefit term that values keeping a
road at all -- coverage, worst-case parcel access, N-1 resilience -- because permeability alone is
content to delete a road and spend the savings elsewhere. Without one the middle of the lattice has
no reason to be selected. See the backlog entry for the full record.

A method proposes TOPOLOGY -- where roads go. What each of those roads should BE is a separate
question, and one the methods currently answer by fiat: every road comes out a full two-way street.
This solves it instead.

## The decision variable is a four-element lattice

A road's state is the set of directions it permits, ordered by inclusion:

        two-way              {forward, backward}          width 7 m
        /       \\
    one-way   one-way        {forward}   {backward}       width 4 m
        \\       /
      don't build                  {}                     width 0

Both benefit and cost rise monotonically up it, and the middle-to-top step is PURE ADDITION: a
one-way road and a two-way road give the permitted direction exactly the same conductance (20.0
either way at the shipped parameters), because `one_way_width` is defined as the width at which they
match. So climbing adds the reverse direction and 3 m of width without trading anything away, which
is what makes the lattice the right algebra here rather than a convenient picture.

Two consequences worth stating, because both remove work:

* **`don't build` is the bottom element**, so choosing widths and choosing WHICH ROADS TO BUILD are
  the same decision. The lenses currently make the second one separately, by walking a prefix of a
  fixed order.
* **There is no Robbins feasibility constraint.** A one-way road's reverse direction falls back to
  footpath conductance, never to zero, so nothing can be stranded and no orientability bookkeeping
  is needed. The metric already prices a bad orientation by itself.

## Why bother: flipping beats truncating

There are exactly two ways to spend less displacement on a given topology -- build fewer roads, or
build thinner one-way ones -- and at equal displacement the second wins: 54/72 block-method pairs,
median +0.0193 permeability, p=0.0020 (`notes/`). The gain comes from networks with SLACK rather
than from loopy ones; `cycle_native`, the only genuinely circulating method, is the one place
flipping loses.

## Cost

Scoring every candidate exactly would need one sparse solve per candidate per round, which is
hopeless at region scale. Instead this uses the same two-stage trick as `ResistanceGreedyReblocker`:
ONE solve per round gives `v = L^-1 b`, every candidate is then ranked in O(edges it covers) by the
first-order sensitivity, and a whole BATCH is committed before re-solving. Ranking error does not
accumulate because the next round re-solves against what was actually built.

The first-order gain is symmetric and therefore OVERSTATES a one-way upgrade, which only improves
one direction. That makes it a ranking heuristic, not a score -- exactly as it already is for the
two-way case, where it also overstates by ignoring the rank-1 denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry import LineString

from reblock.budget import building_radii, displacement
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.orient import one_way_width, strong_orientation
from reblock.permeability import (
    DEFAULT_ROAD_WIDTH_M,
    ONEWAY_COL,
    WIDTH_COL,
    PermeabilityParams,
    _adaptive_r0,
    _footpath_conductance,
    egress_power,
    lane_width,
    road_conductance,
)

BOTTOM, ONE_WAY, TWO_WAY = 0, 1, 2
"""Lattice height. The two one-way directions are incomparable, so they share a height; which of
them a road takes is fixed by `strong_orientation` rather than searched (see `solve`)."""


@dataclass
class WidthAssignment:
    """The solved network plus what it cost, so a caller can report the budget it actually spent."""

    roads: GeoDataFrame
    displacement_frac: float
    permeability: float
    built: int
    one_way: int


def _mesh(block: Block, params: PermeabilityParams, adj: list[set[int]], r0: float,
          width_m: float) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64],
                                   NDArray[np.float64]]:
    """`(i, j, footpath_g, upgrade_delta)` per adjacency edge, and the centroid segment endpoints.

    `upgrade_delta` is what a two-way road of `width_m` would ADD to an edge, floored at zero for
    the same reason the metric takes a `max`: a road never makes an edge worse.
    """
    geoms = list(block.parcels.geometry)
    cent = [g.centroid for g in geoms]
    cx = np.array([c.x for c in cent])
    cy = np.array([c.y for c in cent])
    rows, cols = [], []
    for i in range(len(geoms)):
        for j in adj[i]:
            if j > i:
                rows.append(i)
                cols.append(j)
    ri = np.asarray(rows, dtype=np.int64)
    ci = np.asarray(cols, dtype=np.int64)
    if ri.size == 0:
        z = np.zeros(0)
        return ri, ci, z, z
    dist = np.hypot(cx[ri] - cx[ci], cy[ri] - cy[ci])
    keep = dist > 0.0
    ri, ci, dist = ri[keep], ci[keep], dist[keep]
    foot = _footpath_conductance(dist, r0, params.g_walk)
    road = road_conductance(params, np.full(dist.size, lane_width(params, width_m)), dist)
    return ri, ci, foot, np.maximum(road - foot, 0.0)


def _piece_displacement(geoms: list[LineString], pts: GeoDataFrame, radii: NDArray[np.float64],
                        half_width_m: float) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Per piece, `(building indices, c_b)` for every building its corridor ALONE would graze.

    Mirrors `budget.displacement` term for term -- `c = clip(1 - d/r)`, `r == 0` counting iff
    `d <= 0`, distance measured to the buffered corridor -- so the elementwise MAX of these over the
    built pieces is the exact union displacement rather than an approximation of it. That identity
    holds because distance to a union of corridors is the min over corridors, and `c` decreases in
    distance.

    `resistance_lp.segment_displacement` computes the same thing but prefilters with
    `query_ball_point` around the geometry's VERTICES, which silently misses buildings beside the
    middle of a long span. That is sound there (it is fed short per-edge segments) and wrong here,
    where a piece can be a whole 60 m road -- it under-counted by 22% on the test fixture.
    """
    n = len(pts)
    if n == 0:
        return [(np.zeros(0, dtype=np.int64), np.zeros(0)) for _ in geoms]
    points = shapely.points(np.column_stack([pts.geometry.x.to_numpy(),
                                             pts.geometry.y.to_numpy()]))
    tree = STRtree(points)
    rmax = float(radii.max()) if radii.size else 0.0
    out: list[tuple[NDArray[np.int64], NDArray[np.float64]]] = []
    for g in geoms:
        corridor = g.buffer(half_width_m)
        idx = np.asarray(tree.query(corridor.buffer(rmax), predicate="intersects"), dtype=np.int64)
        if idx.size == 0:
            out.append((idx, np.zeros(0)))
            continue
        d = shapely.distance(points[idx], corridor)
        r = radii[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(r > 0.0, 1.0 - d / r, np.where(d <= 0.0, 1.0, 0.0))
        c = np.clip(c, 0.0, 1.0)
        keep = c > 0.0
        out.append((idx[keep], c[keep]))
    return out


def _edge_lines(block: Block, ri: NDArray[np.int64], ci: NDArray[np.int64]) -> STRtree:
    cent = [g.centroid for g in block.parcels.geometry]
    cx = np.array([c.x for c in cent])
    cy = np.array([c.y for c in cent])
    pts = np.column_stack([np.stack([cx[ri], cx[ci]], axis=1).ravel(),
                           np.stack([cy[ri], cy[ci]], axis=1).ravel()])
    return STRtree(shapely.linestrings(
        pts, indices=np.repeat(np.arange(ri.size, dtype=np.int64), 2)))


@dataclass
class WidthSolver:
    """Greedy over the lattice: promote the road with the best permeability per displaced home.

    Starts every road at TWO-WAY and DEMOTES until the budget is met, rather than starting at
    `don't build` and climbing. That direction is not a detail -- it is the whole measured effect.
    Promoting from the bottom makes this a road-SELECTION algorithm competing with the lens's prefix
    order, and at region scale it loses (6/12, median -0.0134). What actually beat truncation was
    keeping EVERY road and paying in width, so demotion tries narrowing first and only drops a road
    when narrowing is unavailable or insufficient.
    """

    params: PermeabilityParams = field(default_factory=PermeabilityParams)
    road_width_m: float = DEFAULT_ROAD_WIDTH_M
    chunks: int = 8
    """Batches committed between solves. More batches track the true gradient more closely and cost
    one sparse solve each; 8 mirrors `ResistanceLPReblocker`."""

    def solve(self, block: Block, roads: GeoDataFrame, *,
              max_displacement: float) -> WidthAssignment:
        crs = block.crs
        empty = GeoDataFrame(geometry=[], crs=crs)
        empty[WIDTH_COL] = np.zeros(0)
        empty[ONEWAY_COL] = np.zeros(0, dtype=bool)
        if roads is None or len(roads) == 0:
            return WidthAssignment(empty, 0.0, 0.0, 0, 0)

        # Split at orientation boundaries so a road that is only partly orientable can still take a
        # one-way state on the part that is; `oneway=True` also fixes WHICH direction each piece
        # would run, so the solver chooses a lattice HEIGHT and the geometry supplies the side.
        pieces = strong_orientation(block, roads, params=self.params, oneway=True)
        orientable = pieces[ONEWAY_COL].to_numpy(dtype=bool).copy()
        geoms: list[LineString] = [cast(LineString, g) for g in pieces.geometry]
        n = len(geoms)

        adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
        r0 = _adaptive_r0(block, self.params)
        ri, ci, _foot, delta = _mesh(block, self.params, adj, r0, self.road_width_m)
        tree = _edge_lines(block, ri, ci) if ri.size else None

        half = self.road_width_m / 2.0
        covers = [np.asarray(tree.query(g.buffer(half), predicate="intersects"), dtype=np.int64)
                  if tree is not None else np.zeros(0, dtype=np.int64) for g in geoms]

        radii = building_radii(block.building_points)
        n_b = max(len(block.building_points), 1)
        state = np.zeros(n, dtype=np.int64)
        w_one = one_way_width(self.params, self.road_width_m)

        # Per-piece displaced-building sets, so the running cost is an elementwise MAX over what is
        # built rather than a fresh buffer+union of the whole network per candidate. That inner call
        # was the solver's cost: hundreds of pieces x `chunks` rounds of re-unioning made a region
        # take longer than the whole example pipeline. Each state has its own corridor half-width,
        # hence one table per lattice height.
        disp_at = {
            ONE_WAY: _piece_displacement(geoms, block.building_points, radii, w_one / 2.0),
            TWO_WAY: _piece_displacement(geoms, block.building_points, radii, half),
        }
        c_now = np.zeros(len(block.building_points), dtype=float)

        def cost_of(st: NDArray[np.int64]) -> NDArray[np.float64]:
            """Exact union displacement of `st`, as the elementwise max over what is built."""
            c = np.zeros(len(block.building_points), dtype=float)
            for k in np.flatnonzero(st > BOTTOM):
                idx, cb = disp_at[int(st[k])][k]
                if idx.size:
                    np.maximum.at(c, idx, cb)
            return c

        def _with(st: NDArray[np.int64], k: int, nxt: int) -> NDArray[np.int64]:
            t = st.copy()
            t[k] = nxt
            return t

        def frame(st: NDArray[np.int64]) -> GeoDataFrame:
            live = np.flatnonzero(st > BOTTOM)
            if live.size == 0:
                return empty
            out = pieces.iloc[live].reset_index(drop=True)
            one = st[live] == ONE_WAY
            out[WIDTH_COL] = np.where(one, w_one, self.road_width_m)
            out[ONEWAY_COL] = one
            return cast(GeoDataFrame, out)

        cap = max_displacement * n_b
        state[:] = TWO_WAY
        c_now = cost_of(state)
        for _t in range(max(self.chunks, 1)):
            if c_now.sum() <= cap:
                break
            _p, v = egress_power(block, frame(state), self.params, adj=adj, r0=r0)
            # Rank every legal DEMOTION by gain surrendered per home saved, cheapest sacrifice
            # first. A one-way road keeps the permitted direction at full conductance and drops the
            # other to footpath, so it forfeits about half the edge gain; dropping a road forfeits
            # all of it. The next round re-solves, so a misjudged half self-corrects.
            cands: list[tuple[float, int, int]] = []
            for k in range(n):
                for nxt in (ONE_WAY, BOTTOM):
                    if nxt >= state[k] or (nxt == ONE_WAY and not orientable[k]):
                        continue
                    idx = covers[k]
                    raw = (float((delta[idx] * (v[ri[idx]] - v[ci[idx]]) ** 2).sum())
                           if idx.size else 0.0)
                    forfeit = raw * (0.5 if nxt == ONE_WAY else 1.0)
                    saved = c_now.sum() - cost_of(_with(state, k, nxt)).sum()
                    if saved <= 0.0:
                        continue
                    cands.append((forfeit / saved, k, nxt))
            if not cands:
                break
            for _ratio, k, nxt in sorted(cands, key=lambda c: c[0]):
                if nxt >= state[k] or c_now.sum() <= cap:
                    continue
                state[k] = nxt
                c_now = cost_of(state)

        built = frame(state)
        d = (displacement(block.building_points, radii, built) / n_b) if len(built) else 0.0
        p = float(egress_power(block, built if len(built) else None,
                               self.params, adj=adj, r0=r0)[0])
        p0 = float(egress_power(block, None, self.params, adj=adj, r0=r0)[0])
        perm = 0.0 if p0 <= 0 else 1.0 - p / p0
        return WidthAssignment(built, d, perm, int((state > BOTTOM).sum()),
                               int((state == ONE_WAY).sum()))
