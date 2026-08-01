# Road width is a property of each road, not a global (2026-07-31)

`corridor_m` is gone. Every road carries its own `width_m`; the metric has no default and no
fallback. A road set without the column is a `ValueError`, at both entry points
(`permeability.edge_conductances`, `budget.displacement`).

## Why the global had to go

`corridor_m: 3.0` was doing three unrelated jobs under one name:

1. the half-width of the conductance corridor in `permeability`,
2. the half-width of the displacement corridor in `budget`,
3. the fallback *building* radius when a block had fewer than two building points.

(3) was a genuine conflation — a road half-width standing in for a building radius — and is now
`DEFAULT_BUILDING_RADIUS_M`. (1) and (2) are the same quantity, but only per road: once one-way
roads exist, "the corridor width" is not a property of the region.

The blocker was `g_road = 20.0`, "the conductance of a standard road" — meaningful only relative
to a standard width, which is exactly what we were deleting. The replacement is conductance **per
metre of usable width**:

    g = g_road_per_m * max(0, W_dir - road_margin_m) / dist

No reference width appears. `g_road_per_m = 8.0` reproduces the old numbers: a 6 m two-way road
gives each direction `(6 - 1)/2 = 2.5 m` of usable width, and `8.0 * 2.5 = 20.0`.

## What each method now does

Every method carries `road_width_m: float = DEFAULT_ROAD_WIDTH_M` (6.0) and stamps `width_m` on the
roads it emits, via `permeability.with_width`. That is a default a caller overrides, not a global
the metric falls back to — the distinction the whole change turns on. `arterial` and
`osm_footpaths` had their own `corridor_m: 3.0` fields; both became `road_width_m: 6.0`. Internal
half-width parameters were renamed `half_width_m` so the old name cannot creep back, and
`_explode`/`_planarize`/`_greedy_arterials` now require a width rather than defaulting one.

## Does it reduce exactly?

Almost. Measured against `2aab163` (pre-one-way) on 60 real blocks, same clearance road sets:

* **59/60 permeability values bit-identical.**
* 1 block differs by **9.4e-10 relative** — one mesh edge in 19,023.

The cause is the coverage test, not the model. The old code buffered the road union once; per-road
widths make that impossible, so each segment is buffered separately. `buffer` polygonalizes a
circle, and the union's vertices land differently from each segment's, so an edge grazing the
boundary can flip. The flipped edge sits **2.9984 m** from the centreline of a 3 m half-width
corridor — 1.6 mm inside, which the union buffer cut out and the per-segment buffer keeps. The new
answer is the more correct one. The conductance formula and the road geometry are exact.

## Two things fell out

**The footpath clamp is gone.** Under the old rule a road REPLACED the footpath, so every footpath
edge had to be capped at its own would-be upgrade or an upgrade could lower conductance. Roads now
enter through `max(footpath, road)`, which makes monotonicity structural. The clamp was measured
before removal: **0 of 19,023 mesh edges** across 60 blocks would have been affected
(`scratchpad/width/clamp_probe.py`), so no published number moved. `_covered_edges` died with it.

`tests/test_permeability.py` had a test asserting a property of `np.minimum` on values it computed
itself, never calling `egress_power` — it would have passed with the clamp removed either way. It
is now a real guard on `edge_conductances`, fault-injected (replace the `max` with assignment →
fails; restore → passes).

**Widening is superlinear in capacity.** The margin is paid once, so `(W - margin)` grows faster
than `W`. Two 3 m roads carry less than one 6 m road. That is what real street hierarchies look
like, but it is a behavioural change, not just realism, and it is now the model's opinion.

## Consequence for one-way roads

A one-way street is not half a two-way one: both pay the same margin. They match on per-direction
conductance at `W_one = (W_two + margin)/2` — 3.5 m at the defaults, not 3.0 m. That number is
derived from the margin rather than asserted, which is what the earlier
[2026-07-30-oneway-half-width.md](2026-07-30-oneway-half-width.md) note was missing.
