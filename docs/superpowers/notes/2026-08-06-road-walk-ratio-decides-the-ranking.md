# The road/walk ratio decides which method wins (2026-08-06)

A road is currently **~513x more conductive per metre** than a gap between buildings:

    road      g_road_per_m * usable = 6.667 * 3.0 = 20.0     (per unit distance)
    footpath  g_walk * shape        = 0.1 * 0.39  = 0.039
    ratio     513x

That number is not a measured physical fact. `conf/permeability.yaml` records that `g_walk = 0.1`
plus the corridor model were adopted because they "massively improve method discrimination
(D = 10% method-spread ~0.9pts -> ~11.8pts)" — the ratio was **calibrated against method spread**.

## The measurement

`scratchpad/ratio/road_walk_ratio.py`. Lens A (matched displacement `D = 0.10`), 8 blocks x 6
methods. The prefix is chosen by DISPLACEMENT, which is independent of conductance, so each
(block, method) prefix is computed once and scored at every ratio — only `g_road_per_m` varies, and
`footpath_conductance`'s fair-normalization pins the footpath level to `g_walk`.

     ratio   median perm   median spread   as % of 513x spread   rank flips   winner changes
        5x        0.0773          0.0711                19.7%           38            6 / 8
       10x        0.1831          0.1340                37.2%           37            6 / 8
       25x        0.3387          0.2075                57.6%           33            5 / 8
       50x        0.4606          0.2310                64.1%           26            5 / 8
      100x        0.5672          0.2847                79.0%           17            3 / 8
      250x        0.6426          0.3369                93.5%           12            2 / 8
      513x        0.6768          0.3603               100.0%            0            0 / 8

    Kendall tau of each ranking against the 513x ranking
        5x   median +0.386   min -0.600          50x   median +0.314   min +0.067
       10x   median +0.133   min -0.467         100x   median +0.714   min +0.467
       25x   median +0.243   min -0.333         250x   median +0.933   min +0.714

## What it says

**The spread compresses, but that is the lesser finding.** 0.360 -> 0.071 going from 513x to 5x, so
at a realistic ratio methods do look much closer in absolute terms.

**The rankings SCRAMBLE.** Median Kendall tau against today's ranking is **+0.133 at 10x** — close to
random — and the MINIMUM is **negative** at 5x, 10x and 25x (-0.600, -0.467, -0.333), meaning the
ordering substantially reverses on some blocks. The winner changes on **6 of 8 blocks** at 5x.

So it is not that methods become equally good at a realistic ratio. **Which method is best is a
function of the ratio**, and the ratio is a constant chosen because it produced spread.

This closes a loop. The road-first mesh spec's acceptance criterion S1 was rejected in its own
red team as circular — "calibrating for method spread and then accepting on ranking change is nearly
the same operation twice". This is that circularity reappearing as a sensitivity: the published
ordering rests on a knob tuned to make orderings separate.

## Why it surfaced now, and what it forces

It came out of fixing D1/D2. Every formulation of the road term that is both MONOTONE and preserves
D1 reduces the road's effective advantage far below 513x — necessarily, because today's model has no
"walk out to the street" leg at all and prices the whole edge as though you were already on the road.
Making that walk explicit must shrink the advantage, whichever way it is made explicit.

That is why the fix and this question are the same question, and why the fix was amended to price
legs at FOOTPATH rate. Under that model the arithmetic collapses to:

    road wins iff  L_i + L_j + R_len/513 < d
    road conductance ~ (g_walk * shape) / (L_i + L_j)
    benefit factor   ~ d / (L_i + L_j)

Take the road iff detouring to it costs less walking than going direct; the benefit is large for
parcels fronting a road and nil for parcels far from one. **A uniform tuned scalar is replaced by a
per-edge geometric quantity** — which is the structural answer to this finding, not merely a
different value of the same knob.

## Status and caveats

Measured on 8 blocks, 6 methods, Lens A only, one region, and swept on the CURRENT crow-flies road
term. The direction is unambiguous and large; the exact numbers are not load-bearing.

**The analysis must be re-run against the fixed metric**, because the fix changes the road term and
therefore how the ratio enters. Doing it on the old term would measure something being replaced.
That re-run is the open item.

## Why this outranks D1 and D2

D1 and D2 are defects in how a road's SHAPE is priced — worth a few points of permeability. This is
about whether the published ordering of methods means anything. Until it is resolved, any claim that
one reblocking method beats another carries an unstated dependence on a calibrated constant whose
true value is unknown.
