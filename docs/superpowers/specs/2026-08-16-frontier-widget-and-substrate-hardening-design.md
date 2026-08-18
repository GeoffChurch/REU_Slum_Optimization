# Piece D1: the Frontier widget, and hardening the substrate

**Date:** 2026-08-16
**Branch:** to be created off `main`
**Status:** design, approved section by section; not yet implemented
**Scope of this document:** the first sub-piece of piece D
(`specs/2026-08-13-site-redesign-design.md` §5, §7 D). Consumes piece C's substrate
(`specs/2026-08-15-web-bundle-and-widget-substrate-design.md`) and hardens two parts of it. The
remaining three widgets follow in their own specs — see §1 for why each is separated and in what
order.

## Why

Piece C proved the widget substrate with one widget. D1 is where it stops being a substrate-with-one-
consumer and becomes a substrate: a second widget, of a genuinely different *kind* — a chart rather
than a map — plus the two hardening fixes C's reviews identified as harmless with one widget and
hazardous with two.

The Methods index page is pure prose today. `Frontier` gives it the tradeoff the whole project turns
on: permeability against displacement, per method, with the two calibrated standards drawn on it and
draggable.

**What ships:** a committed per-method prefix table for the pinned block, `Frontier` on the Methods
index with the existing chart PNG as its fallback, a chart-capable transform, and two substrate
fixes.

## §1 Scope, and why D is three pieces

Piece D as the parent design frames it is four widgets. They are not four peers — each has a
different dependency, which is why D is split:

| widget | dependency | piece |
|---|---|---|
| `Frontier` | none upstream; its fallback PNG is already committed | **D1, here** |
| `ScreenMap` | a new city tier: 16,451 blocks, measured at 843,838 vertices | D2 |
| `DisplacementField` | no static figure exists; needs one first, as permeability did | D3 |
| `RegionGrow` | `RegionBuilder` discards accretion order (`src/reblock/region.py:353` sorts it away) | D4, or with D3 |

Measured while scoping, and recorded because the parent design's figures are wrong:

* **The city tier is ~2x its budget.** §3 budgets "~3 MB" for quantized polygons. Measured on the
  real data after the `MIN_COUNT = 30` filter: 16,451 blocks, **843,838 vertices** (mean 51.3 per
  block), which is **~6.4 MB** of JSON at 1 m simplification and **4.5 MB** at 3 m. Not a blocker —
  GitHub Pages serves compressed, and `examples/` is already 395 MB — but the simplification
  tolerance is a real design choice with visible angularity at zoom, and D2 owns it.
* **`RegionGrow`'s data does not exist.** `region.py:353` does
  `result.append(sorted(ids[i] for i in cluster))`, so the order blocks were accreted in — the
  widget's entire teaching point — is discarded. Getting it means changing a shipped `RegionBuilder`
  contract, which makes `RegionGrow` the most invasive of the four, not the cheapest.

## §2 The bake

`Frontier` needs, per method, the full drainage-ordered prefix table on the pinned block
`ZAF.9.3.1_1_40972`: arrays of `road_m`, `displacement` and `permeability` for `m = 0…R`. Both axes,
because a target set on either one is answered against the other.

Segment counts, read from the committed `examples/method-comparison/run.log`:

| method | R | | method | R |
|---|---|---|---|---|
| topology | 228 | | resistance_lp | 227 |
| greedy_arterial_access_displacement | 175 | | clearance_looped | 92 |
| clearance | 20 | | cycle_native | 20 |
| osm_footpaths | 13 | | euclidean_grid | 9 |

**784 prefixes across 8 methods.** At the 26 ms/solve calibrated in C's spec for this block that is
~20 s of permeability solving, plus a per-prefix displacement pass. This is the "long pole" the
parent design warned about; at block scale it is not one, and the ~12 hour figure belongs to regions.

Payload: 792 rows x 3 floats, about **30 KB**.

**A separate bundle.** `examples/method-comparison/frontier.json`, with its own generated `.d.ts`,
rather than an extension of `examples/perm-graph/bundle.json`. Same block, different page: folding
the curves into the existing bundle would make the Methods index download PermGraph's 278 KB of
per-prefix potentials and currents in order to draw a chart.

**One duplication fixed rather than compounded.** C's final review flagged that
`scripts/gen_perm_graph.py` and `scripts/gen_web_bundle.py` each define their own `VARIANT`/`METHOD`
and each duplicate the block-and-roads loading, so changing the pin in one silently desynchronises
the other. A third baker makes it triplication. D1 extracts that loading to one shared place and has
all three use it — which also makes it provable that Frontier's chart and PermGraph's widget describe
the same block.

## §3 The substrate changes

### `View` gains independent axes

A chart cannot use the map transform as built. `View` carries a single `scale` and `fitBbox` takes
`Math.min(width / w, height / h)` — a uniform, aspect-preserving fit. That is right for a map, where
metres must be metres on both axes, and wrong for a chart: displacement runs 0–0.25 and permeability
0–1, in different units, so a uniform scale squashes the plot into a sliver. **Piece C's spec claimed
an SVG `Frontier` would reuse the transform layer unchanged; that claim was false, and this is the
correction.**

`View` becomes `{ sx, sy, tx, ty }`:

* `toScreen(v, x, y)` -> `[x * v.sx + v.tx, v.ty - y * v.sy]`, `toWorld` inverting per axis;
* `panned` unchanged — translation only;
* `zoomed` multiplies both scales by the same factor, so their ratio survives: correct for maps,
  harmless for charts;
* `nearest` unchanged — it works in world space.

**`fitBbox` keeps its uniform behaviour, and gets a test that says so.** It sets `sx = sy`, and that
is load-bearing rather than incidental: `render/canvas.ts` converts metres to pixels through the
scale for road widths and node radii, so unequal scales would silently make geographic widths wrong
on one axis while everything still drew. A test asserting `fitBbox` returns `sx === sy` guards the
invariant against a later well-meant generalisation — the same reasoning that made piece B store
`upgraded` rather than let each renderer recompute it. Charts get a separate
`fitAxes(bboxX, bboxY, width, height, pad)` fitting each axis independently.

**Named churn, so a reviewer expects it:** `web/src/view/transform.ts`, its six Node tests,
`web/src/render/canvas.ts` (three `view.scale` reads) and `web/src/widgets/perm-graph.ts`. Every
map-path read becomes `sx`, which is valid precisely because `fitBbox` guarantees equality.

**The risk this carries, stated plainly:** those two files are reviewed, shipped, and live on the
deployed site. A regression breaks a working widget. The `sx === sy` invariant test is what makes the
map path's behaviour provably unchanged, and the implementation plan requires re-verifying the
deployed PermGraph after merge rather than assuming it.

### Two new DOM-free modules

* **`web/src/view/ticks.ts`** — `niceTicks(min, max, target)` returning round tick values. Pure
  arithmetic, and wrong in a way nobody notices (ticks at 0.07, 0.14, 0.21 instead of 0.05, 0.10,
  0.15), so it is unit-tested.
* **`web/src/render/svg.ts`** — the only module that knows SVG namespaces, mirroring
  `render/canvas.ts`'s role: axes, gridlines, one polyline per method, labels, target lines. Text and
  pointer targets being real DOM is the entire reason `Frontier` is SVG rather than canvas —
  selectable labels, screen-reader access, and print fidelity canvas cannot offer.

### The two hardening fixes

Both were deferred from piece C with the note that they are harmless with one widget and hazardous
with two. `Frontier` is the second widget, so D1 is the moment.

* **`mountAll` wraps each widget invocation in try/catch**, so one widget throwing no longer prevents
  every later widget on the page from mounting, and the failure is made visible *in that mount point*
  via C's `showError` rather than console-only.
* **`register` throws on a duplicate name** instead of silently replacing the first registration.

## §4 The widget

The Methods index has no figure today, so D1 puts the committed
`examples/method-comparison/frontier_ZAF.9.3.1_1_40972.png` there through the proven `_copy_asset` +
`_figure` path, with the widget attributes **on the `<figure>` itself** — a wrapping `<div>` escapes
the figure grid's CSS reset, which C's final review caught.

**Fallback parity, as for PermGraph.** That PNG draws its two dashed guides at the calibrated
`matched_displacement = 0.10` and `matched_permeability = 0.60` (`conf/permeability.yaml:39,52`). The
widget therefore **boots with both target lines at those values**, emitted into the bundle and carried
on the mount point as data attributes — the boot state living in the page beside the figure it
replaces, which is where C's ruling put it. Boot anywhere else and the widget contradicts the image it
just removed.

**Controls, with the keyboard path built in rather than bolted on.** Two native
`<input type="range">` controls, one per axis, are the keyboard- and screen-reader-operable route;
dragging the lines on the chart is the mouse affordance. Both write the same state, mirroring
PermGraph's slider. Hovering a curve reads that method's exact pair at that prefix; the legend
isolates a method. A text readout names which methods clear the current targets and at what road
cost, so every number the chart shows is also present as text.

**The answer is exact, not interpolated.** Displacement and permeability are both monotone in prefix,
so "which methods clear `P*`, and at what least road" is a binary search over each method's baked
array — the identical search `budget.prefix_to_permeability` performs in Python, over the identical
sequence. That is the parent design's compute-model claim exercised rather than asserted.

## §5 Testing

**Python.**

* **Parity:** the baked table's per-prefix permeability must equal
  `permeability(block, ordered.iloc[:m])` at every prefix, at the emitted precision — the same shape
  as C's parity test, and the reason to trust a committed artifact nothing recomputes downstream.
* **Artifact vs artifact:** the terminal values must agree with the committed
  `frontier_permeability.csv` and `lens_permeability.csv` rows (e.g. `clearance` at 89.4 m / 0.625391,
  `topology` at 93.0 m / 0.606794). This catches a changed pin — the failure C's review showed a
  parity test structurally *cannot* see, because it shares its loader with the baker.
* **Encoding/target constants:** the bundle's baked `matched_displacement` / `matched_permeability`
  must equal the live `conf/permeability.yaml` values, so the widget's boot state cannot drift from
  the guides drawn on its own fallback PNG.

**Node.**

* `niceTicks` — round values, count near the target, range covered.
* The generalised transform — the six existing tests updated, plus the `fitBbox` returns `sx === sy`
  invariant and `fitAxes`.
* The target binary search as a pure function over a monotone array: least index clearing a target.
  Directly fault-injectable.
* The bundle-evaluation smoke test extended to assert **both** widgets register, not just one.

**Neither.** There are no browser or end-to-end tests in D1. Piece C learned that this boundary is
only defensible if something still evaluates the shipped artifact — a circular import made C's bundle
throw on every page load behind an intact PNG fallback, caught by review rather than by any test, and
needing no browser to catch. The smoke test is that guard and it stays.

Every guard must be shown to fail before it counts, per the standard B and C were held to.

## §6 What D1 is not

* **No `ScreenMap`, no city tier.** D2, with §1's measurements to scope it.
* **No `DisplacementField`.** D3: it needs a static figure built first, exactly as permeability did in
  piece B, because the existing `after_*` renders overlay disks on a heatmap and would make fallback
  parity unsatisfiable by construction.
* **No `RegionGrow`.** It needs `RegionBuilder` to expose accretion order.
* **No region-scale prefix tables.** ~12 hours, recurring per regeneration; still deferred.
* **No URL-as-state.** Piece E; `StateSource` already exists as its injection point.

## Open items

* **Whether hovering a curve should snap to a prefix or read continuously.** The baked table is
  discrete, so continuous readout would interpolate between real prefixes and present a number that
  is not a measurement. Snapping to the nearest prefix is the honest default; decide by using it.
* **Whether `frontier_xmax = 0.40` (`conf/permeability.yaml:62`) should clip the widget's x axis as it
  clips the PNG's.** The PNG clips deliberately — methods have no common terminal, so autoscaling
  squashes the region where the guides sit — but a widget can pan, so the constraint may not transfer.
