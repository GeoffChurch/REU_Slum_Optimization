# Piece C: the data bundle and the widget substrate

**Date:** 2026-08-15
**Branch:** to be created off `main`
**Status:** design, approved section by section; not yet implemented
**Scope of this document:** piece C of the site redesign
(`specs/2026-08-13-site-redesign-design.md` §3–§5, §7 C). Consumes piece B's `GraphFigure`
unchanged (`specs/2026-08-14-perm-graph-artifact-design.md`). Pieces D–F each get their own spec.

## Why

Piece B gave the Permeability page four static figures. This makes one of them live: a reader drags
roads into the block along the drainage order and watches current concentrate into each new corridor
while permeability climbs. The parent design calls this the best teaching image the project has; C is
where it stops being an image.

C is also the piece that decides whether the *rest* of the widgets are cheap or expensive, because it
builds the substrate they mount into. That is why it ships exactly one widget: `PermGraph` is the
smallest payload with the biggest teaching payoff, and B already built a Python twin of it, so the
two can be diffed rather than trusted.

**What ships:** a committed JSON bundle for one block, a generated `.d.ts` over it, a `web/`
TypeScript tree with type-checking and bundling wired into CI, a renderer-agnostic view substrate,
and `PermGraph` mounted on the Permeability page with B's PNG as its fallback.

## §1 Scope, and what measurement removed from it

The parent design's piece C bakes three tiers and all seven flagships' prefix tables. Two
measurements cut that down.

**The city tier belongs to piece D, not C.** The 16,451-block score table (~3 MB) exists for
`ScreenMap`. C's proving widget reads one block. Baking the city tier here builds a payload nothing
in C consumes, so it moves to D where its consumer lives.

**The parent design's prefix-table cost estimate is wrong by roughly 40×.** §3 claims "With R ≲ 60
that is ≤60 rows per (block, method) — *smaller* than what is committed today." R is the
drainage-ordered segment count, and the committed `run.log`s give it directly. Calibrating per-solve
cost from the two curve timings in those same logs — 4.2 s for 8 methods × 21 samples at 263 parcels
(26 ms/solve), 214.2 s for 6 × 21 at 11,006 parcels (1.8 s/solve) — and interpolating linearly in
parcel count:

| flagship | parcels | Σ segments | est. full-table bake |
|---|---|---|---|
| method-comparison | 263 | 784 | 0.3 min |
| multiblock_depth_density | 2,690 | 3,222 | 22 min |
| nairobi/multiblock_density_compactness | 3,547 | 5,509 | 49 min |
| nairobi/multiblock_depth | 4,365 | 6,714 | 74 min |
| multiblock_density_compactness | 4,615 | 5,393 | 63 min |
| nairobi/multiblock_depth_density | 5,095 | 7,246 | 94 min |
| multiblock_depth | 11,006 | 14,728 | 7 h |

**~12 hours**, recurring on every examples regeneration rather than once. Region prefix tables
therefore wait until a widget needs a region, and whoever picks that up owns the cost decision with
these numbers in hand.

**What C bakes instead is one method on one block.** `clearance` on the pinned block
`ZAF.9.3.1_1_40972` is **20 segments / 486 m**, so its full prefix table is 21 states and costs about
**half a second** of solving. One method, not eight: per-prefix fields for all eight would be ~1.4 M
values (~8 MB), method comparison is `Frontier`'s job in piece D, and `clearance` is what B's PNGs
already show — so widget and fallback agree by construction.

## §2 The bake

**New script `scripts/gen_web_bundle.py`**, importing `reblock`. Two corrections to the parent design
follow from CI's shape:

* §3 says `gen_site_pages.py` emits the JSON. **It cannot.** That script is stdlib-only by contract —
  `deploy-site.yml` builds the site with only `mkdocs-material` installed and `reblock` is not
  importable there (see `_load_friendly_method_name`'s docstring, `scripts/gen_site_pages.py:48`) —
  while baking needs `geopandas` and the solver. The bundle is a **committed artifact**, exactly as
  `examples/perm-graph/*.png` already are.
* It lands at `examples/perm-graph/bundle.json` and reaches the site through the existing
  `_copy_asset` path into `docs/assets/perm-graph/`. That reuses a proven route, keeps generated files
  out of tracked `docs/`, and needs no new serving mechanism.

### What the bundle carries

Road-independent, baked once:

* `origin` — the UTM easting/northing subtracted from every coordinate below, so all geometry is
  **local metres**. Two reasons, and the first is a correctness trap: a Cape Town UTM northing is
  ~6,240,000, so rounding coordinates by *significant digits* would quantize them to 10 m and
  dissolve the parcels while the file still parsed and the widget still drew. Coordinates are
  therefore rounded to an absolute 0.01 m, and translating them to a local origin makes that cheap —
  3–4 digits instead of 7. `width_m` is a length, so translation leaves it alone.
* `parcels` — polygon rings, local metres. **No reprojection anywhere:** the canvas fits the bbox
  and draws, and never learns the CRS. 0.01 m is far below one screen pixel at any plausible zoom.
* `nodes` — `cx`, `cy` per parcel (from `GraphFigure`).
* `edges` — `rows`, `cols`, and the road-independent `footpath_g`.
* `roads` — `clearance`'s segments in `street_first_ordered` order (`src/reblock/budget.py:643`),
  each with its `width_m`, so prefix *m* is the first *m* segments and needs no further ordering
  logic in the browser.

Per prefix `m = 0…20`:

* `potential[m]` — 263 floats.
* `current[m]` — 745 floats.

Monotone, so stored once rather than per prefix:

* `first_upgraded_at` — one int per edge: the least *m* at which a road raised that edge, or a
  sentinel for never. `upgraded` is monotone in the road set because conductance enters only through
  `max(footpath, road)`, so the widget derives prefix *m*'s mask as `first_upgraded_at[e] <= m`. This
  replaces 21 × 745 booleans with 745 integers **and** makes the monotonicity that the metric rests on
  explicit in the artifact.

**No per-prefix `conductance` array, and this is not an omission.** The conductance layer's widths
come from `footpath_g` for mesh edges — road-independent, baked once — while road-raised edges draw at
the fixed `_UPGRADED_LW`. So piece B's decision to stop encoding magnitude in upgraded-edge width has a
second payoff here: the conductance layer needs no per-prefix data at all, and the only thing that
changes across the slider on that layer is which edges are blue.

Floats are emitted at 6 significant digits. That is far beyond what a canvas can show or the readout
quotes. The per-prefix arrays come to ~21,900 values ≈ 260 KB, and with parcel rings, edge endpoints
and road segments the bundle lands near **300 KB** — inside the parent design's ~0.5 MB block budget.
§7's parity test is stated against that precision rather than against exact equality.

### The encoding travels with the data

The parent design says the palette comes from Python. C widens that: **the whole encoding does.**
Emitted alongside the geometry —

* the per-layer mesh-only width normalization and `_EDGE_LW_MIN` / `_EDGE_LW_MAX`;
* `_UPGRADED_LW`, the fixed width road-raised edges draw at;
* `_NODE_RADIUS_FRAC`;
* `_ROAD_COLOR`, `_BOUNDARY_COLOR`, `_CONTEXT_OUTLINE`, `_EDGE_GREY`;
* the **sampled `YlOrRd` ramp** — because `_PERM_CMAP` is the *string* `"YlOrRd"`
  (`src/reblock/render.py:41`), a matplotlib colormap name, and the browser has no matplotlib. A
  hand-written JS approximation of that ramp is precisely the "same block renders in two palettes on
  one page" drift §5 warns about. Emit 256 RGB stops.

This is piece B's `upgraded` lesson one level up: computed once in Python, consumed everywhere, so the
widget cannot draw the same data by different rules than the image it replaces.

## §3 The bundle contract

`gen_web_bundle.py` emits `bundle.json` **and** a generated `web/src/bundle.d.ts` describing it, both
committed. The `.d.ts` is what makes a renamed Python field a TypeScript error instead of a blank
panel.

**The drift guard.** One script writes both files, so they move together — but nothing stops someone
editing the Python and re-baking nothing. A structural test asserts the committed bundle's key set
matches the committed `.d.ts` declaration. It is textual and fast: no block loading, no solving, so it
runs in the normal suite rather than being something a contributor must remember.

## §4 Toolchain

**`web/`** holds the TypeScript, with `package.json` and `package-lock.json` committed so `npm ci` is
reproducible. Two dev dependencies, pinned exactly the way `mkdocs-material==9.7.7` already is:

* **`esbuild`** bundles `web/src/` → `docs/js/widgets.js`;
* **`typescript`** checks it. These are separate jobs and both are needed — **esbuild strips types
  without looking at them**, so esbuild alone would leave §3's `.d.ts` no power whatsoever.

**Node appears in two places, deliberately.**

* `nodejs` joins `[tool.pixi.dependencies]`, and `pixi run typecheck` grows `tsc --noEmit` beside
  mypy. `ci.yml` already runs `pixi run typecheck` on pull requests, so a type error fails the PR —
  where the feedback belongs.
* `deploy-site.yml` gains a pinned `actions/setup-node`, `npm ci`, and the esbuild step, ordered
  before `mkdocs build --strict`. It keeps its minimal pip environment; switching that workflow to
  pixi would install the whole scientific stack to build a docs site.

Both Node majors are pinned to the same value. `docs/js/` is added to `.gitignore` beside
`docs/assets/` (`.gitignore:65`), and `mkdocs.yml` gains `extra_javascript: [js/widgets.js]` next to
its existing `extra_css`. `navigation.instant` is confirmed absent from `mkdocs.yml`'s features, so
the parent design's warning about it needs no work — but it must stay absent.

**A guard against the quiet failure.** Because the bundle is gitignored, a site built without the
esbuild step emits a `<script>` tag for a file that does not exist: every widget silently fails to
boot, the PNG fallbacks still render, and the page looks *fine*. So `gen_site_pages.py` asserts
`docs/js/widgets.js` exists whenever a page it writes carries a widget mount point. File existence
needs no imports, so this respects the stdlib-only contract, and it converts a silent 404 into a build
failure.

## §5 The substrate

The parent design calls this "a projected canvas renderer". That conflates two layers, and separating
them is what keeps piece D cheap:

* **`web/src/view/transform.ts`** — renderer-agnostic and DOM-free: world-metres ↔ screen transform,
  fit-to-bbox, pan and zoom, and hit-testing as nearest-mark queries in world space. Pure functions,
  unit-tested in Node.
* **`web/src/render/canvas.ts`** — the only module that knows a 2D context exists. Takes a transform
  plus baked geometry; draws parcels, edges, nodes; handles `devicePixelRatio`.
* **`web/src/state.ts`** — the `StateSource` a widget reads. In C it is built from the mount point's
  `data-*` attributes and mutated locally by the slider. Piece E swaps in a URL-synced shared store,
  and the widget never learns which it has — which is the entire reason to inject it instead of
  growing an `if (embedded)` branch per widget.
* **`web/src/mount.ts`** — scans `[data-widget]`, resolves the name, injects a `StateSource`, replaces
  the fallback image. The name arrives from HTML, a genuinely open boundary, so a string lookup is
  correct here — **with no default, so an unknown widget name throws** rather than rendering nothing.
  Validated once, at that boundary.

**Why the split matters beyond tidiness.** `Frontier` in piece D is a *chart*: axis ticks, series
labels, tooltips, print fidelity, screen-reader access. SVG is straightforwardly better for it, and
forcing it through canvas would mean hand-drawing axis text and losing selectable labels. Meanwhile
`ScreenMap`'s 16,451 polygons make canvas mandatory there — 16k DOM nodes recoloured on every gate
change is SVG's worst case, and recolour *is* the interaction. A transform layer both can share, with
mark-drawing free to differ, is the boundary that serves both. It also makes a later move of
`ScreenMap` to WebGL/WebGPU a contained change rather than a substrate rewrite, should the block count
ever grow by an order of magnitude. At 16k polygons it would buy nothing perceptible today.

**No `overlay.ts` abstraction in C** — that would be inventing a shared pattern from one example.
`PermGraph` owns its own controls DOM. Accessibility is met concretely rather than abstractly: a
native `<input type="range">` for the slider, which is keyboard-reachable and screen-reader-legible
for free, and a text readout of the current prefix's road length and permeability, so every number the
picture shows is also present as text. Any transition honours `prefers-reduced-motion`.

## §6 PermGraph

```html
<div data-widget="perm-graph"
     data-block="ZAF.9.3.1_1_40972"
     data-method="clearance">
  <!-- B's graph_current_after.png sits here as the fallback -->
</div>
```

Controls: a prefix slider over `m = 0…20`, a layer switch between conductance-width and
current-width mirroring `render_graph`'s `layer` argument, a ground-halo toggle, and a hover readout
of a node's φ.

**Fallback parity is a requirement, not a nicety.** The mount point contains
`graph_current_after.png`, whose caption — generated from the artifact — says "89 m of road, reaching
62.5% permeability". That is `clearance`'s Lens-B prefix, 89.4 m of its 486 m total, so **the widget
must boot at that prefix index**, which the baker emits. Booting anywhere else replaces the image with
a visibly different picture while the caption beneath still describes the old one. The slider then
runs *past* the published figure, all the way to the full 486 m network — which is exactly the value
the widget adds over the PNG, since no static figure on the site shows that.

## §7 Testing

Split by language, because the interesting test is not in the browser.

**Python — the parity test, and the reason to trust the bake.** For every prefix `m`, the bundle's
`potential[m]` and `current[m]` must equal `permeability_graph(block, prefix_m).potential` and
`.current`, at the 6-significant-digit precision §2 emits. This is the "diff against the Python twin"
the parent design promised in exchange for building B first. Break the baker's prefix indexing, its
road ordering, or its rounding and this fails. Plus §3's structural bundle ↔ `.d.ts` key check.

**Node — `transform.ts`'s pure functions.** Fit-to-bbox, a pan/zoom round-trip (screen → world →
screen is the identity), and hit-testing returning the true nearest mark. No DOM, no browser.

**Neither — there are no browser or end-to-end tests in C.** The proof is the parity test plus looking
at the page. Stating that plainly is better than implying coverage that does not exist.

Every guard here must be shown to fail before it counts, per the standard piece B was held to.

## §8 What C is not

* **No city tier.** The 16,451-block score table moves to piece D with `ScreenMap`.
* **No region prefix tables.** ~12 hours of compute, deferred to whoever first needs a region widget,
  with §1's per-flagship numbers to decide on.
* **No other widgets.** `DisplacementField`, `Frontier`, `RegionGrow`, `ScreenMap` are piece D.
* **No URL-as-state.** `StateSource` exists so piece E can supply one; C ships the frozen-plus-local
  implementation only.
* **No Pyodide.** Piece F.
* **No second method.** `clearance` alone; the bundle shape generalizes, the payload does not.

## Open items

* **Whether `first_upgraded_at` needs a sentinel at all** on this block — if every mesh edge is
  eventually upgraded by the full 486 m network the field is dense, and if none are late it may
  compress further. A detail for the implementer to observe and note rather than guess at now.
* **Whether the layer switch should offer conductance at all.** B found the conductance-after image
  differs from before only by the blue overlay, since the mesh is road-independent — and §2's
  no-per-prefix-conductance consequence is the same fact stated in data terms: dragging the slider on
  that layer changes *only* which edges are blue. Interactively that may read as a switch that does
  almost nothing, in which case current alone is the honest control and conductance stays a static
  figure. Decide by using it, not in advance.
