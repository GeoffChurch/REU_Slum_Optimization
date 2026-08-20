# Piece D2: the DisplacementField widget, and reflow for the whole substrate

**Parent design:** `2026-08-13-site-redesign-design.md` §5, §7 (piece D).
**Predecessors:** `2026-08-14-perm-graph-artifact-design.md` (B), `2026-08-15-web-bundle-and-widget-substrate-design.md` (C),
`2026-08-16-frontier-widget-and-substrate-hardening-design.md` (D1).

**Sequencing note.** D1's §1 table labels `ScreenMap` as D2 and `DisplacementField` as D3. The parent
design's §7 says `ScreenMap` goes **last within D**, because 16k blocks is the only real rendering-
performance problem in the whole design, and the backlog entry written after D1 shipped says D2 is
`DisplacementField`. Two of three sources agree, and D1's column was a *dependency* table rather
than a sequencing decision. **D2 is `DisplacementField`.** `ScreenMap` stays last.

## Why

`docs/methodology/displacement.md` is 43 lines of prose and **zero figures**. It makes four subtle
modelling claims — width is per-road, overlap is free, parcels are not buildings, gap-hugging is
free — and a reader has to take every one of them on faith. Permeability was in exactly this state
before piece B; it now has the best figure on the site.

Displacement's claims are also *comparative* in a way permeability's are not. Each one is "move the
road, watch the cost change". A static figure can show one road; only an interactive one can show
that moving it into a gap costs nothing, or that pulling two roads apart costs *more*.

## §1 The finding that shapes this piece

**Displacement is exactly computable in a browser.** For a road set where each road carries its own
width,

```
dist(p, ∪ᵢ buffer(Lᵢ, wᵢ/2))  =  minᵢ max(0, dist(p, Lᵢ) − wᵢ/2)
```

because a buffer *is* the set of points within `wᵢ/2` of the line, and distance to a union is the
minimum over the union's parts. So the whole metric needs point-to-segment distance and nothing
else: no polygon union, no geometry library, no Pyodide.

Measured against `budget.displacement` on the pinned block (`ZAF.9.3.1_1_40972`, 263 buildings,
median radius 2.19 m), for all eight methods the example runs:

| method | segments | shapely | closed form | rel |
|---|---|---|---|---|
| `clearance` | 92 | 102.3728 | 102.3888 | 1.6e-04 |
| `clearance_looped` | 274 | 217.6588 | 217.6764 | 8.1e-05 |
| `cycle_native` | 93 | 51.2719 | 51.2880 | 3.1e-04 |
| `euclidean_grid` | 9 | 123.9661 | 123.9667 | 4.9e-06 |
| `greedy_arterial_access_displacement` | 337 | 138.1986 | 138.2589 | 4.4e-04 |
| `osm_footpaths` | 106 | 90.3466 | 90.3619 | 1.7e-04 |
| `resistance_lp` | 227 | 52.5456 | 52.5624 | 3.2e-04 |
| `topology` | 228 | 178.7863 | 178.8301 | 2.4e-04 |

The closed form is **higher every time**, and on a synthetic case the gap collapses quadratically as
shapely's buffer resolution rises — 7.2 mm at `quad_segs=16` (the default), 0.45 mm at 64, 0.028 mm
at 256. So the disagreement is *shapely's* discretisation error, not the formula's: shapely's buffer
is an inscribed polygon, slightly smaller than the true round buffer, so it reports slightly larger
distances and slightly smaller `cᵢ`. The closed form is the more exact of the two.

**Consequence for this piece:** the widget computes the project's real metric on an arbitrary road
position, live, with no Pyodide. `PermGraph` could not — permeability needs a sparse solve, which is
why C baked a prefix table.

**Consequence for piece F:** F was scoped as "draw your own road" behind a pinned Pyodide boot. Half
of F's payload — the *cost* half — needs none of that, and D2 will have shipped the drag UI
separately. F reduces to permeability-in-Pyodide on a road the reader has already drawn.

## §2 What the reader does

The widget boots showing the block: parcels as a pale wireframe, the block boundary and existing
streets, every building as a disk of its own radius `rᵢ`, and **two roads** — one live, one toggled
off. Each road is a straight segment with a draggable handle at each end. A width slider sets the
live road's `width_m`. A readout gives `Σcᵢ` and the fraction, recomputed every frame.

Three of the page's four claims become falsifiable in seconds:

* **gap-hugging is free** — drag the road into a gap between disks and the cost falls to exactly 0,
  because `cᵢ` clips at `dᵢ = rᵢ`;
* **width is per-road** — the slider moves the cost without moving the road;
* **overlap is free** — switch on the second road and drag it onto the first: the cost *drops* as
  the two corridors merge into one.

The fourth (**parcels are not buildings**) is carried by the drawing itself: the wireframe cells and
the disks are visibly different objects, and the cost tracks the disks.

**The width slider floors at 7 m.** `PermeabilityParams.min_road_width_m` is 7.0
(`src/reblock/permeability.py:125`), `DEFAULT_ROAD_WIDTH_M` is 7.0 (`:151`), and a narrower road is
**rejected with an exception** (`:205-209`) as too narrow for two directions. A widget that let a
reader build a road the pipeline refuses would be teaching a model the project does not have. Range
7–20 m, default 7 m, step 0.5 m — all baked, not literals in TypeScript.

**No permeability readout.** The benefit half of the tradeoff needs a sparse solve per road position
(Pyodide, piece F) or region-scale prefix tables (~12 h, still deferred). The widget reports **cost
only**, and the prose must not imply the reader's road is *good* — only what it costs.

### The default road, and why it is a rule rather than a hand-placed line

Both roads are derived deterministically so the bake, the figure and the caption agree:

* **Road 1** runs along the **principal axis of the building field** — the first principal component
  of the building points through their centroid — clipped to the block boundary, longest resulting
  interior segment. The PC sign is normalised so the direction's larger-magnitude component is
  positive, or the axis flips run to run.
* **Road 2** is road 1 translated perpendicular by `3 × width_m`, clipped the same way — far enough
  that the two corridors start disjoint, so merging them is something the reader *does*.

Any deterministic rule would serve; what matters is that it is a rule, so the PNG and the widget's
boot state are the same road and the caption's numbers are measurements of it.

## §3 The Python half: a field figure

D1 deferred this piece precisely because no static figure existed, and the closest existing render
makes parity **unsatisfiable by construction**: `render_after` draws displaced disks at `alpha = cᵢ`
*on top of the depth choropleth* (`src/reblock/render.py:178-186`), so disk shading and parcel fill
compete in the same pixels. A widget drawing disks over a wireframe can never match it.

`render.py` gains `render_field`, modelled on `render_graph` (`:258`), sharing its layering:

1. parcels as a pale wireframe — `_CONTEXT_OUTLINE`, linewidth 0.4 — **never filled**, the same
   finding piece B recorded for `render_graph`: filling states one quantity twice and drowns the
   subject;
2. the corridor, **dissolved per width group and then buffered** at that group's half-width, at
   `_ROAD_COLOR` alpha 0.25. This is the `render.py:304-319` rule and it is load-bearing here: a
   translucent patch per road compounds toward opaque wherever roads overlap, which is exactly the
   region this figure exists to show as *cheap*;
3. `_draw_boundary_and_streets` (`:118`);
4. **every** building as a disk of radius `rᵢ` — grazed ones filled `_DISPLACED_PT` at `alpha = cᵢ`,
   zero-cost ones as a thin outline of the same colour.

(4) is the departure from `render_after`, which draws only the displaced disks. Drawing the
zero-cost disks too is the whole point: without them a reader cannot see that a road **threaded a
gap**, only that some homes went red. `_DISPLACED_PT` (`#c0392b`, `render.py:47`) rather than
`render_after`'s inline `(1.0, 0.0, 0.0, c)`: a named constant is a thing the generator can bake
into the bundle, where a literal inside a function body would have to be duplicated in TypeScript.

`scripts/gen_displacement_field.py` bakes, on the same pinned block every other example baker uses
(`scripts/_example_block.py`, one pin, one module):

* `examples/displacement-field/field.png` — the widget's **boot state**: road 1 alone at 7 m,
  because road 2 defaults off (§2), so fallback parity is checkable against what the reader first
  sees;
* `examples/displacement-field/field.json` — the widget's payload;
* caption numbers, including `Σcᵢ` for the two roads **apart** and for the two roads **dragged
  together**. The no-JS reader gets the overlap-is-free comparison as two measured numbers in the
  caption; the JS reader gets it as motion. One PNG, not a pair.

### Page wiring

`docs/_partials/displacement.md` gains a `<!-- DISPFIELD -->` marker; `gen_site_pages.py` gains a
`_displacement_field_figure()` producer registered in `MARKERS` (`:1119-1132`), emitting the
`<figure>` with its `data-widget`, `data-bundle` and fallback `<img>` exactly as
`_frontier_figure()` (`:382`) does for the Methods index. The generator copies both artifacts into
`docs/assets/displacement-field/` through the existing `_copy_asset` (`:101`), so the mount point's
`data-bundle` is a served URL rather than a path into `examples/`.

`gen_site_pages.py` stays **stdlib-only and must never import `reblock`** — every number it prints
comes from an artifact on disk. The caption's `Σcᵢ` values are read from the baked JSON, not
recomputed.

## §4 The bundle

`examples/displacement-field/field.json`, self-contained, with a generated `web/src/field.d.ts`
beside it — the piece-C contract. Reuses `gen_web_bundle.py`'s quantisers verbatim: `_r` (6
significant figures, field values) and `_c` (absolute 0.01 m, coordinates relative to `origin`).
Never mix them: 6 significant figures on a ~6,240,000 UTM northing quantises to 10 m.

```
block_id, origin: [ox, oy], n_buildings
buildings: { x: [], y: [], r: [] }            # _c, _c, _r
parcels:   [[[x, y], ...], ...]               # _c, pale wireframe
boundary:  [[x, y], ...]
streets:   [[[x, y], ...], ...]
roads:     [{ coords: [[x, y], [x, y]], width_m }, ...]   # road 1, road 2 (§2)
width:     { floor_m: 7.0, max_m: 20.0, step_m: 0.5, default_m: 7.0 }
encoding:  { parcel_color, parcel_lw, boundary_color, boundary_lw, street_lw,
             road_color, road_alpha, disk_color, disk_outline_lw, handle_radius_px, pad }
reference: [{ roads: [...], sum_c, fraction } × 5]        # §6 parity fixtures
```

Payload is trivial: 263 buildings × 3 numbers. `parcels` dominates, and the perm-graph bundle
already carries the same 263 rings at 278 KB total for this block.

**Self-contained rather than sharing `examples/perm-graph/bundle.json`**, even though both widgets
sit on the same block and that bundle already holds `parcels`, `boundary`, `streets` and `origin`.
Sharing would couple two widgets' payloads so that retuning one silently changes the other. The
generator *code* is shared instead — same block loader, same quantisers — which makes data drift
impossible while leaving each widget's artifact its own.

**Close the `.d.ts` gap here.** Piece C left "nothing asserts the committed `.d.ts` equals the
generator's own template" open. The new script gets that guard: a test comparing the committed
`web/src/field.d.ts` to the script's `DTS_TEMPLATE` byte for byte.

## §5 The widget, and three DOM-free modules

* **`web/src/model/displacement.ts`** — DOM-free, the metric: `corridorDistance(bx, by, segs)` and
  `sumC(radii, d)`. DOM-free so §6's parity test needs no fake DOM at all, and so the one piece of
  arithmetic that must agree with Python is testable in isolation.
* **`web/src/render/field.ts`** — the draw, mirroring §3's four layers in the same order.
  `render/canvas.ts`'s `draw` is hardwired to the perm-graph `Bundle`; this is a sibling, not a
  parameterisation of it.
* **`web/src/dom/resize.ts`** — §7.
* **`web/src/dom/fallback.ts`** — `removeFallbackImage(host)`: drops the `<img>` *and* the glightbox
  anchor wrapping it (§9). One place, since all three widgets now do this and two of them do it
  wrong.
* **`web/src/widgets/displacement-field.ts`** — the widget: handles, slider, toggle, readout.

Canvas, not SVG: a drag frame redraws 263 disks plus the corridor, and `sizeCanvas`
(`render/canvas.ts`) already handles `devicePixelRatio`. Interaction uses **Pointer Events**, as
both existing widgets do, so touch works without a second code path.

Per frame the widget recomputes `cᵢ` for all 263 buildings against a handful of segments. For
comparison, the heaviest baked method on this block is 337 segments and Python evaluates it in
milliseconds.

## §6 Testing

Every guard must be shown to fail before it counts — the standard B, C and D1 were held to.

1. **The formula, pinned in Python.** `tests/` asserts
   `displacement_from_distance(radii, closed_form(...)) ≈ displacement(bp, radii, roads)` for all
   eight methods on the pinned block, at the tolerance §1 measured. This pins the identity the
   TypeScript implements *in the language where ground truth lives*, so a future change to
   `corridor_distance` breaks here rather than in a widget nobody re-runs.
2. **Python ↔ TypeScript parity.** The bake writes `reference`: five road configurations with Python's
   `Σcᵢ` for each. A DOM-free test feeds the same coordinates through
   `model/displacement.ts` and asserts agreement within 1e-3 relative. The fixtures must include the
   two default roads apart, the two coincident (overlap), one at 20 m, one threading a gap, and one
   entirely outside the block — the last asserted as **exactly 0**, not within tolerance.
3. **Figure ↔ widget parity.** The widget's boot state against the committed PNG, D1's pixel-level
   pattern, plus the baked colours (§3) so neither can drift from the other.
4. **Reflow.** §7 — containment at 320 / 700 / 1200 px, driven by a fake `ResizeObserver`.
5. **The marker closed set.** `gen_site_pages.py`'s existing unknown-marker test already catches a
   `<!-- DISPFIELD -->` with no producer.

## §7 Reflow — and the recorded plan for it is wrong

The backlog records: *"Both size in absolute pixels with no `viewBox`, so a container narrowing
without a `window` resize event overflows … The real fix is a `viewBox`."* Both halves are wrong.

**What the code actually does.** `perm-graph.ts` sets `cv.style.width = "100%"` with
`aspectRatio = "1 / 1"`, so its canvas cannot overflow. `frontier.ts` measures
`host.getBoundingClientRect().width` at mount (`:244-248`) and renders at that width. Both re-measure
on `window` resize. So **neither overflows at mount, and both already handle window resizes.**

**The real gap** is a *container* resize with no window resize — Material's nav drawer at some
breakpoints, a `<details>` opening, a tab panel switching, print. There, Frontier's absolute-pixel
SVG overflows its narrowed container, and PermGraph's canvas stretches a stale backing store: right
aspect, wrong resolution, and a view fitted to the old box.

**The fix is a `ResizeObserver`**, in one shared `web/src/dom/resize.ts`, replacing both
`window.addEventListener("resize", …)` sites. Not a `viewBox`, which would be actively worse: a
`viewBox` scales text down with the box, so Frontier's 11 px axis labels land at ~5 px on a 320 px
screen. Re-laying out at the measured width keeps type at its designed size and re-nices the ticks
for the narrower span — which is what `niceTicks` is for.

Three things fall out of it:

* **Reflow becomes testable for the first time.** The fake DOM can fire the observer at a chosen
  width, so `svg.test.ts`'s single containment assertion becomes a sweep over widths. A label that
  escapes the plot rect only at 320 px is currently invisible to the whole suite.
* **The deferred zero-width throw resolves honestly.** A zero box means "not laid out yet", so the
  observer skips it and draws on the next callback. `measure`'s throw is deleted, not moved.
* **PermGraph's fallback removal must move.** It removes the `<img>` before the first draw
  (`perm-graph.ts`, right after inserting the canvas), so a zero-width mount would leave a blank
  figure with the static image already gone. Frontier removes it *after* a successful render, which
  D1 introduced for error honesty and which is exactly what makes skip-on-zero safe. PermGraph
  adopts the same order.

## §8 Prose: one truth-pass item on the page D2 is touching

`docs/_partials/displacement.md` says, under *Parcels are not buildings*: **"a parcel with no
building standing on it costs nothing to cross."** But `src/reblock/mesh.py:59` records that parcels
are **Voronoi cells of the building points**, so the correspondence is exactly one point per parcel
— 263 and 263 on the pinned block. A parcel with no building is not a vacant lot; it is a
*degenerate geometry*, which `parcel_radii` handles by assigning radius 0. The published sentence
describes a case this pipeline does not produce.

The distinction the section is reaching for is true and more interesting: displacement is charged
per building against **its own radius `rᵢ = NN/2`**, never per parcel against parcel *area*. A road
crossing one large sparse parcel is charged by its distance to the single building in it, not by how
much land it consumes. Rewrite the section to say that. Same defect class the piece-A truth pass
existed to fix, on the page this piece is already editing.

## §9 Riders

Small, on code paths this piece already touches. Each is a decision, not a sweep:

* **glightbox's orphaned `<a>`.** `mkdocs-glightbox` wraps every figure image in
  `<a class="glightbox" href="…png">`; both widgets remove the `<img>` and leave the anchor —
  focusable, announced by a screen reader as a link with no text, which cuts against the very
  accessibility rationale for drawing SVG. `PermGraph` ships this live today. Remove the anchor when
  the image was its only child.
* **`fitBbox`'s uniformity test** asserts only `scaleX === scaleY`, so a `Math.max`-for-`Math.min`
  regression stays green. Assert the width/height-bound **minimum**.
* **`data-block` is emitted, tested, and read by nobody.** Delete it — from both widgets' mount
  points and from `gen_site_pages.py`. Every bundle already carries `block_id`, and a second source
  of one fact is drift waiting to happen.

## §10 What D2 is not

* **No `ScreenMap`, no city tier.** Last within D: 16,451 blocks / 843,838 vertices, and the
  simplification tolerance is a real design choice that piece owns.
* **No `RegionGrow`.** Still blocked on `RegionBuilder` exposing accretion order
  (`src/reblock/region.py:353` sorts it away).
* **No Pyodide.** §1 is precisely that displacement needs none.
* **No permeability on the reader's road.** §2.
* **No `viewBox`.** §7 — not deferred, *rejected*, with the reason.
* **No region-scale prefix tables.** ~12 h, recurring per regeneration. Still deferred.
* **No URL-as-state.** Piece E; `StateSource` is already its injection point.

## Open items

* **Whether the handles should snap to anything.** They currently will not: a road position is
  continuous and the metric is exact at any position, so there is no grid to snap to and nothing
  dishonest about a continuous readout — unlike D1's frontier hover, where the baked table was
  discrete. Decide by using it.
* **Whether the second road should default on.** Off keeps the boot state simple and matches the
  PNG; on makes the overlap demo discoverable without the reader finding a toggle. Off, unless
  using it says otherwise.
* **Whether `render_field` should also replace the `after_*` figures' disk treatment.** Those draw
  only displaced disks on a choropleth (§3). Not touched here — ~90 committed PNGs regenerate — but
  the same argument for drawing zero-cost disks applies to them.
