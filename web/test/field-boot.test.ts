import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { FieldBundle, ReferenceCase, Road } from "../src/field.js";
import { contributions, corridorDistance, flatten } from "../src/model/displacement.js";
import { handles } from "../src/render/field.js";
import { localState } from "../src/state.js";
import { fitBbox, toScreen, toWorld, type Bbox } from "../src/view/transform.js";
import { displacementField } from "../src/widgets/displacement-field.js";
import {
  armDrawFailure, canvasOf, Call, DPR, FakeElement, fireResize, installStubs, lastFrame,
  mountPoint as mountPointBase,
} from "./harness.js";

/** The DisplacementField widget, mounted for real against the COMMITTED `field.json`.
 *
 * A canvas is the most dangerous surface this branch has tested, because it offers nothing to
 * introspect: you cannot ask it what was drawn, only what calls were made. Every assertion below is
 * therefore about a RECORDING CONTEXT's call sequence -- and each one names the layer it means
 * first (by the bundle colour that identifies it, or by the shape of its path) and only then
 * asserts on it. Task 1 lost two guards to the other habit: a test that searches for *something*
 * satisfying a property accepts a match from whichever layer happens to satisfy it, and one of them
 * was reading the corridor's alpha while claiming to read the disks'.
 *
 * Same minimal-stub spirit as perm-graph-boot.test.ts and frontier-boot.test.ts: no jsdom, one fake
 * element class, one recording 2D context. Neither the widget's module body nor anything it imports
 * touches `document`, `window` or `ResizeObserver` at evaluation time -- only their function bodies
 * do -- so static imports above and stubs installed below are in the right order.
 */
installStubs();

const BUNDLE_PATH = "../examples/displacement-field/field.json";
const bundle = JSON.parse(readFileSync(BUNDLE_PATH, "utf8")) as FieldBundle;
const E = bundle.encoding;
const SIZE = 700;

/** Python's own number for a road configuration, by name -- the fixtures `gen_displacement_field.py`
 * baked with `budget.displacement`. Quoting these rather than recomputing is what makes the readout
 * assertions parity assertions: the number on the page is the number Python measured. */
function reference(name: string): ReferenceCase {
  const c = bundle.reference.find((r) => r.name === name);
  assert.ok(c !== undefined, `field.json has no reference case "${name}"`);
  return c;
}

/** The view the widget must fit: buildings UNIONED WITH the parcel rings. Every expected screen
 * coordinate below comes through this, so a widget fitting the buildings alone (the shape PermGraph
 * ships, which works on this block only because the pad absorbs it) fails on real numbers rather
 * than on a restated intention. `viewFits` below proves the two are actually distinguishable. */
function unionBbox(b: FieldBundle): Bbox {
  const xs = [...b.buildings.x, ...b.parcels.flatMap((ring) => ring.map((p) => p[0]))];
  const ys = [...b.buildings.y, ...b.parcels.flatMap((ring) => ring.map((p) => p[1]))];
  return { minX: Math.min(...xs), minY: Math.min(...ys),
           maxX: Math.max(...xs), maxY: Math.max(...ys) };
}
const VIEW = fitBbox(unionBbox(bundle), SIZE, SIZE, E.pad);

/** Road `r`'s vertex `v` in screen pixels, projected through the same fit the widget builds --
 * computed here, never hardcoded. A hardcoded pixel pair is the defect D1's re-review found as N2:
 * it passes for one canvas size and silently stops meaning anything at any other. */
function handleAt(r: number, v: number): { x: number; y: number } {
  const road = bundle.roads[r];
  assert.ok(road !== undefined, `field.json has no road ${r}`);
  const pt = road.coords[v];
  assert.ok(pt !== undefined, `road ${r} has no vertex ${v}`);
  const [x, y] = toScreen(VIEW, pt[0], pt[1]);
  return { x, y };
}

/** field-boot's own addition to the shared mount point: the bundle URL every generated figure
 * carries in `data-bundle`. The DOM shape itself (figure, glightbox anchor, image, figcaption) comes
 * from the harness -- it is identical to what perm-graph-boot.test.ts mounts against. */
function mountPoint(): FakeElement {
  const figure = mountPointBase();
  figure.dataset["bundle"] = BUNDLE_PATH;
  return figure;
}

async function mount(host: FakeElement, drawFailure: string | null = null,
                     payload: unknown = bundle): Promise<void> {
  armDrawFailure(drawFailure);
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true,
    status: 200,
    statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(payload),
  });
  displacementField(host as unknown as HTMLElement, localState);
  // A macrotask, so the fetch chain has drained by the time this resolves.
  await new Promise((resolve) => setTimeout(resolve, 0));
  armDrawFailure(null);
}

/** Exactly the state a call CONSUMES, and nothing else.
 *
 * A canvas context keeps its last style until something reassigns it, so a frame legitimately
 * starts holding whatever the previous frame left: `clearRect` records a `fillStyle`, and a stroke
 * records a `fillStyle` it cannot possibly use. Comparing whole `Call`s across frames would flag
 * all of that as a leak, so the comparison is over what each op actually paints with -- which keeps
 * `lineCap` on strokes, where a leak really does change the pixels. */
function consumed(c: Call): Record<string, unknown> {
  if (c.op === "clearRect") return { op: c.op };
  if (c.op === "fill") return { op: c.op, fillStyle: c.fillStyle, alpha: c.globalAlpha, path: c.path };
  return { op: c.op, strokeStyle: c.strokeStyle, lineWidth: c.lineWidth, lineCap: c.lineCap,
           alpha: c.globalAlpha, path: c.path };
}

const isArc = (c: Call): boolean => c.path.length === 1 && c.path[0]?.op === "arc";
const isPolyline = (c: Call): boolean => c.path.some((p) => p.op === "lineTo");
const arcArgs = (c: Call): number[] => {
  const p = c.path[0];
  assert.ok(p !== undefined && p.op === "arc", "not an arc path");
  return p.args;
};

/** The four layers `render_field` draws, each named by what identifies it and nothing else. */
function layers(cv: FakeElement): {
  parcels: Call[]; corridor: Call[]; outline: Call[]; streets: Call[];
  diskOutlines: Call[]; diskFills: Call[]; handleFills: Call[]; handleOutlines: Call[];
} {
  const f = lastFrame(cv);
  const strokes = f.filter((c) => c.op === "stroke");
  const fills = f.filter((c) => c.op === "fill");
  // The boundary and the streets share one colour (`_draw_boundary_and_streets` draws both in
  // `_BOUNDARY_COLOR`) and are told apart by ORDER, which is the order the Python draws them in:
  // the block ring first, then every street line.
  const boundaryish = strokes.filter((c) => c.strokeStyle === E.boundary_color && isPolyline(c));
  return {
    parcels: strokes.filter((c) => c.strokeStyle === E.parcel_color),
    corridor: strokes.filter((c) => c.globalAlpha === E.road_alpha),
    outline: boundaryish.slice(0, 1),
    streets: boundaryish.slice(1),
    diskOutlines: strokes.filter((c) => c.strokeStyle === E.disk_color && isArc(c)),
    diskFills: fills.filter((c) => c.fillStyle === E.disk_color && isArc(c)),
    handleFills: fills.filter((c) => c.fillStyle === E.road_color && isArc(c)),
    handleOutlines: strokes.filter((c) => c.strokeStyle === E.boundary_color && isArc(c)),
  };
}

/** `Σcᵢ` as the readout states it, parsed back out of the page. */
function cost(host: FakeElement): number {
  const p = host.find("p");
  assert.ok(p !== null, "the widget wrote no readout");
  const m = /([\d.]+) homes displaced/.exec(p.textContent);
  assert.ok(m !== null, `the readout does not state a cost: ${JSON.stringify(p.textContent)}`);
  return Number(m[1]);
}

/** The picture prices the road it is drawn beside -- asserted against the model for the road set
 * that is live RIGHT NOW, not the one the widget booted with.
 *
 * This is the guard the `contributions()` split exists to make possible, and testing it only at
 * boot is what made it worthless: the reviewer cached the boot frame's `contributions` while
 * leaving the readout live (the picture freezes, the number keeps moving) and stroked the corridor
 * from the BAKED coordinates instead of the dragged ones (the road draws where it started while the
 * cost reports where it was dragged to), and both left the whole suite green. Every interaction
 * test below ends here, so the two halves of the figure are pinned to each other after the reader
 * has actually done something.
 *
 * `roads` is built by the caller from `toWorld(VIEW, ...)` of the very pointer positions it
 * dispatched, so the expected coordinates are bit-identical to the widget's own -- no tolerance is
 * needed anywhere below, and none is used. */
function assertPictureMatchesRoads(cv: FakeElement, roads: Road[], why: string): void {
  const { x, y, r } = bundle.buildings;
  const c = contributions(r, corridorDistance(x, y, flatten(roads)));
  const grazed: number[] = [];
  const missed: number[] = [];
  for (let i = 0; i < r.length; i++) (c[i]! > 0 ? grazed : missed).push(i);
  assert.ok(grazed.length > 0 && missed.length > 0,
    `${why}: this road neither grazes nor misses anything, so both branches must be live here`);
  const l = layers(cv);

  // The CORRIDOR is drawn along the road the readout is pricing. One stroke per distinct width
  // (the slider sets every live road, so in practice one), and the vertices of every road in it, in
  // order -- which is what a corridor stroked from stale coordinates cannot produce.
  const widths = [...new Set(roads.map((road) => road.width_m))];
  assert.equal(l.corridor.length, widths.length, `${why}: one stroke per width group`);
  l.corridor.forEach((stroke, k) => assert.equal(stroke.lineWidth, widths[k]! * VIEW.scaleX,
    `${why}: the corridor is not at the live road width`));
  assert.deepEqual(
    l.corridor.flatMap((stroke) => stroke.path.map((op) => op.args)),
    roads.flatMap((road) => road.coords.map(([wx, wy]) => [...toScreen(VIEW, wx, wy)])),
    `${why}: the corridor is not drawn where the road now is`);

  // The DISKS are shaded by the cost of that same road: one fill per grazed building at exactly its
  // own alpha, one outline per untouched one, each at its own place and radius.
  assert.equal(l.diskFills.length, grazed.length, `${why}: the grazed disks and the model disagree`);
  assert.equal(l.diskOutlines.length, missed.length,
    `${why}: the untouched disks and the model disagree`);
  grazed.forEach((i, k) => {
    const call = l.diskFills[k]!;
    assert.equal(call.globalAlpha, c[i]!,
      `${why}: disk ${i} was filled at alpha ${call.globalAlpha}, its cost is ${c[i]!}`);
    assert.deepEqual(arcArgs(call).slice(0, 3),
      [...toScreen(VIEW, x[i]!, y[i]!), r[i]! * VIEW.scaleX],
      `${why}: disk ${i} is not at its own place and radius`);
  });
  // The zero-cost disks are outlines at FULL alpha -- not faint versions of a grazed disk, but a
  // different statement: this home is not touched at all.
  for (const call of l.diskOutlines) assert.equal(call.globalAlpha, 1, why);
}

/** A road as the widget holds it after a drag: the reader's pointer, put back through the same
 * `toWorld` the widget used. */
function roadWith(road: Road, moved: Record<number, { x: number; y: number }>,
                  width_m = bundle.width.default_m): Road {
  return {
    width_m,
    coords: road.coords.map((pt, v): [number, number] => {
      const to = moved[v];
      return to === undefined ? pt : toWorld(VIEW, to.x, to.y);
    }),
  };
}

function drag(cv: FakeElement, from: { x: number; y: number }, to: { x: number; y: number }): void {
  cv.dispatch("pointerdown", { offsetX: from.x, offsetY: from.y, pointerId: 1 });
  cv.dispatch("pointermove", { offsetX: to.x, offsetY: to.y, pointerId: 1 });
  cv.dispatch("pointerup", { pointerId: 1 });
}

test("boots, draws every layer in render_field's own order, and quotes Python's own number",
  async () => {
    const host = mountPoint();
    await mount(host);
    fireResize(SIZE, SIZE);

    const cv = canvasOf(host);
    assert.ok(!host.find("figcaption")!.textContent.includes("could not load"),
      host.find("figcaption")!.textContent);
    // Sized with an INLINE STYLE, never a presentation attribute: Material's
    // `.md-typeset svg{height:auto;max-width:100%}` beats presentation attributes, which cost D1 a
    // Critical at its final gate.
    assert.equal(cv.style["width"], "100%", "the canvas was not sized with an inline style");
    assert.equal(cv.style["aspectRatio"], "1 / 1");
    // Without this a touch drag scrolls the page and the browser cancels the pointer stream.
    assert.equal(cv.style["touchAction"], "none");
    // The backing store is the observed box times devicePixelRatio -- a stretched canvas with a
    // stale backing store is a blurred picture that nothing reports.
    assert.equal(cv.width, SIZE * DPR);
    assert.equal(cv.height, SIZE * DPR);
    assert.deepEqual(cv.ctx.transforms.at(-1), [DPR, 0, 0, DPR, 0, 0]);

    const l = layers(cv);
    const frame = lastFrame(cv);
    assert.equal(frame[0]?.op, "clearRect", "the frame did not start by clearing");

    // (1) Parcels: one wireframe ring each, at the bundle's own weight and colour, NEVER filled --
    // a filled parcel states one quantity twice and drowns the disks that are the subject.
    assert.equal(l.parcels.length, bundle.parcels.length, "a parcel ring went missing");
    for (const c of l.parcels) assert.equal(c.lineWidth, E.parcel_lw);
    assert.equal(frame.filter((c) => c.op === "fill" && c.fillStyle === E.parcel_color).length, 0,
      "a parcel was filled");

    // (2) The corridor: ONE stroke, translucent, in the bundle's road colour, at the road's own
    // width in screen pixels.
    assert.equal(l.corridor.length, 1, "the corridor was not drawn as a single stroke");
    assert.equal(l.corridor[0]!.strokeStyle, E.road_color);
    assert.equal(l.corridor[0]!.lineWidth, bundle.width.default_m * VIEW.scaleX,
      "the corridor's width is not the road's own width scaled by the fitted view");

    // (3) Boundary and streets: one ring plus one line per street, each at ITS OWN baked weight.
    // Task 3 found `street_lw` baked as 1.0 while render.py draws streets at 1.3 -- a JS-on reader
    // saw thinner streets than the fallback PNG. This pair of assertions is what would catch that
    // divergence reopening from the TypeScript side.
    assert.equal(l.outline.length, 1, "the block boundary was not drawn");
    assert.equal(l.outline[0]!.lineWidth, E.boundary_lw);
    assert.equal(l.streets.length, bundle.streets.length, "a street line went missing");
    for (const c of l.streets) assert.equal(c.lineWidth, E.street_lw);

    // (4) Every building, and every one of them exactly once.
    assert.equal(l.diskOutlines.length + l.diskFills.length, bundle.n_buildings,
      "the disks and the buildings disagree about how many there are");
    for (const c of l.diskOutlines) assert.equal(c.lineWidth, E.disk_outline_lw);

    // Context state, not a count. The corridor is the only layer that wants round caps; every layer
    // after it must have them reset, or the boundary, the streets and the disks quietly inherit
    // them -- and the parcels inherit them on frame 2+ only, which is a first-paint-differs bug
    // that no single frame can show. The frame-identity test below is the other half of this.
    assert.equal(l.corridor[0]!.lineCap, "round");
    for (const c of [...l.outline, ...l.streets, ...l.diskOutlines, ...l.handleOutlines]) {
      assert.equal(c.lineCap, "butt", "a layer drawn after the corridor inherited its round caps");
    }

    // (5) One handle per endpoint of the one live road, and they are drawn LAST -- under nothing.
    assert.equal(l.handleFills.length, bundle.roads[0]!.coords.length);
    assert.equal(l.handleOutlines.length, l.handleFills.length, "a handle lost its outline");
    for (const c of l.handleFills) assert.equal(arcArgs(c)[2], E.handle_radius_px,
      "the handle is not the baked pixel radius");

    // ORDER, which is the half that cannot be seen from counts. The corridor must sit UNDER the
    // disks (render.py's zorder 2 against 5 and 6): a translucent corridor drawn over them would
    // tint the very shading it exists to be read against. And handles last, so a handle sitting on
    // a solid disk is still grabbable.
    const at = (c: Call): number => frame.indexOf(c);
    assert.ok(at(l.parcels.at(-1)!) < at(l.corridor[0]!), "the corridor was drawn under the parcels");
    assert.ok(at(l.corridor[0]!) < at(l.outline[0]!), "the boundary was drawn under the corridor");
    assert.ok(at(l.outline[0]!) < at(l.diskOutlines[0] ?? l.diskFills[0]!),
      "a disk was drawn under the boundary");
    assert.ok(at(l.corridor[0]!) < at(l.diskFills[0]!),
      "the corridor was drawn OVER the disks, tinting the shading it exists to be read against");
    assert.ok(at(l.diskFills.at(-1)!) < at(l.handleFills[0]!), "a handle was buried under a disk");

    // The readout quotes PYTHON's number for the boot state -- road 1 alone at the default width,
    // which is exactly the `road1` fixture `budget.displacement` measured -- and both halves of it,
    // because the page defines displacement as Σcᵢ and reports the fraction.
    const road1 = reference("road1");
    const readout = host.find("p")!.textContent;
    assert.match(readout, new RegExp(`${road1.sum_c.toFixed(1)} homes displaced`.replace(".", "\\.")),
      `readout ${JSON.stringify(readout)} does not quote Python's own ${road1.sum_c}`);
    assert.match(readout, new RegExp(`${(road1.fraction * 100).toFixed(1)}% of `
      + `${bundle.n_buildings} buildings`.replace(".", "\\.")),
      `readout ${JSON.stringify(readout)} does not state the fraction of ${bundle.n_buildings}`);
  });

test("each grazed disk is drawn at exactly its own cost, in its own place, at its own radius",
  async () => {
    // The strongest guard in the file. The disks ARE the metric made visible: alpha is cᵢ, so a
    // widget that shaded them from a stale array, at a fixed alpha, or against the wrong radius
    // would still draw 263 plausible circles and report a correct-looking number beside them.
    const host = mountPoint();
    await mount(host);
    fireResize(SIZE, SIZE);

    assertPictureMatchesRoads(canvasOf(host), [bundle.roads[0]!], "at boot");
    // ...and the untouched disks are the half `render_after` leaves out. Without them a reader
    // cannot see that a road threaded a GAP, only that some homes went red.
    assert.ok(layers(canvasOf(host)).diskOutlines.length > 0);
  });

test("the view fits the parcels too, not just the buildings", async () => {
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);

  // Every parcel vertex the widget drew, in screen pixels, straight off the recording context.
  const drawn = layers(canvasOf(host)).parcels
    .flatMap((c) => c.path.filter((p) => p.op !== "closePath").map((p) => p.args));
  const xs = drawn.map((a) => a[0]!);
  const ys = drawn.map((a) => a[1]!);
  for (const v of [...xs, ...ys]) {
    assert.ok(v >= 0 && v <= SIZE, `a parcel vertex landed at ${v}, outside a ${SIZE} px canvas`);
  }

  // ...and the assertion above is not vacuous: under the fit PermGraph uses -- the BUILDING
  // centroids alone, with parcels drawn anyway -- a ring vertex leaves the canvas. Measured on this
  // block at this pad: the scale inflates 9.8 % and exactly 1 parcel vertex of 1850 lands 9.3 px
  // past the max-x edge of a 700 px canvas (the same vertex is on the boundary ring, which spans
  // the parcel bbox exactly). Small, real, and enough to make this assertion discriminate -- which
  // is all it is here to do.
  const buildingsOnly = fitBbox({
    minX: Math.min(...bundle.buildings.x), minY: Math.min(...bundle.buildings.y),
    maxX: Math.max(...bundle.buildings.x), maxY: Math.max(...bundle.buildings.y),
  }, SIZE, SIZE, E.pad);
  const escaped = bundle.parcels.flatMap((ring) => ring.map((p) => toScreen(buildingsOnly, p[0], p[1])))
    .filter(([sx, sy]) => sx < 0 || sx > SIZE || sy < 0 || sy > SIZE);
  assert.ok(escaped.length > 0,
    "a buildings-only fit would contain the parcels here, so the containment above proves nothing");
});

test("the fallback image and its glightbox anchor go only after a successful draw", async () => {
  const host = mountPoint();
  const img = host.find("img")!;
  const anchor = host.find("a")!;
  await mount(host);
  assert.equal(img.removedAt, null, "the image went before anything was drawn");

  fireResize(SIZE, SIZE);
  const cv = canvasOf(host);
  assert.notEqual(img.removedAt, null, "the fallback image outlived the drawing");
  // ORDER, not merely both: a picture must exist before the static one is taken away.
  assert.ok(img.removedAt! > cv.ctx.firstDrawAt!,
    `fallback removed at ${img.removedAt} but nothing was drawn until ${cv.ctx.firstDrawAt}`);
  // And the glightbox <a> goes with it: an anchor emptied of its image is invisible, still
  // focusable, and announced by a screen reader as a link with no text.
  assert.notEqual(anchor.removedAt, null, "the glightbox anchor outlived its image");
  assert.equal(host.findAll("a").length, 0, "an <a> survives inside the mounted figure");
});

test("a zero-width container draws nothing and leaves the static figure in place", async () => {
  // A hidden tab, a closed <details>, a print layout: the box is not laid out yet, which is not a
  // failure and must not be reported as one -- and the static picture is still the honest one.
  const host = mountPoint();
  await mount(host);
  fireResize(0, 0);

  const cv = canvasOf(host);
  assert.deepEqual(cv.ctx.calls, [], "drew into a zero-width canvas");
  assert.equal(cv.width, 0, "sized a backing store for a box with no width");
  assert.ok(host.find("img") !== null, "removed the fallback without drawing anything");
  assert.ok(host.find("a") !== null, "the lightbox link went with an image that is still needed");
  assert.ok(!host.find("figcaption")!.textContent.includes("could not load"),
    "a container that is merely not laid out yet is not a failure to report");
});

test("a throw while drawing reaches the caption and keeps the static image", async () => {
  // The resize callback runs from the browser's own dispatch, OUTSIDE the mount's
  // `fetch().then(boot).catch(...)` chain, so a throw in it is an uncaught exception and nothing
  // else -- a blank figure, no message, and a page that still looks laid out. That is this
  // branch's signature defect, and `runOrReport` is the only thing standing in front of it.
  const host = mountPoint();
  const img = host.find("img")!;
  await mount(host, "boom on the first draw");
  fireResize(SIZE, SIZE);

  assert.match(host.find("figcaption")!.textContent,
    /DisplacementField could not load interactively .*boom on the first draw/);
  assert.match(host.find("figcaption")!.textContent, /The static image above still applies\./);
  assert.equal(img.removedAt, null,
    "the fallback image was removed although the drawing that replaces it threw");
});

test("dragging a handle moves the road and re-prices it", async () => {
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);
  const cv = canvasOf(host);
  const before = cost(host);

  const from = handleAt(0, 0);
  const to = { x: from.x + 120, y: from.y + 90 };
  drag(cv, from, to);

  // The road MOVED: its handle is redrawn under the cursor, which is the only way to see that the
  // widget acted on the geometry rather than merely on a number.
  const moved = layers(cv).handleFills.map((c) => arcArgs(c).slice(0, 2));
  assert.ok(moved.some(([sx, sy]) => Math.abs(sx! - to.x) < 1e-6 && Math.abs(sy! - to.y) < 1e-6),
    `no handle followed the pointer to (${to.x}, ${to.y}); handles are at ${JSON.stringify(moved)}`);
  // ...and it was RE-PRICED. Σcᵢ is recomputed every frame; a drag that redraws without recomputing
  // leaves a correct picture beside a stale number, which is worse than either alone.
  assert.notEqual(cost(host), before, "the drag changed the road but not what it costs");
  // And the two halves still agree with each other AND with the model, for the road as it is NOW.
  assertPictureMatchesRoads(cv, [roadWith(bundle.roads[0]!, { 0: to })], "after the drag");
});

test("a press on empty canvas moves nothing", async () => {
  // Without the distance test in `pickHandle` the nearest handle teleports to the press -- a
  // hundred-metre jump from a click the reader did not mean as a drag, and their road is gone.
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);
  const cv = canvasOf(host);
  const before = cost(host);
  const handlesBefore = layers(cv).handleFills.map((c) => arcArgs(c));

  const far = handleAt(0, 0);
  drag(cv, { x: far.x + 4 * E.handle_radius_px, y: far.y + 4 * E.handle_radius_px },
       { x: 10, y: 10 });

  assert.deepEqual(layers(cv).handleFills.map((c) => arcArgs(c)), handlesBefore,
    "a press away from every handle still moved one");
  assert.equal(cost(host), before);
});

test("the width slider cannot go below the pipeline's own floor", async () => {
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);

  const slider = host.findAll("input").find((i) => i.type === "range");
  assert.ok(slider !== undefined, "there is no width slider");
  assert.equal(Number(slider.min), bundle.width.floor_m);
  assert.equal(Number(slider.min), 7,
    "permeability.py:205 RAISES below 7 m -- a narrower road is not a road this project has");
  assert.equal(Number(slider.max), bundle.width.max_m);
  assert.equal(Number(slider.step), bundle.width.step_m);
  assert.equal(Number(slider.value), bundle.width.default_m);
});

test("widening the road raises the cost, to Python's own number for that width", async () => {
  // Width is per-road, and the slider moves the cost WITHOUT moving the road -- the second of the
  // page's claims. `widest` is the same road 1 at 20 m, measured by `budget.displacement`.
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);
  const cv = canvasOf(host);
  const before = cost(host);
  const placedBefore = layers(cv).handleFills.map((c) => arcArgs(c));

  const widest = reference("widest");
  const slider = host.findAll("input").find((i) => i.type === "range")!;
  slider.value = String(widest.roads[0]!.width_m);
  slider.dispatch("input");

  assert.ok(cost(host) > before, `widening cost ${cost(host)}, the 7 m road cost ${before}`);
  assert.equal(cost(host).toFixed(1), widest.sum_c.toFixed(1),
    `the widened road costs ${cost(host)}; Python measures ${widest.sum_c}`);
  assert.deepEqual(layers(cv).handleFills.map((c) => arcArgs(c)), placedBefore,
    "the slider moved the road as well as its width");
  assert.equal(layers(cv).corridor[0]!.lineWidth, widest.roads[0]!.width_m * VIEW.scaleX,
    "the corridor was not redrawn at the new width");
  assertPictureMatchesRoads(cv, [roadWith(bundle.roads[0]!, {}, widest.roads[0]!.width_m)],
    "after widening");
});

test("switching the second road on RAISES the cost; dragging them together is what lowers it",
  async () => {
    // The direction matters and is easy to test backwards. A disjoint road can only ADD corridor,
    // so switching it on must cost more -- `apart`. The DROP the page claims comes from merging two
    // corridors into one, which is something the reader DOES -- `coincident`, which is exactly what
    // one road costs. Both numbers are Python's.
    const host = mountPoint();
    await mount(host);
    fireResize(SIZE, SIZE);
    const cv = canvasOf(host);
    const alone = cost(host);
    assert.equal(alone.toFixed(1), reference("road1").sum_c.toFixed(1));

    const toggle = host.findAll("input").find((i) => i.type === "checkbox");
    assert.ok(toggle !== undefined, "there is no second-road toggle");
    toggle.checked = true;
    toggle.dispatch("change");

    const apart = reference("apart");
    assert.ok(cost(host) > alone, `two roads cost ${cost(host)}, one cost ${alone}`);
    assert.equal(cost(host).toFixed(1), apart.sum_c.toFixed(1),
      `two roads apart cost ${cost(host)}; Python measures ${apart.sum_c}`);
    assert.equal(layers(cv).handleFills.length, 4, "the second road brought no handles");
    assertPictureMatchesRoads(cv, [bundle.roads[0]!, bundle.roads[1]!], "with both roads apart");

    // Now drag road 2 onto road 1, endpoint by endpoint. `toWorld` inverts the same `toScreen` the
    // handle positions came through, so the two roads end up coincident to float precision -- and
    // the cost must fall back to what ONE road costs, which is the whole overlap-is-free claim.
    drag(cv, handleAt(1, 0), handleAt(0, 0));
    drag(cv, handleAt(1, 1), handleAt(0, 1));

    const coincident = reference("coincident");
    assert.ok(cost(host) < apart.sum_c - 1,
      `merging the corridors did not lower the cost: ${cost(host)} against ${apart.sum_c} apart`);
    assert.equal(cost(host).toFixed(1), coincident.sum_c.toFixed(1),
      `two coincident roads cost ${cost(host)}; Python measures ${coincident.sum_c}, `
      + `which is what ONE road costs`);
    // Road 2 -- and ONLY road 2 -- ended up on road 1. This is also what pins `Handle.road` to the
    // array the drag writes back into: if the hit test indexed a different array from `pointermove`,
    // the wrong road would have moved and the corridor path below would be drawn somewhere else.
    assertPictureMatchesRoads(cv,
      [bundle.roads[0]!, roadWith(bundle.roads[1]!, { 0: handleAt(0, 0), 1: handleAt(0, 1) })],
      "after dragging road 2 onto road 1");
  });

test("two roads share ONE corridor stroke, so overlapping them cannot compound toward opaque",
  async () => {
    // Each stroke() is an independent compositing operation. A stroke PER ROAD would darken every
    // overlap -- drawing the exact opposite of the claim the test above measures, while every
    // number on the page stayed right. Nothing but the call log can see this.
    const host = mountPoint();
    await mount(host);
    fireResize(SIZE, SIZE);
    const cv = canvasOf(host);
    const toggle = host.findAll("input").find((i) => i.type === "checkbox")!;
    toggle.checked = true;
    toggle.dispatch("change");

    const corridor = layers(cv).corridor;
    assert.equal(corridor.length, 1,
      `two roads of one width were stroked ${corridor.length} times; they share one corridor`);
    assert.equal(corridor[0]!.globalAlpha, E.road_alpha);
    // One path holding both roads: two moveTo's, one per road, in a single stroked path.
    assert.equal(corridor[0]!.path.filter((p) => p.op === "moveTo").length, 2,
      "the single stroke did not carry both roads");
    // The handles prove the second road is really there, so the count above is not "one road drawn".
    assert.equal(layers(cv).handleFills.length, 4);
  });

test("the widget is re-fitted, not merely re-scaled, when its container changes width", async () => {
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);
  fireResize(320, 320);

  const cv = canvasOf(host);
  assert.equal(cv.width, 320 * DPR, "the backing store still matches the old box");
  const narrow = fitBbox(unionBbox(bundle), 320, 320, E.pad);
  const handle = layers(cv).handleFills[0]!;
  const first = handles([bundle.roads[0]!])[0]!;
  assert.deepEqual(arcArgs(handle).slice(0, 2), [...toScreen(narrow, first.x, first.y)],
    "the drawing was not re-fitted to the narrowed box");
  // The handle keeps its PIXEL radius across the reflow: a grab target that shrank with the box
  // would stop being grabbable exactly where the box is smallest.
  assert.equal(arcArgs(handle)[2], E.handle_radius_px);
});

test("every frame is identical: no context state leaks from one draw into the next", async () => {
  // A widget whose first paint differs from its second is a genuinely nasty thing to chase later,
  // and `ctx` is full of state that outlives the call that set it (`lineCap`, `globalAlpha`,
  // `lineWidth`, both styles). Redrawing at the SAME size must produce the same call log, call for
  // call, including every style snapshot -- which is a much stronger statement than any per-layer
  // assertion, and the only one that can see a leak that only exists from frame 2 onwards.
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);
  const first = lastFrame(canvasOf(host));
  fireResize(SIZE, SIZE);
  const second = lastFrame(canvasOf(host));

  assert.deepEqual(second.map(consumed), first.map(consumed),
    "the second paint differs from the first: context state leaked across frames");
  assert.ok(first.length > 500, `only ${first.length} calls compared, which is not a whole frame`);
});

test("the readout is announced, because a canvas says nothing to a screen reader", async () => {
  // A screen-reader user moving the width slider with the arrow keys hears the width they set and
  // nothing about what it cost -- the entire subject of the figure, silent -- unless the one line
  // that changes is a live region. `frontier.ts` set the precedent inside this same branch.
  const host = mountPoint();
  await mount(host);
  fireResize(SIZE, SIZE);

  const readout = host.find("p");
  assert.ok(readout !== null, "the widget wrote no readout");
  assert.equal(readout.getAttribute("aria-live"), "polite",
    "the number that changes on every frame is not announced");
});

test("a bundle that is not exactly two roads is refused, loudly, with the image left in place",
  async () => {
    // External JSON, so this is a boundary check rather than an unreachable guard -- and a THIRD
    // road is the dangerous direction: it would be silently dropped, charged by nothing and drawn
    // by nothing, and the reader would be given a cost for a road set that is not the one on disk.
    for (const [what, roads] of [
      ["a third road", [...bundle.roads, bundle.roads[0]!]],
      ["only one road", [bundle.roads[0]!]],
    ] as const) {
      const host = mountPoint();
      const img = host.find("img")!;
      await mount(host, null, { ...bundle, roads });

      assert.match(host.find("figcaption")!.textContent,
        new RegExp(`DisplacementField could not load interactively .*${roads.length} roads`),
        `${what} was accepted or reported without saying what was wrong`);
      assert.equal(img.removedAt, null, `${what}: the fallback image went anyway`);
      assert.equal(host.find("canvas"), null, `${what}: a canvas was inserted for a refused bundle`);
    }
  });

test("a mount point with no data-bundle names the ATTRIBUTE, not a missing file", async () => {
  // `host.dataset.bundle!` would reach `fetch(undefined)` and surface as "fetch undefined failed:
  // 404", sending whoever wrote the page looking for a file rather than for the attribute they
  // forgot. The throw is synchronous, before the fetch chain exists, which is exactly the path
  // `mountAll` catches and renders through dom/error.ts (mount.test.ts covers that half).
  const host = mountPoint();
  delete host.dataset["bundle"];
  assert.throws(() => displacementField(host as unknown as HTMLElement, localState),
    /DisplacementField: data-bundle is missing/);
});
