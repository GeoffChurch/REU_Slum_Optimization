import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { FieldBundle, ReferenceCase } from "../src/field.js";
import { contributions, corridorDistance, flatten } from "../src/model/displacement.js";
import { handles } from "../src/render/field.js";
import { localState } from "../src/state.js";
import { fitBbox, toScreen, type Bbox } from "../src/view/transform.js";
import { displacementField } from "../src/widgets/displacement-field.js";

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
let clock = 0;

interface PathOp { op: "moveTo" | "lineTo" | "arc" | "closePath"; args: number[] }

/** One drawing call, with the whole style state it was made under and the path it painted.
 *
 * The style snapshot is the point. `ctx.strokeStyle` is a mutable property, so "the widget set
 * road_color at some moment" says nothing about what was stroked with it; recording the value AT
 * the call is what makes "the corridor is drawn in the bundle's road colour at the bundle's alpha"
 * an assertion rather than a hope. The path lets each layer be named by its own shape -- polylines
 * for parcels, boundary and streets, a single arc for a disk or a handle -- which is how the
 * boundary strokes stay distinguishable from the handle outlines that share their colour. */
interface Call {
  op: "clearRect" | "stroke" | "fill";
  strokeStyle: string;
  fillStyle: string;
  lineWidth: number;
  globalAlpha: number;
  path: PathOp[];
}

/** Armed by `mount(host, width, drawFailure)` to make the first drawing call throw. It has to be a
 * module variable rather than a field on the element: the canvas is created inside `boot()`, which
 * runs a microtask after the widget function returns, so no test holds a reference to it in time. */
let NEXT_DRAW_FAILURE: string | null = null;

class RecordingContext {
  readonly calls: Call[] = [];
  failWith: string | null = NEXT_DRAW_FAILURE;
  fillStyle = "";
  strokeStyle = "";
  lineWidth = 0;
  lineCap = "";
  globalAlpha = 1;
  readonly transforms: number[][] = [];
  /** The instant of the first drawing call, on the same counter `FakeElement` stamps creations and
   * removals with -- so a test can assert the fallback image went after a PICTURE existed, not
   * merely after a canvas element did. The element is created either way; the drawing is not. */
  firstDrawAt: number | null = null;
  private path: PathOp[] = [];

  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void {
    this.transforms.push([a, b, c, d, e, f]);
  }
  /** Real `beginPath` discards the current path; real `stroke`/`fill` do NOT. The handles rely on
   * exactly that (one arc, filled and then stroked), so the fake has to model it or the second call
   * would record an empty path and the handle-outline assertions would be vacuous. */
  beginPath(): void { this.path = []; }
  closePath(): void { this.path.push({ op: "closePath", args: [] }); }
  moveTo(x: number, y: number): void { this.path.push({ op: "moveTo", args: [x, y] }); }
  lineTo(x: number, y: number): void { this.path.push({ op: "lineTo", args: [x, y] }); }
  arc(x: number, y: number, r: number, a0: number, a1: number): void {
    this.path.push({ op: "arc", args: [x, y, r, a0, a1] });
  }
  clearRect(): void { this.record("clearRect"); }
  stroke(): void { this.record("stroke"); }
  fill(): void { this.record("fill"); }

  private record(op: Call["op"]): void {
    if (this.failWith !== null) throw new Error(this.failWith);
    this.calls.push({
      op,
      strokeStyle: this.strokeStyle,
      fillStyle: this.fillStyle,
      lineWidth: this.lineWidth,
      globalAlpha: this.globalAlpha,
      path: [...this.path],
    });
    this.firstDrawAt ??= ++clock;
  }
}

class FakeElement {
  readonly tagName: string;
  children: FakeElement[] = [];
  parent: FakeElement | null = null;
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  textContent = "";
  /** Only a canvas has one; the widget reads it back through `getContext`. */
  readonly ctx = new RecordingContext();
  /** The backing store `sizeCanvas` sets from the observed box times devicePixelRatio. */
  width = 0;
  height = 0;
  /** The <input> surface the widget writes and the tests read back. */
  type = "";
  min = "";
  max = "";
  step = "";
  value = "";
  checked = false;
  readonly createdAt = ++clock;
  removedAt: number | null = null;

  /** UPPERCASE, as a real element reports it -- `dom/fallback.ts` decides whether the wrapper around
   * the fallback image is the glightbox anchor by testing `tagName === "A"`. */
  constructor(tagName: string) { this.tagName = tagName.toUpperCase(); }
  getContext(): RecordingContext { return this.ctx; }
  append(...nodes: (FakeElement | string)[]): void {
    for (const n of nodes) {
      if (typeof n === "string") {
        const text = new FakeElement("#TEXT");
        text.textContent = n;
        text.parent = this;
        this.children.push(text);
      } else {
        n.parent = this;
        this.children.push(n);
      }
    }
  }
  insertBefore(node: FakeElement, _ref: FakeElement | null): void {
    node.parent = this;
    this.children.push(node);
  }
  /** DETACHES, as a real `remove()` does -- `dom/fallback.ts` asks whether the anchor has element
   * children left after the image goes, and a fake that only timestamped would answer "yes". */
  remove(): void {
    this.removedAt = ++clock;
    if (this.parent) {
      this.parent.children = this.parent.children.filter((c) => c !== this);
      this.parent = null;
    }
  }
  get parentElement(): FakeElement | null { return this.parent; }
  readonly handlers: Record<string, ((ev: unknown) => void)[]> = {};
  addEventListener(name: string, fn?: (ev: unknown) => void): void {
    if (fn) (this.handlers[name] ??= []).push(fn);
  }
  /** Dispatches a real event object to real handlers, so the pointer tests drive the widget the way
   * a reader's finger does -- not by asserting that a listener was merely registered. */
  dispatch(name: string, ev: unknown = {}): void {
    for (const fn of this.handlers[name] ?? []) fn(ev);
  }
  /** Present so the widget's drag can call them, and deliberately NOT counted: no assertion here
   * could tell a captured drag from an uncaptured one, because the whole difference is what the
   * BROWSER does with events after the pointer leaves the element -- and there is no browser here.
   * Deleting both calls from the widget leaves this file green, which is reported rather than
   * papered over with an assertion that an API was called. */
  setPointerCapture(): void {}
  releasePointerCapture(): void {}
  querySelector(selector: string): FakeElement | null { return this.find(selector); }
  descendants(): FakeElement[] { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  find(tagName: string): FakeElement | null {
    return this.descendants().find((c) => c.tagName === tagName.toUpperCase()) ?? null;
  }
  findAll(tagName: string): FakeElement[] {
    return this.descendants().filter((c) => c.tagName === tagName.toUpperCase());
  }
  text(): string {
    return [this.textContent, ...this.descendants().map((c) => c.textContent)].join(" ");
  }
}

(globalThis as Record<string, unknown>).document = {
  createElement: (tag: string): FakeElement => new FakeElement(tag),
};
/** `render/canvas.ts` reads `devicePixelRatio` off it, and 2 rather than 1 is what makes the
 * backing-store assertion able to fail: at 1 the scaled size and the CSS size coincide. */
const DPR = 2;
(globalThis as Record<string, unknown>).window = { devicePixelRatio: DPR };

/** The observer the widget lays itself out from. It delivers NOTHING on `observe()`: there is no
 * layout engine here, so the test plays layout with `fireResize`, and it does so from its own body
 * -- after the mount's `fetch().then(boot).catch(...)` chain has settled. That is where a real
 * ResizeObserver delivers from too (the browser's own dispatch, never from inside `observe()`), and
 * it is what makes `runOrReport` load-bearing rather than redundant: a throw in the callback has no
 * promise `.catch` above it, which is exactly what the drawing-failure test below exercises. */
class FakeResizeObserver {
  static live: FakeResizeObserver[] = [];
  constructor(private readonly cb:
              (entries: { contentRect: { width: number; height: number } }[]) => void) {
    FakeResizeObserver.live.push(this);
  }
  observe(): void {}
  disconnect(): void {}
  fire(width: number, height: number): void { this.cb([{ contentRect: { width, height } }]); }
}
(globalThis as Record<string, unknown>).ResizeObserver = FakeResizeObserver;

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

/** The mount point the generator will emit: the `<figure>` itself, carrying the bundle URL, with the
 * fallback `<img>` inside the `<a class="glightbox">` mkdocs-glightbox wraps it in, and a
 * `<figcaption>` after it. A fixture with a bare `<img>` would let a widget that removes only the
 * image pass while the live page kept an empty, focusable, screen-reader-announced link. */
function mountPoint(): FakeElement {
  const figure = new FakeElement("figure");
  const anchor = new FakeElement("a");
  anchor.append(new FakeElement("img"));
  figure.append(anchor, new FakeElement("figcaption"));
  figure.dataset["bundle"] = BUNDLE_PATH;
  return figure;
}

async function mount(host: FakeElement, drawFailure: string | null = null): Promise<void> {
  NEXT_DRAW_FAILURE = drawFailure;
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true,
    status: 200,
    statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(bundle),
  });
  displacementField(host as unknown as HTMLElement, localState);
  // A macrotask, so the fetch chain has drained by the time this resolves.
  await new Promise((resolve) => setTimeout(resolve, 0));
  NEXT_DRAW_FAILURE = null;
}

function fireResize(width: number, height: number): void {
  const obs = FakeResizeObserver.live.at(-1);
  assert.ok(obs !== undefined, "the widget never observed anything");
  obs.fire(width, height);
}

function canvasOf(host: FakeElement): FakeElement {
  const cv = host.find("canvas");
  assert.ok(cv !== null, "no canvas was inserted into the figure");
  return cv;
}

/** The calls of the LAST frame only. Every control writes state, and state re-renders, so a test
 * that looked at the whole log would be reading the boot frame and the current one at once. */
function lastFrame(cv: FakeElement): Call[] {
  const starts = cv.ctx.calls.flatMap((c, i) => c.op === "clearRect" ? [i] : []);
  const from = starts.at(-1);
  assert.ok(from !== undefined, "nothing was ever drawn: no clearRect in the call log");
  return cv.ctx.calls.slice(from);
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

    const { x, y, r } = bundle.buildings;
    const c = contributions(r, corridorDistance(x, y, flatten([bundle.roads[0]!])));
    const grazed: number[] = [];
    const missed: number[] = [];
    for (let i = 0; i < r.length; i++) (c[i]! > 0 ? grazed : missed).push(i);
    assert.ok(grazed.length > 0 && missed.length > 0,
      "the boot road neither grazes nor misses anything -- both branches must be live here");

    const l = layers(canvasOf(host));
    assert.equal(l.diskFills.length, grazed.length, "the grazed disks and the model disagree");
    assert.equal(l.diskOutlines.length, missed.length,
      "the untouched disks were not all drawn -- a reader cannot see a road threaded a GAP without them");
    grazed.forEach((i, k) => {
      const call = l.diskFills[k]!;
      assert.equal(call.globalAlpha, c[i]!,
        `disk ${i} was filled at alpha ${call.globalAlpha}, its cost is ${c[i]!}`);
      const [sx, sy] = toScreen(VIEW, x[i]!, y[i]!);
      assert.deepEqual(arcArgs(call).slice(0, 3), [sx, sy, r[i]! * VIEW.scaleX],
        `disk ${i} is not at its own place and radius`);
    });
    // The zero-cost disks are outlines, and their alpha is FULL -- they are not faint versions of a
    // grazed disk, they are a different statement (this home is not touched at all).
    for (const call of l.diskOutlines) assert.equal(call.globalAlpha, 1);
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
  // centroids alone, with parcels drawn anyway -- the same rings leave the canvas. The backlog
  // records that shape working by luck on this block (one vertex of 1850, 0.4 px out, absorbed by
  // the pad); here the buildings span 5.8-149.5 m inside parcels spanning 0-157.8 m, so the luck
  // does not hold, and this is the number that says so.
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
