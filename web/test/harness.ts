import { strict as assert } from "node:assert";

/** The shared fake DOM and recording 2D context every canvas widget test mounts against.
 *
 * Extracted from `field-boot.test.ts` and `perm-graph-boot.test.ts`, which had grown two copies of
 * the fake DOM. Only field-boot's had a `Call`-shaped recording context at all -- perm-graph's
 * `RecordingContext` recorded bare operation-name strings, with no path or style snapshot -- so the
 * richer version is the one kept here, and perm-graph-boot.test.ts's own comparisons against a bare
 * op name are adapted below to read `Call.op` instead. `lineCap` and `globalAlpha` are both
 * load-bearing (see `Call`).
 *
 * No jsdom on purpose -- one fake element class and one recording context, in the minimal-stub
 * spirit the three boot tests already shared. Neither the widgets' module bodies nor anything they
 * import touches `document`, `window` or `ResizeObserver` at evaluation time, only their function
 * bodies do, so a test file's static imports and the stubs `installStubs()` puts in place are in
 * the right order either way.
 */
let clock = 0;

export interface PathOp { op: "moveTo" | "lineTo" | "arc" | "closePath"; args: number[] }

/** One drawing call, with the whole style state it was made under and the path it painted.
 *
 * The style snapshot is the point. `ctx.strokeStyle` is a mutable property, so "the widget set
 * road_color at some moment" says nothing about what was stroked with it; recording the value AT
 * the call is what makes "the corridor is drawn in the bundle's road colour at the bundle's alpha"
 * an assertion rather than a hope. The path lets each layer be named by its own shape -- polylines
 * for parcels, boundary and streets, a single arc for a disk or a handle -- which is how the
 * boundary strokes stay distinguishable from the handle outlines that share their colour. */
export interface Call {
  op: "clearRect" | "stroke" | "fill" | "drawImage";
  strokeStyle: string;
  fillStyle: string;
  lineWidth: number;
  /** Recorded because it is CONTEXT STATE that outlives the call that set it: the corridor wants
   * round caps and nothing else does, so a missing reset gives every later layer round caps AND
   * makes frame 1 (which reaches those layers before the corridor has ever run) differ from frame
   * 2. Neither is visible in a count. */
  lineCap: string;
  globalAlpha: number;
  path: PathOp[];
  /** The fill rule a `fill()` call was made with -- `"nonzero"` (the real canvas default, and what
   * every OTHER op is recorded with too, since none of them take one) unless the caller passed
   * `"evenodd"` explicitly. Added because it used to be silently ignored: `city.ts` fills polygons
   * that carry interior rings (holes) and MUST pass `"evenodd"`, or the holes render solid, and
   * nothing before this field existed could tell the two apart from a recorded `Call`. */
  fillRule: CanvasFillRule;
  /** Set only on a `drawImage` call: the element that was blitted. In this fake it is always
   * another `FakeElement` (created via `document.createElement("canvas")`, the way a real offscreen
   * buffer would be too), so a test can follow this reference to that element's OWN `ctx.calls` --
   * e.g. to confirm an offscreen base layer was painted once and merely re-blitted on every later
   * frame, not rebuilt. */
  image?: FakeElement;
}

/** Armed by each consumer's own `mount()` helper (via `armDrawFailure`) to make the first drawing
 * call throw. It has to be module state rather than a field on the element: the canvas is created
 * inside `boot()`, which runs a microtask after the widget function returns, so no test holds a
 * reference to it in time. Reached only through `armDrawFailure` -- not an exported `let` -- so
 * arming is a deliberate call an importer makes, not a binding any importer can quietly overwrite. */
let NEXT_DRAW_FAILURE: string | null = null;

/** Arms the next drawing call to throw `message`, or clears the arm with `null`. */
export function armDrawFailure(message: string | null): void {
  NEXT_DRAW_FAILURE = message;
}

export class RecordingContext {
  /** Only the three PAINT operations land here -- `clearRect`, `stroke`, `fill` -- not the
   * path-construction calls (`beginPath`/`moveTo`/`lineTo`/`arc`/`closePath`) that build up what
   * they paint; those only mutate `path` below. So `deepEqual(calls, [])` proves no paint happened,
   * not that nothing was called at all -- a hypothetical widget that built a path and never painted
   * it would pass that check too. Not reachable through `draw()` (render/canvas.ts), which always
   * opens with `clearRect` before it builds anything, but it is a real gap in what an empty `calls`
   * array can prove, not a closed one. */
  readonly calls: Call[] = [];
  failWith: string | null = NEXT_DRAW_FAILURE;
  fillStyle = "";
  strokeStyle = "";
  lineWidth = 0;
  lineCap = "butt";      // the real canvas default, so "never set" is not a distinct value
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
  /** No `Path2D` overload: nothing in this project ever passes one (checked -- `region.ts`,
   * `city.ts` and `canvas.ts` all call `fill()` bare or `fill("evenodd")`, always against the
   * context's OWN current path built via `beginPath`/`moveTo`/`lineTo`/`closePath`), so accepting
   * one here would be surface area nothing in this codebase exercises. */
  fill(fillRule?: CanvasFillRule): void { this.record("fill", fillRule); }
  /** Minimal on purpose: only the 3-argument, unscaled form (`drawImage(image, dx, dy)`) is
   * modelled -- the only one `city.ts` calls, blitting an offscreen canvas onto the visible one at
   * its own natural size. `dx`/`dy` are accepted (so the real call shape type-checks against this
   * fake the same way it does against a real `CanvasRenderingContext2D`) but not recorded: nothing
   * here needs to assert WHERE a blit landed, only that it happened, once per frame, and which
   * element it copied from (`Call.image`). */
  drawImage(image: FakeElement, _dx: number, _dy: number): void { this.record("drawImage", undefined, image); }

  private record(op: Call["op"], fillRule?: CanvasFillRule, image?: FakeElement): void {
    if (this.failWith !== null) throw new Error(this.failWith);
    this.calls.push({
      op,
      strokeStyle: this.strokeStyle,
      fillStyle: this.fillStyle,
      lineWidth: this.lineWidth,
      lineCap: this.lineCap,
      globalAlpha: this.globalAlpha,
      path: [...this.path],
      fillRule: fillRule ?? "nonzero",
      image,
    });
    this.firstDrawAt ??= ++clock;
  }
}

export class FakeElement {
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
  readonly attrs = new Map<string, string>();
  setAttribute(name: string, value: string): void { this.attrs.set(name, value); }
  getAttribute(name: string): string | null { return this.attrs.get(name) ?? null; }
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

/** The observer the widget lays itself out from. It delivers NOTHING on `observe()`: there is no
 * layout engine here, so a test plays layout with `fireResize`, and every consumer does so from its
 * own body -- after the mount's `fetch().then(boot).catch(...)` chain has settled. That is where a
 * real ResizeObserver delivers from too (the browser's own dispatch, never from inside `observe()`),
 * and it is what makes `runOrReport` load-bearing rather than redundant: a throw in the callback has
 * no promise `.catch` above it, which is exactly what each consumer's own drawing-failure test
 * exercises.
 *
 * perm-graph-boot.test.ts's pre-extraction version instead auto-fired an initial observation from
 * inside `observe()`, deferred one microtask out with `queueMicrotask` so that delivery stayed
 * outside the fetch chain the same way a real browser's dispatch does. That mechanism is not kept
 * here: field-boot's design above achieves the identical "outside the chain" property more directly
 * -- `fireResize` is always called from a test body that only runs after `mount()`'s own await has
 * resolved, i.e. after the widget's fetch-then-boot chain has fully settled, so there is no promise
 * frame left for a throw to be absorbed by, whether or not any deferral happens in between.
 *
 * That was checked, not assumed: reverting this `observe()` to a direct, synchronous `fire()` call
 * does not by itself redden anything against either widget, with `runOrReport` left in place --
 * `runOrReport` catches it either way, sync or deferred. Isolating the OTHER half on the original,
 * pre-extraction perm-graph-boot.test.ts (which still had the microtask) confirms which half was
 * doing the work: a synchronous `observe()` combined with `runOrReport` bypassed in perm-graph.ts
 * lets "a throw on the FIRST draw is reported" pass when it should fail, while the same bypass with
 * the microtask restored correctly reddens it. So the microtask's job was to keep an in-band failure
 * from being absorbed by the fetch chain's own `.catch`, not to be "async" for its own sake --
 * and the design here gets the same guarantee for free by never delivering in-band at all. The
 * property that actually matters, "a bypassed `runOrReport` reddens the throw tests", is confirmed
 * directly against both widgets' current sources under this design, not inferred from the above. */
export class FakeResizeObserver {
  static live: FakeResizeObserver[] = [];
  constructor(private readonly cb:
              (entries: { contentRect: { width: number; height: number } }[]) => void) {
    FakeResizeObserver.live.push(this);
  }
  observe(): void {}
  disconnect(): void {}
  fire(width: number, height: number): void { this.cb([{ contentRect: { width, height } }]); }
}

/** `render/canvas.ts` reads `devicePixelRatio` off `window`, and 2 rather than 1 is what makes the
 * backing-store assertions able to fail: at 1 the scaled size and the CSS size coincide. Exported so
 * each consumer's own backing-store assertions (`cv.width`, `cv.height`, ...) read it from one place
 * instead of each keeping its own `const DPR = 2` tied to `installStubs()` only by a comment. */
export const DPR = 2;

/** `window.requestAnimationFrame` stub: QUEUES `cb` (a real browser's rAF queues every call, it
 * does not collapse repeated calls into one -- coalescing is a property a WIDGET implements on top
 * of rAF, by guarding its own call to `requestAnimationFrame` behind an "already scheduled" flag,
 * not a property rAF gives away for free) and returns an id `cancelAnimationFrame` can remove by.
 * Nothing here fires on its own, the way a real one fires on the next display refresh -- there is
 * no display -- so a test drives it explicitly with `fireAnimationFrame`, the same shape as
 * `fireResize` playing layout for `ResizeObserver`. */
let pendingFrames: [id: number, cb: FrameRequestCallback][] = [];
let nextFrameId = 0;

/** Flushes every callback currently queued, in the order they were requested -- a widget that
 * scheduled two rAF callbacks before either fired (the coalescing bug this exists to catch) runs
 * BOTH here, which is the point: coalescing is proved by a test asserting the WIDGET only ever
 * queues one, not by this flush silently dropping extras for it. */
export function fireAnimationFrame(time = 0): void {
  const frames = pendingFrames;
  pendingFrames = [];
  for (const [, cb] of frames) cb(time);
}

/** Installs the fake globals every canvas widget boot test needs -- `document.createElement`,
 * `window.devicePixelRatio`/`requestAnimationFrame`/`cancelAnimationFrame` and `ResizeObserver` --
 * in place of the three top-level assignments field-boot.test.ts and perm-graph-boot.test.ts each
 * repeated. One call, before anything that might construct a widget; see this module's own
 * docstring for why the exact position relative to a consumer's own static imports does not
 * matter. Also resets the rAF queue, so a test file that (unusually) leaves a frame unflushed
 * cannot leak it into a later file sharing this module in the same process. */
export function installStubs(): void {
  (globalThis as Record<string, unknown>).document = {
    createElement: (tag: string): FakeElement => new FakeElement(tag),
  };
  (globalThis as Record<string, unknown>).window = { devicePixelRatio: DPR };
  (globalThis as Record<string, unknown>).ResizeObserver = FakeResizeObserver;
  (globalThis as Record<string, unknown>).requestAnimationFrame = (cb: FrameRequestCallback): number => {
    const id = ++nextFrameId;
    pendingFrames.push([id, cb]);
    return id;
  };
  (globalThis as Record<string, unknown>).cancelAnimationFrame = (id: number): void => {
    pendingFrames = pendingFrames.filter(([pid]) => pid !== id);
  };
  pendingFrames = [];
}

/** Plays layout: fires the most recently constructed `ResizeObserver` at `width`x`height`, the way
 * a real one would if there were a layout engine here to trigger it. */
export function fireResize(width: number, height: number): void {
  const obs = FakeResizeObserver.live.at(-1);
  assert.ok(obs !== undefined, "the widget never observed anything");
  obs.fire(width, height);
}

/** The mount point every widget generator emits: a `<figure>` carrying the fallback `<img>` inside
 * the `<a class="glightbox">` mkdocs-glightbox wraps it in, and a `<figcaption>` after it. A fixture
 * with a bare `<img>` would let a widget that removes only the image pass, while the live page kept
 * an empty, focusable, screen-reader-announced link where the picture was. Each consumer adds its
 * own `data-*` attributes (the bundle path, and whatever else its own generator emits) after calling
 * this -- they differ per widget, so this only builds the shape they share. */
export function mountPoint(): FakeElement {
  const figure = new FakeElement("figure");
  const anchor = new FakeElement("a");
  anchor.append(new FakeElement("img"));
  figure.append(anchor, new FakeElement("figcaption"));
  return figure;
}

/** DIRECT CHILDREN only -- never `host.find("canvas")`, a descendant search. Both widgets currently
 * insert the canvas straight into the mount point, so a descendant search would find it either way,
 * which is exactly the trap: it would keep finding it if a widget's insertion code ever changed to
 * wrap the canvas in another element, silently losing this test's ability to fail on exactly that
 * change. perm-graph-boot.test.ts's pre-extraction `canvasOf` already searched `host.children` only;
 * field-boot's searched descendants. Verified by counterfactual (see the task report): wrapping the
 * canvas in a `<div>` before `host.insertBefore` in either widget's source reddens the relevant boot
 * tests against this direct-children search, and reddens none against a descendant search. */
export function canvasOf(host: FakeElement): FakeElement {
  const cv = host.children.find((c) => c.tagName === "CANVAS");
  assert.ok(cv !== undefined, "no canvas was inserted into the figure");
  return cv;
}

/** The calls of the LAST frame only. Every control writes state, and state re-renders, so a test
 * that looked at the whole log would be reading the boot frame and the current one at once. */
export function lastFrame(cv: FakeElement): Call[] {
  const starts = cv.ctx.calls.flatMap((c, i) => c.op === "clearRect" ? [i] : []);
  const from = starts.at(-1);
  assert.ok(from !== undefined, "nothing was ever drawn: no clearRect in the call log");
  return cv.ctx.calls.slice(from);
}
