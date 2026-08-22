import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { AuthoringBlock } from "../src/authoring.js";
import type { PyResult, PyRuntime } from "../src/py/runtime.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import { fitBbox, toWorld, type Bbox } from "../src/view/transform.js";
import {
  canvasOf, fakeLocation, FakeElement, fireAnimationFrame, fireResize, installStubs, mountPoint,
  writeNow,
} from "./harness.js";
import { drawRoadWidget, DRAW_ROAD_URL, type DrawRoadState } from "../src/widgets/draw-road.js";

installStubs();

/** The real committed bundle, not a fixture: it is what the browser fetches, and its arities --
 * 263 parcels, 745 edges -- are what the widget indexes with when it turns a solve into a
 * picture. A hand-rolled stand-in would let an off-by-one in that indexing pass. */
const block = JSON.parse(
  readFileSync("../examples/authoring/block.json", "utf8")) as AuthoringBlock;

// `fetch` is not part of the shared harness (harness.ts's own docstring says so) -- one static
// stub suffices here: every test in this file mounts the same one bundle.
(globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
  ok: true, status: 200, statusText: "OK",
  json: (): Promise<unknown> => Promise.resolve(block),
});

/** A `PyRuntime` that boots instantly, downloads nothing, and remembers what it was asked.
 *
 * This is the whole reason `pyodideRuntime` is injected rather than constructed inside the widget
 * (py/runtime.ts's own docstring): the widget cannot tell the difference, so these tests drive the
 * entire interaction with no network and no 25-35 MB download.
 *
 * `boot()` deliberately does NOT memoise. The real one does (py/runtime.ts), but a fake that also
 * did would count one boot however many times the widget called it, and "the widget boots once"
 * would be this fake's property rather than the widget's. `boots` counts raw calls.
 *
 * `solve` records the road it was handed BEFORE deciding whether to fail, so `roads` says what
 * reached the runtime even on the failure path. Its `potential`/`conductance` are the bundle's own
 * baseline arrays -- the right lengths for the mesh the widget draws, which is what the picture
 * indexes into. */
class FakeRuntime implements PyRuntime {
  boots = 0;
  readonly roads: [number, number][][] = [];
  constructor(private readonly solveThrows: string | null) {}
  boot(): Promise<void> {
    this.boots += 1;
    return Promise.resolve();
  }
  solve(road: [number, number][]): Promise<PyResult> {
    this.roads.push(road);
    if (this.solveThrows !== null) return Promise.reject(new Error(this.solveThrows));
    return Promise.resolve({
      permeability: 0.5,
      roadMetres: 123.4,
      potential: block.baseline.potential,
      conductance: block.edges.footpath_g,
    });
  }
}

/** The view the widget fits, derived here independently of it: the bbox of the block's exterior
 * ring at the mounted size, through the same `fitBbox` every map view on this site uses. Deriving
 * it rather than importing the widget's own is what makes "the click at (10, 10) became THESE
 * metres" an assertion about the transform and not a restatement of it. */
function ringBbox(ring: [number, number][]): Bbox {
  const xs = ring.map(([x]) => x);
  const ys = ring.map(([, y]) => y);
  return { minX: Math.min(...xs), minY: Math.min(...ys),
           maxX: Math.max(...xs), maxY: Math.max(...ys) };
}
const EXTERIOR = block.boundary[0];
assert.ok(EXTERIOR !== undefined, "the committed bundle carries no block boundary");
const MOUNT_PX = 700;
const VIEW = fitBbox(ringBbox(EXTERIOR), MOUNT_PX, MOUNT_PX);

/** The metres a click at these canvas offsets lands on. */
function world(sx: number, sy: number): [number, number] {
  return toWorld(VIEW, sx, sy);
}

/** Mounts the widget over a fake runtime and waits for its fetch chain to settle BEFORE playing
 * the first resize -- screen-map-boot.test.ts's own `mount` shape, positional arguments and all,
 * with `solveThrows` where that one has `drawFailure`.
 *
 * The awaited macrotask matters for the same reason it does there: `fetch(...).then(boot)` means
 * nothing exists to resize until the microtask queue has drained, and `fireResize` before that
 * finds no observer to fire.
 *
 * The first frame is drawn synchronously from inside the resize callback, so this needs no
 * `fireAnimationFrame` of its own; everything a reader's gesture triggers is rAF-coalesced and is
 * flushed by `flush()` below.
 *
 * The state store is the PRODUCTION one (`urlStore` over a `fakeLocation`), never `localState`, so
 * the URL test below exercises the real decode path rather than a second, test-only one. */
async function mount(width = MOUNT_PX, solveThrows: string | null = null, search = ""):
    Promise<{ host: FakeElement; cv: FakeElement; store: StateSource<DrawRoadState>;
              loc: ReturnType<typeof fakeLocation>; runtime: FakeRuntime }> {
  const host = mountPoint();
  host.dataset.bundle = "../examples/authoring/block.json";
  host.dataset.wheel = "../dist/reblock-0.1.0-py3-none-any.whl";
  const runtime = new FakeRuntime(solveThrows);
  const loc = fakeLocation(search);
  const slot = urlStore(loc, writeNow).reserve();
  let bound: StateSource<DrawRoadState> | null = null;
  drawRoadWidget(() => runtime)(host as never, (initial) => {
    bound = slot.bind(DRAW_ROAD_URL, initial);
    return bound;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  fireResize(width, width);
  // Every mount here feeds `boot` the committed bundle, and `boot` reaches `makeState` before it
  // draws anything. A null is a widget that failed before it.
  assert.ok(bound !== null, "the widget never asked for a state store");
  return { host, cv: canvasOf(host), store: bound, loc, runtime };
}

/** Drains the promise queue and then runs the one animation frame an interaction schedules.
 * `state.set` is synchronous, but the redraw is deferred behind `frameScheduled`, and a solve is a
 * promise -- so a test that inspected the readout immediately after a gesture would be reading the
 * moment before the answer arrived. */
async function flush(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  fireAnimationFrame();
}

/** The boot control, found by tag -- this widget writes exactly one button. */
function bootControl(host: FakeElement): FakeElement {
  const el = host.find("button");
  assert.ok(el !== null, "there is no boot control");
  return el;
}

/** The `aria-live="polite"` element's own text -- found by that attribute, not merely by tag, so
 * this fails loudly if the readout is ever rendered without it rather than silently reading some
 * OTHER paragraph (screen-map-boot.test.ts's own `readoutText`). */
function readoutText(host: FakeElement): string {
  const live = host.descendants().find((el) => el.getAttribute("aria-live") === "polite");
  assert.ok(live !== undefined, `there is no aria-live="polite" readout`);
  return live.textContent;
}

async function pressBoot(host: FakeElement): Promise<void> {
  bootControl(host).dispatch("click");
  await flush();
}

/** Draws a polyline the way a reader's pointer does, at CANVAS OFFSETS -- the widget turns them
 * into metres itself.
 *
 * Named `drawPolyline`, not `drawRoad`: `drawRoad` is the widget this file imports.
 *
 * The last point is pressed TWICE before `dblclick`, because that is what a browser delivers: a
 * double-click is two complete clicks, so its second `pointerdown` lands on the spot the first one
 * did, and the road the widget settles ends with the same vertex twice unless something drops it
 * (`withoutRepeats`). Driving the real sequence is what puts that on the tested path instead of
 * leaving it to a comment. */
function drawPolyline(host: FakeElement, points: [number, number][]): void {
  const cv = canvasOf(host);
  for (const [x, y] of points) cv.dispatch("pointerdown", { offsetX: x, offsetY: y });
  const last = points.at(-1);
  assert.ok(last !== undefined, "a road has to start somewhere");
  cv.dispatch("pointerdown", { offsetX: last[0], offsetY: last[1] });
  cv.dispatch("dblclick");
}

test("nothing is solved until the reader boots the runtime", async () => {
  const { runtime } = await mount();
  assert.equal(runtime.boots, 0, "a prose page must not pay 25-35 MB for being read");
});

test("clicking the boot control boots exactly once, however many times it is pressed", async () => {
  const { host, runtime } = await mount();
  bootControl(host).dispatch("click");
  bootControl(host).dispatch("click");
  await flush();
  assert.equal(runtime.boots, 1);
});

test("a two-point road is solved and its permeability reported", async () => {
  const { host, runtime, loc } = await mount();
  await pressBoot(host);
  drawPolyline(host, [[10, 10], [60, 60]]);
  await flush();
  assert.deepEqual(runtime.roads.at(-1), [world(10, 10), world(60, 60)]);
  assert.match(readoutText(host), /permeability/i);
  // The other half of the round trip test 6 reads: a settled road reaches the query string.
  assert.match(loc.written.at(-1) ?? "", /^road=[-\d.,]+$/);
});

test("a one-point road is refused before it reaches the runtime", async () => {
  const { host, runtime } = await mount();
  await pressBoot(host);
  drawPolyline(host, [[10, 10]]);
  await flush();
  assert.equal(runtime.roads.length, 0, "the widget must not hand a degenerate road to Python");
  assert.match(readoutText(host), /two points/i);
});

test("a runtime failure keeps the last good picture and says what failed", async () => {
  const { host } = await mount(MOUNT_PX, "boom");
  await pressBoot(host);
  drawPolyline(host, [[10, 10], [60, 60]]);
  await flush();
  assert.match(readoutText(host), /boom/);
  assert.ok(canvasOf(host), "the canvas is still there; a failed solve is not a blank figure");
});

test("the drawn road survives a URL round trip", async () => {
  const { store, loc } = await mount(MOUNT_PX, null, "road=10,10,60,60");
  assert.deepEqual(store.get().road, [[10, 10], [60, 60]]);
  // A URL the codec accepts verbatim is one the widget has nothing to correct, so it must not
  // rewrite the address bar behind the reader.
  assert.deepEqual(loc.written, [], "a decodable ?road= was rewritten anyway");
});
