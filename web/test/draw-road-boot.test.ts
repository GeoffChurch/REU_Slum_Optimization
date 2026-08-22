import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { AuthoringBlock } from "../src/authoring.js";
import type { PyResult, PyRuntime } from "../src/py/runtime.js";
import { GRAPH_ENCODING } from "../src/render/graph-encoding.generated.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import { fitBbox, toScreen, toWorld, type Bbox } from "../src/view/transform.js";
import {
  Call, canvasOf, fakeLocation, FakeElement, fireAnimationFrame, fireResize, installStubs,
  lastFrame, mountPoint, writeNow,
} from "./harness.js";
import { drawRoadWidget, DRAW_ROAD_URL, type DrawRoadState } from "../src/widgets/draw-road.js";

installStubs();

/** The real committed bundle, not a fixture: it is what the browser fetches, and its arities --
 * 263 parcels, 745 edges -- are what the widget indexes with when it turns a solve into a
 * picture. A hand-rolled stand-in would let an off-by-one in that indexing pass. */
const block = JSON.parse(
  readFileSync("../examples/authoring/block.json", "utf8")) as AuthoringBlock;

/** Every tenth edge is "road-raised" by the fake below. A constant rather than a magic 10, because
 * both the fake and the assertion that counts blue strokes have to mean the same edges. */
const RAISE_EVERY = 10;
const RAISED = block.edges.footpath_g.filter((_, k) => k % RAISE_EVERY === 0).length;

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
 * reached the runtime even on the failure path. What it returns is built to be TELLABLE APART:
 *
 * * `potential` is the baked baseline scaled by `1 / n` for the n-th solve, so no two solves --
 *   and no solve and the baseline -- ever paint the same node colours under a fixed `vmax`, while
 *   staying a uniform rescaling, which is what makes the `vmax` test able to fail (see it);
 * * `conductance` raises every `RAISE_EVERY`-th edge above `footpath_g`, so `first_upgraded_at`
 *   has a known number of raised edges for the picture tests to count.
 *
 * `truncatePotential` is a field rather than a constructor argument so a test can arm it after
 * mounting, without a second mount helper or a wider `mount` signature. */
class FakeRuntime implements PyRuntime {
  boots = 0;
  truncatePotential = false;
  readonly roads: [number, number][][] = [];
  constructor(private readonly solveThrows: string | null) {}
  boot(): Promise<void> {
    this.boots += 1;
    return Promise.resolve();
  }
  solve(road: [number, number][]): Promise<PyResult> {
    this.roads.push(road);
    if (this.solveThrows !== null) return Promise.reject(new Error(this.solveThrows));
    const scale = 1 / (this.roads.length + 1);
    const potential = block.baseline.potential.map((v) => v * scale);
    return Promise.resolve({
      permeability: 0.5,
      roadMetres: 123.4,
      potential: this.truncatePotential ? potential.slice(0, 3) : potential,
      conductance: block.edges.footpath_g.map((g, k) => (k % RAISE_EVERY === 0 ? g * 100 : g)),
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
 * with `solveThrows` where that one has `drawFailure`, plus field-boot.test.ts's `payload`
 * argument (the fetch stub lives in here, so each mount serves its own bundle).
 *
 * The awaited macrotask matters for the same reason it does there: `fetch(...).then(boot)` means
 * nothing exists to resize until the microtask queue has drained, and `fireResize` before that
 * finds no observer to fire.
 *
 * `store` is NULLABLE and nothing is asserted about it here, exactly as in field-boot.test.ts: a
 * bundle `boot` REFUSES -- the wrong block, a boundary with holes -- never reaches `makeState` and
 * never observes anything, so asserting non-null in this helper would turn "the widget correctly
 * refused a broken bundle" into a helper failure. For the same reason the resize is played only
 * when a canvas actually exists.
 *
 * The state store is the PRODUCTION one (`urlStore` over a `fakeLocation`), never `localState`, so
 * the URL test below exercises the real decode path rather than a second, test-only one. */
async function mount(width = MOUNT_PX, solveThrows: string | null = null, search = "",
                     payload: unknown = block):
    Promise<{ host: FakeElement; store: StateSource<DrawRoadState> | null;
              loc: ReturnType<typeof fakeLocation>; runtime: FakeRuntime }> {
  const host = mountPoint();
  host.dataset.bundle = "../examples/authoring/block.json";
  host.dataset.wheel = "../dist/reblock-0.1.0-py3-none-any.whl";
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true, status: 200, statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(payload),
  });
  const runtime = new FakeRuntime(solveThrows);
  const loc = fakeLocation(search);
  const slot = urlStore(loc, writeNow).reserve();
  let bound: StateSource<DrawRoadState> | null = null;
  drawRoadWidget(() => runtime)(host as never, (initial) => {
    bound = slot.bind(DRAW_ROAD_URL, initial);
    return bound;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  if (host.find("canvas") !== null) fireResize(width, width);
  return { host, store: bound, loc, runtime };
}

/** The store a successful mount must have taken. Separate from `mount` so the refusal tests can
 * use the same helper (see its docstring). */
function storeOf(m: { store: StateSource<DrawRoadState> | null }): StateSource<DrawRoadState> {
  assert.ok(m.store !== null, "the widget never asked for a state store");
  return m.store;
}

/** Runs whatever is queued until nothing new is: frames, then promises, three rounds.
 *
 * One round is not enough any more. A gesture schedules a FRAME; that frame renders and asks the
 * runtime for a solve; the solve resolves on the microtask queue; its `finally` schedules another
 * frame, which is the one that actually paints the answer. So the readout and the canvas reach
 * their settled values two frames and two drains after the gesture, and a test that looked after
 * one would be reading the moment before the answer arrived. `fireAnimationFrame` with nothing
 * queued is a no-op, so the extra round costs nothing when the chain is shorter. */
async function flush(): Promise<void> {
  for (let i = 0; i < 3; i++) {
    fireAnimationFrame();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
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
 * With `settleWith: "dblclick"` the last point is pressed TWICE before the `dblclick`, because
 * that is what a browser delivers: a double-click is two complete clicks, so its second
 * `pointerdown` lands on the spot the first one did, and the road the widget settles ends with the
 * same vertex twice unless something drops it (`withoutRepeats`). Driving the real sequence is
 * what puts that on the tested path instead of leaving it to a comment. `"Enter"` settles from the
 * keyboard instead and places each point exactly once. */
function drawPolyline(host: FakeElement, points: [number, number][],
                      settleWith: "dblclick" | "Enter" = "dblclick"): void {
  const cv = canvasOf(host);
  for (const [x, y] of points) cv.dispatch("pointerdown", { offsetX: x, offsetY: y });
  if (settleWith === "Enter") {
    cv.dispatch("keydown", { key: "Enter" });
    return;
  }
  const last = points.at(-1);
  assert.ok(last !== undefined, "a road has to start somewhere");
  cv.dispatch("pointerdown", { offsetX: last[0], offsetY: last[1] });
  cv.dispatch("dblclick");
}

/** Every node's fill colour in the last frame, in node order.
 *
 * `draw` fills exactly one arc per parcel and nothing else fills anything, so this is the potential
 * field as the reader sees it -- the one quantity in the picture that the Pyodide result decides.
 * (The road overlay fills a dot per PENDING vertex, and nothing is pending in any of these tests;
 * the assertions below check the count against the mesh, so a stray fill would show up.) */
function nodeFills(cv: FakeElement): string[] {
  return lastFrame(cv).filter((c) => c.op === "fill").map((c) => c.fillStyle);
}

/** The strokes `draw` makes for road-RAISED edges: the encoding's road colour at its fixed
 * upgraded width, which is the one combination nothing else in the frame uses (mesh edges are grey
 * and variable-width; the reader's own road overlay is the road colour at 2 px). */
function upgradedStrokes(cv: FakeElement): Call[] {
  return lastFrame(cv).filter((c) => c.op === "stroke"
    && c.strokeStyle === GRAPH_ENCODING.road_color
    && c.lineWidth === GRAPH_ENCODING.upgraded_lw);
}

test("nothing is solved until the reader boots the runtime", async () => {
  const { runtime } = await mount();
  assert.equal(runtime.boots, 0, "a prose page must not pay 25-35 MB for being read");
});

test("the boot control says what pressing it downloads", async () => {
  // This widget lands on a prose page a reader may reach with no intention of computing anything
  // (design §3), so the control has to read as an opt-in that states its price, not as decoration.
  const { host } = await mount();
  assert.match(bootControl(host).textContent, /\bMB\b/,
    "the boot control does not say what it will download");
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
  // The other half of the round trip the URL test reads: a settled road reaches the query string.
  assert.match(loc.written.at(-1) ?? "", /^road=[-\d.,]+$/);
});

test("Enter settles a road from the keyboard, and Escape clears it back to the baseline",
  async () => {
    // The canvas is given `tabIndex = 0` FOR these two keys; without a test, renaming either to
    // nonsense is silent, and `clear()` is reachable from nowhere else.
    const m = await mount();
    const cv = canvasOf(m.host);
    await pressBoot(m.host);
    const baseline = nodeFills(cv);

    drawPolyline(m.host, [[20, 20], [90, 40]], "Enter");
    await flush();
    assert.deepEqual(m.runtime.roads.at(-1), [world(20, 20), world(90, 40)],
      "Enter did not settle the road");
    assert.notDeepEqual(nodeFills(cv), baseline, "the solve did not change the picture");

    cv.dispatch("keydown", { key: "Escape" });
    await flush();
    assert.deepEqual(storeOf(m).get().road, [], "Escape left the road in state");
    assert.deepEqual(nodeFills(cv), baseline,
      "Escape left the solved field on the canvas instead of returning to the baseline");
    assert.match(readoutText(m.host), /Click points on the block/,
      "Escape left the previous answer in the readout");
  });

test("a one-point road is refused before it reaches the runtime", async () => {
  const { host, runtime } = await mount();
  await pressBoot(host);
  drawPolyline(host, [[10, 10]]);
  await flush();
  assert.equal(runtime.roads.length, 0, "the widget must not hand a degenerate road to Python");
  assert.match(readoutText(host), /two points/i);
});

test("a road drawn before the runtime is loaded is kept, not solved", async () => {
  // Drawing before booting is a supported order -- the boot handler solves whatever is in hand --
  // so the gate that stops a pre-boot settle reaching `solve` is on the ordinary path. Without it
  // the real runtime rejects with "called before boot() resolved" (py-runtime.test.ts) and the
  // reader is shown that instead of being told to press the button.
  const m = await mount();
  drawPolyline(m.host, [[10, 10], [60, 60]]);
  await flush();
  assert.equal(m.runtime.roads.length, 0, "a road was solved before the runtime was loaded");
  assert.match(readoutText(m.host), /Load the Python runtime/);
  // And it is not lost: pressing the button answers the road already drawn.
  await pressBoot(m.host);
  assert.deepEqual(m.runtime.roads.at(-1), [world(10, 10), world(60, 60)],
    "the road drawn before booting was dropped instead of solved on boot");
});

test("dragging a vertex moves the road and re-solves it", async () => {
  // Design §4: "Dragging an existing vertex moves it", and §1.3 measured a solve at 0.06 s
  // expressly so the number could move WITH the road rather than waiting for a release.
  const { host, runtime } = await mount();
  const cv = canvasOf(host);
  await pressBoot(host);
  drawPolyline(host, [[100, 100], [200, 200]]);
  await flush();
  const before = runtime.roads.length;

  // Press ON the second vertex -- found by projecting the road the widget actually holds back to
  // the screen, not by reusing the offsets it was drawn at, so this stays a hit-test rather than a
  // coincidence.
  const road = runtime.roads.at(-1);
  assert.ok(road !== undefined, "nothing was solved to drag");
  const grabbed = road[1];
  assert.ok(grabbed !== undefined, "the road has no second vertex");
  const [gx, gy] = toScreen(VIEW, grabbed[0], grabbed[1]);
  cv.dispatch("pointerdown", { offsetX: gx, offsetY: gy, pointerId: 1 });
  cv.dispatch("pointermove", { offsetX: 300, offsetY: 120, pointerId: 1 });
  await flush();
  cv.dispatch("pointerup", { offsetX: 300, offsetY: 120, pointerId: 1 });
  await flush();

  assert.ok(runtime.roads.length > before, "the drag never re-solved");
  assert.deepEqual(runtime.roads.at(-1), [world(100, 100), world(300, 120)],
    "the drag placed a new vertex instead of moving the one it grabbed");
});

test("the picture drawn is the SOLVE, at a colour scale fixed by the baseline", async () => {
  // Three properties in one mount, because each needs the previous one's frame to compare against.
  const { host } = await mount();
  const cv = canvasOf(host);
  await pressBoot(host);

  // (1) Before any solve, the frame is the baked baseline: one node fill per parcel, and no edge
  // raised by a road, because there is no road.
  const baseline = nodeFills(cv);
  assert.equal(baseline.length, block.nodes.cx.length, "a node went missing from the baseline");
  assert.equal(upgradedStrokes(cv).length, 0, "an edge was drawn as road-raised with no road");

  // (2) After a solve, the picture is the RESULT: the node colours move, and exactly the edges the
  // runtime raised above `footpath_g` are drawn in the road colour at the fixed upgraded width.
  // A picture still drawn at prefix 0 would show the baseline's colours and no raised edge at all.
  drawPolyline(host, [[10, 10], [60, 60]]);
  await flush();
  const first = nodeFills(cv);
  assert.equal(first.length, block.nodes.cx.length, "a node went missing from the solved frame");
  assert.notDeepEqual(first, baseline, "the solve is not what was drawn: the baseline is");
  assert.equal(upgradedStrokes(cv).length, RAISED,
    "the edges the runtime raised are not the edges drawn as raised");

  // (3) `vmax` comes off prefix 0 -- the BASELINE -- so it is one scale across every solve. The
  // fake's second answer is the same field scaled again, so under a per-solve `vmax` the two
  // frames would paint IDENTICAL colours (a uniform rescale divides out). They must not.
  drawPolyline(host, [[30, 30], [80, 80]]);
  await flush();
  const second = nodeFills(cv);
  assert.notDeepEqual(second, first,
    "two different solves painted the same colours: the ramp is being rescaled per solve, so no "
    + "two roads a reader draws can be compared by colour");
});

test("a bundle that is not the block the drawing scale was baked for is refused, loudly",
  async () => {
    // `GRAPH_ENCODING.width_norm` is a property of perm-graph's MESH, and it is the scale every
    // edge width here is drawn against. The two bundles are baked by two commands, so one can be
    // re-baked and committed without the other -- and a foreign mesh would draw at a silently
    // wrong scale rather than fail.
    const { host, store } = await mount(MOUNT_PX, null, "",
                                        { ...block, block_id: "ZAF.0.0.0_0_00000" });
    assert.equal(store, null, "a refused bundle asked for a state store anyway");
    assert.match(host.find("figcaption")!.textContent,
      /DrawRoad could not load interactively .*ZAF\.0\.0\.0_0_00000/,
      "the block mismatch was accepted, or reported without naming the block");
    assert.equal(host.find("canvas"), null, "a canvas was inserted for a refused bundle");
    assert.equal(host.find("img")!.removedAt, null, "the fallback image went anyway");
  });

test("a block boundary with more than one ring is refused, not drawn as its exterior", async () => {
  // `scripts/_bundle_io.py`'s `polygon_ring` raises on interiors rather than dropping them
  // ("report this instead of silently dropping geometry"), and `draw` strokes one ring, so a
  // block with holes must stop here rather than be drawn as a solid outline it does not have.
  // The pinned bundle has exactly one ring, so the payload is the one place this shape exists.
  const two = [EXTERIOR, EXTERIOR];
  const { host, store } = await mount(MOUNT_PX, null, "", { ...block, boundary: two });
  assert.equal(store, null, "a refused bundle asked for a state store anyway");
  assert.match(host.find("figcaption")!.textContent,
    /DrawRoad could not load interactively .*boundary has 2 rings/,
    "a multi-ring boundary was accepted, or reported without saying what was wrong");
  assert.equal(host.find("canvas"), null, "a canvas was inserted for a refused bundle");
});

test("a runtime whose arrays do not fit the mesh is reported, not drawn", async () => {
  // The silent half of the Pyodide boundary: a short `potential` makes every current NaN, which
  // makes `rampColor` return `undefined`, which a browser IGNORES when assigned to `fillStyle`.
  // Nothing raises; the reader gets a subtly wrong picture.
  const { host, runtime } = await mount();
  const cv = canvasOf(host);
  await pressBoot(host);
  const baseline = nodeFills(cv);
  runtime.truncatePotential = true;
  drawPolyline(host, [[10, 10], [60, 60]]);
  await flush();
  assert.match(readoutText(host), /3 potentials .*263 parcels/,
    "a mis-sized result was accepted");
  assert.deepEqual(nodeFills(cv), baseline,
    "a mis-sized result was drawn instead of leaving the last good picture up");
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
  const m = await mount(MOUNT_PX, null, "road=10,10,60,60");
  assert.deepEqual(storeOf(m).get().road, [[10, 10], [60, 60]]);
  // A URL the codec accepts verbatim is one the widget has nothing to correct, so it must not
  // rewrite the address bar behind the reader.
  assert.deepEqual(m.loc.written, [], "a decodable ?road= was rewritten anyway");
});
