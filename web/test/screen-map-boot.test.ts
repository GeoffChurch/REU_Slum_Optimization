import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { ranking, scores, selectAt, type MetricName } from "../src/model/screen.js";
import { localState } from "../src/state.js";
import { fitBbox, toScreen, type Bbox, type View } from "../src/view/transform.js";
import { canvasOf, Call, fireResize, installStubs, mountPoint } from "./harness.js";
import { screenMap } from "../src/widgets/screen-map.js";

installStubs();

const ct = JSON.parse(
  readFileSync("../examples/screen-map/capetown.json", "utf8")) as CityBundle;
const nb = JSON.parse(
  readFileSync("../examples/screen-map/nairobi.json", "utf8")) as CityBundle;
// `gen_screen_map.py`'s `ENCODING_DICT` is the SAME object baked into both bundles, so reading it
// off either is equivalent -- but reading it off `ct` keeps every Cape Town assertion below
// self-contained against the one bundle it is actually about.
const E = ct.encoding;

// `fetch` is not part of the shared harness (harness.ts's own docstring says so) -- one static
// stub suffices here: every test in this file mounts the same two committed bundles, keyed by
// which of the widget's two `data-bundle-*` URLs the request names.
(globalThis as Record<string, unknown>).fetch = (url: string): Promise<unknown> => {
  const body = url.includes("nairobi") ? nb : ct;
  return Promise.resolve({
    ok: true, status: 200, statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(body),
  });
};

/** Mounts the widget and waits for its two-bundle fetch chain to settle BEFORE playing the first
 * resize -- region-grow-boot.test.ts's own `mount` shape, a `Promise.all`'d fetch aside.
 * `screenMap(...)` only starts a promise chain; it does not run `boot()` -- and so does not
 * construct a canvas or a ResizeObserver -- until that chain's `.then`s have had a turn, which
 * happens no sooner than the next macrotask REGARDLESS of how many `.then` hops are chained (the
 * microtask queue always fully drains before a `setTimeout` callback runs, whether it is one hop
 * or five). Calling `fireResize` before that drain finds no observer to fire. */
async function mount(width = 700): Promise<{ host: ReturnType<typeof mountPoint>; cv: unknown }> {
  const host = mountPoint();
  host.dataset.bundleCapetown = "../examples/screen-map/capetown.json";
  host.dataset.bundleNairobi = "../examples/screen-map/nairobi.json";
  screenMap(host as never, localState);
  await new Promise((resolve) => setTimeout(resolve, 0));
  fireResize(width, width);
  return { host, cv: canvasOf(host) };
}

/** Every draw call recorded on `cv` so far, from index `since`. Deliberately NOT `lastFrame`
 * (harness.ts): that helper slices from the LAST `clearRect`, which is right for a widget that
 * clears and repaints its whole canvas every frame, but this one does not -- `render/city.ts`'s
 * `paintBase` clears and repaints only on the first frame and on a resize/city switch, and a
 * floor or metric change alone goes through `paintSelection` alone, with no new `clearRect`. Two
 * snapshots of `calls.length`, taken immediately before and after one specific interaction,
 * isolate exactly what that interaction added regardless of how many frames came before it. */
function callsSince(cv: unknown, since: number): Call[] {
  return (cv as { ctx: { calls: Call[] } }).ctx.calls.slice(since);
}
function callCount(cv: unknown): number {
  return (cv as { ctx: { calls: Call[] } }).ctx.calls.length;
}
function fillsOf(calls: Call[], style: string): number {
  return calls.filter((c) => c.op === "fill" && c.fillStyle === style).length;
}

function floorSlider(host: ReturnType<typeof mountPoint>): ReturnType<typeof mountPoint> {
  const el = host.findAll("input").find((i) => i.type === "range");
  assert.ok(el !== undefined, "there is no floor slider");
  return el;
}
function metricSelectEl(host: ReturnType<typeof mountPoint>): ReturnType<typeof mountPoint> {
  const el = host.find("select");
  assert.ok(el !== null, "there is no metric selector");
  return el;
}
function cityToggleEl(host: ReturnType<typeof mountPoint>): ReturnType<typeof mountPoint> {
  const el = host.findAll("input").find((i) => i.type === "checkbox");
  assert.ok(el !== undefined, "there is no city toggle");
  return el;
}
/** The `aria-live="polite"` element's own text -- found by that attribute, not merely by tag, so
 * this fails loudly if the readout is ever rendered without it rather than silently reading some
 * OTHER paragraph. */
function readoutText(host: ReturnType<typeof mountPoint>): string {
  const live = host.descendants().find((el) => el.getAttribute("aria-live") === "polite");
  assert.ok(live !== undefined, `there is no aria-live="polite" readout`);
  return live.textContent;
}

/** Drives each control the way a reader's pointer or keyboard would: set `.value`/`.checked`,
 * then dispatch the event the widget listens for -- field-boot.test.ts's and region-grow-boot.
 * test.ts's own `setSlider` shape. */
function setFloor(host: ReturnType<typeof mountPoint>, value: number): void {
  const el = floorSlider(host);
  el.value = String(value);
  el.dispatch("input");
}
function setMetric(host: ReturnType<typeof mountPoint>, metric: MetricName): void {
  const el = metricSelectEl(host);
  el.value = metric;
  el.dispatch("change");
}
function setCity(host: ReturnType<typeof mountPoint>, showNairobi: boolean): void {
  const el = cityToggleEl(host);
  el.checked = showNairobi;
  el.dispatch("change");
}

/** The bbox every shipped block's rings must fit inside -- derived independently here, not
 * imported from the widget, matching region-grow-boot.test.ts's own `hoodBbox` one level up in
 * scale (and its own reasoning: a widget that fit the wrong extent should be caught, not
 * confirmed by its own arithmetic). */
function cityBbox(bundle: CityBundle): Bbox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const rings of bundle.rings) for (const ring of rings) for (const [x, y] of ring) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}
const SIZE = 700;
const VIEW_CT = fitBbox(cityBbox(ct), SIZE, SIZE, E.pad);

/** The LAST fill this widget painted at the exact screen position of `block`'s own first ring
 * vertex -- identifying ONE block's current colour in a call log that may hold several frames'
 * worth of history, without needing a full-path comparison. `render/city.ts`'s `fillBlock` always
 * opens a block's path with `moveTo` at that vertex, so scanning backwards for a fill whose FIRST
 * path op lands there finds that block's most recent paint. */
function lastFillStyleForBlock(cv: unknown, view: View, bundle: CityBundle,
                               blockIndex: number): string | undefined {
  const [x, y] = bundle.rings[blockIndex]![0]![0]!;
  const [sx, sy] = toScreen(view, x, y);
  const calls = (cv as { ctx: { calls: Call[] } }).ctx.calls;
  for (let i = calls.length - 1; i >= 0; i--) {
    const c = calls[i]!;
    const first = c.path[0];
    if (c.op === "fill" && first !== undefined && first.op === "moveTo"
        && first.args[0] === sx && first.args[1] === sy) {
      return c.fillStyle;
    }
  }
  return undefined;
}

/** A block selected under `metricA` at ITS OWN shipped floor and not selected under `metricB` at
 * ITS OWN shipped floor -- found empirically against the committed Cape Town bundle (611 such
 * blocks exist for depth_density_proxy/density_compactness, verified by hand before this test was
 * written) rather than asserted, so the test can never drift from what the two rankings actually
 * produce. */
function findFlip(bundle: CityBundle, metricA: MetricName, metricB: MetricName): number {
  const floorA = bundle.floors.find((f) => f.metric === metricA)!.value;
  const floorB = bundle.floors.find((f) => f.metric === metricB)!.value;
  const orderA = ranking(bundle, metricA);
  const selA = selectAt(bundle, orderA, scores(bundle, metricA), floorA);
  const orderB = ranking(bundle, metricB);
  const selB = selectAt(bundle, orderB, scores(bundle, metricB), floorB);
  const inB = new Set(orderB.slice(0, selB.count));
  for (const i of orderA.slice(0, selA.count)) if (!inB.has(i)) return i;
  throw new Error(`no block is selected under ${metricA} and not under ${metricB} in the `
    + `committed bundle -- pick a different metric pair`);
}

test("the number of selected blocks drawn equals what the model selects", async () => {
  // Identified by the bundle's `selected_color`, never by a path count -- see the comment in
  // region-grow-boot.test.ts.
  const { cv } = await mount();
  const shipped = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  const order = ranking(ct, "depth_density_proxy");
  const s = scores(ct, "depth_density_proxy");
  const sel = selectAt(ct, order, s, shipped.value);
  const drawn = fillsOf(callsSince(cv, 0), E.selected_color);
  assert.equal(drawn, sel.count);
  // Cross-checked against the independently baked pool size too (examples/screen-bakeoff's own
  // route, read into the bundle at bake time) -- the same two-paths-agreeing guard Task 8's own
  // screen-model.test.ts relies on.
  assert.equal(drawn, shipped.n);
});

test("moving the floor changes the drawn selection", async () => {
  // A widget that computed the selection but drew a constant would pass a count test at one
  // floor. Assert at two floors and require the counts to differ.
  const { host, cv } = await mount();
  const countAtDefault = fillsOf(callsSince(cv, 0), E.selected_color);

  // The single highest-scoring block's own score: raising the floor to it selects (about) one
  // block, a difference that cannot be an artefact of two nearly-identical floors.
  const s = scores(ct, "depth_density_proxy");
  let max = -Infinity;
  for (const v of s) if (v > max) max = v;

  const before = callCount(cv);
  setFloor(host, max);
  const countAtMax = fillsOf(callsSince(cv, before), E.selected_color);

  assert.notEqual(countAtMax, countAtDefault,
    `floor change produced the same selected count (${countAtDefault})`);
  assert.equal(countAtMax, selectAt(ct, ranking(ct, "depth_density_proxy"), s, max).count);
});

test("switching metric re-ranks rather than re-filtering the old ranking", async () => {
  // Pick a block that is above the floor under one metric and below it under another, from the
  // committed bundle, and assert its membership flips.
  const { host, cv } = await mount();
  const blockIndex = findFlip(ct, "depth_density_proxy", "density_compactness");

  assert.equal(lastFillStyleForBlock(cv, VIEW_CT, ct, blockIndex), E.selected_color,
    "the chosen block should start out selected under the default metric, depth_density_proxy");

  setMetric(host, "density_compactness");

  assert.equal(lastFillStyleForBlock(cv, VIEW_CT, ct, blockIndex), E.base_color,
    "switching metric left the block selected -- did the widget re-rank, or keep filtering the "
    + "old order?");
});

test("Nairobi shows a pool size and no precision or recall", async () => {
  const { host } = await mount();
  setCity(host, true);
  const text = readoutText(host);

  const shipped = nb.floors.find((f) => f.metric === "depth_density_proxy")!;
  const sel = selectAt(nb, ranking(nb, "depth_density_proxy"), scores(nb, "depth_density_proxy"),
    shipped.value);
  assert.ok(sel.count > 0, "sanity: the shipped floor should select a non-empty pool on Nairobi");

  assert.ok(text.includes(String(sel.count)), `readout does not carry the pool size: "${text}"`);
  assert.ok(!text.includes("%"),
    `readout shows a percentage for a city with no ground-truth layer: "${text}"`);
});

test("the base layer is drawn once, not per frame", async () => {
  // The performance claim, made checkable: count base-colour fills across two frames after a
  // floor change and require it not to grow by 16,451.
  const { host, cv } = await mount();
  const afterMount = fillsOf(callsSince(cv, 0), E.base_color);
  assert.equal(afterMount, ct.n_blocks,
    "the first frame should fill every block's base colour exactly once");

  const shipped = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  const before = callCount(cv);
  setFloor(host, shipped.value * 2); // a stricter floor -- the selection shrinks
  const added = fillsOf(callsSince(cv, before), E.base_color);

  // The correct design undoes exactly the PREVIOUS prefix (`shipped.n` blocks) and touches
  // nothing else -- nowhere near the full base layer. A per-frame rebuild would instead add
  // another `ct.n_blocks` (16,451) base-colour fills here.
  assert.equal(added, shipped.n);
  assert.ok(added < ct.n_blocks,
    `a floor change repainted ${added} base-colour blocks, not far fewer than ${ct.n_blocks}`);
});
