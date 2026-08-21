import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { ranking, scores, selectAt, type MetricName } from "../src/model/screen.js";
import { localState } from "../src/state.js";
import { fitBbox, toScreen, type Bbox, type View } from "../src/view/transform.js";
import {
  canvasOf, Call, FakeElement, fireAnimationFrame, fireResize, installStubs, lastFrame, mountPoint,
} from "./harness.js";
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
 * or five). Calling `fireResize` before that drain finds no observer to fire.
 *
 * The FIRST frame is drawn synchronously from inside the resize callback (screen-map.ts's own
 * `render(true)`, not routed through `requestAnimationFrame`), so this needs no
 * `fireAnimationFrame()` of its own -- only STATE-driven redraws (a floor drag, a metric switch, a
 * city toggle) are rAF-coalesced; see `setFloor`/`setMetric`/`setCity` below. */
async function mount(width = 700): Promise<{ host: ReturnType<typeof mountPoint>; cv: unknown }> {
  const host = mountPoint();
  host.dataset.bundleCapetown = "../examples/screen-map/capetown.json";
  host.dataset.bundleNairobi = "../examples/screen-map/nairobi.json";
  screenMap(host as never, localState);
  await new Promise((resolve) => setTimeout(resolve, 0));
  fireResize(width, width);
  return { host, cv: canvasOf(host) };
}

function callsOf(el: unknown): Call[] {
  return (el as { ctx: { calls: Call[] } }).ctx.calls;
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

/** Drives one control the way a reader's pointer or keyboard would (set `.value`/`.checked`, then
 * dispatch the event the widget listens for -- field-boot.test.ts's and region-grow-boot.test.ts's
 * own `setSlider` shape) and flushes the ONE animation frame that interaction schedules.
 * `state.set()` itself is synchronous, but screen-map.ts's own `scheduleRender` defers the actual
 * redraw to `requestAnimationFrame` (design item 3), so a test that inspected the canvas
 * immediately after `dispatch` would still be looking at the PREVIOUS frame. */
function setFloor(host: ReturnType<typeof mountPoint>, value: number): void {
  const el = floorSlider(host);
  el.value = String(value);
  el.dispatch("input");
  fireAnimationFrame();
}
function setMetric(host: ReturnType<typeof mountPoint>, metric: MetricName): void {
  const el = metricSelectEl(host);
  el.value = metric;
  el.dispatch("change");
  fireAnimationFrame();
}
function setCity(host: ReturnType<typeof mountPoint>, showNairobi: boolean): void {
  const el = cityToggleEl(host);
  el.checked = showNairobi;
  el.dispatch("change");
  fireAnimationFrame();
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

/** The offscreen canvas the last frame's OWN `drawImage` call blitted, found by following the
 * `Call.image` reference `RecordingContext.drawImage` records -- not by assuming there is only
 * ever one other canvas anywhere, so this stays correct even if a page ever holds more than one
 * ScreenMap. */
function offscreenOf(cv: unknown): FakeElement {
  const frame = lastFrame(cv as never);
  const blit = frame.find((c) => c.op === "drawImage");
  assert.ok(blit !== undefined, "no drawImage call in the last frame");
  assert.ok(blit.image !== undefined, "the drawImage call recorded no source image");
  return blit.image;
}

function firstVertexScreen(view: View, bundle: CityBundle, blockIndex: number): [number, number] {
  const [x, y] = bundle.rings[blockIndex]![0]![0]!;
  return toScreen(view, x, y);
}

/** Whether `blockIndex` is painted `E.selected_color` in the CURRENT frame on the MAIN canvas --
 * identified by a fill whose path opens with a `moveTo` at that block's own first vertex, the way
 * `render/city.ts`'s `fillBlock` always starts one. A block that is NOT selected is never filled
 * on the main canvas at all under this design (it shows through from the blitted offscreen base
 * layer instead), so absence here means "not selected", not "selected in some other colour". */
function isSelected(cv: unknown, view: View, bundle: CityBundle, blockIndex: number): boolean {
  const [sx, sy] = firstVertexScreen(view, bundle, blockIndex);
  return lastFrame(cv as never).some((c) => {
    const first = c.path[0];
    return c.op === "fill" && c.fillStyle === E.selected_color && first !== undefined
      && first.op === "moveTo" && first.args[0] === sx && first.args[1] === sy;
  });
}

/** The fill colour `blockIndex` carries on the OFFSCREEN base layer -- `base_color`, or
 * `informal_color` for a Cape Town block the City's own survey records as really informal. Same
 * first-vertex identification as `isSelected`, applied to the offscreen canvas's own call log. */
function baseFillColorOf(offscreen: FakeElement, view: View, bundle: CityBundle,
                         blockIndex: number): string | undefined {
  const [sx, sy] = firstVertexScreen(view, bundle, blockIndex);
  const hit = offscreen.ctx.calls.find((c) => {
    const first = c.path[0];
    return c.op === "fill" && first !== undefined && first.op === "moveTo"
      && first.args[0] === sx && first.args[1] === sy;
  });
  return hit?.fillStyle;
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
  const selectedFills = lastFrame(cv as never)
    .filter((c) => c.op === "fill" && c.fillStyle === E.selected_color);
  assert.equal(selectedFills.length, sel.count);
  // Cross-checked against the independently baked pool size too (examples/screen-bakeoff's own
  // route, read into the bundle at bake time) -- the same two-paths-agreeing guard Task 8's own
  // screen-model.test.ts relies on.
  assert.equal(selectedFills.length, shipped.n);
  for (const c of selectedFills) {
    // Nonzero (the fake's own default when a caller omits the argument) would fill any block that
    // carries interior rings solid instead of cutting the holes out.
    assert.equal(c.fillRule, "evenodd", `block filled with fillRule ${c.fillRule}, not evenodd`);
  }
});

test("moving the floor changes the drawn selection", async () => {
  // A widget that computed the selection but drew a constant would pass a count test at one
  // floor. Assert at two floors and require the counts to differ.
  const { host, cv } = await mount();
  const countAtDefault = lastFrame(cv as never)
    .filter((c) => c.op === "fill" && c.fillStyle === E.selected_color).length;

  // The single highest-scoring block's own score: raising the floor to it selects (about) one
  // block, a difference that cannot be an artefact of two nearly-identical floors.
  const s = scores(ct, "depth_density_proxy");
  let max = -Infinity;
  for (const v of s) if (v > max) max = v;

  setFloor(host, max);
  const countAtMax = lastFrame(cv as never)
    .filter((c) => c.op === "fill" && c.fillStyle === E.selected_color).length;

  assert.notEqual(countAtMax, countAtDefault,
    `floor change produced the same selected count (${countAtDefault})`);
  assert.equal(countAtMax, selectAt(ct, ranking(ct, "depth_density_proxy"), s, max).count);
});

test("switching metric re-ranks rather than re-filtering the old ranking", async () => {
  // Pick a block that is above the floor under one metric and below it under another, from the
  // committed bundle, and assert its membership flips.
  const { host, cv } = await mount();
  const blockIndex = findFlip(ct, "depth_density_proxy", "density_compactness");

  assert.ok(isSelected(cv, VIEW_CT, ct, blockIndex),
    "the chosen block should start out selected under the default metric, depth_density_proxy");

  setMetric(host, "density_compactness");

  assert.ok(!isSelected(cv, VIEW_CT, ct, blockIndex),
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

test("Cape Town's readout pins the actual precision and recall numbers", async () => {
  // The reviewer deleted precision and recall from the readout entirely and all other tests
  // stayed green -- the headline feature was unguarded. This pins it against the bundle's OWN
  // baked figures (examples/screen-bakeoff's independent route), not a recomputed copy of the
  // widget's own formula.
  const { host } = await mount();
  const text = readoutText(host);
  const shipped = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  assert.ok(shipped.precision !== null && shipped.recall !== null,
    "sanity: the shipped Cape Town floor should carry precision/recall");
  const expectedPrecision = `${(shipped.precision! * 100).toFixed(1)}%`;
  const expectedRecall = `${(shipped.recall! * 100).toFixed(1)}%`;
  assert.ok(text.includes(expectedPrecision),
    `readout is missing the precision figure ${expectedPrecision}: "${text}"`);
  assert.ok(text.includes(expectedRecall),
    `readout is missing the recall figure ${expectedRecall}: "${text}"`);
});

test("the base layer is drawn once, not per frame, and blitted every frame", async () => {
  // The performance claim, made checkable the strong way: one drawImage per frame, with the
  // offscreen canvas it names holding exactly one fill per block -- on the first frame AND after
  // a floor change, not growing by another 16,451.
  const { host, cv } = await mount();

  const frame1 = lastFrame(cv as never);
  assert.equal(frame1.filter((c) => c.op === "drawImage").length, 1,
    "each frame should blit the base layer exactly once");
  const offscreen1 = offscreenOf(cv);
  const fills1 = offscreen1.ctx.calls.filter((c) => c.op === "fill");
  assert.equal(fills1.length, ct.n_blocks,
    "the offscreen canvas should hold exactly one fill per block");
  for (const c of fills1) assert.equal(c.fillRule, "evenodd");

  // block_lw consumed: every block outlined once on the offscreen layer, at the bundle's own
  // width -- not a literal, and not a field that ships and is never read.
  const strokes1 = offscreen1.ctx.calls.filter((c) => c.op === "stroke");
  assert.equal(strokes1.length, ct.n_blocks, "every block should be outlined once on the base layer");
  for (const c of strokes1) {
    assert.equal(c.strokeStyle, E.base_color);
    assert.equal(c.lineWidth, E.block_lw);
  }

  const shipped = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  setFloor(host, shipped.value * 2); // a stricter floor -- the selection shrinks

  const frame2 = lastFrame(cv as never);
  assert.equal(frame2.filter((c) => c.op === "drawImage").length, 1,
    "a floor change should still blit exactly once");
  const blit2 = frame2.find((c) => c.op === "drawImage")!;
  assert.equal(blit2.image, offscreen1,
    "the SAME offscreen canvas must be reused across a floor change, not rebuilt");

  const offscreen2 = offscreenOf(cv);
  assert.equal(offscreen2.ctx.calls.filter((c) => c.op === "fill").length, ct.n_blocks,
    "a floor change must not repaint the offscreen base layer -- its fill count must be unchanged");
});

test("Cape Town's real informal blocks are painted informal_color on the base layer", async () => {
  // informal_color consumed: an unguarded field is exactly the defect class this fix round is
  // about, so it gets the same "picked one block and asserted its own colour" treatment as
  // everything else here, not merely a shape check.
  const { cv } = await mount();
  const offscreen = offscreenOf(cv);
  const informalIndex = ct.informal!.findIndex((v) => v === 1);
  const plainIndex = ct.informal!.findIndex((v) => v === 0);
  assert.ok(informalIndex >= 0 && plainIndex >= 0, "sanity: both kinds of block should exist");
  assert.equal(baseFillColorOf(offscreen, VIEW_CT, ct, informalIndex), E.informal_color);
  assert.equal(baseFillColorOf(offscreen, VIEW_CT, ct, plainIndex), E.base_color);
});

test("floor-slider redraws are requestAnimationFrame-coalesced, not synchronous", async () => {
  const { host, cv } = await mount();
  const before = callsOf(cv).length;

  const slider = floorSlider(host);
  slider.value = String(0.02);
  slider.dispatch("input");
  // No fireAnimationFrame() yet -- a widget that redrew synchronously on state.set would already
  // show new calls here.
  assert.equal(callsOf(cv).length, before,
    "the canvas redrew synchronously on state.set, before any animation frame fired");

  slider.value = String(0.03); // a second change before the first has ever been flushed
  slider.dispatch("input");
  fireAnimationFrame();

  // Coalesced: exactly one new frame (one new drawImage), and it reflects the LATEST value
  // (0.03), not two frames and not the first (stale) one.
  const added = callsOf(cv).slice(before);
  const blits = added.filter((c) => c.op === "drawImage");
  assert.equal(blits.length, 1,
    `two rapid floor changes before a flush produced ${blits.length} frames, not one`);
  const expected = selectAt(ct, ranking(ct, "depth_density_proxy"), scores(ct, "depth_density_proxy"),
    0.03).count;
  const drawn = added.filter((c) => c.op === "fill" && c.fillStyle === E.selected_color).length;
  assert.equal(drawn, expected, "the coalesced frame should reflect the LATEST floor, not the first");
});

test("an uncalibrated metric defaults to the shipped default's own pool size", async () => {
  // `density` ships no floors[] entry of its own -- the fallback must not degenerate to "select
  // every block" (which reports the corpus base rate as a screen's own performance and defeats
  // city.ts's O(prefix) design outright).
  const { host, cv } = await mount();
  setMetric(host, "density");

  const shippedDefault = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  const drawn = lastFrame(cv as never)
    .filter((c) => c.op === "fill" && c.fillStyle === E.selected_color).length;
  assert.equal(drawn, shippedDefault.n,
    `density's own default pool should match depth_density_proxy's shipped pool size `
    + `(${shippedDefault.n}), not select every block`);
  assert.notEqual(drawn, ct.n_blocks);
});
