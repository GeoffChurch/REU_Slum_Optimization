import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { ranking, scores, selectAt, type MetricName } from "../src/model/screen.js";
import { FOLLOW_RADIUS_PX } from "../src/render/city.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import { fitBbox, toScreen, type Bbox, type View } from "../src/view/transform.js";
import {
  armDrawFailure, canvasOf, Call, DPR, fakeLocation, FakeElement, fireAnimationFrame, fireResize,
  installStubs, lastFrame, mountPoint, writeNow,
} from "./harness.js";
import { screenMap, SCREEN_MAP_URL, type ScreenState } from "../src/widgets/screen-map.js";

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
 * city toggle) are rAF-coalesced; see `setFloor`/`setMetric`/`setCity` below.
 *
 * `drawFailure` arms `armDrawFailure` (harness.ts) BEFORE `screenMap` runs, so every canvas `boot()`
 * creates during the awaited macrotask below -- the visible one AND `createLayer`'s offscreen one --
 * captures it as its own `RecordingContext.failWith`, region-grow-boot.test.ts's own `mount` doing
 * the same for the same timing reason. Cleared in a `finally` after `fireResize` so a later call to
 * `mount()` with no `drawFailure` is not accidentally armed by a previous test's leftover state.
 *
 * The state store is the PRODUCTION one (`urlStore` over a `fakeLocation`), never `localState`, so
 * the URL tests below exercise the real decode path rather than a second, test-only one --
 * region-grow-boot.test.ts's own `mount` for the same reason. `search` defaults to "": an empty
 * query decodes nothing, and this widget corrects nothing at boot from it either, so `loc.written`
 * stays empty until a control is touched and every caller that passes no search gets exactly the
 * widget's own initial state. */
async function mount(width = 700, drawFailure: string | null = null, search = ""):
    Promise<{ host: ReturnType<typeof mountPoint>; cv: unknown;
              store: StateSource<ScreenState>; loc: ReturnType<typeof fakeLocation> }> {
  const host = mountPoint();
  host.dataset.bundleCapetown = "../examples/screen-map/capetown.json";
  host.dataset.bundleNairobi = "../examples/screen-map/nairobi.json";
  armDrawFailure(drawFailure);
  const loc = fakeLocation(search);
  const urls = urlStore(loc, writeNow);
  let bound: StateSource<ScreenState> | null = null;
  screenMap(host as never, (initial) => {
    bound = urls.bind(SCREEN_MAP_URL, initial);
    return bound;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  try {
    fireResize(width, width);
  } finally {
    armDrawFailure(null);
  }
  // Every mount in this file feeds `boot` the two committed bundles, and `boot` reaches `makeState`
  // before it draws anything -- so even a `drawFailure` mount gets here with a store. A null is a
  // widget that failed before it.
  assert.ok(bound !== null, "the widget never asked for a state store");
  return { host, cv: canvasOf(host), store: bound, loc };
}

function callsOf(el: unknown): Call[] {
  return (el as { ctx: { calls: Call[] } }).ctx.calls;
}
function transformsOf(el: unknown): number[][] {
  return (el as { ctx: { transforms: number[][] } }).ctx.transforms;
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

/** Every call in `calls` whose path opens with a `moveTo` at `blockIndex`'s own first vertex --
 * the way `render/city.ts`'s `tracePath` always starts one, whether the call that follows is a
 * `fill` (the offscreen base layer) or a `stroke` (the selected-prefix outline). Shared by
 * `isSelected`, `baseFillColorOf` and the informal/selection interaction test below, so the
 * first-vertex matching logic exists in exactly one place.
 *
 * Cape Town's own geometry makes that key a real collision risk -- 606 first-vertex keys are shared
 * by 1,213 of 16,451 blocks, measured against the committed bundle -- so a caller could silently be
 * handed another block's draws instead of `blockIndex`'s own. Asserted here, not merely trusted,
 * because three of this file's four callers pick `blockIndex` FROM THE DATA at test time (`findFlip`,
 * `findHit`, `ct.informal!.findIndex(...)`), so a re-bake can move any of them onto a colliding key
 * with no test edit -- turning that into a loud failure instead of a silently weakened assertion. */
function callsForBlock(calls: Call[], view: View, bundle: CityBundle, blockIndex: number): Call[] {
  const [sx, sy] = firstVertexScreen(view, bundle, blockIndex);
  const owners = bundle.rings
    .map((_, i) => firstVertexScreen(view, bundle, i))
    .filter(([ix, iy]) => ix === sx && iy === sy).length;
  assert.equal(owners, 1, `block ${blockIndex}'s first-vertex key (${sx}, ${sy}) is shared by `
    + `${owners} blocks -- callsForBlock would conflate their draws`);
  return calls.filter((c) => {
    const first = c.path[0];
    return first !== undefined && first.op === "moveTo" && first.args[0] === sx
      && first.args[1] === sy;
  });
}

/** The floor `density` and `depth_proxy` fall back to, neither shipping a `floors[]` entry of its
 * own: the score of the block at the SHIPPED DEFAULT metric's own pool size, in THIS metric's own
 * ranking (`screen-map.ts`'s `floorAtShippedPoolSize`). Spelled out here rather than imported so
 * the tests below read without opening the widget -- being the same arithmetic, it pins WHICH
 * BUNDLE resolved a number, not the formula. The formula is pinned by the pool-size assertions
 * beside it, which come from the bundle's own independently baked `n`. */
function poolSizeFloor(bundle: CityBundle, metric: MetricName): number {
  const shipped = bundle.floors.find((f) => f.metric === "depth_density_proxy")!;
  return scores(bundle, metric)[ranking(bundle, metric)[shipped.n - 1]!]!;
}

/** Blocks the picture currently shows as SELECTED, in the CURRENT frame -- identified by the
 * bundle's own `selected_color`, never by a path count (region-grow-boot.test.ts's own layer
 * helpers carry the reasoning). Stroked, not filled: see `isSelected` just below. */
function selectedPaths(cv: unknown): Call[] {
  return lastFrame(cv as never)
    .filter((c) => c.op === "stroke" && c.strokeStyle === E.selected_color);
}

/** Whether `blockIndex` is outlined `E.selected_color` in the CURRENT frame on the MAIN canvas.
 * A block that is NOT selected is never stroked on the main canvas at all under this design (it
 * shows through from the blitted offscreen base layer instead), so absence here means "not
 * selected", not "selected in some other colour". */
function isSelected(cv: unknown, view: View, bundle: CityBundle, blockIndex: number): boolean {
  return callsForBlock(lastFrame(cv as never), view, bundle, blockIndex)
    .some((c) => c.op === "stroke" && c.strokeStyle === E.selected_color);
}

/** The fill colour `blockIndex` carries on the OFFSCREEN base layer -- `base_color`, or
 * `informal_color` for a Cape Town block the City's own survey records as really informal. */
function baseFillColorOf(offscreen: FakeElement, view: View, bundle: CityBundle,
                         blockIndex: number): string | undefined {
  const hit = callsForBlock(offscreen.ctx.calls, view, bundle, blockIndex)
    .find((c) => c.op === "fill");
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

/** A block selected at `metric`'s own shipped floor that the City's own survey also records as
 * really informal -- a "hit" in `_render_screen_map`'s own legend (gold + a red ring). 455 of
 * Cape Town's 682 real informal blocks are hits at the shipped `depth_density_proxy` floor
 * (682 * 0.667155 recall, rounded -- the reviewer's own figure); found empirically here, not
 * hardcoded, so the test can never drift from what the bundle and the model actually produce. */
function findHit(bundle: CityBundle, metric: MetricName): number {
  const informal = bundle.informal;
  assert.ok(informal !== undefined, "findHit needs a bundle with ground truth");
  const shipped = bundle.floors.find((f) => f.metric === metric)!;
  const order = ranking(bundle, metric);
  for (let i = 0; i < shipped.n; i++) {
    const block = order[i]!;
    if (informal[block]) return block;
  }
  throw new Error(`no informal block is selected at ${metric}'s own shipped floor -- pick a `
    + `different metric`);
}

test("the number of selected blocks drawn equals what the model selects", async () => {
  // Identified by the bundle's `selected_color`, never by a path count -- see the comment in
  // region-grow-boot.test.ts.
  const { cv } = await mount();
  const shipped = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  const order = ranking(ct, "depth_density_proxy");
  const s = scores(ct, "depth_density_proxy");
  const sel = selectAt(ct, order, s, shipped.value);
  const selectedStrokes = selectedPaths(cv);
  assert.equal(selectedStrokes.length, sel.count);
  // Cross-checked against the independently baked pool size too (examples/screen-bakeoff's own
  // route, read into the bundle at bake time) -- the same two-paths-agreeing guard Task 8's own
  // screen-model.test.ts relies on.
  assert.equal(selectedStrokes.length, shipped.n);
  for (const c of selectedStrokes) {
    // Matches `_render_screen_map`'s own selected linewidth exactly -- an outline at the base
    // layer's own width would be nearly invisible against it, and a filled rather than stroked
    // selection is the whole defect this fix round closes (see render/city.ts's own docstring).
    assert.equal(c.lineWidth, E.block_lw * 2, `selected outline drawn at ${c.lineWidth}, not block_lw * 2`);
  }
});

test("moving the floor changes the drawn selection", async () => {
  // A widget that computed the selection but drew a constant would pass a count test at one
  // floor. Assert at two floors and require the counts to differ.
  const { host, cv } = await mount();
  const countAtDefault = selectedPaths(cv).length;

  // The single highest-scoring block's own score: raising the floor to it selects (about) one
  // block, a difference that cannot be an artefact of two nearly-identical floors.
  const s = scores(ct, "depth_density_proxy");
  let max = -Infinity;
  for (const v of s) if (v > max) max = v;

  setFloor(host, max);
  const countAtMax = selectedPaths(cv).length;

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

test("the base layer is drawn once, not per frame, and blitted every frame in device-pixel space",
  async () => {
    // The performance claim, made checkable the strong way: one drawImage per frame, with the
    // offscreen canvas it names holding exactly one fill per block -- on the first frame AND after
    // a floor change, not growing by another 16,451 -- and the blit itself bracketed by a reset to
    // the identity transform and back, so a DPR-scaled backing store is copied 1:1 rather than
    // blown up by another factor of dpr.
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
    // width -- not a literal, and not a field that ships and is never read. The outline colour must
    // match THAT BLOCK'S OWN fill, not merely be "one of the two colours in play" -- an `assert.ok`
    // accepting either constant for every block would pass an injection that outlined ALL 16,451
    // blocks gold just as readily as one that left the pre-fix bug (all outlined base_color) in
    // place, since both colours are individually valid somewhere in the bundle. Pairing each stroke
    // with the fill it was drawn immediately after -- `fillBlock`/`strokeBlock` run back-to-back per
    // block inside `paintBase`'s own loop, so `fills1[i]`/`strokes1[i]` are the same block's two
    // calls -- is what actually pins "outline = fill", in both directions, corpus-wide.
    const strokes1 = offscreen1.ctx.calls.filter((c) => c.op === "stroke");
    assert.equal(strokes1.length, ct.n_blocks, "every block should be outlined once on the base layer");
    for (let i = 0; i < strokes1.length; i++) {
      assert.equal(strokes1[i]!.strokeStyle, fills1[i]!.fillStyle,
        `block ${i}'s outline (${strokes1[i]!.strokeStyle}) does not match its own fill `
        + `(${fills1[i]!.fillStyle}) -- one colour would paint over the other`);
      assert.equal(strokes1[i]!.lineWidth, E.block_lw);
    }

    // The blit's own transform bracket: reset to identity, THEN restored to the DPR scale --
    // dropping the reset would blit a DPR-scaled backing store at dpr² the intended size, showing
    // only its top-left 1/dpr² under a correctly-scaled selected overlay on any HiDPI display.
    assert.deepEqual(transformsOf(cv).slice(-2), [[1, 0, 0, 1, 0, 0], [DPR, 0, 0, DPR, 0, 0]]);

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

    // The transform bracket holds again on this SECOND frame too, not just the first.
    assert.deepEqual(transformsOf(cv).slice(-2), [[1, 0, 0, 1, 0, 0], [DPR, 0, 0, DPR, 0, 0]]);
  });

test("the blit reuses the DPR the offscreen canvas was actually sized for, not a fresh read",
  async () => {
    // "One source for one fact": paintBase stores the DPR it actually sized the offscreen canvas
    // for; paintFrame must reuse THAT value on a floor-only change (which never re-runs
    // paintBase), not re-read window.devicePixelRatio independently -- an independent re-read
    // could disagree with what the offscreen canvas is actually sized for if the display's DPR
    // changed with no resize in between, scaling the selected-prefix overlay wrongly relative to
    // the just-blitted base layer underneath it.
    const { host, cv } = await mount();
    const fakeWindow = (globalThis as { window: { devicePixelRatio: number } }).window;
    const original = fakeWindow.devicePixelRatio;
    fakeWindow.devicePixelRatio = 3; // no resize follows, so paintBase never sees this
    try {
      setFloor(host, ct.floors.find((f) => f.metric === "depth_density_proxy")!.value * 2);
      const last2 = transformsOf(cv).slice(-2);
      assert.deepEqual(last2, [[1, 0, 0, 1, 0, 0], [DPR, 0, 0, DPR, 0, 0]],
        `blit transform was ${JSON.stringify(last2)} -- should still use the DPR the offscreen `
        + `canvas was actually sized for (${DPR}), not a freshly re-read devicePixelRatio (3)`);
    } finally {
      fakeWindow.devicePixelRatio = original;
    }
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

test("a known informal block's own outline does not overpaint its gold fill", async () => {
  // The reviewer's finding: `paintBase` filled informal blocks gold but then unconditionally
  // outlined EVERY block, informal or not, in `base_color` at `block_lw` -- and the median informal
  // block is ~0.335 CSS px² at this canvas size, smaller than the outline itself, so that outline
  // covers the whole interior, not merely the edge. The `Call` log cannot see pixel coverage
  // directly, but it can see the colour the outline was drawn in: the fix makes each block's own
  // outline match its own fill, so an informal block's outline can never be a DIFFERENT, opaque
  // colour that would paint over the fill underneath it.
  const { cv } = await mount();
  const offscreen = offscreenOf(cv);
  const informalIndex = ct.informal!.findIndex((v) => v === 1);
  assert.ok(informalIndex >= 0, "sanity: an informal block should exist");

  const calls = callsForBlock(offscreen.ctx.calls, VIEW_CT, ct, informalIndex);
  const fill = calls.find((c) => c.op === "fill");
  const stroke = calls.find((c) => c.op === "stroke");
  assert.ok(fill !== undefined, "sanity: the informal block should be filled on the base layer");
  assert.equal(fill.fillStyle, E.informal_color);
  assert.ok(stroke !== undefined, "sanity: the informal block should be outlined on the base layer");
  assert.equal(stroke.strokeStyle, E.informal_color,
    `the informal block's own outline is drawn in ${stroke.strokeStyle}, which would paint over `
    + `its gold fill at this block's sub-pixel size`);
});

test("a selected block that is really informal still shows gold underneath the red outline",
  async () => {
    // The reviewer's finding: an OPAQUE selected fill erased informal_color, making a hit
    // indistinguishable from a false positive and silently redefining "gold" as "missed" rather
    // than "ground truth". The fix strokes the selection instead of filling it, so this must
    // hold: the block is currently selected, its own base-layer fill is STILL informal_color
    // (paintFrame never touches the offscreen canvas), and the main canvas never fills that
    // block at all -- only strokes it, leaving the blitted gold underneath untouched.
    const { cv } = await mount();
    const hitIndex = findHit(ct, "depth_density_proxy");

    assert.ok(isSelected(cv, VIEW_CT, ct, hitIndex), "sanity: the chosen block should be selected");

    const offscreen = offscreenOf(cv);
    assert.equal(baseFillColorOf(offscreen, VIEW_CT, ct, hitIndex), E.informal_color,
      "the block's own base-layer fill should still be informal_color");

    const fillsForBlock = callsForBlock(lastFrame(cv as never), VIEW_CT, ct, hitIndex)
      .filter((c) => c.op === "fill");
    assert.equal(fillsForBlock.length, 0,
      "the main canvas filled the selected block -- that would erase the gold underneath");
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
  const drawn = added.filter((c) => c.op === "stroke" && c.strokeStyle === E.selected_color).length;
  assert.equal(drawn, expected, "the coalesced frame should reflect the LATEST floor, not the first");
});

test("an uncalibrated metric defaults to the shipped default's own pool size", async () => {
  // `density` ships no floors[] entry of its own -- the fallback must not degenerate to "select
  // every block" (which reports the corpus base rate as a screen's own performance and defeats
  // city.ts's O(prefix) design outright).
  const { host, cv } = await mount();
  setMetric(host, "density");

  const shippedDefault = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  const drawn = selectedPaths(cv).length;
  assert.equal(drawn, shippedDefault.n,
    `density's own default pool should match depth_density_proxy's shipped pool size `
    + `(${shippedDefault.n}), not select every block`);
  assert.notEqual(drawn, ct.n_blocks);
});

test("a throw on the first draw is reported and keeps the static image", async () => {
  // The resize callback runs from the browser's own dispatch, OUTSIDE the mount's
  // `Promise.all([...]).then(boot).catch(...)` chain, so a throw in it is an uncaught exception and
  // nothing else -- a blank figure, no message, and a page that still looks laid out -- unless
  // `screen-map.ts`'s own `runOrReport` catches it. field-boot.test.ts's and
  // perm-graph-boot.test.ts's own boot suites both have this guard; neither new widget's did.
  const { host } = await mount(700, "boom on the first draw");

  assert.match(host.querySelector("figcaption")!.textContent,
    /ScreenMap could not load interactively .*boom on the first draw/);
  assert.match(host.querySelector("figcaption")!.textContent,
    /The static image above still applies\./);
  assert.ok(host.querySelector("img") !== null,
    "the fallback image was removed although the drawing that replaces it threw");
});

test("?metric= alone takes THAT metric's own default floor, not the previous metric's number",
  async () => {
    // Design §1.6's row 3, the one a plain write-back gets wrong. Both kinds of metric, because
    // "this metric's own default" resolves two different ways: `density_compactness` ships a
    // calibration in `ct.floors`, `depth_proxy` ships none and falls back to the shipped default's
    // own pool size. Measured against the committed bundle, a write-back of
    // `depth_density_proxy`'s 0.0128 would select 1 block under `density_compactness` (whose scores
    // top out at 0.00109, so it clamps to the ceiling) and all 16,451 under `depth_proxy` (whose
    // scores start at 0.168, so it clamps to the floor) -- neither of them this metric's own pool,
    // from a URL that asked for nothing unusual.
    const shippedDefault = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
    for (const [metric, expected] of [
      ["density_compactness", ct.floors.find((f) => f.metric === "density_compactness")!.n],
      ["depth_proxy", shippedDefault.n],
    ] as const) {
      const { host, cv, store } = await mount(700, null, `metric=${metric}`);
      assert.equal(store.get().floor, null,
        `${metric}: null means "this metric's own default", and stays null`);
      assert.equal(selectedPaths(cv).length, expected,
        `${metric}: the picture should draw this metric's own default pool`);
      const text = readoutText(host);
      assert.ok(text.includes(`${expected} of ${ct.n_blocks} blocks selected`),
        `${metric}: the readout should carry that same pool -- "${text}"`);
    }
  });

test("?metric= and ?floor= together honour the explicit floor, in state AND on the slider",
  async () => {
    // The desync this task closes: `syncFloor` writes the slider BEFORE `makeState`, so a
    // URL-supplied floor reached the picture and never the control. 0.0004 sits inside
    // `density_compactness`'s own score range on this bundle (its ceiling is ~0.00109), so nothing
    // clamps it and the number the reader typed is the number all three report.
    const { host, cv, store } = await mount(700, null, "metric=density_compactness&floor=0.0004");
    assert.equal(store.get().metric, "density_compactness");
    assert.equal(store.get().floor, 0.0004);
    assert.equal(floorSlider(host).value, "0.0004",
      "the slider initialised from the metric's own default while the picture drew 0.0004");
    const expected = selectAt(ct, ranking(ct, "density_compactness"),
      scores(ct, "density_compactness"), 0.0004).count;
    assert.equal(selectedPaths(cv).length, expected, "the canvas did not draw the URL's floor");
  });

test("switching metric drops ?floor= from the URL", async () => {
  // A metric switch resets to the new metric's own calibration, so the number in the URL is
  // nobody's choice any more. `floor: null` is what lets the store stop emitting the key at all,
  // rather than publishing a number the reader never picked on a scale they never saw.
  const { host, loc } = await mount(700, null, "floor=0.02");
  // `loc.search()`, not `loc.written`: 0.02 is inside `depth_density_proxy`'s own range on this
  // bundle, so boot has nothing to correct and nothing has been written yet -- the URL carrying the
  // floor here is the one the reader arrived with.
  assert.ok(loc.search().includes("floor="), "precondition: an explicit floor is in the URL");
  setMetric(host, "density_compactness");
  assert.equal(loc.written.at(-1), "metric=density_compactness",
    "a metric switch resets to the new metric's calibration, so no floor belongs in the URL");
});

test("switching city carries a floor the READER set, and invents none where they set nothing",
  async () => {
    // `syncFloor`'s own docstring argues that an ABSOLUTE floor carries to a corpus with no
    // calibration rather than being redefined per city -- right for a number the reader picked, and
    // wrong for `null`, which is not an absolute floor at all but "whatever this metric calibrates
    // to". So the switch resolves for the CONTROL either way and pins only a chosen value into
    // STATE (design §1.6, corrected during this task).
    const untouched = await mount();
    setCity(untouched.host, true);
    assert.equal(untouched.store.get().floor, null,
      "an untouched floor is not a choice, and a city switch turned it into one");
    assert.equal(untouched.loc.written.at(-1), "city=nairobi",
      "a `?floor=` the reader never typed is in the URL they would copy");
    // The slider tracks the new bundle even though nothing was pinned -- `syncFloor` is called for
    // its side effect on the CONTROL whichever way the state goes. Nairobi's own
    // depth_density_proxy scores top out well below Cape Town's (0.0537 against 0.158), so a
    // handler that skipped the call would leave the control offering a range this corpus has not
    // got.
    let nbMax = -Infinity;
    for (const v of scores(nb, "depth_density_proxy")) if (v > nbMax) nbMax = v;
    assert.equal(Number(floorSlider(untouched.host).max), nbMax,
      "the slider kept the previous city's bounds");

    const chosen = await mount();
    setFloor(chosen.host, 0.02);
    setCity(chosen.host, true);
    assert.equal(chosen.store.get().floor, 0.02,
      "a floor the reader dragged to should carry to the new corpus, not be redefined by it");
    assert.equal(chosen.loc.written.at(-1), "city=nairobi&floor=0.02",
      "the emitted query should name both the city and the floor it carried");
  });

test("a city round trip returns an untouched floor to where it started", async () => {
  // The concrete defect that corrected design §1.6: pinning a `null` floor at the switch invents a
  // number, and the number belongs to the OTHER corpus. Measured on the committed bundles,
  // `density` (which ships no calibration, so its default is the shipped default's own pool size)
  // resolves to 0.00773 on Cape Town and 0.00625 on Nairobi -- so an unconditional pin sends a
  // reader who only looked at Nairobi home to Nairobi's number, drawing 2,859 blocks where their
  // own city's default draws 1,655, under a `?floor=` they never typed.
  const { host, cv, loc, store } = await mount();
  setMetric(host, "density");
  setCity(host, true);
  setCity(host, false);

  assert.equal(store.get().floor, null, "the round trip pinned a floor the reader never chose");
  assert.equal(Number(floorSlider(host).value), poolSizeFloor(ct, "density"),
    "the slider came home to a number that is not Cape Town's own");
  const shippedDefault = ct.floors.find((f) => f.metric === "depth_density_proxy")!;
  assert.equal(selectedPaths(cv).length, shippedDefault.n,
    "the picture came home to a pool sized by the other corpus's calibration");
  assert.equal(loc.written.at(-1), "metric=density",
    "the URL came home carrying a floor nobody typed");
});

test("an out-of-range ?floor= is clamped AND stops being emitted at that value", async () => {
  // A codec's validation ends at "is this a finite decimal"; where 999 falls in THIS metric's score
  // range is a property of the fetched bundle, which no codec can know (design §2.3). Unclamped it
  // selects nothing at all -- an empty picture from a reader's typo -- while the URL keeps offering
  // the value that produced it.
  const { host, cv, loc, store } = await mount(700, null, "floor=999");
  const s = scores(ct, "depth_density_proxy");
  let max = -Infinity;
  for (const v of s) if (v > max) max = v;

  assert.equal(store.get().floor, max, "the state should land on the bundle's own ceiling");
  assert.equal(floorSlider(host).value, String(max), "the slider should agree with it");
  assert.equal(selectedPaths(cv).length,
    selectAt(ct, ranking(ct, "depth_density_proxy"), s, max).count,
    "the picture should draw the clamped floor's pool, not the empty one 999 selects");
  assert.equal(loc.written.at(-1), `floor=${Number(max.toPrecision(6))}`,
    "the URL should self-correct to the clamped value, at the codec's own 6 significant figures");
});

/** The follow ring: the one stroke whose whole path is a single arc. Every other stroke on this
 * frame is a block outline -- a polyline of moveTo/lineTo/closePath -- so this shape test names the
 * ring by what it IS, rather than by counting ops and hoping. */
function followRings(cv: unknown): Call[] {
  return lastFrame(cv as never).filter(
    (c) => c.op === "stroke" && c.path.length === 1 && c.path[0]!.op === "arc");
}

test("the followed block is ringed, at a fixed screen radius, on the frame layer", async () => {
  const { cv } = await mount(700, null, "");
  const rings = followRings(cv);
  assert.equal(rings.length, 1, "exactly one ring, on the frame -- not one per block");
  const [x, y, r] = rings[0]!.path[0]!.args;
  const expected = toScreen(VIEW_CT, ct.follow!.x, ct.follow!.y);
  assert.ok(Math.abs(x! - expected[0]) < 0.5 && Math.abs(y! - expected[1]) < 0.5,
    `the ring is centred at (${x}, ${y}), not on the followed block's own point (${expected})`);
  assert.equal(r, FOLLOW_RADIUS_PX, "a WORLD radius would shrink to nothing at city zoom");
  // Recorded AT the call, which is the whole point of the style snapshot: "the widget assigned
  // follow_color at some moment" says nothing about what was stroked with it.
  assert.equal(rings[0]!.strokeStyle, E.follow_color);
  // A literal, not city.ts's own FOLLOW_LW_PX: a stroke width read out of the module under test
  // agrees with itself whatever it is set to.
  assert.equal(rings[0]!.lineWidth, 2, "the ring is not drawn at its own line width");
});

test("a bundle with no follow draws no ring", async () => {
  const { cv } = await mount(700, null, "city=nairobi");
  assert.equal(followRings(cv).length, 0);
});

test("a floor change's own frame draws the ring, rather than inheriting it from the blit",
  async () => {
  // What this observes is the VISIBLE canvas's own call log, not the picture. A floor change goes
  // through `render(false)`, which re-blits the offscreen base layer rather than repainting it, so
  // a ring painted into `paintBase` would reach the screen inside that one `drawImage` and this
  // frame would carry no arc-stroke of its own. The picture would still SHOW a ring -- the base
  // layer is copied back whole -- so this pins where the ring is drawn, not whether it is visible.
  // `render/city.ts` gives the two reasons that placement matters; neither is survival. 0.02 is
  // simply a floor inside Cape Town's depth_density_proxy range and different from the shipped
  // default, so the state change is real.
  const { host, cv } = await mount(700, null, "");
  setFloor(host, 0.02);
  assert.equal(followRings(cv).length, 1);
});

/** One block's on-screen area in CSS px², exterior ring less its interiors: the shoelace formula
 * over `toScreen`-projected vertices, which is the size a reader's eye actually gets at this fit. */
function screenAreaPx2(view: View, bundle: CityBundle, blockIndex: number): number {
  const areaOf = (ring: [number, number][]): number => {
    let a = 0;
    for (let i = 0; i < ring.length; i++) {
      const [x1, y1] = toScreen(view, ...ring[i]!);
      const [x2, y2] = toScreen(view, ...ring[(i + 1) % ring.length]!);
      a += x1 * y2 - x2 * y1;
    }
    return Math.abs(a) / 2;
  };
  const rings = bundle.rings[blockIndex]!;
  return rings.slice(1).reduce((acc, hole) => acc - areaOf(hole), areaOf(rings[0]!));
}

test("the followed block is far smaller on screen than the ring that marks it", async () => {
  // The premise a FIXED SCREEN radius rests on, measured here against the committed bundle rather
  // than written as a number into render/city.ts that a re-bake could silently falsify: at the
  // city-wide fit the followed block covers a fraction of one CSS pixel, so its own outline -- or
  // these same constants reinterpreted as world lengths and scaled by `view` -- would put nothing
  // on screen at all.
  //
  // It is also the only guard on the ring's MAGNITUDE: the ring test above compares the drawn
  // radius against FOLLOW_RADIUS_PX itself, which agrees with any value the constant takes. Know
  // what this buys, though: the second assertion is satisfied by any radius above
  // sqrt(area / PI) ~ 0.40 px, so it catches 0.1 and would pass a 1 px ring nobody could see. A
  // real legibility floor needs a threshold in CSS pixels that this task has no basis to set, so
  // this asserts the relation it can actually derive rather than a number it would be inventing.
  const follow = ct.follow!;
  assert.equal(ct.block_id[follow.index], follow.block_id,
    "sanity: follow.index does not address follow.block_id's own row");
  const area = screenAreaPx2(VIEW_CT, ct, follow.index);
  assert.ok(area < 1,
    `the followed block covers ${area.toFixed(3)} CSS px^2 -- if an outline of it were visible, `
    + `the ring would not need to exist`);
  assert.ok(area < Math.PI * FOLLOW_RADIUS_PX ** 2,
    `a radius-${FOLLOW_RADIUS_PX} ring covers ${(Math.PI * FOLLOW_RADIUS_PX ** 2).toFixed(1)} CSS `
    + `px^2, which does not stand out against the ${area.toFixed(3)} px^2 block it marks`);
});

/** The follow ring's own description: the one paragraph in the widget that is NOT the aria-live
 * readout. Found by the ABSENCE of that attribute rather than by position among the paragraphs, so
 * this keeps naming the right element if the two are ever reordered -- and asserts there is exactly
 * one, so a second static paragraph cannot be silently read instead of it. */
function followNoteText(host: ReturnType<typeof mountPoint>): string {
  const notes = host.findAll("p").filter((el) => el.getAttribute("aria-live") === null);
  assert.equal(notes.length, 1, `expected exactly one non-live paragraph, found ${notes.length}`);
  return notes[0]!.textContent;
}

test("the followed block is described in text, outside the live region", async () => {
  // The ring lives in canvas pixels, which carry no accessible text at all -- so a screen-reader
  // user is told the pool size and never told which block the rest of the site follows unless the
  // widget says so somewhere.
  //
  // Somewhere, but NOT in the aria-live readout: that region is re-announced on every frame, and a
  // floor drag produces many, while this sentence never changes within a city. Both halves are
  // asserted, because putting it in the live region is the easy and wrong way to satisfy the first.
  const followed = ct.follow!.block_id;
  const { host } = await mount();
  assert.ok(followNoteText(host).includes(followed),
    `nothing names the followed block ${followed}: "${followNoteText(host)}"`);
  assert.ok(!readoutText(host).includes(followed),
    `the followed block is named inside the aria-live readout, which is re-announced on every `
    + `frame of a floor drag: "${readoutText(host)}"`);
});

test("a city switch clears a description that named the other city's block", async () => {
  // Written from the ACTIVE bundle on every render, not once at mount. Text set at mount only would
  // still name ZAF.9.3.1_1_40972 over Nairobi's map -- a bundle that carries no `follow` at all,
  // and whose canvas carries no ring. Both routes to Nairobi are checked: a URL that starts there,
  // and a toggle that arrives there, since only the second can go stale.
  const followed = ct.follow!.block_id;
  const booted = await mount(700, null, "city=nairobi");
  assert.equal(followNoteText(booted.host), "",
    `Nairobi booted with a description of a block it does not carry: `
    + `"${followNoteText(booted.host)}"`);
  const toggled = await mount();
  assert.ok(followNoteText(toggled.host).includes(followed), "sanity: Cape Town describes its own");
  setCity(toggled.host, true);
  assert.equal(followNoteText(toggled.host), "",
    `switching to Nairobi left Cape Town's followed block described: `
    + `"${followNoteText(toggled.host)}"`);
});
