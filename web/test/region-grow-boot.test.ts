import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { HoodBlock, HoodBundle } from "../src/hood.js";
import { grow, growth } from "../src/model/accretion.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import { fitBbox, toScreen, type Bbox } from "../src/view/transform.js";
import {
  armDrawFailure, canvasOf, Call, fakeLocation, fireResize, installStubs, lastFrame, mountPoint,
  writeNow,
} from "./harness.js";
import { REGION_GROW_URL, regionGrow, type RegionGrowState } from "../src/widgets/region-grow.js";

installStubs();

const bundle = JSON.parse(
  readFileSync("../examples/region-grow/hood.json", "utf8")) as HoodBundle;
const E = bundle.encoding;

// `fetch` is not part of the shared harness (see harness.ts's own docstring for exactly what it
// exports) -- displacement-field's, perm-graph's and frontier's own boot tests each stub it
// locally too. One static stub suffices here: every test in this file mounts the same committed
// bundle, and none exercises a fetch failure or a different payload.
(globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
  ok: true,
  status: 200,
  statusText: "OK",
  json: (): Promise<unknown> => Promise.resolve(bundle),
});

/** Every neighbourhood block's own context outline -- layer (1), drawn first, in `hood_color` at
 * `hood_lw`. Same colour-first reasoning as the other layer helpers below. */
function hoodPaths(cv: unknown): Call[] {
  return lastFrame(cv as never).filter((c) => c.op === "stroke" && c.strokeStyle === E.hood_color);
}

/** Blocks the picture currently shows as REGION -- identified by the bundle's own fill colour,
 * never by a path count. D2's defect #1 was a layer identified by count, in a figure where two
 * layers happened to have the same number of paths; and defect #2 matched a partial alpha that
 * belonged to a different layer entirely. Name the layer by its colour, then assert on it. */
function regionPaths(cv: unknown): Call[] {
  return lastFrame(cv as never).filter((c) => c.op === "fill" && c.fillStyle === E.region_color);
}

/** Blocks the picture currently shows as FRONTIER -- same colour-first reasoning as
 * `regionPaths`, stroked rather than filled (layer 3 is an outline, never a fill). */
function frontierPaths(cv: unknown): Call[] {
  return lastFrame(cv as never)
    .filter((c) => c.op === "stroke" && c.strokeStyle === E.frontier_color);
}

/** The seed's own outline -- there is exactly one seed, so exactly one such stroke. */
function seedPaths(cv: unknown): Call[] {
  return lastFrame(cv as never).filter((c) => c.op === "stroke" && c.strokeStyle === E.seed_color);
}

/** Blocks the picture strokes as REGION -- `region.ts`'s own layer (2) now strokes every accreted
 * block in `region_color` at `region_lw`, matching `hood.png`'s `region.plot(..., edgecolor=
 * region_color, linewidth=region_lw)`. Same colour-first reasoning as `regionPaths`/`frontierPaths`,
 * kept as its own function rather than folded into `frontierPaths` so a mismatch between the two
 * stroke colours (`region_color` vs. `frontier_color`) cannot silently merge them into one count. */
function regionStrokes(cv: unknown): Call[] {
  return lastFrame(cv as never)
    .filter((c) => c.op === "stroke" && c.strokeStyle === E.region_color);
}

/** The `aria-live="polite"` element's own text -- found by that attribute, not merely by tag, so
 * this fails loudly if the readout is ever rendered without it rather than silently reading some
 * OTHER paragraph. Mirrors screen-map-boot.test.ts's own `readoutText`; `ScreenMap` got this guard
 * in an earlier fix round and `RegionGrow` never did. */
function readoutText(host: ReturnType<typeof mountPoint>): string {
  const live = host.descendants().find((el) => el.getAttribute("aria-live") === "polite");
  assert.ok(live !== undefined, `there is no aria-live="polite" readout`);
  return live.textContent;
}

/** Mounts the widget and waits for its fetch chain to settle BEFORE playing the first resize --
 * exactly perm-graph-boot.test.ts's `mount` shape (field-boot.test.ts's and frontier-boot.test.ts's
 * own `mount` helpers agree). `regionGrow(...)` only starts a promise chain; it does not run
 * `boot()` -- and so does not construct a canvas or a ResizeObserver -- until that chain's `.then`s
 * have had a turn, which happens no sooner than the next macrotask (checked empirically, not
 * assumed). Calling `fireResize` before that drain finds no observer to fire.
 *
 * `drawFailure` arms `armDrawFailure` (harness.ts) BEFORE `regionGrow` runs, so the canvas `boot()`
 * creates during the awaited macrotask below captures it as its `RecordingContext.failWith` --
 * perm-graph-boot.test.ts's own `mount` does the same, for the same timing reason. Cleared in a
 * `finally` after `fireResize` so a later call to `mount()` with no `drawFailure` is not
 * accidentally armed by a previous test's leftover state.
 *
 * The state store is the PRODUCTION one (`urlStore` over a `fakeLocation`), never `localState`:
 * `search` defaults to "" -- an empty URL, so nothing is decoded and nothing is written, and a
 * caller that passes no search gets the widget's own initial state exactly as `localState` gave
 * it. (The store CLAIMS every key of the codec regardless of what the URL carries; what an
 * empty query skips is the decode, not the claim.) The URL tests below then get the real
 * decode path rather than a second, test-only one. */
async function mount(width = 700, drawFailure: string | null = null, search = ""):
    Promise<{ host: ReturnType<typeof mountPoint>; cv: unknown;
              store: StateSource<RegionGrowState>; loc: ReturnType<typeof fakeLocation> }> {
  const host = mountPoint();
  host.dataset.bundle = "../examples/region-grow/hood.json";
  armDrawFailure(drawFailure);
  const loc = fakeLocation(search);
  const slot = urlStore(loc, writeNow).reserve();
  let bound: StateSource<RegionGrowState> | null = null;
  regionGrow(host as never, (initial) => {
    bound = slot.bind(REGION_GROW_URL, initial);
    return bound;
  });
  // A macrotask, so the fetch chain has drained -- boot() has run and the canvas exists -- by the
  // time this resolves.
  await new Promise((resolve) => setTimeout(resolve, 0));
  try {
    fireResize(width, width);
  } finally {
    armDrawFailure(null);
  }
  // Every mount in this file feeds `boot` the committed bundle, whose own `seed` is one of its own
  // blocks, so `boot` always reaches `makeState`. A null here is a widget that failed before it.
  assert.ok(bound !== null, "the widget never asked for a state store");
  return { host, cv: canvasOf(host), store: bound, loc };
}

/** Finds the `<input type="range">` the widget wrote -- the one place that lookup happens, so
 * `setSlider` and the bounds test below cannot drift onto two different selectors. */
function budgetSlider(host: ReturnType<typeof mountPoint>): ReturnType<typeof mountPoint> {
  const slider = host.findAll("input").find((i) => i.type === "range");
  assert.ok(slider !== undefined, "there is no budget slider");
  return slider;
}

/** Drives the budget slider the way a reader's pointer or keyboard would: set `.value`, then
 * dispatch the `"input"` event the widget listens for. Same shape as field-boot.test.ts's own
 * width-slider helper, not its selector -- this widget writes exactly one range input, the
 * budget. */
function setSlider(host: ReturnType<typeof mountPoint>, value: number): void {
  const slider = budgetSlider(host);
  slider.value = String(value);
  slider.dispatch("input");
}

/** The bbox every block's rings must fit inside -- derived independently here, not imported from
 * the widget, so a widget that fit the wrong extent would be caught rather than confirmed by its
 * own arithmetic (the same reasoning as field-boot.test.ts's own `unionBbox`). Exterior AND
 * interior rings, matching `region-grow.ts`'s own `hoodBbox`: a hole is still geometry the block
 * occupies on screen even though nothing is drawn inside it. */
function hoodBbox(blocks: HoodBlock[]): Bbox {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const b of blocks) for (const ring of b.rings) for (const [x, y] of ring) {
    xs.push(x);
    ys.push(y);
  }
  return { minX: Math.min(...xs), minY: Math.min(...ys),
           maxX: Math.max(...xs), maxY: Math.max(...ys) };
}
const SIZE = 700;
const VIEW = fitBbox(hoodBbox(bundle.blocks), SIZE, SIZE, E.pad);

/** A point guaranteed inside `block`'s exterior ring: the midpoint of the first even-odd crossing
 * interval on a horizontal ray through the ring's own mean y. Robust for a concave ring, where a
 * vertex or a plain vertex-average centroid is not guaranteed to land inside at all. */
function interiorPoint(block: HoodBlock): [number, number] {
  const ring = block.rings[0]!;
  const y = ring.reduce((sum, [, ry]) => sum + ry, 0) / ring.length;
  const xs: number[] = [];
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]!;
    const [xj, yj] = ring[j]!;
    if ((yi > y) !== (yj > y)) xs.push(xi + (y - yi) * (xj - xi) / (yj - yi));
  }
  xs.sort((a, z) => a - z);
  assert.ok(xs.length >= 2, `block ${block.block_id} has no interior at its own mean y`);
  return [(xs[0]! + xs[1]!) / 2, y];
}

/** Dispatches a `pointerdown` at the screen position of a point known to be inside `block`,
 * projected through the SAME `fitBbox`/`toScreen` the widget builds its own view from (`VIEW`
 * above), at the `mount()` default width every caller of this helper mounts at -- so the click
 * lands exactly where the widget's own `toWorld`/`blockAt` hit test expects it. */
function clickBlock(cv: unknown, block: HoodBlock): void {
  const [wx, wy] = interiorPoint(block);
  const [sx, sy] = toScreen(VIEW, wx, wy);
  (cv as { dispatch: (name: string, ev: unknown) => void })
    .dispatch("pointerdown", { offsetX: sx, offsetY: sy, pointerId: 1 });
}

test("the neighbourhood context is stroked at the bundle's own hood_lw, not a literal", async () => {
  // The same JavaScript-on/off width divergence `region_lw` (finding 4) and D2's own `street_lw`
  // already cost this project, on the one width in this file's draw() that was still never
  // asserted: layer (1)'s `hood_lw`. Setting it to `region_lw` (0.4 -> 1.3 across all 213 blocks)
  // leaves the rest of this suite green.
  const { cv } = await mount();
  const strokes = hoodPaths(cv);
  assert.equal(strokes.length, bundle.blocks.length,
    "every neighbourhood block should be outlined once, in the context layer");
  for (const c of strokes) assert.equal(c.lineWidth, E.hood_lw);
});

test("the region drawn at the default budget is the one the model computes", async () => {
  const { cv } = await mount();
  const expected = grow(bundle.blocks,
    bundle.blocks.findIndex((b) => b.block_id === bundle.seed), bundle.budget.default);
  assert.equal(regionPaths(cv).length, expected.length);
});

test("the region fill is drawn at the bundle's own alpha, not full opacity", async () => {
  // `gen_region_grow.py`'s `_render_hood` draws the region fill at `region_alpha` in the fallback
  // PNG; a widget drawing it at `globalAlpha` 1 would be a live JS-on/JS-off divergence of exactly
  // the kind `street_lw: 1.0` vs. a PNG drawn at 1.3 already cost this project once, undetected.
  const { cv } = await mount();
  const paths = regionPaths(cv);
  assert.ok(paths.length > 0, "no region fill to check the alpha of");
  for (const c of paths) assert.equal(c.globalAlpha, E.region_alpha);
});

test("the region fill cuts interior rings out (evenodd), rather than filling them solid", async () => {
  // screen-map-boot.test.ts:326 asserts this for `city.ts`'s own base layer; `region.ts`'s region
  // layer draws through the SAME `fillBlock` shape but never got the twin assertion. 7 of
  // hood.json's 213 blocks carry interior rings, and any one of them could be in the default-budget
  // region -- a nonzero-winding fill would paint straight over the hole.
  const { cv } = await mount();
  const paths = regionPaths(cv);
  assert.ok(paths.length > 0, "no region fill to check the fill rule of");
  for (const c of paths) assert.equal(c.fillRule, "evenodd");
});

test("the region is stroked in its own colour at region_lw, not merely filled", async () => {
  // `hood.png`'s own `region.plot(..., edgecolor=region_color, linewidth=region_lw,
  // alpha=region_alpha)` strokes every accreted block; a fill-only widget draws JS-off's
  // individually-outlined blocks as a single indistinguishable blob. `region_lw` was otherwise
  // never asserted anywhere in this suite.
  const { cv } = await mount();
  const fillCount = regionPaths(cv).length;
  const strokes = regionStrokes(cv);
  assert.equal(strokes.length, fillCount,
    "the region should be stroked exactly once per block, matching the fill count");
  for (const c of strokes) {
    assert.equal(c.lineWidth, E.region_lw);
    assert.equal(c.globalAlpha, E.region_alpha);
  }
});

test("at the slider floor the region is the seed alone", async () => {
  // The design's §1.3 finding, published rather than hidden. If this stops holding, the widget's
  // caption is wrong.
  const { host, cv } = await mount();
  setSlider(host, bundle.budget.min);
  assert.equal(regionPaths(cv).length, 1);
});

test("the budget slider's bounds come from the bundle, not a literal", async () => {
  // A real `<input type=range>` clamps `.value` to `[min, max]`; the fake DOM this suite mounts
  // against does not (`setSlider` writes `.value` directly), so nothing else here would notice a
  // wrong bound. Mirrors field-boot.test.ts's own floor test for the same reason: that is a real
  // gap the fake DOM leaves open, and a direct attribute assertion is what closes it.
  const { host } = await mount();
  const slider = budgetSlider(host);
  assert.equal(Number(slider.min), bundle.budget.min);
  assert.equal(Number(slider.max), bundle.budget.max);
  assert.equal(Number(slider.step), bundle.budget.step);
});

test("the frontier is drawn one stroke per block adjacent to the region and not in it", async () => {
  // Independently derived from `blocks[i].adj`, not by calling `region-grow.ts`'s own `frontierOf`
  // -- the same reasoning `hoodBbox`/`VIEW` above already follow. The frontier layer is what makes
  // "greedy" visible rather than merely asserted by the caption; a widget that silently failed to
  // draw it would gut that teaching point while every other test here stayed green.
  const { cv } = await mount();
  const seedIndex = bundle.blocks.findIndex((b) => b.block_id === bundle.seed);
  const region = grow(bundle.blocks, seedIndex, bundle.budget.default);
  const inRegion = new Set(region);
  const frontier = new Set<number>();
  for (const i of region) {
    for (const j of bundle.blocks[i]!.adj) if (!inRegion.has(j)) frontier.add(j);
  }
  assert.equal(frontierPaths(cv).length, frontier.size);
});

test("the seed is stroked in the bundle's own seed colour", async () => {
  const { cv } = await mount();
  assert.equal(seedPaths(cv).length, 1);
});

test("the fallback image survives until the first successful draw", async () => {
  // `observeSize` SKIPS a zero width, so a widget that removed the <img> on canvas insertion
  // would leave a blank figure in a collapsed container. D2 closed this; keep it closed.
  const { host } = await mount(0);
  assert.ok(host.querySelector("img"), "zero width drew nothing, so the PNG must remain");
});

test("the picture still matches the model after a reseed and a budget change", async () => {
  // D2's defect #6: drawing was pinned to the model on the BOOT frame only, so every later frame
  // was unguarded. Assert after each interaction, not just at mount.
  const { host, cv } = await mount();
  clickBlock(cv, bundle.blocks[3]!);
  setSlider(host, 600);
  const seed = 3;
  assert.equal(regionPaths(cv).length, grow(bundle.blocks, seed, 600).length);
});

test("the readout pins the region's own block and building counts, live-announced", async () => {
  // The tenth unguarded canvas-only readout on this branch: `ScreenMap`'s twin got exactly this
  // guard in an earlier fix round; `RegionGrow`'s never did. A canvas carries no accessible text of
  // its own, so this `aria-live="polite"` paragraph -- both its presence (asserted inside
  // `readoutText` itself) and its content (asserted below) -- is the ONLY accessible content this
  // widget has.
  const { host } = await mount();
  const seedIndex = bundle.blocks.findIndex((b) => b.block_id === bundle.seed);
  const expected = growth(bundle.blocks, seedIndex, bundle.budget.default);
  const text = readoutText(host);
  assert.ok(text.includes(`${expected.order.length} blocks in the region`),
    `readout is missing the block count: "${text}"`);
  assert.ok(text.includes(`${expected.buildings} buildings`),
    `readout is missing the building count: "${text}"`);
});

test("a throw on the first draw is reported and keeps the static image", async () => {
  // The resize callback runs from the browser's own dispatch, OUTSIDE the mount's
  // `fetch().then(boot).catch(...)` chain, so a throw in it is an uncaught exception and nothing
  // else -- a blank figure, no message, and a page that still looks laid out -- unless
  // `region-grow.ts`'s own `runOrReport` catches it. field-boot.test.ts's and
  // perm-graph-boot.test.ts's own boot suites both have this guard; neither new widget's did.
  const { host } = await mount(700, "boom on the first draw");

  assert.match(host.querySelector("figcaption")!.textContent,
    /RegionGrow could not load interactively .*boom on the first draw/);
  assert.match(host.querySelector("figcaption")!.textContent,
    /The static image above still applies\./);
  assert.ok(host.querySelector("img") !== null,
    "the fallback image was removed although the drawing that replaces it threw");
});

test("a URL budget reaches the SLIDER, not only the canvas", async () => {
  // The desync this widget's own state exists to prevent, in the one direction nothing covered:
  // `boot` drew from state (so the canvas honoured `?budget=`) while the slider was initialised
  // from the bundle's default, leaving the reader a control that disagreed with the picture beside
  // it. Both halves are asserted, so "the picture drew 5000" is observed rather than assumed.
  const { host, cv } = await mount(700, null, "budget=5000");
  assert.equal(budgetSlider(host).value, "5000",
    "the slider initialised from the bundle default while the picture drew 5000");
  const seedIndex = bundle.blocks.findIndex((b) => b.block_id === bundle.seed);
  assert.equal(regionPaths(cv).length, grow(bundle.blocks, seedIndex, 5000).length,
    "the canvas did not draw the URL's budget either");
});

test("the seed is a block_id, so a re-baked hood cannot silently reseed a published link",
  async () => {
    // An INDEX in a published URL points at a different block after any re-bake that reorders
    // hood.json -- no error, right type, right shape, wrong value. An identity cannot do that: it
    // either still names a block or is refused below.
    const { store, cv } = await mount(700, null, `seed=${bundle.blocks[2]!.block_id}`);
    assert.equal(store.get().seed, bundle.blocks[2]!.block_id);
    // The picture half. Without it a `render` that ignored `state.seed` would still pass here: a
    // throw inside `render` is absorbed by `runOrReport` into a caption this test never reads, so
    // the store assertion alone observes the codec and not the widget.
    assert.equal(regionPaths(cv).length, grow(bundle.blocks, 2, bundle.budget.default).length,
      "the region was not grown from the URL's seed");
  });

test("a seed this hood does not carry falls back to the bundle's own AND leaves the URL",
  async () => {
    // A reader's typo is not a broken artifact: it must land on the default view, not an error
    // card, and the key must disappear rather than be carried forward in a URL they might copy.
    const { loc, store } = await mount(700, null, "seed=NOT_A_BLOCK");
    assert.equal(store.get().seed, bundle.seed);
    assert.equal(loc.written.at(-1), "", "the bad key is gone, not carried");
  });

test("an out-of-range ?budget= lands on the slider's own bound, control and picture agreeing",
  async () => {
    // `intParam` checks that a budget is a non-negative integer and stops there -- this hood's own
    // floor and ceiling are properties of the fetched artifact, so an out-of-range one reaches
    // state and must be bounded here.
    //
    // The picture assertion is load-bearing on the HIGH end only, and measured: unclamped,
    // `budget=999999` draws all 213 blocks where the clamped 10000 draws 54. On the low end the
    // greedy stops at the seed for any budget at or below `min`, so `budget=0` and the clamped 150
    // draw the same single block -- there the state and slider assertions above are what
    // distinguish, which is why they come first.
    for (const [search, expected] of [
      ["budget=999999", bundle.budget.max],
      ["budget=0", bundle.budget.min],
    ] as const) {
      const { host, cv, store } = await mount(700, null, search);
      const seedIndex = bundle.blocks.findIndex((b) => b.block_id === bundle.seed);

      assert.equal(store.get().budget, expected, `${search}: the state`);
      assert.equal(budgetSlider(host).value, String(expected), `${search}: the slider`);
      assert.equal(regionPaths(cv).length,
        grow(bundle.blocks, seedIndex, expected).length, `${search}: the picture`);
    }
  });
