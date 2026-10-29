import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { DemoBlock, DemoBundle } from "../src/demo.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import {
  fakeLocation, FakeElement, fireResize, installStubs, mountPoint, writeNow,
} from "./harness.js";
import {
  BUILD_OUT_URL, buildOut, sameAsBudget, type BuildOutState,
} from "../src/widgets/build-out.js";

installStubs();
// The widget draws its plots through render/svg.ts, which creates elements by namespace; the
// shared harness's `document` only knows `createElement` (svg.test.ts adds the same stub).
(globalThis as unknown as { document: Record<string, unknown> }).document.createElementNS =
  (_ns: string, tag: string): FakeElement => new FakeElement(tag);

const committed = JSON.parse(
  readFileSync("../examples/demo/demo.json", "utf8")) as DemoBundle;
const SRC = "../assets/demo/demo.json";
/** The page's name for L_P, handed over as `data-metric-label`; deliberately not the real one, so
 * a label the widget spelled itself would show up as a mismatch. */
const METRIC = "ease of access";
const BLOCK: DemoBlock = committed.block;

let served: DemoBundle = committed;
(globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
  ok: true, status: 200, statusText: "OK",
  json: (): Promise<unknown> => Promise.resolve(served),
});

/** Mounts over the PRODUCTION url store (a fake location), waits out the fetch chain, then plays
 * the first resize -- screen-map-boot.test.ts's `mount`, for this widget. */
async function mount(search = "", bundle: DemoBundle = committed):
    Promise<{ host: FakeElement; store: StateSource<BuildOutState> | null;
              loc: ReturnType<typeof fakeLocation> }> {
  served = bundle;
  const host = mountPoint();
  host.dataset.bundle = SRC;
  host.dataset.metricLabel = METRIC;
  const loc = fakeLocation(search);
  const slot = urlStore(loc, writeNow).reserve();
  let store: StateSource<BuildOutState> | null = null;
  buildOut(host as never, (initial) => {
    store = slot.bind(BUILD_OUT_URL, initial);
    return store;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  if (store !== null) fireResize(600, 400);
  return { host, store, loc };
}

const plate = (host: FakeElement): FakeElement => {
  const img = host.findAll("img").at(-1);
  assert.ok(img !== undefined, "no plate image");
  return img;
};
const slider = (host: FakeElement): FakeElement => {
  const el = host.findAll("input").find((i) => i.type === "range");
  assert.ok(el !== undefined, "no budget slider");
  return el;
};
const readouts = (host: FakeElement): string[] => host.findAll("dd").map((d) => d.textContent);
const circles = (svg: FakeElement, cls: string): FakeElement[] =>
  svg.findAll("circle").filter((c) => c.getAttribute("class") === cls);
const shift = (blk: DemoBlock, k: number): string =>
  `translateY(${(-100 * blk.frames[k]!.row) / blk.sheet.rows}%)`;

test("it boots on the first block's default frame and removes the fallback", async () => {
  const { host } = await mount();
  const d = committed.method.default_frame;
  const f = BLOCK.frames[d]!;
  assert.equal(host.findAll("img").length, 1, "the fallback image is still there");
  assert.equal(plate(host).getAttribute("src"), `../assets/demo/${BLOCK.sheet.file}`);
  assert.equal(plate(host).style.transform, shift(BLOCK, d));
  assert.equal(slider(host).value, String(d));
  assert.deepEqual(readouts(host).slice(0, 3), [
    `${(f.displacement * 100).toFixed(1)}%`, `${f.road_km.toFixed(1)} km`,
    `${(f.permeability * 100).toFixed(1)}%`,
  ]);
});

test("a slider step moves the plate, the readouts and every plot's dot to that frame", async () => {
  const { host } = await mount();
  const k = BLOCK.frames.length - 1;
  const f = BLOCK.frames[k]!;
  slider(host).value = String(k);
  slider(host).dispatch("input");
  assert.equal(plate(host).style.transform, shift(BLOCK, k));
  assert.equal(readouts(host)[2], `${(f.permeability * 100).toFixed(1)}%`);
  const svgs = host.findAll("svg");
  assert.equal(svgs.length, 3, "three plots");
  for (const svg of svgs) {
    // Each frame of the first block is its own network, so its k-th marker is frame k's point.
    const marker = circles(svg, "sbu-curve")[k]!;
    const [dot] = circles(svg, "sbu-cursor");
    assert.ok(dot !== undefined, "no cursor dot");
    assert.equal(dot.getAttribute("cx"), marker.getAttribute("cx"));
    assert.equal(dot.getAttribute("cy"), marker.getAttribute("cy"));
  }
});

test("a network shared by several budgets is one dot, and the readout says whose it is",
     async () => {
  // 5810's every budget is its own network, so share one here: positions 1-3 on one frame.
  const shared = structuredClone(committed);
  const frames = shared.block.frames;
  for (const k of [2, 3]) frames[k] = { ...frames[1]!, budget: frames[k]!.budget };
  const distinct = new Set(frames.map((f) => f.row)).size;
  const { host } = await mount("", shared);
  for (const svg of host.findAll("svg")) {
    assert.equal(circles(svg, "sbu-curve").length, distinct, "one dot per network, not per budget");
  }
  slider(host).value = "3";
  slider(host).dispatch("input");
  assert.equal(sameAsBudget(frames, 3), frames[1]!.budget);
  assert.match(host.text(), new RegExp(
    `same network as at the ${Math.round(frames[1]!.budget * 100)}% budget`));
  slider(host).value = "1";
  slider(host).dispatch("input");
  assert.doesNotMatch(host.text(), /same network/);
});

test("the metric is named as the page names it, never by the widget itself", async () => {
  const { host } = await mount();
  const text = host.text();
  assert.ok(text.includes("Ease of access against displacement"), text);
  assert.ok(!/[Pp]ermeability/.test(text), "the widget spelled the metric's name itself");
  assert.equal(host.findAll("button").length, 0, "there is one view; nothing to switch");
});

test("a URL budget past the last frame is pinned to it and written back", async () => {
  const last = BLOCK.frames.length - 1;
  const { host, loc } = await mount("?budget=999");
  assert.equal(slider(host).value, String(last));
  assert.ok(loc.search().includes(`budget=${last}`), loc.search());
});

test("a frame whose sprite row is not in the sheet fails into the caption", async () => {
  const bad = structuredClone(committed);
  bad.block.frames[1]!.row = bad.block.sheet.rows;
  const { host } = await mount("", bad);
  const caption = host.find("figcaption");
  assert.ok(caption !== null);
  assert.match(caption.textContent, /could not load interactively.*sprite row/);
  assert.equal(host.findAll("img").length, 1, "the fallback must stay when boot failed");
});
