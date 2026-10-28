import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { DemoBundle } from "../src/demo.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import {
  fakeLocation, FakeElement, fireResize, installStubs, mountPoint, writeNow,
} from "./harness.js";
import { BUILD_OUT_URL, buildOut, type BuildOutState } from "../src/widgets/build-out.js";

installStubs();
// The widget draws its plots through render/svg.ts, which creates elements by namespace; the
// shared harness's `document` only knows `createElement` (svg.test.ts adds the same stub).
(globalThis as unknown as { document: Record<string, unknown> }).document.createElementNS =
  (_ns: string, tag: string): FakeElement => new FakeElement(tag);

const committed = JSON.parse(
  readFileSync("../examples/demo/demo.json", "utf8")) as DemoBundle;
const SRC = "../assets/demo/demo.json";

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

test("it boots on the bundle's default frame, network view, and removes the fallback", async () => {
  const { host } = await mount();
  const d = committed.method.default_frame;
  const f = committed.method.frames[d]!;
  assert.equal(host.findAll("img").length, 1, "the fallback image is still there");
  assert.equal(plate(host).getAttribute("src"), `../assets/demo/${committed.sheets.network}`);
  assert.equal(plate(host).style.transform, `translateY(${(-100 * f.row) / committed.sheets.rows}%)`);
  assert.equal(slider(host).value, String(d));
  assert.deepEqual(readouts(host).slice(0, 3), [
    `${(f.displacement * 100).toFixed(1)}%`, `${f.road_km.toFixed(1)} km`,
    `${(f.permeability * 100).toFixed(1)}%`,
  ]);
});

test("a slider step moves the plate, the readouts and every plot's dot to that frame", async () => {
  const { host } = await mount();
  const k = committed.method.frames.length - 1;
  const f = committed.method.frames[k]!;
  slider(host).value = String(k);
  slider(host).dispatch("input");
  assert.equal(plate(host).style.transform, `translateY(${(-100 * f.row) / committed.sheets.rows}%)`);
  assert.equal(readouts(host)[2], `${(f.permeability * 100).toFixed(1)}%`);
  const svgs = host.findAll("svg");
  assert.equal(svgs.length, 3, "three plots");
  for (const svg of svgs) {
    // The curve's k-th marker is frame k's point; the dot must sit exactly on it.
    const marker = circles(svg, "sbu-curve")[k]!;
    const [dot] = circles(svg, "sbu-cursor");
    assert.ok(dot !== undefined, "no cursor dot");
    assert.equal(dot.getAttribute("cx"), marker.getAttribute("cx"));
    assert.equal(dot.getAttribute("cy"), marker.getAttribute("cy"));
  }
});

test("the view switch swaps the sheet and the legend, and the URL says so", async () => {
  const { host, loc } = await mount();
  // `className` is set by the widget; the harness's element does not declare it.
  const legends = host.descendants().filter(
    (e) => (e as unknown as { className?: string }).className === "sbu-buildout__legend");
  assert.equal(legends.length, 3, "one legend per view");
  const button = host.findAll("button").find((b) => b.textContent === "Escape time");
  assert.ok(button !== undefined);
  button.dispatch("click");
  assert.equal(plate(host).getAttribute("src"), `../assets/demo/${committed.sheets.time}`);
  assert.deepEqual(legends.map((l) => (l as unknown as { hidden: boolean }).hidden),
                   [true, false, true]);
  assert.equal(loc.written.at(-1), "view=time");
});

test("a URL budget past the last frame is pinned to it and written back", async () => {
  const last = committed.method.frames.length - 1;
  const { host, loc } = await mount("?budget=999&view=saved");
  assert.equal(slider(host).value, String(last));
  assert.equal(plate(host).getAttribute("src"), `../assets/demo/${committed.sheets.saved}`);
  assert.ok(loc.search().includes(`budget=${last}`), loc.search());
});

test("a frame whose sprite row is not in the sheets fails into the caption", async () => {
  const bad = structuredClone(committed);
  bad.method.frames[1]!.row = bad.sheets.rows;
  const { host } = await mount("", bad);
  const caption = host.find("figcaption");
  assert.ok(caption !== null);
  assert.match(caption.textContent, /could not load interactively.*sprite row/);
  assert.equal(host.findAll("img").length, 1, "the fallback must stay when boot failed");
});
