import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { DemoBuildOut, DemoSeries } from "../src/demo.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import type { UrlCodec } from "../src/url/param.js";
import {
  fakeLocation, FakeElement, fireResize, installStubs, mountPoint, writeNow,
} from "./harness.js";
import {
  BUILD_OUT_COMPARE_URL, BUILD_OUT_URL, buildOut, duration, measuredTime, sameAsBudget,
  sheetOffset,
  type BuildOutState,
} from "../src/widgets/build-out.js";

installStubs();
// The widget draws its plots through render/svg.ts, which creates elements by namespace; the
// shared harness's `document` only knows `createElement` (svg.test.ts adds the same stub).
(globalThis as unknown as { document: Record<string, unknown> }).document.createElementNS =
  (_ns: string, tag: string): FakeElement => new FakeElement(tag);

const read = (name: string): DemoBuildOut =>
  JSON.parse(readFileSync(`../examples/demo/${name}`, "utf8")) as DemoBuildOut;
/** The two build-outs the page carries, as committed: the Demo's block (one series) and the
 * methods compared on another (two). */
const PAGE = read("buildout_ZAF.9.3.1_1_5810.json");
const COMPARED = read("buildout_ZAF.9.3.1_1_41785.json");
const DIR = "../assets/demo/";
/** The page's name for L_P, handed over as `data-metric-label`; deliberately not the real one, so
 * a label the widget spelled itself would show up as a mismatch. */
const METRIC = "ease of access";

let served: DemoBuildOut = PAGE;
(globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
  ok: true, status: 200, statusText: "OK",
  json: (): Promise<unknown> => Promise.resolve(served),
});

/** Mounts over the PRODUCTION url store (a fake location), waits out the fetch chain, then plays
 * the first resize -- screen-map-boot.test.ts's `mount`, for this widget. */
async function mount(bundle: DemoBuildOut, search = "",
                     codec: UrlCodec<BuildOutState> = BUILD_OUT_URL, width = 600):
    Promise<{ host: FakeElement; store: StateSource<BuildOutState> | null;
              loc: ReturnType<typeof fakeLocation> }> {
  served = bundle;
  const host = mountPoint();
  host.dataset.bundle = `${DIR}buildout.json`;
  host.dataset.metricLabel = METRIC;
  const loc = fakeLocation(search);
  const slot = urlStore(loc, writeNow).reserve();
  let store: StateSource<BuildOutState> | null = null;
  buildOut(host as never, (initial) => {
    store = slot.bind(codec, initial);
    return store;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  if (store !== null) fireResize(width, 400);
  return { host, store, loc };
}
const compared = (search = ""): ReturnType<typeof mount> =>
  mount(COMPARED, search, BUILD_OUT_COMPARE_URL, 900);

/** The plate images, one per series, in series order. */
const plates = (host: FakeElement): FakeElement[] =>
  host.findAll("img").filter((i) => i.getAttribute("src")?.endsWith(".webp") ?? false);
const slider = (host: FakeElement): FakeElement => {
  const el = host.findAll("input").find((i) => i.type === "range");
  assert.ok(el !== undefined, "no budget slider");
  return el;
};
const step = (host: FakeElement, k: number): void => {
  slider(host).value = String(k);
  slider(host).dispatch("input");
};
/** Each column's four readouts. */
const readouts = (host: FakeElement): string[][] =>
  host.findAll("dl").map((dl) => dl.findAll("dd").map((d) => d.textContent));
const notes = (host: FakeElement): string[] =>
  host.findAll("p").filter((e) => e.className === "sbu-buildout__same")
    .map((e) => e.textContent);
const byClass = (host: FakeElement, cls: string): string[] =>
  host.findAll("p").filter((e) => e.className === cls).map((e) => e.textContent);
const circles = (svg: FakeElement, cls: string): FakeElement[] =>
  svg.findAll("circle").filter((c) => c.getAttribute("class") === cls);
const shift = (s: DemoSeries, k: number): string => {
  const { x, y } = sheetOffset(s.sheet, s.cells[k]!.step.row);
  return `translate(${x}%, ${y}%)`;
};
const pct1 = (v: number): string => `${(v * 100).toFixed(1)}%`;
const isTime = (svg: FakeElement): boolean =>
  (svg.getAttribute("aria-label") ?? "").includes("against run time");

test("one series boots on its default budget and removes the fallback", async () => {
  const { host } = await mount(PAGE);
  const d = PAGE.default_step;
  const s = PAGE.series[0]!;
  const f = s.cells[d]!.step;
  assert.equal(host.findAll("img").length, 1, "the fallback image is still there");
  assert.equal(plates(host)[0]!.getAttribute("src"), `${DIR}${s.sheet.file}`);
  assert.equal(plates(host)[0]!.style.transform, shift(s, d));
  assert.equal(slider(host).value, String(d));
  assert.deepEqual(readouts(host)[0]!.slice(0, 3), [
    pct1(f.displacement), `${f.road_km.toFixed(2)} km`, pct1(f.permeability),
  ]);
  assert.deepEqual(byClass(host, "sbu-buildout__head"), [], "one series has no heading");
});

test("a slider step moves the plate, the readouts and every plot's dot to that budget",
     async () => {
  const { host } = await mount(PAGE);
  const s = PAGE.series[0]!;
  const k = PAGE.budgets.length - 1;
  const f = s.cells[k]!.step;
  step(host, k);
  assert.equal(plates(host)[0]!.style.transform, shift(s, k));
  assert.equal(readouts(host)[0]![2], pct1(f.permeability));
  assert.equal(readouts(host)[0]![3], duration(f.seconds!));
  const svgs = host.findAll("svg");
  assert.equal(svgs.length, PAGE.plots.length, "a plot per kind the build-out names");
  for (const svg of svgs) {
    // Each of 5810's budgets is its own network, so its row-th marker is this budget's point.
    const marker = circles(svg, "sbu-curve sbu-series-0")[f.row]!;
    const [dot] = circles(svg, "sbu-cursor sbu-series-0");
    assert.ok(dot !== undefined, "no cursor dot");
    assert.equal(dot.getAttribute("cx"), marker.getAttribute("cx"));
    assert.equal(dot.getAttribute("cy"), marker.getAttribute("cy"));
  }
});

test("two series: a headed plate each, both moved by the one slider", async () => {
  const { host } = await compared();
  assert.deepEqual(byClass(host, "sbu-buildout__head"), COMPARED.series.map((s) => s.label));
  assert.deepEqual(byClass(host, "sbu-buildout__time"), COMPARED.series.map((s) => (
    s.run.kind === "measured" ? measuredTime(s.run.seconds) : duration(s.run.seconds))));
  assert.ok(COMPARED.series.some((s) => s.run.kind === "measured"), "no measured run to check");
  assert.deepEqual(plates(host).map((i) => i.getAttribute("src")),
                   COMPARED.series.map((s) => `${DIR}${s.sheet.file}`));
  const k = 3;
  step(host, k);
  COMPARED.series.forEach((s, i) => {
    assert.equal(plates(host)[i]!.style.transform, shift(s, k), s.method);
    assert.equal(readouts(host)[i]![0], pct1(s.cells[k]!.step.displacement), s.method);
  });
  for (const svg of host.findAll("svg")) {
    COMPARED.series.forEach((s, i) => {
      assert.equal(circles(svg, `sbu-cursor sbu-series-${i}`).length, 1, `${s.method}'s dot`);
    });
  }
});

test("the second build-out's slider owns its own URL key", async () => {
  const { host, loc } = await compared("?buildout=2&compare=4");
  assert.equal(slider(host).value, "4");
  step(host, 6);
  assert.ok(loc.search().includes("compare=6"), loc.search());
  assert.ok(loc.search().includes("buildout=2"), "the page block's key was touched");
});

test("a stopped method shows its whole network and says why, short of the budget", async () => {
  const { host } = await compared();
  const i = COMPARED.series.findIndex((s) => s.cells.some((c) => c.kind === "short"));
  assert.ok(i >= 0, "no method stops in the committed build-out");
  const s = COMPARED.series[i]!;
  const k = s.cells.findIndex((c) => c.kind === "short");
  const c = s.cells[k]!;
  assert.ok(c.kind === "short");
  step(host, k);
  const note = notes(host)[i]!;
  assert.ok(note.startsWith(`${s.method} stops at ${Math.round(c.step.displacement * 100)}%: `
    + `${c.why}.`), note);
  assert.ok(note.includes(`short of the ${Math.round(COMPARED.budgets[k]! * 100)}% budget`), note);
  assert.equal(readouts(host)[i]![0], pct1(c.step.displacement));
  // Its dot rests on its curve's last point: the curve ends where the method stopped.
  for (const svg of host.findAll("svg")) {
    const markers = circles(svg, `sbu-curve sbu-series-${i}`);
    const [dot] = circles(svg, `sbu-cursor sbu-series-${i}`);
    assert.equal(dot!.getAttribute("cx"), markers.at(-1)!.getAttribute("cx"));
  }
});

test("each method's Built in is its run's time to the network shown", async () => {
  const { host } = await compared();
  const k = 3;
  step(host, k);
  COMPARED.series.forEach((s, i) => {
    assert.equal(readouts(host)[i]![3], duration(s.cells[k]!.step.seconds), s.method);
  });
});

test("a network shared by several budgets is one dot, and the note says whose it is",
     async () => {
  // Our method's first loop overshoots several budgets: they share its network.
  const { host } = await compared();
  const s = COMPARED.series[0]!;
  const k = s.cells.findIndex((c, j) => j > 0 && c.kind === "met"
                                         && c.step.row === s.cells[j - 1]!.step.row);
  assert.ok(k > 0, "no shared network in the committed build-out");
  const distinct = new Set(s.cells.map((c) => c.step.row)).size;
  for (const svg of host.findAll("svg")) {
    // Run time has no point for the block with no new road (no time, no place on a log scale).
    const timed = isTime(svg) ? 1 : 0;
    assert.equal(circles(svg, "sbu-curve sbu-series-0").length, distinct - timed,
                 "one dot per network, not per budget");
  }
  step(host, k);
  const was = sameAsBudget(COMPARED.budgets, s.cells, k);
  assert.ok(was !== null);
  assert.ok(notes(host)[0]!.includes(`same network as at the ${Math.round(was * 100)}% budget`),
            notes(host)[0]);
  step(host, 1);
  assert.equal(notes(host)[0], "");
});

test("a frame past the sheet's first column is read across, then down", () => {
  const sheet = { file: "x.webp", width: 600, frame_height: 800, frames: 5, columns: 2 };
  assert.deepEqual(sheetOffset(sheet, 0), { x: -0, y: -0 });
  assert.deepEqual(sheetOffset(sheet, 3), { x: -50, y: -100 / 3 });
  assert.deepEqual(sheetOffset(sheet, 4), { x: -0, y: -200 / 3 });
});

test("the metric is named as the page names it, never by the widget itself", async () => {
  const { host } = await mount(PAGE);
  const text = host.text();
  assert.ok(text.includes("Ease of access against displacement"), text);
  assert.ok(!/[Pp]ermeability/.test(text), "the widget spelled the metric's name itself");
});

test("a URL budget past the last is pinned to it and written back", async () => {
  const last = PAGE.budgets.length - 1;
  const { host, loc } = await mount(PAGE, "?buildout=999");
  assert.equal(slider(host).value, String(last));
  assert.ok(loc.search().includes(`buildout=${last}`), loc.search());
});

test("a step whose frame is not in its sheet fails into the caption", async () => {
  const bad = structuredClone(PAGE);
  bad.series[0]!.cells[1]!.step.row = bad.series[0]!.sheet.frames;
  const { host } = await mount(bad);
  const caption = host.find("figcaption");
  assert.ok(caption !== null);
  assert.match(caption.textContent, /could not load interactively.*outside its sheet/);
  assert.equal(host.findAll("img").length, 1, "the fallback must stay when boot failed");
});

test("run times read as the page's generator writes them", () => {
  assert.equal(duration(56), "56 seconds");
  assert.equal(duration(6762.5), "1.9 hours");
  assert.equal(duration(7.44), "7.4 seconds");
});

test("a measured time reads as the page's generator states it", () => {
  // The cases the generator's `_measured_time` is tested against too (tests/test_gen_site_pages.py).
  const fixture = JSON.parse(readFileSync("test/fixtures/measured-times.json", "utf8")) as
    { cases: [number, string][] };
  assert.ok(fixture.cases.length > 0);
  for (const [s, want] of fixture.cases) assert.equal(measuredTime(s), want, String(s));
});

test("the face-off plots run time on a log scale, two series, dots following the slider",
     async () => {
  const { host } = await compared();
  const svgs = host.findAll("svg");
  assert.deepEqual(COMPARED.plots, ["displacement", "road", "time"]);
  assert.equal(svgs.length, 3);
  const time = svgs.find(isTime);
  assert.ok(time !== undefined, "no run-time plot");
  // Human units on the axis.
  const labels = time.findAll("text").map((t) => t.textContent);
  for (const want of ["1 min", "1 h"]) assert.ok(labels.includes(want), `${want} not in ${labels}`);
  // At a budget where both have a network, each dot sits on its series' point for that network,
  // and the axis is logarithmic: equal factors of time are equal distances.
  const k = 3;
  step(host, k);
  COMPARED.series.forEach((s, i) => {
    const f = s.cells[k]!.step;
    const pts = [...new Map(s.cells.filter((c) => c.step.seconds > 0)
      .map((c) => [c.step.row, c.step])).values()];
    const at = pts.findIndex((q) => q.row === f.row);
    const marker = circles(time, `sbu-curve sbu-series-${i}`)[at]!;
    const [dot] = circles(time, `sbu-cursor sbu-series-${i}`);
    assert.equal(dot!.getAttribute("cx"), marker.getAttribute("cx"), s.method);
    assert.notEqual(dot!.getAttribute("visibility"), "hidden");
  });
  const xs = COMPARED.series.map((s) => s.cells[k]!.step.seconds);
  const cx = COMPARED.series.map((_, i) => Number(circles(time, `sbu-cursor sbu-series-${i}`)[0]!
    .getAttribute("cx")));
  const [lo, hi] = ["10 s", "1 h"].map((l) => Number(time.findAll("text")
    .find((t) => t.textContent === l)!.getAttribute("x")));
  const perDecade = (hi! - lo!) / Math.log10(360);
  assert.ok(Math.abs((cx[1]! - cx[0]!) - perDecade * Math.log10(xs[1]! / xs[0]!)) < 0.5,
            "the run-time axis is not logarithmic");
  // The block with no new road took no time: no dot on run time.
  step(host, 0);
  COMPARED.series.forEach((_, i) => {
    assert.equal(circles(time, `sbu-cursor sbu-series-${i}`)[0]!.getAttribute("visibility"),
                 "hidden");
  });
});

test("the Demo's own build-out plots no run time", async () => {
  const { host } = await mount(PAGE);
  assert.deepEqual(PAGE.plots, ["displacement", "road"]);
  assert.ok(!host.findAll("svg").some(isTime));
});
