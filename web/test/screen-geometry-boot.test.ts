import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { DemoBundle } from "../src/demo.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import {
  fakeLocation, FakeElement, fireResize, installStubs, lastFrame, mountPoint, writeNow,
} from "./harness.js";
import {
  geometry, ownShape, SCREEN_GEOMETRY_URL, screenGeometry, type GeometryState,
} from "../src/widgets/screen-geometry.js";

installStubs();

const bundle = JSON.parse(readFileSync("../examples/demo/demo.json", "utf8")) as DemoBundle;
const RED_ROWS = 12;
(globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
  ok: true, status: 200, statusText: "OK",
  json: (): Promise<unknown> => Promise.resolve(bundle),
});

/** The shipped screen, written out here from its definition rather than imported, so the widget's
 * arithmetic is checked against the formula and not against itself. */
const screen = (n: number, area: number, perimeter: number): number =>
  (Math.sqrt(n * area) / perimeter) * (n / area);

async function mount(search = ""): Promise<{ host: FakeElement;
                                            store: StateSource<GeometryState> }> {
  const host = mountPoint();
  host.dataset.bundle = "../assets/demo/demo.json";
  host.dataset.redRows = String(RED_ROWS);
  const slot = urlStore(fakeLocation(search), writeNow).reserve();
  let store: StateSource<GeometryState> | null = null;
  screenGeometry(host as never, (initial) => {
    store = slot.bind(SCREEN_GEOMETRY_URL, initial);
    return store;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  fireResize(500, 200);
  assert.ok(store !== null, "the widget never asked for a state store");
  return { host, store };
}

const readouts = (host: FakeElement): string[] => host.findAll("dd").map((d) => d.textContent);

test("each block's own rectangle has its area and perimeter, so the block's own score", () => {
  for (const blk of [bundle.screen.followed, bundle.screen.typical]) {
    const g = geometry(blk.area_m2, ownShape(blk), blk.n);
    assert.ok(Math.abs(g.width * g.height - blk.area_m2) < 1e-6 * blk.area_m2);
    assert.ok(Math.abs(2 * (g.width + g.height) - blk.perimeter_m) < 1e-6 * blk.perimeter_m);
    assert.ok(Math.abs(g.score - blk.score) < 1e-9, `${g.score} vs ${blk.score}`);
    assert.ok(Math.abs(g.score - screen(blk.n, blk.area_m2, blk.perimeter_m)) < 1e-12);
  }
});

test("the factors multiply to the score: parcels deep times buildings per area", () => {
  const blk = bundle.screen.followed;
  const g = geometry(blk.area_m2, 4, blk.n);
  assert.ok(Math.abs(g.deep - g.thickness / g.parcel) < 1e-12, "parcels deep is A/P over √(A/n)");
  assert.ok(Math.abs(g.score - g.deep * g.crowding) < 1e-15);
});

test("at the same area and buildings, a strip scores lower than a square", () => {
  const blk = bundle.screen.followed;
  const square = geometry(blk.area_m2, 1, blk.n);
  const strip = geometry(blk.area_m2, 32, blk.n);
  assert.ok(strip.score < square.score && strip.deepest < square.deepest);
  assert.equal(strip.crowding, square.crowding, "reshaping must not change how crowded it is");
});

test("it opens on the block's own rectangle and reads out the block's own score", async () => {
  const { host } = await mount();
  assert.equal(host.findAll("img").length, 0, "the fallback picture is still there");
  const s = bundle.screen.followed;
  const score = readouts(host)[4]!;
  assert.ok(score.startsWith(s.score.toPrecision(3)), score);
});

test("each ring of parcels is painted in turn, outermost first, to the deepest", async () => {
  const { host } = await mount("?geo=typical");
  const blk = bundle.screen.typical;
  const g = geometry(blk.area_m2, ownShape(blk), blk.n);
  const fills = lastFrame(host.find("canvas")!).filter((c) => c.op === "fill");
  assert.equal(fills.length, g.deepest, "one fill per ring of parcels");
  // Each ring inset one parcel further than the last: its first corner moves right every time.
  const xs = fills.map((c) => c.path[0]!.args[0]!);
  for (let i = 1; i < xs.length; i++) assert.ok(xs[i]! > xs[i - 1]!, `ring ${i + 1} not inset`);
  // Shaded from the ramp's first colour on the street toward its red.
  assert.equal(fills[0]!.fillStyle, bundle.encoding.escape_ramp[0]);
  assert.ok(host.text().includes(`${g.deepest} ${g.deepest === 1 ? "row" : "rows"} from a street`),
            host.text());
});

test("switching block drops the last block's buildings and shape", async () => {
  const { host, store } = await mount("?homes=50&shape=3");
  const picker = host.findAll("button").find((b) => b.textContent === "A typical block")!;
  picker.dispatch("click");
  assert.deepEqual(store.get(), { block: "typical", homes: null, shape: null });
  const t = bundle.screen.typical;
  assert.ok(readouts(host)[4]!.startsWith(t.score.toPrecision(3)), readouts(host)[4]);
});
