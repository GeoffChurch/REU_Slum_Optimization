import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { DemoBlock, DemoBundle } from "../src/demo.js";
import { createChannel, type Channel } from "../src/page/channel.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import {
  fakeLocation, FakeElement, fireResize, installStubs, lastFrame, mountPoint, writeNow,
} from "./harness.js";
import {
  factors, longestEdge, OUTLINES, perimeterOf, ringsOf, SCREEN_GEOMETRY_URL, screenGeometryOn,
  slotTicks, type GeometryState,
} from "../src/widgets/screen-geometry.js";

installStubs();

const bundle = JSON.parse(readFileSync("../examples/demo/demo.json", "utf8")) as DemoBundle;
const [FIRST, SECOND] = bundle.blocks as [DemoBlock, DemoBlock];
const own = FIRST.proxy;
(globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
  ok: true, status: 200, statusText: "OK",
  json: (): Promise<unknown> => Promise.resolve(bundle),
});

/** The shipped screen, written out from its definition rather than imported, so the widget's
 * arithmetic is checked against the formula and not against itself. */
const screen = (n: number, area: number, perimeter: number): number =>
  (Math.sqrt(n * area) / perimeter) * (n / area);
const close = (a: number, b: number, rel: number): boolean => Math.abs(a - b) <= rel * Math.abs(b);

async function mount(search = "", channel: Channel<string> = createChannel<string>()):
    Promise<{ host: FakeElement; store: StateSource<GeometryState> }> {
  const host = mountPoint();
  host.dataset.bundle = "../assets/demo/demo.json";
  const slot = urlStore(fakeLocation(search), writeNow).reserve();
  let store: StateSource<GeometryState> | null = null;
  screenGeometryOn(channel)(host as never, (initial) => {
    store = slot.bind(SCREEN_GEOMETRY_URL, initial);
    return store;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  fireResize(600, 300);
  assert.ok(store !== null, "the widget never asked for a state store");
  return { host, store };
}

const readouts = (host: FakeElement): string[] => host.findAll("dd").map((d) => d.textContent);
const button = (host: FakeElement, text: string): FakeElement => {
  const b = host.findAll("button").find((x) => x.textContent === text);
  assert.ok(b !== undefined, `no "${text}" button`);
  return b;
};

test("the depth factor is the homes behind each frontage slot, for any shape", () => {
  for (const blk of bundle.blocks.map((b) => b.proxy)) {
    const f = factors(blk.area_m2, blk.perimeter_m, blk.n);
    assert.ok(close(f.slots, blk.perimeter_m / Math.sqrt(blk.area_m2 / blk.n), 1e-12));
    assert.ok(close(f.perSlot, Math.sqrt(blk.n * blk.area_m2) / blk.perimeter_m, 1e-12));
    assert.ok(close(f.score, blk.score, 1e-9), `${f.score} vs ${blk.score}`);
    assert.ok(close(f.score, screen(blk.n, blk.area_m2, blk.perimeter_m), 1e-12));
    assert.ok(close(f.score, f.perSlot / f.side ** 2, 1e-12), "score = homes per slot / s²");
  }
});

test("at one area and one count, shape enters only through the street: the strip scores least",
     () => {
  const scores = OUTLINES.map((o) => factors(own.area_m2, perimeterOf(o, FIRST), own.n));
  for (const f of scores) assert.equal(f.side, scores[0]!.side, "s must not depend on shape");
  const [ownF, disk, square, strip] = scores;
  assert.ok(disk!.score > square!.score && square!.score > strip!.score);
  // 5810's own street is long for its area: it sits below the square, as the page says.
  assert.ok(ownF!.score < square!.score);
});

test("the ticks along the outline number the frontage slots of the drawn street", () => {
  for (const o of OUTLINES) {
    const rings = ringsOf(o, FIRST);
    const side = Math.sqrt(own.area_m2 / own.n);
    const drawn = rings.reduce((s, r) => s + r.reduce((t, p, i) => {
      const q = r[(i + 1) % r.length]!;
      return t + Math.hypot(q[0] - p[0], q[1] - p[1]);
    }, 0), 0);
    const ticks = slotTicks(rings, side);
    assert.ok(Math.abs(ticks.length - drawn / side) <= rings.length,
              `${o}: ${ticks.length} ticks for ${drawn / side} slots of drawn street`);
    // Each tick points into the block: a short step along it lands inside the outline (even-odd,
    // so a step from a hole's edge into the hole would count as outside).
    const inside = (x: number, y: number): boolean => {
      let k = false;
      for (const r of rings) {
        for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
          const [xi, yi] = r[i]!, [xj, yj] = r[j]!;
          if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) k = !k;
        }
      }
      return k;
    };
    const wrong = ticks.filter(([x, y, nx, ny]) => !inside(x + nx * 0.5, y + ny * 0.5));
    // A tick at a sharp concave corner can step out across the other edge; allow a handful.
    assert.ok(wrong.length <= 0.02 * ticks.length, `${o}: ${wrong.length} ticks point outward`);
    const e = longestEdge(rings, side);
    assert.ok(inside(e.x + e.ux * e.len / 2 + e.nx, e.y + e.uy * e.len / 2 + e.ny),
              `${o}: the parcels' edge normal points out of the block`);
    // Its first tick is one of slotTicks' own: the parcels drawn from it span whole slots.
    const tx = e.x + e.ux * e.firstTick, ty = e.y + e.uy * e.firstTick;
    assert.ok(ticks.some(([x, y]) => Math.hypot(x - tx, y - ty) < 1e-6),
              `${o}: the longest edge's first tick is not a slot tick`);
  }
});

test("it opens on the block's own outline and reads out its own numbers", async () => {
  const { host } = await mount();
  assert.equal(host.findAll("img").length, 0, "the fallback picture is still there");
  const f = factors(own.area_m2, own.perimeter_m, own.n);
  const [, slots, perSlot, , score] = readouts(host);
  assert.equal(slots, Number(f.slots.toPrecision(3)).toLocaleString("en-US"));
  assert.equal(perSlot, Number(f.perSlot.toPrecision(3)).toLocaleString("en-US"));
  assert.ok(score!.startsWith(own.score.toPrecision(3)), score);
});

test("the outline drawn is the block's: one fill for the block, then the parcels", async () => {
  const { host } = await mount();
  const frame = lastFrame(host.find("canvas")!);
  const first = frame.find((c) => c.op === "fill")!;
  assert.equal(first.fillRule, "evenodd", "the block's holes must be cut out");
  const moves = first.path.filter((op) => op.op === "moveTo").length;
  assert.equal(moves, own.rings.length, "one subpath per ring of the block's own outline");
});

test("a reference shape keeps the area and the buildings and changes only the street", async () => {
  const { host, store } = await mount();
  button(host, "Strip").dispatch("click");
  assert.equal(store.get().outline, "strip");
  const f = factors(own.area_m2, perimeterOf("strip", FIRST), own.n);
  assert.equal(readouts(host)[1], Number(f.slots.toPrecision(3)).toLocaleString("en-US"));
  assert.equal(readouts(host)[0], `${Number(f.side.toPrecision(2))} m`);
});

test("the page's block switch drops the last block's buildings and keeps the outline",
     async () => {
  const channel = createChannel<string>();
  const { host, store } = await mount("?homes=50&outline=disk", channel);
  channel.set(SECOND.block_id);
  assert.deepEqual(store.get(), { outline: "disk", homes: null });
  const f = factors(SECOND.proxy.area_m2, perimeterOf("disk", SECOND), SECOND.proxy.n);
  assert.ok(readouts(host)[4]!.startsWith(f.score.toPrecision(3)), readouts(host)[4]);
  assert.ok(host.text().includes(`block ${SECOND.block_id}'s area`), host.text());
});

test("it opens on the page's block when the switch has already chosen one", async () => {
  const channel = createChannel<string>();
  channel.set(SECOND.block_id);
  const { host } = await mount("", channel);
  assert.ok(readouts(host)[4]!.startsWith(SECOND.proxy.score.toPrecision(3)), readouts(host)[4]);
});

test("the row of parcels stays on its edge, even on a disk's short edges", async () => {
  for (const outline of ["disk", "own"] as const) {
    const { host } = await mount(`?outline=${outline}`);
    const rings = ringsOf(outline, FIRST);
    const side = Math.sqrt(own.area_m2 / own.n);
    const e = longestEdge(rings, side);
    // The parcels are the only translucent-red fills (the inset's frames are strokes alone).
    const parcels = lastFrame(host.find("canvas")!).filter(
      (c) => c.op === "fill" && c.fillStyle === "rgba(153, 0, 0, 0.18)");
    // Drawn twice (the block and its inset), each a whole number of slots within the edge.
    assert.ok(parcels.length > 0 && parcels.length % 2 === 0, `${outline}: ${parcels.length}`);
    assert.ok(parcels.length / 2 <= Math.floor((e.len - e.firstTick) / side), outline);
  }
});
