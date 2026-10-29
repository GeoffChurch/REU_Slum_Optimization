import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { createLayer } from "../src/render/city.js";
import { sizeCanvas } from "../src/render/canvas.js";
import { toScreen, type View } from "../src/view/transform.js";
import { PixelCanvas, type PixelContext } from "./raster.js";

// The committed bundle's own encoding, so the colours tested are the ones the site ships.
const E = (JSON.parse(readFileSync("../examples/screen-map/capetown.json", "utf8")) as CityBundle)
  .encoding;

// A desktop's 1, where a sub-pixel block is smallest on the device: the worst case for blending.
const DPR = 1;
(globalThis as Record<string, unknown>).document = { createElement: () => new PixelCanvas() };
(globalThis as Record<string, unknown>).window = { devicePixelRatio: DPR };

const square = (x: number, y: number, s: number): [number, number][][] =>
  [[[x, y], [x + s, y], [x + s, y + s], [x, y + s]]];

/** Blocks a reader could see at the city scale, and the sub-pixel kind Cape Town is mostly made
 * of (median ~0.6 CSS px², render/city.ts): A informal, B plain, C informal (never selected),
 * D and E informal and plain at a tenth of a CSS pixel each, side by side in one pixel's
 * neighbourhood, so a selected one's colour has its neighbour and the base under it to bleed
 * from. 1 m = 1 CSS px. */
const RINGS = [square(4, 4, 12), square(20, 4, 12), square(36, 4, 12), square(5.2, 24.2, 0.3),
               square(5.7, 24.2, 0.3)];
const bundle = {
  city: "test", crs_epsg: 0, origin: [0, 0], n_blocks: RINGS.length,
  block_id: RINGS.map((_, i) => `b${i}`), n: RINGS.map(() => 1), area_m2: RINGS.map(() => 1),
  perimeter_m: RINGS.map(() => 1), rings: RINGS, informal: [1, 0, 1, 1, 0], floors: [],
  encoding: E,
} as unknown as CityBundle;
const A = 0, B = 1, C = 2, D = 3;
const SIZE = { width: 52, height: 32 };
const VIEW: View = { scaleX: 1, scaleY: 1, tx: 0, ty: SIZE.height };     // y up, as a map's
// The ranking: A, B, then the sub-pixel D, then the rest -- `k` of these are selected.
const ORDER = new Int32Array([A, B, D, 4, C]);

function frame(k: number): { ctx: PixelContext; paint: (k: number) => void } {
  const layer = createLayer();
  layer.paintBase(bundle, VIEW, E, SIZE);
  const visible = new PixelCanvas();
  sizeCanvas(visible as unknown as HTMLCanvasElement, SIZE);
  const ctx = visible.ctx;
  const paint = (kk: number): void => layer.paintFrame(
    ctx as unknown as CanvasRenderingContext2D, bundle, VIEW, E, ORDER, kk, SIZE);
  paint(k);
  return { ctx, paint };
}

const hex = (c: string): [number, number, number] =>
  [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) as [number, number, number];
const dist = (a: number[], b: number[]): number => Math.hypot(...a.map((v, i) => v - b[i]!));
/** The colour at world point (x, y), at the device pixel it lands in. */
function at(ctx: PixelContext, x: number, y: number): { rgb: number[]; a: number } {
  const [sx, sy] = toScreen(VIEW, x, y);
  return ctx.pixel(Math.floor(sx * DPR), Math.floor(sy * DPR));
}

test("a selected informal block paints red, not gold; unselected keep gold and grey", () => {
  const { ctx } = frame(2);                                      // A and B selected
  const red = hex(E.selected_color), gold = hex(E.informal_color), grey = hex(E.base_color);
  const a = at(ctx, 10, 10), b = at(ctx, 26, 10), c = at(ctx, 42, 10);
  assert.ok(dist(a.rgb, red) < 1 && a.a > 0.999, `selected informal block is ${a.rgb}, not red`);
  assert.ok(dist(b.rgb, red) < 1, `selected plain block is ${b.rgb}, not red`);
  assert.ok(dist(c.rgb, gold) < 1, `unselected informal block is ${c.rgb}, not gold`);
  const { ctx: none } = frame(0);
  assert.ok(dist(at(none, 26, 10).rgb, grey) < 1, "an unselected plain block is not grey");
});

test("a selected sub-pixel informal block reads red, not gold", () => {
  // Cape Town's typical block covers part of one pixel, so what a reader sees is its pixels'
  // blend: it must sit far nearer the red than the gold, as a selected plain one's does.
  const red = hex(E.selected_color), gold = hex(E.informal_color);
  const { ctx } = frame(3);                                      // ... and D
  // D's device pixels: its world square through the view, at DPR.
  const [x0, y1] = toScreen(VIEW, 5.2, 24.2).map((v) => v * DPR) as [number, number];
  const [x1, y0] = toScreen(VIEW, 5.5, 24.5).map((v) => v * DPR) as [number, number];
  let worst = 0, seen = 0;
  for (let y = Math.floor(y0); y < Math.ceil(y1); y++) {
    for (let x = Math.floor(x0); x < Math.ceil(x1); x++) {
      const p = ctx.pixel(x, y);
      if (p.a < 0.05) continue;
      seen++;
      worst = Math.max(worst, dist(p.rgb, red) / dist(p.rgb, gold));
    }
  }
  assert.ok(seen > 0, "the sub-pixel block painted nothing");
  assert.ok(worst < 0.5, `a pixel of the selected sub-pixel block is ${worst.toFixed(2)} `
    + "as far from red as from gold");
});

test("dragging the floor down and back up leaves no residue", () => {
  // The pixels after a round trip are bit-for-bit the pixels before it: each frame starts from
  // the base layer's own untouched copy, so no select/deselect cycle can leave colour behind --
  // the sub-pixel blocks, all antialiased edge, included.
  const { ctx, paint } = frame(0);
  const before = Float64Array.from(ctx.data);
  for (const k of [1, 2, 3, 4, 5, 3, 1]) paint(k);
  paint(0);
  assert.deepEqual(ctx.data, before);
  paint(5);
  const low = Float64Array.from(ctx.data);
  paint(1);
  paint(5);
  assert.deepEqual(ctx.data, low);
});
