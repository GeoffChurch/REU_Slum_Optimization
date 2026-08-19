import { strict as assert } from "node:assert";
import { test } from "node:test";
import { fitAxes } from "../src/view/transform.js";
import { niceTicks } from "../src/view/ticks.js";
import { createSvg, drawAxes, drawPolyline } from "../src/render/svg.js";

/** svg.ts's real `document.createElementNS(...)` calls are all *inside* functions, never at
 * module top level (unlike mount.ts's `document.addEventListener`), so -- unlike
 * widgets-bundle.test.ts, which has to eval the pre-bundled artifact in a `vm` sandbox because it
 * needs the whole module graph resolved -- this file can `import` svg.ts's exports directly and
 * only needs to stand in for `document` itself before calling them. No browser, no jsdom: just
 * enough of a fake DOM to record what svg.ts would have built, the same minimal-stub spirit as
 * widgets-bundle.test.ts's `{ addEventListener }` stub.
 */
class FakeElement {
  readonly tagName: string;
  readonly attrs = new Map<string, string>();
  readonly children: FakeElement[] = [];
  textContent = "";
  constructor(tagName: string) { this.tagName = tagName; }
  setAttribute(name: string, value: string): void { this.attrs.set(name, value); }
  getAttribute(name: string): string | null { return this.attrs.get(name) ?? null; }
  append(...nodes: FakeElement[]): void { this.children.push(...nodes); }
}

// Textual order relative to the `import` above does not matter: svg.ts's top-level module body
// never touches `document` (only the function bodies imported above do), and those functions are
// only called from inside `test(...)` callbacks below, which run after this whole module's
// top-level body -- including this assignment -- has finished executing.
(globalThis as Record<string, unknown>).document = {
  createElementNS: (_ns: string, tag: string): FakeElement => new FakeElement(tag),
};

// No real browser exists in this environment (or for the reviewer) to measure actual glyph
// metrics, so these two constants are a stand-in for what a browser's layout engine would report
// for `font-size` and average character advance width. They exist ONLY in this test, to convert
// an SVG `em`/character-count into a pixel box for the bounds check below -- svg.ts itself never
// assumes a font size (see its own module doc: that is deliberate, since its signature is given
// no font-size parameter to know one with). A different assumed size would shift the exact margins
// this test tolerates, but not change PASS/FAIL for the containment bug this test guards: that bug
// put labels entirely outside the box regardless of font size.
const EM_PX = 16;
const CHAR_PX = 9;

function parseDyEm(raw: string | null): number {
  if (raw === null) return 0;
  const m = /^(-?[\d.]+)em$/.exec(raw);
  if (!m) throw new Error(`unexpected dy format: ${raw}`);
  return Number(m[1]);
}

/** The glyph's bounding box in the text element's own local (pre-transform) coordinate frame:
 * horizontal extent from `text-anchor` and an assumed character width, vertical extent from
 * `dominant-baseline` and the assumed em size. */
function localGlyphBox(t: FakeElement): [xMin: number, xMax: number, yMin: number, yMax: number] {
  const x = Number(t.getAttribute("x") ?? "0");
  const y = Number(t.getAttribute("y") ?? "0") + parseDyEm(t.getAttribute("dy")) * EM_PX;
  const w = t.textContent.length * CHAR_PX;
  const anchor = t.getAttribute("text-anchor") ?? "start";
  const [xMin, xMax] = anchor === "middle" ? [x - w / 2, x + w / 2]
    : anchor === "end" ? [x - w, x]
    : [x, x + w];
  const baseline = t.getAttribute("dominant-baseline") ?? "alphabetic";
  // hanging: anchor is the TOP of the glyph, grows down. middle: centered. alphabetic (default):
  // anchor is near the bottom, glyph mostly above with a small descender allowance below.
  const [yMin, yMax] = baseline === "hanging" ? [y, y + EM_PX]
    : baseline === "middle" ? [y - EM_PX / 2, y + EM_PX / 2]
    : [y - EM_PX * 0.8, y + EM_PX * 0.2];
  return [xMin, xMax, yMin, yMax];
}

/** Applies the element's own `transform="rotate(angle cx cy)"`, if present, to `localGlyphBox`'s 4
 * corners and returns the axis-aligned bounding box of the rotated result -- this is what "room
 * for the glyph box" means for the y-axis title, which is the element the rotation-sign bug was
 * in. Elements with no `transform` return their local box unchanged. */
function effectiveBox(t: FakeElement): [xMin: number, xMax: number, yMin: number, yMax: number] {
  const [xMin, xMax, yMin, yMax] = localGlyphBox(t);
  const raw = t.getAttribute("transform");
  if (raw === null) return [xMin, xMax, yMin, yMax];
  const m = /^rotate\((-?[\d.]+)[ ,]+(-?[\d.]+)[ ,]+(-?[\d.]+)\)$/.exec(raw);
  if (!m) throw new Error(`unexpected transform format: ${raw}`);
  const angle = (Number(m[1]) * Math.PI) / 180;
  const cx = Number(m[2]);
  const cy = Number(m[3]);
  const corners: [number, number][] = [[xMin, yMin], [xMin, yMax], [xMax, yMin], [xMax, yMax]];
  const rotated = corners.map(([px, py]) => {
    const dx = px - cx;
    const dy = py - cy;
    return [cx + dx * Math.cos(angle) - dy * Math.sin(angle),
            cy + dx * Math.sin(angle) + dy * Math.cos(angle)] as [number, number];
  });
  const xs = rotated.map(([rx]) => rx);
  const ys = rotated.map(([, ry]) => ry);
  return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
}

const WIDTH = 400;
const HEIGHT = 300;

function draw(): FakeElement {
  const host = new FakeElement("div");
  // pad = 0 deliberately: the contract this test guards is containment with NO gutter to spare
  // (see drawAxes's own doc). [0, 0.4] / [0, 1] is the real Frontier axis pairing from the brief,
  // so the first/last tick on each axis lands EXACTLY on the box edge -- the tightest case, not a
  // hypothetical one.
  const svg = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT);
  const v = fitAxes([0, 0.4], [0, 1], WIDTH, HEIGHT, 0);
  const xTicks = niceTicks(0, 0.4, 5);
  const yTicks = niceTicks(0, 1, 5);
  drawAxes(svg, v, xTicks, yTicks, "Displacement", "Permeability");
  return svg as unknown as FakeElement;
}

// Absorbs floating-point noise from the rotation matrix (cos(-90 deg) is ~6e-17, not exactly 0, so
// a point meant to land exactly on an edge can come out as e.g. -3.3e-15) without weakening the
// check against a real overflow, which is on the order of whole pixels -- the same role `1e-9`
// plays in transform.test.ts and ticks.test.ts.
const EPS = 1e-6;

test("drawAxes renders every label inside the SVG box, even at pad = 0", () => {
  const svg = draw();
  const texts = svg.children.filter((c) => c.tagName === "text");
  assert.ok(texts.length >= 4, `expected tick labels + 2 titles, got ${texts.length} text nodes`);
  for (const t of texts) {
    const [xMin, xMax, yMin, yMax] = effectiveBox(t);
    assert.ok(xMin >= -EPS && xMax <= WIDTH + EPS,
      `"${t.textContent}" x-range [${xMin}, ${xMax}] escapes [0, ${WIDTH}] ` +
      `(attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
    assert.ok(yMin >= -EPS && yMax <= HEIGHT + EPS,
      `"${t.textContent}" y-range [${yMin}, ${yMax}] escapes [0, ${HEIGHT}] ` +
      `(attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
  }
});

test("drawAxes never emits an empty label, and both axis titles are present", () => {
  const svg = draw();
  const texts = svg.children.filter((c) => c.tagName === "text");
  for (const t of texts) {
    assert.ok(t.textContent.length > 0, `empty text content (attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
  }
  const labels = texts.map((t) => t.textContent);
  assert.ok(labels.includes("Displacement"), "x-axis title missing");
  assert.ok(labels.includes("Permeability"), "y-axis title missing");
});

test("drawPolyline throws on an xs/ys length mismatch instead of emitting NaN points", () => {
  const host = new FakeElement("div");
  const svg = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT);
  const v = fitAxes([0, 1], [0, 1], WIDTH, HEIGHT, 0);
  assert.throws(() => drawPolyline(svg, v, [0, 1, 2], [0, 1], "red", 2),
    /xs and ys must be the same length/);
});
