import { strict as assert } from "node:assert";
import { test } from "node:test";
import { fitAxes } from "../src/view/transform.js";
import { niceTicks } from "../src/view/ticks.js";
import { createSvg, drawAxes, drawMarkers, drawPolyline } from "../src/render/svg.js";

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
  /** Inline style, which is what actually sizes the chart -- see createSvg's own doc. A plain record
   * is enough: the code under test only assigns to it. */
  readonly style: Record<string, string> = {};
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
/** A phone-sized column: a 390 px viewport gives about this, and `scripts/gen_site_pages.py` records
 * the project's own expectation that a phone is how most readers arrive. The aspect is the committed
 * fallback PNG's own IHDR ratio, which is what the widget sizes its box by. */
const NARROW = { width: 358, height: Math.round(358 / 1.398) };
// [0, 0.4] / [0, 1] is the real Frontier axis pairing from the brief -- both `draw` regimes below
// use it, so the first/last tick on each axis always lands EXACTLY on the world range's own ends
// (niceTicks(0, 0.4, 5) = [0, 0.1, 0.2, 0.3, 0.4], niceTicks(0, 1, 5) = [0, 0.2, ..., 1]), and the
// plot-rect recovery in svg.ts ("the tick extremes coincide exactly with the world range ends" --
// its own doc comment) is exact for every test below, not approximate.
const X_TICKS = niceTicks(0, 0.4, 5);
const Y_TICKS = niceTicks(0, 1, 5);

/** `pad = 0`: no gutter at all, the tightest case and the one fix round 1's containment guard
 * exists for. A generous `pad` (e.g. 0.15, ~45-60px of gutter on a 300-400px box -- comfortably
 * more than this test's own assumed EM_PX/CHAR_PX) exercises the OTHER proven regime fix round 2
 * added: gutter used, tick labels/titles moved off the plot rect. See drawAxes's own doc comment
 * for why only these two regimes are proven, not a continuum between them. */
function draw(pad: number, formatTick: (t: number) => string = String,
              box: { width: number; height: number } = { width: WIDTH, height: HEIGHT },
             ): { svg: FakeElement; rect: Rect } {
  const host = new FakeElement("div");
  const svg = createSvg(host as unknown as HTMLElement, box.width, box.height);
  const v = fitAxes([0, 0.4], [0, 1], box.width, box.height, pad);
  // `String` by default, which is exactly what drawAxes hard-coded before fix round 1 gave it a
  // required `formatTick` -- so every assertion below still runs on the same label TEXT it was
  // written against, and the containment/gutter guards (one of them a Critical from the previous
  // task) are unchanged rather than re-tuned. PERCENT_TICK below covers the units the Frontier
  // widget actually passes, whose labels are up to four characters instead of one.
  drawAxes(svg, v, X_TICKS, Y_TICKS, "Displacement", "Permeability", formatTick, GRID_OPACITY);
  const fake = svg as unknown as FakeElement;
  return { svg: fake, rect: plotRectAttrs(fake) };
}

/** The Frontier widget's own formatter: both axes are fractions in [0, 1] drawn as percentages,
 * mirroring the `PercentFormatter(xmax=1)` on the matplotlib figure the widget replaces. */
const PERCENT_TICK = (t: number): string => `${(t * 100).toFixed(0)}%`;

/** Gridline weight the fixture draws with. The real value comes from the bundle (`grid_opacity`);
 * what matters here is that whatever the caller passes is what lands on every gridline. */
const GRID_OPACITY = 0.12;

/** The plot rect `drawAxes` recorded on `svg` via its own `setPlotRect` (see svg.ts) -- read back
 * the same way `drawGuide` itself recovers it, so this test's notion of "the plot rect" is
 * guaranteed to be the SAME one the production code used, not a value this test recomputes and
 * could quietly drift from it. */
interface Rect { left: number; right: number; top: number; bottom: number }
function plotRectAttrs(svg: FakeElement): Rect {
  return {
    left: Number(svg.getAttribute("data-plot-left")),
    right: Number(svg.getAttribute("data-plot-right")),
    top: Number(svg.getAttribute("data-plot-top")),
    bottom: Number(svg.getAttribute("data-plot-bottom")),
  };
}

function rectsOverlap(a: [number, number, number, number], b: [number, number, number, number]): boolean {
  const [aXMin, aXMax, aYMin, aYMax] = a;
  const [bXMin, bXMax, bYMin, bYMax] = b;
  return aXMin < bXMax && aXMax > bXMin && aYMin < bYMax && aYMax > bYMin;
}

/** `drawAxes` draws, in order: one line+text pair per x tick, one line+text pair per y tick, the
 * x title, then the y title (see its own source) -- so the `<text>` elements alone, in the same
 * order, split cleanly into x-tick labels, y-tick labels, x title, y title by COUNT, without
 * needing to pattern-match attributes or content (content alone cannot disambiguate: this widget's
 * x and y ticks both include the value 0). */
function splitLabels(svg: FakeElement): { xTickLabels: FakeElement[]; yTickLabels: FakeElement[]; xTitle: FakeElement; yTitle: FakeElement } {
  const texts = svg.children.filter((c) => c.tagName === "text");
  const xTickLabels = texts.slice(0, X_TICKS.length);
  const yTickLabels = texts.slice(X_TICKS.length, X_TICKS.length + Y_TICKS.length);
  const xTitle = texts[X_TICKS.length + Y_TICKS.length];
  const yTitle = texts[X_TICKS.length + Y_TICKS.length + 1];
  if (xTitle === undefined || yTitle === undefined) throw new Error("drawAxes did not emit both titles");
  return { xTickLabels, yTickLabels, xTitle, yTitle };
}

// Absorbs floating-point noise from the rotation matrix (cos(-90 deg) is ~6e-17, not exactly 0, so
// a point meant to land exactly on an edge can come out as e.g. -3.3e-15) without weakening the
// check against a real overflow, which is on the order of whole pixels -- the same role `1e-9`
// plays in transform.test.ts and ticks.test.ts.
const EPS = 1e-6;

function assertContained(t: FakeElement,
                        box: { width: number; height: number } = { width: WIDTH, height: HEIGHT },
                       ): void {
  const [xMin, xMax, yMin, yMax] = effectiveBox(t);
  assert.ok(xMin >= -EPS && xMax <= box.width + EPS,
    `"${t.textContent}" x-range [${xMin}, ${xMax}] escapes [0, ${box.width}] ` +
    `(attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
  assert.ok(yMin >= -EPS && yMax <= box.height + EPS,
    `"${t.textContent}" y-range [${yMin}, ${yMax}] escapes [0, ${box.height}] ` +
    `(attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
}

test("drawAxes renders every label inside the SVG box, even at pad = 0", () => {
  const { svg } = draw(0);
  const texts = svg.children.filter((c) => c.tagName === "text");
  assert.ok(texts.length >= 4, `expected tick labels + 2 titles, got ${texts.length} text nodes`);
  for (const t of texts) assertContained(t);
});

test("drawAxes never emits an empty label, and both axis titles are present", () => {
  const { svg } = draw(0);
  const texts = svg.children.filter((c) => c.tagName === "text");
  for (const t of texts) {
    assert.ok(t.textContent.length > 0, `empty text content (attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
  }
  const labels = texts.map((t) => t.textContent);
  assert.ok(labels.includes("Displacement"), "x-axis title missing");
  assert.ok(labels.includes("Permeability"), "y-axis title missing");
});

// Fix round 2: with a gutter (unlike the pad = 0 test above), containment alone is not the whole
// contract any more -- gridlines, guides, and DATA live in the plot rect; tick labels and titles
// live in the gutter around it. GENEROUS_PAD is chosen once, up top, specifically big enough
// (checked below) for this test's own assumed glyph metrics to clear it, so the two assertions
// that follow are testing the "gutter has room" regime drawAxes's doc comment claims to guarantee,
// not merely whichever pad happened to be lying around.
const GENEROUS_PAD = 0.15;

test("drawAxes keeps every label inside the box, and outside the plot rect, once the gutter has room", () => {
  const { svg, rect } = draw(GENEROUS_PAD);
  // Sanity check on the fixture itself: if this ever fails, GENEROUS_PAD is not generous enough
  // for EM_PX/CHAR_PX any more and the test below would be exercising the wrong regime silently.
  // Two ROWS below the plot (tick labels then the axis title) and, on the left, the widest tick
  // label plus the rotated title -- not the one em this asserted before, which understated what the
  // layout needs by about 3x and so did not actually establish the "gutter has room" regime it
  // claims to (final review, I1).
  const widestYLabel = Math.max(...Y_TICKS.map((t) => String(t).length)) * CHAR_PX;
  assert.ok(HEIGHT - rect.bottom > 2 * EM_PX && rect.left > widestYLabel + EM_PX,
    `GENEROUS_PAD ${GENEROUS_PAD} leaves too little gutter to test the "room" regime: ` +
    `bottom gutter ${HEIGHT - rect.bottom} (needs > ${2 * EM_PX}), ` +
    `left gutter ${rect.left} (needs > ${widestYLabel + EM_PX})`);

  const plotBox: [number, number, number, number] = [rect.left, rect.right, rect.top, rect.bottom];
  const texts = svg.children.filter((c) => c.tagName === "text");
  assert.ok(texts.length >= 4, `expected tick labels + 2 titles, got ${texts.length} text nodes`);
  for (const t of texts) {
    assertContained(t);
    const box = effectiveBox(t);
    assert.ok(!rectsOverlap(box, plotBox),
      `"${t.textContent}" box [${box.join(", ")}] overlaps the plot rect [${plotBox.join(", ")}] ` +
      `(attrs: ${JSON.stringify(Object.fromEntries(t.attrs))})`);
  }
});

test("the y-axis title does not overlap any y-tick label", () => {
  const { svg } = draw(GENEROUS_PAD);
  const { yTickLabels, yTitle } = splitLabels(svg);
  const titleBox = effectiveBox(yTitle);
  for (const label of yTickLabels) {
    assert.ok(!rectsOverlap(titleBox, effectiveBox(label)),
      `y title box [${titleBox.join(", ")}] overlaps y-tick label "${label.textContent}" box ` +
      `[${effectiveBox(label).join(", ")}]`);
  }
});

test("drawPolyline throws on an xs/ys length mismatch instead of emitting NaN points", () => {
  const host = new FakeElement("div");
  const svg = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT);
  const v = fitAxes([0, 1], [0, 1], WIDTH, HEIGHT, 0);
  assert.throws(() => drawPolyline(svg, v, [0, 1, 2], [0, 1], "red", 2),
    /xs and ys must be the same length/);
});

test("percent tick labels -- four characters, not one -- still stay in the gutter", () => {
  // The regime the Frontier widget actually draws in. "100%" is four times the width of "1", and
  // y-tick labels grow LEFT from the plot rect while the rotated y-axis title grows right from
  // x = 0, so a wider tick column is exactly what would collide with it -- silently, since nothing
  // in SVG complains about overlapping text. Containment is asserted the same way as for the
  // one-character labels, plus the plot rect and the y-title, so this is the same three guarantees
  // over the label set that ships.
  const { svg, rect } = draw(GENEROUS_PAD, PERCENT_TICK);
  const { xTickLabels, yTickLabels, yTitle } = splitLabels(svg);
  const plotBox: [number, number, number, number] = [rect.left, rect.right, rect.top, rect.bottom];

  assert.deepEqual(xTickLabels.map((t) => t.textContent), ["0%", "10%", "20%", "30%", "40%"]);
  assert.deepEqual(yTickLabels.map((t) => t.textContent),
    ["0%", "20%", "40%", "60%", "80%", "100%"]);

  for (const t of [...xTickLabels, ...yTickLabels]) {
    assertContained(t);
    const box = effectiveBox(t);
    assert.ok(!rectsOverlap(box, plotBox),
      `"${t.textContent}" box [${box.join(", ")}] overlaps the plot rect [${plotBox.join(", ")}]`);
  }
  const titleBox = effectiveBox(yTitle);
  for (const label of yTickLabels) {
    assert.ok(!rectsOverlap(titleBox, effectiveBox(label)),
      `y title [${titleBox.join(", ")}] overlaps percent y-tick label "${label.textContent}" ` +
      `[${effectiveBox(label).join(", ")}]`);
  }
});

test("drawMarkers puts one dot on every sample, at the polyline's own projected positions", () => {
  // The marks exist because a curve clipped to a SINGLE sample has no segment to stroke and draws
  // nothing at all -- so the count and the coincidence with the line are the whole contract.
  const host = new FakeElement("div");
  const svg = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT);
  const v = fitAxes([0, 0.4], [0, 1], WIDTH, HEIGHT, GENEROUS_PAD);
  const xs = [0, 0.1, 0.4];
  const ys = [0, 0.55, 1];
  const line = drawPolyline(svg, v, xs, ys, "#123456", 2.5) as unknown as FakeElement;
  const dots = drawMarkers(svg, v, xs, ys, "#123456", 2.5) as unknown as FakeElement[];

  assert.equal(dots.length, xs.length);
  const points = line.getAttribute("points")!.split(" ");
  assert.equal(points.length, dots.length);
  for (const [i, dot] of dots.entries()) {
    assert.equal(`${dot.getAttribute("cx")},${dot.getAttribute("cy")}`, points[i],
      "a marker sits somewhere the line does not");
    assert.ok(Number(dot.getAttribute("r")) > 0, "a zero-radius marker draws nothing");
    assert.equal(dot.getAttribute("fill"), "#123456");
    // The samples are already conveyed by the line and by the widget's text readout; announcing 229
    // of them per curve would drown the readout that carries the actual numbers.
    assert.equal(dot.getAttribute("aria-hidden"), "true");
  }

  // A single sample: no polyline segment exists, so the dot is the only thing that renders.
  const lone = drawMarkers(svg, v, [0.2], [0.5], "#123456", 2.5);
  assert.equal(lone.length, 1);
});

test("drawMarkers throws on an xs/ys length mismatch instead of emitting NaN centres", () => {
  const host = new FakeElement("div");
  const svg = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT);
  const v = fitAxes([0, 1], [0, 1], WIDTH, HEIGHT, 0);
  assert.throws(() => drawMarkers(svg, v, [0, 1, 2], [0, 1], "red", 2),
    /xs and ys must be the same length/);
});

test("createSvg sizes the chart with an INLINE STYLE, which a stylesheet rule cannot override", () => {
  // Final review C1. The pinned mkdocs-material ships
  // `.md-typeset img,.md-typeset svg,.md-typeset video{height:auto;max-width:100%}` and wraps page
  // content in `.md-typeset`, so a presentation attribute -- specificity zero, sorted before every
  // author sheet -- loses the height. The box that renders would then not be the box drawAxes
  // computed every gutter and tick row from: nothing throws, and most of the chart is simply not
  // there. An inline style wins the cascade, which is the same mechanism the shipped sibling widget
  // relies on (perm-graph sizes its canvas with `cv.style.width`).
  const host = new FakeElement("div");
  const svg = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT) as unknown as FakeElement;

  assert.equal(svg.style["width"], `${WIDTH}px`, "no inline width: a stylesheet rule can override it");
  assert.equal(svg.style["height"], `${HEIGHT}px`,
    "no inline height: Material's `.md-typeset svg{height:auto}` wins and the box collapses");
  // The attributes stay too -- `sizeOf` reads the box back off them, so dropping them would make
  // drawAxes compute its layout from NaN.
  assert.equal(svg.getAttribute("width"), String(WIDTH));
  assert.equal(svg.getAttribute("height"), String(HEIGHT));
});

test("the x-axis title does not overlap any x-tick label -- the mirror of the y-side guard", () => {
  // Final review I1: the y pair has been guarded since round 2 of the previous task; this pair was
  // guarded nowhere, and at the fixture's own 400x300 geometry the two were already overlapping by
  // ~4.6 px on this file's glyph model. Checked at BOTH the fixture box and a phone-sized column,
  // because the failure was a function of how much gutter the box leaves.
  for (const box of [{ width: WIDTH, height: HEIGHT }, NARROW]) {
    const { svg } = draw(GENEROUS_PAD, PERCENT_TICK, box);
    const { xTickLabels, xTitle } = splitLabels(svg);
    const titleBox = effectiveBox(xTitle);
    assertContained(xTitle, box);
    for (const label of xTickLabels) {
      assertContained(label, box);
      assert.ok(!rectsOverlap(titleBox, effectiveBox(label)),
        `at ${box.width}x${box.height}: x title box [${titleBox.join(", ")}] overlaps x-tick label ` +
        `"${label.textContent}" box [${effectiveBox(label).join(", ")}]`);
    }
  }
});

test("gridlines are drawn at the caller's opacity, not at full body ink", () => {
  // Final review I2: they were `currentColor` at full strength -- 11 near-black lines under the
  // data -- while the matplotlib figure the widget replaces draws no gridlines at all. The weight is
  // the caller's (it comes from the bundle), and `currentColor` stays so the lines follow the site's
  // light/dark scheme rather than pinning a grey that vanishes in one of them.
  const { svg } = draw(GENEROUS_PAD, PERCENT_TICK);
  const grid = svg.children.filter((c) => c.tagName === "line");
  assert.equal(grid.length, X_TICKS.length + Y_TICKS.length, "one gridline per tick");
  for (const line of grid) {
    assert.equal(line.getAttribute("stroke"), "currentColor");
    assert.equal(Number(line.getAttribute("stroke-opacity")), GRID_OPACITY,
      `gridline at full ink: ${JSON.stringify(Object.fromEntries(line.attrs))}`);
    assert.equal(line.getAttribute("aria-hidden"), "true");
  }
  // Zero is a legitimate weight and means "no gridlines" -- exact parity with the fallback figure.
  const { svg: none } = ((): { svg: FakeElement } => {
    const host = new FakeElement("div");
    const s2 = createSvg(host as unknown as HTMLElement, WIDTH, HEIGHT);
    drawAxes(s2, fitAxes([0, 0.4], [0, 1], WIDTH, HEIGHT, GENEROUS_PAD), X_TICKS, Y_TICKS,
             "Displacement", "Permeability", PERCENT_TICK, 0);
    return { svg: s2 as unknown as FakeElement };
  })();
  for (const line of none.children.filter((c) => c.tagName === "line")) {
    assert.equal(Number(line.getAttribute("stroke-opacity")), 0);
  }
});
