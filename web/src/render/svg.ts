import { toScreen, type View } from "../view/transform.js";

/** The only module that knows SVG namespaces, mirroring render/canvas.ts's role: canvas.ts draws
 * PermGraph's marks, this one draws Frontier's. The split exists because the chart's axis labels
 * must be real, selectable, screen-reader-legible, printable text, which a <canvas> raster cannot
 * give -- see transform.ts's module doc.
 *
 * Every function here does exactly one thing: build element(s) via
 * `document.createElementNS(SVG_NS, …)`, position them through `toScreen(v, …)` (which already
 * applies transform.ts's y-flip -- nothing in this file flips again), set attributes, append, and
 * return. No layout algorithm, no retained state, no event wiring, and -- deliberately -- no
 * colour, stroke-width or font-size chosen in here: those are exactly the kind of per-widget
 * visual choice the project's global rule says must come from configuration, not a TypeScript
 * literal. `drawPolyline` and `drawGuide` take colour/width as parameters because their signatures
 * have room for them; `drawAxes` does not (its signature carries only what a caller cannot
 * possibly derive here -- tick values, titles, and since fix round 1 a tick FORMATTER; see its own
 * doc), so its chrome uses `currentColor` for paint -- a keyword that defers the actual
 * value to whatever CSS `color` is cascaded onto the widget's host, never a value this file picks
 * -- and leaves stroke-width unset, taking SVG's own initial value (1) rather than a chosen one.
 */
const SVG_NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(name: K): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NS, name);
}

/** Create the root <svg>, sized to `width`x`height` CSS pixels, and mount it in `host`. */
export function createSvg(host: HTMLElement, width: number, height: number): SVGSVGElement {
  const svg = el("svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  host.append(svg);
  return svg;
}

/** The pixel size `createSvg` gave `svg` -- read back rather than threaded through every call. */
function sizeOf(svg: SVGSVGElement): { width: number; height: number } {
  return { width: Number(svg.getAttribute("width")), height: Number(svg.getAttribute("height")) };
}

interface Rect { left: number; right: number; top: number; bottom: number }

/** The plot rect: where the DATA lives, as opposed to the gutter around it where axis chrome
 * lives. Recovered from the tick extremes rather than threaded through as a parameter (fix round
 * 2's finding): `toScreen(v, xTicks[0], yTicks[0])` is one corner, the last ticks the other. For
 * this widget's axes ([0, 0.4] step 0.1, [0, 1] step 0.2) the tick extremes coincide exactly with
 * the world range ends, so the recovery is exact, not approximate -- and `niceTicks`'s own "ticks
 * span the range and stay inside it" invariant guarantees it is never a poorer approximation than
 * that in general: the plot rect is always a subset of the SVG box, never larger.
 *
 * Falls back to the full box when a caller passes no ticks on an axis (nothing to derive a
 * smaller rect FROM in that case -- not a silently-wrong default, there is no better answer). */
function plotRect(v: View, xTicks: number[], yTicks: number[], box: { width: number; height: number }): Rect {
  const x0 = xTicks[0];
  const x1 = xTicks[xTicks.length - 1];
  const y0 = yTicks[0];
  const y1 = yTicks[yTicks.length - 1];
  if (x0 === undefined || x1 === undefined || y0 === undefined || y1 === undefined) {
    return { left: 0, right: box.width, top: 0, bottom: box.height };
  }
  const [left, bottom] = toScreen(v, x0, y0);
  const [right, top] = toScreen(v, x1, y1);
  return { left, right, top, bottom };
}

/** `drawGuide`'s signature has no `xTicks`/`yTicks` parameter to recover a plot rect from the way
 * `drawAxes` does, so `drawAxes` records the one it computed as data-* attributes on `svg` itself
 * -- the same "read state back off the element `createSvg` already returned" idiom `sizeOf` uses
 * for the box's own width/height, just for a second, smaller rect nested inside it. */
function setPlotRect(svg: SVGSVGElement, rect: Rect): void {
  svg.setAttribute("data-plot-left", String(rect.left));
  svg.setAttribute("data-plot-right", String(rect.right));
  svg.setAttribute("data-plot-top", String(rect.top));
  svg.setAttribute("data-plot-bottom", String(rect.bottom));
}

/** Reads back what `setPlotRect` stored. Falls back to the full box if `drawAxes` has not run yet
 * on this `svg` -- a guide drawn before any axes has no smaller rect to recover, so the box is the
 * only defensible default (matching `plotRect`'s own empty-ticks fallback above). */
function plotRectOf(svg: SVGSVGElement): Rect {
  const box = sizeOf(svg);
  const left = svg.getAttribute("data-plot-left");
  const right = svg.getAttribute("data-plot-right");
  const top = svg.getAttribute("data-plot-top");
  const bottom = svg.getAttribute("data-plot-bottom");
  if (left === null || right === null || top === null || bottom === null) {
    return { left: 0, right: box.width, top: 0, bottom: box.height };
  }
  return { left: Number(left), right: Number(right), top: Number(top), bottom: Number(bottom) };
}

/** Draw both axes into `svg`: gridlines confined to the plot rect (fix round 2 -- see below), a
 * real `<text>` label per tick, and the two axis titles. `xTicks`/`yTicks` are `niceTicks` output:
 * round-step values the caller already computed, drawn here and nowhere invented.
 *
 * UNITS (fix round 1 of the Frontier task, and the one thing that reopened this file): `formatTick`
 * turns a tick VALUE into its label TEXT, and it is REQUIRED -- no default. This module previously
 * printed `String(t)`, so a caller whose data are fractions got bare `0.6` where the matplotlib
 * figure the widget replaces prints `60%` (emit.compare_report puts a `PercentFormatter(xmax=1)` on
 * both axes). A defaulted parameter would have let that divergence come back silently at the next
 * call site, which is the failure mode this project keeps finding: nothing throws, the chart draws,
 * and only a human comparing two pictures notices. Required means a new caller must state its units
 * or fail to compile. The formatter's OUTPUT is deliberately unconstrained -- an empty string is a
 * legitimate "hide this tick" -- so nothing here validates it beyond drawing it.
 *
 * CONTAINMENT (fix round 1, Critical, kept): every label still renders inside
 * `[0, width] x [0, height]`, even at `pad = 0` where the plot rect fills the whole box and there
 * is no gutter to spare. That fix picked, for each label, an anchor and a growth direction proven
 * safe by construction rather than by assuming a caller-chosen pad left room -- unchanged here.
 *
 * GUTTER (fix round 2): round 1 fixed containment by anchoring every label at the ABSOLUTE box
 * edge and growing inward. That is safe, but it ignores the plot rect entirely, so three things
 * collide whenever a real gutter exists: gridlines (still spanning the full box) cross the label
 * rows; the y-axis title and the y-tick labels are both pinned to x = 0 and print on top of each
 * other; and nothing uses the gutter a generous `pad` actually provides. The fix is the standard
 * chart layout: the SVG box is the gutter plus the plot rect. Gridlines and `drawGuide` now span
 * the PLOT RECT (`plotRect` above), not the box -- so they stop before the label rows regardless
 * of gutter size, which is what removes the gridline/label collision structurally, with no
 * gutter-size reasoning needed for that half of the fix.
 *
 * Tick labels move OUTWARD to the far side of the gutter when one exists (`dominant-baseline`
 * "hanging" for x-tick labels growing down from the plot rect's bottom edge; `text-anchor="end"`
 * for y-tick labels growing left from the plot rect's left edge), which is what lets the y-axis
 * title -- still anchored at the absolute x = 0 -- stop overlapping them: the tick-label column no
 * longer sits at x = 0 once a gutter exists, so the two no longer share an anchor. Whether "one
 * exists" is decided by comparing the gutter's SIZE against zero (with a floating-point epsilon,
 * `GUTTER_EPS`, not an assumed font size), never by asking whether it is big ENOUGH: this file is
 * given no font-size parameter to answer that with (see the module doc), so it cannot know how
 * many pixels a label needs. The two regimes this file DOES guarantee are the two the bounds test
 * below exercises: a wholly degenerate `pad = 0` (falls back to round 1's proven-safe inward
 * clamp -- containment holds, labels may overlap the plot, exactly the round-1 behaviour, kept
 * unchanged) and a generously padded chart (falls back to nothing -- the natural outward placement
 * has room and is used directly). A caller-chosen pad that is nonzero but still too small for
 * whatever font actually renders sits between those two proven regimes and is not specifically
 * defended against -- doing so would need either a font-size parameter or a real DOM measurement
 * (`getBBox`), neither available here; flagged plainly rather than silently left unmentioned.
 */
export function drawAxes(svg: SVGSVGElement, v: View, xTicks: number[], yTicks: number[],
                         xLabel: string, yLabel: string,
                         formatTick: (t: number) => string): void {
  const box = sizeOf(svg);
  const { width, height } = box;
  const rect = plotRect(v, xTicks, yTicks, box);
  setPlotRect(svg, rect);

  // See the function doc's GUTTER paragraph: "a gutter exists" is a zero-vs-nonzero comparison
  // (guarded by a small epsilon against floating-point noise in the rotation/scale arithmetic,
  // the same failure mode round 1 hit with `cos(-90 deg)`), never a "big enough" one.
  const GUTTER_EPS = 1e-6;
  const hasBottomGutter = height - rect.bottom > GUTTER_EPS;
  const hasLeftGutter = rect.left > GUTTER_EPS;

  // A tick whose screen position lands exactly on the box edge (the real Frontier x axis is
  // [0, 0.4] with pad 0, so its first/last tick DOES land exactly on x = 0 / x = width) must not
  // grow its label symmetrically about that point -- half the glyph would sit outside. Comparing
  // against the known-safe absolute edges (0 and the box's own width/height, not an assumed glyph
  // size) is enough to pick a direction that is always safe, and falls back to the ordinary
  // centered/symmetric mode for every interior tick. Unchanged from round 1: this is about the
  // PERPENDICULAR dimension to the gutter fix above (x-tick labels' own horizontal position,
  // y-tick labels' own vertical position), which the gutter does not touch.
  const edgeAnchor = (s: number): "start" | "middle" | "end" =>
    s <= 0 ? "start" : s >= width ? "end" : "middle";
  const edgeBaseline = (s: number): "hanging" | "middle" | "alphabetic" =>
    s <= 0 ? "hanging" : s >= height ? "alphabetic" : "middle";
  // "alphabetic" grows mostly upward from its anchor, but still reserves a small allowance BELOW
  // the anchor for a descender (g, y, p, ...) -- so a glyph anchored EXACTLY on an edge still
  // pokes out by that allowance (caught by the round-1 bounds test). Digits/'.'/'-' never actually
  // descend in any common font, but the margin is closed structurally rather than by trusting
  // today's character set. Negative is always the safe direction (see the function doc), so the
  // exact magnitude is not load-bearing the way it would be if it could push the wrong way.
  const alphabeticDy = "-0.25em";

  for (const t of xTicks) {
    const [sx] = toScreen(v, t, 0);
    const line = el("line");
    line.setAttribute("x1", String(sx));
    line.setAttribute("y1", String(rect.top));
    line.setAttribute("x2", String(sx));
    line.setAttribute("y2", String(rect.bottom));
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("aria-hidden", "true");
    svg.append(line);

    const label = el("text");
    label.setAttribute("x", String(sx));
    label.setAttribute("y", String(rect.bottom));
    label.setAttribute("text-anchor", edgeAnchor(sx));
    if (hasBottomGutter) {
      // Room to spare: grow DOWN into the gutter, away from the plot rect, the natural position.
      label.setAttribute("dominant-baseline", "hanging");
    } else {
      // No gutter (pad = 0, or a caller-chosen pad too small even for the plot rect's own tick
      // extremes to clear the box edge): round 1's proven-safe inward clamp, unchanged.
      label.setAttribute("dy", alphabeticDy);
      label.setAttribute("dominant-baseline", "alphabetic");
    }
    label.setAttribute("fill", "currentColor");
    label.textContent = formatTick(t);
    svg.append(label);
  }

  for (const t of yTicks) {
    const [, sy] = toScreen(v, 0, t);
    const line = el("line");
    line.setAttribute("x1", String(rect.left));
    line.setAttribute("y1", String(sy));
    line.setAttribute("x2", String(rect.right));
    line.setAttribute("y2", String(sy));
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("aria-hidden", "true");
    svg.append(line);

    const baseline = edgeBaseline(sy);
    const label = el("text");
    // Same gutter test as the x-tick labels above, mirrored onto this axis's own gutter
    // (`hasLeftGutter`): grow LEFT, away from the plot rect, when there is room; otherwise fall
    // back to round 1's x = 0, grow-right clamp.
    label.setAttribute("x", String(hasLeftGutter ? rect.left : 0));
    label.setAttribute("y", String(sy));
    if (baseline === "alphabetic") label.setAttribute("dy", alphabeticDy);
    label.setAttribute("text-anchor", hasLeftGutter ? "end" : "start");
    label.setAttribute("dominant-baseline", baseline);
    label.setAttribute("fill", "currentColor");
    label.textContent = formatTick(t);
    svg.append(label);
  }

  const xTitle = el("text");
  xTitle.setAttribute("x", String(width / 2));
  xTitle.setAttribute("y", String(height));
  // Negative: further INWARD (up) than the tick-label row, one row's worth of em. The old code
  // pushed positive (down, "hanging"), which is the direction that overflowed. Left keyed off the
  // absolute `height` (not `rect.bottom`) rather than round 1: once x-tick labels move into the
  // gutter above (`hasBottomGutter`), they sit BETWEEN `rect.bottom` and `height`, so a title still
  // anchored at `height` and pushed inward by ~1 line naturally lands just past them, outward
  // (closer to the true edge) of the tick-label row -- correct chart ordering (title outermost) --
  // for a gutter with room for both rows; see the function doc's GUTTER paragraph for the regime
  // this is and is not proven for.
  xTitle.setAttribute("dy", "-1.3em");
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("dominant-baseline", "alphabetic");
  xTitle.setAttribute("fill", "currentColor");
  xTitle.textContent = xLabel;
  svg.append(xTitle);

  const yTitle = el("text");
  yTitle.setAttribute("x", "0");
  yTitle.setAttribute("y", String(height / 2));
  yTitle.setAttribute("text-anchor", "middle");
  // "hanging" makes the glyph grow in +y pre-rotation; through this element's own -90 deg
  // rotation, (0, +h) maps to (+h, 0) -- +x, i.e. inward from x = 0. No extra `dy` push: unlike
  // the unrotated titles, ANY dy here is also rotated, and round 0's `dy="-1.6em"` (chosen to read
  // as "inward" in the pre-rotation frame) became a horizontal push in the WRONG direction
  // post-rotation -- the bug fix round 1 corrected. Left at the absolute x = 0 rather than
  // `rect.left`, same reasoning as the x-title above: once y-tick labels move to `rect.left` and
  // grow LEFT (`hasLeftGutter`), the tick-label column vacates x = 0, so the title -- still there,
  // growing right by about one line height -- stops overlapping it whenever the gutter has room
  // for both; see the function doc's GUTTER paragraph.
  yTitle.setAttribute("dominant-baseline", "hanging");
  yTitle.setAttribute("fill", "currentColor");
  yTitle.setAttribute("transform", `rotate(-90 0 ${height / 2})`);
  yTitle.textContent = yLabel;
  svg.append(yTitle);
}

/** One data series as a real `<polyline>`, `xs`/`ys` paired positionally and projected through
 * `toScreen`. `fill` is fixed to "none" -- not a visual choice but the structural fact that a
 * polyline drawn as a line chart must not also close and fill a shape; every actual visual
 * property (colour, width, dashing) is the caller's parameter. A dashed stroke's pattern is
 * expressed as a multiple of the caller's own `width` (never an absolute pixel count of ours),
 * so the dash scales with whatever line weight the caller chose instead of looking broken at one
 * weight and fine at another.
 *
 * `xs`/`ys` must be the same length: a mismatch would otherwise silently pair a value with
 * `undefined`, producing a `NaN` screen coordinate, and a `<polyline>` with any `NaN` in its
 * `points` attribute renders NOTHING -- the whole series vanishes with no error anywhere. Throwing
 * here instead is loud in exactly the way `mount.ts`'s per-widget try/catch expects: caught and
 * shown on the page, rather than a blank chart that looks like an empty-data widget.
 */
export function drawPolyline(svg: SVGSVGElement, v: View, xs: number[], ys: number[],
                             colour: string, width: number, dashed?: boolean): SVGPolylineElement {
  if (xs.length !== ys.length) {
    throw new Error(`drawPolyline: xs and ys must be the same length (${xs.length} vs ${ys.length})`);
  }
  const points = xs.map((x, i) => toScreen(v, x, ys[i]!)).map(([sx, sy]) => `${sx},${sy}`).join(" ");
  const line = el("polyline");
  line.setAttribute("points", points);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", colour);
  line.setAttribute("stroke-width", String(width));
  if (dashed) line.setAttribute("stroke-dasharray", `${width * 3} ${width * 2}`);
  svg.append(line);
  return line;
}

/** A single reference line at a fixed axis value, spanning the PLOT RECT (fix round 2 -- see
 * `drawAxes`'s own doc) -- vertical for `axis === "x"`, horizontal for `axis === "y"` -- the same
 * rect its own gridlines now use, recovered via `plotRectOf` since this signature has no ticks
 * parameter to derive one from directly. Decorative, not data, so it is hidden from assistive tech
 * the same way `drawAxes`'s gridlines are.
 */
export function drawGuide(svg: SVGSVGElement, v: View, axis: "x" | "y", value: number,
                          colour: string): SVGLineElement {
  const rect = plotRectOf(svg);
  const line = el("line");
  if (axis === "x") {
    const [sx] = toScreen(v, value, 0);
    line.setAttribute("x1", String(sx));
    line.setAttribute("y1", String(rect.top));
    line.setAttribute("x2", String(sx));
    line.setAttribute("y2", String(rect.bottom));
  } else {
    const [, sy] = toScreen(v, 0, value);
    line.setAttribute("x1", String(rect.left));
    line.setAttribute("y1", String(sy));
    line.setAttribute("x2", String(rect.right));
    line.setAttribute("y2", String(sy));
  }
  line.setAttribute("stroke", colour);
  line.setAttribute("aria-hidden", "true");
  svg.append(line);
  return line;
}
