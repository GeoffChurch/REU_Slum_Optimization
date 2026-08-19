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
 * have room for them; `drawAxes` does not (its signature is fixed by the brief this module was
 * built against), so its chrome uses `currentColor` for paint -- a keyword that defers the actual
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

/** The pixel size `createSvg` gave `svg` -- read back rather than threaded through every call, so
 * axis chrome and guides can span the full drawing surface without the caller repeating it. */
function sizeOf(svg: SVGSVGElement): { width: number; height: number } {
  return { width: Number(svg.getAttribute("width")), height: Number(svg.getAttribute("height")) };
}

/** Draw both axes into `svg`: one full-span gridline per tick (so a tick's "length" is the data --
 * the other axis's own drawing extent -- rather than an arbitrary chosen pixel), a real `<text>`
 * label per tick, and the two axis titles. `xTicks`/`yTicks` are `niceTicks` output: round-step
 * values the caller already computed, drawn here and nowhere invented.
 *
 * CONTAINMENT (fix round 1, Critical): every label must render inside `[0, width] x [0, height]`,
 * even at `pad = 0` -- i.e. even when a tick's screen position lands EXACTLY on the SVG's own
 * outer edge, with no gutter to spare. Round 0 got this backwards: it anchored text at the edge
 * and then grew it OUTWARD (`dominant-baseline="hanging"` at `y = height`, a rotated title's `dy`
 * whose sign flipped the wrong way through the rotation matrix) -- safe only when a caller-chosen
 * `pad` happened to leave room, which the contract explicitly does not get to assume.
 *
 * The fix does not compute a numeric pixel clamp, because that would need to know how many pixels
 * "1em" resolves to, and `drawAxes` is deliberately given no font-size parameter to know it with
 * (see the module doc). Instead every label's anchor and growth direction is chosen so containment
 * holds by CONSTRUCTION, given only geometry already in hand:
 *   - the row that sits at the fixed outer edge (x-tick labels, both titles) grows INWARD from
 *     that edge (`dominant-baseline="alphabetic"` grows up from `y = height`; the y-title's
 *     `dominant-baseline="hanging"` grows, after its -90 deg rotation, in +x from `x = 0` -- worked
 *     through by hand: pre-rotation offset (0, +h) maps to post-rotation (+h, 0), i.e. toward the
 *     box interior, never past x = 0);
 *   - the row whose anchor moves PER TICK (x-tick labels horizontally, y-tick labels vertically)
 *     picks its anchor/baseline by comparing that tick's own screen position against the box edge
 *     (`edgeAnchor`/`edgeBaseline` below), so a tick sitting exactly on the boundary grows away
 *     from it instead of straddling it, and an interior tick still centers normally.
 * Row-stacking (tick label under the axis line, title further inward again) still uses `em`
 * (font-relative units) for the SEPARATION between the two rows -- that reasoning was reviewed and
 * kept: it is a structural constant a working two-row stack needs, the same category as
 * niceTicks's [1, 2, 2.5, 5, 10], not a drawn value. What changed is only the SIGN: both offsets
 * now push inward (negative, toward the box), so no magnitude of the row gap can carry a label
 * outside -- it can at most overshoot the OPPOSITE edge on a pathologically short chart, not this
 * one.
 *
 * One consequence worth stating plainly: because inward is the only direction proven safe without
 * a font-size, a caller's `pad` (`fitAxes`'s gutter) is not spent even when it exists -- labels
 * always hug the inner side of the outer edge rather than floating in the middle of a generous
 * gutter. Adaptive use of a generous pad would need either a font-size parameter this signature
 * does not have, or a real DOM measurement (`getBBox`) unavailable outside a browser -- both out of
 * scope for this fix, which is about containment, not layout polish.
 */
export function drawAxes(svg: SVGSVGElement, v: View, xTicks: number[], yTicks: number[],
                         xLabel: string, yLabel: string): void {
  const { width, height } = sizeOf(svg);

  // A tick whose screen position lands exactly on the box edge (the real Frontier x axis is
  // [0, 0.4] with pad 0, so its first/last tick DOES land exactly on x = 0 / x = width) must not
  // grow its label symmetrically about that point -- half the glyph would sit outside. Comparing
  // against the known-safe absolute edges (0 and the box's own width/height, not an assumed glyph
  // size) is enough to pick a direction that is always safe, and falls back to the ordinary
  // centered/symmetric mode for every interior tick.
  const edgeAnchor = (s: number): "start" | "middle" | "end" =>
    s <= 0 ? "start" : s >= width ? "end" : "middle";
  const edgeBaseline = (s: number): "hanging" | "middle" | "alphabetic" =>
    s <= 0 ? "hanging" : s >= height ? "alphabetic" : "middle";
  // "alphabetic" grows mostly upward from its anchor, but still reserves a small allowance BELOW
  // the anchor for a descender (g, y, p, ...) -- so a glyph anchored EXACTLY on the bottom/right
  // edge still pokes out by that allowance, which is what the bounds test below caught in this
  // fix's first pass. Digits/'.'/'-' never actually descend in any common font, but a future tick
  // format could add a unit suffix that does, so the margin is closed structurally instead of by
  // trusting today's character set: nudge alphabetic-anchored text a small further step inward.
  // Negative is always the safe direction here (see the function doc), so the exact magnitude is
  // not load-bearing the way it would be if it could push the wrong way.
  const alphabeticDy = "-0.25em";

  for (const t of xTicks) {
    const [sx] = toScreen(v, t, 0);
    const line = el("line");
    line.setAttribute("x1", String(sx));
    line.setAttribute("y1", "0");
    line.setAttribute("x2", String(sx));
    line.setAttribute("y2", String(height));
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("aria-hidden", "true");
    svg.append(line);

    const label = el("text");
    label.setAttribute("x", String(sx));
    label.setAttribute("y", String(height));
    label.setAttribute("dy", alphabeticDy);
    label.setAttribute("text-anchor", edgeAnchor(sx));
    label.setAttribute("dominant-baseline", "alphabetic");
    label.setAttribute("fill", "currentColor");
    label.textContent = String(t);
    svg.append(label);
  }

  for (const t of yTicks) {
    const [, sy] = toScreen(v, 0, t);
    const line = el("line");
    line.setAttribute("x1", "0");
    line.setAttribute("y1", String(sy));
    line.setAttribute("x2", String(width));
    line.setAttribute("y2", String(sy));
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("aria-hidden", "true");
    svg.append(line);

    const baseline = edgeBaseline(sy);
    const label = el("text");
    label.setAttribute("x", "0");
    label.setAttribute("y", String(sy));
    if (baseline === "alphabetic") label.setAttribute("dy", alphabeticDy);
    label.setAttribute("text-anchor", "start");
    label.setAttribute("dominant-baseline", baseline);
    label.setAttribute("fill", "currentColor");
    label.textContent = String(t);
    svg.append(label);
  }

  const xTitle = el("text");
  xTitle.setAttribute("x", String(width / 2));
  xTitle.setAttribute("y", String(height));
  // Negative: further INWARD (up) than the tick-label row, one row's worth of em. The old code
  // pushed positive (down, "hanging"), which is the direction that overflowed.
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
  // the unrotated titles, ANY dy here is also rotated, and the old code's `dy="-1.6em"` (chosen to
  // read as "inward" in the pre-rotation frame) became a horizontal push in the WRONG direction
  // post-rotation -- exactly the bug the review's hand-derived matrix caught. Leaving it at 0
  // keeps the title flush against the tick-label column rather than further separated from it;
  // that overlap is a cosmetic cost accepted in this fix, not a containment violation.
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

/** A single reference line at a fixed axis value, spanning the full drawing surface -- vertical
 * for `axis === "x"`, horizontal for `axis === "y"` -- the same full-span convention `drawAxes`
 * uses for its own tick gridlines, so a guide reads as part of the same chart grammar. Decorative,
 * not data, so it is hidden from assistive tech the same way `drawAxes`'s gridlines are.
 */
export function drawGuide(svg: SVGSVGElement, v: View, axis: "x" | "y", value: number,
                          colour: string): SVGLineElement {
  const { width, height } = sizeOf(svg);
  const line = el("line");
  if (axis === "x") {
    const [sx] = toScreen(v, value, 0);
    line.setAttribute("x1", String(sx));
    line.setAttribute("y1", "0");
    line.setAttribute("x2", String(sx));
    line.setAttribute("y2", String(height));
  } else {
    const [, sy] = toScreen(v, 0, value);
    line.setAttribute("x1", "0");
    line.setAttribute("y1", String(sy));
    line.setAttribute("x2", String(width));
    line.setAttribute("y2", String(sy));
  }
  line.setAttribute("stroke", colour);
  line.setAttribute("aria-hidden", "true");
  svg.append(line);
  return line;
}
