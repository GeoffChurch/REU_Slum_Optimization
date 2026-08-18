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
 * Row-stacking (tick label under the axis line, title under the tick label) needs *some* vertical
 * separation between the two text rows, and there is no pixel parameter to spend on it -- so both
 * offsets below are in `em` (font-relative units), scaling with whatever font-size the page's CSS
 * gives the widget, the same way niceTicks's [1, 2, 2.5, 5, 10] is the step algorithm's basis
 * rather than a drawn value: a structural constant a working two-row label stack needs, not an
 * aesthetic pixel guess.
 */
export function drawAxes(svg: SVGSVGElement, v: View, xTicks: number[], yTicks: number[],
                         xLabel: string, yLabel: string): void {
  const { width, height } = sizeOf(svg);

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
    label.setAttribute("dy", "1em");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("dominant-baseline", "hanging");
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

    const label = el("text");
    label.setAttribute("x", "0");
    label.setAttribute("y", String(sy));
    label.setAttribute("text-anchor", "start");
    label.setAttribute("dominant-baseline", "middle");
    label.setAttribute("fill", "currentColor");
    label.textContent = String(t);
    svg.append(label);
  }

  const xTitle = el("text");
  xTitle.setAttribute("x", String(width / 2));
  xTitle.setAttribute("y", String(height));
  xTitle.setAttribute("dy", "2.2em");
  xTitle.setAttribute("text-anchor", "middle");
  xTitle.setAttribute("dominant-baseline", "hanging");
  xTitle.setAttribute("fill", "currentColor");
  xTitle.textContent = xLabel;
  svg.append(xTitle);

  const yTitle = el("text");
  yTitle.setAttribute("x", "0");
  yTitle.setAttribute("y", String(height / 2));
  yTitle.setAttribute("dy", "-1.6em");
  yTitle.setAttribute("text-anchor", "middle");
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
 */
export function drawPolyline(svg: SVGSVGElement, v: View, xs: number[], ys: number[],
                             colour: string, width: number, dashed?: boolean): SVGPolylineElement {
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
