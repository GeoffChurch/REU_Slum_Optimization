import type { FrontierBundle, MethodCurve } from "../frontier.js";
// Type-only, both of these: erased at compile time, so this module has NO runtime import of
// mount.js. A runtime one would recreate the cycle that made the whole bundle throw during module
// evaluation (see mount.ts's registration comment) -- this file must never import `register`.
import type { Widget } from "../mount.js";
import type { StateFactory } from "../state.js";
import { createSvg, drawAxes, drawGuide, drawPolyline } from "../render/svg.js";
import { niceTicks } from "../view/ticks.js";
import { fitAxes, nearest, toScreen, toWorld, type View } from "../view/transform.js";

/** Least index whose value reaches `target`, or -1. Binary search, valid because the baked arrays
 * are monotone in prefix -- the same search budget.prefix_to_permeability performs in Python, over
 * the same sequence. Asserted monotone by tests/test_frontier_bundle.py. */
export function leastClearing(values: number[], target: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid]! >= target) hi = mid; else lo = mid + 1;
  }
  return lo < values.length ? lo : -1;
}

/** How many leading samples sit inside the displayed x window. A count, not a filter, because
 * monotone x makes the inside samples a PREFIX -- which is what lets a hover result found among
 * them index straight back into the baked arrays with no index map. */
function insideCount(xs: number[], xMax: number): number {
  const first = xs.findIndex((x) => x > xMax);
  return first === -1 ? xs.length : first;
}

/** The samples inside the displayed x window, plus a final point ON the window's right edge when
 * the curve runs past it (clearance_looped reaches 0.83 displacement against a 0.4 window).
 *
 * That edge point is interpolated, and interpolating is a DRAWING act here, never a reported one:
 * matplotlib's own axis clipping draws exactly this line on the fallback PNG (emit.py's
 * `set_xlim`), while every number this widget shows as text comes from `leastClearing` over the
 * baked samples. Truncating at the last sample inside the window instead would end the line short
 * of the edge, which reads as the method terminating there.
 *
 * `xs[n] > xMax >= xs[n - 1]` holds by construction (monotone x, `insideCount` cutting at the
 * FIRST sample past the window), so the divisor below is strictly positive -- worth stating
 * because a zero-width segment would put NaN in the `points` attribute and a `<polyline>` with any
 * NaN in it renders NOTHING, silently. The real bundle does contain zero-width segments
 * (greedy_arterial_access_displacement's first three samples all sit at displacement 0.0), but
 * they can only occur INSIDE the window, never at the crossing.
 */
export function clipToXMax(xs: number[], ys: number[], xMax: number): { xs: number[]; ys: number[] } {
  if (xs.length !== ys.length) {
    throw new Error(`clipToXMax: xs and ys must be the same length (${xs.length} vs ${ys.length})`);
  }
  const n = insideCount(xs, xMax);
  const outX = xs.slice(0, n);
  const outY = ys.slice(0, n);
  const px = xs[n - 1];
  const py = ys[n - 1];
  const qx = xs[n];
  const qy = ys[n];
  if (px !== undefined && py !== undefined && qx !== undefined && qy !== undefined) {
    outX.push(xMax);
    outY.push(py + ((xMax - px) / (qx - px)) * (qy - py));
  }
  return { xs: outX, ys: outY };
}

// ------------------------------------------------------------------ the mount point's own data
//
// Every colour, width, label, tick target and pad this widget draws with arrives from the mount
// point, emitted by scripts/gen_site_pages.py from the bundle and from the values the fallback PNG
// itself was drawn with (see `_frontier_figure` there). None is chosen in this file: the project's
// rule is that a visual choice is configuration, resolved once upstream, not a TypeScript literal.
// Two JSON attributes rather than a dozen data-* ones, converted ONCE here at the boundary into
// declared types, so exactly one place in the widget deals in strings.

export interface MethodStyle { label: string; colour: string }

export interface ChartStyle {
  xLabel: string;
  yLabel: string;
  lineWidth: number;
  guideColour: string;
  guideDash: string;
  tickTarget: number;
  pad: number;
  aspect: number;
  sliderDivisions: number;
  permeabilityMax: number;
}

function jsonObject(raw: string, what: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${what} is not a JSON object: ${raw}`);
  }
  return parsed as Record<string, unknown>;
}

function chartNumber(o: Record<string, unknown>, key: string): number {
  if (!(key in o)) throw new Error(`data-chart is missing "${key}"`);
  const v = o[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`data-chart's "${key}" is not a finite number: ${JSON.stringify(v)}`);
  }
  return v;
}

function chartString(o: Record<string, unknown>, key: string): string {
  if (!(key in o)) throw new Error(`data-chart is missing "${key}"`);
  const v = o[key];
  if (typeof v !== "string" || v === "") {
    throw new Error(`data-chart's "${key}" is not a non-empty string: ${JSON.stringify(v)}`);
  }
  return v;
}

/** Finite is not enough for these five: 0 passes `Number.isFinite` and then draws nothing while
 * throwing nowhere. A zero `aspect` makes the box height Infinity and every y = 0 coordinate NaN
 * (and one NaN voids a whole `<polyline>`); a zero `lineWidth` strokes nothing; a zero
 * `sliderDivisions` makes the slider step Infinity; a zero `permeabilityMax` collapses the y axis;
 * a zero `tickTarget` divides by zero inside niceTicks. */
function chartPositive(o: Record<string, unknown>, key: string): number {
  const v = chartNumber(o, key);
  if (!(v > 0)) throw new Error(`data-chart's "${key}" must be positive, got ${v}`);
  return v;
}

/** Convert `data-chart` into a declared type, or throw. A field the generator stopped emitting
 * would otherwise reach `stroke-width="NaN"` or a NaN-padded view -- both of which draw nothing
 * while throwing nowhere, which is exactly the failure shape this widget must not have. */
export function parseChart(raw: string): ChartStyle {
  const o = jsonObject(raw, "data-chart");
  // `pad` is the one that may legitimately be zero (svg.ts supports a gutterless chart), but it is
  // a fraction of the box applied to BOTH sides, so at 0.5 the plot has no width left at all --
  // fitAxes would return a zero or negative scale and the chart would collapse silently.
  const pad = chartNumber(o, "pad");
  if (!(pad >= 0 && pad < 0.5)) throw new Error(`data-chart's "pad" must be in [0, 0.5), got ${pad}`);
  return {
    xLabel: chartString(o, "xLabel"),
    yLabel: chartString(o, "yLabel"),
    lineWidth: chartPositive(o, "lineWidth"),
    guideColour: chartString(o, "guideColour"),
    guideDash: chartString(o, "guideDash"),
    tickTarget: chartPositive(o, "tickTarget"),
    pad,
    aspect: chartPositive(o, "aspect"),
    sliderDivisions: chartPositive(o, "sliderDivisions"),
    permeabilityMax: chartPositive(o, "permeabilityMax"),
  };
}

/** A label and a colour for every method the BUNDLE carries -- keyed off `Object.keys(b.methods)`,
 * never a list written here (the keys are longer than they look: the arterial one is
 * `greedy_arterial_access_displacement`). Missing means the page's styles and the fetched bundle
 * disagree about which methods exist, which would otherwise silently drop one curve from a chart
 * of eight on a page that still looks entirely correct. */
export function parseMethodStyles(raw: string, keys: readonly string[]): Map<string, MethodStyle> {
  const o = jsonObject(raw, "data-methods");
  const out = new Map<string, MethodStyle>();
  for (const key of keys) {
    const entry = o[key];
    const label = typeof entry === "object" && entry !== null
      ? (entry as Record<string, unknown>)["label"] : undefined;
    const colour = typeof entry === "object" && entry !== null
      ? (entry as Record<string, unknown>)["colour"] : undefined;
    if (typeof label !== "string" || label === "" || typeof colour !== "string" || colour === "") {
      throw new Error(`data-methods has no label/colour for method ${key}: ${JSON.stringify(entry)}`);
    }
    out.set(key, { label, colour });
  }
  return out;
}

/** A target read off the mount point. Throws rather than defaulting, unlike PermGraph's cosmetic
 * `data-prefix`: these two ARE the calibrated standards the caption beside them claims, so a
 * missing or malformed one must fail loudly instead of booting a chart whose guides quietly
 * contradict the sentence under it. */
function requireFinite(raw: string | undefined, what: string): number {
  const n = raw === undefined ? Number.NaN : Number(raw);
  if (raw === undefined || raw === "" || !Number.isFinite(n)) {
    throw new Error(`frontier: ${what} is missing or not a number (${String(raw)})`);
  }
  return n;
}

function requireAttr(raw: string | undefined, what: string): string {
  if (raw === undefined || raw === "") throw new Error(`frontier: ${what} is missing`);
  return raw;
}

// ------------------------------------------------------------------------------------ the widget

interface FrontierState { targetDisplacement: number; targetPermeability: number; isolated: string | null }

export const frontier: Widget = (host, makeState) => {
  const src = requireAttr(host.dataset.bundle, "data-bundle");
  // A 404, a renamed bundle field, or any throw inside boot() must be VISIBLE on the page, not an
  // unhandled rejection in the console while the PNG fallback keeps the page looking correct --
  // the same pattern PermGraph settled on, for the same reason.
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<FrontierBundle>;
    })
    .then((b) => boot(host, makeState, b))
    .catch((err: unknown) => showError(host, err));
};

function showError(host: HTMLElement, err: unknown): void {
  const message = `Frontier failed to load: ${err instanceof Error ? err.message : String(err)}`;
  const caption = host.querySelector("figcaption");
  if (caption) {
    caption.textContent = message;
  } else {
    const p = document.createElement("p");
    p.textContent = message;
    host.append(p);
  }
}

/** The chart's pixel box: the host element's own width, and a height from the fallback PNG's true
 * aspect ratio (`data-chart`'s `aspect`, read from the PNG's own header by the generator), so the
 * widget occupies the shape of the image it replaces. Throws on a non-positive width rather than
 * building a zero-width SVG, which would look exactly like a widget that mounted and then failed
 * to draw. */
function measure(el: HTMLElement, aspect: number): { width: number; height: number } {
  const { width } = el.getBoundingClientRect();
  if (!(width > 0)) throw new Error(`frontier: the chart box has no width (${width})`);
  return { width, height: width / aspect };
}

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Quantise a DRAGGED target onto the slider's own step, so the guide, the slider thumb and the
 * readout can never disagree about where the target is (a range input snaps any value assigned to
 * it onto its step, so an unquantised drag would leave the thumb somewhere the line is not). The
 * BOOT targets are deliberately not snapped: those are the calibrated standards, and they must
 * appear exactly as the caption states them. */
function snap(v: number, step: number): number {
  return Math.round(v / step) * step;
}

function boot(host: HTMLElement, makeState: StateFactory, b: FrontierBundle): void {
  const keys = Object.keys(b.methods);
  if (keys.length === 0) throw new Error("frontier: the bundle carries no methods");
  const styles = parseMethodStyles(requireAttr(host.dataset.methods, "data-methods"), keys);
  const chart = parseChart(requireAttr(host.dataset.chart, "data-chart"));

  // The displayed window. x is clipped to the bundle's own `frontier_xmax` -- display only, the
  // same clip the fallback PNG applies, and nothing measured is lost by it (the full table stays
  // in the bundle and drives every number in the readout). y spans the whole permeability
  // fraction, which is what makes the last tick land exactly on the axis top, so svg.ts's plot
  // rect (recovered from the tick extremes) is the true data area rather than an inset of it.
  const xMax = b.frontier_xmax;
  const yMax = chart.permeabilityMax;
  const xStep = xMax / chart.sliderDivisions;
  const yStep = yMax / chart.sliderDivisions;
  const xTicks = niceTicks(0, xMax, chart.tickTarget);
  const yTicks = niceTicks(0, yMax, chart.tickTarget);

  const state = makeState<FrontierState>({
    targetDisplacement: requireFinite(host.dataset.targetDisplacement, "data-target-displacement"),
    targetPermeability: requireFinite(host.dataset.targetPermeability, "data-target-permeability"),
    isolated: null,
  });
  const s0 = state.get();

  const caption = host.querySelector("figcaption");
  const fallback = host.querySelector("img");

  const chartHost = document.createElement("div");
  // Pointer events on the guides are drags, not scrolls: without this a touch drag pans the page.
  chartHost.style.touchAction = "none";

  const controls = document.createElement("div");

  // Both native range inputs -- the keyboard and screen-reader path to the same two targets the
  // pointer drags. `min` is 0 on both axes because both are fractions rooted at zero, which the
  // bundle itself asserts (tests/test_frontier_bundle.py::test_starts_at_zero).
  const xSlider = document.createElement("input");
  xSlider.type = "range";
  xSlider.min = "0";
  xSlider.max = String(xMax);
  xSlider.step = String(xStep);
  xSlider.value = String(s0.targetDisplacement);
  xSlider.addEventListener("input", () => {
    state.set({ targetDisplacement: clamp(Number(xSlider.value), 0, xMax) });
  });
  const xLabelEl = document.createElement("label");
  // Named off the axis titles themselves, and phrased as the CONSTRAINT each target expresses: the
  // x guide is a ceiling on cost, the y guide a floor on benefit, which is what the readout below
  // then answers against ("N of 8 methods reach ... within ...").
  xLabelEl.append(`Most ${chart.xLabel} allowed `, xSlider);

  const ySlider = document.createElement("input");
  ySlider.type = "range";
  ySlider.min = "0";
  ySlider.max = String(yMax);
  ySlider.step = String(yStep);
  ySlider.value = String(s0.targetPermeability);
  ySlider.addEventListener("input", () => {
    state.set({ targetPermeability: clamp(Number(ySlider.value), 0, yMax) });
  });
  const yLabelEl = document.createElement("label");
  yLabelEl.append(`Least ${chart.yLabel} required `, ySlider);

  // Legend entries are real <button>s, so isolating one method is reachable by keyboard and
  // announces its state; each is painted in its own curve's colour, which is the only place the
  // legend and the chart have to agree.
  const legend = document.createElement("div");
  const buttons = new Map<string, HTMLButtonElement>();
  for (const key of keys) {
    const style = styles.get(key)!;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = style.label;
    btn.style.color = style.colour;
    btn.addEventListener("click", () => {
      state.set({ isolated: state.get().isolated === key ? null : key });
    });
    buttons.set(key, btn);
    legend.append(btn);
  }

  const summary = document.createElement("p");
  // The one line that changes on every drag, announced: a screen-reader user moving the slider with
  // the arrow keys hears the answer change, instead of the picture changing silently.
  summary.setAttribute("aria-live", "polite");
  const verdicts = document.createElement("ul");
  const hoverOut = document.createElement("p");
  controls.append(xLabelEl, yLabelEl, legend, summary, verdicts, hoverOut);

  // Chart then controls, both BEFORE the figcaption, so this figure keeps the
  // picture-then-caption reading order every sibling figure on the page uses.
  if (caption) {
    host.insertBefore(chartHost, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(chartHost, controls);
  }

  let size = measure(chartHost, chart.aspect);
  let view: View = fitAxes([0, xMax], [0, yMax], size.width, size.height, chart.pad);

  // Screen-space positions of the DRAWN samples, per visible method, rebuilt by render(). Screen
  // space and not world space because "nearest" must mean nearest in pixels: a world-space
  // distance would mix a displacement in [0, 0.4] with a permeability in [0, 1] and snap to the
  // wrong sample near a steep rise. Index i here is index i in the baked arrays -- the drawn
  // samples are a prefix (see insideCount) -- so a hover result reports measured values directly.
  let drawnScreen: { key: string; xs: number[]; ys: number[] }[] = [];

  const render = (): void => {
    const s = state.get();
    chartHost.replaceChildren();
    const svg = createSvg(chartHost, size.width, size.height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label",
      `${chart.yLabel} against ${chart.xLabel} for ${keys.length} methods, with a target line on `
      + `each axis. Every number in the chart is repeated as text below it.`);
    drawAxes(svg, view, xTicks, yTicks, chart.xLabel, chart.yLabel);

    drawnScreen = [];
    for (const key of keys) {
      if (s.isolated !== null && s.isolated !== key) continue;
      const curve = b.methods[key]!;
      const clipped = clipToXMax(curve.displacement, curve.permeability, xMax);
      drawPolyline(svg, view, clipped.xs, clipped.ys, styles.get(key)!.colour, chart.lineWidth);
      const n = insideCount(curve.displacement, xMax);
      const xs: number[] = [];
      const ys: number[] = [];
      for (let i = 0; i < n; i++) {
        const [sx, sy] = toScreen(view, curve.displacement[i]!, curve.permeability[i]!);
        xs.push(sx);
        ys.push(sy);
      }
      drawnScreen.push({ key, xs, ys });
    }

    for (const [axis, value] of [["x", s.targetDisplacement],
                                 ["y", s.targetPermeability]] as const) {
      drawGuide(svg, view, axis, value, chart.guideColour)
        .setAttribute("stroke-dasharray", chart.guideDash);
    }

    xSlider.value = String(s.targetDisplacement);
    ySlider.value = String(s.targetPermeability);
    for (const [key, btn] of buttons) {
      btn.setAttribute("aria-pressed", String(s.isolated === key));
    }
    writeVerdicts(s);
  };

  /** Every number the picture shows, as text: for the two current targets, which methods clear
   * them and at what least road. The least road is `leastClearing`'s answer over the baked
   * permeability column -- the same binary search Python runs over the same sequence -- and the
   * displacement reported beside it is the baked value at that same index, never an interpolation
   * between two real prefixes. */
  const writeVerdicts = (s: FrontierState): void => {
    verdicts.replaceChildren();
    let cleared = 0;
    for (const key of keys) {
      const curve: MethodCurve = b.methods[key]!;
      const label = styles.get(key)!.label;
      const item = document.createElement("li");
      const i = leastClearing(curve.permeability, s.targetPermeability);
      if (i === -1) {
        const best = curve.permeability[curve.permeability.length - 1]!;
        item.textContent = `${label}: never reaches ${pct(s.targetPermeability)} permeability `
          + `— it tops out at ${pct(best)}.`;
      } else if (curve.displacement[i]! <= s.targetDisplacement) {
        cleared++;
        item.textContent = `${label}: clears both at ${curve.road_m[i]!.toFixed(0)} m of road `
          + `(road ${i} of ${curve.road_m.length - 1}, ${pct(curve.displacement[i]!)} displaced).`;
      } else {
        item.textContent = `${label}: reaches ${pct(s.targetPermeability)} permeability only at `
          + `${pct(curve.displacement[i]!)} displaced, past the ${pct(s.targetDisplacement)} `
          + `budget (${curve.road_m[i]!.toFixed(0)} m of road).`;
      }
      verdicts.append(item);
    }
    summary.textContent = `${cleared} of ${keys.length} methods reach `
      + `${pct(s.targetPermeability)} permeability within ${pct(s.targetDisplacement)} `
      + `displacement on block ${b.block_id}.`;
  };

  state.subscribe(render);

  const localPoint = (ev: PointerEvent): [number, number] => {
    const r = chartHost.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  };

  // Which guide a press takes hold of is decided by which one the pointer is NEARER to, in
  // pixels. No grab radius to pick (a pixel tolerance would be one more drawn number), and a
  // press always takes hold of something, which is what makes the affordance discoverable at all.
  let dragging: "x" | "y" | null = null;

  const dragTo = (sx: number, sy: number): void => {
    const [wx, wy] = toWorld(view, sx, sy);
    if (dragging === "x") {
      state.set({ targetDisplacement: clamp(snap(wx, xStep), 0, xMax) });
    } else {
      state.set({ targetPermeability: clamp(snap(wy, yStep), 0, yMax) });
    }
  };

  /** Hover snaps to the nearest measured prefix instead of interpolating between two of them: the
   * baked table is discrete, so a continuous readout would put a number on the page that is not a
   * measurement (spec §Open items). */
  const hoverAt = (sx: number, sy: number): void => {
    let bestKey: string | null = null;
    let bestIndex = -1;
    let bestDist = Infinity;
    for (const d of drawnScreen) {
      const i = nearest(d.xs, d.ys, sx, sy);
      if (i < 0) continue;
      const dist = (d.xs[i]! - sx) ** 2 + (d.ys[i]! - sy) ** 2;
      if (dist < bestDist) { bestDist = dist; bestKey = d.key; bestIndex = i; }
    }
    if (bestKey === null) { hoverOut.textContent = ""; return; }
    const curve = b.methods[bestKey]!;
    hoverOut.textContent = `Nearest measured prefix: ${styles.get(bestKey)!.label}, road `
      + `${bestIndex} of ${curve.road_m.length - 1} — ${curve.road_m[bestIndex]!.toFixed(0)} m, `
      + `${pct(curve.displacement[bestIndex]!)} displaced, `
      + `${pct(curve.permeability[bestIndex]!)} permeability.`;
  };

  chartHost.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    const [sx, sy] = localPoint(ev);
    const s = state.get();
    const [guideX] = toScreen(view, s.targetDisplacement, 0);
    const [, guideY] = toScreen(view, 0, s.targetPermeability);
    dragging = Math.abs(sx - guideX) <= Math.abs(sy - guideY) ? "x" : "y";
    chartHost.setPointerCapture(ev.pointerId);
    dragTo(sx, sy);
  });
  chartHost.addEventListener("pointermove", (ev) => {
    const [sx, sy] = localPoint(ev);
    if (dragging !== null) dragTo(sx, sy); else hoverAt(sx, sy);
  });
  const endDrag = (): void => { dragging = null; };
  chartHost.addEventListener("pointerup", endDrag);
  chartHost.addEventListener("pointercancel", endDrag);

  window.addEventListener("resize", () => {
    size = measure(chartHost, chart.aspect);
    view = fitAxes([0, xMax], [0, yMax], size.width, size.height, chart.pad);
    render();
  });

  render();
  // The fallback image goes only once a real chart has been drawn in its place. A throw anywhere
  // above lands in the widget's own `.catch`, which replaces the caption with the failure -- and
  // that message is only honest while the static image it points at is still on the page.
  if (fallback) fallback.remove();
}
