import type { ChartStyle, FrontierBundle, MethodCurve } from "../frontier.js";
// Type-only, both of these: erased at compile time, so this module has NO runtime import of
// mount.js. A runtime one would recreate the cycle that made the whole bundle throw during module
// evaluation (see mount.ts's registration comment) -- this file must never import `register`.
import type { Widget } from "../mount.js";
import type { StateFactory } from "../state.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { createSvg, drawAxes, drawGuide, drawMarkers, drawPolyline } from "../render/svg.js";
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

// ---------------------------------------------------------------- what the widget draws with
//
// All of it comes from the BUNDLE (`scripts/gen_frontier_bundle.py`, whose `CHART` block takes five
// of these straight from `reblock.emit`, the module that draws the fallback PNG this widget
// replaces) -- no colour, width, label, tick target or pad is chosen in this file, and none is
// restated on the page. Fix round 1 moved them out of two JSON `data-*` attributes and into the
// bundle: same single source, and no literal `{`/`}` left in a markdown raw-HTML block, which was a
// risk that was measurable after all: `/usr/bin/python3` here has python-markdown 3.5.2 and every
// extension mkdocs.yml configures is core, and rendering the generated page through exactly that set
// shows the figure and all six attributes surviving intact. Scalars are still the better shape --
// but the earlier claim that nothing here could check it was wrong (fix round 2).
//
// The bundle is still a BOUNDARY -- it arrives over the network, and a page can outlive the artifact
// it was generated beside -- so these two converters validate it once, here, and throw rather than
// let a missing field reach an SVG attribute as "NaN".

export interface MethodStyle { label: string; colour: string }

function fields(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${what} is not an object: ${JSON.stringify(value)}`);
  }
  return value as Record<string, unknown>;
}

// `what` names the containing object in the error, so one set of validators serves the `chart` block
// and the bundle's own top level. Both are the same boundary: a page can outlive the artifact it was
// generated beside, and the top level is where the two numbers that scale the WHOLE chart live.
const CHART_WHAT = "frontier.json's chart";
const BUNDLE_WHAT = "frontier.json";

function numberField(o: Record<string, unknown>, key: string, what: string): number {
  if (!(key in o)) throw new Error(`${what} is missing "${key}"`);
  const v = o[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`${what} "${key}" is not a finite number: ${JSON.stringify(v)}`);
  }
  return v;
}

function stringField(o: Record<string, unknown>, key: string, what: string): string {
  if (!(key in o)) throw new Error(`${what} is missing "${key}"`);
  const v = o[key];
  if (typeof v !== "string" || v === "") {
    throw new Error(`${what} "${key}" is not a non-empty string: ${JSON.stringify(v)}`);
  }
  return v;
}

/** Finite is not enough for any of these: 0 passes `Number.isFinite` and then draws nothing while
 * throwing nowhere. A zero `line_width`, `guide_width` or `marker_radius` strokes nothing; a zero
 * `slider_step` makes every drag `NaN` (`Math.round(v / 0)`); a zero `permeability_max` collapses
 * the y axis; a zero `tick_target` divides by zero inside niceTicks; and a missing or zero
 * `frontier_xmax` -- the review's finding I1 -- put `NaN` in every one of the eight polylines while
 * the widget reported success and removed the fallback image, leaving a blank frame with no message
 * anywhere. That is the exact failure shape this file is built against, and it got in because the
 * one number that scales the whole x axis was read straight off the bundle. */
function positiveField(o: Record<string, unknown>, key: string, what: string): number {
  const v = numberField(o, key, what);
  if (!(v > 0)) throw new Error(`${what} "${key}" must be positive, got ${v}`);
  return v;
}

/** Convert the bundle's `chart` block into the declared type, or throw. A field the baker stopped
 * emitting would otherwise reach `stroke-width="NaN"` or a NaN-padded view -- both of which draw
 * nothing while throwing nowhere, which is exactly the failure shape this widget must not have. */
export function parseChart(value: unknown): ChartStyle {
  const o = fields(value, "frontier.json's chart");
  // `pad` is the one that may legitimately be zero (svg.ts supports a gutterless chart), but it is
  // a fraction of the box applied to BOTH sides, so at 0.5 the plot has no width left at all --
  // fitAxes would return a zero or negative scale and the chart would collapse silently.
  const pad = numberField(o, "pad", CHART_WHAT);
  if (!(pad >= 0 && pad < 0.5)) {
    throw new Error(`${CHART_WHAT} "pad" must be in [0, 0.5), got ${pad}`);
  }
  // Unlike the widths, 0 is MEANINGFUL here -- it is "no gridlines", which is exactly what the
  // fallback PNG draws -- so this is a range check rather than a positivity one.
  const gridOpacity = numberField(o, "grid_opacity", CHART_WHAT);
  if (!(gridOpacity >= 0 && gridOpacity <= 1)) {
    throw new Error(`${CHART_WHAT} "grid_opacity" must be in [0, 1], got ${gridOpacity}`);
  }
  return {
    x_label: stringField(o, "x_label", CHART_WHAT),
    y_label: stringField(o, "y_label", CHART_WHAT),
    line_width: positiveField(o, "line_width", CHART_WHAT),
    guide_colour: stringField(o, "guide_colour", CHART_WHAT),
    guide_width: positiveField(o, "guide_width", CHART_WHAT),
    guide_dash: stringField(o, "guide_dash", CHART_WHAT),
    marker_radius: positiveField(o, "marker_radius", CHART_WHAT),
    grid_opacity: gridOpacity,
    tick_target: positiveField(o, "tick_target", CHART_WHAT),
    pad,
    slider_step: positiveField(o, "slider_step", CHART_WHAT),
    permeability_max: positiveField(o, "permeability_max", CHART_WHAT),
  };
}

/** The legend name and curve colour for every method the bundle carries, keyed off its own
 * `Object.keys(methods)` -- never a list written here (the keys are longer than they look: the
 * arterial one is `greedy_arterial_access_displacement`, and its label, "Frontage (street-priced)",
 * cannot be reconstructed from it at all). Both are baked by the same `friendly_method_name` and
 * `method_colors` the fallback PNG's legend and curves use, so a missing one means the fetched
 * bundle predates that bake -- a curve would otherwise be drawn unlabelled, or with
 * `stroke="undefined"`, which renders as nothing. */
export function parseMethodStyles(methods: unknown): Map<string, MethodStyle> {
  const o = fields(methods, "frontier.json's methods");
  const out = new Map<string, MethodStyle>();
  for (const key of Object.keys(o)) {
    const curve = fields(o[key], `frontier.json's method ${key}`);
    const label = curve["label"];
    const colour = curve["colour"];
    if (typeof label !== "string" || label === "" || typeof colour !== "string" || colour === "") {
      throw new Error(`frontier.json has no label/colour for method ${key}: `
        + `${JSON.stringify({ label, colour })}`);
    }
    out.set(key, { label, colour });
  }
  return out;
}

/** A target read off the mount point. Throws rather than defaulting, unlike PermGraph's cosmetic
 * `data-prefix`: these two ARE the calibrated standards the caption beside them claims, so a
 * missing or malformed one must fail loudly instead of booting a chart whose guides quietly
 * contradict the sentence under it. `data-aspect` is read the same way -- it is the fallback
 * image's own shape, measured off that PNG's header by the generator. */
function requireFinite(raw: string | undefined, what: string): number {
  const n = raw === undefined ? Number.NaN : Number(raw);
  if (raw === undefined || raw === "" || !Number.isFinite(n)) {
    throw new Error(`frontier: ${what} is missing or not a number (${String(raw)})`);
  }
  return n;
}

function requirePositive(raw: string | undefined, what: string): number {
  const n = requireFinite(raw, what);
  if (!(n > 0)) throw new Error(`frontier: ${what} must be positive, got ${n}`);
  return n;
}

/** A boot target must sit ON the axis it marks -- 0 through the axis maximum inclusive. Throws
 * rather than clamping: a clamped target would silently disagree with the caption that states it,
 * which is the same contradiction the whole fix round was opened for. */
function inRange(value: number, max: number, what: string): number {
  if (!(value >= 0 && value <= max)) {
    throw new Error(`frontier: ${what} (${value}) is outside its axis [0, ${max}]`);
  }
  return value;
}

// ------------------------------------------------------------------------------------ the widget

interface FrontierState { targetDisplacement: number; targetPermeability: number; isolated: string | null }

/** The name every failure of this widget is reported under -- one constant, because it is used from
 * two unrelated places (the fetch chain and the resize callback) and two spellings of it would be a
 * reader seeing two different widgets fail. */
const LABEL = "Frontier";

export const frontier: Widget = (host, makeState) => {
  const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
  // A 404, a renamed bundle field, or any throw inside boot() must be VISIBLE on the page, not an
  // unhandled rejection in the console while the PNG fallback keeps the page looking correct --
  // the same pattern PermGraph settled on, for the same reason.
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<FrontierBundle>;
    })
    .then((b) => boot(host, makeState, b))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

/** Both axes are fractions in [0, 1] shown as PERCENTAGES, mirroring the
 * `PercentFormatter(xmax=1)` emit.compare_report puts on both axes of the figure this widget
 * replaces. Before fix round 1 this widget drew bare fractions, so a reader with JS off saw `60%`
 * where a reader with JS on saw `0.6`: the same chart contradicting itself on units. */
function percent(x: number, digits: number): string {
  return `${(x * 100).toFixed(digits)}%`;
}

// A TARGET is always a whole number of percentage points -- the bundle's `slider_step` is one
// percentage point, and both calibrated standards sit on that grid -- so it prints exactly under
// emit's own `{:.0%}`. A MEASURED value (a prefix's displacement or permeability) is on no grid at
// all, so it keeps a decimal: rounding 7.14% to 7% would be printing a number that was not measured.
const TARGET_DIGITS = 0;
const MEASURED_DIGITS = 1;

const formatTarget = (x: number): string => percent(x, TARGET_DIGITS);
const formatMeasured = (x: number): string => percent(x, MEASURED_DIGITS);

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
  const styles = parseMethodStyles(b.methods);
  const chart = parseChart(b.chart);
  // The one number that is genuinely the PAGE's and not the data's: the fallback image's aspect
  // ratio, which the generator measures off that PNG's own IHDR header. It describes the picture
  // this figure replaces, so it belongs beside it rather than in a bundle about methods.
  // M1: POSITIVE, not merely finite -- the resize callback divides by it, so 0 makes the box height Infinity
  // and every polyline coordinate NaN: finding I1's failure shape in the one field the I1 sweep did
  // not reach. Unreachable from the generator (a valid PNG cannot report a zero height), but the
  // rule this file states for the bundle's numbers applies to the page's number too.
  const aspect = requirePositive(host.dataset.aspect, "data-aspect");

  // The displayed window. x is clipped to the bundle's own `frontier_xmax` -- display only, the
  // same clip the fallback PNG applies, and nothing measured is lost by it (the full table stays
  // in the bundle and drives every number in the readout). y spans the whole permeability
  // fraction, which is what makes the last tick land exactly on the axis top, so svg.ts's plot
  // rect (recovered from the tick extremes) is the true data area rather than an inset of it.
  // I1: `frontier_xmax` scales the entire x axis and bounds `clipToXMax`, and `block_id` is quoted
  // in the readout, so both go through the same gate the chart block does. Read off the bundle
  // unvalidated they produced the branch's signature failure in its purest form -- eight all-NaN
  // polylines, no error text, and the fallback image removed anyway.
  const top = fields(b, BUNDLE_WHAT);
  const xMax = positiveField(top, "frontier_xmax", BUNDLE_WHAT);
  const blockId = stringField(top, "block_id", BUNDLE_WHAT);
  const yMax = chart.permeability_max;
  // One step for both axes, in the units the reader sees: a percentage point.
  const step = chart.slider_step;
  const xTicks = niceTicks(0, xMax, chart.tick_target);
  const yTicks = niceTicks(0, yMax, chart.tick_target);

  // M2: a target OUTSIDE its axis is finite, so `requireFinite` accepts it -- and then `drawGuide`
  // draws a line outside the plot rect, where the reader cannot see it, while the readout keeps
  // answering truthfully about a guide that is not on the chart. Both bounds are already in hand.
  const state = makeState<FrontierState>({
    targetDisplacement: inRange(
      requireFinite(host.dataset.targetDisplacement, "data-target-displacement"),
      xMax, "data-target-displacement"),
    targetPermeability: inRange(
      requireFinite(host.dataset.targetPermeability, "data-target-permeability"),
      yMax, "data-target-permeability"),
    isolated: null,
  });
  const s0 = state.get();

  const caption = host.querySelector("figcaption");

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
  xSlider.step = String(step);
  xSlider.value = String(s0.targetDisplacement);
  xSlider.addEventListener("input", () => {
    state.set({ targetDisplacement: clamp(Number(xSlider.value), 0, xMax) });
  });
  // Each slider carries its guide's CURRENT value, formatted exactly as the fallback PNG formats
  // the same two numbers in its legend (`matched displacement = 10%`, i.e. `{:.0%}`) -- so the two
  // charts state the two standards identically, and a dragged guide still names itself.
  const xValue = document.createElement("span");
  const xLabelEl = document.createElement("label");
  // Named off the axis titles themselves, and phrased as the CONSTRAINT each target expresses: the
  // x guide is a ceiling on cost, the y guide a floor on benefit, which is what the readout below
  // then answers against ("N of 8 methods reach ... within ...").
  xLabelEl.append(`Most ${chart.x_label} allowed: `, xValue, " ", xSlider);

  const ySlider = document.createElement("input");
  ySlider.type = "range";
  ySlider.min = "0";
  ySlider.max = String(yMax);
  ySlider.step = String(step);
  ySlider.value = String(s0.targetPermeability);
  ySlider.addEventListener("input", () => {
    state.set({ targetPermeability: clamp(Number(ySlider.value), 0, yMax) });
  });
  const yValue = document.createElement("span");
  const yLabelEl = document.createElement("label");
  yLabelEl.append(`Least ${chart.y_label} required: `, yValue, " ", ySlider);

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

  // Both assigned by the observer at the bottom of this function and by nothing else. Everything
  // that reads them -- `render`, the drag and hover handlers -- is wired from inside the first sized
  // callback, so there is no ordering in which they are read before that callback has run.
  let size: { width: number; height: number };
  let view: View;

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
    // NO `role="img"` (final review, M6). It plus an `aria-label` makes the whole subtree
    // presentational, which hides the `<text>` tick labels and axis titles from assistive tech --
    // and those being real, reachable text is half the stated reason this widget draws SVG rather
    // than canvas (see render/svg.ts's module doc). The chart's meaning is not left to them either:
    // the aria-live summary, the per-method verdicts and both guide labels below carry every number
    // as announced prose, and the figure keeps its own <figcaption>.
    drawAxes(svg, view, xTicks, yTicks, chart.x_label, chart.y_label, formatTarget,
             chart.grid_opacity);

    drawnScreen = [];
    for (const key of keys) {
      if (s.isolated !== null && s.isolated !== key) continue;
      const curve = b.methods[key]!;
      const clipped = clipToXMax(curve.displacement, curve.permeability, xMax);
      const colour = styles.get(key)!.colour;
      drawPolyline(svg, view, clipped.xs, clipped.ys, colour, chart.line_width);
      const n = insideCount(curve.displacement, xMax);
      // M1: the samples themselves, as the fallback PNG draws them (`marker="o"`). Only the REAL
      // samples get a dot -- never `clipToXMax`'s interpolated edge point, which is a drawing
      // artifact and not a measurement, and which the hover readout must never be able to name.
      drawMarkers(svg, view, curve.displacement.slice(0, n), curve.permeability.slice(0, n),
                  colour, chart.marker_radius);
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
      const guide = drawGuide(svg, view, axis, value, chart.guide_colour);
      guide.setAttribute("stroke-dasharray", chart.guide_dash);
      guide.setAttribute("stroke-width", String(chart.guide_width));
    }

    xSlider.value = String(s.targetDisplacement);
    ySlider.value = String(s.targetPermeability);
    xValue.textContent = formatTarget(s.targetDisplacement);
    yValue.textContent = formatTarget(s.targetPermeability);
    // M6: without this a screen reader announces the raw `0.1` while the label beside it reads 10%.
    xSlider.setAttribute("aria-valuetext", xValue.textContent);
    ySlider.setAttribute("aria-valuetext", yValue.textContent);
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
        item.textContent = `${label}: never reaches ${formatTarget(s.targetPermeability)} `
          + `permeability — it tops out at ${formatMeasured(best)}.`;
      } else if (curve.displacement[i]! <= s.targetDisplacement) {
        cleared++;
        item.textContent = `${label}: clears both at ${curve.road_m[i]!.toFixed(0)} m of road `
          + `(road ${i} of ${curve.road_m.length - 1}, `
          + `${formatMeasured(curve.displacement[i]!)} displaced).`;
      } else {
        item.textContent = `${label}: reaches ${formatTarget(s.targetPermeability)} permeability `
          + `only at ${formatMeasured(curve.displacement[i]!)} displaced, past the `
          + `${formatTarget(s.targetDisplacement)} budget `
          + `(${curve.road_m[i]!.toFixed(0)} m of road).`;
      }
      verdicts.append(item);
    }
    summary.textContent = `${cleared} of ${keys.length} methods reach `
      + `${formatTarget(s.targetPermeability)} permeability within `
      + `${formatTarget(s.targetDisplacement)} displacement on block ${blockId}.`;
  };

  /** A pointer event's position inside the chart, in the same pixel frame `view` maps world
   * coordinates into.
   *
   * TWO DIFFERENT BOXES MEET HERE, and they used to be one. `view` is built from the observer's
   * `contentRect`, which is the CONTENT box; `getBoundingClientRect()` returns the BORDER box,
   * because client coordinates exist in no other frame -- there is no API that gives a pointer's
   * position relative to an element's content box, so this cannot be avoided, only stated.
   *
   * They coincide exactly while `chartHost` has no border and no padding, which it has none of
   * today (it is a bare <div> this file creates, and it sets only `touch-action`). Give it either
   * and every drag is offset by that much: the guide would land where the pointer was not, on a
   * chart that still looks perfectly drawn, with nothing thrown anywhere. If such a rule ever
   * arrives it lands in docs/stylesheets/sbu.css, most plausibly on
   * `.md-typeset .sbu-figure-grid > figure` or a descendant of it -- and the fix is then to subtract
   * the computed border and padding here, not to re-measure `view` from the border box.
   */
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
    const s = state.get();
    // M2: a render rebuilds the whole SVG -- 8 polylines, 548 markers, 11 gridlines -- and a
    // pointermove that does not move the guide past a step boundary would rebuild all of it to draw
    // the identical picture. Skipping those is not the fix for the rebuild itself (that stays
    // deferred), just the redundant half of it.
    //
    // M4: `dragging` is explicitly "y" here rather than "everything that is not x", so the `null`
    // case is unrepresentable instead of silently meaning "drag the y guide".
    if (dragging === "x") {
      const next = clamp(snap(wx, step), 0, xMax);
      if (next !== s.targetDisplacement) state.set({ targetDisplacement: next });
    } else if (dragging === "y") {
      const next = clamp(snap(wy, step), 0, yMax);
      if (next !== s.targetPermeability) state.set({ targetPermeability: next });
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
      + `${formatMeasured(curve.displacement[bestIndex]!)} displaced, `
      + `${formatMeasured(curve.permeability[bestIndex]!)} permeability.`;
  };

  const wireInteraction = (): void => {
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
  };

  // The chart is laid out at the width the CONTAINER reports, every time that width changes -- not
  // measured once at mount and then only when the window moves, which missed every container-only
  // resize (Material's nav drawer, a <details>, print) and left this absolute-pixel SVG overflowing.
  // Height comes from the fallback PNG's own aspect ratio, so the widget keeps occupying the shape
  // of the image it replaces; only the width is ever measured.
  //
  // `runOrReport`: this callback is outside the mount's `.catch`, so without it a throw in here is
  // an unhandled rejection and a blank figure (see dom/error.ts).
  //
  // `drawnWidth` is BOTH the "have we drawn yet" flag (-1 until the first draw) and the guard
  // against a redraw feeding the observer back into itself: `chartHost` is 0 px tall until the SVG
  // lands in it, so our own first render changes the content box it is being observed by, and the
  // second callback carries the same width and would rebuild the identical picture -- 8 polylines
  // and 548 markers of it.
  let drawnWidth = -1;
  observeSize(chartHost, ({ width }) => runOrReport(host, LABEL, () => {
    if (width === drawnWidth) return;
    const first = drawnWidth < 0;
    size = { width, height: width / aspect };
    view = fitAxes([0, xMax], [0, yMax], size.width, size.height, chart.pad);
    render();
    drawnWidth = width;
    if (first) {
      state.subscribe(render);
      wireInteraction();
      // The fallback image goes only once a real chart has been drawn in its place. A throw anywhere
      // above lands in `runOrReport`, which replaces the caption with the failure -- and that message
      // is only honest while the static image it points at is still on the page.
      removeFallbackImage(host);
    }
  }));
}
