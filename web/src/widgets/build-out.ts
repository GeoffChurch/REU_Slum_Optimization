// Type-only: erased at compile time, so this module has NO runtime import of mount.js (the cycle
// mount.ts's registration comment describes). This file must never import `register`.
import type {
  DemoBuildOut, DemoEncoding, DemoMet, DemoSeries, DemoSheet, DemoShort, DemoStep,
} from "../demo.js";
import type { Widget } from "../mount.js";
import type { StateFactory, StateSource } from "../state.js";
import { intParam, type UrlCodec } from "../url/param.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { createSvg, drawAxes, drawCursor, drawMarkers, drawPolyline,
         type Cursor } from "../render/svg.js";
import { niceTicks } from "../view/ticks.js";
import type { View } from "../view/transform.js";

/** One block's build-out under one budget slider, for one method or several side by side
 * (a DemoBuildOut: its budgets, and a series per method with a cell per budget).
 *
 * One series is the Demo's own block: the plate beside its three plots. Several are the methods
 * compared on another block: a plate per method side by side, each headed by its method and its
 * whole run time, and the three plots beneath with a curve and a moving dot per method. Both are
 * this widget; only the layout class differs, so the slider, the readouts, the plots and the
 * honesty rules below cannot diverge between them.
 *
 * Honesty: a method's cell at a budget is met (the shortest stretch of its build order
 * displacing at least the budget) or short (it stopped below the budget: its whole network, and
 * why it stopped). A short cell says so beside the plate, and so does a met budget whose network
 * an earlier budget already had -- the plate never silently shows another budget's network as
 * this one's. */
const LABEL = "BuildOut";

/** `budget` is the slider's POSITION -- an index into the build-out's `budgets`, the block with
 * no new road at 0 -- not a share: an index cannot name a budget the bundle has no cell for. */
export interface BuildOutState {
  budget: number;
}

/** A page can carry two build-outs (the Demo's block, and the methods compared on another), so
 * each mount's slider owns its own URL key: one codec per registration, the same widget. */
export function buildOutCodec(key: string): UrlCodec<BuildOutState> {
  return { budget: intParam(key) };
}

// Not `budget`: region-grow's slider has it, and keys are unique across the shipped widgets.
export const BUILD_OUT_URL: UrlCodec<BuildOutState> = buildOutCodec("buildout");
export const BUILD_OUT_COMPARE_URL: UrlCodec<BuildOutState> = buildOutCodec("compare");

type Cell = DemoMet | DemoShort;

const pct = (v: number, digits: number): string => `${(v * 100).toFixed(digits)}%`;
const km = (v: number): string => `${v.toFixed(2)} km`;
const m2 = (v: number): string =>
  (v >= 1000 ? `${Math.round(v / 1000)}k m²` : `${Math.round(v)} m²`);
const capital = (w: string): string => w.charAt(0).toUpperCase() + w.slice(1);
/** gen_site_pages.py's `_SPANS`, longest first. */
const SPANS = [["centuries", 100 * 3.156e7], ["years", 3.156e7], ["months", 2.63e6],
               ["days", 86400], ["hours", 3600], ["minutes", 60]] as const;

/** A run time as (value, unit) in the largest unit it is at least one and a half of:
 * gen_site_pages.py's `_span`. */
function span(s: number): readonly [number, string] {
  return SPANS.reduce<readonly [number, string] | null>(
    (got, [u, len]) => got ?? (s >= 1.5 * len ? [s / len, u] : null), null) ?? [s, "seconds"];
}

/** A measured whole run's time, to three figures (1.89 hours): gen_site_pages.py's
 * `_measured_time`, so the column's heading and the page's prose state one run alike
 * (web/test/fixtures/measured-times.json holds the cases both are tested against). */
export function measuredTime(s: number): string {
  const [v, unit] = span(s);
  return `${Number(v.toPrecision(3))} ${unit}`;
}

/** Any other run time, to two figures below ten (42 minutes, 1.6 hours): gen_site_pages.py's
 * `_duration`, so the widget and the page agree. */
export function duration(s: number): string {
  const [v, unit] = span(s);
  return `${v >= 10 ? v.toFixed(0) : v.toFixed(1)} ${unit}`;
}

/** The budget at which the network in cell `k` first appears, where an earlier budget's cell
 * already showed it: a run whose step overshoots several budgets builds one network for all of
 * them, and the readout says so rather than implying a new network at every step. `null` where
 * cell `k` is its network's first. */
export function sameAsBudget(budgets: readonly number[], cells: readonly Cell[],
                             k: number): number | null {
  const row = cells[k]!.step.row;
  const first = cells.findIndex((c) => c.step.row === row);
  return first < k ? budgets[first]! : null;
}

/** What the plate says about a series at budget position `k`, or "" where the cell is plainly
 * this budget's own network. */
export function cellNote(budgets: readonly number[], s: DemoSeries, k: number): string {
  const c = s.cells[k]!;
  const b = pct(budgets[k]!, 0);
  switch (c.kind) {
    case "short":
      return `${s.method} stops at ${pct(c.step.displacement, 0)}: ${c.why}. Its whole network `
        + `is shown, short of the ${b} budget.`;
    case "met": {
      const was = sameAsBudget(budgets, s.cells, k);
      return was === null ? ""
        : `The same network as at the ${pct(was, 0)} budget: its roads already displace `
          + `${pct(c.step.displacement, 1)}, past this budget too.`;
    }
  }
}

/** One of the three plots: permeability (y, shared) against one cost of a step (x). */
interface PlotSpec {
  readonly title: string;
  readonly xLabel: string;
  readonly x: (f: DemoStep) => number;
  readonly formatX: (t: number) => string;
  /** What the plot's heading says about a step after its permeability. */
  readonly at: (f: DemoStep) => string;
}

/** `metric` is the page's name for L_P (`data-metric-label`, lower case), so a rename is one
 * change in the generator rather than a hunt through this file's strings. */
const plotsFor = (metric: string): readonly PlotSpec[] => [
  { title: `${capital(metric)} against displacement`, xLabel: "displaced, share of floor area",
    x: (f) => f.displacement, formatX: (t) => pct(t, 0),
    at: (f) => `${pct(f.displacement, 1)} displaced` },
  { title: "… against new road", xLabel: "new road, km",
    x: (f) => f.road_km, formatX: (t) => String(Number(t.toFixed(1))),
    at: (f) => km(f.road_km) },
  { title: "… against conductance added", xLabel: "conductance added, m²",
    x: (f) => f.conductance_m2,
    formatX: (t) => (t === 0 ? "0" : `${Number((t / 1000).toFixed(1))}k`),
    at: (f) => `${m2(f.conductance_m2)} added` },
];

/** A chart axis from 0 up to the first `niceTicks` step at or past `max`: `niceTicks` alone stops
 * at the last step INSIDE [0, max], which would leave the curve's top above the frame. The ticks
 * are that step's multiples, not a second `niceTicks` call over [0, top]: `top` is a float sum
 * (0.6000000000000001 for three steps of 0.2), and re-picking a step for it can pick a different
 * one whose last tick stops short of `top` again. */
function axisTo(max: number, target: number): { top: number; ticks: number[] } {
  const first = niceTicks(0, max, target);
  const step = first.length > 1 ? first[1]! - first[0]! : max;
  const n = Math.ceil(max / step - 1e-9);
  const ticks = Array.from({ length: n + 1 }, (_, i) => Number((i * step).toFixed(10)));
  return { top: ticks[n]!, ticks };
}

/** The plot rect inside the svg, in CSS pixels: room on the left for the rotated title and the
 * percentage labels, below for the tick labels and the title. */
const GUTTER = { left: 50, right: 16, top: 8, bottom: 36 } as const;
/** The figure width (px) from which several methods' plots sit three across, and the gap between
 * them: a chart narrower than about 180 px cannot hold its tick labels. */
const ACROSS_PX = 600;
const PLOT_GAP_PX = 24;
function plotHeight(width: number): number {
  return Math.min(200, Math.max(140, Math.round(width * 0.44)));
}
function plotView(width: number, height: number, xTop: number, yTop: number): View {
  const scaleX = (width - GUTTER.left - GUTTER.right) / xTop;
  const scaleY = (height - GUTTER.top - GUTTER.bottom) / yTop;
  return { scaleX, scaleY, tx: GUTTER.left, ty: height - GUTTER.bottom };
}

/** The bundle crosses a network boundary, and its declaration (demo.d.ts) cannot promise what
 * the sprite sheets hold: every series has a cell per budget, every step's row is one of its
 * sheet's rows, and the opening position is one of the budgets. Checked once, here, so nothing
 * below indexes past any of them. */
function checkBundle(b: DemoBuildOut): void {
  if (b.series.length === 0) throw new Error(`${LABEL}: the build-out has no series`);
  for (const s of b.series) {
    if (s.cells.length !== b.budgets.length) {
      throw new Error(`${LABEL}: ${s.method} has ${s.cells.length} cells for `
        + `${b.budgets.length} budgets`);
    }
    for (const { step } of s.cells) {
      if (!(Number.isInteger(step.row) && step.row >= 0 && step.row < s.sheet.frames)) {
        throw new Error(`${LABEL}: ${s.method}'s frame ${step.row} is outside its sheet's `
          + `${s.sheet.frames} frames`);
      }
    }
  }
  const d = b.default_step;
  if (!(Number.isInteger(d) && d >= 0 && d < b.budgets.length)) {
    throw new Error(`${LABEL}: default_step ${d} is not one of the ${b.budgets.length} budgets`);
  }
}

/** Where frame `i` sits on its sheet (read across, then down), as the translate that brings it
 * into the plate: percentages of the image's own size. */
export function sheetOffset(sheet: DemoSheet, i: number): { x: number; y: number } {
  const down = Math.ceil(sheet.frames / sheet.columns);
  return { x: (-100 * (i % sheet.columns)) / sheet.columns,
           y: (-100 * Math.floor(i / sheet.columns)) / down };
}

/** The sheets sit beside the build-out file (gen_site_pages.py copies them into one directory),
 * so each one's URL is the file's with its name swapped in. */
function besideBundle(src: string, name: string): string {
  return src.slice(0, src.lastIndexOf("/") + 1) + name;
}

export const buildOut: Widget<BuildOutState> = (host, makeState) => {
  const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<DemoBuildOut>;
    })
    .then((b) => boot(host, makeState, b, src))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

function div(cls: string): HTMLDivElement {
  const el = document.createElement("div");
  el.className = cls;
  return el;
}

function para(cls: string, text = ""): HTMLParagraphElement {
  const el = document.createElement("p");
  el.className = cls;
  el.textContent = text;
  return el;
}

/** A legend's colour bar with its tick labels under it. */
function ramp(stops: readonly string[], ticks: readonly string[]): HTMLDivElement {
  const box = div("sbu-buildout__ramp");
  const bar = div("sbu-buildout__ramp-bar");
  bar.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
  const labels = div("sbu-buildout__ramp-ticks");
  for (const t of ticks) {
    const span = document.createElement("span");
    span.textContent = t;
    labels.append(span);
  }
  box.append(bar, labels);
  return box;
}

/** One swatch and its words; `paint` styles the swatch (a fill, a gradient, a line). */
function swatch(text: string, paint: (sw: HTMLElement) => void): HTMLSpanElement {
  const item = document.createElement("span");
  item.className = "sbu-buildout__key";
  const sw = document.createElement("i");
  sw.className = "sbu-buildout__swatch";
  paint(sw);
  item.append(sw, text);
  return item;
}

/** The plates' legend: escape time's ramp (to 100% of the block's scale and past it, hence the
 * "+"), the road, and a destroyed building's ghost -- distinct encodings, since a road and a
 * destroyed home are different losses -- and a damaged home's outline. */
function legend(e: DemoEncoding): HTMLDivElement {
  const box = div("sbu-buildout__legend");
  box.append(
    ramp(e.escape_ramp,
         [0, 0.25, 0.5, 0.75, 1].map((t) => (t === 1 ? `${pct(t, 0)}+` : pct(t, 0)))),
    swatch("New road", (sw) => { sw.style.background = e.road; }),
    swatch("Destroyed building", (sw) => {
      sw.style.background = e.ghost_fill;
      sw.style.borderColor = e.ghost_edge;
    }),
    swatch("Damaged, still a home", (sw) => {
      sw.style.background = e.escape_ramp[1]!;
      sw.style.borderColor = e.ghost_edge;
    }));
  return box;
}

/** The block's edge: where people walk out, and the two kinds of edge they cannot cross on foot
 * except at its ends or not at all. */
function edgeKey(e: DemoEncoding): HTMLDivElement {
  const key = div("sbu-buildout__legend sbu-buildout__edge");
  key.append(
    // On white, as on the plate: the street ink is near-black, invisible on the dark scheme's page.
    swatch("Street", (sw) => {
      sw.style.background = `linear-gradient(${e.street}, ${e.street}) center / 100% 3px `
        + "no-repeat, #ffffff";
    }),
    swatch("Railway", (sw) => {
      sw.style.background = `linear-gradient(${e.street}, ${e.street}) center / 100% 1px `
        + `no-repeat, repeating-linear-gradient(90deg, ${e.street} 0 1.5px, #ffffff 1.5px 6px)`;
    }),
    swatch("Ditch", (sw) => {
      sw.style.background = e.water;
      sw.style.borderColor = e.edge;
    }));
  return key;
}

/** One method's column: (with several) its heading and whole run time, then its plate, its
 * readouts and the note on its cell. */
interface Column {
  readonly series: DemoSeries;
  readonly img: HTMLImageElement;
  readonly readouts: HTMLElement[];
  readonly note: HTMLParagraphElement;
}

function column(s: DemoSeries, src: string, metric: string, headed: boolean,
                parent: HTMLElement): Column {
  const col = div("sbu-buildout__col");
  if (headed) {
    const how = s.run.kind === "recorded" ? "recorded" : "measured";
    // The heading's time is the whole run's; each step's "Built in" is the same run's time to it.
    col.append(para("sbu-buildout__head", s.label),
               // A measured run as the page's prose states it (1.89 hours), a recorded one as
               // the page states ours.
               para("sbu-buildout__time", s.run.kind === "measured"
                 ? measuredTime(s.run.seconds) : duration(s.run.seconds)),
               para("sbu-buildout__sub", `whole run, ${how}, ${s.run.machine}`));
  }
  const plate = div("sbu-buildout__plate");
  plate.style.aspectRatio = `${s.sheet.width} / ${s.sheet.frame_height}`;
  const img = document.createElement("img");
  img.alt = "";
  // The sheet is `columns` frames wide: the plate shows one, so the image is that many plates.
  img.style.width = `${100 * s.sheet.columns}%`;
  img.setAttribute("src", besideBundle(src, s.sheet.file));
  plate.append(img);
  const read = document.createElement("dl");
  read.className = "sbu-buildout__read";
  // Announced: the one place a screen-reader user hears what a slider step built.
  read.setAttribute("aria-live", "polite");
  const readouts = ["Displaced", "New road", capital(metric), "Built in"].map((term) => {
    const item = div("");
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    item.append(dt, dd);
    read.append(item);
    return dd;
  });
  const note = para("sbu-buildout__same");
  note.setAttribute("aria-live", "polite");
  col.append(plate, read, note);
  parent.append(col);
  return { series: s, img, readouts, note };
}

function boot(host: HTMLElement, makeState: StateFactory<BuildOutState>, b: DemoBuildOut,
              src: string): void {
  checkBundle(b);
  const metric = requireAttr(host.dataset.metricLabel, "data-metric-label", LABEL);
  const PLOTS = plotsFor(metric);
  const multi = b.series.length > 1;
  const last = b.budgets.length - 1;
  const state: StateSource<BuildOutState> = makeState({ budget: b.default_step });
  // The URL key decodes any non-negative integer; how many budgets there are is the bundle's to
  // say, so a position past the last is pinned to it here and written back, which rewrites the
  // URL to the frame actually shown (screen-map.ts's `?floor=` reasoning).
  if (state.get().budget > last) state.set({ budget: last });

  const caption = host.querySelector("figcaption");
  const wrap = div(multi ? "sbu-buildout sbu-buildout--multi" : "sbu-buildout");

  // ---- the bar: the budget slider, and (on a phone, by CSS) a compact readout
  const bar = div("sbu-buildout__bar");
  const sliderLabel = document.createElement("label");
  sliderLabel.className = "sbu-buildout__budget";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(last);
  slider.step = "1";
  const out = document.createElement("output");
  sliderLabel.append("Budget ", slider, out);
  slider.addEventListener("input", () => state.set({ budget: Number(slider.value) }));
  // The compact readout repeats the plates' figures for a phone, where a plate fills the screen
  // and its readouts scroll away under it; hidden from assistive tech, which has those.
  const mini = para("sbu-buildout__mini");
  mini.setAttribute("aria-hidden", "true");
  bar.append(sliderLabel, mini);

  // ---- the plates: a column per method
  const stage = div("sbu-buildout__stage");
  const cols = div("sbu-buildout__cols");
  const columns = b.series.map((s) => column(s, src, metric, multi, cols));
  const left = div("sbu-buildout__left");
  left.append(cols, legend(b.encoding), edgeKey(b.encoding));

  // ---- the plots: permeability against each cost, each method's curve and its dot
  const plots = div("sbu-buildout__plots");
  if (multi) {
    const key = div("sbu-buildout__series");
    b.series.forEach((s, i) => {
      const item = document.createElement("span");
      item.className = `sbu-buildout__serie sbu-series-${i}`;
      item.textContent = s.method;
      key.append(item);
    });
    plots.append(key);
  }
  const plotParts = PLOTS.map((spec) => {
    const box = div("sbu-buildout__plot");
    const title = para("sbu-buildout__plot-title");
    const name = document.createElement("span");
    name.textContent = spec.title;
    const now = document.createElement("span");
    title.append(name, now);
    const chart = div("sbu-buildout__chart");
    box.append(title, chart);
    plots.append(box);
    return { spec, now, chart };
  });
  stage.append(left, plots);
  wrap.append(bar, stage);
  if (caption) host.insertBefore(wrap, caption); else host.append(wrap);

  // Each method's distinct networks: a dot for each, not one per budget, on axes spanning all.
  const distinct = b.series.map((s) => {
    const seen = new Set<number>();
    const steps: DemoStep[] = [];
    for (const { step } of s.cells) {
      if (!seen.has(step.row)) { seen.add(step.row); steps.push(step); }
    }
    return steps;
  });
  const all = distinct.flat();
  const yAxis = axisTo(Math.max(...all.map((f) => f.permeability)), 3);
  let cursors: Cursor[][] = [];
  let svgs: SVGSVGElement[] = [];

  /** Every chart, laid out afresh for `width`: axes, each method's curve and points, and a cursor
   * per method. */
  const layout = (width: number): void => {
    for (const s of svgs) s.remove();
    const height = plotHeight(width);
    svgs = [];
    cursors = plotParts.map(({ spec, chart }) => {
      const xAxis = axisTo(Math.max(...all.map(spec.x)), 4);
      const view = plotView(width, height, xAxis.top, yAxis.top);
      const svg = createSvg(chart, width, height);
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label",
                       `${spec.title.replace("…", capital(metric))}, every network`);
      // The y axis's 0 goes unlabelled: the x axis's 0 sits under it at the same corner.
      drawAxes(svg, view, xAxis.ticks, yAxis.ticks, spec.xLabel, metric, spec.formatX,
               (t) => (t === 0 ? "" : pct(t, 0)), 0.15);
      distinct.forEach((steps, i) => {
        const xs = steps.map(spec.x);
        const ys = steps.map((f) => f.permeability);
        const cls = `sbu-curve sbu-series-${i}`;
        drawPolyline(svg, view, xs, ys, "currentColor", 1.5).setAttribute("class", cls);
        for (const m of drawMarkers(svg, view, xs, ys, "currentColor", 2)) {
          m.setAttribute("class", cls);
        }
      });
      const cs = distinct.map((_, i) => {
        const cursor = drawCursor(svg, view, "currentColor", 6);
        cursor.dot.setAttribute("class", `sbu-cursor sbu-series-${i}`);
        return cursor;
      });
      svgs.push(svg);
      return cs;
    });
  };

  const render = (): void => {
    const k = state.get().budget;
    const budget = b.budgets[k]!;
    slider.value = String(k);
    out.textContent = pct(budget, 0);
    const steps = columns.map((c) => c.series.cells[k]!.step);
    slider.setAttribute("aria-valuetext", `${pct(budget, 0)} budget, ` + columns.map((c, i) => {
      const f = steps[i]!;
      return `${c.series.method} ${pct(f.displacement, 1)} displaced, `
        + `${metric} ${pct(f.permeability, 1)}`;
    }).join("; "));
    columns.forEach((c, i) => {
      const f = steps[i]!;
      const { x, y } = sheetOffset(c.series.sheet, f.row);
      c.img.style.transform = `translate(${x}%, ${y}%)`;
      c.img.alt = `${c.series.method}: the block's homes coloured by escape time, `
        + (k === 0 ? "before any new road." : `at ${pct(f.displacement, 1)} displaced.`);
      const [dDisp, dRoad, dPerm, dTime] = c.readouts;
      dDisp!.textContent = pct(f.displacement, 1);
      dRoad!.textContent = km(f.road_km);
      dPerm!.textContent = pct(f.permeability, 1);
      // Since the run began, never an average: a run's early roads can take most of it.
      dTime!.textContent = duration(f.seconds);
      c.note.textContent = cellNote(b.budgets, c.series, k);
    });
    // Each item's words held together (no-break spaces), so a narrow bar wraps between items
    // rather than leaving "km" alone on a line.
    const keep = (t: string): string => t.replaceAll(" ", "\u00a0");
    mini.textContent = columns.map((c, i) => {
      const f = steps[i]!;
      const who = multi ? `${c.series.method}: ` : "";
      return [`${who}${capital(metric)} ${pct(f.permeability, 1)}`,
              `Displaced ${pct(f.displacement, 1)}`, `Road ${km(f.road_km)}`].map(keep)
        .join(" · ");
    }).join(" | ");
    plotParts.forEach(({ spec, now }, j) => {
      now.textContent = columns.map((c, i) => {
        const f = steps[i]!;
        return `${multi ? `${c.series.method} ` : ""}${pct(f.permeability, 1)} at ${spec.at(f)}`;
      }).join(" · ");
      steps.forEach((f, i) => cursors[j]![i]!.move(spec.x(f), f.permeability));
    });
  };

  let drawnWidth = -1;
  observeSize(plots, ({ width }) => runOrReport(host, LABEL, () => {
    if (width === drawnWidth) return;
    const first = drawnWidth < 0;
    // Several methods, on a wide enough figure: the three plots laid across, sharing the width;
    // otherwise stacked, each taking all of it. Set here, since each chart is sized to it.
    const across = multi && width >= ACROSS_PX ? 3 : 1;
    plots.style.gridTemplateColumns = `repeat(${across}, minmax(0, 1fr))`;
    plots.style.columnGap = `${PLOT_GAP_PX}px`;
    layout(Math.floor((width - PLOT_GAP_PX * (across - 1)) / across));
    render();
    drawnWidth = width;
    if (first) {
      state.subscribe(render);
      removeFallbackImage(host);
    }
  }));
}
