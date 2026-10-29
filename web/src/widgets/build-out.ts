// Type-only: erased at compile time, so this module has NO runtime import of mount.js (the cycle
// mount.ts's registration comment describes). This file must never import `register`.
import type { DemoBundle, DemoEncoding, DemoFrame } from "../demo.js";
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

/** The name every failure of this widget is reported under (region-grow.ts's `LABEL`). */
const LABEL = "BuildOut";

/** `budget` is the slider's POSITION -- an index into the bundle's `method.frames`, the block with
 * no new road at 0 -- not a share: the frames' budgets are whatever the producer baked, and an
 * index cannot name a budget the bundle has no frame for.
 *
 * One picture per frame, each home's escape time with its isochrones: the owner kept that view
 * alone (2026-10-28), so there is no `view` to carry. */
export interface BuildOutState {
  budget: number;
}

export const BUILD_OUT_URL: UrlCodec<BuildOutState> = {
  budget: intParam("budget"),
};

const pct = (v: number, digits: number): string => `${(v * 100).toFixed(digits)}%`;
const km = (v: number): string => `${v.toFixed(1)} km`;
const m2 = (v: number): string =>
  (v >= 1000 ? `${Math.round(v / 1000)}k m²` : `${Math.round(v)} m²`);
const capital = (w: string): string => w.charAt(0).toUpperCase() + w.slice(1);
const duration = (s: number): string =>
  (s < 60 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

/** One of the three stacked plots: permeability (y, shared) against one cost of the frame (x). */
interface PlotSpec {
  readonly title: string;
  readonly xLabel: string;
  readonly x: (f: DemoFrame) => number;
  readonly formatX: (t: number) => string;
  /** What the plot's heading says about the current frame after its permeability. */
  readonly at: (f: DemoFrame) => string;
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
 * percentage labels, below for the tick labels and the title. Fixed pixels rather than
 * `fitAxes`'s symmetric fraction: three short plots stacked cannot spare a right-hand gutter the
 * size of the left one. */
const GUTTER = { left: 50, right: 16, top: 8, bottom: 36 } as const;
/** A plot's height for its width: the prototype's 380 x 168, bounded so a full-width plot on a
 * phone is not a sliver and one on a wide screen is not a wall. */
function plotHeight(width: number): number {
  return Math.min(200, Math.max(140, Math.round(width * 0.44)));
}
function plotView(width: number, height: number, xTop: number, yTop: number): View {
  const scaleX = (width - GUTTER.left - GUTTER.right) / xTop;
  const scaleY = (height - GUTTER.top - GUTTER.bottom) / yTop;
  return { scaleX, scaleY, tx: GUTTER.left, ty: height - GUTTER.bottom };
}

/** The bundle crosses a network boundary, and its declaration (demo.d.ts) cannot promise what
 * the sprite sheets hold: every frame's row must be one of the sheets' rows, and the frame the
 * slider opens on must exist. Checked once, here, so nothing below indexes past either. */
function checkBundle(b: DemoBundle): void {
  const frames = b.method.frames;
  if (frames.length === 0) throw new Error(`${LABEL}: the bundle carries no frames`);
  for (const f of frames) {
    if (!(Number.isInteger(f.row) && f.row >= 0 && f.row < b.sheets.rows)) {
      throw new Error(`${LABEL}: a frame's sprite row ${f.row} is outside the sheets' `
        + `${b.sheets.rows} rows`);
    }
  }
  const d = b.method.default_frame;
  if (!(Number.isInteger(d) && d >= 0 && d < frames.length)) {
    throw new Error(`${LABEL}: default_frame ${d} is not one of the ${frames.length} frames`);
  }
}

/** The sheets sit beside the bundle (gen_site_pages.py copies all four into one directory), so
 * each one's URL is the bundle's with its file name swapped in. */
function besideBundle(src: string, name: string): string {
  return src.slice(0, src.lastIndexOf("/") + 1) + name;
}

export const buildOut: Widget<BuildOutState> = (host, makeState) => {
  const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<DemoBundle>;
    })
    .then((b) => boot(host, makeState, b, src))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

function div(cls: string): HTMLDivElement {
  const el = document.createElement("div");
  el.className = cls;
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

/** The plate's legend: escape time's ramp, to 100% of its scale and past it (hence the "+"), and
 * the grey that is a new road's corridor or a wholly destroyed building. */
function legend(e: DemoEncoding): HTMLDivElement {
  const box = div("sbu-buildout__legend");
  box.append(
    ramp(e.escape_ramp,
         [0, 0.25, 0.5, 0.75, 1].map((t) => (t === 1 ? `${pct(t, 0)}+` : pct(t, 0)))),
    swatch("New road, or a destroyed building", (sw) => { sw.style.background = e.road; }));
  return box;
}

/** The block's edge: where people walk out, and the two kinds of edge
 * they cannot cross on foot except at its ends or not at all. */
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

function boot(host: HTMLElement, makeState: StateFactory<BuildOutState>, b: DemoBundle,
              src: string): void {
  checkBundle(b);
  const metric = requireAttr(host.dataset.metricLabel, "data-metric-label", LABEL);
  const PLOTS = plotsFor(metric);
  const frames = b.method.frames;
  const last = frames.length - 1;
  const state: StateSource<BuildOutState> = makeState({
    budget: b.method.default_frame,
  });
  // `?budget=` decodes any non-negative integer; how many frames there are is the bundle's to
  // say, so a position past the last is pinned to it here and written back, which rewrites the
  // URL to the frame actually shown (screen-map.ts's `?floor=` reasoning).
  if (state.get().budget > last) state.set({ budget: last });

  const caption = host.querySelector("figcaption");
  const wrap = div("sbu-buildout");

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
  // The compact readout repeats the dl below for a phone, where the plate fills the screen and the
  // dl scrolls away under it; hidden from assistive tech, which already has the dl.
  const mini = document.createElement("p");
  mini.className = "sbu-buildout__mini";
  mini.setAttribute("aria-hidden", "true");
  bar.append(sliderLabel, mini);

  // ---- the plate: the sprite sheet, the frame chosen by translating it
  const stage = div("sbu-buildout__stage");
  const left = div("sbu-buildout__left");
  const plate = div("sbu-buildout__plate");
  plate.style.aspectRatio = `${b.sheets.width} / ${b.sheets.frame_height}`;
  const img = document.createElement("img");
  img.alt = "";
  plate.append(img);
  img.setAttribute("src", besideBundle(src, b.sheets.sheet));

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
  left.append(plate, read, legend(b.encoding), edgeKey(b.encoding));

  // ---- the plots: permeability against each cost, the method's whole curve and this frame's dot
  const plots = div("sbu-buildout__plots");
  const plotParts = PLOTS.map((spec) => {
    const box = div("sbu-buildout__plot");
    const title = document.createElement("p");
    title.className = "sbu-buildout__plot-title";
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

  const yAxis = axisTo(Math.max(...frames.map((f) => f.permeability)), 3);
  let cursors: Cursor[] = [];
  let svgs: SVGSVGElement[] = [];

  /** Every chart, laid out afresh for `width`: axes, the curve and its points, and a cursor. */
  const layout = (width: number): void => {
    for (const s of svgs) s.remove();
    const height = plotHeight(width);
    svgs = [];
    cursors = plotParts.map(({ spec, chart }) => {
      const xs = frames.map(spec.x);
      const ys = frames.map((f) => f.permeability);
      const xAxis = axisTo(Math.max(...xs), 4);
      const view = plotView(width, height, xAxis.top, yAxis.top);
      const svg = createSvg(chart, width, height);
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", `${spec.title.replace("…", capital(metric))}, every frame`);
      // The y axis's 0 goes unlabelled: the x axis's 0 sits under it at the same corner.
      drawAxes(svg, view, xAxis.ticks, yAxis.ticks, spec.xLabel, metric, spec.formatX,
               (t) => (t === 0 ? "" : pct(t, 0)), 0.15);
      drawPolyline(svg, view, xs, ys, "currentColor", 1.5).setAttribute("class", "sbu-curve");
      for (const m of drawMarkers(svg, view, xs, ys, "currentColor", 2)) {
        m.setAttribute("class", "sbu-curve");
      }
      const cursor = drawCursor(svg, view, "currentColor", 6);
      cursor.dot.setAttribute("class", "sbu-cursor");
      svgs.push(svg);
      return cursor;
    });
  };

  const render = (): void => {
    const s = state.get();
    const f = frames[s.budget]!;
    slider.value = String(s.budget);
    out.textContent = pct(f.budget, 0);
    slider.setAttribute("aria-valuetext", `${pct(f.budget, 0)} budget, `
      + `${pct(f.displacement, 1)} displaced, ${metric} ${pct(f.permeability, 1)}`);
    img.style.transform = `translateY(${(-100 * f.row) / b.sheets.rows}%)`;
    img.alt = `Block ${b.block_id}, each home's escape time with isochrones, `
      + (s.budget === 0 ? "before any new road." : `at ${pct(f.displacement, 1)} displaced.`);
    const [dDisp, dRoad, dPerm, dTime] = readouts;
    dDisp!.textContent = pct(f.displacement, 1);
    dRoad!.textContent = km(f.road_km);
    dPerm!.textContent = pct(f.permeability, 1);
    dTime!.textContent = duration(f.seconds);
    mini.textContent = `${capital(metric)} ${pct(f.permeability, 1)} · Displaced `
      + `${pct(f.displacement, 1)} · Road ${km(f.road_km)}`;
    plotParts.forEach(({ spec, now }, i) => {
      now.textContent = `${pct(f.permeability, 1)} at ${spec.at(f)}`;
      cursors[i]?.move(spec.x(f), f.permeability);
    });
  };

  let drawnWidth = -1;
  observeSize(plots, ({ width }) => runOrReport(host, LABEL, () => {
    if (width === drawnWidth) return;
    const first = drawnWidth < 0;
    layout(width);
    render();
    drawnWidth = width;
    if (first) {
      state.subscribe(render);
      removeFallbackImage(host);
    }
  }));
}
