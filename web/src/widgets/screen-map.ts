// Type-only: erased at compile time, so this module has NO runtime import of mount.js. A runtime
// one would recreate the import cycle that once made the whole bundle throw during module
// evaluation while the page still looked fine (see mount.ts's registration comment). This file
// must never import `register`.
import type { Widget } from "../mount.js";
import type { CityBundle } from "../screen_map.js";
import type { StateFactory, StateSource } from "../state.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { ranking, scores, selectAt, type MetricName, type Selection } from "../model/screen.js";
import { createLayer } from "../render/city.js";
import { sizeCanvas } from "../render/canvas.js";
import { fitBbox, type Bbox, type View } from "../view/transform.js";

/** The name every failure of this widget is reported under -- see region-grow.ts's own `LABEL`
 * for why one constant rather than a string repeated at each call site. */
const LABEL = "ScreenMap";

/** `city` picks which already-fetched bundle is active; `metric` and `floor` are indices into
 * THAT bundle's own scoring, not display state -- exactly `RegionGrowState`'s reasoning
 * (region-grow.ts) applied to a second axis (which city) on top of the first (which metric, at
 * what floor). Keeping all three here is what makes a metric-switch-then-drag sequence replay
 * through the same `scores`/`ranking`/`selectAt` calls every render, instead of the picture and
 * the model drifting apart across two different pieces of mutable state. */
interface ScreenState { city: "capetown" | "nairobi"; metric: MetricName; floor: number }

/** All four candidate screens (design §3.1), calibrated or not -- `MetricName`'s own four members,
 * spelled out rather than read off `Object.keys(METRICS)`: an object's keys are plain `string[]`,
 * which would need re-asserting back to `MetricName` anyway, and this list is what the `<select>`
 * below is built from, so a fifth metric added to the model without a line added here is a
 * compile error at THIS array instead of a silently short menu. Calibrated metrics first (the two
 * `bundle.floors` actually ships a floor for), so the shipped default is also the first option. */
const METRIC_NAMES: readonly MetricName[] =
  ["depth_density_proxy", "density_compactness", "density", "depth_proxy"];

/** The bbox every shipped block's rings must fit inside -- exterior AND interior rings, mirroring
 * region-grow.ts's own `hoodBbox` one level up in scale. A `Math.min(...xs)`-over-spread version
 * (fine at RegionGrow's 213 blocks) would risk "too many arguments" at Cape Town's ~200,000+
 * ring vertices, so this accumulates in a loop instead. */
function cityBbox(bundle: CityBundle): Bbox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const rings of bundle.rings) for (const ring of rings) for (const [x, y] of ring) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

/** Exhaustive over `ScreenState["city"]` by construction: a switch with no `default` over a
 * two-member literal union fails to compile if a third city is ever added and this is not, so the
 * reader message can never silently fall back to the wrong name. */
function cityName(city: ScreenState["city"]): string {
  switch (city) {
    case "capetown": return "Cape Town";
    case "nairobi": return "Nairobi";
  }
}

/** The `<select>`'s value is a real DOM string, not a literal this module wrote -- a genuinely
 * open boundary even though every `<option>` coming out of it was populated from `METRIC_NAMES`
 * a few lines up, so it is validated rather than merely cast (the CLI-argument/plugin-key case
 * the checkability rule carves out, not a closed-set field access). No default: an unrecognised
 * value raises rather than silently picking a metric the reader did not choose. */
function asMetric(raw: string): MetricName {
  for (const m of METRIC_NAMES) if (m === raw) return m;
  throw new Error(`${LABEL}: unrecognised metric "${raw}"`);
}

/** The floor slider's live bounds and value for (bundle, metric). `preferred === null` means
 * "reset to this metric's own default": its shipped calibration in `bundle.floors` if it has one
 * (`depth_density_proxy`/`density_compactness` do -- design §3.1), else this metric's own minimum
 * score. `density`/`depth_proxy` ship no calibration at all, and inventing a numeric floor for
 * them would be worse than admitting none exists: every block passes at the minimum, and the
 * reader drags up from there. Otherwise `preferred` itself, clamped into the freshly computed
 * range -- the city toggle takes this path, since the whole point of an ABSOLUTE floor (design
 * §3.4) is that the same number carries over to a corpus with no calibration at all, rather than
 * being redefined on every switch the way a percentile would be. */
function syncFloor(floorSlider: HTMLInputElement, bundle: CityBundle, metric: MetricName,
                   preferred: number | null): number {
  const sc = scores(bundle, metric);
  let min = Infinity, max = -Infinity;
  for (const v of sc) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const shipped = bundle.floors.find((f) => f.metric === metric)?.value ?? null;
  const target = preferred ?? shipped ?? min;
  const floor = Math.min(Math.max(target, min), max);
  floorSlider.min = String(min);
  floorSlider.max = String(max);
  floorSlider.step = String((max - min) / 1000 || 1e-9);
  floorSlider.value = String(floor);
  return floor;
}

/** The aria-live text: pool size always, and -- only where the bundle carries `informal` -- the
 * precision/recall that pool scores against it. `sel.precision`/`sel.recall` are `null` together
 * (model/screen.ts's `selectAt`: both come from the same `informal === undefined` check), so
 * either one being absent is read as "no ground-truth layer for this city" rather than as a
 * partial result to paper over with a placeholder number.
 *
 * `city` is the caller's OWN `ScreenState["city"]`, not `bundle.city` -- the bundle's own field is
 * a plain `string` off a JSON boundary, and the caller already holds the validated union value
 * that came from `state.get()`, so re-deriving it from the bundle would be a second, uncertain
 * path to a fact already known for certain. */
function describeSelection(city: ScreenState["city"], bundle: CityBundle, sel: Selection): string {
  const pool = `${sel.count} of ${bundle.n_blocks} blocks selected.`;
  if (sel.precision === null || sel.recall === null) {
    return `${pool} ${cityName(city)} has no ground-truth informal-settlement layer, `
         + `so precision and recall cannot be shown.`;
  }
  return `${pool} Precision ${(sel.precision * 100).toFixed(1)}%, `
       + `recall ${(sel.recall * 100).toFixed(1)}%, against the City of Cape Town's own `
       + `informal-structure survey.`;
}

function fetchBundle(src: string): Promise<CityBundle> {
  return fetch(src).then((r) => {
    if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
    return r.json() as Promise<CityBundle>;
  });
}

export const screenMap: Widget = (host, makeState) => {
  // Not `host.dataset.bundleCapetown!`: a missing attribute then reaches `fetch(undefined)` and
  // surfaces as "fetch undefined failed: 404", which sends the reader looking for a missing FILE
  // rather than the missing ATTRIBUTE that is actually wrong (region-grow.ts's own reasoning).
  const srcCapetown = requireAttr(host.dataset.bundleCapetown, "data-bundle-capetown", LABEL);
  const srcNairobi = requireAttr(host.dataset.bundleNairobi, "data-bundle-nairobi", LABEL);
  // Both cities fetched eagerly, not the active one only: the city toggle is then an instant
  // client-side swap with no second network round trip, and "7.33 MB / 2.42 MB gz" (this branch's
  // own spec) is already the combined figure for both tiers together.
  void Promise.all([fetchBundle(srcCapetown), fetchBundle(srcNairobi)])
    .then(([capetown, nairobi]) => boot(host, makeState, { capetown, nairobi }))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

interface Bundles { capetown: CityBundle; nairobi: CityBundle }

function boot(host: HTMLElement, makeState: StateFactory, bundles: Bundles): void {
  const caption = host.querySelector("figcaption");

  const cv = document.createElement("canvas");
  // Inline styles, never presentation attributes -- Material's `.md-typeset svg{height:auto;
  // max-width:100%}` beats a presentation attribute (region-grow.ts's own reasoning, D1's Critical).
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";

  const controls = document.createElement("div");

  const metricLabel = document.createElement("label");
  const metricSelect = document.createElement("select");
  for (const m of METRIC_NAMES) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    metricSelect.append(opt);
  }
  metricLabel.append("Metric ", metricSelect);

  const floorLabel = document.createElement("label");
  const floorSlider = document.createElement("input");
  floorSlider.type = "range";
  floorLabel.append("Floor ", floorSlider);

  const cityLabel = document.createElement("label");
  const cityToggle = document.createElement("input");
  cityToggle.type = "checkbox";
  cityLabel.append(cityToggle, " Show Nairobi instead of Cape Town");

  const readout = document.createElement("p");
  // The one line that changes on every frame -- floor drag, metric switch, city toggle --
  // announced. A canvas carries no accessible text at all, so without this a screen-reader user
  // moving the slider hears nothing about which blocks it now selects (region-grow.ts's own
  // reasoning for its budget readout). `polite`, not `assertive`: a drag produces many frames and
  // must not interrupt itself.
  readout.setAttribute("aria-live", "polite");

  controls.append(metricLabel, floorLabel, cityLabel, readout);

  // Canvas and controls go BEFORE any <figcaption>, so the mount point keeps picture-then-caption
  // reading order. The fallback <img> is NOT removed here: it goes after the first successful
  // draw, below, so a canvas that mounts into a zero-width container or a draw that throws leaves
  // the static picture the error text points the reader at.
  if (caption) {
    host.insertBefore(cv, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(cv, controls);
  }

  const initialCity: ScreenState["city"] = "capetown";
  const initialMetric: MetricName = "depth_density_proxy";
  const initialFloor = syncFloor(floorSlider, bundles[initialCity], initialMetric, null);
  const state: StateSource<ScreenState> = makeState<ScreenState>({
    city: initialCity, metric: initialMetric, floor: initialFloor,
  });
  metricSelect.value = state.get().metric;
  cityToggle.checked = state.get().city === "nairobi";

  floorSlider.addEventListener("input", () => state.set({ floor: Number(floorSlider.value) }));
  metricSelect.addEventListener("change", () => {
    const metric = asMetric(metricSelect.value);
    const floor = syncFloor(floorSlider, bundles[state.get().city], metric, null);
    state.set({ metric, floor });
  });
  cityToggle.addEventListener("change", () => {
    const city: ScreenState["city"] = cityToggle.checked ? "nairobi" : "capetown";
    const s = state.get();
    const floor = syncFloor(floorSlider, bundles[city], s.metric, s.floor);
    state.set({ city, floor });
  });

  // Assigned only from inside the first sized callback below, exactly like region-grow.ts's own
  // `size`/`view` -- everything that reads them (`render`) is wired from there too, so there is no
  // ordering in which they are read before that callback has run.
  let size: { width: number; height: number };
  let view: View;
  const ctx = cv.getContext("2d")!;

  // The ranking currently painted, and which (city, metric) it belongs to -- `null` before the
  // first render forces `render` to compute both on its first call regardless of `force`.
  let order: Int32Array;
  let s: Float64Array;
  let rankedFor: { city: ScreenState["city"]; metric: MetricName } | null = null;
  // One instance for this mount, never shared -- see render/city.ts's own `CityLayer` comment for
  // why its paint history must not be module state the way its geometry cache is.
  const layer = createLayer();

  // `force`: true from the resize callback (a new view invalidates every cached screen coordinate,
  // so the base layer must be repainted regardless of what else changed) and from a city switch (a
  // different bundle's blocks need their own bbox/view/base layer); false from every other state
  // change, so a floor or metric change alone never repaints the ~16,451-block base layer -- see
  // render/city.ts's own module comment for why that split is the whole performance point.
  const render = (force: boolean): void => {
    const st = state.get();
    const bundle = bundles[st.city];
    const cityChanged = rankedFor === null || rankedFor.city !== st.city;
    const metricChanged = rankedFor === null || rankedFor.metric !== st.metric;
    if (force || cityChanged) {
      view = fitBbox(cityBbox(bundle), size.width, size.height, bundle.encoding.pad);
      layer.paintBase(ctx, bundle, view, bundle.encoding, size);
    }
    if (cityChanged || metricChanged) {
      s = scores(bundle, st.metric);
      order = ranking(bundle, st.metric);
      rankedFor = { city: st.city, metric: st.metric };
    }
    const sel = selectAt(bundle, order, s, st.floor);
    layer.paintSelection(ctx, bundle, view, bundle.encoding, order, sel.count);
    // Every number the picture shows is also present as text, computed from the same `sel` the
    // picture was drawn from -- there is no second call to `selectAt` that could disagree with it.
    readout.textContent = describeSelection(st.city, bundle, sel);
  };

  // The observed element is the CANVAS ITSELF, exactly region-grow.ts's own reasoning: `width:
  // 100%` makes its content box track the container's width, answering "how many CSS pixels am I
  // drawing into?" in one hop.
  //
  // `runOrReport`: a real ResizeObserver delivers from the browser's own dispatch, OUTSIDE the
  // fetch chain above, so that chain's `.catch` cannot see a throw in here -- it would be an
  // uncaught exception with a blank figure and no message (dom/error.ts).
  let firstDraw = true;
  observeSize(cv, (measured) => runOrReport(host, LABEL, () => {
    size = measured;
    sizeCanvas(cv, size);
    render(true);
    if (firstDraw) {
      firstDraw = false;
      // Controls are wired above, at creation time, not deferred behind this flag: unlike
      // RegionGrow's canvas click (which needs `view` to hit-test), none of the three controls
      // here read anything that is not ready until now -- they only ever call `state.set`, which
      // is harmless before any subscriber exists. Only the SUBSCRIPTION itself is deferred, so a
      // later resize can never register a second one and double-render every state change.
      state.subscribe(() => render(false));
      // Only now: the static picture is the honest one until a real one has replaced it.
      removeFallbackImage(host);
    }
  }));
}
