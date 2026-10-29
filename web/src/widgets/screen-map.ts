// Type-only: erased at compile time, so this module has NO runtime import of mount.js. A runtime
// one would recreate the import cycle that once made the whole bundle throw during module
// evaluation while the page still looked fine (see mount.ts's registration comment). This file
// must never import `register`.
import type { Widget } from "../mount.js";
import type { CityBundle } from "../screen_map.js";
import type { StateFactory, StateSource } from "../state.js";
import { enumParam, nullableNumberParam, type UrlCodec } from "../url/param.js";
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
 * the model drifting apart across two different pieces of mutable state.
 *
 * `floor` is `number | null`, and `null` means **this metric's own default** -- exactly what
 * `syncFloor`'s `preferred` parameter already meant. A resolved number cannot express that: the
 * four metrics score on unrelated scales (`model/screen.ts`'s own four formulas), so a URL
 * supplying only `?metric=` would hand the NEW metric the PREVIOUS metric's number, which lands
 * wherever that number happens to fall in a distribution nobody calibrated it against -- past the
 * top of the new range at one extreme, below all of it (every block in the city selected) at the
 * other. Keeping "unset" spellable is also what keeps `?floor=` out of the URL for a reader who
 * only switched metric (design §1.6). */
export interface ScreenState {
  city: "capetown" | "nairobi";
  metric: MetricName;
  floor: number | null;
}

/** All four candidate screens (design §3.1), calibrated or not -- `MetricName`'s own four members,
 * spelled out rather than read off `Object.keys(METRICS)`: an object's keys are plain `string[]`,
 * which would need re-asserting back to `MetricName` anyway, and this list is what the `<select>`
 * below is built from. Calibrated metrics first (the two `bundle.floors` actually ships a floor
 * for), so the shipped default is also the first option. Completeness against `MetricName` is NOT
 * this array's own doing -- see `_AllMetricsAreListed` just below, which is what actually turns a
 * missing fifth metric into a compile error. */
const METRIC_NAMES = [
  "depth_density_proxy", "density_compactness", "density", "depth_proxy",
] as const satisfies readonly MetricName[];

/** Compile-time exhaustiveness over `MetricName`, because the list above is a CLOSED set whose
 * completeness nothing else checks: a fifth metric that never reaches it ships a short <select>
 * and a URL grammar that silently rejects the new name.
 *
 * `METRIC_NAMES` must NOT carry an explicit `readonly MetricName[]` annotation. That widens the
 * literal type, `(typeof METRIC_NAMES)[number]` becomes `MetricName`, the `Exclude` below is
 * vacuously `never`, and the check can never fail -- a guard that cannot fire, which is worse than
 * no guard because it looks like one. `as const satisfies` keeps the literal members while still
 * requiring each to be a real `MetricName`. */
type AssertNever<T extends never> = T;
type _AllMetricsAreListed = AssertNever<Exclude<MetricName, (typeof METRIC_NAMES)[number]>>;

export const SCREEN_MAP_URL: UrlCodec<ScreenState> = {
  city: enumParam("city", ["capetown", "nairobi"] as const),
  // METRIC_NAMES, not Object.keys(METRICS): the same closed, spelled-out list the <select> is built
  // from, so a fifth metric added to the model without a line added to METRIC_NAMES is a compile
  // error at `_AllMetricsAreListed`'s declaration (above), rather than a silently short menu and a
  // silently narrower URL grammar.
  metric: enumParam("metric", METRIC_NAMES),
  // Nullable, because `null` is a real value of `ScreenState.floor` -- "this metric's own default"
  // -- and `nullableNumberParam` spells it as the key's ABSENCE. A reader who only switched metric
  // therefore publishes no `?floor=` at all, and a `?floor=` this codec refuses falls back to the
  // initial, which is that same `null` rather than some other metric's number.
  floor: nullableNumberParam("floor", 6),
};

/** The metric every OTHER metric's own no-calibration floor default is measured against -- the
 * design's own "shipped default" (§3.1), and, as of both committed bundles, present in
 * `bundle.floors` for both cities (checked directly against `capetown.json`/`nairobi.json`, not
 * assumed). Nothing STRUCTURALLY guarantees a future bake keeps shipping it -- `bundle.floors`
 * arrives over the network like the rest of the bundle -- so `floorAtShippedPoolSize` below checks
 * and throws rather than silently defaulting if it is ever missing. */
const DEFAULT_METRIC: MetricName = "depth_density_proxy";

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

/** `density`/`depth_proxy` ship no calibration of their own (only `depth_density_proxy` and
 * `density_compactness` do -- design §3.1). Defaulting them to "select everything" reported the
 * corpus base rate as if it were a screen's own performance (16,451 of 16,451 selected, precision
 * equal to the informal share of the whole city) and defeated `city.ts`'s O(prefix) design
 * outright. Instead default to the score that selects exactly the SAME pool size as the shipped
 * `DEFAULT_METRIC` floor on this bundle (1,655 for Cape Town, 169 for Nairobi): the score of the
 * block at that rank in THIS metric's own ranking. This is exactly how `examples/screen-bakeoff/
 * screen_comparison.csv` compares screens that ship no floor of their own -- at equal pool
 * fractions, not at an arbitrary threshold, and it is a number this bundle already carries, not
 * one invented here. */
function floorAtShippedPoolSize(bundle: CityBundle, metric: MetricName, sc: Float64Array): number {
  const shipped = bundle.floors.find((f) => f.metric === DEFAULT_METRIC);
  if (shipped === undefined) {
    throw new Error(`${LABEL}: bundle carries no shipped ${DEFAULT_METRIC} floor to size `
      + `${metric}'s own default against`);
  }
  const order = ranking(bundle, metric);
  return sc[order[shipped.n - 1]!]!;
}

/** This metric's own default floor: its shipped calibration where `bundle.floors` carries one,
 * else `floorAtShippedPoolSize`'s own non-invented fallback.
 *
 * One function rather than that `??` chain written twice, because `ScreenState.floor` is `null` for
 * "this metric's own default" and TWO callers now resolve that `null`: `syncFloor` for the SLIDER,
 * `render` for the PICTURE. (`syncFloor` goes on to clamp what it gets into the slider's live
 * bounds; that is its contract for an arbitrary `preferred` value, not part of what "the default"
 * means.)
 *
 * Takes the already-computed `sc` rather than scoring again inside -- `render` has it cached beside
 * `rankedFor`. What that saves is bounded, and worth stating so nobody reads it as free: the
 * calibrated branch returns out of `bundle.floors` without reading `sc` at all, while the
 * uncalibrated one falls through to `floorAtShippedPoolSize`, whose `ranking` call scores every
 * block again internally and then sorts -- so that path pays a full scoring pass AND a sort per
 * resolution whatever is handed in, and `sc` merely supplies the array the chosen rank's score is
 * finally read out of.
 *
 * `metric` and `sc` are two parameters whose agreement nothing declares or checks: pair one
 * metric's name with another metric's scores and the result is a number of the right type and the
 * wrong value, with nothing to raise. Both call sites hold it today -- `syncFloor` scores `metric`
 * on the line above its call, and `render` recomputes `s` whenever `rankedFor` moves -- but Task 5
 * turned that from one caller's local invariant into a shared one. */
function defaultFloorFor(bundle: CityBundle, metric: MetricName, sc: Float64Array): number {
  return bundle.floors.find((f) => f.metric === metric)?.value
      ?? floorAtShippedPoolSize(bundle, metric, sc);
}

/** The floor slider's live bounds and value for (bundle, metric). `preferred === null` means "reset
 * to this metric's own default", which `defaultFloorFor` just above resolves. Otherwise `preferred`
 * itself: the whole point of an ABSOLUTE floor (design §3.4) is that the same number carries over to
 * a corpus with no calibration at all, rather than being redefined on every switch the way a
 * percentile would be. `boot` and the city toggle are the two callers that can take either path:
 * `boot` for a URL-supplied `?floor=` or its absence, the toggle for a floor the reader set or one
 * they never touched -- the latter not an absolute floor to carry, but a request for whatever this
 * metric calibrates to. The metric handler always passes `null`. Either way the result is clamped
 * into the range just computed, written to the slider, and returned. */
function syncFloor(floorSlider: HTMLInputElement, bundle: CityBundle, metric: MetricName,
                   preferred: number | null): number {
  const sc = scores(bundle, metric);
  let min = Infinity, max = -Infinity;
  for (const v of sc) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const target = preferred ?? defaultFloorFor(bundle, metric, sc);
  const floor = Math.min(Math.max(target, min), max);
  floorSlider.min = String(min);
  floorSlider.max = String(max);
  // No zero-width-range fallback: verified against both committed bundles that every one of the
  // four metrics has a strictly positive, finite score range (min < max, min 0.0005627 on
  // Nairobi's density_compactness, the narrowest of the eight city/metric pairs) -- a `|| 1e-9`
  // here would be a silencer for a failure that has not happened and cannot happen from this data.
  floorSlider.step = String((max - min) / 1000);
  floorSlider.value = String(floor);
  return floor;
}

/** The aria-live text: pool size always, and -- only where the bundle carries `informal` -- the
 * precision/recall that pool scores against it. NOT the follow ring: that sentence is invariant
 * within a city, and this region is re-announced on every frame a floor drag produces, so it lives
 * in its own static element instead (`describeFollow`, just below). `sel.precision`/`sel.recall`
 * are `null` together
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

/** The follow ring, in text. A canvas carries no accessible text at all, so a marker that exists
 * only in its pixels is one a screen-reader user is never told about.
 *
 * Deliberately NOT part of the `aria-live` readout beside it. That region is the line that changes
 * every frame, and a floor drag produces many; this sentence is invariant within a city, so folding
 * it in there would make a screen-reader user re-hear an unchanging clause once per drag frame. A
 * plain paragraph is read once, where the reader reaches it, and is silent on every later frame.
 *
 * Written from the ACTIVE bundle on every render rather than once at mount, which is what empties
 * it on a switch to a city that carries no `follow` -- text set once would leave Cape Town's block
 * named over Nairobi's map. Rewriting a NON-live element announces nothing, so per-render writing
 * costs the reader no repeats. The id comes off the bundle, never typed here. */
function describeFollow(bundle: CityBundle): string {
  const follow = bundle.follow;
  return follow === undefined ? ""
    : `Block ${follow.block_id} is ringed on the map: it is the one the rest of the site follows.`;
}

function fetchBundle(src: string): Promise<CityBundle> {
  return fetch(src).then((r) => {
    if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
    return r.json() as Promise<CityBundle>;
  });
}

export const screenMap: Widget<ScreenState> = (host, makeState) => {
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

function boot(host: HTMLElement, makeState: StateFactory<ScreenState>, bundles: Bundles): void {
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

  // Static, no `aria-live`: see `describeFollow`. It sits between the canvas and the controls so a
  // screen reader meets the picture's own description where the picture is, before the controls
  // that change it.
  const followNote = document.createElement("p");

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
    host.insertBefore(followNote, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(cv, followNote, controls);
  }

  const state: StateSource<ScreenState> = makeState({
    city: "capetown", metric: "depth_density_proxy", floor: null,
  });
  // A URL (piece E) may have set any of the three before this line, so all three controls are
  // written from STATE rather than from the initials above -- and the slider's bounds and value are
  // resolved against the state's OWN city and metric, with `state.floor` handed straight to
  // `preferred`, since `null` there already means "this metric's own default" (design §1.6).
  const s0 = state.get();
  metricSelect.value = s0.metric;
  cityToggle.checked = s0.city === "nairobi";
  const resolved = syncFloor(floorSlider, bundles[s0.city], s0.metric, s0.floor);
  // `?floor=` is a bare finite number; where it falls in THIS metric's score range is a property of
  // the fetched bundle, which no codec can know (design §2.3). `syncFloor` has already pinned the
  // slider to the nearest usable value, so writing that value back is what keeps the control and
  // the picture on one number -- and the write it triggers rewrites `?floor=` to the clamped value.
  // Only when the URL actually supplied one: resolving a `null` into state here would publish a
  // `?floor=` to a reader who never set one and -- being then indistinguishable from a number they
  // did set -- carry it across the next CITY switch and into any URL they copied. (A metric switch
  // would still clear it: the handler below writes `floor: null` unconditionally.)
  if (s0.floor !== null && resolved !== s0.floor) state.set({ floor: resolved });

  floorSlider.addEventListener("input", () => state.set({ floor: Number(floorSlider.value) }));
  metricSelect.addEventListener("change", () => {
    const metric = asMetric(metricSelect.value);
    syncFloor(floorSlider, bundles[state.get().city], metric, null);
    // `null`, not the number `syncFloor` just wrote onto the slider: both sides resolve `null`
    // through the same `defaultFloorFor`, and `null` additionally keeps `?floor=` out of the URL
    // for a reader who only switched metric.
    state.set({ metric, floor: null });
  });
  cityToggle.addEventListener("change", () => {
    const city: ScreenState["city"] = cityToggle.checked ? "nairobi" : "capetown";
    const s = state.get();
    // Called for its side effect on the SLIDER whichever way the state below goes: the new bundle
    // scores on its own range, so the control's bounds and displayed value have to track it even
    // when there is nothing to pin.
    const resolved = syncFloor(floorSlider, bundles[city], s.metric, s.floor);
    // Into STATE, though, only a floor the READER chose. `syncFloor`'s own docstring argues that an
    // ABSOLUTE floor carries across corpora rather than being redefined per city, and pinning the
    // number here is what carries it -- saying so in the URL is honest, because the reader chose to
    // carry it. A `null` floor is not an absolute floor: it says "whatever this metric calibrates
    // to", so pinning it would invent a choice nobody made, publish a `?floor=` nobody typed, and
    // leave a there-and-back city switch on the OTHER corpus's number (design §1.6).
    state.set({ city, floor: s.floor === null ? null : resolved });
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
      layer.paintBase(bundle, view, bundle.encoding, size);
    }
    if (cityChanged || metricChanged) {
      s = scores(bundle, st.metric);
      order = ranking(bundle, st.metric);
      rankedFor = { city: st.city, metric: st.metric };
    }
    // `null` is "this metric's own default" (`ScreenState`), resolved HERE rather than carried in
    // state -- which is what draws a `?metric=` that arrived without a `?floor=` at the new
    // metric's own calibration instead of at the previous metric's number.
    const floor = st.floor ?? defaultFloorFor(bundle, st.metric, s);
    const sel = selectAt(bundle, order, s, floor);
    layer.paintFrame(ctx, bundle, view, bundle.encoding, order, sel.count, size);
    // Every number the picture shows is also present as text, computed from the same `sel` the
    // picture was drawn from -- there is no second call to `selectAt` that could disagree with it.
    readout.textContent = describeSelection(st.city, bundle, sel);
    // From the SAME `bundle` the frame above was painted from, so the sentence and the ring can
    // never name different cities.
    followNote.textContent = describeFollow(bundle);
  };

  // Coalesces every state-driven redraw (a floor drag, a metric switch, a city toggle) onto the
  // next animation frame, so a drag firing many "input" events in a row cannot queue more frames
  // than the display can render -- each call before the frame fires just moves WHICH state the
  // eventual `render(false)` reads (`state.get()`, inside `render` itself), it never queues a
  // second one. `frameScheduled` is the guard that makes this coalescing rather than merely async:
  // `requestAnimationFrame` on its own queues every call it is given (see harness.ts's own
  // comment), so without this flag a fast drag would queue one callback per "input" event, not one
  // per frame.
  let frameScheduled = false;
  const scheduleRender = (): void => {
    if (frameScheduled) return;
    frameScheduled = true;
    // `runOrReport`: an animation-frame callback runs from the browser's own dispatch, exactly as
    // far outside the fetch chain above as the ResizeObserver callback below is -- without this, a
    // throw here is an uncaught exception with a blank figure and no message (dom/error.ts).
    requestAnimationFrame(() => runOrReport(host, LABEL, () => {
      frameScheduled = false;
      render(false);
    }));
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
      state.subscribe(() => scheduleRender());
      // Only now: the static picture is the honest one until a real one has replaced it.
      removeFallbackImage(host);
    }
  }));
}
