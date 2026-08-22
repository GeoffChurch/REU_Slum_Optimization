import type { Bundle } from "../bundle.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { draw, sizeCanvas } from "../render/canvas.js";
// Type-only: erased at compile time, so this produces no runtime import of mount.js. A runtime
// import here would recreate the mount<->widget cycle that made the bundle throw on load (see
// mount.ts's registration comment) -- this module must never import `register` from mount.js.
import type { Widget } from "../mount.js";
import { fitBbox, nearest, panned, toWorld, zoomed, type View } from "../view/transform.js";
import type { StateFactory, StateSource } from "../state.js";
import { boolParam, enumParam, intParam, type UrlCodec } from "../url/param.js";

export interface PermGraphState {
  prefix: number;
  layer: "conductance" | "current";
  halos: boolean;
}

export const PERM_GRAPH_URL: UrlCodec<PermGraphState> = {
  prefix: intParam("prefix"),
  layer: enumParam("layer", ["conductance", "current"] as const),
  halos: boolParam("halos"),
};

/** The name every failure of this widget is reported under -- one constant, because it is used from
 * two unrelated places (the fetch chain and the resize callback) and two spellings of it would be a
 * reader seeing two different widgets fail. */
const LABEL = "PermGraph";

/** Parse `data-prefix` into a non-negative integer, never NaN.
 *
 * The attribute is a string from HTML -- absent, empty, or hand-typed-wrong are all reachable --
 * and `prefix` goes on to index straight into baked arrays (`b.roads.slice(0, prefix)`,
 * `b.prefix.current[prefix]`, ...). `Number(raw)` alone lets a malformed attribute (missing,
 * "abc", "-1", "2.5") become NaN or a value that is numeric but not a valid array index; either
 * would surface downstream as a silent `undefined` read or a `.slice` with a nonsensical bound
 * instead of failing here, at the boundary, where the bad string is still in hand. Falling back to
 * 0 (rather than throwing) matches `layer`/`halos` below, which also coerce a malformed attribute
 * to a valid default instead of rejecting the whole mount over one cosmetic attribute.
 */
function parsePrefix(raw: string | undefined): number {
  const n = raw === undefined ? 0 : Number(raw);
  return Number.isInteger(n) && n >= 0 ? n : 0;
}

function initialState(el: HTMLElement): PermGraphState {
  const layer = el.dataset.layer === "conductance" ? "conductance" : "current";
  return { prefix: parsePrefix(el.dataset.prefix), layer, halos: el.dataset.halos !== "false" };
}

export const permGraph: Widget<PermGraphState> = (host, makeState) => {
  // Not `host.dataset.bundle!`: a missing attribute then reaches `fetch(undefined)` and surfaces as
  // "fetch undefined failed: 404", which sends the reader (and whoever wrote the page) looking for a
  // missing FILE rather than the missing ATTRIBUTE that is actually wrong. This was the third call
  // site of a helper extracted for the other two.
  const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
  // I9: a 404, a renamed bundle field, or any throw inside boot() must be VISIBLE on the page, not
  // an unhandled rejection sitting silently in the console while the PNG fallback stays up and the
  // page looks fine -- verbatim the defect class this branch already found once (mount.ts's
  // registration-cycle comment).
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<Bundle>;
    })
    .then((b) => boot(host, makeState, b))
    // Shared with mount.ts and the Frontier widget (final review, M7): three copies of this had
    // already diverged, and only one of them mentioned the static image the reader is still looking
    // at. PermGraph's behaviour is otherwise unchanged -- same trigger, same destination, one
    // sentence added.
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

function boot(host: HTMLElement, makeState: StateFactory<PermGraphState>, b: Bundle): void {
  const state: StateSource<PermGraphState> = makeState(initialState(host));
  // `?prefix=` is a bare non-negative integer; how many prefixes THIS block has is a property of the
  // fetched bundle, which no codec can know (design §2.3). Clamp rather than throw -- an
  // out-of-range number must land on the last prefix, not on an error card -- and the write the
  // clamp triggers replaces the out-of-range key with the clamped one, so the URL self-corrects in
  // front of the reader (and drops `?prefix=` entirely when the clamp lands on this mount point's
  // own `data-prefix`, which is the value the store diffs against).
  const maxPrefix = b.n_prefixes - 1;
  if (state.get().prefix > maxPrefix) state.set({ prefix: maxPrefix });
  const caption = host.querySelector("figcaption");

  const cv = document.createElement("canvas");
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";
  // The canvas PANS on a pointer drag, so without this a touch drag scrolls the page and the
  // browser CANCELS the pointer stream mid-drag -- the view lurches and then sticks. It is the one
  // line that makes Pointer Events actually serve touch rather than only appear to, and it was
  // missing here while both other widgets carried it.
  cv.style.touchAction = "none";

  const controls = document.createElement("div");

  const s0 = state.get();

  const sliderLabel = document.createElement("label");
  const slider = document.createElement("input");
  slider.type = "range";                    // native: keyboard- and screen-reader-reachable
  slider.min = "0";
  slider.max = String(b.n_prefixes - 1);
  slider.value = String(s0.prefix);
  slider.addEventListener("input", () => state.set({ prefix: Number(slider.value) }));
  sliderLabel.append("Road prefix ", slider);

  // Layer switch (spec §6 / fix wave I8): conductance-width vs current-width, mirroring
  // render_graph's `layer` argument. Native <select>, keyboard- and screen-reader-reachable.
  const layerLabel = document.createElement("label");
  const layerSelect = document.createElement("select");
  for (const [value, text] of [["current", "Current"], ["conductance", "Conductance"]] as const) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = text;
    layerSelect.append(opt);
  }
  layerSelect.value = s0.layer;
  layerSelect.addEventListener("change", () => {
    state.set({ layer: layerSelect.value as PermGraphState["layer"] });
  });
  layerLabel.append("Layer ", layerSelect);

  // Ground-halo toggle (spec §6 / fix wave I8). Native checkbox, same keyboard/screen-reader
  // reachability as the other two controls.
  const haloLabel = document.createElement("label");
  const haloCheckbox = document.createElement("input");
  haloCheckbox.type = "checkbox";
  haloCheckbox.checked = s0.halos;
  haloCheckbox.addEventListener("change", () => state.set({ halos: haloCheckbox.checked }));
  haloLabel.append(haloCheckbox, " Ground halos");

  const readout = document.createElement("p");
  controls.append(sliderLabel, layerLabel, haloLabel, readout);

  // Insert the canvas and controls BEFORE the figcaption -- so the mount point (now the <figure>
  // itself, fix wave I4) keeps picture-then-caption reading order, the same order every sibling
  // figure in .sbu-figure-grid uses. Appending after </figure> (the prior behaviour) put this one
  // cell's caption ahead of its picture, out of step with its neighbours.
  //
  // The fallback <img> is NOT removed here. It used to go on this line, before anything had been
  // drawn -- so a canvas that mounted into a zero-width container, or a draw that threw, left a
  // blank figure with the static picture already gone. It now goes where Frontier has always put
  // it: after the first successful draw, below.
  if (caption) {
    host.insertBefore(cv, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(cv, controls);
  }

  const xs = b.nodes.cx, ys = b.nodes.cy;
  const bbox = { minX: Math.min(...xs), minY: Math.min(...ys),
                 maxX: Math.max(...xs), maxY: Math.max(...ys) };
  // Both assigned by the observer below and by nothing else. Everything that reads them -- `render`,
  // the pan/zoom handlers -- is wired from inside the first sized callback, so there is no ordering
  // in which they are read before that callback has run.
  let size: { width: number; height: number };
  let view: View;
  const ctx = cv.getContext("2d")!;

  const render = (): void => {
    const s = state.get();
    draw(ctx, b, { view, ...s }, size);
    // Every number the picture shows is also present as text.
    readout.textContent =
      `${b.prefix.road_m[s.prefix]!.toFixed(0)} m of road · ` +
      `${(b.prefix.permeability[s.prefix]! * 100).toFixed(1)}% permeability`;
  };

  const wireInteraction = (): void => {
    let dragging: [number, number] | null = null;
    cv.addEventListener("pointerdown", (ev) => {
      dragging = [ev.offsetX, ev.offsetY];
      // So a pan that leaves the canvas still tracks -- without it the view freezes at the edge and
      // the pointerup never arrives, leaving the pan glued to the cursor until the next press.
      cv.setPointerCapture(ev.pointerId);
    });
    // `pointercancel` as well as `pointerup`: a captured stream can still be cancelled (the OS takes
    // the gesture, the element is removed), and only releasing on `pointerup` leaves `dragging` set
    // so the next hover pans instead of reading out a potential.
    const release = (ev: PointerEvent): void => {
      if (dragging === null) return;
      dragging = null;
      cv.releasePointerCapture(ev.pointerId);
    };
    cv.addEventListener("pointerup", release);
    cv.addEventListener("pointercancel", release);
    cv.addEventListener("pointermove", (ev) => {
      if (dragging) {
        view = panned(view, ev.offsetX - dragging[0], ev.offsetY - dragging[1]);
        dragging = [ev.offsetX, ev.offsetY];
        render();
        return;
      }
      const [wx, wy] = toWorld(view, ev.offsetX, ev.offsetY);
      const i = nearest(xs, ys, wx, wy);
      cv.title = `φ = ${b.prefix.potential[state.get().prefix]![i]!.toPrecision(4)}`;
    });
    cv.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      view = zoomed(view, ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.offsetX, ev.offsetY);
      render();
    }, { passive: false });
  };

  // The observed element is the CANVAS ITSELF, not the container. `cv.style.width = "100%"` makes
  // the canvas's own content box track the container's content width, so observing it answers the
  // question this widget actually has ("how many CSS pixels am I drawing into?") in one hop instead
  // of two -- and it is the box `sizeCanvas` scales the backing store to, so there is no second
  // element whose padding or border could put the two out of step.
  //
  // Re-fitting on every such change is the fix: `width: 100%` already stretched the canvas when its
  // container narrowed, but nothing re-sized the backing store or re-fitted the bbox, so the drawing
  // stayed scaled for the width it had at mount -- and a window listener never saw it, because the
  // window had not moved. No dedupe on width here (as Frontier needs): `aspect-ratio: 1 / 1` ties
  // this box's height to its width, and `sizeCanvas` touches only the backing store, so the observer
  // cannot be re-entered by our own drawing.
  //
  // `runOrReport`: a real ResizeObserver delivers its callbacks from the browser's own dispatch,
  // NOT from inside boot()'s promise chain, so that chain's `.catch` cannot see a throw in here --
  // it would be an uncaught exception with a blank figure and no message (see dom/error.ts).
  let firstDraw = true;
  observeSize(cv, (measured) => runOrReport(host, LABEL, () => {
    size = measured;
    sizeCanvas(cv, size);
    view = fitBbox(bbox, size.width, size.height);
    render();
    if (firstDraw) {
      firstDraw = false;
      state.subscribe(render);
      wireInteraction();
      // Only now: the static picture is the honest one until a real one has replaced it.
      removeFallbackImage(host);
    }
  }));
}
