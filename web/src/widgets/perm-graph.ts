import type { Bundle } from "../bundle.js";
import { showWidgetError } from "../dom/error.js";
import { draw, sizeCanvas } from "../render/canvas.js";
// Type-only: erased at compile time, so this produces no runtime import of mount.js. A runtime
// import here would recreate the mount<->widget cycle that made the bundle throw on load (see
// mount.ts's registration comment) -- this module must never import `register` from mount.js.
import type { Widget } from "../mount.js";
import { fitBbox, nearest, panned, toWorld, zoomed, type View } from "../view/transform.js";
import type { StateFactory, StateSource } from "../state.js";

interface PermGraphState { prefix: number; layer: "conductance" | "current"; halos: boolean }

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

export const permGraph: Widget = (host, makeState) => {
  const src = host.dataset.bundle!;
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
    .catch((err: unknown) => showWidgetError(host, "PermGraph", err));
};

function boot(host: HTMLElement, makeState: StateFactory, b: Bundle): void {
  const state: StateSource<PermGraphState> = makeState(initialState(host));
  const fallback = host.querySelector("img");
  const caption = host.querySelector("figcaption");

  const cv = document.createElement("canvas");
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";

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

  // Insert the canvas and controls BEFORE the figcaption, and remove the <img> -- so the mount
  // point (now the <figure> itself, fix wave I4) keeps picture-then-caption reading order, the
  // same order every sibling figure in .sbu-figure-grid uses. Appending after </figure> (the prior
  // behaviour) put this one cell's caption ahead of its picture, out of step with its neighbours.
  if (caption) {
    host.insertBefore(cv, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(cv, controls);
  }
  if (fallback) fallback.remove();

  const xs = b.nodes.cx, ys = b.nodes.cy;
  const bbox = { minX: Math.min(...xs), minY: Math.min(...ys),
                 maxX: Math.max(...xs), maxY: Math.max(...ys) };
  let size = sizeCanvas(cv);
  let view: View = fitBbox(bbox, size.width, size.height);
  const ctx = cv.getContext("2d")!;

  const render = (): void => {
    const s = state.get();
    draw(ctx, b, { view, ...s }, size);
    // Every number the picture shows is also present as text.
    readout.textContent =
      `${b.prefix.road_m[s.prefix]!.toFixed(0)} m of road · ` +
      `${(b.prefix.permeability[s.prefix]! * 100).toFixed(1)}% permeability`;
  };
  state.subscribe(render);

  let dragging: [number, number] | null = null;
  cv.addEventListener("pointerdown", (ev) => { dragging = [ev.offsetX, ev.offsetY]; });
  cv.addEventListener("pointerup", () => { dragging = null; });
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
  window.addEventListener("resize", () => {
    size = sizeCanvas(cv);
    view = fitBbox(bbox, size.width, size.height);
    render();
  });

  render();
}
