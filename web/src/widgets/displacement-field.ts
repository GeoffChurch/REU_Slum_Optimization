import type { FieldBundle, Road } from "../field.js";
// Type-only: erased at compile time, so this module has NO runtime import of mount.js. A runtime
// one would recreate the mount<->widget cycle that made the whole bundle throw during module
// evaluation while the page still looked fine (see mount.ts's registration comment). This file must
// never import `register`.
import type { Widget } from "../mount.js";
import type { StateFactory, StateSource } from "../state.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { contributions, corridorDistance, flatten, sumC } from "../model/displacement.js";
import { sizeCanvas } from "../render/canvas.js";
import { drawField, handles, type Handle } from "../render/field.js";
import { fitBbox, nearest, toScreen, toWorld, type Bbox, type View } from "../view/transform.js";

/** Both roads, always -- `second` says whether road 2 is switched ON, not whether it exists. Keeping
 * road 2's geometry in state while it is off means toggling it back on returns it exactly where the
 * reader last dragged it, rather than snapping it back to the baked line and quietly undoing their
 * work. */
interface FieldState { roads: Road[]; second: boolean }

/** The name every failure of this widget is reported under -- one constant, because it is used from
 * two unrelated places (the fetch chain and the resize callback) and two spellings of it would read
 * to a reader as two different widgets failing. */
const LABEL = "DisplacementField";

/** The roads whose corridor is charged and drawn. Road 1 is always live; road 2 is the toggle. */
function activeRoads(s: FieldState): Road[] {
  return s.second ? s.roads : s.roads.slice(0, 1);
}

/** The view fits the buildings UNIONED WITH the parcel rings, never the buildings alone.
 *
 * `PermGraph` fits to node centroids while drawing parcels, streets and roads, and the backlog
 * records the consequence measured on this very block: one parcel vertex of 1850 lands 0.4 px
 * outside a 600 px canvas, absorbed by the 4 % pad -- it works by luck. Here the gap is far larger
 * (buildings span 5.8-149.5 m against the parcels' 0-157.8 m), so inheriting that luck would clip
 * the wireframe, the boundary and the streets on every side. */
function fieldBbox(b: FieldBundle): Bbox {
  const xs: number[][] = [b.buildings.x, ...b.parcels.map((ring) => ring.map((p) => p[0]))];
  const ys: number[][] = [b.buildings.y, ...b.parcels.map((ring) => ring.map((p) => p[1]))];
  const fold = (vs: number[][], f: (...n: number[]) => number, seed: number): number =>
    vs.reduce((acc, v) => f(acc, ...v), seed);
  return {
    minX: fold(xs, Math.min, Infinity),
    minY: fold(ys, Math.min, Infinity),
    maxX: fold(xs, Math.max, -Infinity),
    maxY: fold(ys, Math.max, -Infinity),
  };
}

export const displacementField: Widget = (host, makeState) => {
  const src = host.dataset.bundle!;
  // A 404, a renamed bundle field, or any throw inside boot() must be VISIBLE on the page, not an
  // unhandled rejection sitting in the console while the PNG fallback keeps the page looking fine
  // -- verbatim the defect class this branch has found repeatedly.
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<FieldBundle>;
    })
    .then((b) => boot(host, makeState, b))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

function boot(host: HTMLElement, makeState: StateFactory, b: FieldBundle): void {
  const e = b.encoding;
  // The bundle is a BOUNDARY: it arrives over the network, and a page can outlive the artifact it
  // was generated beside. Two roads is not a preference here, it is what the widget IS -- the
  // toggle, the overlap demonstration and the handle hit test all assume road 2 exists -- so a
  // bundle carrying fewer must fail loudly, on the page, with the static picture left in place,
  // rather than reach a `!` and draw a widget with a dead checkbox.
  const [road1, road2] = b.roads;
  if (road1 === undefined || road2 === undefined) {
    throw new Error(`field.json carries ${b.roads.length} roads; this widget needs two`);
  }
  const width0 = b.width.default_m;
  const state: StateSource<FieldState> = makeState<FieldState>({
    roads: [{ ...road1, width_m: width0 }, { ...road2, width_m: width0 }],
    second: false,
  });

  const caption = host.querySelector("figcaption");

  const cv = document.createElement("canvas");
  // Inline styles, never presentation attributes: Material's `.md-typeset svg{height:auto;
  // max-width:100%}` beats a presentation attribute, which cost D1 a Critical at its final gate.
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";
  // Without this a touch drag scrolls the page and the browser CANCELS the pointer stream mid-drag,
  // so the handle sticks. It is the one line that makes Pointer Events actually serve touch rather
  // than only appear to.
  cv.style.touchAction = "none";

  const controls = document.createElement("div");

  const widthLabel = document.createElement("label");
  const slider = document.createElement("input");
  slider.type = "range";                       // native: keyboard- and screen-reader-reachable
  // The floor IS `PermeabilityParams.min_road_width_m`: permeability.py:205 RAISES below it, as too
  // narrow for two directions. A widget that let a reader build a road the pipeline rejects would
  // be teaching a model this project does not have. Every bound is baked, none is a literal here.
  slider.min = String(b.width.floor_m);
  slider.max = String(b.width.max_m);
  slider.step = String(b.width.step_m);
  slider.value = String(width0);
  slider.addEventListener("input", () => {
    // The width applies to EVERY live road. Width is per-road in the model (permeability.py's
    // module docstring) and the widget offers one control, so the choice is which roads it moves:
    // moving all of them is what keeps the overlap demonstration exact, since two coincident roads
    // of one width are algebraically one road, while two coincident roads of different widths are
    // just the wider one and the reader would be watching a weaker claim.
    const w = Number(slider.value);
    state.set({ roads: state.get().roads.map((r) => ({ ...r, width_m: w })) });
  });
  widthLabel.append("Road width ", slider);

  const secondLabel = document.createElement("label");
  const secondBox = document.createElement("input");
  secondBox.type = "checkbox";
  secondBox.checked = state.get().second;
  secondBox.addEventListener("change", () => state.set({ second: secondBox.checked }));
  secondLabel.append(secondBox, " Second road");

  const readout = document.createElement("p");
  controls.append(widthLabel, secondLabel, readout);

  // Canvas and controls go BEFORE any <figcaption>, so the mount point keeps picture-then-caption
  // reading order -- the same order every sibling figure on the site uses.
  //
  // The fallback <img> is NOT removed here: it goes after the first successful draw, below, so a
  // canvas that mounts into a zero-width container (a hidden tab, a closed <details>, print) or a
  // draw that throws leaves the static picture the error text points the reader at.
  if (caption) {
    host.insertBefore(cv, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(cv, controls);
  }

  const bbox = fieldBbox(b);
  // Both assigned by the observer below and by nothing else. Everything that reads them -- `render`,
  // the pointer handlers -- is wired from inside the first sized callback, so there is no ordering
  // in which they are read before that callback has run.
  let size: { width: number; height: number };
  let view: View;
  const ctx = cv.getContext("2d")!;

  const render = (): void => {
    const roads = activeRoads(state.get());
    const d = corridorDistance(b.buildings.x, b.buildings.y, flatten(roads));
    // The disks' shading and the number under them come from ONE distance array through one
    // formula (`contributions`, which `sumC` is a sum of), so the picture and the readout cannot
    // disagree about what the road costs.
    drawField(ctx, b, { view, roads, c: contributions(b.buildings.r, d) }, size);
    const total = sumC(b.buildings.r, d);
    // Cost only -- both numbers, every frame. The page defines displacement as Σcᵢ and reports the
    // fraction, so quoting one and not the other would make the widget disagree with the prose
    // above it. There is deliberately no verdict here: the benefit half of the tradeoff is
    // permeability, which needs a sparse solve this widget does not do.
    readout.textContent =
      `Cost: ${total.toFixed(1)} homes displaced · `
      + `${(total / b.n_buildings * 100).toFixed(1)}% of ${b.n_buildings} buildings · `
      // `roads[0]` is road 1, which always exists: boot builds the array from the two roads it
      // validated, and every writer here `.map`s it rather than resizing it.
      + `road width ${state.get().roads[0]!.width_m.toFixed(1)} m`;
  };

  /** The handle under a press, or null if the press was not on one.
   *
   * Screen space, not world: `handle_radius_px` is a pixel radius (a grab target that shrank with
   * the view would stop being grabbable), and the same `handles()` the drawing uses supplies the
   * positions, so a handle can never be drawn somewhere the reader cannot grab it. */
  const pickHandle = (sx: number, sy: number): Handle | null => {
    const hs = handles(activeRoads(state.get()));
    const pts = hs.map((h) => toScreen(view, h.x, h.y));
    // `nearest` returns -1 only for an EMPTY list, and the live set always contains road 1 --
    // `activeRoads` builds it from the two-road bundle validated at boot. A guard here would be the
    // unreachable kind this project's directives forbid.
    const i = nearest(pts.map((p) => p[0]), pts.map((p) => p[1]), sx, sy);
    const [hx, hy] = pts[i]!;
    // A press further than this takes NOTHING, deliberately: without it, a press on empty canvas
    // teleports the nearest handle a hundred metres and the reader's road is destroyed by a click
    // they did not mean as a drag.
    return Math.hypot(hx - sx, hy - sy) <= e.handle_radius_px * 2 ? hs[i]! : null;
  };

  const wireInteraction = (): void => {
    let dragging: Handle | null = null;
    cv.addEventListener("pointerdown", (ev) => {
      const hit = pickHandle(ev.offsetX, ev.offsetY);
      if (hit === null) return;
      dragging = hit;
      // So a drag that leaves the canvas still tracks -- without it the road freezes at the edge
      // and the pointerup never arrives, leaving the handle stuck to the cursor.
      cv.setPointerCapture(ev.pointerId);
    });
    cv.addEventListener("pointermove", (ev) => {
      const held = dragging;
      if (held === null) return;
      const [wx, wy] = toWorld(view, ev.offsetX, ev.offsetY);
      state.set({
        roads: state.get().roads.map((road, r) => r !== held.road ? road : {
          width_m: road.width_m,
          coords: road.coords.map((pt, v): [number, number] => v === held.vertex ? [wx, wy] : pt),
        }),
      });
    });
    const release = (ev: PointerEvent): void => {
      if (dragging === null) return;
      dragging = null;
      cv.releasePointerCapture(ev.pointerId);
    };
    cv.addEventListener("pointerup", release);
    cv.addEventListener("pointercancel", release);
  };

  // The observed element is the CANVAS ITSELF: `width: 100%` makes its content box track the
  // container's width, so observing it answers this widget's actual question ("how many CSS pixels
  // am I drawing into?") in one hop, and it is the box `sizeCanvas` scales the backing store to.
  //
  // `runOrReport`: a real ResizeObserver delivers from the browser's own dispatch, OUTSIDE the
  // fetch chain above, so that chain's `.catch` cannot see a throw in here -- it would be an
  // uncaught exception with a blank figure and no message (see dom/error.ts).
  let firstDraw = true;
  observeSize(cv, (measured) => runOrReport(host, LABEL, () => {
    size = measured;
    sizeCanvas(cv, size);
    view = fitBbox(bbox, size.width, size.height, e.pad);
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
