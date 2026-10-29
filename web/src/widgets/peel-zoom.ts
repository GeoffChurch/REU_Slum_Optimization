// Type-only: erased at compile time, so this module has NO runtime import of mount.js (the cycle
// mount.ts's registration comment describes). This file must never import `register`.
import type { Widget } from "../mount.js";
import type { StateSource } from "../state.js";
import { numberParam, type UrlCodec } from "../url/param.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport } from "../dom/error.js";
import { observeSize } from "../dom/resize.js";

/** The Demo's peel figure as one zoomable frame (owner, 2026-10-29): the flagship run's depth
 * render, whole at first; pinch, Ctrl/⌘-wheel, the +/− buttons or a double tap/click zoom in, a
 * drag pans once zoomed, and a button resets. The view is URL state, so a link -- or the page's QR code -- carries a zoomed
 * view.
 *
 * Nothing hijacks the page's scroll: a plain wheel and a one-finger swipe on an unzoomed frame
 * scroll the page as anywhere else (the frame's `touch-action` lets the browser have them); only a
 * deliberate gesture -- a modified wheel, two fingers, a drag on a zoomed image -- moves the view.
 *
 * No library: pointer events and one CSS transform on the image. The page's generator writes the
 * frame, the image (marked skip-lightbox: a lightbox would fight the gestures) and the image's
 * native width (`data-native-px`), which caps the zoom: past it there is no detail
 * left to show. */
const LABEL = "PeelZoom";

/** The view: how many times the frame's width the image is drawn at (1 = whole), and the image
 * point at the frame's centre, as fractions of the image's width and height. */
export interface PeelView {
  zoom: number;
  cx: number;
  cy: number;
}

export const PEEL_URL: UrlCodec<PeelView> = {
  zoom: numberParam("peel", 3),
  cx: numberParam("peelx", 4),
  cy: numberParam("peely", 4),
};

export const WHOLE: PeelView = { zoom: 1, cx: 0.5, cy: 0.5 };

/** `v` made legal: the zoom between the whole image and `maxZoom`, and the centre where the
 * visible window stays inside the image -- so no view, from a drag, a gesture or a hand-edited
 * URL, shows anything but the image. */
export function clampView(v: PeelView, maxZoom: number): PeelView {
  const zoom = Math.min(Math.max(v.zoom, 1), Math.max(1, maxZoom));
  const half = 0.5 / zoom;
  const inside = (c: number): number => Math.min(Math.max(c, half), 1 - half);
  return { zoom, cx: inside(v.cx), cy: inside(v.cy) };
}

/** Zoom by `factor` about the frame point (`fx`, `fy`), fractions of the frame: the image point
 * under it stays under it, as far as the clamp allows. */
export function zoomAt(v: PeelView, factor: number, fx: number, fy: number,
                       maxZoom: number): PeelView {
  const ux = v.cx + (fx - 0.5) / v.zoom;
  const uy = v.cy + (fy - 0.5) / v.zoom;
  const zoom = Math.min(Math.max(v.zoom * factor, 1), Math.max(1, maxZoom));
  return clampView({ zoom, cx: ux - (fx - 0.5) / zoom, cy: uy - (fy - 0.5) / zoom }, maxZoom);
}

/** Drag the image by (`dx`, `dy`), fractions of the frame. */
export function panBy(v: PeelView, dx: number, dy: number, maxZoom: number): PeelView {
  return clampView({ zoom: v.zoom, cx: v.cx - dx / v.zoom, cy: v.cy - dy / v.zoom }, maxZoom);
}

/** The image's transform for `v`, its origin at its top-left: scaled by the zoom and moved so the
 * view's window fills the frame. Percentages of the image's own box, which is the frame's. */
export function transformOf(v: PeelView): string {
  const tx = -(v.cx - 0.5 / v.zoom) * v.zoom * 100;
  const ty = -(v.cy - 0.5 / v.zoom) * v.zoom * 100;
  return `translate(${tx}%, ${ty}%) scale(${v.zoom})`;
}

const STEP = 1.6;            // one +/− press, or a double tap
const WHEEL = 0.002;         // zoom per pixel of a modified wheel's delta, exponentially
const TAP_MS = 300;          // two taps this close make a double tap
const TAP_PX = 24;           // ... landing this close together
const DRAG_PX = 4;           // a pointer that moves this far is a drag, not a tap

function button(cls: string, text: string, label: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `sbu-peel__btn ${cls}`;
  b.textContent = text;
  b.setAttribute("aria-label", label);
  return b;
}

// A throw while setting up (a missing attribute) reaches mountAll, which reports it on the
// figure like every widget's; after that, a drawing failure is reported by `runOrReport`.
export const peelZoom: Widget<PeelView> = (host, makeState) => {
  const native = Number(requireAttr(host.dataset.nativePx, "data-native-px", LABEL));
  const img = host.querySelector("img");
  if (img === null) throw new Error(`${LABEL}: no image`);
  const frame = img.parentElement!;

  let width = 0;
  // The cap: the image's native pixels across the frame's CSS width. Unknown until laid out, so
  // a URL's zoom is clamped again once the frame has a width.
  const maxZoom = (): number => (width > 0 ? native / width : 1);
  const state: StateSource<PeelView> = makeState(WHOLE);

  const controls = document.createElement("div");
  controls.className = "sbu-peel__controls";
  const zoomIn = button("sbu-peel__in", "+", "Zoom in");
  const zoomOut = button("sbu-peel__out", "−", "Zoom out");
  const reset = button("sbu-peel__reset", "Reset", "Show the whole block");
  const hint = document.createElement("p");
  hint.className = "sbu-peel__hint";
  hint.textContent = "Ctrl-scroll or pinch to zoom; drag to pan";
  controls.append(zoomIn, zoomOut, reset, hint);
  frame.parentElement!.insertBefore(controls, frame);

  const set = (v: PeelView): void => state.set(clampView(v, maxZoom()));
  zoomIn.addEventListener("click", () => set(zoomAt(state.get(), STEP, 0.5, 0.5, maxZoom())));
  zoomOut.addEventListener("click", () => set(zoomAt(state.get(), 1 / STEP, 0.5, 0.5, maxZoom())));
  reset.addEventListener("click", () => set(WHOLE));

  const render = (): void => {
    const v = state.get();
    img.style.transform = transformOf(v);
    // Unzoomed, a one-finger swipe is the page's to scroll; zoomed, a drag pans the image.
    frame.style.touchAction = v.zoom > 1 ? "none" : "pan-x pan-y";
    frame.dataset.zoomed = v.zoom > 1 ? "yes" : "no";
    zoomIn.disabled = v.zoom >= maxZoom() - 1e-9;
    zoomOut.disabled = v.zoom <= 1;
  };

  /** A client point as fractions of the frame. */
  const at = (x: number, y: number): [number, number] => {
    const r = frame.getBoundingClientRect();
    return [(x - r.left) / r.width, (y - r.top) / r.height];
  };

  frame.addEventListener("wheel", (ev: WheelEvent) => {
    // A plain wheel scrolls the page; only Ctrl/⌘ (and a trackpad pinch, which browsers report
    // as a Ctrl-wheel) zooms.
    if (!(ev.ctrlKey || ev.metaKey)) return;
    ev.preventDefault();
    const [fx, fy] = at(ev.clientX, ev.clientY);
    set(zoomAt(state.get(), Math.exp(-ev.deltaY * WHEEL), fx, fy, maxZoom()));
  }, { passive: false });

  // Pointers down on the frame: one pans (zoomed) or taps; two pinch.
  const down = new Map<number, { x: number; y: number }>();
  let moved = false;
  let lastTap: { t: number; x: number; y: number } | null = null;
  frame.addEventListener("pointerdown", (ev: PointerEvent) => {
    down.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (down.size === 1) moved = false;
    if (state.get().zoom > 1 || down.size === 2) frame.setPointerCapture(ev.pointerId);
  });
  frame.addEventListener("pointermove", (ev: PointerEvent) => {
    const prev = down.get(ev.pointerId);
    if (prev === undefined) return;
    const r = frame.getBoundingClientRect();
    if (down.size === 2) {
      const [a, b] = [...down.values()];
      const other = a === prev ? b! : a!;
      const before = Math.hypot(prev.x - other.x, prev.y - other.y);
      const after = Math.hypot(ev.clientX - other.x, ev.clientY - other.y);
      const [fx, fy] = at((ev.clientX + other.x) / 2, (ev.clientY + other.y) / 2);
      if (before > 0) set(zoomAt(state.get(), after / before, fx, fy, maxZoom()));
      moved = true;
    } else if (state.get().zoom > 1) {
      set(panBy(state.get(), (ev.clientX - prev.x) / r.width, (ev.clientY - prev.y) / r.height,
                maxZoom()));
    }
    if (Math.hypot(ev.clientX - prev.x, ev.clientY - prev.y) > DRAG_PX) moved = true;
    down.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  });
  const up = (ev: PointerEvent): void => {
    const was = down.size;
    down.delete(ev.pointerId);
    if (ev.type !== "pointerup" || was !== 1 || moved) return;
    // A tap: the second of two close in time and place zooms in there.
    const t = ev.timeStamp;
    if (lastTap !== null && t - lastTap.t < TAP_MS
        && Math.hypot(ev.clientX - lastTap.x, ev.clientY - lastTap.y) < TAP_PX) {
      const [fx, fy] = at(ev.clientX, ev.clientY);
      set(zoomAt(state.get(), STEP, fx, fy, maxZoom()));
      lastTap = null;
    } else {
      lastTap = { t, x: ev.clientX, y: ev.clientY };
    }
  };
  frame.addEventListener("pointerup", up);
  frame.addEventListener("pointercancel", up);

  observeSize(frame, (size) => runOrReport(host, LABEL, () => {
    const first = width === 0;
    width = size.width;
    // A URL zoom past this width's cap, or a resize that lowered it, is pinned to the cap.
    const legal = clampView(state.get(), maxZoom());
    if (legal.zoom !== state.get().zoom || legal.cx !== state.get().cx
        || legal.cy !== state.get().cy) {
      state.set(legal);
    }
    render();
    if (first) state.subscribe(render);
  }));
};
