// Type-only: erased at compile time, so this module has NO runtime import of mount.js (the cycle
// mount.ts's registration comment describes). This file must never import `register`.
import type { Widget } from "../mount.js";
import type { StateSource } from "../state.js";
import { numberParam, type UrlCodec } from "../url/param.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport } from "../dom/error.js";
import { observeSize } from "../dom/resize.js";
import { controlButton } from "../dom/controls.js";

/** The Demo's peel figure as one zoomable frame (owner, 2026-10-29): the flagship run's depth
 * render, whole at first; a two-finger pinch, Ctrl/⌘-wheel, the +/− buttons or a double
 * tap/click zoom in, two fingers or a mouse drag pan once zoomed, and a button resets. The view is
 * URL state, so a link -- or the page's QR code -- carries a zoomed view.
 *
 * Nothing hijacks the page's scroll, as on a map: a plain wheel, and one finger on a touch screen
 * whether zoomed or not, scroll the page as anywhere else; a one-finger drag only says, briefly,
 * "Use two fingers to move the image". The view moves for a deliberate gesture: a Ctrl/⌘-wheel,
 * two fingers (pan and pinch), or a mouse or pen drag on a zoomed image.
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
const NOTICE_MS = 1500;      // how long the two-finger notice stays before it fades

interface Point { x: number; y: number }

/** Two fingers moving from `from` to `to` (client points; `rect` the frame's box): the image
 * follows their midpoint and scales by the change in their spread, about that midpoint. */
export function twoFingers(v: PeelView, from: readonly [Point, Point], to: readonly [Point, Point],
                           rect: { left: number; top: number; width: number; height: number },
                           maxZoom: number): PeelView {
  const mid = (p: readonly [Point, Point]): Point =>
    ({ x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 });
  const spread = (p: readonly [Point, Point]): number =>
    Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
  const [m0, m1] = [mid(from), mid(to)];
  const panned = panBy(v, (m1.x - m0.x) / rect.width, (m1.y - m0.y) / rect.height, maxZoom);
  const factor = spread(from) > 0 ? spread(to) / spread(from) : 1;
  return zoomAt(panned, factor, (m1.x - rect.left) / rect.width, (m1.y - rect.top) / rect.height,
                maxZoom);
}

function button(cls: string, text: string, label: string): HTMLButtonElement {
  const b = controlButton(text, cls);
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

  /** A tap at a client point: the second of two close in time and place zooms in there. */
  let lastTap: { t: number; x: number; y: number } | null = null;
  const tap = (t: number, x: number, y: number): void => {
    if (lastTap !== null && t - lastTap.t < TAP_MS
        && Math.hypot(x - lastTap.x, y - lastTap.y) < TAP_PX) {
      const [fx, fy] = at(x, y);
      set(zoomAt(state.get(), STEP, fx, fy, maxZoom()));
      lastTap = null;
    } else {
      lastTap = { t, x, y };
    }
  };

  // ---- mouse and pen: drag to pan once zoomed; a double click zooms in.
  let drag: { id: number; x: number; y: number; moved: boolean } | null = null;
  frame.addEventListener("pointerdown", (ev: PointerEvent) => {
    if (ev.pointerType === "touch") return;          // touch is the touch handlers' (below)
    drag = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, moved: false };
    if (state.get().zoom > 1) frame.setPointerCapture(ev.pointerId);
  });
  frame.addEventListener("pointermove", (ev: PointerEvent) => {
    if (drag === null || ev.pointerId !== drag.id) return;
    const r = frame.getBoundingClientRect();
    if (state.get().zoom > 1) {
      set(panBy(state.get(), (ev.clientX - drag.x) / r.width, (ev.clientY - drag.y) / r.height,
                maxZoom()));
    }
    if (Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y) > DRAG_PX) drag.moved = true;
    drag = { ...drag, x: ev.clientX, y: ev.clientY };
  });
  const release = (ev: PointerEvent): void => {
    if (drag === null || ev.pointerId !== drag.id) return;
    if (ev.type === "pointerup" && !drag.moved) tap(ev.timeStamp, ev.clientX, ev.clientY);
    drag = null;
  };
  frame.addEventListener("pointerup", release);
  frame.addEventListener("pointercancel", release);

  // ---- touch, as on a map: one finger always scrolls the page (the frame's touch-action leaves
  // it to the browser, and nothing here cancels it), two fingers pan and pinch. A one-finger drag
  // on the image says so, briefly.
  const notice = document.createElement("div");
  notice.className = "sbu-peel__notice";
  notice.textContent = "Use two fingers to move the image";
  notice.setAttribute("aria-hidden", "true");
  frame.append(notice);
  let noticeTimer: number | null = null;
  const showNotice = (): void => {
    notice.dataset.shown = "yes";
    if (noticeTimer !== null) clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { notice.dataset.shown = "no"; },
                             NOTICE_MS) as unknown as number;
  };
  let pair: [Point, Point] | null = null;
  let one: { x: number; y: number; moved: boolean; noticed: boolean } | null = null;
  const points = (list: TouchList): Point[] =>
    Array.from({ length: list.length }, (_, i) => ({ x: list[i]!.clientX, y: list[i]!.clientY }));
  frame.addEventListener("touchstart", (ev: TouchEvent) => {
    const p = points(ev.touches);
    pair = p.length === 2 ? [p[0]!, p[1]!] : null;
    one = p.length === 1 ? { ...p[0]!, moved: false, noticed: false } : null;
  });
  frame.addEventListener("touchmove", (ev: TouchEvent) => {
    const p = points(ev.touches);
    if (p.length === 2 && pair !== null) {
      // Two fingers are the image's: neither the page's scroll nor its zoom.
      ev.preventDefault();
      const r = frame.getBoundingClientRect();
      set(twoFingers(state.get(), pair, [p[0]!, p[1]!], r, maxZoom()));
      pair = [p[0]!, p[1]!];
    } else if (p.length === 1 && one !== null) {
      if (Math.hypot(p[0]!.x - one.x, p[0]!.y - one.y) > DRAG_PX) one.moved = true;
      if (one.moved && !one.noticed) { one.noticed = true; showNotice(); }
    }
  }, { passive: false });
  frame.addEventListener("touchend", (ev: TouchEvent) => {
    if (one !== null && !one.moved && ev.touches.length === 0) {
      const [c] = points(ev.changedTouches);
      if (c !== undefined) tap(ev.timeStamp, c.x, c.y);
    }
    pair = null;
    one = null;
  });

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
