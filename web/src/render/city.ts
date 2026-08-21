/** The ScreenMap canvas: 16,451 Cape Town blocks (3,500 Nairobi), a pale base layer painted once
 * onto an offscreen canvas and blitted back every frame, with a red selected-prefix layer filled
 * directly on top.
 *
 * **Why a real blit, not a "paint minus undo" delta.** An earlier version of this module painted
 * the base layer once onto the VISIBLE canvas and, on every later frame, tried to restore just the
 * previously-selected blocks by re-filling them in `base_color` before filling the new prefix in
 * `selected_color`. That does not work at Cape Town's own geometry: the median block is ~0.6 CSS
 * px² at a 700 px canvas (measured against the committed bundle), which is entirely antialiased
 * edge coverage -- there is no interior pixel a re-fill can cleanly overwrite. Source-over
 * compositing `selected_color` onto a partially-covered pixel and then `base_color` back over it
 * does not restore the original colour; repeated select/deselect cycles converge toward a
 * permanent pink residue (a coverage-α pixel converges to (1−α)/(2−α) of the way to
 * `selected_color`, never back to 0). A `drawImage` pixel COPY has no such residue: every frame
 * starts from the offscreen canvas's own untouched pixels, unconditionally.
 *
 * **Why the offscreen canvas is a plain, uninserted `<canvas>`, not `OffscreenCanvas`.** Both give
 * the same 2D context and `drawImage` accepts either; a second `<canvas>` element needs no new
 * global and is what `web/test/harness.ts`'s fake `document.createElement` already knows how to
 * produce (its own `RecordingContext`, `drawImage`-able like any other).
 *
 * `region.ts`'s neighbourhood tops out at 213 blocks, so redrawing everything every frame there is
 * free. At city scale it is not: `paintBase` below is the ONE O(n_blocks) pass this module ever
 * makes for a given (bundle, view) pair, run once into the offscreen canvas. Every later frame --
 * a floor drag, a metric switch -- goes through `paintFrame` alone: one `clearRect`, one
 * `drawImage` (the whole base layer, correctly restored, in one call), and O(k) fills for the
 * current selected prefix -- never another O(n_blocks) pass. That is what keeps a floor-slider
 * drag cheap at 1,655 selected blocks rather than 16,451; `web/test/screen-map-boot.test.ts`'s
 * base-layer test makes the claim checkable by following a frame's own `drawImage` call to the
 * offscreen canvas it names and counting what THAT canvas's own log holds. */
import type { CityBundle, CityEncoding } from "../screen_map.js";
import { sizeCanvas } from "./canvas.js";
import { toScreen, type View } from "../view/transform.js";

/** One block's rings (exterior first, then any interiors), projected to screen space. */
type ScreenRings = [number, number][][];

/** The last VIEW this module projected for, and every block's rings already computed for it --
 * mirrors `render/region.ts`'s own `screenRingsOf` cache, one level up in scale. Keyed on `bundle`
 * by reference: a city switch is a different bundle object, which invalidates it exactly like a
 * resized `view` does. Sharing this across widget instances is harmless (worst case, an extra
 * recompute), unlike the offscreen canvas itself -- see `createLayer`'s own comment. */
let cache: { bundle: CityBundle; view: View; screen: ScreenRings[] } | null = null;

function screenRingsOf(bundle: CityBundle, view: View): ScreenRings[] {
  if (cache !== null && cache.bundle === bundle && cache.view === view) return cache.screen;
  const screen = bundle.rings.map(
    (rings) => rings.map((ring) => ring.map(([x, y]) => toScreen(view, x, y))));
  cache = { bundle, view, screen };
  return screen;
}

function tracePath(ctx: CanvasRenderingContext2D, rings: ScreenRings): void {
  ctx.beginPath();
  for (const ring of rings) {
    ring.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.closePath();
  }
}

/** One beginPath()/fill() PER block, matching `region.ts`'s own reasoning: the boot test counts
 * selected blocks by counting `fill()` calls in `selected_color`, and a single batched path across
 * many blocks would collapse that count to one regardless of how many blocks it covers. */
function fillBlock(ctx: CanvasRenderingContext2D, rings: ScreenRings, style: string): void {
  ctx.fillStyle = style;
  tracePath(ctx, rings);
  // Even-odd, per screen_map.d.ts's own note on `rings`: "Exterior ring first, then interiors."
  // Nonzero (the default) would fill Cape Town's 6,990 holes solid instead of cutting them out.
  ctx.fill("evenodd");
}

function strokeBlock(ctx: CanvasRenderingContext2D, rings: ScreenRings, style: string,
                     lineWidth: number): void {
  ctx.strokeStyle = style;
  ctx.lineWidth = lineWidth;
  tracePath(ctx, rings);
  ctx.stroke();
}

export interface CityLayer {
  /** The one expensive pass: paints the OFFSCREEN base layer (every block filled `e.base_color`,
   * Cape Town's real informal blocks filled `e.informal_color` instead, every block outlined in
   * `e.base_color` at `e.block_lw` so adjacent same-coloured blocks stay distinguishable). Call
   * only when the view or the active bundle changes (a resize or a city switch) -- never for a
   * floor or metric change, which `paintFrame` alone handles by re-blitting this unchanged. */
  paintBase(bundle: CityBundle, view: View, e: CityEncoding,
           size: { width: number; height: number }): void;
  /** Every frame: clear `ctx`, blit the offscreen base layer onto it (a single `drawImage`, in
   * device-pixel space so a DPR-scaled backing store is copied 1:1 rather than re-scaled), then
   * fill `order[0..k)` in `e.selected_color` directly on `ctx`, in the CSS-pixel space every other
   * draw call in this codebase uses. */
  paintFrame(ctx: CanvasRenderingContext2D, bundle: CityBundle, view: View, e: CityEncoding,
            order: Int32Array, k: number, size: { width: number; height: number }): void;
}

/** Builds one OFFSCREEN `<canvas>` (never inserted into the document -- a pure pixel buffer) and
 * closes over it, so its contents belong to ONE widget instance. Deliberately not module state the
 * way `cache` above is: two ScreenMaps sharing one offscreen canvas would blit whichever one last
 * painted it into BOTH widgets' visible canvases. */
export function createLayer(): CityLayer {
  const base = document.createElement("canvas");
  const baseCtx = base.getContext("2d")!;
  return {
    paintBase(bundle, view, e, size) {
      sizeCanvas(base, size); // matches the DESTINATION's own backing-store size exactly, so
                              // `paintFrame`'s blit is a 1:1 device-pixel copy, never a rescale
      baseCtx.clearRect(0, 0, size.width, size.height);
      const screen = screenRingsOf(bundle, view);
      const informal = bundle.informal;
      for (let i = 0; i < screen.length; i++) {
        const rings = screen[i]!;
        fillBlock(baseCtx, rings, informal?.[i] ? e.informal_color : e.base_color);
        strokeBlock(baseCtx, rings, e.base_color, e.block_lw);
      }
    },
    paintFrame(ctx, bundle, view, e, order, k, size) {
      ctx.clearRect(0, 0, size.width, size.height);
      // The blit happens in DEVICE-PIXEL space: `base` and `ctx`'s own canvas were sized to the
      // same backing-store dimensions by the same `sizeCanvas` call (above, and in screen-map.ts's
      // resize handler), so copying them 1:1 needs the identity transform, not the CSS-pixel-space
      // one `sizeCanvas` leaves active -- drawing `base` under THAT transform would scale its
      // already-DPR-sized pixels up by another factor of `dpr`, blurring the whole picture.
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(base, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // back to CSS-pixel space for the fills below
      const screen = screenRingsOf(bundle, view);
      for (let i = 0; i < k; i++) fillBlock(ctx, screen[order[i]!]!, e.selected_color);
    },
  };
}
