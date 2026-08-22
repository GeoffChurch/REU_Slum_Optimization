/** The ScreenMap canvas: 16,451 Cape Town blocks (3,500 Nairobi), a pale base layer painted once
 * onto an offscreen canvas and blitted back every frame, with the selected prefix OUTLINED on top
 * of it -- never filled over it.
 *
 * **Why the selected prefix is a stroke, not a fill.** An opaque `selected_color` fill over the
 * base layer erases whatever `paintBase` painted underneath, including `e.informal_color`. That
 * is not merely a colour swap: `gen_screen_map.py`'s `_render_screen_map` draws Cape Town's real
 * informal blocks filled gold (zorder 2) and the current selection OUTLINED red on top of that
 * (zorder 3, `facecolor="none"`), so a hit reads as "gold with a red ring", a false positive as
 * "red ring, no gold", and a miss as "gold, no ring" -- the legend `_render_screen_map`'s own
 * docstring states. A fill collapses the first two into indistinguishable solid red, silently
 * turning "gold" from "ground truth" into "what the screen missed", which shrinks toward zero as
 * the reader lowers the floor -- the opposite of what the widget is showing. Stroking at
 * `e.block_lw * 2` (matching the PNG's own selected linewidth) restores the legend exactly, at
 * the same O(k) cost a fill would have been.
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
 * `drawImage` (the whole base layer, correctly restored, in one call), and O(k) strokes for the
 * current selected prefix -- never another O(n_blocks) pass. That is what keeps a floor-slider
 * drag cheap at 1,655 selected blocks rather than 16,451; `web/test/screen-map-boot.test.ts`'s
 * base-layer test makes the claim checkable by following a frame's own `drawImage` call to the
 * offscreen canvas it names and counting what THAT canvas's own log holds. */
import type { CityBundle, CityEncoding } from "../screen_map.js";
import { sizeCanvas } from "./canvas.js";
import { toScreen, type View } from "../view/transform.js";

/** The follow ring's radius and line width, in CSS pixels -- SCREEN sizes, never world ones.
 * `view` fits a whole metro's extent into one canvas, which is what leaves a single block covering
 * a fraction of a pixel there (this module's own docstring above carries the measured median), so
 * the followed block's own outline would put nothing on screen. Nor can these two numbers be read
 * as world lengths and scaled: at that fit, six metres is hundredths of a pixel. (Six HUNDRED
 * metres would be legible -- the point is not that world units cannot work, it is that a number
 * chosen as pixels cannot be reinterpreted as metres, which is what multiplying by `view.scaleX`
 * would do.) `screen-map-boot.test.ts` measures the block-against-ring size relation on the
 * committed bundle, so a re-bake that pushed the followed block above either threshold there -- or
 * that broke the `index`/`block_id` pairing -- fails loudly instead of outdating a number written
 * into this comment. A re-bake onto a DIFFERENT sub-pixel block is not caught, and needs no
 * catching: nothing about it would be wrong. */
export const FOLLOW_RADIUS_PX = 6;
const FOLLOW_LW_PX = 2;

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

/** One beginPath()/fill() PER block, matching `region.ts`'s own reasoning: a test that counts
 * layer membership by counting `fill()` calls in a given colour needs one call per block, not one
 * batched path across many. Used only for the offscreen base layer now -- the selected prefix is
 * `strokeBlock`, below. */
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
   * Cape Town's real informal blocks filled `e.informal_color` instead, every block outlined at
   * `e.block_lw` in ITS OWN fill colour -- never unconditionally `e.base_color`, which would paint
   * over an informal block's gold: the median informal block is ~0.335 CSS px² at the shipped
   * canvas size, smaller than the outline itself, so an outline in a DIFFERENT colour from the
   * fill covers the whole interior, not merely the edge). Call only when the view or the active
   * bundle changes (a resize or a city switch) -- never for a floor or metric change, which
   * `paintFrame` alone handles by re-blitting this unchanged. */
  paintBase(bundle: CityBundle, view: View, e: CityEncoding,
           size: { width: number; height: number }): void;
  /** Every frame: clear `ctx`, blit the offscreen base layer onto it (a single `drawImage`, in
   * device-pixel space so a DPR-scaled backing store is copied 1:1 rather than re-scaled), then
   * OUTLINE `order[0..k)` in `e.selected_color` at `e.block_lw * 2` directly on `ctx`, in the
   * CSS-pixel space every other draw call in this codebase uses -- never filled, so whatever
   * `paintBase` painted underneath (in particular `e.informal_color`) stays visible through the
   * ring. Finally, where the bundle carries one, `bundle.follow` gets a `FOLLOW_RADIUS_PX` ring in
   * `e.follow_color`, drawn last so it sits over those outlines rather than under them. That draw
   * order is one of the two reasons it belongs on THIS layer; the other is that `paintBase` is the
   * strictly per-block pass, one fill and one outline for each of `bundle.rings` and nothing else.
   * Survival is NOT among the reasons: the base layer is blitted back whole on every frame, so a
   * ring painted into it would still reach the screen after a floor change (verified). */
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
  // The DPR `base`'s backing store was actually sized for -- read exactly once, inside `paintBase`
  // (the same moment `sizeCanvas` reads `window.devicePixelRatio` to size `base` itself, so the
  // two can never disagree even though they are technically two reads of the same global: nothing
  // can mutate it between two synchronous statements in the same call). `paintFrame` reuses THIS
  // value for its own transform reset/restore around the blit rather than re-deriving it
  // independently on every frame -- a floor-only change never re-runs `paintBase`, so an
  // independent re-read there could disagree with what `base` is actually sized for if the
  // display's DPR changed with no resize in between (ResizeObserver fires on a CSS box-size
  // change, not a DPR-only one -- ScreenMap does not special-case that, matching every other
  // widget in this codebase). Reusing the stored value keeps the selected-prefix overlay at the
  // SAME scale as the just-blitted base layer regardless, rather than silently drifting.
  let dpr = 1;
  return {
    paintBase(bundle, view, e, size) {
      dpr = window.devicePixelRatio || 1;
      sizeCanvas(base, size); // matches the DESTINATION's own backing-store size exactly, so
                              // `paintFrame`'s blit is a 1:1 device-pixel copy, never a rescale
      baseCtx.clearRect(0, 0, size.width, size.height);
      const screen = screenRingsOf(bundle, view);
      const informal = bundle.informal;
      for (let i = 0; i < screen.length; i++) {
        const rings = screen[i]!;
        // Fill and outline in the SAME colour, so the outline can never obliterate the fill it
        // surrounds -- see this function's own docstring for why that is not merely cosmetic at
        // this bundle's block sizes.
        const style = informal?.[i] ? e.informal_color : e.base_color;
        fillBlock(baseCtx, rings, style);
        strokeBlock(baseCtx, rings, style, e.block_lw);
      }
    },
    paintFrame(ctx, bundle, view, e, order, k, size) {
      ctx.clearRect(0, 0, size.width, size.height);
      // The blit happens in DEVICE-PIXEL space: `base` and `ctx`'s own canvas were sized to the
      // same backing-store dimensions (this module's own `dpr`, above), so copying them 1:1 needs
      // the identity transform, not the CSS-pixel-space one `sizeCanvas` leaves active -- drawing
      // `base` under THAT transform would scale its already-DPR-sized pixels up by another factor
      // of `dpr`, blitting only the top-left 1/dpr² of the base layer at the wrong scale.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(base, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // back to CSS-pixel space for the stroke below
      const screen = screenRingsOf(bundle, view);
      for (let i = 0; i < k; i++) {
        strokeBlock(ctx, screen[order[i]!]!, e.selected_color, e.block_lw * 2);
      }
      // Drawn last, so wherever it meets a selected block's outline it sits over that outline
      // rather than under it -- which is the first of the two reasons it is on the FRAME and not
      // the base layer. The second: `paintBase` is the strictly per-block pass, and a marker
      // belonging to no block has no place in a loop whose stroke count is exactly `n_blocks`
      // (`screen-map-boot.test.ts`'s base-layer test counts it). Note which reason is NOT in that
      // list: a ring painted into the base layer would still be VISIBLE after a floor change,
      // because the base layer is copied back whole on every frame. What it would not be is drawn
      // by this frame.
      //
      // `follow` is absent for Nairobi (screen_map.d.ts), so this branch is a real optional, not a
      // guard: a bundle that carries no followed block draws no ring.
      const follow = bundle.follow;
      if (follow !== undefined) {
        const [fx, fy] = toScreen(view, follow.x, follow.y);
        ctx.beginPath();
        ctx.arc(fx, fy, FOLLOW_RADIUS_PX, 0, Math.PI * 2);
        ctx.strokeStyle = e.follow_color;
        ctx.lineWidth = FOLLOW_LW_PX;
        ctx.stroke();
      }
    },
  };
}
