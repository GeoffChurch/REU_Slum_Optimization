/** The ScreenMap canvas: every block of a whole metro (Cape Town's bundle carries tens of
 * thousands, Nairobi's a few thousand), a base layer painted once onto an offscreen canvas, and the
 * selected prefix FILLED red over a copy of it.
 *
 * **The encoding: three fills.** Each block is red if the current floor selects it, whatever the
 * City's survey says; gold if it is not selected but the survey marks it informal; grey otherwise
 * (owner, 2026-10-29; `gen_screen_map.py`'s `_render_screen_map` draws the static figure the same
 * way). The base layer carries the last two -- every block in its unselected colour -- and each
 * frame fills the selected prefix in `e.selected_color` over it. Hiding the gold under a selected
 * block is the encoding, not a loss: the map answers "what does the screen pick", and gold is
 * what it leaves.
 *
 * **Why the selected fill is edged at twice the base outline's width.** Most of Cape Town's
 * blocks are smaller than a pixel (the median is ~0.6 CSS px² at a 700 px canvas, measured against
 * the committed bundle), so a block is nothing but antialiased edge: its colour reaches the screen
 * as partial coverage, composited over what is already there. Red painted over the block's own
 * base-layer gold at the SAME coverage would leave part of the gold in every such pixel, and a
 * selected informal block would come out orange beside a selected plain one's red -- the survey
 * leaking back into a colour that is meant to ignore it. Edging the red at `e.block_lw * 2`, twice
 * the base outline, covers the block's whole base-layer footprint with room to spare, so its
 * pixels read red (`web/test/city-raster.test.ts` rasterizes a tenth-of-a-pixel informal block at
 * a desktop's DPR and checks; at `e.block_lw` it comes out nearer the gold). The PNG's selected
 * linewidth is the same.
 *
 * **Why every frame starts from a blit, so a floor drag leaves no residue.** Painting the base
 * layer once onto the VISIBLE canvas and, on each later frame, restoring the previously selected
 * blocks by re-filling them in their base colour does not work at this geometry: a sub-pixel block
 * has no interior pixel a re-fill can cleanly overwrite, and source-over compositing red onto a
 * partially covered pixel and then the base colour back over it does not restore the original --
 * repeated select/deselect cycles converge toward a permanent pink residue (a coverage-α pixel
 * converges to (1−α)/(2−α) of the way to `selected_color`, never back to 0). A filled selection
 * would make that worse, not better. So red is only ever ADDED, never taken back: every prefix
 * starts from a pixel COPY (`drawImage`) of the untouched base layer, or of a saved copy of an
 * earlier prefix of the same ranking, and fills blocks over it in ranking order; each frame clears
 * the visible canvas and copies that prefix onto it. What a frame shows depends only on its own
 * floor, never on the frames before it (`city-raster.test.ts` drags the floor down and back and
 * compares every pixel, and compares a random walk of floors against a fresh layer's).
 *
 * **Why the offscreen canvas is a plain, uninserted `<canvas>`, not `OffscreenCanvas`.** Both give
 * the same 2D context and `drawImage` accepts either; a second `<canvas>` element needs no new
 * global and is what `web/test/harness.ts`'s fake `document.createElement` already knows how to
 * produce (its own `RecordingContext`, `drawImage`-able like any other).
 *
 * `region.ts`'s neighbourhood tops out at 213 blocks, so redrawing everything every frame there is
 * free. At city scale it is not: `paintBase` below is the ONE O(n_blocks) pass this module ever
 * makes for a given (bundle, view) pair, run once into the offscreen canvas. Nor is the selected
 * prefix repainted whole on every frame: it runs to nearly every block over most of the floor
 * track (on Cape Town the log track spends its left two-thirds above three-quarters of the
 * city), where a repaint per frame costs a full repaint -- 90 to 240 ms in headless Chromium at
 * DPR 1 to 2, measured 2026-11-25. `paintFrame` instead keeps the prefix on a second offscreen
 * canvas and only ever ADDS blocks to it, in `order`, which leaves it pixel for pixel what a
 * repaint from the base layer would draw (the same fills, in the same order, over the same copy);
 * a shrinking floor restarts from a saved copy at or below the new prefix (`PREFIX_CHECKPOINTS`).
 * So a frame paints the blocks the drag added since the last one, or at most
 * n_blocks / PREFIX_CHECKPOINTS of them on a shrink, then one blit and the ring.
 * `web/test/screen-map-boot.test.ts`'s base-layer test makes the claims checkable by following a
 * frame's own `drawImage` calls to the offscreen canvases they name and counting what THOSE
 * canvases' own logs hold, and `city-raster.test.ts` compares a random walk of floors against a
 * fresh layer pixel for pixel. */
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
 * batched path across many. */
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
  /** Every frame: bring the offscreen PREFIX layer to the base layer with `order[0..k)` FILLED
   * over it in `e.selected_color`, edged in it at `e.block_lw * 2` -- in the CSS-pixel space every
   * other draw call in this codebase uses, over whatever `paintBase` painted there, gold included
   * (this module's docstring) -- then clear `ctx` and blit that layer onto it (a single
   * `drawImage`, in device-pixel space so a DPR-scaled backing store is copied 1:1 rather than
   * re-scaled). The prefix layer is kept between calls: a larger `k` for the same `order` fills
   * only the blocks it adds, a smaller one restarts from the last saved state at or below it
   * (`PREFIX_CHECKPOINTS`), and a new `order` or a `paintBase` restarts from the base layer.
   * Finally, where the bundle carries one, `bundle.follow` gets a `FOLLOW_RADIUS_PX` ring in
   * `e.follow_color` on `ctx` itself, drawn last so it sits over those fills rather than under
   * them. */
  paintFrame(ctx: CanvasRenderingContext2D, bundle: CityBundle, view: View, e: CityEncoding,
            order: Int32Array, k: number, size: { width: number; height: number }): void;
}

/** How many saved prefix states a layer keeps between metric switches: `paintFrame` saves the
 * prefix layer each time it grows past another `n_blocks / PREFIX_CHECKPOINTS` blocks, and a
 * SHRINKING floor restarts from the last one at or below it instead of from the bare base layer,
 * so no frame repaints more than that many blocks plus the ones it adds. A constant, not a
 * setting: it trades memory for speed and cannot change a pixel (`city-raster.test.ts` checks
 * every frame of a random walk against a fresh layer's). Each one costs a backing store the size
 * of the visible canvas -- 1 MB for a 500 CSS px canvas at DPR 1, 4 MB at DPR 2. */
const PREFIX_CHECKPOINTS = 8;

/** Builds one OFFSCREEN `<canvas>` (never inserted into the document -- a pure pixel buffer) and
 * closes over it, so its contents belong to ONE widget instance. Deliberately not module state the
 * way `cache` above is: two ScreenMaps sharing one offscreen canvas would blit whichever one last
 * painted it into BOTH widgets' visible canvases. */
export function createLayer(): CityLayer {
  const base = document.createElement("canvas");
  const baseCtx = base.getContext("2d")!;
  // The PREFIX layer: the base layer with `order[0..preK)` filled over it, kept between frames so a
  // floor drag paints only the blocks it adds (module docstring). `preFor` is the ranking it was
  // painted for, compared by reference: `render` makes a fresh `order` on every metric or city
  // switch, and `paintBase` resets it to `null`, since a repainted base invalidates every prefix.
  const pre = document.createElement("canvas");
  const preCtx = pre.getContext("2d")!;
  let preFor: Int32Array | null = null;
  let preK = 0;
  // Saved prefix states: `saved[j]` holds what `pre` held at `preK === (j + 1) * step` for
  // `preFor`, for every j below `nSaved`. A pool, so a metric switch reuses the canvases rather
  // than allocating eight more. `pre` only ever grows one block at a time from a restore, so it
  // passes, and saves, every multiple of `step` at or below the highest `preK` it has reached --
  // which is why a shrink to `k` always finds `saved[floor(k / step) - 1]`.
  const saved: HTMLCanvasElement[] = [];
  let nSaved = 0;
  // The DPR `base`'s (and `pre`'s) backing store was actually sized for -- read exactly once,
  // inside `paintBase` (the same moment `sizeCanvas` reads `window.devicePixelRatio` to size
  // `base` itself, so the two can never disagree even though they are technically two reads of the same global: nothing
  // can mutate it between two synchronous statements in the same call). `paintFrame` reuses THIS
  // value for its own transform reset/restore around the blit rather than re-deriving it
  // independently on every frame -- a floor-only change never re-runs `paintBase`, so an
  // independent re-read there could disagree with what `base` is actually sized for if the
  // display's DPR changed with no resize in between (ResizeObserver fires on a CSS box-size
  // change, not a DPR-only one -- ScreenMap does not special-case that, matching every other
  // widget in this codebase). Reusing the stored value keeps the selected-prefix overlay at the
  // SAME scale as the just-blitted base layer regardless, rather than silently drifting.
  let dpr = 1;

  /** `src`'s pixels into `pre`, 1:1 in device-pixel space, over a cleared canvas: an exact copy. */
  const restore = (src: HTMLCanvasElement, size: { width: number; height: number }): void => {
    preCtx.clearRect(0, 0, size.width, size.height);
    preCtx.setTransform(1, 0, 0, 1, 0, 0);
    preCtx.drawImage(src, 0, 0);
    preCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  /** `pre`'s pixels into the next saved slot, an exact copy likewise. Sized off `pre` itself, never
   * a fresh `devicePixelRatio` read, for `dpr`'s own reason above. */
  const save = (): void => {
    let cp = saved[nSaved];
    if (cp === undefined) {
      cp = document.createElement("canvas");
      saved.push(cp);
    }
    if (cp.width !== pre.width || cp.height !== pre.height) {
      cp.width = pre.width;
      cp.height = pre.height;
    }
    const cpCtx = cp.getContext("2d")!;
    cpCtx.setTransform(1, 0, 0, 1, 0, 0);
    cpCtx.clearRect(0, 0, cp.width, cp.height);
    cpCtx.drawImage(pre, 0, 0);
    nSaved++;
  };

  return {
    paintBase(bundle, view, e, size) {
      dpr = window.devicePixelRatio || 1;
      sizeCanvas(base, size); // matches the DESTINATION's own backing-store size exactly, so
                              // `paintFrame`'s blit is a 1:1 device-pixel copy, never a rescale
      sizeCanvas(pre, size);
      preFor = null;
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
      const screen = screenRingsOf(bundle, view);
      const step = Math.ceil(order.length / PREFIX_CHECKPOINTS);
      if (preFor !== order) {
        restore(base, size);
        preFor = order;
        preK = 0;
        nSaved = 0;
      } else if (k < preK) {
        // A shrink: back to the last saved state at or below `k` (the bare base layer below the
        // first), and grow from there.
        const j = Math.floor(k / step);
        restore(j === 0 ? base : saved[j - 1]!, size);
        preK = j * step;
      }
      while (preK < k) {
        const rings = screen[order[preK]!]!;
        fillBlock(preCtx, rings, e.selected_color);
        strokeBlock(preCtx, rings, e.selected_color, e.block_lw * 2);
        preK++;
        if (preK % step === 0 && preK / step > nSaved) save();
      }
      ctx.clearRect(0, 0, size.width, size.height);
      // The blit happens in DEVICE-PIXEL space: `pre` and `ctx`'s own canvas were sized to the
      // same backing-store dimensions (this module's own `dpr`, above), so copying them 1:1 needs
      // the identity transform, not the CSS-pixel-space one `sizeCanvas` leaves active -- drawing
      // `pre` under THAT transform would scale its already-DPR-sized pixels up by another factor
      // of `dpr`, blitting only the top-left 1/dpr² of it at the wrong scale.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(pre, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // back to CSS-pixel space for the ring below
      // Drawn on the VISIBLE canvas, never into `pre`: `pre` is kept and grown between frames, so
      // a ring painted there would sit under every block a later frame adds. It also stays off the
      // base layer, `paintBase` being the strictly per-block pass (`screen-map-boot.test.ts`'s
      // base-layer test counts it).
      //
      // `follow` is absent for Nairobi (screen_map.d.ts), so this branch is a real optional, not a
      // guard: a bundle that carries no followed block draws no ring.
      drawFollow(ctx, bundle, view, e, false);
    },
  };
}

/** `bundle.follow`'s ring in `e.follow_color`, over whatever the frame has drawn; `halo` puts a
 * white ring under it first, for the score view's dark ground and blue blocks, where the blue
 * alone would sink in. Absent for Nairobi (screen_map.d.ts), so a bundle that carries no followed
 * block draws no ring. */
function drawFollow(ctx: CanvasRenderingContext2D, bundle: CityBundle, view: View, e: CityEncoding,
                    halo: boolean): void {
  const follow = bundle.follow;
  if (follow === undefined) return;
  const [fx, fy] = toScreen(view, follow.x, follow.y);
  const ring = (style: string, lineWidth: number): void => {
    ctx.beginPath();
    ctx.arc(fx, fy, FOLLOW_RADIUS_PX, 0, Math.PI * 2);
    ctx.strokeStyle = style;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  };
  if (halo) ring(FOLLOW_HALO, FOLLOW_LW_PX * 2);
  ring(e.follow_color, FOLLOW_LW_PX);
}

/** The score view's ramp: blue below the floor, pale at it, red above (ColorBrewer's RdBu, seven
 * classes, read as a continuous ramp), and the ground it is drawn on -- dark in BOTH colour
 * schemes, since the pale colours at the floor vanish on a light page. The widget's own, not the
 * bundle's: the static figure and gen_screen_map.py draw only the selection view. */
export const SCORE_RAMP = [
  "#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b",
] as const;
export const SCORE_GROUND = "#2b2e33";
const FOLLOW_HALO = "#ffffff";

const RAMP_RGB = SCORE_RAMP.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));

/** The colour at `u` (0 to 1) along SCORE_RAMP, linear in sRGB between its stops. */
export function rampColour(u: number): string {
  const t = Math.max(0, Math.min(1, u)) * (RAMP_RGB.length - 1);
  const i = Math.min(RAMP_RGB.length - 2, Math.floor(t));
  const f = t - i, a = RAMP_RGB[i]!, b = RAMP_RGB[i + 1]!;
  return `rgb(${a.map((v, j) => Math.round(v + (b[j]! - v) * f)).join(",")})`;
}

/** The score view's frame: the ground, then EVERY block in its own shade (`shades`, model/screen.ts
 * `rankShades`), filled and outlined in it at `e.block_lw` as the base layer's blocks are (a
 * sub-pixel block is nothing but its outline), then the follow ring with its halo. An O(n_blocks)
 * pass per frame, unlike the selection view's blit: a floor move changes every block's rank on
 * its side, so there is no unchanged layer to keep. Drawn straight onto the visible canvas, never
 * into the selection view's offscreen base layer, which stays valid for a switch back. */
export function paintScoreFrame(ctx: CanvasRenderingContext2D, bundle: CityBundle, view: View,
                                e: CityEncoding, shades: Float64Array,
                                size: { width: number; height: number }): void {
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = SCORE_GROUND;
  ctx.fillRect(0, 0, size.width, size.height);
  const screen = screenRingsOf(bundle, view);
  for (let i = 0; i < screen.length; i++) {
    const style = rampColour(shades[i]!);
    fillBlock(ctx, screen[i]!, style);
    strokeBlock(ctx, screen[i]!, style, e.block_lw);
  }
  drawFollow(ctx, bundle, view, e, true);
}
