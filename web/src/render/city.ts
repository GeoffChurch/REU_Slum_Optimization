/** The ScreenMap canvas: 16,451 Cape Town blocks (3,500 Nairobi), a pale base layer painted once
 * and a red selected-prefix layer repainted on every floor/metric change.
 *
 * `region.ts`'s neighbourhood tops out at 213 blocks, so redrawing everything every frame is free
 * there. At city scale it is not: `paintBase` below is the ONE O(n_blocks) pass this module ever
 * makes for a given (bundle, view) pair. Every later frame -- a floor drag, a metric switch --
 * goes through `paintSelection` alone, which touches only the blocks whose selected/unselected
 * status actually changed: it undoes exactly the PREVIOUS prefix (back to `base_color`) and fills
 * exactly the NEW one (`selected_color`), never the ~14,800 blocks outside both. That is what
 * keeps a floor-slider drag cheap at 1,655 selected blocks rather than 16,451 -- see
 * `web/test/screen-map-boot.test.ts`'s base-layer test, which makes the claim checkable by
 * counting `base_color` fills across two frames.
 *
 * A real `OffscreenCanvas`/`drawImage` pixel blit would do the same job in a browser, but this
 * project's test harness (`web/test/harness.ts`) records draw CALLS, not pixels, and its
 * `RecordingContext` deliberately has no `drawImage` -- so the "paint once, reuse" property has to
 * be achieved with the calls the harness can see, which is what `paintBase`/`paintSelection`'s
 * split does. */
import type { CityBundle, CityEncoding } from "../screen_map.js";
import { toScreen, type View } from "../view/transform.js";

/** One block's rings (exterior first, then any interiors), projected to screen space. */
type ScreenRings = [number, number][][];

/** The last VIEW this module projected for, and every block's rings already computed for it --
 * mirrors `render/region.ts`'s own `screenRingsOf` cache, one level up in scale. Keyed on `bundle`
 * by reference: a city switch is a different bundle object, which invalidates it exactly like a
 * resized `view` does. */
let cache: { bundle: CityBundle; view: View; screen: ScreenRings[] } | null = null;

function screenRingsOf(bundle: CityBundle, view: View): ScreenRings[] {
  if (cache !== null && cache.bundle === bundle && cache.view === view) return cache.screen;
  const screen = bundle.rings.map(
    (rings) => rings.map((ring) => ring.map(([x, y]) => toScreen(view, x, y))));
  cache = { bundle, view, screen };
  return screen;
}

/** One beginPath()/fill() PER block, matching `region.ts`'s own reasoning: the boot test counts
 * selected blocks by counting `fill()` calls in `selected_color`, and a single batched path across
 * many blocks would collapse that count to one regardless of how many blocks it covers. */
function fillBlock(ctx: CanvasRenderingContext2D, rings: ScreenRings, style: string): void {
  ctx.fillStyle = style;
  ctx.beginPath();
  for (const ring of rings) {
    ring.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.closePath();
  }
  // Even-odd, per screen_map.d.ts's own note on `rings`: "Exterior ring first, then interiors."
  ctx.fill("evenodd");
}

/** What the last `paintSelection` call painted, so the next one can undo exactly that prefix
 * rather than the whole base layer -- `null` right after `paintBase` (nothing selected on a fresh
 * base layer). Closed over by `createLayer` below, ONE PER WIDGET INSTANCE, deliberately not
 * module state the way `cache` above is: `cache` keyed on `bundle`/`view` degrades to a harmless
 * extra recompute if two ScreenMaps ever shared it, but two instances sharing this one would
 * corrupt each other's picture outright -- instance B's `paintSelection` would "undo" instance
 * A's prefix. `screenRingsOf`'s geometry cache stays module-level (region.ts's own precedent, and
 * genuinely harmless to share); only the paint HISTORY needs its own home per instance. */
export interface CityLayer {
  paintBase(ctx: CanvasRenderingContext2D, bundle: CityBundle, view: View, e: CityEncoding,
           size: { width: number; height: number }): void;
  paintSelection(ctx: CanvasRenderingContext2D, bundle: CityBundle, view: View, e: CityEncoding,
                order: Int32Array, k: number): void;
}

export function createLayer(): CityLayer {
  let lastPainted: { order: Int32Array; k: number } | null = null;
  return {
    // The one expensive pass: clear the canvas and fill every block in `e.base_color`. Call this
    // only when the view or the active bundle changes (a resize or a city switch) -- never for a
    // floor or metric change, which `paintSelection` alone handles.
    paintBase(ctx, bundle, view, e, size) {
      ctx.clearRect(0, 0, size.width, size.height);
      const screen = screenRingsOf(bundle, view);
      for (const rings of screen) fillBlock(ctx, rings, e.base_color);
      lastPainted = null;
    },
    // Undoes the previously painted prefix (back to `e.base_color`) and fills `order[0..k)` in
    // `e.selected_color`. Safe to call every frame: its cost is O(previous prefix + new prefix),
    // not O(n_blocks) -- `paintBase` is the only call that touches every block.
    paintSelection(ctx, bundle, view, e, order, k) {
      const screen = screenRingsOf(bundle, view);
      if (lastPainted !== null) {
        for (let i = 0; i < lastPainted.k; i++) {
          fillBlock(ctx, screen[lastPainted.order[i]!]!, e.base_color);
        }
      }
      for (let i = 0; i < k; i++) fillBlock(ctx, screen[order[i]!]!, e.selected_color);
      lastPainted = { order, k };
    },
  };
}
