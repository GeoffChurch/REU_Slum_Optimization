import type { HoodBlock, HoodBundle } from "../hood.js";
// Type-only: erased at compile time, so this module has NO runtime import of mount.js. A runtime
// one would recreate the import cycle that once made the whole bundle throw during module
// evaluation while the page still looked fine (see mount.ts's registration comment). This file
// must never import `register`.
import type { Widget } from "../mount.js";
import type { StateFactory, StateSource } from "../state.js";
import { intParam, stringParam, type UrlCodec } from "../url/param.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { growth } from "../model/accretion.js";
import { sizeCanvas } from "../render/canvas.js";
import { blockAt, draw } from "../render/region.js";
import { fitBbox, toWorld, type Bbox, type View } from "../view/transform.js";

/** `seed` is a **block_id**, `budget` a building_count on the slider's own scale. `seed` is an
 * identity rather than a position deliberately: it is citable in a URL (piece E), and an array
 * index there would point at a DIFFERENT block after any re-bake that reorders `hood.json` -- no
 * error, right type, right shape, wrong value. Keeping both here rather than as loose widget
 * variables is what makes a click-then-slide sequence replay through the SAME `growth()` call every
 * render, instead of the picture and the model drifting apart across two pieces of mutable state. */
export interface RegionGrowState { seed: string; budget: number }

export const REGION_GROW_URL: UrlCodec<RegionGrowState> = {
  seed: stringParam("seed"),
  budget: intParam("budget"),
};

/** The name every failure of this widget is reported under -- one constant, because it is used
 * from two unrelated places (the fetch chain and the resize callback) and two spellings of it
 * would read to a reader as two different widgets failing. */
const LABEL = "RegionGrow";

/** The bbox every shipped block's rings must fit inside -- exterior AND interior rings, since a
 * hole is still geometry the block occupies on screen even though nothing is drawn inside it. */
function hoodBbox(blocks: HoodBlock[]): Bbox {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const b of blocks) for (const ring of b.rings) for (const [x, y] of ring) {
    xs.push(x);
    ys.push(y);
  }
  return { minX: Math.min(...xs), minY: Math.min(...ys),
           maxX: Math.max(...xs), maxY: Math.max(...ys) };
}

/** Blocks adjacent to `region` and not already in it -- the same frontier `growth`'s own loop
 * computes at every step, recomputed once here at the FINAL region so the picture can show it:
 * it is what makes "greedy" visible as next-candidates, rather than merely asserted by prose. */
function frontierOf(blocks: HoodBlock[], region: number[]): number[] {
  const inRegion = new Set(region);
  const frontier = new Set<number>();
  for (const i of region) for (const j of blocks[i]!.adj) if (!inRegion.has(j)) frontier.add(j);
  return [...frontier];
}

export const regionGrow: Widget<RegionGrowState> = (host, makeState) => {
  // Not `host.dataset.bundle!`: a missing attribute then reaches `fetch(undefined)` and surfaces
  // as "fetch undefined failed: 404", which sends the reader (and whoever wrote the page) looking
  // for a missing FILE rather than the missing ATTRIBUTE that is actually wrong.
  const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
  // A 404, a renamed bundle field, or any throw inside boot() must be VISIBLE on the page, not an
  // unhandled rejection sitting in the console while the PNG fallback keeps the page looking fine
  // -- verbatim the defect class this branch has found repeatedly.
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<HoodBundle>;
    })
    .then((b) => boot(host, makeState, b))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

function boot(host: HTMLElement, makeState: StateFactory<RegionGrowState>, b: HoodBundle): void {
  const e = b.encoding;
  const blocks = b.blocks;
  // block_id -> index, built once. `growth()` and `draw()` both take a POSITION (and `hood.json`'s
  // `reference` fixtures pin accretion by index), so the conversion lives at the two boundaries
  // rather than in the model.
  const indexOf = new Map(blocks.map((blk, i) => [blk.block_id, i]));
  // The bundle is a BOUNDARY: it arrives over the network, and a page can outlive the artifact it
  // was generated beside. A `seed` that no longer names one of `blocks` would otherwise reach
  // `growth()` as a negative index and fail far from here, with no message a reader could act on.
  if (!indexOf.has(b.seed)) {
    throw new Error(`hood.json's seed "${b.seed}" is not among its own ${blocks.length} blocks`);
  }
  const state: StateSource<RegionGrowState> = makeState({
    seed: b.seed, budget: b.budget.default,
  });
  // A URL (piece E) may name a block this hood does not carry. That is a reader's typo, not a
  // broken artifact, so reset rather than throw -- and because the reset makes the field equal its
  // initial, the store stops emitting `?seed=` and the URL self-corrects (design §2.3).
  if (!indexOf.has(state.get().seed)) state.set({ seed: b.seed });

  const caption = host.querySelector("figcaption");

  const cv = document.createElement("canvas");
  // Inline styles, never presentation attributes: Material's `.md-typeset svg{height:auto;
  // max-width:100%}` beats a presentation attribute, which cost D1 a Critical at its final gate.
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";
  // No drag here (the click is a single reseed, not a gesture), but this still keeps a tap from
  // being swallowed by the browser's own double-tap-to-zoom gesture recognizer on touch.
  cv.style.touchAction = "none";

  const controls = document.createElement("div");

  const budgetLabel = document.createElement("label");
  const slider = document.createElement("input");
  slider.type = "range";                      // native: keyboard- and screen-reader-reachable
  slider.min = String(b.budget.min);
  slider.max = String(b.budget.max);
  slider.step = String(b.budget.step);
  // From STATE, not from `b.budget.default`: the bundle's default is what `makeState` was seeded
  // with, but a URL (piece E) may have overridden it before this line, and a slider showing 3000
  // beside a region grown to 5000 is the exact desync this widget's own state exists to prevent.
  slider.value = String(state.get().budget);
  slider.addEventListener("input", () => state.set({ budget: Number(slider.value) }));
  budgetLabel.append("Budget ", slider);

  const readout = document.createElement("p");
  // The one line that changes on every frame -- reseed, slider drag, both -- announced. A canvas
  // carries no accessible text at all, so without this a screen-reader user moving the slider
  // hears the budget they set and NOTHING about the region it grew. `polite` rather than
  // `assertive`: it must not interrupt, and a drag produces a frame per pointer move, which an
  // assertive region would announce over itself continuously.
  readout.setAttribute("aria-live", "polite");
  controls.append(budgetLabel, readout);

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

  const bbox = hoodBbox(blocks);
  // Both assigned by the observer below and by nothing else. Everything that reads them --
  // `render`, the pointer handler -- is wired from inside the first sized callback, so there is
  // no ordering in which they are read before that callback has run.
  let size: { width: number; height: number };
  let view: View;
  const ctx = cv.getContext("2d")!;

  const render = (): void => {
    const s = state.get();
    // `!` because every value `state.seed` can hold is one of `blocks`: `boot` above resets a
    // non-member (a URL's) to `b.seed` before this ever runs, and the `pointerdown` handler below
    // only ever writes the block_id of a block `blockAt` has just found.
    const seed = indexOf.get(s.seed)!;
    const g = growth(blocks, seed, s.budget);
    const frontier = frontierOf(blocks, g.order);
    draw(ctx, blocks, e, { view, region: g.order, frontier, seed }, size);
    // Every number the picture shows is also present as text, and both come from the same `g` the
    // picture was drawn from -- there is no second call to `growth()` that could disagree with it.
    readout.textContent =
      `${g.order.length} blocks in the region · ${g.buildings} buildings`
      + (g.stoppedAtEdge ? ` · Growth reached the edge of the loaded neighbourhood.` : "");
  };

  const wireInteraction = (): void => {
    cv.addEventListener("pointerdown", (ev) => {
      const [wx, wy] = toWorld(view, ev.offsetX, ev.offsetY);
      const hit = blockAt(blocks, wx, wy);
      // A click that lands in no block (outside every ring) leaves the seed alone, rather than
      // clearing it to something the picture cannot draw.
      if (hit === -1) return;
      state.set({ seed: blocks[hit]!.block_id });
    });
  };

  // The observed element is the CANVAS ITSELF: `width: 100%` makes its content box track the
  // container's width, so observing it answers this widget's actual question ("how many CSS
  // pixels am I drawing into?") in one hop, and it is the box `sizeCanvas` scales the backing
  // store to.
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
