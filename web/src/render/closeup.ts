/** ScreenMap's optional close-up: the followed block and its neighbours, drawn at a scale where
 * the block is a shape rather than a ring.
 *
 * The city canvas cannot show the followed block itself -- `city.ts`'s own docstring measures the
 * median block at well under a CSS pixel there, which is why `follow` is a fixed-size RING on that
 * canvas and never an outline. The Demo page is about that one block, so it asks for a second
 * canvas framed on it (`data-closeup-pad-m`, the margin around the block in metres) in the SAME
 * encoding as the city map: every block filled `base_color`, or `informal_color` where the City's
 * survey calls it informal, the current selection OUTLINED in `selected_color` (never filled, for
 * `city.ts`'s reason: a fill would hide the gold), and the followed block outlined in
 * `follow_color`. So what the close-up says about the screen is what the city map says, at a scale
 * where it can be read.
 *
 * A few hundred blocks at most fall in the window, so this redraws everything on every frame --
 * `region.ts`'s reasoning at its 213 blocks, not `city.ts`'s offscreen blit at 16,000.
 */
import type { CityBundle } from "../screen_map.js";
import { sizeCanvas } from "./canvas.js";
import { fitBbox, toScreen, type Bbox, type View } from "../view/transform.js";

/** The separator between neighbouring blocks, in CSS pixels. At the close-up's scale two blocks
 * filled the same grey would otherwise read as one; the city canvas needs none because there no
 * block is big enough for its edge to be seen. */
const SEPARATOR_LW_PX = 1;
const SEPARATOR_COLOR = "#ffffff";
/** The selection outline and the followed block's, in CSS pixels: thicker than the city map's,
 * because here they are the shapes the reader is looking at. */
const SELECTED_LW_PX = 1.5;
const FOLLOW_LW_PX = 3;

/** The window: the followed block's exterior bbox, squared about its centre and grown by `padM`
 * on every side -- the prototype's own framing (build.py's `screen_bake`), so the close-up is the
 * shape the owner approved. */
export function closeupBox(bundle: CityBundle, index: number, padM: number): Bbox {
  const ring = bundle.rings[index]![0]!;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of ring) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const half = Math.max(maxX - minX, maxY - minY) / 2 + padM;
  return { minX: cx - half, minY: cy - half, maxX: cx + half, maxY: cy + half };
}

/** Every block whose exterior ring's bbox meets `box`, in bundle order (a bbox test, so a block
 * that only grazes a corner is drawn and clipped by the canvas edge, as a map would). */
export function blocksIn(bundle: CityBundle, box: Bbox): number[] {
  const out: number[] = [];
  bundle.rings.forEach((rings, i) => {
    const ring = rings[0]!;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of ring) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (maxX >= box.minX && minX <= box.maxX && maxY >= box.minY && minY <= box.maxY) out.push(i);
  });
  return out;
}

function trace(ctx: CanvasRenderingContext2D, view: View, rings: [number, number][][]): void {
  ctx.beginPath();
  for (const ring of rings) {
    ring.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
  }
}

export interface Closeup {
  /** The close-up's own canvas, for the widget to place beside the city map. */
  readonly canvas: HTMLCanvasElement;
  /** The canvas's measured CSS size; call from its own ResizeObserver. Repaints the last frame. */
  resize(size: { width: number; height: number }): void;
  /** Draw `bundle`'s followed block's window with `order[0..k)` selected. A bundle with no
   * `follow` (Nairobi) has nothing to frame, so the canvas is HIDDEN rather than left showing the
   * previous city's block. Before the first `resize` nothing is drawn; `resize` draws it. */
  paint(bundle: CityBundle, order: Int32Array, k: number): void;
}

export function createCloseup(padM: number): Closeup {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.aspectRatio = "1 / 1";
  const ctx = canvas.getContext("2d")!;
  let size: { width: number; height: number } | null = null;
  let last: { bundle: CityBundle; order: Int32Array; k: number } | null = null;
  // The window and the blocks in it, per bundle (a city switch is a different bundle object).
  let framed: { bundle: CityBundle; box: Bbox; near: number[] } | null = null;

  const draw = (): void => {
    if (size === null || last === null) return;
    const { bundle, order, k } = last;
    const follow = bundle.follow;
    if (follow === undefined) {
      canvas.style.display = "none";
      return;
    }
    canvas.style.display = "";
    if (framed === null || framed.bundle !== bundle) {
      const box = closeupBox(bundle, follow.index, padM);
      framed = { bundle, box, near: blocksIn(bundle, box) };
    }
    const e = bundle.encoding;
    const view = fitBbox(framed.box, size.width, size.height, 0);
    sizeCanvas(canvas, size);
    ctx.clearRect(0, 0, size.width, size.height);
    const selected = new Set<number>();
    for (let i = 0; i < k; i++) selected.add(order[i]!);
    const informal = bundle.informal;
    for (const i of framed.near) {
      trace(ctx, view, bundle.rings[i]!);
      ctx.fillStyle = informal?.[i] ? e.informal_color : e.base_color;
      ctx.fill("evenodd");
      ctx.strokeStyle = SEPARATOR_COLOR;
      ctx.lineWidth = SEPARATOR_LW_PX;
      ctx.stroke();
    }
    for (const i of framed.near) {
      if (!selected.has(i)) continue;
      trace(ctx, view, bundle.rings[i]!);
      ctx.strokeStyle = e.selected_color;
      ctx.lineWidth = SELECTED_LW_PX;
      ctx.stroke();
    }
    // Last, so it sits over a selected neighbour's outline where the two share an edge.
    trace(ctx, view, bundle.rings[follow.index]!);
    ctx.strokeStyle = e.follow_color;
    ctx.lineWidth = FOLLOW_LW_PX;
    ctx.stroke();
  };

  return {
    canvas,
    resize(measured) {
      size = measured;
      draw();
    },
    paint(bundle, order, k) {
      last = { bundle, order, k };
      draw();
    },
  };
}
