/** The RegionGrow canvas: neighbourhood, region, frontier, seed -- in that order.
 *
 * Recolouring must not touch geometry, so each block's screen-space path is rebuilt only when the
 * VIEW changes, never when the budget or the seed does. */
import type { HoodBlock, HoodEncoding } from "../hood.js";
import { toScreen, type View } from "../view/transform.js";

export interface RegionFrame {
  view: View;
  region: number[];
  frontier: number[];
  seed: number;
}

/** One block's rings (exterior first, then any interiors), projected to screen space. */
type ScreenRings = [number, number][][];

/** The last VIEW `draw` was called with, and every block's rings already projected for it.
 *
 * Keyed on `blocks` as well as `view` -- by reference, not by content -- so a genuinely new
 * bundle invalidates it too, even though in practice one widget instance only ever calls `draw`
 * with the one `blocks` array it fetched. Module state rather than a closure because `draw`'s
 * signature is the fixed widget contract this file's own docstring describes: the budget and the
 * seed pass through `f.region`/`f.frontier`/`f.seed` on every call, while `f.view` is the same
 * object across a whole slider drag or a reseed and only becomes a new one on resize. */
let cache: { blocks: HoodBlock[]; view: View; screen: ScreenRings[] } | null = null;

function screenRingsOf(blocks: HoodBlock[], view: View): ScreenRings[] {
  if (cache !== null && cache.blocks === blocks && cache.view === view) return cache.screen;
  const screen = blocks.map((b) => b.rings.map((ring) => ring.map(([x, y]) => toScreen(view, x, y))));
  cache = { blocks, view, screen };
  return screen;
}

/** One beginPath() covering every ring of a block -- exterior, then interiors -- as its own
 * closed subpath, so a single stroke() outlines all of them and a single fill() (evenodd) cuts
 * the interiors out of the exterior rather than doubly filling them. */
function tracePath(ctx: CanvasRenderingContext2D, rings: ScreenRings): void {
  ctx.beginPath();
  for (const ring of rings) {
    ring.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.closePath();
  }
}

function strokeBlock(ctx: CanvasRenderingContext2D, rings: ScreenRings): void {
  tracePath(ctx, rings);
  ctx.stroke();
}

function fillBlock(ctx: CanvasRenderingContext2D, rings: ScreenRings): void {
  tracePath(ctx, rings);
  // Even-odd, per hood.d.ts's own note on `rings`: "Exterior ring first, then interiors." Filling
  // nonzero (the default) would paint an interior ring's area a second time instead of cutting it
  // out, since both rings wind the same way coming out of the same polygon simplification.
  ctx.fill("evenodd");
}

export function draw(ctx: CanvasRenderingContext2D, blocks: HoodBlock[], e: HoodEncoding,
                     f: RegionFrame, size: { width: number; height: number }): void {
  ctx.clearRect(0, 0, size.width, size.height);
  const screen = screenRingsOf(blocks, f.view);

  // (1) Every neighbourhood block, outlined only -- the pale context every later layer sits on.
  ctx.strokeStyle = e.hood_color;
  ctx.lineWidth = e.hood_lw;
  for (const rings of screen) strokeBlock(ctx, rings);

  // (2) The grown region, filled AND stroked -- matching hood.png's own `region.plot(...,
  // facecolor=region_color, edgecolor=region_color, linewidth=region_lw, alpha=region_alpha)`.
  // A fill-only region draws JS-off's eleven individually-outlined blocks as a single
  // indistinguishable blob: the accretion is the widget's whole teaching point, and the stroke is
  // what separates one accreted block from the next one drawn on top of it. One beginPath()/
  // fill()+stroke() PER block rather than one path for the whole region: `region-grow-boot.test.ts`
  // counts region blocks by counting fill() calls in `region_color`, and a single batched path
  // would collapse that count to one regardless of how many blocks are actually in the region.
  //
  // `globalAlpha` is reset to 1 immediately after, not merely for tidiness: it is context state
  // that outlives the call that set it (field.ts's corridor layer resets it for the same reason),
  // so leaving it at `region_alpha` would tint the frontier and seed outlines drawn after it in
  // THIS frame, and the hood outline drawn first in the NEXT one.
  ctx.fillStyle = e.region_color;
  ctx.strokeStyle = e.region_color;
  ctx.lineWidth = e.region_lw;
  ctx.globalAlpha = e.region_alpha;
  for (const i of f.region) { fillBlock(ctx, screen[i]!); strokeBlock(ctx, screen[i]!); }
  ctx.globalAlpha = 1;

  // (3) The frontier -- adjacent to the region, not yet in it -- outlined so growth's next
  // candidates are visible rather than merely asserted by the caption.
  ctx.strokeStyle = e.frontier_color;
  ctx.lineWidth = e.region_lw;
  for (const i of f.frontier) strokeBlock(ctx, screen[i]!);

  // (4) The seed, outlined last so it is never buried under the region fill it sits inside.
  ctx.strokeStyle = e.seed_color;
  ctx.lineWidth = e.region_lw;
  strokeBlock(ctx, screen[f.seed]!);
}

/** Even-odd point-in-ring (the Franklin PNPOLY test): crossing count of a rightward ray from
 * `(px, py)` through the ring's edges, taken mod 2. */
function ringContains(ring: readonly [number, number][], px: number, py: number): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]!;
    const [xj, yj] = ring[j]!;
    const crosses = (yi > py) !== (yj > py);
    if (crosses && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Shoelace area, unsigned -- only the magnitude is needed, to compare which of two containing
 * rings is the smaller one. */
function ringArea(ring: readonly [number, number][]): number {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]!;
    const [xj, yj] = ring[j]!;
    sum += xj * yi - xi * yj;
  }
  return Math.abs(sum) / 2;
}

/** Index of the block whose exterior ring contains `(wx, wy)`, or -1. Even-odd ray cast over the
 * exterior only: a point inside a HOLE belongs to whatever block fills that hole, and that block
 * is tested on its own ring.
 *
 * That makes more than one exterior ring a candidate whenever the point is in a hole: the block
 * whose hole it is also contains it, over its own (larger) exterior ring. Measured against every
 * one of hood.json's 213 blocks, exactly one such pair exists -- block 42 nests inside block 41's
 * hole -- so the tie cannot be dismissed as merely theoretical. It is broken by area: the smaller
 * ring is the one actually occupying the point, since a hole's occupant nests entirely inside the
 * ring that was cut around it. */
export function blockAt(blocks: HoodBlock[], wx: number, wy: number): number {
  let best = -1;
  let bestArea = Infinity;
  for (let i = 0; i < blocks.length; i++) {
    // Every block ships at least its exterior ring (hood.d.ts: "Exterior ring first, then
    // interiors"), so index 0 always exists.
    const ext = blocks[i]!.rings[0]!;
    if (!ringContains(ext, wx, wy)) continue;
    const area = ringArea(ext);
    if (area < bestArea) { bestArea = area; best = i; }
  }
  return best;
}
