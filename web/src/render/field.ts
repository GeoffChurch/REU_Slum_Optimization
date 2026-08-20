import type { Encoding, FieldBundle, Road } from "../field.js";
import { toScreen, type View } from "../view/transform.js";

/** One draggable road endpoint: which road, which vertex, and where it is in WORLD metres.
 *
 * A named record rather than a bare `[number, number]` pair, because the widget's hit test has to
 * answer "which vertex did the reader grab" and a positional tuple makes that a `h[0]`/`h[1]` the
 * checker cannot audit (owner directive on closed sets). */
export interface Handle { road: number; vertex: number; x: number; y: number }

/** Every vertex of every ACTIVE road, in draw order.
 *
 * Exported and used by BOTH the drawing and the widget's pointer hit test, deliberately: two
 * derivations of "where the handles are" is how a handle comes to be drawn somewhere the reader
 * cannot grab it -- a failure that looks like nothing at all, since the picture is still correct.
 */
export function handles(roads: readonly Road[]): Handle[] {
  const out: Handle[] = [];
  roads.forEach((road, r) => {
    road.coords.forEach(([x, y], v) => out.push({ road: r, vertex: v, x, y }));
  });
  return out;
}

/** What one frame of the field draws: the fitted view, the roads that are switched on, and the
 * per-building contribution `cᵢ` that shades each disk. `c` is passed in rather than recomputed
 * here because the widget also SUMS it for the readout, and a picture shaded from one array while
 * the number beside it is summed from another is precisely the disagreement this widget exists to
 * make impossible. */
export interface FieldFrame { view: View; roads: readonly Road[]; c: Float64Array }

/** Screen-space radius of a world-space length. `scaleX` (rather than a mean of both scales) is
 * correct because `fitBbox` guarantees `scaleX === scaleY` for a map view -- the same read, for the
 * same stated reason, as `render/canvas.ts`'s corridor width. */
function px(view: View, metres: number): number {
  return metres * view.scaleX;
}

function polyline(ctx: CanvasRenderingContext2D, view: View,
                  points: readonly [number, number][]): void {
  points.forEach(([x, y], i) => {
    const [sx, sy] = toScreen(view, x, y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  });
}

function circle(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number): void {
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
}

/** The displacement model drawn literally, mirroring `render_field` (src/reblock/render.py:413)
 * layer for layer and in the same order:
 *
 *   1. parcels as a pale wireframe, never filled (filling would state one quantity twice);
 *   2. the road corridor, translucent, UNDER everything above it -- a corridor drawn over the disks
 *      would tint the shading it exists to be read against (`_draw_corridor`, zorder 2);
 *   3. the block boundary and the existing streets (`_draw_boundary_and_streets`);
 *   4. EVERY building as a disk of its own radius: the ones the corridor misses as a thin outline
 *      (zorder 5), the ones it grazes filled at `alpha = cᵢ` (zorder 6). Drawing the zero-cost disks
 *      too is the whole point -- without them a reader cannot see that a road threaded a GAP, only
 *      that some homes went red;
 *   5. the drag handles, LAST, so a handle is never buried under a disk it happens to sit on.
 *
 * Every colour, weight and alpha comes from `b.encoding` and none is written here. Task 3 found why
 * concretely: `street_lw` had been baked as 1.0 while `render.py` draws streets at 1.3, so a reader
 * with JS on would have seen thinner streets than the fallback PNG shows. The bundle now carries
 * 1.3, pinned to `render.py`'s `_BOUNDARY_LW` by a Python test -- a literal here reopens exactly
 * that divergence, silently, in the one direction no Python test can see.
 */
export function drawField(ctx: CanvasRenderingContext2D, b: FieldBundle, f: FieldFrame,
                          size: { width: number; height: number }): void {
  const e: Encoding = b.encoding;
  ctx.clearRect(0, 0, size.width, size.height);

  // (1) Parcels: the wireframe that shows a reader parcels are not buildings. Never filled.
  ctx.strokeStyle = e.parcel_color;
  ctx.lineWidth = e.parcel_lw;
  for (const ring of b.parcels) {
    ctx.beginPath();
    polyline(ctx, f.view, ring);
    ctx.closePath();
    ctx.stroke();
  }

  // (2) The corridor: ONE beginPath() per width group covering every road in it, then ONE stroke().
  // Each stroke() is an independent compositing operation, so a stroke PER ROAD compounds
  // translucency toward opaque wherever two roads overlap -- drawing the exact opposite of the
  // overlap-is-free claim this widget exists to demonstrate. This is `render.py:156-180`'s
  // dissolve-per-width-group rule in canvas terms, and it is the reason `lineWidth` is set per
  // group: lineWidth is a single value per stroke() call, so a group IS a width.
  ctx.globalAlpha = e.road_alpha;
  ctx.strokeStyle = e.road_color;
  ctx.lineCap = "round";
  const byWidth = new Map<number, Road[]>();
  for (const road of f.roads) {
    const group = byWidth.get(road.width_m);
    if (group) group.push(road); else byWidth.set(road.width_m, [road]);
  }
  for (const [width_m, group] of byWidth) {
    ctx.lineWidth = px(f.view, width_m);
    ctx.beginPath();
    for (const road of group) polyline(ctx, f.view, road.coords);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // (3) Boundary and streets, in one colour and two weights -- both read from the bundle, which is
  // where `render.py`'s own `_BOUNDARY_LW` was baked.
  ctx.strokeStyle = e.boundary_color;
  ctx.lineWidth = e.boundary_lw;
  ctx.beginPath();
  polyline(ctx, f.view, b.boundary);
  ctx.stroke();
  ctx.lineWidth = e.street_lw;
  for (const line of b.streets) {
    ctx.beginPath();
    polyline(ctx, f.view, line);
    ctx.stroke();
  }

  // (4) The disks. Two passes in `render_field`'s own zorder order: the untouched ones as outlines
  // first, the grazed ones filled at their own alpha on top.
  const { x, y, r } = b.buildings;
  ctx.strokeStyle = e.disk_color;
  ctx.lineWidth = e.disk_outline_lw;
  for (let i = 0; i < r.length; i++) {
    // Indexing is safe without a guard for the same reason `model/displacement.ts` states at
    // length: tests/test_displacement_field_bundle.py asserts x, y and r all have length
    // n_buildings at the artifact boundary, so a ragged bundle is unconstructible rather than
    // merely unlikely. `f.c` is produced from these same arrays by `corridorDistance`.
    if (f.c[i]! > 0) continue;
    const [sx, sy] = toScreen(f.view, x[i]!, y[i]!);
    circle(ctx, sx, sy, px(f.view, r[i]!));
    ctx.stroke();
  }
  ctx.fillStyle = e.disk_color;
  for (let i = 0; i < r.length; i++) {
    const ci = f.c[i]!;
    if (ci <= 0) continue;
    const [sx, sy] = toScreen(f.view, x[i]!, y[i]!);
    // Opacity IS the cost: a barely-grazed home is nearly transparent, a certainly-displaced one
    // solid. Same encoding as `render_field`'s per-disk facecolor alpha.
    ctx.globalAlpha = ci;
    circle(ctx, sx, sy, px(f.view, r[i]!));
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // (5) Handles last. A fixed SCREEN radius, not a world one: a grab target that shrank with the
  // zoom would stop being grabbable, and `handle_radius_px` is baked in pixels for that reason.
  for (const h of handles(f.roads)) {
    const [sx, sy] = toScreen(f.view, h.x, h.y);
    ctx.fillStyle = e.road_color;
    circle(ctx, sx, sy, e.handle_radius_px);
    ctx.fill();
    ctx.strokeStyle = e.boundary_color;
    ctx.lineWidth = e.boundary_lw;
    ctx.stroke();
  }
}
