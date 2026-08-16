import type { Bundle } from "../bundle.js";
import { toScreen, type View } from "../view/transform.js";

/** Resize the backing store for devicePixelRatio and return the CSS-pixel size to draw in. */
export function sizeCanvas(cv: HTMLCanvasElement): { width: number; height: number } {
  const dpr = window.devicePixelRatio || 1;
  const { width, height } = cv.getBoundingClientRect();
  cv.width = Math.round(width * dpr);
  cv.height = Math.round(height * dpr);
  const ctx = cv.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

function rampColor(ramp: string[], t: number): string {
  const i = Math.min(ramp.length - 1, Math.max(0, Math.round(t * (ramp.length - 1))));
  return ramp[i]!;
}

export interface Frame { view: View; prefix: number; layer: "conductance" | "current"; halos: boolean }

export function draw(ctx: CanvasRenderingContext2D, b: Bundle, f: Frame,
                     size: { width: number; height: number }): void {
  const e = b.encoding;
  ctx.clearRect(0, 0, size.width, size.height);

  // Parcels as a pale wireframe, never filled: filling them by potential would state the same
  // quantity twice in two shapes and drown the graph (piece B's finding).
  ctx.strokeStyle = e.parcel_color;
  ctx.lineWidth = 0.4;
  for (const ring of b.parcels) {
    ctx.beginPath();
    ring.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(f.view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
    ctx.stroke();
  }

  // The road corridor, drawn once per width group as a translucent stroke of that group's
  // segments, all joined into ONE path before ONE stroke(). A drainage-ordered prefix is many
  // short segments meeting at junctions; each `stroke()` call is an independent compositing
  // operation, so a stroke PER ROAD compounds translucency toward opaque at every junction just
  // as surely as a fill would (this is the bug src/reblock/render.py:304-319 documents and avoids
  // by unioning roads per width group before a single draw -- mirrored here: one beginPath() per
  // width group covering every road in it, then one stroke()). Grouping by width_m also keeps a
  // group's lineWidth well-defined, since lineWidth is a single value per stroke() call.
  ctx.globalAlpha = 0.25;
  ctx.strokeStyle = e.road_color;
  ctx.lineCap = "round";
  const byWidth = new Map<number, typeof b.roads>();
  for (const r of b.roads.slice(0, f.prefix)) {
    const group = byWidth.get(r.width_m);
    if (group) group.push(r); else byWidth.set(r.width_m, [r]);
  }
  for (const [width_m, group] of byWidth) {
    ctx.lineWidth = width_m * f.view.scale;
    ctx.beginPath();
    for (const r of group) {
      r.coords.forEach(([x, y], i) => {
        const [sx, sy] = toScreen(f.view, x, y);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // The block outline and the existing street network, beneath the graph -- fallback parity with
  // src/reblock/render.py's _draw_boundary_and_streets, which every PNG (including
  // graph_current_after.png, the image this widget replaces) draws in the same layer position:
  // after the parcel wireframe and road corridor, before the mesh edges and nodes.
  ctx.strokeStyle = e.boundary_color;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  b.boundary.forEach(([x, y], i) => {
    const [sx, sy] = toScreen(f.view, x, y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  });
  ctx.stroke();
  for (const line of b.streets) {
    ctx.beginPath();
    line.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(f.view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  }

  // Edges. Width encodes the chosen quantity for MESH edges only; road-raised edges draw at the
  // fixed upgraded_lw, because their computed width would be a saturated non-measurement.
  const quantity = f.layer === "current" ? b.prefix.current[f.prefix]! : b.edges.footpath_g;
  const norm = e.width_norm[f.layer];
  const { rows, cols, first_upgraded_at } = b.edges;
  const isUpgraded = (k: number): boolean =>
    first_upgraded_at[k]! >= 0 && first_upgraded_at[k]! <= f.prefix;
  const strokeEdge = (k: number, strokeStyle: string, lineWidth: number): void => {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    const [x0, y0] = toScreen(f.view, b.nodes.cx[rows[k]!]!, b.nodes.cy[rows[k]!]!);
    const [x1, y1] = toScreen(f.view, b.nodes.cx[cols[k]!]!, b.nodes.cy[cols[k]!]!);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  };
  // Two passes -- every mesh (grey) edge, THEN every upgraded (blue) edge -- mirroring
  // render_graph's zorder=3/zorder=4 split (src/reblock/render.py): a single index-order pass
  // interleaves grey and blue, and a mesh edge can reach 3 px against an upgraded edge's fixed
  // 1.0 px, so blue could be partially hidden by grey drawn after it -- weakening the exact
  // corridor signal this widget exists to show (fix wave, minor). Splitting into two passes makes
  // blue undraws-over-able the same way the PNG's LineCollection z-order already guarantees.
  for (let k = 0; k < rows.length; k++) {
    if (isUpgraded(k)) continue;
    const frac = norm > 0 ? Math.min(1, Math.abs(quantity[k]!) / norm) : 0;
    strokeEdge(k, e.edge_color, e.edge_lw_min + frac * (e.edge_lw_max - e.edge_lw_min));
  }
  for (let k = 0; k < rows.length; k++) {
    if (isUpgraded(k)) strokeEdge(k, e.road_color, e.upgraded_lw);
  }

  // Nodes, coloured by potential on the ramp Python sampled. vmax is prefix 0's maximum: roads
  // only lower potentials, so this is the shared scale across every slider position.
  const pot = b.prefix.potential[f.prefix]!;
  const vmax = Math.max(...b.prefix.potential[0]!);
  const r = e.node_radius_frac * medianEdgeLength(b) * f.view.scale;
  for (let i = 0; i < pot.length; i++) {
    const [sx, sy] = toScreen(f.view, b.nodes.cx[i]!, b.nodes.cy[i]!);
    if (f.halos && b.nodes.ground_g[i]! > 0) {
      ctx.strokeStyle = e.boundary_color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 1.6, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = rampColor(e.ramp, vmax > 0 ? pot[i]! / vmax : 0);
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Matches `np.median` (src/reblock/render.py:347-349): the average of the two middle values on
 * an even-length array, not just the upper-middle one -- otherwise the widget's node radius
 * quietly diverges from the PNG's whenever the edge count happens to be even. */
function medianEdgeLength(b: Bundle): number {
  const ds = b.edges.rows.map((ri, k) => {
    const ci = b.edges.cols[k]!;
    return Math.hypot(b.nodes.cx[ri]! - b.nodes.cx[ci]!, b.nodes.cy[ri]! - b.nodes.cy[ci]!);
  }).sort((a, z) => a - z);
  if (ds.length === 0) return 1;
  const mid = ds.length / 2;
  return ds.length % 2 === 0 ? (ds[mid - 1]! + ds[mid]!) / 2 : ds[Math.floor(mid)]!;
}
