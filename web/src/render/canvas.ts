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

  // The road corridor, drawn once as a translucent stroke of the prefix's segments at their own
  // width. Stroked rather than buffered+filled because overlapping translucent fills compound
  // toward opaque -- exactly the bug that made piece B's corridor unreadable.
  ctx.globalAlpha = 0.25;
  ctx.strokeStyle = e.road_color;
  ctx.lineCap = "round";
  for (const r of b.roads.slice(0, f.prefix)) {
    ctx.lineWidth = r.width_m * f.view.scale;
    ctx.beginPath();
    r.coords.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(f.view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Edges. Width encodes the chosen quantity for MESH edges only; road-raised edges draw at the
  // fixed upgraded_lw, because their computed width would be a saturated non-measurement.
  const quantity = f.layer === "current" ? b.prefix.current[f.prefix]! : b.edges.footpath_g;
  const norm = e.width_norm[f.layer];
  const { rows, cols, first_upgraded_at } = b.edges;
  for (let k = 0; k < rows.length; k++) {
    const up = first_upgraded_at[k]! >= 0 && first_upgraded_at[k]! <= f.prefix;
    const frac = Math.min(1, Math.abs(quantity[k]!) / norm);
    ctx.strokeStyle = up ? e.road_color : e.edge_color;
    ctx.lineWidth = up ? e.upgraded_lw : e.edge_lw_min + frac * (e.edge_lw_max - e.edge_lw_min);
    const [x0, y0] = toScreen(f.view, b.nodes.cx[rows[k]!]!, b.nodes.cy[rows[k]!]!);
    const [x1, y1] = toScreen(f.view, b.nodes.cx[cols[k]!]!, b.nodes.cy[cols[k]!]!);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
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

function medianEdgeLength(b: Bundle): number {
  const ds = b.edges.rows.map((ri, k) => {
    const ci = b.edges.cols[k]!;
    return Math.hypot(b.nodes.cx[ri]! - b.nodes.cx[ci]!, b.nodes.cy[ri]! - b.nodes.cy[ci]!);
  }).sort((a, z) => a - z);
  return ds[Math.floor(ds.length / 2)] ?? 1;
}
