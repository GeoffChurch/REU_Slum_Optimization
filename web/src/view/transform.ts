/** World (projected UTM metres) <-> screen, with pan/zoom and nearest-mark queries.
 *
 * Renderer-agnostic and DOM-free on purpose. The parent design called the substrate "a canvas
 * renderer", which conflates two layers: this one is shared by every widget, while what draws the
 * marks is free to differ -- piece D's Frontier chart wants SVG for its axis text, and ScreenMap's
 * 16k polygons want canvas. Keeping them apart is also what makes this file unit-testable.
 *
 * Parcels arrive already projected, so there is no reprojection anywhere: fit the bbox and draw.
 */
export interface Bbox { minX: number; minY: number; maxX: number; maxY: number }

/** screenX = x * scaleX + tx; screenY = ty - y * scaleY  (y flips: world up is screen up).
 *
 * Two scales, not one, because a chart and a map want different things: a map must not stretch
 * (metres are metres on both axes) while a chart's axes are different quantities entirely --
 * displacement 0-0.25 against permeability 0-1. `fitBbox` serves maps and keeps scaleX === scaleY;
 * `fitAxes` serves charts. Piece C's spec claimed one transform served both; it did not.
 */
export interface View { scaleX: number; scaleY: number; tx: number; ty: number }

export function fitBbox(b: Bbox, width: number, height: number, pad = 0.04): View {
  const w = Math.max(b.maxX - b.minX, 1e-9);
  const h = Math.max(b.maxY - b.minY, 1e-9);
  const scale = Math.min(width / w, height / h) * (1 - 2 * pad);
  return {
    scaleX: scale,
    scaleY: scale,
    tx: (width - w * scale) / 2 - b.minX * scale,
    ty: (height + h * scale) / 2 + b.minY * scale,
  };
}

/** Chart fit: each axis independently, so the plot fills the box. */
export function fitAxes(bx: [number, number], by: [number, number],
                        width: number, height: number, pad = 0.04): View {
  const w = Math.max(bx[1] - bx[0], 1e-9);
  const h = Math.max(by[1] - by[0], 1e-9);
  const scaleX = (width / w) * (1 - 2 * pad);
  const scaleY = (height / h) * (1 - 2 * pad);
  return {
    scaleX,
    scaleY,
    tx: (width - w * scaleX) / 2 - bx[0] * scaleX,
    ty: (height + h * scaleY) / 2 + by[0] * scaleY,
  };
}

export function toScreen(v: View, x: number, y: number): [number, number] {
  return [x * v.scaleX + v.tx, v.ty - y * v.scaleY];
}

export function toWorld(v: View, sx: number, sy: number): [number, number] {
  return [(sx - v.tx) / v.scaleX, (v.ty - sy) / v.scaleY];
}

export function panned(v: View, dxScreen: number, dyScreen: number): View {
  return { scaleX: v.scaleX, scaleY: v.scaleY, tx: v.tx + dxScreen, ty: v.ty + dyScreen };
}

/** Zoom about a screen anchor, keeping the world point under it fixed. */
export function zoomed(v: View, factor: number, sx: number, sy: number): View {
  return {
    scaleX: v.scaleX * factor,
    scaleY: v.scaleY * factor,
    tx: sx - (sx - v.tx) * factor,
    ty: sy - (sy - v.ty) * factor,
  };
}

/** Index of the nearest of `xs`/`ys` to a world point. Linear: 263 nodes needs no index. */
export function nearest(xs: number[], ys: number[], wx: number, wy: number): number {
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i]! - wx;
    const dy = ys[i]! - wy;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
