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

/** screenX = x * scale + tx; screenY = ty - y * scale  (y flips: world up is screen up). */
export interface View { scale: number; tx: number; ty: number }

export function fitBbox(b: Bbox, width: number, height: number, pad = 0.04): View {
  const w = Math.max(b.maxX - b.minX, 1e-9);
  const h = Math.max(b.maxY - b.minY, 1e-9);
  const scale = Math.min(width / w, height / h) * (1 - 2 * pad);
  const tx = (width - w * scale) / 2 - b.minX * scale;
  const ty = (height + h * scale) / 2 + b.minY * scale;
  return { scale, tx, ty };
}

export function toScreen(v: View, x: number, y: number): [number, number] {
  return [x * v.scale + v.tx, v.ty - y * v.scale];
}

export function toWorld(v: View, sx: number, sy: number): [number, number] {
  return [(sx - v.tx) / v.scale, (v.ty - sy) / v.scale];
}

export function panned(v: View, dxScreen: number, dyScreen: number): View {
  return { scale: v.scale, tx: v.tx + dxScreen, ty: v.ty + dyScreen };
}

/** Zoom about a screen anchor, keeping the world point under it fixed. */
export function zoomed(v: View, factor: number, sx: number, sy: number): View {
  const scale = v.scale * factor;
  return { scale, tx: sx - (sx - v.tx) * factor, ty: sy - (sy - v.ty) * factor };
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
