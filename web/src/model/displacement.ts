import type { Road } from "../field.js";

/** One road segment as the metric needs it: two endpoints and that road's own half-width. */
export interface Segment { x0: number; y0: number; x1: number; y1: number; hw: number }

/** Flatten roads to segments. Mirrors `scripts/_default_road.segments`, so a parity failure is a
 * failure of the FORMULA and never of two different flattenings. */
export function flatten(roads: readonly Road[]): Segment[] {
  const out: Segment[] = [];
  for (const road of roads) {
    const hw = road.width_m / 2;
    for (let i = 1; i < road.coords.length; i++) {
      const [x0, y0] = road.coords[i - 1]!;
      const [x1, y1] = road.coords[i]!;
      out.push({ x0, y0, x1, y1, hw });
    }
  }
  return out;
}

/** Per-building distance to the road corridor, without ever constructing the corridor.
 *
 *     dist(p, U_i buffer(L_i, w_i/2)) == min_i max(0, dist(p, L_i) - w_i/2)
 *
 * A buffer IS the set of points within w/2 of the line, and distance to a union is the minimum over
 * its parts -- so this is exact, and it is what lets this widget compute the project's real metric
 * on an arbitrary road position with no Pyodide and no geometry library. The reference
 * implementation is `scripts/_default_road.closed_form_distance`, and
 * `tests/test_displacement_closed_form.py` pins it against `budget.displacement` for all eight
 * methods.
 *
 * With no segments every distance is Infinity, which `sumC` turns into zero cost -- the same answer
 * `budget.displacement` gives for an empty road set.
 */
export function corridorDistance(px: readonly number[], py: readonly number[],
                                 segs: readonly Segment[]): Float64Array {
  const out = new Float64Array(px.length).fill(Infinity);
  for (let i = 0; i < px.length; i++) {
    const x = px[i]!, y = py[i]!;
    let best = Infinity;
    for (const s of segs) {
      const dx = s.x1 - s.x0, dy = s.y1 - s.y0;
      const l2 = dx * dx + dy * dy;
      // A zero-length road is its own endpoint. Without this, t is 0/0 and every distance is NaN --
      // and NaN propagates silently through Math.min to a readout of "NaN homes".
      const t = l2 > 0 ? Math.min(1, Math.max(0, ((x - s.x0) * dx + (y - s.y0) * dy) / l2)) : 0;
      const d = Math.hypot(x - (s.x0 + t * dx), y - (s.y0 + t * dy)) - s.hw;
      if (d < best) best = d;
    }
    out[i] = Math.max(0, best);
  }
  return out;
}

/** `Σ clip(1 - d_i/r_i, 0, 1)`. Mirrors `budget.displacement_from_distance`, including its r == 0
 * case: a coincident-points building counts iff the corridor actually touches it. */
export function sumC(radii: readonly number[], d: Float64Array): number {
  let total = 0;
  for (let i = 0; i < radii.length; i++) {
    const r = radii[i]!, di = d[i]!;
    const c = r > 0 ? 1 - di / r : (di <= 0 ? 1 : 0);
    total += Math.min(1, Math.max(0, c));
  }
  return total;
}
