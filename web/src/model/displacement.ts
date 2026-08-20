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
    // px[i]/py[i] cannot actually be undefined: every real caller passes
    // bundle.buildings.x/.y, and tests/test_displacement_field_bundle.py asserts both have
    // length n_buildings at the artifact boundary -- a length mismatch here is unconstructible,
    // not merely unlikely. Same reasoning web/src/view/transform.ts's nearest() already relies on
    // for its own xs[i]!/ys[i]!, left unguarded for the same reason. A runtime length check here
    // would be exactly the unreachable guard this project's directives forbid -- and worth writing
    // down, because the failure it would catch (an x/y mismatch reading as silent zero cost) is
    // the same silent-and-plausible shape this module keeps finding elsewhere.
    const x = px[i]!, y = py[i]!;
    let best = Infinity;
    for (const s of segs) {
      const dx = s.x1 - s.x0, dy = s.y1 - s.y0;
      const l2 = dx * dx + dy * dy;
      // A zero-length road is its own endpoint. Without this, t is 0/0 = NaN, d becomes NaN, and
      // "NaN < best" is always false -- so best never leaves its initial Infinity and the
      // degenerate road silently contributes ZERO cost, not a visible NaN. That is the more
      // dangerous failure: silent and plausible (a road reads as free) instead of loud and
      // obviously wrong, and it's exactly what this guard prevents.
      const t = l2 > 0 ? Math.min(1, Math.max(0, ((x - s.x0) * dx + (y - s.y0) * dy) / l2)) : 0;
      const d = Math.hypot(x - (s.x0 + t * dx), y - (s.y0 + t * dy)) - s.hw;
      if (d < best) best = d;
    }
    out[i] = Math.max(0, best);
  }
  return out;
}

/** Per-building `c_i = clip(1 - d_i/r_i, 0, 1)`. Mirrors `budget.displacement_contributions`.
 *
 * Split out from `sumC` for the same reason ruling R8 split it out in Python: the widget needs the
 * per-building values (each disk is shaded at `alpha = c_i`) AND their sum (the readout), and
 * writing the formula twice is how a picture comes to disagree with the number printed beside it.
 * One formula, two callers, and `sumC` below is now literally a sum of this. */
export function contributions(radii: readonly number[], d: Float64Array): Float64Array {
  const out = new Float64Array(radii.length);
  for (let i = 0; i < radii.length; i++) {
    const r = radii[i]!, di = d[i]!;
    const c = r > 0 ? 1 - di / r : (di <= 0 ? 1 : 0);
    // The upper bound here is unreachable: corridorDistance always returns d >= 0 (its own
    // Math.max(0, best) floor), so c = 1 - d/r <= 1 whenever r > 0, and c is exactly 1 or 0 when
    // r == 0 -- c can never exceed 1 through this pipeline. Same story in the Python reference:
    // corridor_distance is a shapely .distance(), also always >= 0, so
    // budget.displacement_contributions's own upper clip is equally dead there. Kept anyway --
    // this module's job is to mirror that formula line for line, and the mirror is worth more than
    // the dead branch costs. This comment exists so a future reader neither tests that branch nor
    // trusts it to defend against anything. The LOWER bound is the live one: d > r gives c < 0 (any
    // building outside its corridor-touch radius), and Math.max(0, ...) is what turns that into
    // zero cost rather than negative cost.
    out[i] = Math.min(1, Math.max(0, c));
  }
  return out;
}

/** `Σ clip(1 - d_i/r_i, 0, 1)`. Mirrors `budget.displacement_from_distance`, including its r == 0
 * case: a coincident-points building counts iff the corridor actually touches it. A sum over
 * `contributions` rather than a second copy of the formula -- the six baked fixtures pin THIS
 * function against `budget.displacement`, so the widget's readout and the widget's shading are
 * pinned by the same measurement. */
export function sumC(radii: readonly number[], d: Float64Array): number {
  let total = 0;
  for (const c of contributions(radii, d)) total += c;
  return total;
}
