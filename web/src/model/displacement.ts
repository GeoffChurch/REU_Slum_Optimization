import type { Road } from "../field.js";

/** One road segment as the metric needs it: two endpoints and that road's own half-width. */
export interface Segment { x0: number; y0: number; x1: number; y1: number; hw: number }

type Pt = readonly [number, number];
/** One polygon as the bundle carries it: closed rings, exterior first. */
type Polygon = readonly (readonly Pt[])[];

/** Flatten roads to segments, one per consecutive pair of vertices. */
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

/** Vertices per quarter circle -- shapely's `buffer` default (`quad_segs=16`), which is what
 * `budget.road_corridor` buffers every road with. */
const QUAD_SEGS = 16;

/** One segment's corridor, as the convex polygon GEOS buffers it to: the two offset sides plus a
 * round cap at each end, every cap vertex at `QUAD_SEGS` steps per quarter turn from the segment's
 * own normal -- GEOS's `addDirectedFillet` places them on the same angles. CCW, open (no repeated
 * closing vertex). Matching GEOS's polygon rather than the true round capsule is the point: the
 * parity fixtures then measure the formula, not two different discretisations of one circle.
 *
 * A zero-length segment is a POINT, which GEOS buffers to a full circle starting at angle 0; so
 * does this. Without that case the direction is atan2(0, 0) = 0 and the "capsule" would still come
 * out as a circle -- but only by accident of which way that degenerate angle happens to point. */
export function capsule(s: Segment): Pt[] {
  const step = Math.PI / (2 * QUAD_SEGS);
  const out: Pt[] = [];
  const arc = (cx: number, cy: number, from: number, n: number): void => {
    for (let k = 0; k < n; k++) {
      const a = from + k * step;
      out.push([cx + s.hw * Math.cos(a), cy + s.hw * Math.sin(a)]);
    }
  };
  if (s.x0 === s.x1 && s.y0 === s.y1) {
    arc(s.x0, s.y0, 0, 4 * QUAD_SEGS);
    return out;
  }
  const theta = Math.atan2(s.y1 - s.y0, s.x1 - s.x0);
  // Far cap from the right side round to the left, then the near cap from the left round to the
  // right; the two straight sides are the edges joining them. 2*QUAD_SEGS + 1 vertices per cap:
  // both of its ends lie on the offset sides.
  arc(s.x1, s.y1, theta - Math.PI / 2, 2 * QUAD_SEGS + 1);
  arc(s.x0, s.y0, theta + Math.PI / 2, 2 * QUAD_SEGS + 1);
  return out;
}

/** Twice the signed area (shoelace): positive for a CCW ring, negative for a CW one. */
function area2(ring: readonly Pt[]): number {
  let s = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    s += (ring[j]![0] - ring[i]![0]) * (ring[j]![1] + ring[i]![1]);
  }
  return s;
}

/** Sutherland--Hodgman: `subject` clipped to the CONVEX, CCW polygon `clip`.
 *
 * `subject` need not be convex. For a concave one the output can carry zero-width "bridges" along
 * a clip edge where the true intersection has several pieces -- so it is not a valid polygon to
 * DRAW -- but its signed area is still exactly the area of the intersection: each bridge is
 * traversed once in each direction along one line and contributes nothing. Area is all this module
 * takes from it, including when the output is clipped again (see `taken`). It keeps `subject`'s
 * orientation, so a CW hole ring comes back with negative area. */
function clipToConvex(subject: readonly Pt[], clip: readonly Pt[]): Pt[] {
  let out: Pt[] = subject.slice();
  for (let i = 0; i < clip.length && out.length > 0; i++) {
    const [ax, ay] = clip[i]!;
    const [bx, by] = clip[(i + 1) % clip.length]!;
    const side = (p: Pt): number => (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax);
    const inp = out;
    out = [];
    for (let j = 0; j < inp.length; j++) {
      const p = inp[j]!, q = inp[(j + 1) % inp.length]!;
      const sp = side(p), sq = side(q);
      if (sp >= 0) out.push(p);
      if ((sp >= 0) !== (sq >= 0)) {
        const t = sp / (sp - sq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return out;
}

interface Bbox { minX: number; minY: number; maxX: number; maxY: number }

function bboxOf(rings: readonly (readonly Pt[])[]): Bbox {
  const b = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (const ring of rings) {
    for (const [x, y] of ring) {
      b.minX = Math.min(b.minX, x); b.minY = Math.min(b.minY, y);
      b.maxX = Math.max(b.maxX, x); b.maxY = Math.max(b.maxY, y);
    }
  }
  return b;
}

const overlaps = (a: Bbox, b: Bbox): boolean =>
  a.minX <= b.maxX && b.minX <= a.maxX && a.minY <= b.maxY && b.minY <= a.maxY;

/** One building, prepared once at boot: every ring of every part, open, exteriors CCW and holes
 * CW, so that one signed sum over ALL of them is the building's area -- and, clipped, the area of
 * any piece of it. The bake orients them (`building_polygons`); nothing here re-derives which ring
 * is a hole. */
export interface Outline { rings: Pt[][]; area2: number; bbox: Bbox }

/** Prepare a building's outline from the bundle's polygons (each a list of closed rings). */
export function outline(polygons: readonly Polygon[]): Outline {
  // Drop each ring's closing vertex: shapely writes rings closed, and SH on a closed ring would
  // walk one zero-length edge -- harmless to the area, but a vertex with no business being there.
  const rings = polygons.flatMap((poly) => poly.map((ring) => ring.slice(0, -1)));
  return { rings, area2: rings.reduce((s, r) => s + area2(r), 0), bbox: bboxOf(rings) };
}

/** Twice the area of `rings` inside the UNION of `caps`, by inclusion--exclusion over the caps.
 *
 * The union is never constructed: |R & (C1 | ... | Ck)| is the alternating sum of
 * |R & C_S| over nonempty subsets S, and each R & C_S is `rings` clipped by every cap in S in
 * turn -- all of them convex, which is what makes SH exact here. A subset whose intersection is
 * already empty prunes every superset of it. Exponential in k, the number of capsules whose box
 * meets THIS building: the widget has at most two roads of one segment each, so k <= 2. */
function taken(rings: readonly Pt[][], caps: readonly Pt[][], start: number, sign: number): number {
  let total = 0;
  for (let i = start; i < caps.length; i++) {
    const clipped = rings.map((r) => clipToConvex(r, caps[i]!));
    const a = clipped.reduce((s, r) => s + area2(r), 0);
    if (a === 0) continue;
    total += sign * a + taken(clipped, caps, i + 1, -sign);
  }
  return total;
}

/** Per-building `c_i = area(outline_i & corridor) / area(outline_i)` -- the share of each building
 * the road set takes. Mirrors `buildings._overlap_fraction` on the same outlines, at every tier:
 * a disc tier ships its 64-gon, a footprint tier its real polygon, and this never knows which.
 *
 * Split out from `sumC` for the same reason ruling R8 split it out in Python: the widget needs the
 * per-building values (each building is shaded at `alpha = c_i`) AND their sum (the readout), and
 * writing the formula twice is how a picture comes to disagree with the number printed beside it.
 *
 * The CLAMP is live, not decoration: inclusion--exclusion adds and subtracts, so a building that
 * two overlapping corridors cover between them -- but neither alone -- comes out at a1 + a2 - a12,
 * which lands an ulp above 1 for about one such building in 600 (measured by random search).
 * Canvas silently IGNORES a `globalAlpha` outside [0, 1] and keeps the previous building's, so an
 * unclamped c would shade one home at its neighbour's cost. */
export function contributions(buildings: readonly Outline[], segs: readonly Segment[]): Float64Array {
  const caps = segs.map(capsule);
  const boxes = caps.map((c) => bboxOf([c]));
  const out = new Float64Array(buildings.length);
  for (let i = 0; i < buildings.length; i++) {
    const b = buildings[i]!;
    const near = caps.filter((_, k) => overlaps(boxes[k]!, b.bbox));
    if (near.length === 0) continue;
    out[i] = Math.min(1, Math.max(0, taken(b.rings, near, 0, 1) / b.area2));
  }
  return out;
}

/** `Σ c_i` -- displacement, in buildings. A sum over `contributions` rather than a second copy of
 * the formula: the baked fixtures pin THIS against `budget.displacement`, so the widget's readout
 * and its shading are pinned by the same measurement. */
export function sumC(c: Float64Array): number {
  let total = 0;
  for (const v of c) total += v;
  return total;
}
