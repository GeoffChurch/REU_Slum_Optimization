import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import type { FieldBundle, Road } from "../src/field.js";
import { contributions, flatten, outline, sumC } from "../src/model/displacement.js";

const bundle = JSON.parse(
  readFileSync("../examples/displacement-field/field.json", "utf8")) as FieldBundle;
const OUTLINES = bundle.buildings.map(outline);

// 1e-5 relative, and it is the STORE's resolution, not the model's. The widget clips the same
// quantised outlines Python measured, against the same polygon GEOS buffers each road to
// (`capsule`), so the two agree to float noise; what they cannot agree past is `sum_c` itself,
// which the bake writes at 6 significant figures -- half a step of that is up to 5e-6 relative.
const TOL = 1e-5;

const cost = (roads: readonly Road[]): number => sumC(contributions(OUTLINES, flatten(roads)));

test("every baked fixture's sum_c is reproduced from its own coordinates", () => {
  assert.equal(bundle.reference.length, 6);
  for (const c of bundle.reference) {
    const got = cost(c.roads);
    const rel = Math.abs(got - c.sum_c) / Math.max(c.sum_c, 1);
    assert.ok(rel < TOL, `${c.name}: TS ${got} vs Python ${c.sum_c} (rel ${rel})`);
  }
});

test("the outside-the-block fixture is exactly zero, not merely close", () => {
  const outside = bundle.reference.find((c) => c.name === "outside")!;
  assert.strictEqual(cost(outside.roads), 0,
    "outside's road clears every building by hundreds of metres, pinning compact support: a "
    + "corridor that reaches no outline costs exactly nothing");
});

test("a road drawn twice costs what one costs", () => {
  // The honest form of "overlap is free". The corridor is the UNION of the roads' buffers, so two
  // coincident roads occupy one corridor and are charged once. Equal to float noise, not bit for
  // bit: inclusion--exclusion reaches the union by adding both and subtracting their overlap,
  // where Python's union never double-counts at all. A port that SUMMED per-road shares would
  // charge this twice and fail by a factor of two, not by an ulp.
  const one = cost([bundle.roads[0]!]);
  const twice = cost([bundle.roads[0]!, bundle.roads[0]!]);
  assert.ok(Math.abs(twice - one) <= 1e-9 * one, `one road ${one}, the same road twice ${twice}`);
  assert.ok(cost(bundle.roads) > one, "a disjoint second road must add cost");
});

/** A closed CCW ring from its corners. */
const ring = (...pts: [number, number][]): [number, number][] => [...pts, pts[0]!];
const square = (x0: number, y0: number, x1: number, y1: number): [number, number][] =>
  ring([x0, y0], [x1, y0], [x1, y1], [x0, y1]);
/** A straight road far longer than any shape below, so its caps are nowhere near them. */
const band = (y: number, width_m: number): Road =>
  ({ coords: [[-100, y], [100, y]], width_m });
const along = (x: number, width_m: number): Road =>
  ({ coords: [[x, -100], [x, 100]], width_m });
const c1 = (polygons: [number, number][][][], roads: Road[]): number =>
  contributions([outline(polygons)], flatten(roads))[0]!;

test("c is the share of the building the corridor covers, not whether it touches", () => {
  // Corridor |y| <= 1; the square spans y 0..2, so the corridor takes exactly its lower half. The
  // retired centre-distance rule would have read this building as FULLY displaced (its centre,
  // at y = 1, is inside), which is the over-count this model exists to remove.
  assert.ok(Math.abs(c1([[square(0, 0, 1, 2)]], [band(0, 2)]) - 0.5) < 1e-12);
});

test("two crossing roads are charged their UNION, not the sum of their shares", () => {
  // Square [-1,1]^2, one 1 m road along each axis: each takes 2 of its 4 m^2, together only 3 --
  // the 1 m^2 where they cross is taken once. A per-road sum would read 1.0.
  const got = c1([[square(-1, -1, 1, 1)]], [band(0, 1), along(0, 1)]);
  assert.ok(Math.abs(got - 0.75) < 1e-12, `union share ${got}, expected 0.75`);
});

test("a hole is not building: a CW interior ring subtracts", () => {
  // A 4x4 courtyard house with a 2x2 open courtyard (hole CW, as the bake orients it), crossed by
  // a 1 m road through the middle: the road covers 4 m^2 of the square but 2 of those are the
  // courtyard, so it takes 2 m^2 of a 12 m^2 building. Ignoring the hole's orientation reads 1/4.
  const courtyard: [number, number][] = [[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]];
  const got = c1([[square(-2, -2, 2, 2), courtyard]], [band(0, 1)]);
  assert.ok(Math.abs(got - 2 / 12) < 1e-12, `courtyard house share ${got}, expected 1/6`);
});

test("a concave building cut into two pieces is charged both of them", () => {
  // A U whose two prongs a 1 m road crosses: the clip is two disjoint 1x1 pieces, which the
  // clipper returns as ONE ring bridged along the corridor's edge. The bridge carries no area, so
  // the share is 2 of the U's 7 m^2 -- a clipper that dropped the second piece would read 1/7.
  const u = ring([0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]);
  const got = c1([[u]], [band(2, 1)]);
  assert.ok(Math.abs(got - 2 / 7) < 1e-12, `U share ${got}, expected 2/7`);
});

test("the corridor ends at the segment's round caps, not on the infinite line", () => {
  // Road (0,0)-(10,0) at 4 m. A unit square just past the end sits wholly inside the cap (its far
  // corner is 1.58 m from the endpoint); one 50 m further along the SAME line is on the infinite
  // line's corridor but nowhere near this road's.
  const road: Road = { coords: [[0, 0], [10, 0]], width_m: 4 };
  assert.equal(c1([[square(10.5, -0.5, 11.5, 0.5)]], [road]), 1);
  assert.equal(c1([[square(60, -0.5, 61, 0.5)]], [road]), 0);
});

test("a zero-length road is a disc of its own half-width, not a NaN", () => {
  // GEOS buffers a zero-length line to a circle, and so does `capsule`. A 2x2 square centred on it
  // lies inside radius 3.5 (corner at 1.41 m); one 10 m away does not reach it.
  const dot: Road = { coords: [[0, 0], [0, 0]], width_m: 7 };
  assert.equal(c1([[square(-1, -1, 1, 1)]], [dot]), 1);
  assert.equal(c1([[square(9, -1, 11, 1)]], [dot]), 0);
});

test("c never leaves [0, 1], because canvas ignores an out-of-range alpha", () => {
  // A square covered by the UNION of two overlapping bands but by neither alone: inclusion--
  // exclusion reaches its area as a1 + a2 - a12, and on this one (found by random search -- 345 of
  // 200,000 such squares do it) the float sum lands at 1 + 4.4e-16 unclamped. Canvas drops a
  // `globalAlpha` above 1 and keeps the previous building's, so the clamp is what keeps each shade
  // its own. A building inside every corridor is NOT a test of it: each clip then returns the
  // subject unchanged, bit for bit, and the sum is exact.
  const sq: [number, number][] = [
    [8.482439517974854, 2.8757596015930176], [11.081203699111938, 2.8757596015930176],
    [11.081203699111938, 5.4745237827301025], [8.482439517974854, 5.4745237827301025],
    [8.482439517974854, 2.8757596015930176]];
  const got = c1([[sq]], [band(2.8757596015930176, 4.626065880815358),
                          band(5.4745237827301025, 3.7541263096212445)]);
  assert.equal(got, 1, `c = ${got} for a building its two corridors cover between them`);
});
