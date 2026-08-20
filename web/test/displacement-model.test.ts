import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import type { FieldBundle } from "../src/field.js";
import { corridorDistance, flatten, sumC } from "../src/model/displacement.js";

const bundle = JSON.parse(
  readFileSync("../examples/displacement-field/field.json", "utf8")) as FieldBundle;

// 1e-3 relative. This is SHAPELY's residual, not slack invented for this test -- and the mechanism
// is specific, not "bigger roads have bigger residuals" (an earlier draft of this comment claimed
// exactly that on two data points and was wrong; see task-4-report.md's fix round 2). A buffer's
// STRAIGHT SIDES are exact offsets of the line, so both shapely and this closed form agree on them
// to float noise (~1e-16, confirmed for road1/coincident/in_a_gap, where every building's nearest
// corridor point lands on a side). Only the ROUND CAPS at a segment's ends -- and any joins, for a
// multi-segment road -- are polygonised by shapely, so only a building whose nearest corridor point
// falls past an end, on a cap, can disagree at all; and the size of that disagreement still scales
// with the corridor's half-width once it does. That is why "apart" (a 7 m fixture, but TWO roads,
// hence four caps instead of two) shows a residual where the single 7 m roads do not, and why
// "widest" (road1's own geometry, just 20 m instead of 7 m wide) shows the worst one: the same
// capped buildings, a wider corridor. Recomputing all six fixtures from the bundle's own quantised
// (x, y, r) against the baked `sum_c` values, worst relative disagreement is 6.07e-05 ("widest"), so
// TOL carries ~16.5x headroom.
const TOL = 1e-3;

test("every baked fixture's sum_c is reproduced from its own coordinates", () => {
  const { x, y, r } = bundle.buildings;
  assert.equal(bundle.reference.length, 6);
  for (const c of bundle.reference) {
    const got = sumC(r, corridorDistance(x, y, flatten(c.roads)));
    const rel = Math.abs(got - c.sum_c) / Math.max(c.sum_c, 1);
    assert.ok(rel < TOL, `${c.name}: TS ${got} vs Python ${c.sum_c} (rel ${rel})`);
  }
});

test("the outside-the-block fixture is exactly zero, not merely close", () => {
  const { x, y, r } = bundle.buildings;
  const outside = bundle.reference.find((c) => c.name === "outside")!;
  assert.strictEqual(sumC(r, corridorDistance(x, y, flatten(outside.roads))), 0,
    "outside's road clears every building by hundreds of metres, pinning compact support (far " +
    "away costs exactly nothing) -- not the clip boundary itself, which needs its own case below");
});

test("a road drawn twice costs exactly what one costs", () => {
  // The honest form of "overlap is free". Each road is buffered on its OWN width and only then
  // unioned, so two coincident roads occupy one corridor and are charged once -- an equality, not
  // a discount. A TypeScript port that summed per-road distances instead of minimising over
  // segments would pass a `coincident < apart` check and fail this one.
  const { x, y, r } = bundle.buildings;
  const cost = (name: string): number => {
    const c = bundle.reference.find((k) => k.name === name)!;
    return sumC(r, corridorDistance(x, y, flatten(c.roads)));
  };
  assert.equal(cost("coincident"), cost("road1"));
  assert.ok(cost("apart") > cost("road1"), "a disjoint second road must add cost");
});

test("a zero-length road is its own endpoint rather than a NaN", () => {
  const segs = flatten([{ coords: [[0, 0], [0, 0]], width_m: 7 }]);
  const d = corridorDistance([10], [0], segs);
  assert.ok(Number.isFinite(d[0]!), `degenerate road produced ${d[0]}`);
  assert.equal(d[0], 10 - 3.5);
});

test("distance clamps to the segment, not the infinite line it lies on", () => {
  // No baked fixture reaches this: every fixture road is a chord spanning (or nearly spanning) the
  // whole block, so every building's perpendicular foot lands within [0, 1] of the segment and the
  // clamp never binds there (confirmed by recomputing every fixture's projection parameter directly
  // -- see task-4-report.md). This synthetic case is chosen so the clamp dominates: a short segment
  // and a building well past one end, positioned so the clamped and unclamped answers are nowhere
  // close.
  const segs = flatten([{ coords: [[0, 0], [10, 0]], width_m: 4 }]);
  const d = corridorDistance([60], [0], segs);
  // Clamped: the nearest point on the SEGMENT is its (10, 0) endpoint, so distance = 50 - hw = 48.
  // Unclamped, the projection parameter is t = 6, landing the "nearest point" at (60, 0) -- i.e. on
  // top of the building -- for a distance near 0. 48 and 0 are not close by any tolerance.
  assert.equal(d[0], 50 - 2);
});

test("the clip's real boundary: d == r gives exactly zero, d just inside gives something positive", () => {
  // "outside" pins compact support, but its road clears everything by hundreds of metres, nowhere
  // near this boundary. "Gap-hugging is free" is entirely about d == r, so it needs a direct case.
  const eps = 1e-6;
  assert.strictEqual(sumC([10], new Float64Array([10])), 0, "d == r must clip to exactly 0");
  assert.ok(sumC([10], new Float64Array([10 - eps])) > 0, "d just inside r must be strictly positive");
});

test("r == 0 (a coincident-points building) contributes exactly 1 or 0, never a fraction", () => {
  // Mirrors tests/test_budget.py::test_displacement_contributions_pins_the_r_equals_zero_convention.
  // Untested otherwise: this bundle's minimum baked radius is 1.13973, so r == 0 is unreachable
  // through any baked fixture. r_i = 0 is the one branch sumC's total can hide -- a 0-or-1
  // contribution moves an aggregate either way just as easily as any other -- so pin it directly:
  // a coincident point sitting exactly on the corridor (d <= 0) is fully displaced, with no radius
  // to graze it partially; one that is not touching (d > 0) contributes nothing at all, however
  // close.
  assert.equal(sumC([0], new Float64Array([0])), 1);
  assert.equal(sumC([0], new Float64Array([0.001])), 0);
});

test("no roads means no cost, not an empty-array minimum of Infinity leaking into sumC", () => {
  assert.strictEqual(sumC(bundle.buildings.r, corridorDistance(
    bundle.buildings.x, bundle.buildings.y, [])), 0);
});
