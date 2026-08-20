import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import type { FieldBundle } from "../src/field.js";
import { corridorDistance, flatten, sumC } from "../src/model/displacement.js";

const bundle = JSON.parse(
  readFileSync("../examples/displacement-field/field.json", "utf8")) as FieldBundle;

// 1e-3 relative. This is SHAPELY's residual, not slack invented for this test: I recomputed all six
// fixtures from the bundle's own quantised (x, y, r) with this exact closed form and diffed against
// the baked `sum_c` values. Worst relative disagreement is 6.07e-05 (the "widest" fixture, a 20 m
// road), so TOL carries ~16x headroom. The residual is smallest for the 7 m roads and largest for
// "widest" because it scales with radius: shapely's buffer is an INSCRIBED polygon (its corners cut
// the true circular arc's sagitta), so shapely reports slightly larger distances -- and slightly
// smaller c -- than this closed form's exact circular offset, by an amount proportional to the
// buffer's own width. That gap is the entire budget TOL has to cover; it is not measuring anything
// about this module's correctness.
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
    "c must clip to exactly 0 at d = r -- a tolerance here would hide a soft tail");
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

test("no roads means no cost, not an empty-array minimum of Infinity leaking into sumC", () => {
  assert.strictEqual(sumC(bundle.buildings.r, corridorDistance(
    bundle.buildings.x, bundle.buildings.y, [])), 0);
});
