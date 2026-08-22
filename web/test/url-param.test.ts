import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  boolParam, enumParam, intParam, nullableNumberParam, nullableStringParam,
  numberParam, roadsParam, stringParam,
} from "../src/url/param.js";
import type { Road } from "../src/field.js";

test("intParam round-trips and rejects everything that is not a non-negative integer", () => {
  const p = intParam("prefix");
  assert.deepEqual(p.keys, ["prefix"]);
  assert.deepEqual(p.encode(14, 0), { prefix: "14" });
  assert.equal(p.decode({ prefix: "14" }, 0), 14);
  for (const bad of ["", " ", "1.5", "-1", "abc", "NaN", "Infinity", "1e3x"]) {
    assert.equal(p.decode({ prefix: bad }, 0), null, bad);
  }
});

test("numberParam keeps six significant figures and rejects non-finite input", () => {
  const p = numberParam("floor", 6);
  assert.deepEqual(p.encode(0.012799999, 0), { floor: "0.0128" });
  assert.equal(p.decode({ floor: "0.0128" }, 0), 0.0128);
  for (const bad of ["", "NaN", "Infinity", "-Infinity", "0x10", "abc"]) {
    assert.equal(p.decode({ floor: bad }, 0), null, bad);
  }
});

test("nullableNumberParam spells null as the ABSENT key, both ways", () => {
  const p = nullableNumberParam("floor", 6);
  assert.deepEqual(p.encode(null, null), {});
  assert.deepEqual(p.encode(0.02, null), { floor: "0.02" });
  assert.equal(p.decode({ floor: "0.02" }, null), 0.02);
  assert.ok(p.same(null, null));
  assert.ok(!p.same(null, 0.02));
});

test("boolParam is 0/1 and refuses anything else -- not a truthiness test", () => {
  const p = boolParam("halos");
  assert.deepEqual(p.encode(false, true), { halos: "0" });
  assert.equal(p.decode({ halos: "1" }, false), true);
  assert.equal(p.decode({ halos: "0" }, true), false);
  for (const bad of ["true", "yes", "", "2", "maybe"]) {
    assert.equal(p.decode({ halos: bad }, false), null, bad);
  }
});

test("enumParam admits exactly its declared members", () => {
  const p = enumParam("city", ["capetown", "nairobi"] as const);
  assert.equal(p.decode({ city: "nairobi" }, "capetown"), "nairobi");
  assert.equal(p.decode({ city: "kampala" }, "capetown"), null);
  // A prototype key must not be admitted by an `in`-style membership test.
  assert.equal(p.decode({ city: "toString" }, "capetown"), null);
});

test("nullableStringParam round-trips a slug and spells null as the absent key", () => {
  const p = nullableStringParam("method");
  assert.deepEqual(p.encode(null, null), {});
  assert.deepEqual(p.encode("clearance", null), { method: "clearance" });
  assert.equal(p.decode({ method: "clearance" }, null), "clearance");
  assert.equal(p.decode({ method: "" }, null), null);
});

test("stringParam round-trips a block id and rejects the empty string", () => {
  const p = stringParam("seed");
  assert.deepEqual(p.encode("ZAF.9.3.1_1_40972", "X"), { seed: "ZAF.9.3.1_1_40972" });
  assert.equal(p.decode({ seed: "ZAF.9.3.1_1_40972" }, "X"), "ZAF.9.3.1_1_40972");
  assert.equal(p.decode({ seed: "" }, "X"), null);
});

const ROADS: Road[] = [
  { coords: [[132.53, 3.82], [40.24, 113.92]], width_m: 7 },
  { coords: [[101.81, 7.77], [26.28, 97.88]], width_m: 7 },
];

test("roadsParam emits ONLY the sub-keys that changed", () => {
  const p = roadsParam("road1", "road2", "width");
  assert.deepEqual(p.keys, ["road1", "road2", "width"]);
  const wider = ROADS.map((r) => ({ ...r, width_m: 12 }));
  assert.deepEqual(p.encode(wider, ROADS), { width: "12" });
  const moved = [{ ...ROADS[0]!, coords: [[1, 2], [3, 4]] as [number, number][] }, ROADS[1]!];
  assert.deepEqual(p.encode(moved, ROADS),
                   { road1: "1,2,3,4", road2: "101.8,7.8,26.3,97.9" });
});

test("roadsParam decodes a width-only URL against the initial geometry", () => {
  const p = roadsParam("road1", "road2", "width");
  const got = p.decode({ width: "12" }, ROADS);
  assert.deepEqual(got?.map((r) => r.width_m), [12, 12]);
  assert.deepEqual(got?.[0]!.coords, ROADS[0]!.coords);
});

test("roadsParam refuses a half-specified pair and any malformed segment", () => {
  const p = roadsParam("road1", "road2", "width");
  assert.equal(p.decode({ road1: "1,2,3,4" }, ROADS), null, "road2 missing");
  assert.equal(p.decode({ road1: "1,2,3", road2: "1,2,3,4" }, ROADS), null, "three numbers");
  assert.equal(p.decode({ road1: "1,2,3,x", road2: "1,2,3,4" }, ROADS), null, "not a number");
  assert.equal(p.decode({ width: "0" }, ROADS), null, "non-positive width");
  assert.equal(p.decode({ width: "-3" }, ROADS), null, "negative width");
});

test("roadsParam.same ignores object identity and sees a moved vertex", () => {
  const p = roadsParam("road1", "road2", "width");
  assert.ok(p.same(ROADS, ROADS.map((r) => ({ coords: [...r.coords], width_m: r.width_m }))));
  assert.ok(!p.same(ROADS, [{ ...ROADS[0]!, coords: [[0, 0], [1, 1]] }, ROADS[1]!]));
});
