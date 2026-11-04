import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import {
  METRICS, rankShades, ranking, scores, selectAt, type MetricName,
} from "../src/model/screen.js";

const ct = JSON.parse(
  readFileSync("../examples/screen-map/capetown.json", "utf8")) as CityBundle;
const nb = JSON.parse(
  readFileSync("../examples/screen-map/nairobi.json", "utf8")) as CityBundle;

test("each metric is its published formula", () => {
  assert.equal(METRICS.density(100, 10000, 400), 100 / 10000);
  assert.equal(METRICS.depth_proxy(100, 10000, 400), Math.sqrt(100 * 10000) / 400);
  assert.equal(METRICS.density_compactness(100, 10000, 400), 100 / (400 * 400));
  assert.equal(METRICS.depth_density_proxy(100, 10000, 400),
    (Math.sqrt(100 * 10000) / 400) * (100 / 10000));
});

test("the ranking is sorted descending and is a permutation", () => {
  const s = scores(ct, "depth_density_proxy");
  const order = ranking(ct, "depth_density_proxy");
  assert.equal(order.length, ct.n_blocks);
  assert.equal(new Set(order).size, ct.n_blocks, "a permutation, not a resampling");
  for (let i = 1; i < order.length; i++) {
    assert.ok(s[order[i - 1]!]! >= s[order[i]!]!, `out of order at ${i}`);
  }
});

test("selection at each shipped floor reproduces the baked pool size and precision/recall", () => {
  // The bundle's `floors` were READ from the bake-off CSV, which computed them by a different
  // route entirely. Two independent paths agreeing is the strongest guard on this widget.
  assert.ok(ct.floors.length >= 2,
    `only ${ct.floors.length} floor(s) in the bundle; this loop would assert almost nothing`);
  for (const f of ct.floors) {
    const metric = f.metric as MetricName;
    const s = scores(ct, metric);
    const got = selectAt(ct, ranking(ct, metric), s, f.value);
    assert.equal(got.count, f.n, `${f.metric} pool size`);
    if (f.precision !== null) {
      // The bundle's floor precision/recall are FIELD VALUES: rounded to 6 significant digits at
      // bake time (scripts/_bundle_io.py's `sigfig`, "%.6g") to keep the payload small. A live,
      // full-precision recomputation is bit-for-bit identical to screen_comparison.csv's own
      // UNROUNDED floor_prec/floor_recall (verified by hand: 0.27492447129909364 and
      // 0.6671554252199413 for depth_density_proxy, 0.24452554744525548 and 0.5894428152492669 for
      // density_compactness) but differs from the bundle's rounded copy by ~5e-7 -- so 1e-9 here
      // would fail against a provably correct implementation. 1e-6 comfortably covers that rounding
      // noise while still failing on any real formula or prefix-sum bug, which moves precision/
      // recall by whole percentage points, not fractions of a ppm. Python's own equivalent check
      // (tests/test_screen_map_bundle.py::test_precision_and_recall_at_the_shipped_floor_match_the_bakeoff)
      // compares against the CSV directly and uses `rel_tol=1e-6` for the same reason.
      assert.ok(Math.abs(got.precision! - f.precision) < 1e-6, `${f.metric} precision`);
      assert.ok(Math.abs(got.recall! - f.recall!) < 1e-6, `${f.metric} recall`);
    }
  }
});

test("a city with no ground truth reports no precision or recall", () => {
  const s = scores(nb, "depth_density_proxy");
  const got = selectAt(nb, ranking(nb, "depth_density_proxy"), s, 0.0128);
  assert.ok(got.count > 0, "the pool is still counted");
  assert.equal(got.precision, null);
  assert.equal(got.recall, null);
});

test("raising the floor never enlarges the selection", () => {
  // Monotonicity. It is the property the prefix representation depends on, and a sort comparator
  // with a sign error would break it while leaving every count plausible.
  const s = scores(ct, "density");
  const order = ranking(ct, "density");
  let prev = Infinity;
  const counts: number[] = [];
  for (const floor of [0, 1e-4, 1e-3, 1e-2, 1e-1]) {
    const n = selectAt(ct, order, s, floor).count;
    assert.ok(n <= prev, `floor ${floor} selected ${n} after ${prev}`);
    counts.push(n);
    prev = n;
  }
  // A selection that never changes satisfies `n <= prev` at every step while testing nothing.
  assert.ok(new Set(counts).size > 2,
    `the floor sweep produced ${new Set(counts).size} distinct pool size(s): ${counts}`);
});

test("score shades: each side of the floor takes half the ramp, by area-weighted mid-rank", () => {
  // Brute force on a five-block bundle, against the definition: a block's shade is the share of
  // its side's area scoring lower than it, its own counted half, put into its side's half.
  const area = [10, 30, 20, 40, 50];
  const b = { n_blocks: 5, area_m2: area } as unknown as CityBundle;
  const s = Float64Array.from([0.1, 0.5, 0.3, 0.9, 0.7]);
  const order = Int32Array.from([3, 4, 1, 2, 0]);               // descending by score
  const floor = 0.5;                                            // selects 1, 3, 4 (>= floor)
  const t = rankShades(b, order, s, floor);
  const below = 10 + 20, above = 30 + 50 + 40;
  const want = [0.5 * (5 / below), 0.5 + 0.5 * (15 / above), 0.5 * ((10 + 10) / below),
                0.5 + 0.5 * ((30 + 50 + 20) / above), 0.5 + 0.5 * ((30 + 25) / above)];
  want.forEach((w, i) => assert.ok(Math.abs(t[i]! - w) < 1e-12, `block ${i}: ${t[i]} not ${w}`));
});

test("score shades on Cape Town: sides split at 1/2, ranks ordered, each side's area mean its middle",
  () => {
  const s = scores(ct, "depth_density_proxy");
  const order = ranking(ct, "depth_density_proxy");
  const floor = ct.floors.find((f) => f.metric === "depth_density_proxy")!.value;
  const t = rankShades(ct, order, s, floor);
  let prev = -1, aBelow = 0, aAbove = 0, mBelow = 0, mAbove = 0;
  for (let i = order.length - 1; i >= 0; i--) {
    const k = order[i]!, a = ct.area_m2[k]!;
    const up = s[k]! >= floor;
    assert.ok(up ? t[k]! >= 0.5 && t[k]! <= 1 : t[k]! >= 0 && t[k]! < 0.5,
      `block ${k} (score ${s[k]}) shaded ${t[k]} on the wrong side of the floor`);
    assert.ok(t[k]! >= prev, `shade falls at block ${k} though its score does not`);
    prev = t[k]!;
    if (up) { aAbove += a; mAbove += a * t[k]!; } else { aBelow += a; mBelow += a * t[k]!; }
  }
  // An area-weighted mid-rank's area-weighted mean is exactly the middle of its range: 1/4 and
  // 3/4. A count-weighted rank, or a rank over the whole city, misses both.
  assert.ok(Math.abs(mBelow / aBelow - 0.25) < 1e-9, `below: ${mBelow / aBelow}`);
  assert.ok(Math.abs(mAbove / aAbove - 0.75) < 1e-9, `above: ${mAbove / aAbove}`);
});
