import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { METRICS, ranking, scores, selectAt, type MetricName } from "../src/model/screen.js";

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
