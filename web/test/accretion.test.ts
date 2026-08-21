import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { HoodBlock, HoodBundle } from "../src/hood.js";
import { depthProxy, grow, growth } from "../src/model/accretion.js";

const bundle = JSON.parse(
  readFileSync("../examples/region-grow/hood.json", "utf8")) as HoodBundle;
const indexOf = new Map(bundle.blocks.map((b, i) => [b.block_id, i]));

test("depthProxy is sqrt(n*A)/P", () => {
  assert.equal(depthProxy(100, 10000, 400), Math.sqrt(100 * 10000) / 400);
});

test("depthProxy is zero-safe on a degenerate perimeter", () => {
  // `_depth_proxy` in region.py is documented zero-safe; a NaN here would silently win every
  // argmax, since NaN comparisons are false and `min` would keep whatever it saw first.
  assert.equal(depthProxy(10, 0, 0), 0);
});

test("every reference case reproduces production's accretion ORDER", () => {
  // Order, not membership. A set comparison passes against any permutation, and the order IS what
  // the widget draws -- which is why Task 1 changed RegionBuilder to stop discarding it.
  for (const c of bundle.reference) {
    const seed = indexOf.get(c.seed);
    assert.notEqual(seed, undefined, `reference seed ${c.seed} is not in the bundle`);
    const got = grow(bundle.blocks, seed!, c.max_buildings)
      .map((i) => bundle.blocks[i]!.block_id);
    assert.deepEqual(got, c.order, `${c.seed} @ ${c.max_buildings}`);
  }
});

test("at least one reference order differs from its own sorted order", () => {
  // Without this the test above would pass against a `sorted()` implementation and guard nothing.
  // D2's defect #7 was exactly a fixture satisfied by its own twin.
  const informative = bundle.reference.filter(
    (c) => c.order.length > 3 && c.order.join() !== [...c.order].sort().join());
  assert.ok(informative.length >= 2,
    `only ${informative.length} reference cases have order != sorted; the fixture set cannot ` +
    `distinguish accretion order from sorted order`);
});

test("growth reports reaching the edge of the loaded neighbourhood", () => {
  // The production builder's own `if not frontier: break`. A budget far past what 213 blocks can
  // supply must stop and SAY so, not silently return a short region.
  const seed = indexOf.get(bundle.seed)!;
  const huge = growth(bundle.blocks, seed, 10 ** 9);
  assert.equal(huge.stoppedAtEdge, true);
  assert.equal(huge.order.length, bundle.blocks.length,
    "an unbounded budget consumes the seed's whole connected component");
});

test("growth does not report the edge when the budget bound it", () => {
  const seed = indexOf.get(bundle.seed)!;
  assert.equal(growth(bundle.blocks, seed, bundle.budget.default).stoppedAtEdge, false);
});

test("the block_id tie-break decides when proxy and count both tie", () => {
  // SYNTHETIC ON PURPOSE. Task 4 reported that flipping this tie-break in the Python builder
  // reddened NOTHING: across all 12 reference cases, no accretion step ever has two frontier
  // blocks with identical depth proxy AND identical building_count, so the third sort key is
  // never reached on real data. The bundle fixtures therefore cannot exercise it, and a
  // TypeScript bug in this exact comparison would ship unobserved.
  //
  // Real data cannot produce the tie, so the test constructs it: two neighbours identical in
  // every quantity the first two keys read, differing only in block_id. region.py breaks such a
  // tie by LOWER block_id (`min(..., key=lambda j: (-score, -count, ids[j]))`).
  const seed: HoodBlock = {
    block_id: "seed", n: 10, area_m2: 10000, perimeter_m: 400, rings: [], adj: [1, 2],
  };
  const tie = (id: string): HoodBlock => ({
    block_id: id, n: 50, area_m2: 40000, perimeter_m: 800, rings: [], adj: [0],
  });
  // "b_lower" sorts before "c_upper"; both are identical in n, area and perimeter.
  const blocks = [seed, tie("c_upper"), tie("b_lower")];
  assert.equal(depthProxy(50, 40000, 800), depthProxy(50, 40000, 800), "the fixture must tie");
  const order = grow(blocks, 0, 40).map((i) => blocks[i]!.block_id);
  assert.deepEqual(order, ["seed", "b_lower"],
    "on a full tie the LOWER block_id must win, matching region.py's third sort key");
});

test("growth is nested in the budget", () => {
  const seed = indexOf.get(bundle.seed)!;
  const small = grow(bundle.blocks, seed, 600);
  const big = grow(bundle.blocks, seed, 3000);
  // Guard the guard: if the two budgets happened to grow the same region, `big.slice(0, n)`
  // would equal `small` no matter what the ordering did, and this test would assert nothing.
  assert.ok(big.length > small.length,
    `both budgets grew ${big.length} blocks, so nesting is trivially satisfied here`);
  assert.deepEqual(big.slice(0, small.length), small);
});
