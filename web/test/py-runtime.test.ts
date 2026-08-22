import { strict as assert } from "node:assert";
import { test } from "node:test";
import { PYODIDE_INDEX_URL, type PyResult, type PyRuntime } from "../src/py/runtime.js";

test("the pinned index URL is an exact version, not a floating one", () => {
  // jsDelivr serves immutable versioned paths, which is the entire stability argument: geopandas
  // and pyproj both disappeared from the distribution at Pyodide 0.28.0 (present through 0.27.7,
  // absent through 0.29.1) and returned only at 0.29.2 -- shapely, by contrast, was present the
  // whole span with no gap. MEASURED by fetching v0.25.1 through v0.29.2's own pyodide-lock.json
  // directly (this task's report; the brief's claim that geopandas alone was "removed in 0.27"
  // and all three were "disabled in 0.28" does not match: the disappearance starts at 0.28.0, not
  // 0.27, and shapely never disappears). A `latest` or major-only URL re-opens that.
  assert.match(PYODIDE_INDEX_URL, /\/v\d+\.\d+\.\d+\/full\/$/);
  assert.ok(PYODIDE_INDEX_URL.includes("v0.29.2"));
});

/** A runtime that records its calls and resolves with whatever it was handed.
 *
 * `boots` counts the FAKE's own calls to `boot()` -- this fake has no memoisation of its own, so
 * the test below that reads `r.boots` is a CONTRACT test (what a caller may assume `PyRuntime`
 * looks like) rather than a guard on `pyodideRuntime`'s own idempotence (`runtime.ts`'s `??=`).
 * This file documents that contract rather than guarding the real runtime's memoisation, and
 * confirms it by fault injection rather than asserting it: dropping `pyodideRuntime`'s `??=` does
 * NOT redden this file (this task's fault injection 2; see the task report), because the real
 * runtime is never constructed here.
 *
 * DEBT: nothing in this task's own suite boots a real Pyodide, so nothing here can observe the
 * `??=` actually preventing a second load. `web/test/pyodide-parity.test.ts` (Task 5) is where a
 * real boot happens and is where that observation belongs; this test does not claim Task 5 (not
 * yet written at this commit) reuses `pyodideRuntime` itself rather than its own boot sequence --
 * only that a real boot is the only place the claim could be checked.
 */
function fakeRuntime(result: PyResult): PyRuntime & { boots: number; roads: [number, number][][] } {
  const roads: [number, number][][] = [];
  let boots = 0;
  return {
    get boots() { return boots; },
    roads,
    async boot() { boots += 1; },
    async solve(road) { roads.push(road); return result; },
  };
}

test("a fake satisfies the contract the widget depends on", async () => {
  const r = fakeRuntime({ permeability: 0.5, roadMetres: 10, potential: [1], conductance: [2] });
  await r.boot();
  await r.boot();
  assert.equal(r.boots, 2, "the fake counts boots; idempotence is the real runtime's job");
  const got = await r.solve([[0, 0], [1, 1]]);
  assert.equal(got.permeability, 0.5);
  assert.deepEqual(r.roads.at(-1), [[0, 0], [1, 1]]);
});
