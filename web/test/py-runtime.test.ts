import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { AuthoringBlock } from "../src/authoring.js";
import {
  PYODIDE_INDEX_URL, pyodideRuntime, type PyResult, type PyRuntime,
} from "../src/py/runtime.js";

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
 * Nothing in THIS file boots a real Pyodide, so nothing here can observe the `??=` actually
 * preventing a second load. `web/test/pyodide-parity.test.ts` is where a real boot happens, it
 * boots `pyodideRuntime` itself rather than a boot sequence of its own, and it holds the
 * assertion that observes the memoisation: a second `boot()` returns in a fraction of a
 * millisecond where a re-boot costs seconds (measured 0.4 ms against 8274 ms with the `??=`
 * deleted, Task 5's report).
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

test("solve refuses a call made before boot, instead of reaching into a runtime that is not there",
  async () => {
    // Closes the debt Task 3 recorded: `pyodideRuntime.solve`'s null-`state` check shipped
    // untested. Reaching it costs nothing -- constructing a `pyodideRuntime` does no work and
    // starts no download (its own docstring), and the check is the first statement in `solve` --
    // so this needs neither a network round trip nor the 25-35 MB the CDN would serve.
    //
    // `assert.rejects` with the MESSAGE, not merely "it rejected": with the check deleted, the
    // destructuring on the next line rejects too, with a TypeError about destructuring `null`.
    // Both are rejections; only one of them is the guard.
    const block = JSON.parse(
      readFileSync("../examples/authoring/block.json", "utf8")) as AuthoringBlock;
    await assert.rejects(
      () => pyodideRuntime(block, "reblock-0.1.0-py3-none-any.whl").solve([[0, 0], [1, 1]]),
      /called before boot\(\) resolved/);
  });
