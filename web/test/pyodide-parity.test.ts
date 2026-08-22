import { strict as assert } from "node:assert";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import type { AuthoringBlock } from "../src/authoring.js";
import { pyodideRuntime } from "../src/py/runtime.js";

/** Boots the pinned Pyodide for real and checks what `reblock.permeability` computes under it
 * against the numbers CPython baked into `examples/authoring/block.json`.
 *
 * This is the only test in the repo that runs `reblock` under the runtime the browser will use,
 * and three failure modes are invisible to any static import scan (design §5):
 *
 *   * a package leaving the distribution -- `geopandas` and `pyproj` are both absent from Pyodide
 *     0.28.0's own lockfile and present in 0.27.7's and 0.29.2's (measured by fetching each
 *     release's `pyodide-lock.json`, Task 3's report), so the pin is load-bearing and a bump can
 *     take an import's package away without touching a line of source;
 *   * the pandas major skew: 2.3.3 in the distribution against 3.0.3 in this checkout;
 *   * behavioural drift at call time, including through a deferred import -- `mesh.parcel_radii`
 *     imports `reblock.budget` inside the function body, so nothing but running the solve reaches
 *     it.
 *
 * Relative paths here are relative to `web/`, which is where `scripts/test.sh` runs `node --test`
 * from -- the same assumption every other file in this directory makes when it reads
 * `../examples/authoring/block.json`.
 */

/** The committed bundle, and the source of BOTH sides of the comparison: `reference[].road` is
 * the input handed to Pyodide, `reference[].permeability` is what CPython returned for it at bake
 * time. Nothing re-rounds the geometry in between, so every difference the assertion below can
 * see is the two runtimes' own arithmetic. */
const BUNDLE = JSON.parse(
  readFileSync("../examples/authoring/block.json", "utf8")) as AuthoringBlock;

/** How far the two runtimes are allowed to disagree about a permeability, in absolute terms.
 *
 * The claim started as exact equality and did not survive contact with the runtime. MEASURED at
 * this commit, and identical across repeated runs: `crossing` agrees to the last bit (difference
 * exactly 0), `spur` does not -- Pyodide returns 0.3637809513169823 where CPython bakes
 * 0.3637809513169825, a difference of 2.220446e-16, which at that magnitude is four units in the
 * last place. CPython on this machine reproduces both baked numbers exactly, so the disagreement
 * is Pyodide's arithmetic against CPython's, not a stale bake. The stacks differ underneath:
 * numpy 2.2.5 / scipy 1.14.1 on wasm against numpy 2.5.0 / scipy 1.18.0 on x86-64, and the
 * quantity is the tail of a sparse solve.
 *
 * 1e-13 is a STATED tolerance, not one widened until the suite went green:
 *
 *   * it is ~450x the largest disagreement actually observed, which is the headroom for a
 *     differently-built BLAS to accumulate a few more last bits than this one did;
 *   * it is ~1.7e10 times SMALLER than 1.66e-03, the smallest effect `authoring.d.ts` records as
 *     one a parity guard must not absorb (what rounding this bundle's geometry to centimetres
 *     does to `crossing`; `spur` moves by 1.96e-03). Design §1.4's 4.71e-05 is the same effect
 *     measured on the clearance method's road set, and is also enormous next to this;
 *   * it is strictly below the 1e-12 perturbation this task's fault injection 1 applies, so that
 *     injection reddens the test with an order of magnitude to spare.
 *
 * If this ever needs raising, the difference and its magnitude are the finding -- record them the
 * way this comment does. A tolerance widened until the test passes has stopped measuring the
 * runtime, which is the only thing this test exists to measure. */
const PARITY_TOL = 1e-13;

/** `node_modules/pyodide/`, absolute, with the trailing slash `pyodideRuntime` concatenates
 * `pyodide.mjs` onto.
 *
 * NOT `PYODIDE_INDEX_URL`, even though this file exists to check that pin's behaviour: under Node
 * `pyodide.mjs` resolves `indexURL` as a filesystem path whatever scheme the string carries, so a
 * CDN URL there becomes a relative path and the boot dies in module resolution (measured, Task 3's
 * report). What is booted is still the pinned distribution: `web/package.json` pins the `pyodide`
 * devDependency to the same 0.29.2 `PYODIDE_INDEX_URL` names (`py-runtime.test.ts` asserts that
 * version), the npm package carries the interpreter and `pyodide-lock.json`, and the package
 * wheels it does not carry are fetched by `loadPackage` from exactly
 * `https://cdn.jsdelivr.net/pyodide/v0.29.2/full/` and cached into `node_modules/pyodide/` --
 * observed in this task's runs, and the reason a first run needs a network and a second does not.
 */
const LOCAL_INDEX_URL = `${resolve("node_modules/pyodide")}/`;

/** The `reblock` wheel `micropip` installs, as `pixi run pip wheel` names it.
 *
 * `scripts/test.sh` rebuilds this immediately before running the suite, so what is installed here
 * is this checkout's `src/reblock` rather than whatever a previous build left behind. */
const WHEEL = resolve("../dist/reblock-0.1.0-py3-none-any.whl");

/** The runtime under test: the SHIPPING one, `pyodideRuntime` itself, not a boot sequence written
 * here that could drift from it. That is what puts `SOLVE_PACKAGES`, `SOLVE_PY_SOURCE`, the
 * `toPy`/`toJs` conversions and `block_from_bundle`'s reconstruction inside what this file
 * measures -- none of which any other test in the suite executes, because every other test injects
 * a fake `PyRuntime` instead. */
const runtime = pyodideRuntime(BUNDLE, pathToFileURL(WHEEL).href, LOCAL_INDEX_URL);

/** How long the first `boot()` call took. Set once, by whichever test boots first. */
let coldBootMs = 0;

/** Awaits `runtime.boot()` and returns how long THIS call took. After the first call that is
 * `pyodideRuntime`'s memoised promise resolving, not a second load -- which is what the third test
 * below measures. */
async function boot(): Promise<number> {
  if (!existsSync(WHEEL)) {
    // Reachable: `dist/` is gitignored, so a fresh checkout has no wheel until something builds
    // one. Without this, `micropip.install` fails on a missing `file://` URL from inside Python
    // and the traceback says nothing about how to fix it.
    throw new Error(
      `${WHEEL} does not exist. Build it with \`pixi run pip wheel --no-deps --wheel-dir dist .\` `
      + `from the repo root, or run the suite through \`npm test\`, which builds it first.`);
  }
  const started = performance.now();
  await runtime.boot();
  const elapsed = performance.now() - started;
  if (coldBootMs === 0) {
    coldBootMs = elapsed;
  }
  return elapsed;
}

test("the pinned runtime reproduces CPython's answer for every reference road", async () => {
  await boot();
  // A vacuous pass is the failure mode a `for ... of` invites: an emptied or renamed `reference`
  // would make every assertion below run zero times and the test go green having checked nothing.
  assert.ok(BUNDLE.reference.length > 0, "the bundle carries no reference cases to check");
  for (const c of BUNDLE.reference) {
    const got = await runtime.solve(c.road);
    const difference = Math.abs(got.permeability - c.permeability);
    // `assert.ok` on the difference rather than `assert.equal` on the values, because the two
    // runtimes are measured to differ by four units in the last place on one of these two roads --
    // see PARITY_TOL, which carries the numbers and the reasoning for how far apart they are
    // allowed to be. The message reports the difference either way, so a failure states the
    // magnitude rather than leaving it to be re-derived.
    assert.ok(difference <= PARITY_TOL,
      `${c.name}: Pyodide returned ${got.permeability}, CPython baked ${c.permeability} `
      + `(|difference| ${difference.toExponential(6)}, tolerance ${PARITY_TOL.toExponential(0)})`);
    // The arities the widget indexes the picture with. A runtime that answered with a plausible
    // number over a different graph would pass the line above and fail these.
    assert.equal(got.potential.length, BUNDLE.nodes.cx.length,
      `${c.name}: one potential per parcel`);
    assert.equal(got.conductance.length, BUNDLE.edges.rows.length,
      `${c.name}: one conductance per edge`);
  }
});

test("the runtime's pandas is the one the browser will get, not this checkout's", async () => {
  // The skew is real and pinning cannot remove it (design §1.1): the browser runs pandas 2.3.3
  // whatever we pin, while CI runs 3.0.3. This asserts the skew EXISTS rather than pretending it
  // does not, so that the day the two converge someone reads this line and deletes it.
  //
  // Booted directly rather than through `pyodideRuntime`: `PyRuntime` exposes `boot` and `solve`
  // and nothing else, deliberately (design §3), and widening it with a `runPython` escape hatch so
  // a test could reach the interpreter would be inventing API for this file. Both boots read the
  // same pinned distribution at `LOCAL_INDEX_URL`, which is where the version comes from; only
  // `pandas` is loaded here because that is all this assertion reads.
  const mod = (await import(`${LOCAL_INDEX_URL}pyodide.mjs`)) as typeof import("pyodide");
  const py = await mod.loadPyodide({ indexURL: LOCAL_INDEX_URL });
  await py.loadPackage("pandas");
  assert.match(py.runPython("import pandas; pandas.__version__"), /^2\./);
});

test("a second boot() resolves the first one's promise instead of loading Pyodide again",
  async () => {
    // Closes the debt `py-runtime.test.ts` records against this file: `pyodideRuntime`'s `??=`
    // memoisation ships, but every other test in the suite injects a fake `PyRuntime` and so
    // cannot observe it. This is the only place a real boot happens.
    //
    // Time is the only observable: a runtime that re-booted would return the same answers, just
    // after building a second interpreter, re-installing the wheel and re-running solve.py. The
    // memoised call awaits an already-settled promise and returns in about a tenth of a
    // millisecond (measured); a real second boot costs seconds even with every package already
    // cached, so a tenth of the first boot's own cost separates them by orders of magnitude.
    // CONFIRMED by fault injection (this task's report): deleting the `??=` reddens this.
    const again = await boot();
    assert.ok(again < coldBootMs / 10,
      `second boot() took ${again.toFixed(1)} ms against the first boot's `
      + `${coldBootMs.toFixed(1)} ms -- it re-ran the boot sequence rather than reusing it`);
  });
