/** The Pyodide seam: the one place on the site that boots a Python runtime in the browser.
 *
 * `web/src/widgets/draw-road.ts` (Task 4) takes a `PyRuntime` the way every other widget on this
 * site takes a `StateFactory` (`state.ts`) -- injected, never constructed by the widget itself --
 * so the widget cannot tell whether it holds this or a hand-written fake. That is what lets Task
 * 4's boot tests run with no network and no 25-35 MB download: they inject a fake that satisfies
 * `PyRuntime`'s shape and never call `pyodideRuntime` at all.
 */
import type { AuthoringBlock } from "../authoring.js";
// Generated from this file's sibling solve.py by web/scripts/gen-solve-source.mjs, whose own
// header comment explains why solve.py's source is inlined into the bundle this way rather than
// fetched separately at boot.
import { SOLVE_PY_SOURCE } from "./solve-source.generated.js";

/** jsDelivr serves immutable versioned paths: `v0.28.0/full/` and `v0.29.2/full/` carry different
 * bytes, which is the entire stability argument for pinning an exact version rather than a
 * floating `latest` or major-only tag. MEASURED (this task's report, by fetching each version's
 * own `pyodide-lock.json` directly): `geopandas` and `pyproj` both disappear from the
 * distribution starting at 0.28.0 (present through 0.27.7, absent through 0.29.1) and return only
 * at 0.29.2; `shapely` is present throughout that whole span, with no gap.
 *
 * Kept in lockstep with `web/package.json`'s pinned `pyodide` devDependency at the SAME version:
 * that package is what `web/test/pyodide-parity.test.ts` (Task 5) boots under Node to check this
 * URL's real behaviour, so a version mismatch between the two would make that check compare the
 * wrong distribution against itself. */
export const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/";

/** Mirrors `web/src/py/solve.py`'s `PyResult` TypedDict field for field -- that file's own
 * docstring says the two are meant to read alike across the runtime boundary, and Task 5's parity
 * test depends on both sides agreeing about this shape. */
export interface PyResult {
  /** `1 - P(road)/p0` against the bundle's baked `baseline.p0`, computed from one
   * `solve_egress` call rather than the two `permeability()` would run (it recomputes the
   * road-invariant baseline on every call). `solve.py`'s own docstring cites Task 2's fault
   * injection 2, which measured the two agreeing bit for bit on both reference roads -- so this
   * is a cost choice, not a claim that the numbers would otherwise differ. */
  permeability: number;
  roadMetres: number;
  /** Per parcel, in the bundle's `nodes` order. */
  potential: number[];
  /** Per edge, in the bundle's `edges` order; the widget derives per-edge current from this and
   * `potential` as `conductance[i] * (potential[rows[i]] - potential[cols[i]])`, the same
   * expression `gen_web_bundle.py` already bakes. */
  conductance: number[];
}

export interface PyRuntime {
  /** Idempotent: a second call returns the first call's promise rather than booting Pyodide
   * twice. Resolves once `solve` is callable. A boot that rejects (the CDN unreachable, the pin
   * 404ing) stays rejected -- `boot()`'s memoisation keys on the promise existing, not on how it
   * settles, so this does not retry on a later call. */
  boot(): Promise<void>;
  /** `road` is a polyline in the bundle's own projected metres. Assumes `boot()` has already
   * resolved -- this never boots on the caller's behalf, because the whole point of a boot
   * control the reader presses is that a reader who never presses it never pays for it. */
  solve(road: [number, number][]): Promise<PyResult>;
}

/** Exactly the packages `web/src/py/solve.py`'s own imports need beyond the standard library --
 * `geopandas`, `pyproj`, `shapely`, `networkx` directly, and `numpy`/`scipy`/`pandas` that those
 * pull in (`solve.py`'s own module docstring traces the chain). Loaded through Pyodide's package
 * index (`loadPackage`), not `micropip`: these ship as part of the pinned distribution itself,
 * unlike `reblock`, which `micropip.install(wheelUrl)` fetches separately from `dist/`. */
const SOLVE_PACKAGES = ["numpy", "scipy", "pandas", "geopandas", "pyproj", "shapely", "networkx"];

/** The `pyodide` npm package's own module shape, used ONLY as a type -- `import type` erases
 * entirely at compile time, so this line adds no runtime import of "pyodide" to the bundle. The
 * real, running Pyodide module is loaded at `boot()` time from `indexUrl`, a plain string,
 * dynamically -- see `boot`'s own comment for why. Keeping the type derived from the npm package
 * rather than hand-declaring `loadPyodide`'s shape is what makes a `pyodide` version bump that
 * changes this signature a compile error here instead of a silent mismatch discovered only when
 * Task 5 boots the real thing. */
type PyodideModule = typeof import("pyodide");
type Pyodide = Awaited<ReturnType<PyodideModule["loadPyodide"]>>;

/** `PyProxy`'s own call/property signatures are declared `any` throughout the `pyodide` package's
 * types (see `node_modules/pyodide/pyodide.d.ts`) except for a handful of real methods it
 * declares concretely, `toJs`/`destroy` among them -- narrowing the untyped surface further here
 * would invent a contract the library itself does not state. */
type PyProxy = import("pyodide/ffi").PyProxy;

/** What `boot()` leaves behind for `solve()` to call: the booted runtime, the `Block` rebuilt
 * ONCE from `bundle` (design §1.6 -- a drawn road changes only two arrays this file never touches;
 * rebuilding the whole block on every edit would repeat work nothing about the road invalidates),
 * and the Python `solve` function itself, both still-live `PyProxy` handles into the running
 * interpreter. */
interface Booted {
  readonly pyodide: Pyodide;
  readonly block: PyProxy;
  readonly solveFn: (block: PyProxy, road: PyProxy, p0: number) => PyProxy;
}

/** Builds one `PyRuntime` closed over `bundle` and `wheelUrl`. Neither is read until `boot()` is
 * called -- constructing this does no work and starts no download.
 *
 * `indexUrl` defaults to the pinned CDN distribution; a caller can override it (Task 5 points the
 * equivalent boot sequence at the npm package's own local files instead, to check that pin's
 * behaviour without a network round trip on every test run). */
export function pyodideRuntime(
  bundle: AuthoringBlock, wheelUrl: string, indexUrl: string = PYODIDE_INDEX_URL,
): PyRuntime {
  let booting: Promise<void> | null = null;
  let state: Booted | null = null;

  const boot = (): Promise<void> => (booting ??= (async () => {
    // A dynamic `import()` of an absolute URL, not a static `import ... from "pyodide"`: the
    // devDependency exists so Task 5 can boot the SAME version under Node and check this pin's
    // behaviour, not so this bundle ships its own copy of Pyodide's loader -- the CDN already
    // serves that at indexUrl, and a browser's dynamic import fetches an ES module from an
    // absolute URL natively, no bundler cooperation required. esbuild cannot resolve a
    // non-literal specifier at bundle time and leaves a call shaped like this alone, which is
    // exactly what a runtime-computed CDN URL needs.
    const mod = (await import(`${indexUrl}pyodide.mjs`)) as PyodideModule;
    const pyodide = await mod.loadPyodide({ indexURL: indexUrl });
    await pyodide.loadPackage(SOLVE_PACKAGES);
    // "micropip" is itself one of Pyodide's own loadable packages (not preinstalled), and has to
    // be loaded before it can be `pyimport`ed -- the standard two-step Pyodide uses to install a
    // package that is not part of its own pinned distribution.
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(wheelUrl);
    // Runs solve.py's module body -- its imports, its TypedDicts, `block_from_bundle`, `solve` --
    // once, in Pyodide's global namespace, exactly as if this were run inside `python3
    // web/src/py/solve.py` after the packages above were on the path.
    pyodide.runPython(SOLVE_PY_SOURCE);
    const blockFromBundle: (b: PyProxy) => PyProxy = pyodide.globals.get("block_from_bundle");
    const solveFn: (block: PyProxy, road: PyProxy, p0: number) => PyProxy =
      pyodide.globals.get("solve");
    // `toPy`, not a raw JS object handed straight to the call: measured directly (see this task's
    // report) that a Pyodide `PyProxy` function called with an un-converted JS array/object
    // receives it on the Python side as an opaque `JsProxy`, not a Python list/dict -- `toPy`
    // recursively converts nested JS arrays/objects into Python lists/dicts, which is what
    // `block_from_bundle`'s dict-subscript reads (`bundle["parcel_id"]`, etc.) actually need.
    const block = blockFromBundle(pyodide.toPy(bundle));
    state = { pyodide, block, solveFn };
  })());

  return {
    boot,
    async solve(road) {
      if (state === null) {
        throw new Error("pyodideRuntime.solve: called before boot() resolved");
      }
      const { pyodide, block, solveFn } = state;
      const resultProxy = solveFn(block, pyodide.toPy(road), bundle.baseline.p0);
      // `dict_converter: Object.fromEntries` turns the outer Python dict into a plain JS object
      // rather than the default `Map`. Its two array-valued fields need no converter of their
      // own: `solve.py`'s `PyResult.potential`/`conductance` are already plain Python lists of
      // floats (`.tolist()`'d before returning), and `toJs`'s default recursion converts a list
      // of floats straight into a JS array of numbers with no leftover `PyProxy` -- measured
      // directly (this task's report) that nothing beyond `resultProxy` itself remains to
      // destroy.
      const result: PyResult = resultProxy.toJs({ dict_converter: Object.fromEntries });
      resultProxy.destroy();
      return result;
    },
  };
}
