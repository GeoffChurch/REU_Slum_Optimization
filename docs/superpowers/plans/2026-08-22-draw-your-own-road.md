# Draw-your-own-road (piece F) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reader draws an arbitrary road on the spine block and gets its real permeability, computed by `reblock.permeability` itself running in the browser under a pinned Pyodide.

**Architecture:** A committed full-precision bundle carries everything road-invariant (geometry, the footpath mesh, the no-roads baseline). A `PyRuntime` Protocol — booted lazily, injected at mount — runs `solve_egress` under Pyodide and returns the two arrays a road actually changes. The widget draws a polyline, re-solves on every settled edit, and redraws the egress graph through the renderer that already exists. A Node test boots the same pinned runtime and checks its answers against CPython's, baked into the bundle.

**Tech Stack:** Pyodide 0.29.2 (pinned CDN), TypeScript 5.9 (no framework), esbuild, Node's built-in test runner, Python 3 + geopandas/shapely/scipy for the baker, MkDocs Material.

**Spec:** `docs/superpowers/specs/2026-08-22-draw-your-own-road-design.md`

## Global Constraints

- `scripts/gen_site_pages.py` stays **stdlib-only** and must **never** import `reblock`.
- **`[project] dependencies` in `pyproject.toml` stays empty.** A dependency there breaks the browser install silently — `micropip` would try to resolve it from PyPI, where no wasm wheel exists.
- **`navigation.instant` stays off.** Two documented mechanisms fail the moment it is on; see `web/src/mount.ts`'s `DOMContentLoaded` comment and `web/src/dom/resize.ts`'s disposer.
- Generated bundles and their `.d.ts` are generated and committed, **never hand-edited**. The **wheel** is built and **never committed**.
- `docs/js/` and `docs/assets/` are gitignored — never stage them.
- No `# type: ignore`, no mypy excludes, no `eslint-disable`, no unreachable guards, no legacy shims.
- Never reach into a closed, known-at-authoring-time set with a runtime string, position or count; dynamic access over a genuinely open set has **no default**.
- Every number in generated prose is read from an artifact, never typed.
- **Comments must describe the code as it stands and must not claim more than it does.** Piece E produced fifteen findings of that shape, several arriving inside a fix for a previous one. Prefer describing the mechanism to asserting the property, and keep artifact-specific measurements in the fixture-pinned test file rather than in code a re-bake could falsify. The cheapest check: *which artifacts does this code actually read?*
- **Fault injection is the acceptance criterion.** Break what a guard guards, observe RED, restore. An injection that will not redden is **reported, not tuned**. Note that an assertion whose expected value is the system's null state — zero, empty, unchanged — is satisfied by any failure that stops the code short, so absence must be proved with a **non-throwing** injection.
- Node here is **v24.12.0**; `node --test`'s default reporter prints `ℹ pass 179`, **not** TAP's `# pass 179`. Verify any grep against a GREEN baseline first.
- `node:assert` **aborts a test body at its first failing assertion.** Name the assertion that actually threw; where you infer rather than observe, say so.
- Restore after an injection with `cp` snapshots, **never `git checkout <file>`** — a previous task wiped its own work that way.

---

## File Structure

**Create**

| File | Responsibility |
| --- | --- |
| `scripts/gen_authoring_block.py` | bakes the bundle + `.d.ts`; asserts its own round trip |
| `examples/authoring/block.json` | the block at full float64, its mesh, baseline and reference cases (committed) |
| `examples/authoring/README.md` | generated, documents the bundle from the bundle |
| `web/src/authoring.d.ts` | generated, never hand-edited |
| `web/src/py/solve.py` | bundle dict → `Block` → `solve_egress` → result dict. **Runs under CPython too**, so pytest tests it without Pyodide |
| `web/src/py/runtime.ts` | `PyRuntime`, `pyodideRuntime`, the pinned index URL |
| `web/src/widgets/draw-road.ts` | the widget and its `UrlCodec` |
| `tests/test_authoring_bundle.py` | schema, `.d.ts` agreement, round trip |
| `tests/test_solve_py.py` | `solve.py` under CPython against the bundle |
| `web/test/py-runtime.test.ts` | the seam over a fake: boot idempotence, error surfacing, road validation |
| `web/test/draw-road-boot.test.ts` | the widget over a fake runtime |
| `web/test/pyodide-parity.test.ts` | the guard: pinned Pyodide vs the baked reference |

**Modify**

| File | Change |
| --- | --- |
| `docs/_partials/permeability.md` | `<!-- DRAWROAD -->` and its framing prose |
| `scripts/gen_site_pages.py` | `_draw_road_figure` producer, `DRAWROAD` marker, `data-wheel` path rewrite, wheel asset copy |
| `web/src/mount.ts` | `register("draw-road", drawRoad, DRAW_ROAD_URL)` |
| `web/package.json` | pinned `pyodide` devDependency |
| `web/scripts/test.sh` | selects or defers the parity test by its measured cost |
| `.gitignore` | `dist/` |
| `.github/workflows/deploy-site.yml` | build the wheel, copy it into `docs/assets/` |
| `pyproject.toml` | `gen_authoring_block.py` on the mypy list |

---

### Task 1: The authoring bundle

**Files:**
- Create: `scripts/gen_authoring_block.py`, `tests/test_authoring_bundle.py`
- Generate + commit: `examples/authoring/block.json`, `examples/authoring/README.md`, `web/src/authoring.d.ts`
- Modify: `pyproject.toml` (both mypy lists)

**Interfaces:**
- Consumes: `scripts._example_block.load_example_block`, `reblock.permeability.{solve_egress, PermeabilityParams}`, `reblock.mesh.footpath_mesh`.
- Produces: `examples/authoring/block.json` matching `web/src/authoring.d.ts`, whose shape is spec §2 verbatim — `block_id`, `crs_epsg`, `boundary`, `parcel_id`, `parcels`, `streets`, `building_points`, `nodes {cx, cy, ground}`, `edges {rows, cols, footpath_g}`, `baseline {p0, potential}`, `reference []`.

**Why full precision:** spec §1.4 — cm rounding moves permeability by 4.71e-05 on this block, which Task 5's parity test cannot tolerate. Do **not** use `scripts/_bundle_io.py`'s `cm`/`sigfig`/`polygon_rings`; they exist for the render bundles and quantise by design.

- [ ] **Step 1: Write the failing test**

Create `tests/test_authoring_bundle.py`:

```python
"""The authoring bundle: the block Pyodide reconstructs, at full precision.

Full float64 and not the cm-rounded form every render bundle ships (design §1.4): quantising
moves permeability by 4.71e-05 on this block, and `web/test/pyodide-parity.test.ts` asserts the
browser reproduces CPython's number on the SAME bundle -- a tolerance that would have to be
widened past the quantisation error to accommodate rounding is a tolerance that has stopped
measuring the runtime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.dts_keys import json_keys, ts_field_names

OUT = Path("examples/authoring/block.json")
DTS = Path("web/src/authoring.d.ts")
SPINE = "ZAF.9.3.1_1_40972"


@pytest.fixture(scope="session")
def bundle() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8"))
    return result


def test_dts_declares_exactly_the_keys_the_bundle_carries(bundle: dict[str, Any]) -> None:
    assert json_keys(bundle) - ts_field_names(DTS.read_text(encoding="utf-8")) == set()
    assert ts_field_names(DTS.read_text(encoding="utf-8")) - json_keys(bundle) == set()


def test_it_is_the_spine_block_and_a_projected_crs(bundle: dict[str, Any]) -> None:
    """The same block every other stage of the site follows. 32734 is UTM 34S, projected --
    `Block.__post_init__` rejects a geographic CRS, and that failure in the browser would surface
    as a Pyodide traceback in a figcaption rather than as anything a reader could act on."""
    assert bundle["block_id"] == SPINE
    assert bundle["crs_epsg"] == 32734


def test_the_coordinates_are_not_quantised(bundle: dict[str, Any]) -> None:
    """The defect this bundle exists to avoid. A cm-rounded coordinate is an exact multiple of
    0.01, so a bundle that went through `_bundle_io.cm` would have EVERY coordinate land on that
    grid. Real UTM metres do not."""
    xs = [p[0] for ring in bundle["parcels"] for r in ring for p in r]
    on_grid = sum(1 for x in xs if math.isclose(x, round(x, 2), rel_tol=0, abs_tol=1e-12))
    assert on_grid < len(xs) // 2, (
        f"{on_grid} of {len(xs)} parcel x-coordinates sit exactly on the cm grid; this bundle "
        "must ship full float64 (design §1.4)")


def test_every_per_parcel_column_has_one_entry_per_parcel(bundle: dict[str, Any]) -> None:
    n = len(bundle["parcels"])
    assert len(bundle["parcel_id"]) == n
    assert len(bundle["nodes"]["cx"]) == n
    assert len(bundle["nodes"]["cy"]) == n
    assert len(bundle["nodes"]["ground"]) == n
    assert len(bundle["baseline"]["potential"]) == n


def test_every_per_edge_column_has_one_entry_per_edge(bundle: dict[str, Any]) -> None:
    m = len(bundle["edges"]["rows"])
    assert len(bundle["edges"]["cols"]) == m
    assert len(bundle["edges"]["footpath_g"]) == m


def test_edge_endpoints_are_valid_parcel_indices(bundle: dict[str, Any]) -> None:
    """An out-of-range index would reach `potential[rows[i]]` in the widget's own current
    calculation and read `undefined`, which arithmetic turns into NaN and canvas turns into
    nothing drawn -- silent, and shaped exactly like an empty graph."""
    n = len(bundle["parcels"])
    for key in ("rows", "cols"):
        assert all(0 <= i < n for i in bundle["edges"][key]), key


def test_building_points_are_one_per_parcel(bundle: dict[str, Any]) -> None:
    """`mesh.parcel_radii` resolves point-to-parcel by CONTAINMENT, not by index -- parcels are
    Voronoi cells of these points, so the count matches while the order does not."""
    assert len(bundle["building_points"]) == len(bundle["parcels"])


def test_the_baseline_is_the_no_roads_solve(bundle: dict[str, Any]) -> None:
    """Road-invariant (design §1.6), so it is baked rather than recomputed in the browser on
    every edit. A finite positive p0 is what `permeability` divides by."""
    assert math.isfinite(bundle["baseline"]["p0"])
    assert bundle["baseline"]["p0"] > 0.0


def test_reference_cases_are_present_and_shaped(bundle: dict[str, Any]) -> None:
    """What `web/test/pyodide-parity.test.ts` compares the browser against, so that the parity
    test needs no Python process at test time -- the same device `field.json` and `hood.json`
    already use."""
    ref = bundle["reference"]
    assert len(ref) >= 2
    for c in ref:
        assert c["name"] and len(c["road"]) >= 2
        assert all(len(p) == 2 for p in c["road"])
        assert math.isfinite(c["permeability"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run python -m pytest tests/test_authoring_bundle.py -q 2>&1 | tail -5`
Expected: collection error or failures — `examples/authoring/block.json` does not exist.

- [ ] **Step 3: Write the baker**

Create `scripts/gen_authoring_block.py`. Model its shape on `scripts/gen_region_grow.py` (constants at the top, a `main()` that writes bundle + README + `.d.ts`, a `DTS_TEMPLATE` string). Key content:

```python
SPINE = "ZAF.9.3.1_1_40972"
OUT = Path("examples/authoring")
DTS = Path("web/src/authoring.d.ts")

# Fixed roads whose CPython answers are baked for the parity test. Two, deliberately: one
# crossing the block and one short spur, so a runtime that agreed on a trivial case and not a
# real one cannot pass.
REFERENCE_ROADS: dict[str, list[tuple[float, float]]] = {...}
```

The extraction is deliberately raw — `list(geom.exterior.coords)` and friends, **no `cm`, no `sigfig`** — with a comment saying why (§1.4's 4.71e-05). Build the mesh once with `footpath_mesh(block, params, radii=parcel_radii(block, params))` and read `mesh.rows`, `mesh.cols`, `mesh.footpath_g`, `mesh.ground`; take node centroids from `block.parcels.geometry.centroid`. Solve the baseline with `solve_egress(block, None, params)`.

For each reference road, build a one-row `GeoDataFrame` with `geometry` and `width_m` (use `params.min_road_width_m`), call `solve_egress(block, roads, params)`, and record `1 - p/p0`.

**Bake-time assertion, not a test** — if the bundle cannot reproduce the number there is nothing to ship:

```python
def _block_from_bundle(bundle: AuthoringBundle) -> Block:
    """Rebuild a Block from the JSON about to be written, to prove the JSON is sufficient.

    Task 2 EXTRACTS this into `web/src/py/solve.py` and this module imports it back, so the
    browser and the baker share one reconstruction. Written here first on purpose: the baker
    cannot depend on a module that is tested against the bundle the baker has not written yet.
    """


def _assert_round_trip(bundle: AuthoringBundle, params: PermeabilityParams) -> None:
    """A bundle that cannot reproduce its own reference answers is not shippable, so this is a
    bake-time exit rather than a test -- there is nothing to commit if it fails."""
    rebuilt = _block_from_bundle(bundle)
    p0 = bundle["baseline"]["p0"]
    for case in bundle["reference"]:
        sol = solve_egress(rebuilt, _road_frame(case["road"], params, rebuilt.crs), params)
        got = 1.0 - sol.p / p0
        if got != case["permeability"]:
            raise SystemExit(
                f"round trip failed for {case['name']}: {got!r} != {case['permeability']!r}")
```

**No import from `web/src/py/` here.** That module is Task 2's and is tested against *this* bundle, so depending on it now would be circular. Task 2 lifts `_block_from_bundle` out of this file into `solve.py` and leaves this one importing it back — one reconstruction, shared, once both exist.

- [ ] **Step 4: Bake and run the tests**

Run: `pixi run python -m scripts.gen_authoring_block && pixi run python -m pytest tests/test_authoring_bundle.py -q 2>&1 | tail -5`
Expected: bundle, README and `.d.ts` written; all tests PASS.

- [ ] **Step 5: Add to the mypy gate**

Add `scripts/gen_authoring_block.py` to **both** `pyproject.toml` lists — the `typecheck-py` task's explicit file arguments **and** `[tool.mypy] files`. `tests/test_typecheck_config.py` pins the two together, so adding it to one and not the other fails.

Run: `pixi run typecheck && pixi run lint`
Expected: both clean.

- [ ] **Step 6: Fault injection**

Each: make the edit, run the named command, record the assertion that actually failed and its message, restore from a `cp` snapshot, re-verify green.
1. Route the coordinate extraction through `_bundle_io.cm` ⇒ `test_the_coordinates_are_not_quantised` must redden. Record the on-grid count.
2. Drop one entry from `nodes.ground` ⇒ the per-parcel column test must redden.
3. Add 1 to every `edges.rows` entry ⇒ the valid-indices test must redden.
4. Perturb one `reference[].permeability` by 1e-9 ⇒ the baker's **round-trip assertion** must exit non-zero. This one proves the bake-time guard, not a test.

- [ ] **Step 7: Commit**

```bash
git add scripts/gen_authoring_block.py tests/test_authoring_bundle.py examples/authoring web/src/authoring.d.ts pyproject.toml
git commit -m "feat(authoring): bake the spine block at full precision for the browser"
```

---

### Task 2: `solve.py` — the Python the runtime runs

**Files:**
- Create: `web/src/py/solve.py`, `tests/test_solve_py.py`

**Interfaces:**
- Consumes: `examples/authoring/block.json` (Task 1); `reblock.contracts.Block`, `reblock.permeability.{solve_egress, PermeabilityParams}`.
- Produces:
  ```python
  def block_from_bundle(bundle: dict) -> Block
  def solve(block: Block, road: list[list[float]], p0: float) -> dict
      # -> {"permeability": float, "roadMetres": float,
      #     "potential": list[float], "conductance": list[float]}
  ```

**The point of this task:** `solve.py` is ordinary CPython. It is tested by pytest here, with no Pyodide anywhere, so Task 5's parity test has to prove only that *the runtime* agrees — not that the logic is right.

- [ ] **Step 1: Write the failing test**

Create `tests/test_solve_py.py`:

```python
"""`web/src/py/solve.py` under CPython.

This module is the payload Pyodide runs, and it is plain Python -- so its logic is tested here,
where a failure is a stack trace rather than a browser console. `web/test/pyodide-parity.test.ts`
then has one job only: prove the RUNTIME agrees with this. Splitting it that way is what keeps
the expensive test cheap to interpret.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

# `web/src/py/` is not an importable package -- no `__init__.py`, and `web/` is not on the path
# -- because it is a directory of things that ship to a browser, not a Python package. Load it by
# path, the way `scripts/gen_site_pages.py` already loads `method_labels.py`.
def _load_solve():
    spec = importlib.util.spec_from_file_location("solve", Path("web/src/py/solve.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_solve_mod = _load_solve()
block_from_bundle = _solve_mod.block_from_bundle
solve = _solve_mod.solve

BUNDLE = json.loads(Path("examples/authoring/block.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def block():
    return block_from_bundle(BUNDLE)


def test_reconstruction_matches_the_source_block(block) -> None:
    assert block.block_id == BUNDLE["block_id"]
    assert len(block.parcels) == len(BUNDLE["parcels"])
    assert len(block.building_points) == len(BUNDLE["building_points"])


def test_every_reference_case_reproduces_its_baked_answer(block) -> None:
    """Exact equality, not a tolerance: both sides are CPython on the same float64 input, so any
    difference is a defect rather than noise. The parity test is where a tolerance belongs, and
    only if the two RUNTIMES are shown to disagree."""
    for case in BUNDLE["reference"]:
        got = solve(block, case["road"], BUNDLE["baseline"]["p0"])
        assert got["permeability"] == case["permeability"], case["name"]


def test_the_returned_arrays_are_the_shapes_the_widget_indexes(block) -> None:
    """The widget computes per-edge current as `conductance[i] * (potential[rows[i]] -
    potential[cols[i]])`, so a short array reads `undefined` there, becomes NaN, and draws
    nothing -- indistinguishable from an empty graph."""
    got = solve(block, BUNDLE["reference"][0]["road"], BUNDLE["baseline"]["p0"])
    assert len(got["potential"]) == len(BUNDLE["parcels"])
    assert len(got["conductance"]) == len(BUNDLE["edges"]["rows"])


def test_road_length_is_the_polyline_length(block) -> None:
    road = [[0.0, 0.0], [3.0, 4.0], [3.0, 14.0]]
    got = solve(block, road, BUNDLE["baseline"]["p0"])
    assert math.isclose(got["roadMetres"], 15.0, rel_tol=1e-12)


def test_a_road_that_is_a_single_point_is_refused(block) -> None:
    """Refused HERE as well as in the widget: the widget's check is for the reader, this one is
    the contract. A one-point 'polyline' reaches shapely as a degenerate LineString."""
    with pytest.raises(ValueError, match="at least two"):
        solve(block, [[0.0, 0.0]], BUNDLE["baseline"]["p0"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run python -m pytest tests/test_solve_py.py -q 2>&1 | tail -5`
Expected: FAIL — `FileNotFoundError` on `web/src/py/solve.py`, from `_load_solve`.

- [ ] **Step 3: Write `web/src/py/solve.py`**

```python
"""The Python that runs inside Pyodide.

Lives under `web/src/` because it ships to the browser like everything else here, and is plain
CPython so `tests/test_solve_py.py` can test it without a runtime. It imports `reblock` and
nothing else beyond the closure `reblock.permeability` already pulls in -- adding an import here
is adding one to the browser's install, which `web/test/pyodide-parity.test.ts` is what catches.

It calls `solve_egress`, NOT `permeability`: `permeability` runs two solves because it recomputes
the no-roads baseline every call, and that baseline is road-invariant (design §1.6) and baked.
One solve per edit is what makes re-solve-while-dragging affordable.
"""
```

`block_from_bundle` builds `GeoDataFrame`s from the bundle's raw coordinate lists with `crs=f"EPSG:{bundle['crs_epsg']}"`, a `parcel_id` column, and `building_points`. `solve` builds a one-row road `GeoDataFrame` with `width_m`, calls `solve_egress(block, roads, params)`, and returns the four fields, converting numpy arrays with `.tolist()`.

Guard the degenerate road explicitly — `if len(road) < 2: raise ValueError("a road needs at least two points")` — and note in a comment that the widget refuses it first, so this is the contract rather than the user-facing message.

- [ ] **Step 4: Run the tests**

Run: `pixi run python -m pytest tests/test_solve_py.py -q 2>&1 | tail -5`
Expected: all PASS.

- [ ] **Step 5: Fault injection**

1. Return `potential` truncated by one element ⇒ the array-shape test must redden.
2. Replace `solve_egress` with `permeability` and drop the baseline argument ⇒ the reference-case test must redden (or, if it does not, **report that** — it would mean the two paths agree numerically and the distinction is only about cost).
3. Delete the `len(road) < 2` guard ⇒ the refusal test must redden. Confirm the failure is the **assertion**, not an unrelated shapely `TypeError` from deeper down; if it is the latter, say so.

- [ ] **Step 6: Commit**

```bash
git add web/src/py/solve.py tests/test_solve_py.py
git commit -m "feat(py): the solve payload Pyodide runs, tested under CPython"
```

---

### Task 3: The runtime seam

**Files:**
- Create: `web/src/py/runtime.ts`, `web/test/py-runtime.test.ts`
- Modify: `web/package.json` (pinned `pyodide` devDependency), `.gitignore` (`dist/`)

**Interfaces:**
- Consumes: `AuthoringBlock` from `web/src/authoring.js` (Task 1).
- Produces:
  ```ts
  export const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/";
  export interface PyResult { permeability: number; roadMetres: number; potential: number[]; conductance: number[] }
  export interface PyRuntime { boot(): Promise<void>; solve(road: [number, number][]): Promise<PyResult> }
  export function pyodideRuntime(bundle: AuthoringBlock, wheelUrl: string, indexUrl?: string): PyRuntime;
  ```

**Injected, not reached for** — the widget takes a `PyRuntime` and never learns whether it is real, exactly as it takes a `StateFactory`. That is what lets Task 4's tests run with no network and no 25–35 MB download.

- [ ] **Step 1: Write the failing test**

Create `web/test/py-runtime.test.ts`. It tests the **contract** over a hand-written fake, plus the parts of `pyodideRuntime` reachable without booting:

```ts
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { PYODIDE_INDEX_URL, type PyResult, type PyRuntime } from "../src/py/runtime.js";

test("the pinned index URL is an exact version, not a floating one", () => {
  // jsDelivr serves immutable versioned paths, which is the entire stability argument: geopandas
  // was REMOVED in Pyodide 0.27 and geopandas/pyproj/shapely were all disabled in 0.28, returning
  // only in 0.29.2. A `latest` or major-only URL re-opens that.
  assert.match(PYODIDE_INDEX_URL, /\/v\d+\.\d+\.\d+\/full\/$/);
  assert.ok(PYODIDE_INDEX_URL.includes("v0.29.2"));
});

/** A runtime that records its calls and resolves with whatever it was handed. */
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npm test 2>&1 | tail -20`
Expected: FAIL — `Cannot find module '../src/py/runtime.js'`.

- [ ] **Step 3: Write `web/src/py/runtime.ts`**

`pyodideRuntime` closes over the bundle and the wheel URL. `boot()` is **idempotent by memoising its own promise** — a second call returns the first one rather than loading Pyodide twice:

```ts
  let booting: Promise<void> | null = null;
  const boot = (): Promise<void> => (booting ??= (async () => { /* load, install, run solve.py */ })());
```

Inside: `loadPyodide({ indexURL })`, `loadPackage(["numpy","scipy","pandas","geopandas","pyproj","shapely","networkx"])`, `micropip.install(wheelUrl)`, then run `solve.py`'s source and hand the bundle across once. `solve()` calls into it and returns a plain `PyResult`.

**`solve.py`'s source is inlined at build time**, not fetched: esbuild bundles `web/src/` into one file and a second network round trip for a 4 KB Python file is a failure mode with no upside. Use an esbuild text loader or a generated `.ts` string — say which you chose and why in your report.

- [ ] **Step 4: Pin the devDependency**

Add `"pyodide": "0.29.2"` to `web/package.json`'s `devDependencies` — the **same** version as `PYODIDE_INDEX_URL`, since Task 5 boots the npm package and asserts against the CDN pin's behaviour. Add `dist/` to `.gitignore`.

Run: `cd web && npm install && npm run check && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: type-check clean, all tests pass.

- [ ] **Step 5: Fault injection**

1. Change `PYODIDE_INDEX_URL` to `.../pyodide/v0.29/full/` ⇒ the exact-version test must redden.
2. Make `boot()` non-memoising (drop the `??=`) ⇒ **this will NOT redden**, because the fake counts its own boots and the real runtime is never booted here. **Report it rather than tuning it** — memoisation is proved in Task 5, where a real boot happens, and this test's own comment says so. This is the plan predicting a non-redden; confirm it.

- [ ] **Step 6: Commit**

```bash
git add web/src/py/runtime.ts web/test/py-runtime.test.ts web/package.json web/package-lock.json .gitignore
git commit -m "feat(web): the Pyodide runtime seam, pinned and injected"
```

---

### Task 4: The widget

**Files:**
- Create: `web/src/widgets/draw-road.ts`, `web/test/draw-road-boot.test.ts`
- Modify: `web/src/mount.ts`

**Interfaces:**
- Consumes: `PyRuntime`/`PyResult` (Task 3), `AuthoringBlock` (Task 1), `Widget<T>`/`register`/`UrlCodec` (piece E), `render/canvas.ts`'s `draw`, `view/transform.ts`'s `fitBbox`/`toScreen`/`toWorld`, `dom/{attrs,error,fallback,resize}.ts`.
- Produces: `export const drawRoad: Widget<DrawRoadState>`, `export interface DrawRoadState { road: [number, number][] }`, `export const DRAW_ROAD_URL: UrlCodec<DrawRoadState>`.

- [ ] **Step 1: Write the failing test**

Create `web/test/draw-road-boot.test.ts`, following `web/test/screen-map-boot.test.ts`'s shape: `installStubs()`, a `mount(width, drawFailure, search)` helper that fetches the real committed `examples/authoring/block.json`, injects a **fake** `PyRuntime`, and drives `fireResize`. Tests:

```ts
test("nothing is solved until the reader boots the runtime", async () => {
  const { runtime } = await mount();
  assert.equal(runtime.boots, 0, "a prose page must not pay 25-35 MB for being read");
});

test("clicking the boot control boots exactly once, however many times it is pressed", async () => {
  const { host, runtime } = await mount();
  bootButton(host).fire("click");
  bootButton(host).fire("click");
  await settle();
  assert.equal(runtime.boots, 1);
});

test("a two-point road is solved and its permeability reported", async () => {
  const { host, runtime, readout } = await mount();
  await boot(host);
  drawRoad(host, [[10, 10], [60, 60]]);
  await settle();
  assert.deepEqual(runtime.roads.at(-1), [[10, 10], [60, 60]]);
  assert.match(readout.textContent, /permeability/i);
});

test("a one-point road is refused before it reaches the runtime", async () => {
  const { host, runtime } = await mount();
  await boot(host);
  drawRoad(host, [[10, 10]]);
  await settle();
  assert.equal(runtime.roads.length, 0, "the widget must not hand a degenerate road to Python");
  assert.match(captionText(host), /two points/i);
});

test("a runtime failure keeps the last good picture and says what failed", async () => {
  const { host } = await mount({ solveThrows: "boom" });
  await boot(host);
  drawRoad(host, [[10, 10], [60, 60]]);
  await settle();
  assert.match(captionText(host), /boom/);
  assert.ok(canvasOf(host), "the canvas is still there; a failed solve is not a blank figure");
});

test("the drawn road survives a URL round trip", async () => {
  const { store, loc } = await mount({ search: "road=10,10,60,60" });
  assert.deepEqual(store.get().road, [[10, 10], [60, 60]]);
  void loc;
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: the six new tests fail; everything else passes.

- [ ] **Step 3: Write the widget**

Structure it on `web/src/widgets/screen-map.ts`: `requireAttr` for `data-bundle` and `data-wheel`, a `fetch(...).then(boot).catch(showWidgetError)` chain, `observeSize`, a `frameScheduled` single-frame coalesce, `removeFallbackImage` only after the first successful draw.

**The boot control states its cost.** This sits on a prose page, so the button reads as an explicit opt-in rather than a decoration, and says what pressing it downloads.

**Drawing:** `pointerdown` on the canvas appends `toWorld(view, ev.offsetX, ev.offsetY)`; `pointermove` previews the pending segment; `dblclick` or `Enter` settles; `Escape` clears. Each settled edit calls `runtime.solve(state.road)`.

**Redraw:** synthesise the `Bundle` shape `render/canvas.ts`'s `draw` already consumes, from the authoring bundle's road-invariant halves plus the result's two arrays, with `prefix: 0`. Per-edge current is `conductance[i] * (potential[rows[i]] - potential[cols[i]])` — the same expression `gen_web_bundle.py` bakes, and a comment should say so rather than re-deriving the reasoning.

**Codec:** this needs a param that does not exist yet. `web/src/url/param.ts` ships `roadsParam`, which is specific to DisplacementField's two fixed two-point segments; an arbitrary polyline is a different shape. **Add `polylineParam(key: string): Param<[number, number][]>`** to that file, beside `roadsParam` and reusing its `COORD_DP` and `fmtCoord`:

* encodes a flat `x1,y1,x2,y2,…` list at 0.1 m — the same precision `roadsParam` already uses, so one file does not carry two coordinate spellings;
* decodes `null` (which makes the store drop the key and fall back to the initial) on an odd coordinate count, any non-finite value, or fewer than two points;
* `same` compares element-wise, since `===` on an array would rewrite the URL on every render.

Then `DRAW_ROAD_URL` is `{ road: polylineParam("road") }`. Add its round-trip and rejection cases to `web/test/url-param.test.ts` alongside `roadsParam`'s, and fault-inject each rejection.

- [ ] **Step 4: Register it**

In `web/src/mount.ts`, after `REGISTRY` exists, following the five existing registrations **and keeping their comments** — they record the module-evaluation cycle that once made the whole bundle throw:

```ts
// Sixth widget, same shape, same reason -- registered HERE, after REGISTRY exists, never from
// inside the widget module (see the paragraph above).
import { drawRoad, DRAW_ROAD_URL } from "./widgets/draw-road.js";
register("draw-road", drawRoad, DRAW_ROAD_URL);
```

Then extend `web/test/mount.test.ts`'s pinned key-list test with `draw-road`'s keys — a URL-key rename is not a type error and would silently break published links.

- [ ] **Step 5: Run the tests**

Run: `cd web && npm run check && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: type-check clean, `fail 0`.

- [ ] **Step 6: Fault injection**

1. Boot the runtime at mount instead of on click ⇒ the no-solve-until-boot test must redden.
2. Drop the `??=` memoisation in the widget's boot handler ⇒ the boot-once test must redden.
3. Remove the `< 2` road check ⇒ the refusal test must redden. Verify the failure is the **assertion** and not an exception from the runtime fake.
4. Make the solve-failure path clear the canvas ⇒ the last-good-picture test must redden. Note this is an **absence-shaped** assertion in one half (`canvasOf` present), so confirm the redness comes from the caption half too, and use a non-throwing injection for the half that asserts presence.

- [ ] **Step 7: Commit**

```bash
git add web/src/widgets/draw-road.ts web/test/draw-road-boot.test.ts web/src/mount.ts web/test/mount.test.ts
git commit -m "feat(web): draw a road, solve it under Pyodide, redraw the graph"
```

---

### Task 5: The parity guard

**Files:**
- Create: `web/test/pyodide-parity.test.ts`
- Modify: `web/scripts/test.sh`

**Interfaces:**
- Consumes: the pinned `pyodide` npm package (Task 3), `examples/authoring/block.json`'s `reference` (Task 1), `web/src/py/solve.py` (Task 2), the built wheel.

**What this proves that nothing else can** — spec §5: a package that left the distribution, the pandas **2.3.3 vs 3.0.3** major skew, and behavioural drift at call time including anything reached only through a deferred import like `mesh.parcel_radii`'s `reblock.budget`. A static import scan sees none of the three.

- [ ] **Step 1: Build the wheel and write the failing test**

```bash
pixi run pip wheel --no-deps --wheel-dir dist .
```
Expected: `dist/reblock-0.1.0-py3-none-any.whl`, ~220 KB, `py3-none-any`.

Create `web/test/pyodide-parity.test.ts`: `loadPyodide({ indexURL })` against the **npm package's own** distribution, `loadPackage` the seven, `micropip.install` the `file://` wheel from `dist/`, run `solve.py`, and for each `reference` case assert the returned permeability equals the baked one.

```ts
test("the pinned runtime reproduces CPython's answer for every reference road", async () => {
  const py = await bootPyodide();
  for (const c of BUNDLE.reference) {
    const got = solveIn(py, c.road);
    // Exact equality is the claim until a runtime is SHOWN to differ. If this ever fails by
    // float noise, the finding is the difference and its magnitude -- record it and switch to a
    // stated absolute tolerance. A tolerance widened until the test passes has stopped measuring
    // the runtime, which is the only thing this test exists to measure.
    assert.equal(got.permeability, c.permeability, c.name);
  }
});

test("the runtime's pandas is the one the browser will get, not this checkout's", async () => {
  // The skew is real and pinning cannot remove it (design §1.1): the browser runs pandas 2.3.3
  // whatever we pin, while CI runs 3.0.3. This asserts the skew EXISTS rather than pretending it
  // does not, so that the day the two converge someone reads this line and deletes it.
  const py = await bootPyodide();
  assert.match(py.runPython("import pandas; pandas.__version__"), /^2\./);
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npx tsc -p tsconfig.test.json --outDir /tmp/f5 --noEmit false && node --test /tmp/f5/test/pyodide-parity.test.js 2>&1 | tail -20`
Expected: FAIL — the module, the wheel, or the runtime is missing.

- [ ] **Step 3: Make it pass, and measure its cost**

Implement the boot helper. **Record the wall-clock time of one full boot** — the plan needs the number, not an impression.

- [ ] **Step 4: Wire it into the gate by its measured cost**

If a boot is **under 90 s**, run it in `web/scripts/test.sh` with everything else. If it is over, give it its own npm script and its own CI job, and **say so in `test.sh`'s comment with the measured number** — the same way this repo's slow Python tests document their own deselection. Do not silently deselect it.

Run: `cd web && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: `fail 0`.

- [ ] **Step 5: Fault injection**

1. Perturb one `reference[].permeability` in the committed bundle by 1e-12 ⇒ the parity test must redden. Record the reported difference — that number is the runtimes' real agreement, and it belongs in your report.
2. Remove `"geopandas"` from the `loadPackage` list ⇒ boot must fail with an import error naming geopandas, **not** a silent zero. This is the disappearance the pin exists to survive.
3. Assert `/^3\./` in the pandas test ⇒ it must redden, confirming the skew assertion reads the runtime rather than the host.

- [ ] **Step 6: Commit**

```bash
git add web/test/pyodide-parity.test.ts web/scripts/test.sh
git commit -m "test(web): boot the pinned Pyodide and check it against CPython"
```

---

### Task 6: The page

**Files:**
- Modify: `docs/_partials/permeability.md`, `scripts/gen_site_pages.py`, `.github/workflows/deploy-site.yml`
- Test: `tests/test_gen_site_pages.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `<!-- DRAWROAD -->` filled on `docs/methodology/permeability.md`, served at `url_depth=2`.

- [ ] **Step 1: Write the failing test**

In `tests/test_gen_site_pages.py`, using the file's existing `render_page(name)` and `<name>_body` fixture patterns:

```python
def test_the_permeability_page_carries_the_draw_road_widget(permeability_body: str) -> None:
    assert permeability_body.count('data-widget="draw-road"') == 1
    assert 'data-bundle="assets/authoring/block.json"' in permeability_body
    assert 'data-wheel="assets/wheels/' in permeability_body


def test_the_draw_road_paths_are_rewritten_for_the_served_depth() -> None:
    """permeability.md serves at <base>/methodology/permeability/ -- url_depth 2. `data-wheel` is
    a fetch URL exactly as `data-bundle` is, so it needs its own `.replace()` in `_write_page`:
    `data-bundle="assets/` is not a substring of `data-wheel="assets/`, which is the same trap
    ScreenMap's two attributes sprang."""
    page = render_page("permeability")
    assert 'data-bundle="../../assets/authoring/block.json"' in page
    assert 'data-wheel="../../assets/wheels/' in page
    for attr in ("data-bundle", "data-wheel"):
        assert f'{attr}="assets/' not in page, attr
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run python -m pytest tests/test_gen_site_pages.py -q 2>&1 | tail -5`
Expected: 2 failures.

- [ ] **Step 3: Add the producer and the marker**

`_draw_road_figure()` in `scripts/gen_site_pages.py`: copy `examples/authoring/block.json` as an asset, locate the built wheel under `dist/`, copy it into `docs/assets/wheels/`, and emit the mount point through `_figure(...)` with `data-widget`, `data-bundle`, `data-wheel`. Every number in the caption is **read from the bundle** — the parcel count, the block id — never typed. Register `"DRAWROAD": _draw_road_figure` in `MARKERS`.

Add `data-wheel` to `_write_page`'s `url_depth` rewrites, beside the three `data-bundle*` lines.

**Stdlib only.** Locating the wheel is `sorted(Path("dist").glob("reblock-*.whl"))`; if it is absent, raise with a message naming the build command, the same shape as `_assert_widget_bundle_present`.

- [ ] **Step 4: Write the partial's prose**

Add `<!-- DRAWROAD -->` to `docs/_partials/permeability.md` after the existing `<!-- PERMGRAPHFIGS -->`, with two or three sentences of framing. **No numerals** — every figure on the page carries an artifact-read caption, and a typed number is the drift class this site has a documented history of. Say plainly that this one computes rather than looks up, and that pressing the button downloads a Python runtime.

- [ ] **Step 5: Wire the deploy**

In `.github/workflows/deploy-site.yml`, before `gen_site_pages.py`, build the wheel:
```yaml
      - name: Build the reblock wheel for Pyodide
        run: pixi run pip wheel --no-deps --wheel-dir dist .
```

- [ ] **Step 6: Run everything**

```bash
pixi run python scripts/gen_site_pages.py
pixi run python -m pytest tests/test_gen_site_pages.py -q
pixi run lint && pixi run typecheck
~/.cache/rattler/cache/cached-envs-v0/4937c48afb8986c1/bin/mkdocs build --strict --site-dir "$(mktemp -d)"
```
Expected: all clean, `mkdocs` exit 0 with no WARNING. Confirm the built `methodology/permeability/index.html` carries `data-widget="draw-road"` with both paths resolving two directories up.

- [ ] **Step 7: Fault injection**

1. Delete the `data-wheel` line from `_write_page` ⇒ the path-rewrite test must redden.
2. Remove `DRAWROAD` from `MARKERS` ⇒ the existing both-directions marker test must redden (an orphan marker).
3. Rename the wheel in `dist/` ⇒ the producer must **raise** with the build command in its message, not emit a broken path.

- [ ] **Step 8: Commit**

```bash
git add docs/_partials/permeability.md scripts/gen_site_pages.py tests/test_gen_site_pages.py .github/workflows/deploy-site.yml
git commit -m "feat(site): draw-your-own-road on the Permeability page"
```
