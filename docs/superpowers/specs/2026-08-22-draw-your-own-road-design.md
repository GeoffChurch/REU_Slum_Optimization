# Draw-your-own-road (site redesign, piece F) — design

**Parent design:** `docs/superpowers/specs/2026-08-13-site-redesign-design.md` (§2 *Pyodide: verified
feasible, with pins*; §7 piece F).

**Predecessors:** C (`2026-08-15-web-bundle-and-widget-substrate-design.md`), D1–D3, and E
(`2026-08-21-explore-chain-and-url-state-design.md`).

Piece F is the last piece of the redesign and the only one that is pure upside — everything else is
complete without it. It lets a reader draw an arbitrary road on the site's spine block and get its
**real permeability**, computed by the actual `reblock.permeability` running in the browser under
Pyodide. It is the one widget that ever boots Pyodide, and the only place on the site where a number
is computed rather than looked up.

---

## §1 Measurements

Every figure below was measured on this checkout, against the pinned distribution, today.

### §1.1 The runtime is available and the pin is real

`https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide-lock.json` carries every package the closure
needs:

| package | Pyodide 0.29.2 | this checkout | skew |
| --- | --- | --- | --- |
| numpy | 2.2.5 | 2.5.0 | minor |
| scipy | 1.14.1 | 1.18.0 | minor |
| **pandas** | **2.3.3** | **3.0.3** | **MAJOR** |
| geopandas | 1.1.1 | 1.1.4 | minor |
| pyproj | 3.7.2 | 3.7.2 | none |
| shapely | 2.0.7 | 2.1.2 | minor |
| networkx | 3.4.2 | 3.6.1 | minor |
| **pyarrow** | **22.0.0** | **24.0.0** | **MAJOR** |

**The pin is genuinely immutable.** `v0.28.0/full/pyodide-lock.json` and
`v0.29.2/full/pyodide-lock.json` have different md5s, so the versioned paths serve distinct builds.
Note that v0.29.2's lockfile reports `"version": "0.28.0.dev0"` in its own `info` block — that is
stale metadata inside the release, **not** evidence the pin is broken, and it should not be
"corrected" by chasing a different URL.

**The pandas skew is real and pinning cannot fix it.** The browser will run `permeability` under
pandas 2.3.3 whatever we pin, while CI runs it under 3.0.3. Anything that depends on pandas-3-only
behaviour passes every test here and dies in the reader's browser. §5's guard exists to make that
loud rather than silent; it is not a hypothetical.

### §1.2 Two of the parent spec's feasibility claims have gone stale

Both are recorded because they shaped §5, not to score points against a nine-day-old document.

**"No `pyarrow`" is false.** `pandas` 3.0 imports it eagerly, so the import closure of
`reblock.permeability` on this checkout is: `certifi`, `charset_normalizer`, `cython_runtime`,
`dateutil`, `geopandas`, `networkx`, `numpy`, `packaging`, `pandas`, `pyarrow`, `pyproj`, `scipy`,
`shapely`, `six`. It does not block anything — `pyarrow` 22.0.0 ships in the pinned distribution —
but **nobody touched `reblock` to cause it.** The parent spec framed the guard as protecting against
"if someone later adds an import to it"; the drift arrived through a dependency upgrade, in a
package the closure already had.

**"`budget` is `TYPE_CHECKING` only" is false at runtime.** `mesh.parcel_radii` does
`from reblock.budget import building_radii` **inside the function body**, deliberately, to break a
`budget`↔`permeability` import cycle. `reblock.budget` therefore enters the closure when
`permeability` is *called*, never when it is imported. Measured: it drags in no new third-party
top-level package (it imports only `networkx`, `numpy`, `pandas`, `shapely`, `geopandas`, `scipy`,
all already present) — but a guard that measured the **import** closure would not have seen it at
all, and could not have known that.

Together these set §5's shape: **the guard runs the function in the target runtime**, because the
two things that actually drifted are invisible to a static import scan.

### §1.3 A solve is 0.06 s, so the road can re-solve while it is dragged

On the spine block `ZAF.9.3.1_1_40972` (263 parcels, 1 street, 263 building points, EPSG:32734):

| call | time |
| --- | --- |
| `parcel_radii` | < 0.01 s |
| `permeability(block, None)` | 0.06 s |
| `permeability(block, roads)` — 20 road rows | 0.07 s |

CPython. WASM is slower, but a 10× penalty still leaves a solve under a second, so the design does
**not** need a solve-on-release interaction, a spinner, or a debounce longer than one animation
frame. Re-solve on drag, and let the number move with the road.

**What this measurement does not capture, found during Task 4.** `pyodideRuntime.solve` is `async`
but contains no `await`: the call into Pyodide runs synchronously and **blocks the main thread**. So
for the duration of each solve the drag's own pointer events do not dispatch, and the gesture will
feel *stepped* rather than merely slow — a quality this number cannot predict and no test in this
repo can observe, since the harness has no real runtime and no real pointer. It is the strongest
argument for moving the solve to a worker, which §4's in-flight guard is already written against.
Unmeasured, and recorded as such rather than assumed away.

### §1.4 cm rounding moves the answer, so the render bundle cannot be reused

The bundles piece C established ship coordinates as **cm-rounded, origin-relative** metres, which is
right for drawing. On this block, computing permeability against that quantisation instead of the
exact geometry:

| geometry | permeability |
| --- | --- |
| exact | 0.9508990541 |
| cm-rounded | 0.9508519775 |
| difference | **4.71 × 10⁻⁵** |

Invisible on screen and far too large for §5's guard to assert equality across. So piece F ships its
**own** bundle at full float64 precision rather than extending `perm-graph/bundle.json`. The two
bundles answer different questions and should not be made to answer both.

### §1.5 What a `Block` actually requires

`Block.__post_init__` (`src/reblock/contracts.py:52`) enforces a projected CRS, a non-empty
`parcels` carrying `parcel_id` and `geometry`, and a `streets` carrying `geometry`. Beyond that,
`mesh.parcel_radii` reads `block.building_points` and resolves one point per parcel **by
containment** — parcels are Voronoi cells of the building points, but *not* in index order. So the
authoring bundle must carry the building points; the render bundle has no use for them and does not.

### §1.6 The mesh is road-invariant, so most of the graph can be baked

`solve_egress` (`src/reblock/permeability.py:292`) builds `footpath_mesh(block, params, ...)` — which
takes **no `roads` argument** — and then computes `edge_conductances(..., roads, params)`. Grounding
comes from `mesh.ground`, parcels within `STREET_TOL` of a *street*, and is likewise untouched by
roads. So a drawn road changes exactly two arrays and nothing else:

| quantity | depends on the road? | where it comes from |
| --- | --- | --- |
| parcel centroids, edge `rows`/`cols`, `ground`, `footpath_g` | **no** | baked once into the bundle |
| per-edge `conductance` | **yes** | returned by the runtime |
| per-parcel `potential` | **yes** | returned by the runtime |
| the no-roads baseline `p0` | **no** | baked once into the bundle |

Three consequences the design takes: the runtime returns two arrays rather than a mesh; the widget
computes per-edge current as `conductance[i] * (potential[rows[i]] - potential[cols[i])`) in JS,
exactly as `gen_web_bundle.py` already bakes it; and `permeability` need not be called at all —
`solve_egress` returns `EgressSolution(p, potential, mesh, conductance)` in **one** solve, where
`permeability()` would run two (it recomputes the road-invariant baseline every call).

---

## §2 The authoring bundle — `examples/authoring/block.json`

Baked by `scripts/gen_authoring_block.py`, committed, with a generated `web/src/authoring.d.ts`
beside it exactly as every other bundle on this site.

```ts
export interface AuthoringBlock {
  block_id: string;
  /** Projected, and asserted so by the baker: `Block.__post_init__` rejects a geographic CRS,
   * and a browser-side failure there would surface as a Pyodide traceback in a figcaption. */
  crs_epsg: number;
  /** FULL float64, NOT the cm-rounded form the render bundles ship (§1.4): quantising moves
   * permeability by 4.71e-05 on this block, which is more than §5's guard can tolerate. */
  boundary: [number, number][][];
  parcel_id: string[];
  parcels: [number, number][][][];
  streets: [number, number][][];
  /** One per parcel, but NOT in parcel order -- `parcel_radii` resolves the correspondence by
   * containment, and this bundle preserves whatever order the source had rather than inventing
   * one the Python does not rely on. */
  building_points: [number, number][];
  /** CPython's own answer for fixed roads, baked here so §5's parity test needs no Python at test
   * time -- the same shape `field.json`'s `reference` and `hood.json`'s `reference` already use,
   * for the same reason. */
  reference: { name: string; road: [number, number][]; permeability: number }[];

  /** The road-INVARIANT half of the graph (§1.6), baked once so the runtime returns two arrays
   * rather than a mesh. Parcel centroids and the footpath edge list do not move when a road is
   * drawn; only the conductances on those edges do. */
  nodes: { cx: number[]; cy: number[]; ground: boolean[] };
  edges: { rows: number[]; cols: number[]; footpath_g: number[] };
  /** `solve_egress(block, None)`'s own answer: the no-roads baseline permeability divides against,
   * and the potentials the "before" picture shows. Road-invariant, so computing it in the browser
   * on every edit would be waste. */
  baseline: { p0: number; potential: number[] };
}
```

Absolute coordinates, no origin offset: the origin trick exists to shorten cm-rounded strings, and
at full precision it buys nothing while adding a subtraction the browser would have to undo before
handing the numbers to shapely.

The baker asserts the round trip it is responsible for: **`permeability` computed on the reconstructed
block equals `permeability` on the source block exactly.** That is a bake-time assertion, not a test
— if the bundle cannot reproduce the number, there is nothing to ship.

---

## §3 The runtime seam — `web/src/py/`

```ts
export interface PyResult {
  /** 1 - P(road)/p0, against the bundle's baked `baseline.p0` -- the same definition
   * `permeability()` uses, computed from ONE solve rather than two (§1.6). */
  permeability: number;
  roadMetres: number;
  /** Per parcel, in `nodes` order. */
  potential: number[];
  /** Per edge, in `edges` order. The widget derives per-edge current from this and `potential`. */
  conductance: number[];
}

export interface PyRuntime {
  /** Idempotent. Resolves when `permeability` is callable. */
  boot(): Promise<void>;
  /** `road` is a polyline in the bundle's own projected metres. */
  solve(road: [number, number][]): Promise<PyResult>;
}

export function pyodideRuntime(bundle: AuthoringBlock, wheelUrl: string, indexUrl?: string): PyRuntime;
```

Built once at mount and **injected**, the way `StateFactory` is (piece C/E) — the widget never learns
whether it holds a real Pyodide or a fake, which is what makes it testable in `web/test/harness.ts`
with no network and no 25–35 MB download.

**Pinned:** `indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/"`, verified immutable in
§1.1. Self-hosting the wheels is **not** proposed: pinning solves the stability problem, and
self-hosting buys only the removal of third-party requests from readers' browsers. Revisit if SBU
policy requires that; nothing else changes.

**`[project] dependencies = []` must stay empty.** Everything heavy sits under
`[tool.pixi.dependencies]`, which is what lets the hatchling wheel install through `micropip`
without triggering dependency resolution. A dependency added there breaks the browser install and
nothing in the Python test suite would notice.

**How `reblock` reaches the browser.** `pixi run pip wheel --no-deps --wheel-dir dist .` produces
`reblock-0.1.0-py3-none-any.whl` — **220 KB, pure Python**, measured — which `micropip` installs
after the runtime's own packages. `--no-deps` is belt-and-braces on top of the empty
`[project] dependencies`: the wheel must never trigger resolution, because every scientific package
it would resolve is already provided by the distribution and a resolver would try to fetch them from
PyPI, where no wasm wheel exists.

The wheel is **built, never committed** — it is a compiled artifact of `src/`, and a stale committed
copy would let the browser run different code from CI while both looked healthy. Two consumers, one
build:

* `web/test/pyodide-parity.test.ts` loads it from `dist/` through Node's filesystem.
* `deploy-site.yml` builds it and copies it into `docs/assets/`, and the mount point carries the
  resulting path as `data-wheel`, rewritten by `_write_page` exactly as `data-bundle` is.

**Boot is lazy and its cost is stated.** Piece F sits on the Permeability page, a prose page a reader
may reach without ever intending to compute anything, so the button that boots Pyodide says what it
will download. Nothing is fetched until it is pressed.

---

## §4 The widget — `web/src/widgets/draw-road.ts`

Registered as `draw-road`, mounted from `<!-- DRAWROAD -->` on `docs/_partials/permeability.md`,
directly below the four-panel egress grid the reader has just been shown.

**Drawing.** Click places a vertex; the polyline follows the pointer; double-click or `Enter` closes
it; `Escape` clears. Dragging an existing vertex moves it. This is a polyline, not the two fixed
segments `DisplacementField` drags — an arbitrary road is the entire point, and it is why no baked
prefix table can answer the question.

**Solving.** Every settled edit calls `runtime.solve(road)`. At §1.3's timings that is sub-second, so
the readout and the picture both move with the road. Two throttles, not one: a single-frame coalesce
(the `frameScheduled` guard `screen-map.ts` already uses), **and** an in-flight guard that keeps at
most one solve outstanding and remembers only the newest road.

(This said the coalesce "is the only throttling needed" until Task 4 shipped the drag. The coalesce
alone is sufficient against *today's* runtime, whose `solve` blocks the main thread and therefore
cannot be re-entered — but `PyRuntime` promises no ordering, and the obvious remedy for a blocking
sub-second solve is to move it to a worker, which would both reorder answers and let a drag pile up
roughly sixty solves a second behind the coalesce. The guard is written against the interface rather
than against the one implementation that happens to exist.)

**Reporting.** Permeability for the drawn road, its length in metres, and the egress graph redrawn
from the returned `potential` and `conductance` through the existing `render/canvas.ts` — the same
picture the Permeability page teaches, for a road the reader invented. **Pyodide returns numbers,
never pixels.**

(This said "permeability **before and after**" until Task 4's review. There is no "before" to show:
permeability is `1 − P(roads)/P(no roads)`, so the no-road value is **definitionally zero** and
printing it tells a reader nothing. Corrected here rather than implemented, because a spec asking
for a constant is a spec to fix — unlike vertex dragging, also missing from the same section, which
is a real capability this design measured for in §1.3 and which Task 4 implemented.)

**State.** `DrawRoadState { road: [number, number][] }`, with a `UrlCodec` under piece E's contract
so a drawn road is citable. Coordinates at 0.1 m, matching `roadsParam`'s existing precision, and the
codec validates arity and finiteness before anything reaches the runtime.

---

## §5 The CI guard — run the function in the target runtime

`web/test/pyodide-parity.test.ts` boots the **pinned** Pyodide under Node, installs the reblock
wheel through `micropip`, runs `permeability` on `examples/authoring/block.json` for each of the
bundle's `reference` roads, and asserts it matches the baked CPython answer.

**It is a Node test, not a pytest, and that is forced rather than chosen:** Pyodide is a WebAssembly
build of CPython that runs inside a JavaScript runtime, so nothing in `pytest` can host it. Baking
CPython's answers into the bundle (§2) is what lets the comparison happen without a Python process
at test time — and it is the same device `field.json` and `hood.json` already use to pin their own
browser models against production.

**Both sides read the same JSON**, so any disagreement is the runtime and not the quantiser — which
is why §1.4's bundle exists at full precision. The baker asserts its `reference` numbers against the
live `permeability` as it writes them, so a stale reference cannot survive a re-bake.

This catches, in one test, all three things a static import scan cannot:

1. **A package that left the distribution.** **Measured across every release** (`pyodide-lock.json` at each pinned path): `geopandas` and
   `pyproj` are present in 0.26.4, 0.27.0 and 0.27.7, **absent in 0.28.0**, and present again in
   0.29.2. `shapely` is present in every one of them. So the disappearance is **two packages in one
   release**, not three across two — and it is still the reason to pin, because the two that vanished
   are load-bearing here.
   (An earlier draft of this section said `geopandas` was removed in 0.27 and all three were
   disabled in 0.28; Task 3 measured it and neither half was right. The claim came from the parent
   design and was repeated here without checking.)
2. **The pandas major skew** (§1.1). The browser runs pandas 2.3.3; CI runs 3.0.3.
3. **Behavioural drift at call time**, including anything reached only through a deferred import
   like `mesh.parcel_radii`'s `reblock.budget` (§1.2).

**Tolerance is stated, not discovered.** The assertion is exact equality if the runtimes agree
bit-for-bit, and otherwise a documented absolute tolerance with the observed difference recorded
beside it — never a tolerance widened until the test passes. If the two runtimes disagree beyond
float noise, that is the finding, and it is reported rather than tuned away.

**Measured, Task 5: they do not agree exactly.** `crossing` matches to the last bit; **`spur` differs
by 2.220446e-16 — four units in the last place** (Pyodide `0.3637809513169823` against the baked
`0.3637809513169825`), deterministic across runs. CPython on this machine reproduces both baked
numbers exactly, so the difference is arithmetic in the WASM build rather than a stale bake. The
shipped assertion is therefore a stated absolute **1e-13**, chosen from three anchors rather than
from what made the test pass: ~450× the observed disagreement, ~1.7e10× **below** the 1.66e-03 that
centimetre-rounding costs on these roads (§1.4), and strictly below the 1e-12 that the tolerance's
own fault injection perturbs by — which still reddens, with 10× margin.

This is the result the guard exists to produce. A design that had assumed bit-identity would have
been wrong, and a tolerance picked after seeing a failure would have hidden that.

**Cost.** One Pyodide boot in CI, measured and recorded in the plan. If it exceeds the budget the
plan sets, it becomes a separate slower job rather than being deleted.

---

## §6 Failure modes

Every one of them lands on the committed PNG, which is what the fallback exists for:

* **No JS, print, a reader who never clicks** — the static egress grid, exactly today's page.
* **The CDN is unreachable or the pin 404s** — `boot()` rejects, and the failure renders in the
  figcaption through `dom/error.ts`'s `showWidgetError`, naming the URL that failed.
* **Pyodide throws** — a Python traceback is not a reader-facing message, so the widget reports that
  the computation failed and keeps the last good picture rather than clearing to blank.
* **A road that degenerates** (fewer than two distinct vertices, or a zero-length segment) is refused
  in the widget before it reaches the runtime, with the reason shown.

---

## §7 Out of scope

* **Self-hosting the Pyodide wheels** (§3).
* **A JS reimplementation of permeability.** Avoiding a second implementation is the entire reason
  this piece boots a Python runtime; the parent design settled it and nothing here reopens it.
* **Arbitrary block selection.** One block, the site's spine, consistent with every other stage.
* **Editing anything but the road** — parcels, streets and buildings are fixed input.
* **A `Scorer` Protocol shared with `Frontier`.** The parent design's §2 rules this out explicitly:
  `PrefixTable` answers "what does prefix m of method X score?" instantly from a lookup, `Scorer`
  answers "what does this arbitrary road set score?" in seconds through Pyodide. One interface over
  both produces a `TableScorer` that throws on inputs its type claims to accept.

---

## §8 File structure

**New**

| File | Responsibility |
| --- | --- |
| `scripts/gen_authoring_block.py` | bakes `examples/authoring/block.json` + `web/src/authoring.d.ts`; asserts the round trip |
| `examples/authoring/block.json` | the block at full float64 (committed) |
| `web/src/authoring.d.ts` | generated, never hand-edited |
| `web/src/py/runtime.ts` | `PyRuntime`, `pyodideRuntime`, the pinned index URL |
| `web/src/py/solve.py` | the Python the runtime runs: bundle → `Block` → `permeability` |
| `web/src/widgets/draw-road.ts` | the widget |
| `web/test/draw-road-boot.test.ts` | boot tests over a fake `PyRuntime` |
| `web/test/py-runtime.test.ts` | the seam: boot idempotence, error surfacing, road validation |
| `web/test/pyodide-parity.test.ts` | §5's guard: pinned Pyodide under Node vs the bundle's baked reference |
| `tests/test_authoring_bundle.py` | schema, `.d.ts` agreement, the round-trip assertion |

**Modified**

| File | Change |
| --- | --- |
| `docs/_partials/permeability.md` | the `<!-- DRAWROAD -->` marker and its framing prose |
| `scripts/gen_site_pages.py` | the `DRAWROAD` producer and marker |
| `web/src/mount.ts` | `register("draw-road", drawRoad, DRAW_ROAD_URL)` |
| `web/package.json` | the pinned `pyodide` devDependency |
| `.gitignore` | `dist/` — the wheel is built, never committed |
| `web/scripts/test.sh` | how the parity test is selected or deselected, if it needs its own job |
| `.github/workflows/ci.yml` | the Pyodide job, if §5's measured cost demands a separate one |

---

## §9 Global constraints (binding on every task)

* `scripts/gen_site_pages.py` stays **stdlib-only** and must **never** import `reblock`.
* **`[project] dependencies` stays empty** (§3) — a dependency there breaks the browser install
  silently.
* **`navigation.instant` stays off.** Two documented mechanisms fail the moment it is on; see
  `web/src/mount.ts`'s `DOMContentLoaded` comment and `web/src/dom/resize.ts`'s disposer.
* Generated bundles and their `.d.ts` are generated and committed, **never hand-edited**.
* `docs/js/` and `docs/assets/` are gitignored — never stage them.
* No `# type: ignore`, no mypy excludes, no `eslint-disable`, no unreachable guards, no
  legacy-compatibility shims.
* Never reach into a closed, known-at-authoring-time set with a runtime string, position or count;
  dynamic access over a genuinely open set has **no default**.
* Every number in generated prose is read from an artifact, never typed.
* **Comments must describe the code as it stands and must not claim more than it does.** Piece E
  produced fifteen findings of exactly that shape, several arriving inside a fix for a previous one,
  and §1.2 records two more in this piece's own parent spec. Before writing that something is
  guaranteed, checked or prevented, verify it — and prefer describing the mechanism to asserting the
  property. The cheapest check that catches it: *which artifacts does this code actually read?*
* **Fault injection is the acceptance criterion.** Break the thing a guard guards, observe RED,
  restore. An injection that will not redden is **reported, not tuned**. Note that an assertion whose
  expected value is the system's null state — zero, empty, unchanged — is satisfied by any failure
  that stops the code short, so absence must be proved with a non-throwing injection.
