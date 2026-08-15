# Web Bundle and Widget Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one of piece B's static permeability figures live — a reader drags roads into the block along the drainage order and watches current concentrate while permeability climbs — and build the substrate the remaining four widgets will mount into.

**Architecture:** A Python baker emits a committed JSON bundle plus a generated `.d.ts`; a `web/` TypeScript tree splits into a renderer-agnostic transform layer, a canvas mark-drawing layer, an injected `StateSource`, and a mount scanner; `esbuild` bundles it in CI while `tsc` type-checks it under pixi. Exactly one widget, `PermGraph`, proves the whole path, and a Python parity test asserts the bundle equals what `permeability_graph` produces.

**Tech Stack:** Python 3.11 (geopandas, numpy, matplotlib for colormap sampling only), TypeScript, esbuild, Node, pytest, mypy --strict, ruff, pixi, MkDocs Material.

**Spec:** `docs/superpowers/specs/2026-08-15-web-bundle-and-widget-substrate-design.md`

## Global Constraints

- **Run everything through pixi:** `pixi run test`, `pixi run typecheck`, `pixi run lint`, `pixi run web`. `python -m scripts.<name>` works only under pixi (pythonpath is configured for pytest otherwise).
- **`scripts/gen_site_pages.py` must stay stdlib-only and must NEVER import `reblock`.** `deploy-site.yml` builds the site with only `mkdocs-material` installed. File-existence checks need no imports and are fine.
- **`src/` is `mypy --strict`.** `scripts/gen_web_bundle.py` is NOT added to `[tool.mypy] files` — do not add it.
- **The bundle is a committed artifact.** `examples/perm-graph/bundle.json` and `web/src/bundle.d.ts` are both committed, both emitted by the same script.
- **No number and no colour is ever retyped in TypeScript or CSS.** Everything the widget draws with — colours, the sampled `YlOrRd` ramp, width normalizations, `_UPGRADED_LW`, `_NODE_RADIUS_FRAC` — comes from the bundle. `_PERM_CMAP` is the *string* `"YlOrRd"` (`src/reblock/render.py:41`), a matplotlib colormap name, so the ramp must be sampled in Python.
- **Pinned block:** `ZAF.9.3.1_1_40972`. **Method:** `clearance` (20 segments, 486 m). **Prefix count:** 21 states, `m = 0…20`. **Lens-B index:** `len(prefix_to_permeability(...)[0])`.
- **esbuild does not type-check.** `tsc --noEmit` must run under `pixi run typecheck`, or the generated `.d.ts` has no power at all.
- **Both Node majors pinned to the same value** — the pixi `nodejs` dependency and `actions/setup-node` in `deploy-site.yml`.
- **Every guard test must be shown to fail** before it counts. Break it, run it, paste the failure into the task report, restore.
- **Accessibility, concretely:** a native `<input type="range">` (keyboard- and screen-reader-reachable without extra work), and every number the picture shows also present as text. `prefers-reduced-motion` needs no handling because nothing here animates — the slider redraws on input, it does not tween. If you add any transition, it needs the media query.

## File Structure

| file | responsibility |
|---|---|
| `scripts/gen_web_bundle.py` | **new** — bake `bundle.json` + `bundle.d.ts` from the pinned block |
| `examples/perm-graph/bundle.json` | **new, committed** — the artifact |
| `web/src/bundle.d.ts` | **new, committed, generated** — types over the artifact |
| `web/package.json`, `web/package-lock.json` | **new, committed** — pinned esbuild + typescript |
| `web/tsconfig.json` | **new** — strict TS config |
| `web/src/view/transform.ts` | **new** — world↔screen, pan/zoom, hit-test. DOM-free, unit-tested |
| `web/src/render/canvas.ts` | **new** — the only module touching a 2D context |
| `web/src/state.ts` | **new** — `StateSource` |
| `web/src/mount.ts` | **new** — `[data-widget]` scan, registry with no default |
| `web/src/widgets/perm-graph.ts` | **new** — the widget |
| `web/test/transform.test.ts` | **new** — Node tests for the pure layer |
| `tests/test_web_bundle.py` | **new** — parity + `.d.ts` structural guard |
| `pyproject.toml` | add `nodejs` to the dev feature; `web` task; extend `typecheck` |
| `.gitignore` | add `docs/js/` |
| `mkdocs.yml` | add `extra_javascript` |
| `.github/workflows/deploy-site.yml` | setup-node + `npm ci` + esbuild before `mkdocs build` |
| `scripts/gen_site_pages.py` | mount point in the partial; assert the bundle exists |
| `docs/_partials/permeability.md` | wrap the fallback figure in the mount point |

---

### Task 1: Bake the bundle

**Files:**
- Create: `scripts/gen_web_bundle.py`
- Create (generated, committed): `examples/perm-graph/bundle.json`, `web/src/bundle.d.ts`

**Interfaces:**
- Consumes: `reblock.perm_graph.permeability_graph(block, roads, params, *, adj=None, radii=None) -> GraphFigure` with fields `cx, cy, potential, ground_g, rows, cols, conductance, footpath_g, upgraded, current, n, p`; `reblock.budget.street_first_ordered(block, roads, tol) -> GeoDataFrame`; `reblock.budget.prefix_to_permeability(block, roads, p_star, params, *, tol) -> tuple[GeoDataFrame, bool]`; `reblock.compare.load_permeability_config()`.
- Produces: `examples/perm-graph/bundle.json` with the exact key set below, and `web/src/bundle.d.ts` declaring it. Tasks 2, 6 and 7 read those keys.

- [ ] **Step 1: Write the baker**

Create `scripts/gen_web_bundle.py`. It mirrors `scripts/gen_perm_graph.py`'s config/block loading — read that file first and reuse its `initialize_config_dir` / `build_regions` / `propose` sequence verbatim, including its `VARIANT = "method_comparison"` and `METHOD = "clearance"` constants.

**Expose that loading as a module-level function, not inline in `main()`:**

```python
def load_block_and_roads() -> tuple[Block, GeoDataFrame]:
    """The pinned block and `clearance`'s full road set. A FUNCTION rather than inline setup because
    tests/test_web_bundle.py's parity test re-derives the same inputs to check the committed bundle
    against them -- if the test loaded the block a second, independent way, the two could drift and
    the parity check would be comparing the wrong things."""
```

`main()` calls it; Task 2's test imports it.

```python
"""Bake the browser bundle for the Permeability page's PermGraph widget.

Everything the widget draws is computed HERE, in Python, and read there: geometry, the
per-prefix fields, the colour ramp, and the width rules. The widget's only freedoms are which
prefix and which layer. That is deliberate -- the widget replaces a PNG whose caption quotes
numbers from `perm_graph.json`, so a second opinion about how to draw the same data would put two
pictures under one caption.

Committed, not built in CI: `scripts/gen_site_pages.py` is stdlib-only (CI builds the site with
only mkdocs-material installed) and baking needs geopandas and the solver.

Run:  pixi run python -m scripts.gen_web_bundle
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from reblock.budget import prefix_to_permeability, street_first_ordered
from reblock.derive.access import STREET_TOL
from reblock.perm_graph import permeability_graph
from reblock.render import (
    _BOUNDARY_COLOR, _CONTEXT_OUTLINE, _EDGE_GREY, _EDGE_LW_MAX, _EDGE_LW_MIN,
    _NODE_RADIUS_FRAC, _PERM_CMAP, _ROAD_COLOR, _UPGRADED_LW,
)

OUT = Path("examples/perm-graph")
DTS = Path("web/src/bundle.d.ts")
SIGFIGS = 6


def _r(x: float) -> float:
    """FIELD VALUES at 6 significant digits -- far beyond what a canvas shows or the readout
    quotes, and it keeps the payload near 300 KB. tests/test_web_bundle.py's parity assertion is
    stated at this precision, so changing it means changing that tolerance too.

    NOT for coordinates -- see `_c`."""
    return float(f"%.{SIGFIGS}g" % x)


def _c(x: float) -> float:
    """COORDINATES, as centimetres of absolute precision.

    Significant digits are the wrong tool here and dangerously so: a Cape Town UTM northing is
    ~6,240,000, so `%.6g` would round it to the nearest 10 METRES and dissolve the parcel geometry.
    Coordinates are emitted relative to `origin` (see below), which both fixes the precision problem
    and shrinks the payload, since local metres are 3-4 digits instead of 7."""
    return round(x, 2)


def _ramp(name: str, n: int = 256) -> list[str]:
    """The colormap sampled to hex stops. `_PERM_CMAP` is the STRING "YlOrRd" -- a matplotlib
    colormap name -- and the browser has no matplotlib, so a hand-rolled JS approximation would put
    the same block in two palettes on one page (the drift the parent design warns about)."""
    cmap = colormaps[name]
    return ["#%02x%02x%02x" % tuple(int(round(c * 255)) for c in cmap(t)[:3])
            for t in np.linspace(0.0, 1.0, n)]
```

Then the body: load block and `clearance` roads exactly as `gen_perm_graph.py` does, then

```python
    ordered = street_first_ordered(block, roads, STREET_TOL)
    prefix, reached = prefix_to_permeability(block, roads, pcfg.matched_permeability, params,
                                            tol=STREET_TOL)
    if not reached:
        raise SystemExit(f"{METHOD} never reached P*={pcfg.matched_permeability}")
    # prefix_to_permeability returns `ordered.iloc[:lo]`, so its LENGTH is the index into the
    # canonical sequence -- this is the prefix graph_current_after.png shows, and the widget must
    # boot here or the caption below it describes a different picture.
    lens_b_index = len(prefix)

    figs = [permeability_graph(block, ordered.iloc[:m], params) for m in range(len(ordered) + 1)]
    base = figs[0]

    # `upgraded` is monotone in the road set (conductance enters only through max(footpath, road)),
    # so store the first m at which each edge is raised instead of 21 x 745 booleans. -1 = never.
    first_upgraded_at = np.full(len(base.rows), -1, dtype=int)
    for m, f in enumerate(figs):
        newly = f.upgraded & (first_upgraded_at < 0)
        first_upgraded_at[newly] = m

    # Mesh-only width norms, matching render_graph's rule exactly (see gen_perm_graph.py): the
    # road-dominated max would collapse the mesh into a sub-pixel band.
    mesh = ~figs[-1].upgraded
    width_norm = {
        "conductance": _r(float(np.percentile(np.abs(base.footpath_g[mesh]), 99))),
        "current": _r(float(np.percentile(
            np.abs(np.concatenate([f.current[mesh] for f in figs])), 99))),
    }
```

The emitted object — this key set is the contract Tasks 2/6/7 depend on:

```python
    # Everything geometric is emitted RELATIVE to this, in metres. The canvas works in local metres
    # and never learns the CRS; `width_m` is a length, so translation leaves it alone.
    ox, oy = float(base.cx.min()), float(base.cy.min())

    bundle = {
        "block_id": block.block_id,
        "method": METHOD,
        "lens_b_index": lens_b_index,
        "n_prefixes": len(figs),
        "origin": [ox, oy],
        "parcels": [[[_c(x - ox), _c(y - oy)] for x, y in g.exterior.coords]
                    for g in block.parcels.geometry],
        "nodes": {"cx": [_c(v - ox) for v in base.cx], "cy": [_c(v - oy) for v in base.cy],
                  "ground_g": [_r(v) for v in base.ground_g]},
        "edges": {"rows": base.rows.tolist(), "cols": base.cols.tolist(),
                  "footpath_g": [_r(v) for v in base.footpath_g],
                  "first_upgraded_at": first_upgraded_at.tolist()},
        "roads": [{"coords": [[_c(x - ox), _c(y - oy)] for x, y in g.coords],
                   "width_m": float(w)}
                  for g, w in zip(ordered.geometry, ordered["width_m"], strict=True)],
        "prefix": {
            "potential": [[_r(v) for v in f.potential] for f in figs],
            "current": [[_r(v) for v in f.current] for f in figs],
            "permeability": [_r(1.0 - f.p / base.p) for f in figs],
            "road_m": [_r(float(ordered.geometry.iloc[:m].length.sum()))
                       for m in range(len(figs))],
        },
        "encoding": {
            "width_norm": width_norm,
            "edge_lw_min": _EDGE_LW_MIN, "edge_lw_max": _EDGE_LW_MAX,
            "upgraded_lw": _UPGRADED_LW, "node_radius_frac": _NODE_RADIUS_FRAC,
            "ramp": _ramp(_PERM_CMAP),
            "road_color": _ROAD_COLOR, "boundary_color": _BOUNDARY_COLOR,
            "parcel_color": _CONTEXT_OUTLINE, "edge_color": _EDGE_GREY,
        },
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")
```

Note `parcels` uses `g.exterior.coords`: Voronoi parcels are simple polygons with no holes. If any parcel turns out to be a MultiPolygon or to have interior rings, raise rather than silently dropping geometry, and report it.

- [ ] **Step 2: Emit the `.d.ts` from the same script**

Append to the same script, so the two files can never be regenerated independently:

```python
DTS_TEMPLATE = '''// GENERATED by scripts/gen_web_bundle.py -- do not edit.
// Regenerate: pixi run python -m scripts.gen_web_bundle
// This file is what makes a renamed Python field a TypeScript error instead of a blank panel.
export interface Encoding {
  width_norm: { conductance: number; current: number };
  edge_lw_min: number;
  edge_lw_max: number;
  upgraded_lw: number;
  node_radius_frac: number;
  ramp: string[];
  road_color: string;
  boundary_color: string;
  parcel_color: string;
  edge_color: string;
}
export interface Bundle {
  block_id: string;
  method: string;
  lens_b_index: number;
  n_prefixes: number;
  /** UTM easting/northing subtracted from every coordinate below; all geometry is local metres. */
  origin: [number, number];
  parcels: [number, number][][];
  nodes: { cx: number[]; cy: number[]; ground_g: number[] };
  edges: { rows: number[]; cols: number[]; footpath_g: number[]; first_upgraded_at: number[] };
  roads: { coords: [number, number][]; width_m: number }[];
  prefix: {
    potential: number[][];
    current: number[][];
    permeability: number[];
    road_m: number[];
  };
  encoding: Encoding;
}
'''
    DTS.parent.mkdir(parents=True, exist_ok=True)
    DTS.write_text(DTS_TEMPLATE, encoding="utf-8")
```

- [ ] **Step 3: Run it and inspect**

Run: `pixi run python -m scripts.gen_web_bundle`
Expected: `examples/perm-graph/bundle.json` and `web/src/bundle.d.ts` written.

Run: `python3 -c "import json;b=json.load(open('examples/perm-graph/bundle.json'));print(b['n_prefixes'], b['lens_b_index'], len(b['parcels']), len(b['edges']['rows']), len(b['encoding']['ramp']))"`
Expected: `21 <index> 263 745 256`, with `<index>` between 1 and 20.

Run: `ls -la examples/perm-graph/bundle.json`
Expected: roughly 250–400 KB. If it is over 1 MB, report the actual figure and which array dominates rather than silently shipping it.

Sanity-check three values against the committed artifact — `examples/perm-graph/perm_graph.json` says `permeability_after` is `0.6253906117970942` and `road_m` `89.43251802592889`:

Run: `python3 -c "
import json
b=json.load(open('examples/perm-graph/bundle.json')); i=b['lens_b_index']
print('perm at lens_b:', b['prefix']['permeability'][i])
print('road_m at lens_b:', b['prefix']['road_m'][i])
print('perm at 0:', b['prefix']['permeability'][0])"`
Expected: permeability ≈ 0.625391 and road_m ≈ 89.4325 at the Lens-B index (they must agree with `perm_graph.json` to the emitted precision — that is the whole point of booting the widget there), and exactly `0.0` at prefix 0.

If they disagree, the road ordering or the prefix indexing is wrong. Do not adjust the expected numbers.

- [ ] **Step 4: Lint**

Run: `pixi run lint`
Expected: clean. Note the script imports several `_`-prefixed constants from `reblock.render`; that is deliberate (the spec requires the palette come from Python) — if ruff objects, keep the import and record which rule fired.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_web_bundle.py examples/perm-graph/bundle.json web/src/bundle.d.ts
git commit -m "feat: bake the browser bundle for PermGraph

Everything the widget draws is computed in Python and read there --
geometry, per-prefix fields, the sampled YlOrRd ramp, the width rules --
so the interactive version cannot draw the same data by different rules
than the PNG it replaces.

upgraded is monotone in the road set, so the bundle stores the first
prefix at which each edge is raised instead of 21x745 booleans, which
also puts the monotonicity the metric rests on into the artifact."
```

---

### Task 2: The parity test and the `.d.ts` guard

This is the task that makes the bake trustworthy. It is the payoff for having built piece B first.

**Files:**
- Create: `tests/test_web_bundle.py`

**Interfaces:**
- Consumes: `examples/perm-graph/bundle.json` and `web/src/bundle.d.ts` from Task 1.
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing tests**

```python
"""The bundle is a committed artifact, so nothing recomputes it on the way to the browser. These
tests are the only thing standing between a bad bake and a wrong picture."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path("examples/perm-graph/bundle.json")
DTS = Path("web/src/bundle.d.ts")


@pytest.fixture(scope="module")
def bundle() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_shapes_are_internally_consistent(bundle) -> None:
    n_nodes, n_edges = len(bundle["nodes"]["cx"]), len(bundle["edges"]["rows"])
    assert len(bundle["nodes"]["cy"]) == len(bundle["nodes"]["ground_g"]) == n_nodes
    assert len(bundle["edges"]["cols"]) == len(bundle["edges"]["footpath_g"]) == n_edges
    assert len(bundle["edges"]["first_upgraded_at"]) == n_edges
    assert bundle["n_prefixes"] == len(bundle["roads"]) + 1
    for key in ("potential", "current", "permeability", "road_m"):
        assert len(bundle["prefix"][key]) == bundle["n_prefixes"]
    assert all(len(p) == n_nodes for p in bundle["prefix"]["potential"])
    assert all(len(c) == n_edges for c in bundle["prefix"]["current"])
    assert 0 < bundle["lens_b_index"] < bundle["n_prefixes"]


def test_first_upgraded_at_is_monotone_and_in_range(bundle) -> None:
    """-1 means never raised. Any other value must be a real prefix index -- an off-by-one here
    would silently paint the wrong edges blue at every slider position."""
    n = bundle["n_prefixes"]
    fu = np.asarray(bundle["edges"]["first_upgraded_at"])
    assert ((fu == -1) | ((fu >= 0) & (fu < n))).all()
    assert (fu != 0).all(), "no edge can be road-raised at prefix 0: there are no roads"


def test_coordinates_are_local_metres_at_centimetre_precision(bundle) -> None:
    """Guards a bug that would ship as *slightly wrong geometry* rather than as a crash.

    Coordinates are emitted relative to `origin`. If someone rounds them with significant digits
    instead of absolute precision, a Cape Town UTM northing (~6,240,000) rounds to the nearest 10 m
    and the parcels dissolve -- while the file still parses, the widget still draws, and the picture
    is merely wrong. So: coordinates must be small (local, not UTM), and the extent they span must
    be a plausible block, not a degenerate or continental one."""
    xs = [x for ring in bundle["parcels"] for x, _ in ring]
    ys = [y for ring in bundle["parcels"] for _, y in ring]
    assert max(abs(v) for v in xs + ys) < 10_000, "coordinates look like UTM, not local metres"
    assert 20 < (max(xs) - min(xs)) < 2_000, f"implausible x extent {max(xs) - min(xs)}"
    assert 20 < (max(ys) - min(ys)) < 2_000, f"implausible y extent {max(ys) - min(ys)}"
    # Centimetre rounding must leave at least 3 distinct values per 10 m of extent; 10 m rounding
    # would collapse a ~200 m block to ~20 distinct coordinates.
    assert len(set(xs)) > (max(xs) - min(xs)) / 10 * 3, "coordinate resolution looks too coarse"
    assert bundle["origin"][1] > 1_000_000, "origin should carry the real UTM northing"


def test_permeability_is_zero_at_prefix_zero_and_monotone(bundle) -> None:
    perm = bundle["prefix"]["permeability"]
    assert perm[0] == 0.0
    assert all(b >= a - 1e-9 for a, b in zip(perm, perm[1:])), "permeability must not fall"


def test_dts_declares_exactly_the_bundle_keys(bundle) -> None:
    """Catches 'regenerated one file, not the other'. Structural and fast -- no solving."""
    dts = DTS.read_text(encoding="utf-8")
    declared = set(re.findall(r"^\s{2}(\w+)[?]?:", dts, flags=re.M))
    for key in bundle:
        assert key in declared, f"bundle key {key!r} missing from bundle.d.ts"
    for key in bundle["encoding"]:
        assert key in declared, f"encoding key {key!r} missing from bundle.d.ts"


@pytest.mark.slow
def test_bundle_matches_permeability_graph_at_every_prefix(bundle) -> None:
    """THE parity test: the committed bundle must equal what the Python twin produces, at the 6
    significant digits the baker emits. This is what B being built first bought us."""
    from scripts.gen_web_bundle import load_block_and_roads   # Task 1 exposes this
    from reblock.budget import street_first_ordered
    from reblock.compare import load_permeability_config
    from reblock.derive.access import STREET_TOL
    from reblock.perm_graph import permeability_graph

    block, roads = load_block_and_roads()
    params = load_permeability_config().params
    ordered = street_first_ordered(block, roads, STREET_TOL)

    for m in range(bundle["n_prefixes"]):
        fig = permeability_graph(block, ordered.iloc[:m], params)
        np.testing.assert_allclose(bundle["prefix"]["potential"][m], fig.potential, rtol=1e-5)
        np.testing.assert_allclose(bundle["prefix"]["current"][m], fig.current, rtol=1e-5,
                                   atol=1e-9)
```

Task 1's script must therefore expose its loading as a reusable `load_block_and_roads() -> tuple[Block, GeoDataFrame]` rather than inlining it in `main()`. If it does not yet, refactor it to do so as part of this task and say so in the report.

- [ ] **Step 2: Run and verify**

Run: `pixi run pytest tests/test_web_bundle.py -v -m "not slow"`
Expected: the four fast tests PASS.

`slow` is not a registered marker in this repo — only `network` is (`pyproject.toml:268`). Register it in the same block, following that entry's style:

```toml
markers = [
    "network: hits the network (Geofabrik/Overpass/Open Buildings); deselect with -m 'not network'",
    "slow: needs a warm derivation cache and re-solves a block; deselect with -m 'not slow'",
]
```

Run: `pixi run pytest tests/test_web_bundle.py -v -m slow`
Expected: PASS. It needs the pinned block, so the derivation cache must be warm; at ~21 solves on 263 parcels it is roughly a second of solving plus block loading.

- [ ] **Step 3: Fault-inject the parity test**

The parity test passes on correct code, which is not evidence. Prove it. For each, make the edit, run `pixi run pytest tests/test_web_bundle.py -m slow`, paste the failure into the report, then revert and re-bake.

| # | break in `scripts/gen_web_bundle.py` | must fail |
|---|---|---|
| 1 | off-by-one the prefix loop: `range(len(ordered))` instead of `range(len(ordered) + 1)` | shape test, then parity |
| 2 | drop the ordering: use `roads` instead of `ordered` in the `figs` comprehension | parity, at some prefix |
| 3 | round to 2 significant digits (`SIGFIGS = 2`) | parity, on `rtol=1e-5` |

Row 3 is the one that proves the tolerance is meaningful rather than decorative. If it does *not* fail, the tolerance is too loose — report that and tighten it.

- [ ] **Step 4: Confirm restored**

Run: `git diff --stat scripts/gen_web_bundle.py examples/perm-graph/bundle.json`
Expected: empty (re-bake after the last revert).

- [ ] **Step 5: Commit**

```bash
git add tests/test_web_bundle.py scripts/gen_web_bundle.py
git commit -m "test: the bundle must equal its Python twin at every prefix

Nothing recomputes the bundle between the baker and the browser, so this
parity assertion is the only thing between a bad bake and a wrong
picture. Fault-injected first: an off-by-one prefix loop, a dropped
drainage ordering, and a slackened rounding all observed failing."
```

---

### Task 3: The `web/` toolchain

**Files:**
- Create: `web/package.json`, `web/package-lock.json`, `web/tsconfig.json`
- Modify: `pyproject.toml` (dev deps, `web` task, `typecheck` task; the `slow` marker is added in Task 2), `.gitignore`, `mkdocs.yml`
- Modify: `.github/workflows/deploy-site.yml`

**Interfaces:**
- Produces: `pixi run web` builds `web/src/mount.ts` → `docs/js/widgets.js`; `pixi run typecheck` also runs `tsc --noEmit`. Tasks 4–6 rely on both.

- [ ] **Step 1: Scaffold `web/`**

`web/package.json` — pin both dev dependencies exactly, the way `mkdocs-material==9.7.7` already is in `deploy-site.yml`:

```json
{
  "name": "reblock-web",
  "private": true,
  "type": "module",
  "scripts": {
    "build": "esbuild src/mount.ts --bundle --format=iife --minify --outfile=../docs/js/widgets.js",
    "check": "tsc --noEmit"
  },
  "devDependencies": {
    "esbuild": "0.25.0",
    "typescript": "5.7.2"
  }
}
```

Use whatever exact versions `npm install` resolves at implementation time and record them; the requirement is that they are pinned to a single version with no range prefix, and that `package-lock.json` is committed.

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noEmit": true,
    "lib": ["ES2022", "DOM"],
    "types": []
  },
  "include": ["src", "test"]
}
```

`noUncheckedIndexedAccess` is deliberate: this code indexes baked arrays constantly, and it is the flag that turns a prefix off-by-one into a type error.

Run `npm install` inside `web/` (not `npm ci` — there is no lock file yet), then commit the lock file it generates.

- [ ] **Step 2: Wire pixi**

In `pyproject.toml`, add to `[tool.pixi.feature.dev.dependencies]` (dev-only tooling, matching that section's stated intent):

```toml
# Builds and type-checks the web/ widget bundle. Pinned to the same major as
# .github/workflows/deploy-site.yml's actions/setup-node, so a type error cannot pass locally and
# fail in the site build (or the reverse).
nodejs = "22.*"
```

Then in `[tool.pixi.tasks]`:

```toml
web = { cmd = "npm ci && npm run build", cwd = "web" }
web-check = { cmd = "npm ci && npm run check", cwd = "web" }
typecheck = { depends-on = ["typecheck-py", "web-check"] }
```

and rename the existing `typecheck` line to `typecheck-py`, keeping its argument list byte-identical:

```toml
typecheck-py = "mypy --strict src tests scripts/crossblock_probe.py scripts/calibrate_permeability.py scripts/perf/records.py scripts/perf/region_cap_report.py"
```

This matters: `ci.yml` runs `pixi run typecheck`, so `tsc` now gates pull requests. **esbuild strips types without checking them** — without this step the generated `.d.ts` would be decorative.

- [ ] **Step 3: Wire gitignore and mkdocs**

`.gitignore`, beside the existing `docs/assets/` at line 65:

```
# Built by `pixi run web` (esbuild) and by deploy-site.yml in CI -- generated, like docs/assets/.
docs/js/
```

`mkdocs.yml`, after the `extra_css` block:

```yaml
extra_javascript:
  - js/widgets.js
```

- [ ] **Step 4: Wire CI**

In `.github/workflows/deploy-site.yml`, add before the `python3 scripts/gen_site_pages.py` step:

```yaml
      # The widget bundle. Node is pinned to the same major as pyproject.toml's pixi `nodejs`
      # dependency, so the tsc that gates pull requests and the esbuild that ships are the same
      # runtime. `npm ci` (not install) so package-lock.json is authoritative.
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - run: npm ci
        working-directory: web
      - run: npm run build
        working-directory: web
```

Order matters: `mkdocs build --strict` must come after this, and it already does.

- [ ] **Step 5: Verify and commit**

Run: `pixi run web`
Expected: `docs/js/widgets.js` written. It will fail until `web/src/mount.ts` exists — create a one-line placeholder `export {};` for this task only if needed, and note it; Task 6 replaces it.

Run: `pixi run typecheck`
Expected: mypy clean AND tsc clean.

Run: `git status --porcelain docs/js`
Expected: empty — `docs/js/` is ignored.

```bash
git add web/package.json web/package-lock.json web/tsconfig.json pyproject.toml .gitignore mkdocs.yml .github/workflows/deploy-site.yml
git commit -m "build: the web/ TypeScript toolchain

esbuild bundles, tsc checks, and they are separate jobs because esbuild
strips types without looking at them -- so tsc runs under pixi where
ci.yml already picks it up, and a type error fails the PR rather than
the deploy. Node is pinned to one major across the pixi dependency and
setup-node so both run the same runtime."
```

---

### Task 4: `transform.ts` — the renderer-agnostic layer

The only module with real unit tests, and the one an SVG chart or a WebGPU map reuses unchanged.

**Files:**
- Create: `web/src/view/transform.ts`, `web/test/transform.test.ts`
- Modify: `web/package.json` (a `test` script)

**Interfaces:**
- Produces:
  ```ts
  export interface Bbox { minX: number; minY: number; maxX: number; maxY: number }
  export interface View { scale: number; tx: number; ty: number }
  export function fitBbox(b: Bbox, width: number, height: number, pad?: number): View
  export function toScreen(v: View, x: number, y: number): [number, number]
  export function toWorld(v: View, sx: number, sy: number): [number, number]
  export function panned(v: View, dxScreen: number, dyScreen: number): View
  export function zoomed(v: View, factor: number, sx: number, sy: number): View
  export function nearest(xs: number[], ys: number[], wx: number, wy: number): number
  ```
  Tasks 5 and 6 consume all of these.

- [ ] **Step 1: Write the failing tests**

`web/test/transform.test.ts`, using Node's built-in test runner (no new dependency):

```ts
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { fitBbox, nearest, panned, toScreen, toWorld, zoomed } from "../src/view/transform.js";

const BOX = { minX: 0, minY: 0, maxX: 100, maxY: 50 };

test("fitBbox centres the box and preserves aspect ratio", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  // The box is twice as wide as tall, so width binds: 400/100 = 4.
  assert.equal(v.scale, 4);
  const [, topY] = toScreen(v, 0, 50);
  const [, botY] = toScreen(v, 0, 0);
  // Vertically centred: equal slack above and below a 200px-tall drawing in 400px.
  assert.ok(Math.abs(topY - 100) < 1e-9, `topY ${topY}`);
  assert.ok(Math.abs(botY - 300) < 1e-9, `botY ${botY}`);
});

test("y is flipped: world up is screen up", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  const [, yLow] = toScreen(v, 0, 0);
  const [, yHigh] = toScreen(v, 0, 50);
  assert.ok(yHigh < yLow, "larger world y must give smaller screen y");
});

test("toWorld inverts toScreen", () => {
  const v = fitBbox(BOX, 400, 400, 0.1);
  for (const [x, y] of [[0, 0], [100, 50], [37.5, 12.25]] as [number, number][]) {
    const [sx, sy] = toScreen(v, x, y);
    const [wx, wy] = toWorld(v, sx, sy);
    assert.ok(Math.abs(wx - x) < 1e-9 && Math.abs(wy - y) < 1e-9, `${wx},${wy} != ${x},${y}`);
  }
});

test("zoom keeps the cursor's world point under the cursor", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  const anchor: [number, number] = [123, 210];
  const before = toWorld(v, ...anchor);
  const after = toWorld(zoomed(v, 2.5, ...anchor), ...anchor);
  assert.ok(Math.abs(before[0] - after[0]) < 1e-9, "world x under cursor moved");
  assert.ok(Math.abs(before[1] - after[1]) < 1e-9, "world y under cursor moved");
});

test("pan moves by exactly the screen delta", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  const [x0, y0] = toScreen(v, 10, 10);
  const [x1, y1] = toScreen(panned(v, 25, -8), 10, 10);
  assert.equal(x1 - x0, 25);
  assert.equal(y1 - y0, -8);
});

test("nearest returns the closest index, not merely a close one", () => {
  const xs = [0, 10, 20];
  const ys = [0, 0, 0];
  assert.equal(nearest(xs, ys, 9.4, 0), 1);
  assert.equal(nearest(xs, ys, 5.1, 0), 1);
  assert.equal(nearest(xs, ys, 4.9, 0), 0);
});
```

Add to `web/package.json`'s scripts: `"test": "node --test --experimental-strip-types test/"` (Node 22 runs TypeScript tests directly with type stripping — no bundler or test framework needed). If that flag is unavailable in the pinned Node, compile with `tsc` to a temp dir and run the JS instead, and record which route you took.

- [ ] **Step 2: Run to verify failure**

Run: `cd web && npm test`
Expected: FAIL — cannot resolve `../src/view/transform.js`.

- [ ] **Step 3: Implement**

`web/src/view/transform.ts`:

```ts
/** World (projected UTM metres) <-> screen, with pan/zoom and nearest-mark queries.
 *
 * Renderer-agnostic and DOM-free on purpose. The parent design called the substrate "a canvas
 * renderer", which conflates two layers: this one is shared by every widget, while what draws the
 * marks is free to differ -- piece D's Frontier chart wants SVG for its axis text, and ScreenMap's
 * 16k polygons want canvas. Keeping them apart is also what makes this file unit-testable.
 *
 * Parcels arrive already projected, so there is no reprojection anywhere: fit the bbox and draw.
 */
export interface Bbox { minX: number; minY: number; maxX: number; maxY: number }

/** screenX = x * scale + tx; screenY = ty - y * scale  (y flips: world up is screen up). */
export interface View { scale: number; tx: number; ty: number }

export function fitBbox(b: Bbox, width: number, height: number, pad = 0.04): View {
  const w = Math.max(b.maxX - b.minX, 1e-9);
  const h = Math.max(b.maxY - b.minY, 1e-9);
  const scale = Math.min(width / w, height / h) * (1 - 2 * pad);
  const tx = (width - w * scale) / 2 - b.minX * scale;
  const ty = (height + h * scale) / 2 + b.minY * scale;
  return { scale, tx, ty };
}

export function toScreen(v: View, x: number, y: number): [number, number] {
  return [x * v.scale + v.tx, v.ty - y * v.scale];
}

export function toWorld(v: View, sx: number, sy: number): [number, number] {
  return [(sx - v.tx) / v.scale, (v.ty - sy) / v.scale];
}

export function panned(v: View, dxScreen: number, dyScreen: number): View {
  return { scale: v.scale, tx: v.tx + dxScreen, ty: v.ty + dyScreen };
}

/** Zoom about a screen anchor, keeping the world point under it fixed. */
export function zoomed(v: View, factor: number, sx: number, sy: number): View {
  const scale = v.scale * factor;
  return { scale, tx: sx - (sx - v.tx) * factor, ty: sy - (sy - v.ty) * factor };
}

/** Index of the nearest of `xs`/`ys` to a world point. Linear: 263 nodes needs no index. */
export function nearest(xs: number[], ys: number[], wx: number, wy: number): number {
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i]! - wx;
    const dy = ys[i]! - wy;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd web && npm test`
Expected: all six PASS.

Run: `pixi run typecheck`
Expected: clean.

The y-flip and the zoom-anchor tests are the two that catch real bugs here; if either passes trivially (e.g. because `scale` is 1), change the fixture until it does not, and say so.

- [ ] **Step 5: Commit**

```bash
git add web/src/view/transform.ts web/test/transform.test.ts web/package.json
git commit -m "feat: renderer-agnostic view transform

World-metres <-> screen with pan, zoom-about-a-point and nearest-mark
queries, DOM-free so it unit-tests in Node and so an SVG chart or a GPU
map can reuse it unchanged. Splitting this from mark-drawing is what
keeps piece D from either rewriting the substrate or bending a chart
into a canvas."
```

---

### Task 5: `state.ts` and `mount.ts`

**Files:**
- Create: `web/src/state.ts`, `web/src/mount.ts`

**Interfaces:**
- Produces:
  ```ts
  // state.ts
  export interface WidgetState { prefix: number; layer: "conductance" | "current"; halos: boolean }
  export interface StateSource {
    get(): WidgetState;
    set(patch: Partial<WidgetState>): void;
    subscribe(fn: (s: WidgetState) => void): void;
  }
  export function localState(initial: WidgetState): StateSource
  // mount.ts
  export type Widget = (host: HTMLElement, state: StateSource) => void
  export function register(name: string, w: Widget): void
  export function mountAll(root?: ParentNode): void
  ```
  Task 6 registers `perm-graph` and consumes `StateSource`.

- [ ] **Step 1: Implement `state.ts`**

```ts
/** The state a widget reads, injected at mount.
 *
 * In prose pages this is a local store seeded from the mount point's data-* attributes; piece E
 * swaps in a URL-synced shared store for the Explore page. The widget never learns which it has --
 * that is the whole reason to inject it rather than grow an `if (embedded)` branch per widget, with
 * the branch count climbing alongside the widget count.
 */
export interface WidgetState {
  prefix: number;
  layer: "conductance" | "current";
  halos: boolean;
}

export interface StateSource {
  get(): WidgetState;
  set(patch: Partial<WidgetState>): void;
  subscribe(fn: (s: WidgetState) => void): void;
}

export function localState(initial: WidgetState): StateSource {
  let current = { ...initial };
  const listeners: ((s: WidgetState) => void)[] = [];
  return {
    get: () => current,
    set(patch) {
      current = { ...current, ...patch };
      for (const fn of listeners) fn(current);
    },
    subscribe(fn) { listeners.push(fn); },
  };
}
```

- [ ] **Step 2: Implement `mount.ts`**

```ts
/** The mount contract: a page carries a placeholder and nothing else. */
import { localState, type StateSource, type WidgetState } from "./state.js";

export type Widget = (host: HTMLElement, state: StateSource) => void;

const REGISTRY = new Map<string, Widget>();

export function register(name: string, w: Widget): void {
  REGISTRY.set(name, w);
}

function initialState(el: HTMLElement): WidgetState {
  const layer = el.dataset.layer === "conductance" ? "conductance" : "current";
  return { prefix: Number(el.dataset.prefix ?? 0), layer, halos: el.dataset.halos !== "false" };
}

export function mountAll(root: ParentNode = document): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-widget]"))) {
    const name = el.dataset.widget!;
    const widget = REGISTRY.get(name);
    // No default. The name arrives from HTML -- a genuinely open boundary, so a string lookup is
    // right here -- but an unknown one must throw rather than leave a silently empty mount point
    // that looks like a widget which merely failed to draw.
    if (widget === undefined) throw new Error(`unknown data-widget: ${name}`);
    widget(el, localState(initialState(el)));
  }
}

document.addEventListener("DOMContentLoaded", () => mountAll());
```

`navigation.instant` is confirmed absent from `mkdocs.yml`, so `DOMContentLoaded` is sufficient and no Material `document$` hook is needed. If that feature is ever enabled, this line is what breaks.

- [ ] **Step 3: Verify**

Run: `pixi run typecheck`
Expected: clean.

Run: `pixi run web`
Expected: `docs/js/widgets.js` builds (mount.ts is the entry point).

- [ ] **Step 4: Commit**

```bash
git add web/src/state.ts web/src/mount.ts
git commit -m "feat: StateSource injection and the mount contract

A widget reads an injected StateSource and never asks whether it is
inline or in the explorer, so piece E can supply a URL-synced store
without touching any widget. The registry lookup has no default: an
unknown data-widget name throws rather than leaving a mount point that
looks like a widget which merely failed to draw."
```

---

### Task 6: `PermGraph`, on the page

**Files:**
- Create: `web/src/render/canvas.ts`, `web/src/widgets/perm-graph.ts`
- Modify: `web/src/mount.ts` (import the widget so esbuild includes it)
- Modify: `docs/_partials/permeability.md`, `scripts/gen_site_pages.py`

**Interfaces:**
- Consumes: `transform.ts`'s exports (Task 4), `StateSource`/`register` (Task 5), `Bundle` from `web/src/bundle.d.ts` (Task 1).
- Produces: the mounted widget. Nothing downstream in C.

- [ ] **Step 1: Implement the canvas mark layer**

`web/src/render/canvas.ts` — the only module that knows a 2D context exists:

```ts
import type { Bundle } from "../bundle.js";
import { toScreen, type View } from "../view/transform.js";

/** Resize the backing store for devicePixelRatio and return the CSS-pixel size to draw in. */
export function sizeCanvas(cv: HTMLCanvasElement): { width: number; height: number } {
  const dpr = window.devicePixelRatio || 1;
  const { width, height } = cv.getBoundingClientRect();
  cv.width = Math.round(width * dpr);
  cv.height = Math.round(height * dpr);
  const ctx = cv.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

function rampColor(ramp: string[], t: number): string {
  const i = Math.min(ramp.length - 1, Math.max(0, Math.round(t * (ramp.length - 1))));
  return ramp[i]!;
}

export interface Frame { view: View; prefix: number; layer: "conductance" | "current"; halos: boolean }

export function draw(ctx: CanvasRenderingContext2D, b: Bundle, f: Frame,
                     size: { width: number; height: number }): void {
  const e = b.encoding;
  ctx.clearRect(0, 0, size.width, size.height);

  // Parcels as a pale wireframe, never filled: filling them by potential would state the same
  // quantity twice in two shapes and drown the graph (piece B's finding).
  ctx.strokeStyle = e.parcel_color;
  ctx.lineWidth = 0.4;
  for (const ring of b.parcels) {
    ctx.beginPath();
    ring.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(f.view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
    ctx.stroke();
  }

  // The road corridor, drawn once as a translucent stroke of the prefix's segments at their own
  // width. Stroked rather than buffered+filled because overlapping translucent fills compound
  // toward opaque -- exactly the bug that made piece B's corridor unreadable.
  ctx.globalAlpha = 0.25;
  ctx.strokeStyle = e.road_color;
  ctx.lineCap = "round";
  for (const r of b.roads.slice(0, f.prefix)) {
    ctx.lineWidth = r.width_m * f.view.scale;
    ctx.beginPath();
    r.coords.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(f.view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Edges. Width encodes the chosen quantity for MESH edges only; road-raised edges draw at the
  // fixed upgraded_lw, because their computed width would be a saturated non-measurement.
  const quantity = f.layer === "current" ? b.prefix.current[f.prefix]! : b.edges.footpath_g;
  const norm = e.width_norm[f.layer];
  const { rows, cols, first_upgraded_at } = b.edges;
  for (let k = 0; k < rows.length; k++) {
    const up = first_upgraded_at[k]! >= 0 && first_upgraded_at[k]! <= f.prefix;
    const frac = Math.min(1, Math.abs(quantity[k]!) / norm);
    ctx.strokeStyle = up ? e.road_color : e.edge_color;
    ctx.lineWidth = up ? e.upgraded_lw : e.edge_lw_min + frac * (e.edge_lw_max - e.edge_lw_min);
    const [x0, y0] = toScreen(f.view, b.nodes.cx[rows[k]!]!, b.nodes.cy[rows[k]!]!);
    const [x1, y1] = toScreen(f.view, b.nodes.cx[cols[k]!]!, b.nodes.cy[cols[k]!]!);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  // Nodes, coloured by potential on the ramp Python sampled. vmax is prefix 0's maximum: roads
  // only lower potentials, so this is the shared scale across every slider position.
  const pot = b.prefix.potential[f.prefix]!;
  const vmax = Math.max(...b.prefix.potential[0]!);
  const r = e.node_radius_frac * medianEdgeLength(b) * f.view.scale;
  for (let i = 0; i < pot.length; i++) {
    const [sx, sy] = toScreen(f.view, b.nodes.cx[i]!, b.nodes.cy[i]!);
    if (f.halos && b.nodes.ground_g[i]! > 0) {
      ctx.strokeStyle = e.boundary_color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 1.6, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = rampColor(e.ramp, vmax > 0 ? pot[i]! / vmax : 0);
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function medianEdgeLength(b: Bundle): number {
  const ds = b.edges.rows.map((ri, k) => {
    const ci = b.edges.cols[k]!;
    return Math.hypot(b.nodes.cx[ri]! - b.nodes.cx[ci]!, b.nodes.cy[ri]! - b.nodes.cy[ci]!);
  }).sort((a, z) => a - z);
  return ds[Math.floor(ds.length / 2)] ?? 1;
}
```

- [ ] **Step 2: Implement the widget**

`web/src/widgets/perm-graph.ts`. It must boot at `bundle.lens_b_index`, because the fallback image it replaces is `graph_current_after.png` and the caption beneath quotes that prefix's numbers.

```ts
import type { Bundle } from "../bundle.js";
import { draw, sizeCanvas } from "../render/canvas.js";
import { register, type Widget } from "../mount.js";
import { fitBbox, nearest, panned, toWorld, zoomed, type View } from "../view/transform.js";
import type { StateSource } from "../state.js";

const permGraph: Widget = (host, state) => {
  const src = host.dataset.bundle!;
  void fetch(src).then((r) => r.json()).then((b: Bundle) => boot(host, state, b));
};

function boot(host: HTMLElement, state: StateSource, b: Bundle): void {
  // The fallback PNG shows clearance's Lens-B prefix; boot anywhere else and the page swaps in a
  // picture the caption below it does not describe.
  state.set({ prefix: b.lens_b_index });

  const fallback = host.querySelector("img");
  const cv = document.createElement("canvas");
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";
  const controls = document.createElement("div");
  const slider = document.createElement("input");
  slider.type = "range";                    // native: keyboard- and screen-reader-reachable
  slider.min = "0";
  slider.max = String(b.n_prefixes - 1);
  slider.value = String(b.lens_b_index);
  slider.setAttribute("aria-label", "road prefix");
  const readout = document.createElement("p");
  controls.append(slider, readout);
  host.append(cv, controls);
  if (fallback) fallback.remove();

  const xs = b.nodes.cx, ys = b.nodes.cy;
  const bbox = { minX: Math.min(...xs), minY: Math.min(...ys),
                 maxX: Math.max(...xs), maxY: Math.max(...ys) };
  let size = sizeCanvas(cv);
  let view: View = fitBbox(bbox, size.width, size.height);
  const ctx = cv.getContext("2d")!;

  const render = (): void => {
    const s = state.get();
    draw(ctx, b, { view, ...s }, size);
    // Every number the picture shows is also present as text.
    readout.textContent =
      `${b.prefix.road_m[s.prefix]!.toFixed(0)} m of road · ` +
      `${(b.prefix.permeability[s.prefix]! * 100).toFixed(1)}% permeability`;
  };
  state.subscribe(render);

  slider.addEventListener("input", () => state.set({ prefix: Number(slider.value) }));

  let dragging: [number, number] | null = null;
  cv.addEventListener("pointerdown", (ev) => { dragging = [ev.offsetX, ev.offsetY]; });
  cv.addEventListener("pointerup", () => { dragging = null; });
  cv.addEventListener("pointermove", (ev) => {
    if (dragging) {
      view = panned(view, ev.offsetX - dragging[0], ev.offsetY - dragging[1]);
      dragging = [ev.offsetX, ev.offsetY];
      render();
      return;
    }
    const [wx, wy] = toWorld(view, ev.offsetX, ev.offsetY);
    const i = nearest(xs, ys, wx, wy);
    cv.title = `φ = ${b.prefix.potential[state.get().prefix]![i]!.toPrecision(4)}`;
  });
  cv.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    view = zoomed(view, ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.offsetX, ev.offsetY);
    render();
  }, { passive: false });
  window.addEventListener("resize", () => {
    size = sizeCanvas(cv);
    view = fitBbox(bbox, size.width, size.height);
    render();
  });

  render();
}

register("perm-graph", permGraph);
```

Add `import "./widgets/perm-graph.js";` to `web/src/mount.ts` so esbuild includes it.

- [ ] **Step 3: Wire the mount point and the missing-bundle guard**

In `docs/_partials/permeability.md`, the marker stays; the change is in the producer. In `scripts/gen_site_pages.py`'s `_perm_graph_figures()`, wrap **only** the `graph_current_after.png` figure in the mount point, leaving the other three as plain figures:

```python
    mount = (f'<div data-widget="perm-graph" data-block="{block}"'
             f' data-bundle="{bundle_url}" data-layer="current">\n{fig_html}</div>\n')
```

where `bundle_url` comes from `_copy_asset(PERMGRAPH / "bundle.json", "perm-graph")` — the same path the PNGs already travel, so no new serving mechanism.

Then the guard, in the same module:

```python
def _assert_widget_bundle_present(pages_with_widgets: bool) -> None:
    """A site built without `pixi run web` emits a <script> tag for a file that is not there: every
    widget silently fails to boot, the PNG fallbacks still render, and the page looks FINE. Turn
    that into a build failure. File existence needs no imports, so this respects the stdlib-only
    contract."""
    if pages_with_widgets and not (DOCS / "js" / "widgets.js").exists():
        raise SystemExit(
            "docs/js/widgets.js is missing but a page carries a widget mount point -- run "
            "`pixi run web` (CI does this in deploy-site.yml before mkdocs build)")
```

Call it at the end of `main()`.

- [ ] **Step 4: Verify**

Run: `pixi run web && pixi run typecheck && pixi run python -m scripts.gen_site_pages`
Expected: all clean; the generated `docs/methodology/permeability.md` contains the `data-widget="perm-graph"` div wrapping the `graph_current_after.png` figure, and `docs/assets/perm-graph/bundle.json` exists.

**Fault-inject the guard** — this is a required step, not optional:

```bash
mv docs/js/widgets.js /tmp/widgets.js.bak
pixi run python -m scripts.gen_site_pages   # must FAIL with the message above
mv /tmp/widgets.js.bak docs/js/widgets.js
```

Paste the failure into the report. If it does not fail, the guard is not wired.

Run: `pixi run test`
Expected: 623+ passing, including Task 2's tests and the existing marker tests.

- [ ] **Step 5: Verify what is verifiable, and say what is not**

`mkdocs` is installed in no environment here, so **there is no built HTML to open and no way to see this widget render locally.** The first real render is CI's `mkdocs build --strict` plus the deployed page. Do not claim otherwise.

What you can and must check:

```bash
grep -c 'data-widget="perm-graph"' docs/methodology/permeability.md   # expect 1
grep -o 'data-bundle="[^"]*"' docs/methodology/permeability.md        # expect ../../assets/...
ls -la docs/assets/perm-graph/bundle.json docs/js/widgets.js
node -e "const b=require('./examples/perm-graph/bundle.json');console.log(b.n_prefixes,b.lens_b_index)"
```

Confirm: exactly one mount point, wrapping the `graph_current_after.png` figure and not the other three; the bundle URL carries the same `../../` prefix the images do (`depth=1, url_depth=2`); both files exist; the bundle parses as JSON in Node.

Then **state plainly in your report** that the widget was not observed rendering, and that the mount point, bundle path, bundle validity and successful esbuild bundling are what you verified. An honest gap is worth more here than a claim I cannot check.

- [ ] **Step 6: Commit**

```bash
git add web/src docs/_partials/permeability.md scripts/gen_site_pages.py
git commit -m "feat: PermGraph, and Permeability goes interactive

The widget boots at clearance's Lens-B prefix because that is the state
the fallback PNG shows and the caption beneath it quotes -- then the
slider runs past it, to the full 486 m network no static figure shows.
Width, colour and the ramp all come from the bundle, so the interactive
version cannot draw the same data by different rules than the image it
replaces.

gen_site_pages now fails the build if the widget bundle is missing: a
site built without esbuild would emit a script tag for a file that is
not there, and every widget would silently not boot behind an
intact-looking page."
```

---

## Task summary

| task | deliverable | reviewable alone |
|---|---|---|
| 1 | `bundle.json` + `bundle.d.ts`, baked | yes — inspect the artifact |
| 2 | parity test, fault-injected | yes — the report is the evidence |
| 3 | `web/` toolchain, pixi + CI wiring | yes — `pixi run web` and `typecheck` |
| 4 | `transform.ts` + Node tests | yes — pure functions |
| 5 | `StateSource` + mount contract | yes — type-checks and builds |
| 6 | `PermGraph` on the page | yes — the page carries it |

Tasks 3–5 need no data. Tasks 1, 2 and 6 need `capetown_full` in `~/.cache/reblock` and a warm derivation cache.
