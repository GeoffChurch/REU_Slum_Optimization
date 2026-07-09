# Redesign Layer 3 — derivations through `derive()` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every expensive per-block derivation (Voronoi build, before/after access, after-geometric, method proposal) through the single `derive()` primitive, keyed on the L2 identities — making F2's four `cached_*` wrappers dead (deleted in L6).

**Architecture:** A new `reblock.derivations` module wraps the existing pure algorithm bodies (reused verbatim) as `derive()` calls. Before/after are **separate functions** (`access_before(block)` vs `access_after(block, proposal)`), so their distinct `fn.identity` gives distinct keys — no `roads_key`. The Voronoi build takes a small identified `VoronoiInput`. Methods populate `Proposal.block_identity = block.identity` and expose a `Method.identity`, so `propose` and the after-derivations cache. The eval and `KblockSource` are rewired to call `reblock.derivations`; `reblock.cache` is left orphaned (deleted in L6).

**Tech Stack:** Python 3.12, geopandas/shapely, joblib (via `derive_graph`), pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently 129 tests.
- **Results are IDENTICAL to before** — the wrappers only memoize; the pinned kblock peel/geometric values and all `run`/eval tests must produce the same numbers cold or warm. The algorithm bodies are reused verbatim.
- **Keys come only from L2 identities** — `derive()` keys on `(fn.identity, input identities)`; a `None` identity (empty `source_content_hash`, or a `Proposal` with no `block_identity`) bypasses. Heavy geometry is never hashed.
- **`reblock.cache` is NOT edited or deleted here** — it is simply no longer called by `src/` after this layer (its tests keep passing until L6 deletes both). No import of `reblock.cache` remains in `KComplexityEval`/`KblockSource`/`run` after Task 3.
- **No import cycle** — `reblock.derivations` may import `derive_graph`, `contracts`, `derive.access`, `derive.geometric_access` at module level; it local-imports `kblock._voronoi_parcels` inside a function (kblock imports `derivations`). Methods import `derivations` at module level; `derivations` must not import the method modules at module level.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `reblock.derivations` — the derive()-wrapped derivations

**Files:**
- Create: `src/reblock/derivations.py`
- Test: `tests/test_derivations.py`

**Interfaces:**
- Consumes: `derive_graph.derive`; `derive.access.parcel_access_layers`; `derive.geometric_access.geometric_access_distances`; `kblock._voronoi_parcels` (local import); `contracts.Block`/`Proposal`.
- Produces:
  - `access_before(block: Block) -> pd.Series`
  - `access_after(block: Block, proposal: Proposal) -> pd.Series`
  - `geometric_after(block: Block, proposal: Proposal) -> pd.Series`
  - `VoronoiInput` (frozen, `.identity`) and `voronoi(vin: VoronoiInput) -> GeoDataFrame | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_derivations.py`. Build a small real `KblockSource`-style block via the committed Cape Town fixture so identities are non-empty and caching is exercised; assert (a) `access_before`/`access_after` differ in key (a call-counter spy shows each impl runs once), (b) results equal the direct `parcel_access_layers` values, (c) a block with empty `source_content_hash` bypasses (recomputes):

```python
from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Proposal
import reblock.derivations as D
from reblock.derive.access import parcel_access_layers

UTM = CRS.from_epsg(32643)


def _grid_block(hash_: str) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


def test_access_before_matches_direct_and_caches(monkeypatch) -> None:
    box = {"n": 0}
    real = parcel_access_layers

    def spy(block, roads=None, **kw):
        box["n"] += 1
        return real(block, roads)
    monkeypatch.setattr(D, "parcel_access_layers", spy)

    block = _grid_block("deadbeef")
    out1 = D.access_before(block)
    out2 = D.access_before(block)                 # cache hit
    assert box["n"] == 1
    assert out1.equals(out2)
    assert out1.equals(real(block, None))         # value identical to direct call


def test_before_and_after_use_distinct_keys(monkeypatch) -> None:
    box = {"n": 0}
    real = parcel_access_layers

    def spy(block, roads=None, **kw):
        box["n"] += 1
        return real(block, roads)
    monkeypatch.setattr(D, "parcel_access_layers", spy)

    block = _grid_block("deadbeef")
    prop = Proposal(block_id="g", crs=UTM, block_identity=block.identity, proposal_id="peel")
    D.access_before(block)
    D.access_after(block, prop)                   # distinct fn.identity -> distinct key
    assert box["n"] == 2


def test_bypass_when_hash_empty() -> None:
    block = _grid_block("")                        # identity None -> uncacheable
    out = D.access_before(block)
    assert isinstance(out, pd.Series)             # computes directly, no cache touch
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_derivations.py -v`
Expected: FAIL — `No module named 'reblock.derivations'`.

- [ ] **Step 3: Implement `src/reblock/derivations.py`**

```python
"""Per-block derivations, each memoized through the single derive() primitive
(reblock.derive_graph) on the L2 identities. Before/after are SEPARATE functions
so their distinct fn.identity gives distinct cache keys -- no roads_key. The
algorithm bodies live in reblock.derive.* / reblock.data.kblock and are reused
verbatim; this module only adds the derive() memoization layer (superseding the
four reblock.cache wrappers, deleted in a later layer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from shapely.geometry import Point, Polygon

from reblock.contracts import Block, Proposal
from reblock.derive.access import parcel_access_layers
from reblock.derive.geometric_access import geometric_access_distances
from reblock.derive_graph import derive


def _access_before_impl(block: Block) -> pd.Series:
    return parcel_access_layers(block, None)


def access_before(block: Block) -> pd.Series:
    return derive(_access_before_impl, block)


def _access_after_impl(block: Block, proposal: Proposal) -> pd.Series:
    return parcel_access_layers(block, proposal.roads)


def access_after(block: Block, proposal: Proposal) -> pd.Series:
    return derive(_access_after_impl, block, proposal)


def _geometric_after_impl(block: Block, proposal: Proposal) -> pd.Series:
    return geometric_access_distances(block, proposal.roads)


def geometric_after(block: Block, proposal: Proposal) -> pd.Series:
    return derive(_geometric_after_impl, block, proposal)


@dataclass(frozen=True)
class VoronoiInput:
    """Identified carrier for the Voronoi build: derive() keys on .identity
    (never the geometry); a missing source_id makes it uncacheable (bypass)."""
    source_id: str
    block_id: str
    poly: Polygon
    points: list[Point]
    crs: Any

    @property
    def identity(self) -> tuple[str, str, str] | None:
        return ("voronoi", self.source_id, self.block_id) if self.source_id else None


def _voronoi_impl(vin: VoronoiInput) -> Any:
    from reblock.data.kblock import _voronoi_parcels   # local import avoids a cycle
    return _voronoi_parcels(vin.poly, vin.points, vin.crs)


def voronoi(vin: VoronoiInput) -> Any:
    return derive(_voronoi_impl, vin)
```

- [ ] **Step 4: Run to verify pass + full check**

Run: `pixi run pytest tests/test_derivations.py -v` then `pixi run check`
Expected: PASS (3 tests). `pixi run check` green — additive module; the conftest already isolates the cache and clears `derive_graph._L1` per test, so these cache-backed tests are hermetic. 132 tests.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/derivations.py tests/test_derivations.py
git commit -m "$(cat <<'EOF'
feat: reblock.derivations -- derive()-wrapped per-block derivations (redesign L3)

access_before/access_after/geometric_after + voronoi, each memoized through the
single derive() primitive on the L2 identities. Separate before/after functions
give distinct keys (no roads_key). Algorithm bodies reused verbatim; supersedes
reblock.cache's four wrappers (deleted later). Additive -- consumers rewired next.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: methods populate `block_identity` + expose `Method.identity`; `propose` via `derive()`

**Files:**
- Modify: `src/reblock/methods/topology.py`, `src/reblock/methods/peel.py`
- Modify: `src/reblock/derivations.py` (add `propose`)
- Modify: `src/reblock/contracts.py` (add `identity` to the `Method` protocol)
- Test: `tests/methods/test_topology_method.py`, `tests/test_derivations.py`

**Interfaces:**
- Produces:
  - Each `Method` has an `identity` property (a stable per-params tuple, e.g. `("topology", alpha, seed)` / `("peel",)`).
  - Every `Proposal` a method returns has `block_identity=block.identity`.
  - `derivations.propose(method, block) -> Proposal` = `derive(_propose_impl, method, block)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/methods/test_topology_method.py`: a proposal from `TopologyMethod(...).propose(block)` has `block_identity == block.identity`, and `TopologyMethod(alpha=2.0, seed=0).identity == ("topology", 2.0, 0)`. (Use the file's existing `_grid` helper; give the block a non-empty `source_content_hash` so `identity` is set.) Add to `tests/test_derivations.py`: `D.propose(method, block)` returns a Proposal whose `.roads` equals the direct `method.propose(block).roads`, and repeat calls recompute the underlying propose once (spy on `method.propose`).

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/methods/test_topology_method.py -k "identity or block_identity" tests/test_derivations.py -k propose -v`
Expected: FAIL — methods have no `identity`; proposals have no `block_identity`; `derivations.propose` undefined.

- [ ] **Step 3: Implement**

- In `contracts.py`, add `identity` to the `Method` protocol: `class Method(Protocol): identity: object; def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...` (match the existing `propose` signature exactly — read it first).
- In `topology.py`: add an `identity` property to `TopologyMethod` returning `("topology", self.alpha, self.seed)`; set `block_identity=block.identity` in the `Proposal(...)` it returns.
- In `peel.py`: add an `identity` property to `PeelReblocker` returning `("peel",)` (plus any params it has — read it); set `block_identity=block.identity` in the `Proposal(...)` it returns.
- In `derivations.py`, add:

```python
from reblock.contracts import Method   # add to imports


def _propose_impl(method: Method, block: Block) -> Proposal:
    return method.propose(block)


def propose(method: Method, block: Block) -> Proposal:
    return derive(_propose_impl, method, block)
```

`derive(_propose_impl, method, block)` keys on `(fn.identity, method.identity, block.identity)`. `method.identity` is a small tuple (not the heavy method object); `block.identity` composes.

- [ ] **Step 4: Run + full check**

Run: `pixi run pytest tests/methods tests/test_derivations.py -v` then `pixi run check`
Expected: PASS — proposals carry `block_identity`, methods have `identity`, `propose` caches. Existing method/efficacy tests unaffected (the proposal geometry is unchanged; only `block_identity` is added).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/topology.py src/reblock/methods/peel.py src/reblock/derivations.py src/reblock/contracts.py tests/methods/test_topology_method.py tests/test_derivations.py
git commit -m "$(cat <<'EOF'
feat: Method.identity + Proposal.block_identity; propose via derive() (redesign L3)

Methods expose a stable per-params identity and stamp every Proposal with
block_identity=block.identity, so proposals get a composed identity and the
after-derivations + propose memoize. derivations.propose wraps method.propose
through derive().

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 3: rewire `KComplexityEval` + `KblockSource` to `reblock.derivations`

**Files:**
- Modify: `src/reblock/eval/kcomplexity.py`, `src/reblock/data/kblock.py`, `src/reblock/run.py`
- Test: existing suites (no new tests; results must be unchanged)

**Interfaces:**
- Consumes: `reblock.derivations.access_before`/`access_after`/`geometric_after`/`voronoi`/`propose`.
- Produces: no `reblock.cache` import remains in these three files.

- [ ] **Step 1: Rewire the eval**

Read `KComplexityEval.score` — it currently calls `reblock.cache.cached_access_layers(block, None, "__before__")`, `cached_access_layers(block, proposal.roads, after_key)`, and `cached_geometric(block, proposal.roads, after_key)`. Replace with:

```python
        pre = access_before(block)
        post = access_after(block, proposal)
        geo = geometric_after(block, proposal)
```

Import `from reblock.derivations import access_after, access_before, geometric_after`; drop the `reblock.cache` import and the now-unused `after_key` line. `street_connectivity` stays called directly (it is cheap and not memoized).

- [ ] **Step 2: Rewire `KblockSource`**

In `kblock.py` `_blocks_from`, replace the `cache.cached_voronoi_parcels(poly, pts, utm, block_id=..., source_content_hash=...)` call with a `derivations.voronoi(...)` call over a `VoronoiInput`:

```python
            parcels = voronoi(VoronoiInput(source_id=source_content_hash, block_id=str(row["block_id"]),
                                           poly=poly, points=pts, crs=utm))
```

Import `from reblock.derivations import VoronoiInput, voronoi`; drop the `from reblock.cache import cached_voronoi_parcels` import. (`source_content_hash` is the same value stamped onto the Block, so the VoronoiInput identity matches the block identity's hash component.)

- [ ] **Step 3: Rewire `run()`**

In `run.py`, replace `from reblock.cache import cached_propose` + the `cached_propose(method, block)` call with `from reblock.derivations import propose` + `proposal = propose(method, block)`.

- [ ] **Step 4: Run the full suite — results must be UNCHANGED**

Run: `pixi run check`
Expected: PASS, 132 tests. The pinned kblock peel/geometric values (`test_kblock_source.py`), the `run`/eval efficacy tests, and the block_ids/capetown tests must all produce the SAME numbers — the derivations are byte-identical, only the memoization layer changed. Confirm no `reblock.cache` import remains in the three rewired files (`grep -rn "reblock.cache\|reblock import cache\|from reblock import cache" src/reblock/{eval/kcomplexity,data/kblock,run}.py` → none). `test_cache.py` still passes (cache.py is untouched, just no longer used by src).

- [ ] **Step 5: Commit**

```bash
git add src/reblock/eval/kcomplexity.py src/reblock/data/kblock.py src/reblock/run.py
git commit -m "$(cat <<'EOF'
refactor: rewire eval + KblockSource + run to reblock.derivations (redesign L3)

KComplexityEval, KblockSource, and run() now memoize through the single
derive() primitive (reblock.derivations) instead of reblock.cache's four
wrappers. Results are byte-identical (algorithm bodies unchanged); cache.py is
now orphaned in src/ (deleted with its tests in the final layer).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (L3):** derivations re-expressed as pure functions through `derive()` (Task 1), methods carry identity + stamp proposals so the graph composes (Task 2), consumers rewired to the one primitive (Task 3). The Voronoi build is now a `derive()` call (the Source's build responsibility becomes a cached derivation) — the practical form of the ingest/build split under Option Y. ✓

**Placeholder scan:** Task 1 has complete module code. Tasks 2–3 direct the implementer to READ the current call sites (methods' `Proposal(...)`, `KComplexityEval.score`, `_blocks_from`, `run()`) and apply the shown replacements — necessary because those bodies are what's being rewired; the exact replacement lines are given. No TBD.

**Type consistency:** `access_before(block)`, `access_after(block, proposal)`, `geometric_after(block, proposal)`, `voronoi(VoronoiInput)`, `propose(method, block)` are defined in Task 1/2 and called with matching arities in Task 3. `Method.identity` (Task 2 protocol) is what `derivations.propose`'s `derive()` reads. `Proposal.block_identity` (L2 field) is set by the methods (Task 2) and read by `Proposal.identity` (L2) which `access_after`/`geometric_after` key on.

**Results-unchanged is the load-bearing guarantee** — Task 3's full-suite run against the pinned kblock values is the proof the memoization rewrite changed nothing but caching.
