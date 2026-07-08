# Flow-refactor F2 — L2 per-block persistent cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the expensive **pure** per-block derivations (block build, before/after access + geometric layers, method proposal) to a content-addressed joblib disk cache, so re-runs and F3's screen-then-reblock reuse them instead of recomputing.

**Architecture:** A new `reblock.cache` module wraps `joblib.Memory` with content-addressed keys: the heavy geometry (`block`, `roads`, `method`) is passed via joblib's `ignore=`, and the key is only lightweight strings — `(block_id, source_content_hash, geos_version, proj_version, code_version, roads_key | method_repr)`. Each `Source` hashes its source file(s) once and stamps every `Block.source_content_hash`; an empty hash (synthetic/test blocks) **bypasses** the cache entirely. Call sites (`KblockSource` build, `KComplexityEval.score`, the method call in `run()`) route through cached wrappers.

**Tech Stack:** Python 3.12, joblib 1.5.x, geopandas/shapely 2.1, numpy, pandas, pyproj, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` must stay green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + `pytest`. Suite is currently 110 tests.
- **Cache keys are content-addressed and lightweight** — `(block_id, source_content_hash, geos_version, proj_version, code_version[, roads_key | method_repr])`. Never hash raw geometry (heavy + GEOS-fragile); pass `block`/`roads`/`method` via joblib `ignore=`.
- **Separate before/after keys** — the before derivation (`roads=None`) and after (`roads=proposal.roads`) MUST use distinct keys (`roads_key="__before__"` vs the proposal's `proposal_id`), or the after wrongly returns the before.
- **GEOS and PROJ versions in every key** — parcel counts are GEOS-sensitive and derivations run on reprojected geometry; a GEOS *or* PROJ upgrade must be a clean miss, not a stale hit.
- **`code_version` in every key** — an auto hash of the cached-derivation module files, so editing derivation logic auto-invalidates (joblib alone only hashes the thin wrapper). No hand-bumped tag.
- **Empty `source_content_hash` bypasses the cache** — compute directly, no store/lookup. Synthetic/test blocks (which don't set it) never cache and can't key-collide.
- **Coarse-but-safe invalidation** — the whole-source-file hash means any source-file edit invalidates all its blocks (accepted tradeoff per spec §6).
- **Cached values must be deterministic and picklable** — `pd.Series`, `GeoDataFrame`, `Block`, `Proposal` all pickle faithfully; the derivations are pure (Task F1 already made `TopologyMethod` RNG side-effect-free).
- **Cache location:** `~/.cache/reblock/derivations/` (XDG user cache, never the repo).
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: Cache foundation — `reblock.cache` + joblib dependency

**Files:**
- Modify: `pyproject.toml` (add `joblib` to dependencies)
- Create: `src/reblock/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces:
  - `SOURCE_HASH_UNSET = ""` — the sentinel meaning "uncacheable block".
  - `source_hash(*paths: Path) -> str` — sha256 over the sorted paths' names + bytes.
  - `memory` — a module-level `joblib.Memory` at `~/.cache/reblock/derivations`.
  - `key_parts() -> tuple[str, str, str]` — `(geos_version, proj_version, code_version)` read live (so tests can monkeypatch).
  - `cached(impl, ignore)` — returns `memory.cache(impl, ignore=ignore)`; impls are module-level functions in the derivation modules (Tasks 3–4), each taking the heavy objects (ignored) plus the lightweight key args.

- [ ] **Step 1: Add joblib to dependencies**

In `pyproject.toml`, add `joblib` to the `[project].dependencies` (or the pixi `[tool.pixi.dependencies]`/`[tool.pixi.pypi-dependencies]` block the project uses — match the existing style for geopandas/networkx). Then:

Run: `pixi install` (or `pixi run python -c "import joblib; print(joblib.__version__)"`)
Expected: joblib importable (≥1.5).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_cache.py`:

```python
from pathlib import Path

import reblock.cache as cache


def test_source_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    h1 = cache.source_hash(a)
    h2 = cache.source_hash(a)
    assert h1 == h2 and h1 != ""
    a.write_bytes(b"HELLO")
    assert cache.source_hash(a) != h1


def test_source_hash_covers_all_paths_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"; a.write_bytes(b"aaa")
    b = tmp_path / "b.bin"; b.write_bytes(b"bbb")
    assert cache.source_hash(a, b) == cache.source_hash(b, a)   # sorted internally
    assert cache.source_hash(a, b) != cache.source_hash(a)


def test_key_parts_reports_live_versions() -> None:
    geos, proj, code = cache.key_parts()
    assert geos and proj and code                     # all non-empty strings
    assert isinstance(geos, str) and isinstance(code, str)


def test_cached_wrapper_hits_and_key_invalidates(tmp_path: Path, monkeypatch) -> None:
    # Point the joblib Memory at a temp dir so the test never touches ~/.cache.
    import joblib
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    calls = {"n": 0}

    def _impl(heavy, *, key: str) -> int:
        calls["n"] += 1
        return len(heavy) + calls["n"] * 0   # value depends only on heavy, keyed on `key`

    fn = cache.cached(_impl, ignore=["heavy"])
    r1 = fn("abcd", key="k1")
    r2 = fn("XXXX", key="k1")   # same key, different (ignored) heavy -> cache HIT, stale-by-design
    assert calls["n"] == 1 and r1 == r2 == 4    # heavy ignored: 2nd call returns cached r1
    fn("abcd", key="k2")        # different key -> recompute
    assert calls["n"] == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run pytest tests/test_cache.py -v`
Expected: FAIL — `No module named 'reblock.cache'`.

- [ ] **Step 4: Implement `src/reblock/cache.py`**

```python
"""L2 per-block derivation cache: a content-addressed joblib.Memory.

Cached derivations are pure functions of a block (and optionally roads/params).
We never hash the heavy geometry (slow + GEOS-fragile); instead each cached
wrapper passes the heavy objects via joblib `ignore=` and keys ONLY on
lightweight strings: (block_id, source_content_hash, geos, proj, code_version,
roads_key | method_repr). An empty source_content_hash bypasses the cache
(synthetic/test blocks). See docs/.../2026-07-07-atomic-flow-...-design.md §6.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pyproj
import shapely

SOURCE_HASH_UNSET = ""

_CACHE_DIR = Path.home() / ".cache" / "reblock" / "derivations"
memory = joblib.Memory(location=str(_CACHE_DIR), verbose=0)

# Modules whose source defines the cached derivations; hashed into every key so
# an edit to derivation logic auto-invalidates (joblib alone only hashes the
# thin wrapper, not its callees). Coarse but safe: any edit to these files
# invalidates all cached derivations.
_DERIVATION_MODULE_FILES = (
    Path(__file__).with_name("cache.py"),
    Path(__file__).parent / "derive" / "access.py",
    Path(__file__).parent / "derive" / "geometric_access.py",
    Path(__file__).parent / "data" / "kblock.py",
    Path(__file__).parent / "methods" / "topology.py",
    Path(__file__).parent / "methods" / "peel.py",
)


def source_hash(*paths: Path) -> str:
    """sha256 over the sorted paths' names + bytes. Stable, content-sensitive,
    order-independent. Used both for a Source's data files and for code_version."""
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        h.update(str(Path(p).name).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


_CODE_VERSION = source_hash(*_DERIVATION_MODULE_FILES)


def key_parts() -> tuple[str, str, str]:
    """(geos_version, proj_version, code_version) — read live so tests can
    monkeypatch and force a clean miss."""
    geos = ".".join(str(x) for x in shapely.geos_version)
    return geos, pyproj.proj_version_str, _CODE_VERSION


def cached(impl: Callable[..., Any], ignore: list[str]) -> Callable[..., Any]:
    """Wrap `impl` with the module joblib.Memory, ignoring the named heavy args
    when computing the cache key."""
    return memory.cache(impl, ignore=ignore)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_cache.py -v`
Expected: PASS (4 tests). Then `pixi run check` (ruff + mypy --strict + full suite) green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/reblock/cache.py tests/test_cache.py
git commit -m "$(cat <<'EOF'
feat: reblock.cache foundation (content-addressed joblib.Memory) (F2)

Adds joblib dep + a Memory at ~/.cache/reblock/derivations, source_hash()
(sha256 over source bytes), key_parts() (geos/proj/code versions read live,
code_version = hash of the cached-derivation module files so logic edits
auto-invalidate), and cached(impl, ignore) to memoize on lightweight keys
while ignoring heavy geometry args.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: `Block.source_content_hash` + Source stamping

**Files:**
- Modify: `src/reblock/contracts.py` (add field to `Block`)
- Modify: `src/reblock/data/kblock.py` (`region()` computes + stamps the hash)
- Modify: `src/reblock/data/shapefile.py` (`region()` computes + stamps the hash)
- Test: `tests/data/test_kblock_source.py`, `tests/data/test_shapefile_source.py`

**Interfaces:**
- Consumes: `reblock.cache.source_hash`.
- Produces: `Block.source_content_hash: str = ""` — a new frozen-dataclass field, defaulting to `""` (uncacheable). Real Sources set it to `source_hash(<their source files>)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_kblock_source.py`:

```python
def test_kblock_blocks_carry_source_content_hash() -> None:
    from reblock.data.kblock import KblockSource
    src = KblockSource(str(BLOCKS), str(BUILDINGS), region_id="kblock")  # existing fixture paths
    blocks = list(src.region().blocks)
    assert blocks, "expected at least one built block"
    h = blocks[0].source_content_hash
    assert h and all(b.source_content_hash == h for b in blocks)  # same hash for all blocks
```

(Use the same fixture path constants the existing tests in that file already define — reuse them; do not invent new ones.)

Add to `tests/data/test_shapefile_source.py` an analogous assertion that a `ShapefileSource`-built block has a non-empty `source_content_hash`, using that file's existing fixture path constant.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/data/test_kblock_source.py -k source_content_hash -v`
Expected: FAIL — `Block` has no attribute `source_content_hash` (or it's `""`).

- [ ] **Step 3: Add the field to `Block`**

In `src/reblock/contracts.py`, add the field to the frozen `Block` dataclass (after `streets`, before `attrs` so the defaulted fields stay last):

```python
@dataclass(frozen=True)
class Block:
    block_id: str
    crs: CRS
    boundary: Polygon
    parcels: GeoDataFrame
    streets: GeoDataFrame
    source_content_hash: str = ""   # content hash of the Source's file(s); "" => uncacheable
    attrs: Mapping[str, object] = field(default_factory=dict)
```

(The default `""` means existing/synthetic block constructions need no change and bypass the L2 cache.)

- [ ] **Step 4: Stamp the hash in both Sources**

In `src/reblock/data/kblock.py` `region()`, compute the hash once and thread it into `_blocks_from`. Import `from reblock.cache import source_hash`. After reading paths, compute `sch = source_hash(self.blocks_path, self.buildings_path)` and pass it through:

```python
    def region(self) -> Region:
        ...
        bld = gpd.read_parquet(self.buildings_path, columns=["geometry"])
        sch = source_hash(self.blocks_path, self.buildings_path)
        return Region(region_id=self.region_id, crs=utm,
                      blocks=self._blocks_from(blocks.to_crs(utm), bld.to_crs(utm), sch))
```

Update `_blocks_from` to accept `source_content_hash: str` and pass it to the `Block(...)` constructor:

```python
    def _blocks_from(self, blocks: gpd.GeoDataFrame, bld: gpd.GeoDataFrame,
                     source_content_hash: str) -> Iterator[Block]:
        ...
            yield Block(block_id=str(row["block_id"]), crs=utm, boundary=poly,
                        parcels=parcels, streets=streets,
                        source_content_hash=source_content_hash,
                        attrs={"kblock_k": float(row["k_complexity"])})
```

In `src/reblock/data/shapefile.py` `region()`, compute `source_hash(self.path)` once and pass `source_content_hash=...` into the `Block(...)` constructor at the existing yield site (mirror the kblock change; thread it through whatever per-block loop/helper the file uses).

- [ ] **Step 5: Run to verify pass**

Run: `pixi run pytest tests/data/test_kblock_source.py tests/data/test_shapefile_source.py -v`
Expected: PASS. Then `pixi run check` green (the pinned kblock value tests are unaffected — the new field doesn't change geometry/parcels).

- [ ] **Step 6: Commit**

```bash
git add src/reblock/contracts.py src/reblock/data/kblock.py src/reblock/data/shapefile.py tests/data/test_kblock_source.py tests/data/test_shapefile_source.py
git commit -m "$(cat <<'EOF'
feat: Block.source_content_hash, stamped by each Source (F2)

Block gains an optional source_content_hash (default "" => uncacheable);
KblockSource and ShapefileSource compute source_hash() over their files once
per load and stamp every block, so the L2 cache can key per-block. Synthetic/
test blocks keep the "" default and bypass the cache.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 3: Cache the access + geometric derivations (before/after) in the eval

**Files:**
- Modify: `src/reblock/cache.py` (add the cached access/geometric wrappers)
- Modify: `src/reblock/eval/kcomplexity.py` (`KComplexityEval.score` routes through them)
- Test: `tests/test_cache.py` (before/after distinct keys; hit; bypass)

**Interfaces:**
- Consumes: `parcel_access_layers(block, roads=None, *, tol=STREET_TOL) -> pd.Series`, `geometric_access_distances(block, roads=None, *, tol=STREET_TOL) -> pd.Series`, `Block.source_content_hash`, `cache.key_parts`, `cache.cached`, `cache.SOURCE_HASH_UNSET`.
- Produces:
  - `cache.cached_access_layers(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series`
  - `cache.cached_geometric(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series`
  - Convention: `roads_key == "__before__"` when `roads is None`; otherwise the proposal's `proposal_id`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache.py` (build a small real-ish block with a fake non-empty hash so caching is exercised, and a counter spy on the underlying derivation):

```python
import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block

_UTM = CRS.from_epsg(32643)


def _grid_block(hash_: str) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=_UTM)
    boundary = parcels.geometry.union_all()
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=_UTM)
    return Block(block_id="g", crs=_UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


def test_cached_access_before_after_use_distinct_keys(tmp_path, monkeypatch) -> None:
    import joblib
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    # rebind the wrappers onto the fresh Memory
    monkeypatch.setattr(cache, "_access_impl_cached", cache.cached(cache._access_impl, ignore=["block", "roads"]))
    calls = {"n": 0}
    real = cache.parcel_access_layers

    def spy(block, roads=None, **kw):
        calls["n"] += 1
        return real(block, roads)
    monkeypatch.setattr(cache, "parcel_access_layers", spy)
    # rebind again so the wrapper closes over the spy
    monkeypatch.setattr(cache, "_access_impl_cached", cache.cached(cache._access_impl, ignore=["block", "roads"]))

    block = _grid_block("deadbeef")
    before1 = cache.cached_access_layers(block, None, "__before__")
    before2 = cache.cached_access_layers(block, None, "__before__")   # HIT
    after = cache.cached_access_layers(block, block.streets, "peel")  # distinct key -> MISS
    assert calls["n"] == 2                          # before computed once, after once
    assert before1.equals(before2)
    # before and after must NOT be the same cached object under one key:
    # (both computed here from the same geometry so may be equal in value, but
    #  the recompute count proves they used different keys)


def test_cached_access_bypasses_when_hash_unset(tmp_path, monkeypatch) -> None:
    import joblib
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    block = _grid_block(cache.SOURCE_HASH_UNSET)  # ""
    out = cache.cached_access_layers(block, None, "__before__")
    assert isinstance(out, pd.Series)
    # bypass path: nothing written to the joblib store
    assert not any(tmp_path.glob("**/*.pkl"))
```

> Implementer note: the exact monkeypatch mechanics for the spy may need adjusting to however the wrappers close over `parcel_access_layers`. The behavioral assertions that MUST hold: (a) two identical before-calls compute once (hit); (b) a before-call and an after-call compute separately (distinct keys — the collapse guard); (c) an unset hash never writes to the joblib store (bypass). Keep those assertions; adapt the plumbing.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cache.py -k "before_after or bypass" -v`
Expected: FAIL — `cache.cached_access_layers` / `cache._access_impl` not defined.

- [ ] **Step 3: Implement the cached wrappers in `cache.py`**

Add to `src/reblock/cache.py`:

```python
from geopandas import GeoDataFrame  # add to imports
import pandas as pd                 # add to imports

from reblock.contracts import Block  # add to imports
from reblock.derive.access import parcel_access_layers
from reblock.derive.geometric_access import geometric_access_distances


def _access_impl(block: Block, roads: GeoDataFrame | None, *, block_id: str,
                 src_hash: str, geos: str, proj: str, code: str, roads_key: str) -> pd.Series:
    return parcel_access_layers(block, roads)


def _geometric_impl(block: Block, roads: GeoDataFrame | None, *, block_id: str,
                    src_hash: str, geos: str, proj: str, code: str, roads_key: str) -> pd.Series:
    return geometric_access_distances(block, roads)


_access_impl_cached = cached(_access_impl, ignore=["block", "roads"])
_geometric_impl_cached = cached(_geometric_impl, ignore=["block", "roads"])


def cached_access_layers(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series:
    if block.source_content_hash == SOURCE_HASH_UNSET:
        return parcel_access_layers(block, roads)
    geos, proj, code = key_parts()
    return _access_impl_cached(block, roads, block_id=block.block_id,
                               src_hash=block.source_content_hash, geos=geos, proj=proj,
                               code=code, roads_key=roads_key)


def cached_geometric(block: Block, roads: GeoDataFrame | None, roads_key: str) -> pd.Series:
    if block.source_content_hash == SOURCE_HASH_UNSET:
        return geometric_access_distances(block, roads)
    geos, proj, code = key_parts()
    return _geometric_impl_cached(block, roads, block_id=block.block_id,
                                  src_hash=block.source_content_hash, geos=geos, proj=proj,
                                  code=code, roads_key=roads_key)
```

Watch for an import cycle: `cache.py` importing `reblock.derive.access` / `reblock.contracts` must not create a cycle (access.py and contracts.py must not import cache.py). They don't today — keep it that way.

- [ ] **Step 4: Route `KComplexityEval.score` through the cached wrappers**

In `src/reblock/eval/kcomplexity.py`, change the three derivation calls in `score` (lines 46-48) to use the cached wrappers with the before/after `roads_key`:

```python
    def score(self, block: Block, proposal: Proposal) -> Metrics:
        after_key = proposal.proposal_id or proposal.method or "__after__"
        pre = cached_access_layers(block, None, "__before__")
        post = cached_access_layers(block, proposal.roads, after_key)
        geo = cached_geometric(block, proposal.roads, after_key)
        ...
```

Add the import `from reblock.cache import cached_access_layers, cached_geometric` and drop the now-unused direct imports of `parcel_access_layers`/`geometric_access_distances` from `score` (keep `street_connectivity`, still called directly). `after_key` falls back to `proposal.method` then `"__after__"` so an empty `proposal_id` still differs from `"__before__"`.

- [ ] **Step 5: Run tests to verify pass**

Run: `pixi run pytest tests/test_cache.py tests/eval -v`
Expected: PASS. Then `pixi run check` green — the kcomplexity eval's existing tests still pass (values unchanged; caching is transparent because synthetic test blocks bypass and real ones return identical results).

- [ ] **Step 6: Commit**

```bash
git add src/reblock/cache.py src/reblock/eval/kcomplexity.py tests/test_cache.py
git commit -m "$(cat <<'EOF'
feat: cache before/after access + geometric derivations (F2)

cached_access_layers/cached_geometric memoize the peel + geometric distances
per (block_id, source_hash, geos, proj, code, roads_key); KComplexityEval.score
routes through them with roads_key="__before__" vs the proposal_id so the
before and after never collapse to one key. Unset source hash bypasses.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 4: Cache the block build + method proposal

**Files:**
- Modify: `src/reblock/cache.py` (cached Voronoi-parcels build + cached propose)
- Modify: `src/reblock/data/kblock.py` (`_blocks_from` uses the cached build)
- Modify: `src/reblock/run.py` (`run()` uses cached propose)
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `_voronoi_parcels(poly, points, crs) -> GeoDataFrame | None` (kblock.py), `Method.propose(block) -> Proposal`.
- Produces:
  - `cache.cached_voronoi_parcels(poly, points, crs, *, block_id, source_content_hash) -> GeoDataFrame | None`
  - `cache.cached_propose(method: Method, block: Block) -> Proposal` — keyed on `(block_id, source_hash, geos, proj, code, repr(method))`; bypasses on unset hash.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache.py`:

```python
def test_cached_propose_hits_and_bypasses(tmp_path, monkeypatch) -> None:
    import joblib
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(cache, "_propose_impl_cached",
                        cache.cached(cache._propose_impl, ignore=["method", "block"]))

    from reblock.methods.topology import TopologyMethod
    m = TopologyMethod(alpha=2.0, seed=0)
    block = _grid_block("cafe1234")
    p1 = cache.cached_propose(m, block)
    p2 = cache.cached_propose(m, block)          # HIT
    assert p1.proposal_id == p2.proposal_id == "topology_a2.0_s0"
    # bypass path when hash unset writes nothing
    monkeypatch.setattr(cache, "memory", joblib.Memory(location=str(tmp_path / "b"), verbose=0))
    monkeypatch.setattr(cache, "_propose_impl_cached",
                        cache.cached(cache._propose_impl, ignore=["method", "block"]))
    cache.cached_propose(m, _grid_block(cache.SOURCE_HASH_UNSET))
    assert not any((tmp_path / "b").glob("**/*.pkl"))
```

> The behavioral assertions that MUST hold: repeated propose on the same (method, block) recomputes once (verify via the returned proposal equality and/or a spy on the method); an unset-hash block bypasses (nothing written). Adapt monkeypatch plumbing as needed.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_cache.py -k propose -v`
Expected: FAIL — `cache.cached_propose` / `cache._propose_impl` not defined.

- [ ] **Step 3: Implement the cached build + propose in `cache.py`**

```python
from shapely.geometry import Point, Polygon  # add to imports
from reblock.contracts import Proposal        # add to imports
from reblock.contracts import Method          # add to imports (Protocol)


def _voronoi_impl(poly: Polygon, points: list[Point], crs: Any, *, block_id: str,
                  src_hash: str, geos: str, proj: str, code: str) -> Any:
    from reblock.data.kblock import _voronoi_parcels   # local import avoids a cycle
    return _voronoi_parcels(poly, points, crs)


def _propose_impl(method: Any, block: Block, *, block_id: str, src_hash: str,
                  geos: str, proj: str, code: str, method_repr: str) -> Proposal:
    return method.propose(block)


_voronoi_impl_cached = cached(_voronoi_impl, ignore=["poly", "points", "crs"])
_propose_impl_cached = cached(_propose_impl, ignore=["method", "block"])


def cached_voronoi_parcels(poly: Polygon, points: list[Point], crs: Any, *,
                           block_id: str, source_content_hash: str) -> Any:
    from reblock.data.kblock import _voronoi_parcels
    if source_content_hash == SOURCE_HASH_UNSET:
        return _voronoi_parcels(poly, points, crs)
    geos, proj, code = key_parts()
    return _voronoi_impl_cached(poly, points, crs, block_id=block_id,
                                src_hash=source_content_hash, geos=geos, proj=proj, code=code)


def cached_propose(method: Method, block: Block) -> Proposal:
    if block.source_content_hash == SOURCE_HASH_UNSET:
        return method.propose(block)
    geos, proj, code = key_parts()
    return _propose_impl_cached(method, block, block_id=block.block_id,
                                src_hash=block.source_content_hash, geos=geos, proj=proj,
                                code=code, method_repr=repr(method))
```

Note: `_voronoi_parcels` is imported locally inside the impl/wrapper to avoid an import cycle (`kblock.py` imports `cache`). Confirm `TopologyMethod`/`PeelReblocker` are `@dataclass`es so `repr(method)` is stable and encodes their params (it is for topology: `TopologyMethod(alpha=2.0, seed=0)`; `PeelReblocker()` has none today — fine).

- [ ] **Step 4: Wire the cached build into `KblockSource`**

In `src/reblock/data/kblock.py` `_blocks_from`, replace the direct `_voronoi_parcels(poly, pts, utm)` call with the cached wrapper, threading the block's `source_content_hash`:

```python
            parcels = cached_voronoi_parcels(poly, pts, utm, block_id=str(row["block_id"]),
                                             source_content_hash=source_content_hash)
```

Add `from reblock.cache import cached_voronoi_parcels` at the top of kblock.py (module-level import is fine; the cycle is only the reverse direction, handled by the local import inside cache.py's impl).

- [ ] **Step 5: Wire the cached propose into `run()`**

In `src/reblock/run.py`, change the per-block `method.propose(block)` call to `cached_propose(method, block)`:

```python
    from reblock.cache import cached_propose   # top-of-file import
    ...
    for block in islice(region.blocks, cfg.max_blocks):
        proposal = cached_propose(method, block)
        metrics = tuple(ev.score(block, proposal) for ev in evals)
        results.append(Result(block=block, proposal=proposal, metrics=metrics))
```

- [ ] **Step 6: Run tests to verify pass + full suite**

Run: `pixi run pytest tests/test_cache.py -v` then `pixi run check`
Expected: PASS. The kblock pinned-value tests and the run/eval tests still pass — real blocks now route through the cache (identical results), synthetic ones bypass. Note: `test_run.py`'s real-data tests (phule/dji/capetown) will populate `~/.cache/reblock/derivations` on first run; that's expected and harmless (deterministic results).

- [ ] **Step 7: Commit**

```bash
git add src/reblock/cache.py src/reblock/data/kblock.py src/reblock/run.py tests/test_cache.py
git commit -m "$(cat <<'EOF'
feat: cache the Voronoi block build + method proposal (F2)

cached_voronoi_parcels memoizes the per-block tessellation (keyed on
block_id+source_hash+versions); cached_propose memoizes method.propose per
(block, repr(method)). KblockSource._blocks_from and run() route through them;
unset source hash bypasses. Completes the three cached derivations (build,
access/geometric before+after, proposal).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 5: Measurement pass — cold-vs-warm timings + cache disk usage

**Files:**
- Create: `scripts/bench_cache.py`
- (No src/ changes; this task produces a measurement report.)

**Interfaces:**
- Consumes: `reblock.cache.memory` (to clear), `reblock.data.provision.cached_kblock_source` (real Cape Town data), `reblock.run.run`, `RunConfig`.

- [ ] **Step 1: Write the benchmark script**

Create `scripts/bench_cache.py` — it reblocks a small real Cape Town multi-block set (a handful of flagged block_ids) COLD (cache cleared) then WARM, timing each and reporting the cache-dir disk footprint. It writes nothing to src/ and is not a test.

```python
"""Benchmark the F2 L2 cache: cold (cleared) vs warm wall-time for a real
Cape Town multi-block reblock, plus the derivation cache's disk footprint.
Usage: PYTHONPATH=. pixi run python scripts/bench_cache.py
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from reblock import cache
from reblock.data.provision import ensure_city_data
from reblock.data.kblock import KblockSource
from reblock.run import RunConfig, run

BLOCK_IDS = ["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_42413", "ZAF.9.3.1_1_21255"]


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _timed_run(blocks_path: Path, buildings_path: Path) -> float:
    cfg = RunConfig(
        max_blocks=len(BLOCK_IDS),
        data={"_target_": "reblock.data.kblock.KblockSource",
              "blocks_path": str(blocks_path), "buildings_path": str(buildings_path),
              "region_id": "capetown", "block_ids": BLOCK_IDS},
        method={"_target_": "reblock.methods.peel.PeelReblocker"},
        eval=[{"_target_": "reblock.eval.kcomplexity.KComplexityEval"}],
    )
    t0 = time.perf_counter()
    run(cfg)
    return time.perf_counter() - t0


def main() -> None:
    blocks_path, buildings_path = ensure_city_data("capetown")
    cache_dir = Path(cache._CACHE_DIR)

    cache.memory.clear(warn=False)
    cold = _timed_run(blocks_path, buildings_path)
    cold_disk = _dir_bytes(cache_dir)

    warm = _timed_run(blocks_path, buildings_path)
    warm_disk = _dir_bytes(cache_dir)

    print(f"blocks: {len(BLOCK_IDS)}  method=peel")
    print(f"COLD (cache cleared): {cold:6.2f}s")
    print(f"WARM (cache hit):     {warm:6.2f}s   speedup {cold / warm:4.1f}x")
    print(f"cache disk: {cold_disk/1e6:6.2f} MB after cold, {warm_disk/1e6:6.2f} MB after warm")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark and capture the report**

Run: `PYTHONPATH=. pixi run python scripts/bench_cache.py`
Expected: prints cold/warm timings + a speedup + cache disk footprint. (First cold run may also trigger a data download if `~/.cache/reblock` isn't populated — that's outside the timed region.)

Record the printed output verbatim in the task report (this is the deliverable of Task 5 — the empirical answer to "how long each phase took and how much disk").

- [ ] **Step 3: Confirm it's lint/type clean and commit**

Run: `pixi run ruff check scripts/bench_cache.py` (scripts/ is not in the mypy path, so ruff-clean suffices).
Then:

```bash
git add scripts/bench_cache.py
git commit -m "$(cat <<'EOF'
feat: bench_cache.py — cold-vs-warm L2 cache benchmark (F2)

Reblocks a small real Cape Town multi-block set cold (cache cleared) then
warm, reporting per-run wall-time, speedup, and the derivation cache's disk
footprint. Measurement deliverable for F2.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (F2 slice of the atomic-flow spec §6):**
- L2 per-block persistent cache (joblib) → Tasks 1,3,4. ✓
- Content-addressed key `(block_id, source_content_hash, geos_version, proj_version[, params])` → Task 1 (`key_parts`) + Tasks 3–4 (wrappers); `code_version` added for logic-edit safety. ✓
- `Block` carries `source_content_hash`, computed once at load → Task 2. ✓
- Separate before/after keys → Task 3 (`roads_key`). ✓
- GEOS **and** PROJ in the key → Task 1 (`key_parts`). ✓
- Coarse whole-source-hash invalidation → Task 1 (`source_hash`), Task 2 (stamped). ✓
- All three cached derivations (build, before/after derivations, proposal) → Tasks 3 (derivations) + 4 (build, proposal). ✓
- Measurement (timings + disk) → Task 5. ✓
- Out of F2 (later): L1 in-process reuse + the compare aggregate (F4); screen stage + flagged-map (F3). ✓

**Placeholder scan:** the two "adapt monkeypatch plumbing as needed" notes in Tasks 3/4 tests are deliberate — the *behavioral* assertions are fully specified (hit computes once; before≠after keys; bypass writes nothing); only the spy-wiring mechanics are left to the implementer because they depend on joblib's rebinding. Every src/ code step shows complete code. No TBD.

**Type consistency:** `cached(impl, ignore)`, `key_parts() -> (geos, proj, code)`, `source_hash(*paths)`, `SOURCE_HASH_UNSET`, `Block.source_content_hash`, `cached_access_layers/cached_geometric(block, roads, roads_key)`, `cached_voronoi_parcels(..., *, block_id, source_content_hash)`, `cached_propose(method, block)` are used identically at every definition and call site (Task 1 defs → Tasks 2–4 call sites). `roads_key` convention (`"__before__"` vs `proposal_id`) is consistent between Task 3's wrapper and the eval.

**Import-cycle check:** `cache.py` imports `reblock.contracts`, `reblock.derive.access`, `reblock.derive.geometric_access` at module level (none import `cache`), and imports `reblock.data.kblock._voronoi_parcels` *locally inside functions* (kblock.py imports `cache` at module level) — so the one back-edge (kblock→cache) is broken by the local import. `run.py` and `eval/kcomplexity.py` import `cache` at module level (cache doesn't import them). No cycle.
