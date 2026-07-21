# Loop-Closure Candidate Cap Fix + Region Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `LoopClosureRefiner` candidate-cap bias that starves large blocks/regions, and wire a working `clearance_looped` region showcase into the depth example.

**Architecture:** The `max_candidates` cap in `loop_candidates` currently uses `_knn_bounded_pairs`
(each node's k-NEAREST neighbours). The `min_loop_len_m` floor rejects short pairs, so the nearest-k
cap keeps exactly the floor-rejected pairs; with many nodes k→1-2 and the candidate pool starves to
near-zero (6619-parcel block: 8 candidates, commute_ratio ρ 0.013; 11k-parcel region: 63 candidates,
ρ 0.06). Replace it with a **uniform-stride subsample** of the raw index-sorted `query_pairs`, which
preserves the distance distribution so a cap of C yields ~C valid candidates (6619-parcel block ρ
0.013→0.505, external held). Then retune defaults (`max_candidates` 4000→1500, the ρ plateau) and add
a region-example orchestrator override so `clearance_looped` reaches ρ≈0.50 on the depth region.

**Tech Stack:** Python, geopandas/shapely/networkx/scipy, Hydra configs, pixi, pytest.

## Global Constraints

- **Migrate, never accommodate:** REPLACE `_knn_bounded_pairs` entirely — no dual-path, no kept-for-
  compat kNN branch. Rename to `_subsample_pairs` and delete the old implementation and its tests.
- `pixi run check` (ruff lint + mypy --strict + pytest) MUST be green at the end of every task.
- ruff forbids E702 semicolons, E501 lines >100 chars, B905 bare `zip`.
- Continuous colormap only for any example coloring (no scheme/binning) — unchanged here.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x`
- Ground-truth calibration (from the 11k-parcel depth region + 3 member blocks, all external held):
  - uniform cap ρ PLATEAUS for caps ≥1500 (budget-bound past ~1300 valid candidates); cap 1500 is
    ~6× faster than uncapped.
  - default base stays `depth_target=1` (dt=3 UNDER-clears individual blocks: 1353-parcel block dt=3
    → base 298 m / external 0.19 vs dt=1 → 6622 m / external 0.74).
  - default `search_radius_m` stays 45 (sr=45+uniform already recovers large blocks); sr=60 is a
    region-ONLY override (the sparse dt=3 region base needs the wider reach — sr=45 caps region ρ
    ~0.24 even uncapped; sr=60 → ρ 0.515).
  - region showcase recipe = base dt=3/mr=3000 + search_radius_m=60 + budget_frac=0.30 → ρ≈0.515,
    external≈0.954.

---

### Task 1: Uniform-stride candidate subsample (replace kNN cap)

**Files:**
- Modify: `src/reblock/methods/loop_closure.py` (`_knn_bounded_pairs` → `_subsample_pairs`; the cap
  call site + docstring in `loop_candidates`)
- Modify: `tests/test_loop_closure.py` (import line; the three `_knn_bounded_pairs`/cap tests at
  ~L247-301)

**Interfaces:**
- Produces: `_subsample_pairs(pairs: list[tuple[int, int]], max_candidates: int) -> list[tuple[int, int]]`
  — uniform stride over the index-sorted pair list. Replaces `_knn_bounded_pairs(nodes, kdt,
  search_radius_m, max_candidates)`.
- Consumes: `loop_candidates` already computes `pairs = sorted(kdt.query_pairs(search_radius_m))`.

- [ ] **Step 1: Update the failing tests first**

In `tests/test_loop_closure.py`, change the import `_knn_bounded_pairs` → `_subsample_pairs`.
Replace the section header + the two `_knn_bounded_pairs_*` unit tests (~L247-276) with tests for the
new helper. `_grid_nodes()` may be deleted if unused after this (check: it is only used by the two
removed tests). New tests:

```python
# --- _subsample_pairs / max_candidates cap ------------------------------------------------------
# A uniform stride bounds pair volume WITHOUT the nearest-k bias that starved dense meshes (kept only
# the short, floor-rejected pairs). It must preserve the distance SPREAD -- keep long pairs, not just
# short ones -- so real (min_loop_len-clearing) loop-closers survive the cap.

def test_subsample_pairs_noop_when_within_cap() -> None:
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert _subsample_pairs(pairs, max_candidates=10) == pairs


def test_subsample_pairs_shrinks_volume_to_about_cap() -> None:
    pairs = [(0, j) for j in range(1, 101)]           # 100 sorted pairs
    bounded = _subsample_pairs(pairs, max_candidates=10)
    assert 0 < len(bounded) <= 10
    for p in bounded:
        assert p in pairs                             # subset, no fabricated pairs


def test_subsample_pairs_preserves_distance_spread_not_just_shortest() -> None:
    # Regression guard for the kNN starvation bug: the cap must retain pairs from ACROSS the list
    # (which, index-sorted, spans the node-coordinate space), not collapse to the shortest/first few.
    pairs = [(0, j) for j in range(1, 1001)]          # 1000 sorted pairs
    bounded = _subsample_pairs(pairs, max_candidates=50)
    seconds = [j for _i, j in bounded]
    assert min(seconds) < 100 and max(seconds) > 900  # spans low AND high, not just the head
```

The `test_loop_candidates_max_candidates_caps_pairs_and_stays_valid` test (public API) must still
pass, but with uniform stride at `max_candidates=1` only `pairs[0]` survives and may not be the loop.
Change its cap from `1` to a value that keeps the real loop while still capping — use
`max_candidates=4` (the `_gap_block` fixture yields few pairs; 4 keeps the loop yet asserts the cap
path runs) and keep the existing validity assertions. Leave
`test_loop_candidates_max_candidates_none_is_unbounded_default` unchanged.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run python -m pytest tests/test_loop_closure.py -k "subsample or max_candidates" -q`
Expected: FAIL (ImportError on `_subsample_pairs`).

- [ ] **Step 3: Replace `_knn_bounded_pairs` with `_subsample_pairs`**

In `src/reblock/methods/loop_closure.py`, delete `_knn_bounded_pairs` (L128-150) and add:

```python
def _subsample_pairs(pairs: list[tuple[int, int]], max_candidates: int
                     ) -> list[tuple[int, int]]:
    """Uniformly subsample the index-sorted `pairs` to ~`max_candidates`, bounding the pair volume
    handed to `loop_candidates`' expensive per-pair `_snap` + shortest-path WITHOUT biasing which
    pairs survive. The previous nearest-k scheme kept each node's CLOSEST neighbours -- exactly the
    short, low-perimeter pairs the `min_loop_len_m` floor rejects -- so on a dense mesh (many nodes
    -> k collapses to 1-2) it starved the candidate pool to near-zero valid loop-closers (an
    11k-parcel region fell to 63 candidates / commute_ratio 0.06). A uniform stride over the
    index-sorted list preserves the straight-line-distance distribution, so the fraction of pairs
    that clear the loop floor is retained: a cap of C yields ~C valid candidates, not a handful
    (same region -> ~1300 candidates / commute_ratio ~0.50). `pairs` is assumed sorted (as
    `sorted(query_pairs(...))` returns), so the stride samples evenly across the node-index space."""
    if len(pairs) <= max_candidates:
        return pairs
    stride = math.ceil(len(pairs) / max_candidates)
    return pairs[::stride]
```

In `loop_candidates`, change the cap block (L189-191) to:

```python
    pairs = sorted(kdt.query_pairs(search_radius_m))
    if max_candidates is not None and len(pairs) > max_candidates:
        pairs = _subsample_pairs(pairs, max_candidates)
```

- [ ] **Step 4: Rewrite the `max_candidates` paragraph of `loop_candidates`' docstring**

Replace the final paragraph (the `_knn_bounded_pairs` description, ~L174-181) with:

```
    `max_candidates` bounds the PAIR VOLUME `query_pairs` hands to the expensive per-pair `_snap`
    Dijkstra + shortest-path below -- both scale with pair count, so this is where the volume must be
    capped, before either runs. On a dense clearance mesh `query_pairs(search_radius_m)` can return
    tens of thousands of pairs; past `max_candidates`, `_subsample_pairs` takes a uniform stride over
    the index-sorted pairs, preserving the distance distribution so real (floor-clearing) loop-closers
    survive the cap (see `_subsample_pairs`). `None` (the default) leaves `query_pairs`' own radius
    cutoff as the only bound -- unchanged behavior for callers that don't opt in.
```

- [ ] **Step 5: Run tests + full check**

Run: `pixi run python -m pytest tests/test_loop_closure.py -q`
Expected: PASS.
Run: `pixi run check`
Expected: ruff + mypy --strict + full pytest all green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/loop_closure.py tests/test_loop_closure.py
git commit -m "fix: uniform-stride loop-candidate cap (was nearest-k, starved dense meshes)"
```

---

### Task 2: Retune defaults + region-example orchestrator override

**Files:**
- Modify: `src/reblock/methods/loop_closure.py` (`LoopClosureRefiner.max_candidates` default +
  comment)
- Modify: `conf/compare_config.yaml` (`clearance_looped.max_candidates` + stale `search_radius_m`
  comment)
- Modify: `conf/method/loop_closure.yaml` (`max_candidates` + stale `search_radius_m` comment)
- Modify: `scripts/gen_multiblock_example.py` (region override for `clearance_looped`)
- Modify: `tests/test_loop_closure.py` ONLY if a test asserts the old `max_candidates=4000` default
  (search for `4000`; if none, no test change)

**Interfaces:**
- Consumes: `_subsample_pairs` cap from Task 1.
- Produces: default `max_candidates=1500`; region-example `clearance_looped` instantiated with base
  `depth_target=3, max_roads=3000`, `budget_frac=0.30`, `search_radius_m=60`.

- [ ] **Step 1: Lower the dataclass default + fix its comment**

In `src/reblock/methods/loop_closure.py`, change `max_candidates: int | None = 4000` →
`max_candidates: int | None = 1500` and replace its field comment with:

```python
    max_candidates: int | None = 1500
    # Uniform-subsample cap on `loop_candidates`' pair volume (see `_subsample_pairs`): bounds the
    # per-pair `_snap` cost regardless of mesh density. 1500 is the commute_ratio PLATEAU -- caps
    # 1500/2500/4000 all reach ~the same ρ on an 11k-parcel region (budget-bound past ~1300 valid
    # candidates), and 1500 is ~6x faster than uncapped. search_radius_m stays 45 by default (that
    # already recovers large blocks with the uniform cap); the sparse dt=3 REGION base overrides to
    # 60 (see scripts/gen_multiblock_example.py).
```

Leave `search_radius_m: float = 45.0`, `budget_frac: float = 0.12` unchanged; if their comments
mention the old kNN cap or a "20 m" radius, correct them to reference `_subsample_pairs` and 45 m.

- [ ] **Step 2: Retune both config copies**

In `conf/compare_config.yaml` (`clearance_looped`, ~L41-42) and `conf/method/loop_closure.yaml`
(~L13-17): set `max_candidates: 1500` and replace the stale `search_radius_m` comment (it wrongly
says "20 m") with:

```yaml
    # search_radius_m 45 m: sufficient for block-scale bases with the uniform max_candidates cap,
    # which bounds the expensive per-pair snap volume regardless of mesh density (see _subsample_pairs).
    search_radius_m: 45.0
    max_candidates: 1500
```

Keep `base` (`depth_target: 1`), `budget_frac: 0.12` as the block-safe shipped defaults.

- [ ] **Step 3: Add the region override in the orchestrator**

In `scripts/gen_multiblock_example.py`, in the `compose(...)` `overrides=[...]` list (~L61-66),
append these four entries (base matches the example's `clearance` line so the showcase reads as
"clearance + loops"; sr=60/budget=0.30 are the region recipe):

```python
        "all_methods.clearance_looped.base.depth_target=3",
        "all_methods.clearance_looped.base.max_roads=3000",
        "all_methods.clearance_looped.budget_frac=0.30",
        "all_methods.clearance_looped.search_radius_m=60",
```

- [ ] **Step 4: Guard the default value in a test**

Add to `tests/test_loop_closure.py` (near the other refiner tests):

```python
def test_loop_closure_refiner_default_max_candidates_is_the_plateau() -> None:
    # The ρ-plateau default -- documents the calibrated cap so a silent regression to the old
    # starving value is caught.
    r = LoopClosureRefiner(base=cast(Method, _StubMethod()))
    assert r.max_candidates == 1500
```

If `_StubMethod`/a stub `Method` fixture is not already present in the file, reuse the existing fake
used by `test_loop_closure_refiner_identity_*` (search for how those construct `LoopClosureRefiner`)
rather than adding a new one.

- [ ] **Step 5: Run check**

Run: `pixi run check`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/loop_closure.py conf/compare_config.yaml conf/method/loop_closure.yaml scripts/gen_multiblock_example.py tests/test_loop_closure.py
git commit -m "feat: retune loop-closure cap to the ρ plateau + wire region showcase override"
```

---

### Task 3: Regenerate the depth region example

**Files:**
- Modify (generated artifacts): `examples/multiblock_depth/` (maps, GIF, lens CSVs, README, meta)

**Interfaces:**
- Consumes: Tasks 1-2 (the fixed cap + the orchestrator override).

- [ ] **Step 1: Regenerate the example**

Run (from repo root, ~5-10 min; reads the full Cape Town parquet):
`PYTHONPATH=$(pwd) pixi run python -m scripts.gen_multiblock_example depth`
Expected: prints `wrote examples/multiblock_depth: 12 blocks / 11006 parcels ...`.

- [ ] **Step 2: Verify the looped result reached the calibrated ρ**

Read `examples/multiblock_depth/lens_a_external.csv` and the matched-budget lens CSV. Confirm the
`clearance_looped` row shows internal connectivity (commute_ratio) ≈0.4-0.55 and external ≈0.95
(NOT ≈0.01). If ρ is still ≈0.01, STOP and report — the override did not take effect.

- [ ] **Step 3: Sanity-check the artifacts exist**

Confirm `examples/multiblock_depth/` contains the regenerated `after_clearance_looped_*.jpg`,
`reblock_clearance_looped.gif`, and an updated `README.md` referencing clearance_looped's ρ.

- [ ] **Step 4: Commit**

```bash
git add examples/multiblock_depth
git commit -m "docs: regenerate depth region example with looped clearance (ρ≈0.5)"
```

---

## Self-Review

- **Spec coverage:** kNN→uniform fix (Task 1); default retune to plateau + region override (Task 2);
  example regeneration + ρ verification (Task 3). All three ground-truth findings (uniform fix, dt=1
  default base, sr=60 region-only, cap-1500 plateau) are encoded.
- **Placeholder scan:** none — all code blocks are literal.
- **Type consistency:** `_subsample_pairs(pairs, max_candidates)` signature is used identically at
  its definition, the call site, and the tests. `max_candidates` default `1500` matches across the
  dataclass, both YAMLs, and the guard test.
