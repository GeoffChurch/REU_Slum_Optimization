# Redesign Layer 2 — composed identity on the data types — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `Block` and `Proposal` a composed content-address `.identity` so `derive()` can key derivations on them, with `None` meaning "uncacheable" (preserving the F2 bypass semantics).

**Architecture:** `Block.identity = (source_content_hash, block_id)` — or `None` when the hash is empty (synthetic/test blocks stay uncacheable, so they never key-collide). `Proposal` gains a `block_identity` field (the identity of the block it was proposed for) and `Proposal.identity = (block_identity, proposal_id)` — `None` if `block_identity` is `None`. `proposal_id` already encodes method+params, so it fully distinguishes proposals for a block. Additive: the fields default so nothing existing breaks; L3 wires the methods to populate `block_identity` and routes derivations through `derive()`.

**Tech Stack:** Python 3.12, dataclasses, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `ruff format --check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently 124 tests.
- **Additive only** — add `.identity` to `Block`, add a defaulted `block_identity` field + `.identity` to `Proposal`. Do NOT change existing field order in a way that breaks positional construction (all construction sites use keywords, but keep defaulted fields last). Do NOT route any derivation through `derive()` yet (that's L3).
- **`None` identity = uncacheable** — `Block.identity` is `None` when `source_content_hash == ""`; `Proposal.identity` is `None` when `block_identity is None`. This matches `derive()`'s bypass-on-`None`-identity, so synthetic/test data never caches or collides.
- **Identity must be hashable** — tuples of strings (and nested identities). No geometry in the identity.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `Block.identity` + `Proposal.identity`

**Files:**
- Modify: `src/reblock/contracts.py` (`Block`, `Proposal`)
- Test: `tests/test_contracts_identity.py` (new)

**Interfaces:**
- Produces:
  - `Block.identity -> tuple[str, str] | None` — `(source_content_hash, block_id)`, or `None` if the hash is empty.
  - `Proposal.block_identity: Hashable | None = None` (new field) and `Proposal.identity -> tuple[Hashable, str] | None` — `(block_identity, proposal_id)`, or `None` if `block_identity is None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_contracts_identity.py`:

```python
from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Proposal

UTM = CRS.from_epsg(32643)


def _block(hash_: str) -> Block:
    parcels = gpd.GeoDataFrame({"parcel_id": [0]},
                               geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


def test_block_identity_composes_hash_and_id() -> None:
    assert _block("deadbeef").identity == ("deadbeef", "b")


def test_block_identity_is_none_when_hash_empty() -> None:
    assert _block("").identity is None            # default => uncacheable


def test_block_identity_is_hashable() -> None:
    hash({_block("h").identity})                  # usable as a dict/joblib key


def test_proposal_identity_from_block_identity_and_proposal_id() -> None:
    p = Proposal(block_id="b", crs=UTM, block_identity=("deadbeef", "b"),
                 proposal_id="topology_a2.0_s0")
    assert p.identity == (("deadbeef", "b"), "topology_a2.0_s0")


def test_proposal_identity_is_none_without_block_identity() -> None:
    assert Proposal(block_id="b", crs=UTM, proposal_id="peel").identity is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_contracts_identity.py -v`
Expected: FAIL — `Block` has no `identity`; `Proposal` has no `block_identity`/`identity`.

- [ ] **Step 3: Add `.identity` to `Block`**

In `src/reblock/contracts.py`, add an `identity` property to the frozen `Block` dataclass (after `__post_init__`):

```python
    @property
    def identity(self) -> tuple[str, str] | None:
        """Content-address for the derivation cache: (source hash, block_id), or
        None when the source hash is unknown (synthetic/test blocks -> uncacheable,
        so they never key-collide). See reblock.derive_graph.derive."""
        return (self.source_content_hash, self.block_id) if self.source_content_hash else None
```

- [ ] **Step 4: Add `block_identity` + `.identity` to `Proposal`**

Add a defaulted `block_identity` field (keep it among the defaulted fields; place it before `params` or after `proposal_id`/`method` — all defaulted, order among them is free) and an `identity` property:

```python
@dataclass(frozen=True)
class Proposal:
    block_id: str
    crs: CRS
    roads: GeoDataFrame | None = None
    edges: GeoDataFrame | None = None
    proposal_id: str = ""
    method: str = ""
    params: Mapping[str, object] = field(default_factory=dict)
    block_identity: Hashable | None = None   # the identity of the block this was proposed for

    @property
    def identity(self) -> tuple[Hashable, str] | None:
        """(block_identity, proposal_id) -- proposal_id encodes method+params, so
        it distinguishes proposals for a block. None when block_identity is unknown."""
        return (self.block_identity, self.proposal_id) if self.block_identity is not None else None
```

Add `Hashable` to the `from collections.abc import ...` import at the top of `contracts.py` (it currently imports `Iterable, Mapping`).

- [ ] **Step 5: Run to verify pass + full check**

Run: `pixi run pytest tests/test_contracts_identity.py -v` then `pixi run check`
Expected: PASS (5 tests). `pixi run check` green — additive fields with defaults don't break existing `Block`/`Proposal` construction (all sites use keywords; the pinned-value and topology tests are unaffected). 129 tests.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/contracts.py tests/test_contracts_identity.py
git commit -m "$(cat <<'EOF'
feat: composed .identity on Block and Proposal (redesign L2)

Block.identity = (source_content_hash, block_id) or None when the hash is
empty (uncacheable); Proposal gains a block_identity field and
Proposal.identity = (block_identity, proposal_id) or None. These are the
content-addresses derive() keys on; None preserves the F2 bypass semantics
so synthetic/test data never caches or collides. Additive -- L3 wires methods
to set block_identity and routes derivations through derive().

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage (L2 of the migration):** "immutable data with identity" — composed `.identity` on `Block` and `Proposal`, `None` = uncacheable (bypass). Additive; no derivation routed through `derive()` yet (L3). ✓

**Placeholder scan:** every step has complete code; no TBD.

**Type consistency:** `Block.identity -> tuple[str,str] | None` and `Proposal.identity -> tuple[Hashable,str] | None` are what L3's `derive(fn, block, proposal)` calls will read; `None` aligns with `derive()`'s bypass-on-`None`. `Proposal.block_identity` is the field L3's methods populate from `block.identity`. `Hashable` imported from `collections.abc`.

**Note on scope:** L2 is deliberately one small additive task — the identity foundation. L3 (the large layer) routes the derivations through `derive()`, populates `Proposal.block_identity` in the methods, and splits `KblockSource` so its Voronoi build is a `derive()` call.
