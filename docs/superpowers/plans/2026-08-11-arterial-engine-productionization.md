# Arterial Engine Productionization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace five conditionally-relevant fields on `GreedyArterialReblocker` with three injected strategies, make `max_anchors` a real cap, and ship the tier-2 shortlist engine so the access objective runs at region scale.

**Architecture:** `arterial.py` + `arterial_lazy.py` become a package with one module per concern. `ArterialEngine` (Exact | Lazy | Shortlist) replaces `lazy`; `ChordRealizer` (SnapToBoundary | IdealChord) replaces `mode` + `lam`; `CandidatePolicySpec` (Grow | Fixed | Faithful) replaces `candidate_policy`. Each is a frozen dataclass built once by Hydra and passed down, so no downstream code asks which one it has.

**Tech Stack:** Python 3.11+, shapely/geopandas, Hydra (`_target_` instantiation), pytest, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-08-11-arterial-engine-productionization-design.md`

## Global Constraints

- **No legacy paths.** When a format or interface changes, migrate and delete the old path. Never add a branch whose only justification is history.
- **Every test must be proven to guard something.** Before accepting a new guard, break the thing it guards and confirm the test fails. Three guards in this plan fail on `main` today — run them against `main` first.
- `pixi run lint` (ruff, line-length 100) and `pixi run typecheck` (mypy --strict) must be green before every commit.
- Run scripts as `pixi run python -m scripts.<name>` — the pythonpath is pytest-only.
- Run tests as `pixi run pytest`. The suite uses `-n auto`; add `-p no:randomly` never — it is not installed.
- **Behaviour must not change** in Tasks 1 and 3–7. The pinned oracles (`test_arterial_proposal_wkt_unchanged`, `test_arterial_parallel_matches_reference_1808`) are the proof. Only Tasks 2 (non-binding cap case) and 9 (access rollout) change any output.
- `_rnd` and `_boundary_graph` come from `reblock.methods.boundary_graph`.
- Commit after every task. Do not squash tasks together.

---

### Task 1: Split `arterial.py` and `arterial_lazy.py` into a package

Pure motion. No signature changes, no behaviour changes. This is the largest blast radius and the lowest risk, because the existing suite is the oracle.

**Files:**
- Create: `src/reblock/methods/arterial/__init__.py`
- Create: `src/reblock/methods/arterial/primitives.py`
- Create: `src/reblock/methods/arterial/realize.py`
- Create: `src/reblock/methods/arterial/scoring.py`
- Create: `src/reblock/methods/arterial/policies.py`
- Create: `src/reblock/methods/arterial/engines.py`
- Create: `src/reblock/methods/arterial/reblocker.py`
- Delete: `src/reblock/methods/arterial.py`, `src/reblock/methods/arterial_lazy.py`
- Modify: `src/reblock/methods/loop_closure.py` (one import line)
- Modify: `tests/methods/test_arterial.py`, `tests/methods/test_arterial_lazy.py`, `tests/test_scoring_equivalence.py`
- Modify: 10 files under `scripts/perf/` (import updates only)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the package layout every later task edits. Public names re-exported from `reblock.methods.arterial`: `GreedyArterialReblocker`, `ArterialIdentity`.

**Exact symbol map** — every top-level symbol, by destination:

| destination | symbols (from `arterial.py` unless noted) |
|---|---|
| `primitives.py` | `_xy`, `_anchor_points`, `_deep_targets`, `_candidate_chords`, `_SnapGraph`, `_snap_graph`, `_merge`, `_explode`, `_planarize`, `_union_with` |
| `realize.py` | `_snap` |
| `scoring.py` | `_score`, `_StepState`, `_STEP_STATE`, `_PARALLEL_THRESHOLD`, `eval_candidate`, `_best_candidate` |
| `policies.py` | from `arterial_lazy.py`: `_road_vertices`, `_committed_gdf`, `_FixedPolicy`, `_GrowPolicy`, `_FaithfulPolicy`, `_make_policy` |
| `engines.py` | `_greedy_arterials`; from `arterial_lazy.py`: `_score_all`, `_iter_live`, `_greedy_arterials_lazy` |
| `reblocker.py` | `ArterialIdentity`, `GreedyArterialReblocker` |

- [ ] **Step 1: Confirm the suite is green before touching anything**

```bash
pixi run pytest tests/methods/test_arterial.py tests/methods/test_arterial_lazy.py -q
```
Expected: PASS. Record the count — it must be identical after the move.

- [ ] **Step 2: Create the package and move the symbols**

```bash
mkdir -p src/reblock/methods/arterial
git mv src/reblock/methods/arterial.py src/reblock/methods/arterial/_old_arterial.py
git mv src/reblock/methods/arterial_lazy.py src/reblock/methods/arterial/_old_lazy.py
```

Then cut each symbol from `_old_arterial.py` / `_old_lazy.py` into its destination module per the table above, carrying its docstring verbatim. Each new module gets `from __future__ import annotations` plus only the imports it actually uses. Delete both `_old_*.py` files when empty.

`_STEP_STATE` is a module-level mutable global read by the fork pool. It moves to `scoring.py` and **all writers must reference it as `scoring._STEP_STATE`**, never via a `from … import _STEP_STATE` binding (that would rebind a copy, and the workers would see `None`).

- [ ] **Step 3: Write `__init__.py` re-exporting the public surface**

```python
"""Greedy arterial reblocking: insert straight arterials one at a time by best gain per cost.

Split into one module per concern -- primitives (geometry and candidate generation), realize (how
a chord becomes a road), scoring (per-candidate evaluation), policies (which candidates the lazy
engine keeps alive), engines (the search strategies), reblocker (the public method). Re-exported
here so `reblock.methods.arterial.GreedyArterialReblocker` keeps resolving from config.
"""
from __future__ import annotations

from reblock.methods.arterial.reblocker import ArterialIdentity, GreedyArterialReblocker

__all__ = ["ArterialIdentity", "GreedyArterialReblocker"]
```

- [ ] **Step 4: Update the one src consumer**

`src/reblock/methods/loop_closure.py` imports `_snap, _snap_graph` from `reblock.methods.arterial`. Change to:

```python
from reblock.methods.arterial.primitives import _snap_graph
from reblock.methods.arterial.realize import _snap
```

- [ ] **Step 5: Update tests and harnesses**

```bash
grep -rln "methods.arterial" --include=*.py tests scripts
```

For each hit, repoint the import at the module that now owns the symbol (use the table in this task). Rename `tests/methods/test_arterial_lazy.py` → `tests/methods/test_engines.py`; its imports of `reblock.methods.arterial_lazy as lz` become `reblock.methods.arterial.engines as lz` and `reblock.methods.arterial.policies`.

- [ ] **Step 6: Run the full suite — it must be identical**

```bash
pixi run pytest -q
pixi run lint && pixi run typecheck
```
Expected: same pass count as Step 1, zero failures. If `test_arterial_proposal_wkt_unchanged` or `test_arterial_parallel_matches_reference_1808` fails, the move changed behaviour — find it before continuing.

- [ ] **Step 7: Commit**

```bash
git add -A src/reblock/methods tests scripts
git commit -m "refactor: split arterial into a package, one module per concern

Pure motion -- no signature or behaviour change. arterial.py (549 lines, mixing
primitives, the exact engine and the public method) and arterial_lazy.py become
primitives/realize/scoring/policies/engines/reblocker, with dependencies flowing one
way. __init__ re-exports so config _target_ paths are unchanged.

The pinned WKT and reference-block oracles pass unchanged, which is what proves
nothing moved."
```

---

### Task 2: `max_anchors` becomes a real cap

Behaviour change, deliberate and scoped: only the **non-binding** case changes. Two of the three guards below fail on `main`.

**Files:**
- Modify: `src/reblock/methods/arterial/primitives.py` (`_anchor_points`)
- Test: `tests/methods/test_arterial.py`

**Interfaces:**
- Consumes: `_anchor_points(network, n, max_anchors=0) -> list[tuple[float, float]]` from Task 1.
- Produces: same signature, new semantics. `len(_anchor_points(net, n, cap)) <= len(_anchor_points(net, n, 0))` for every `cap`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/methods/test_arterial.py`:

```python
def test_max_anchors_never_pessimises() -> None:
    """A cap must never produce MORE anchors than uncapped. Today `max_anchors > 0` REPLACES the
    per-vertex family with arc-length samples, so a cap above the vertex count inflates the set --
    measured 1.69x wall clock at block scale. The name promises a maximum; this makes it one."""
    coords = [(float(i), 0.0 if i % 2 == 0 else 1.0) for i in range(40)]
    net = [LineString(coords)]
    uncapped = _anchor_points(net, n=8, max_anchors=0)
    for cap in (4, 8, 16, 32, 64, 128, 256):
        got = _anchor_points(net, n=8, max_anchors=cap)
        assert len(got) <= len(uncapped), (
            f"max_anchors={cap} produced {len(got)} anchors, more than uncapped's {len(uncapped)}")


def test_max_anchors_above_the_anchor_count_is_a_no_op() -> None:
    """A cap the network never reaches must be exactly uncapped -- not a different anchor family."""
    coords = [(float(i), 0.0 if i % 2 == 0 else 1.0) for i in range(40)]
    net = [LineString(coords)]
    uncapped = _anchor_points(net, n=8, max_anchors=0)
    assert _anchor_points(net, n=8, max_anchors=10_000) == uncapped
```

- [ ] **Step 2: Run them and confirm they fail ON THE CURRENT CODE**

```bash
pixi run pytest tests/methods/test_arterial.py -k "never_pessimises or above_the_anchor_count" -v
```
Expected: **both FAIL.** `never_pessimises` reports 65 anchors at cap=64 against uncapped's 47; the no-op test reports 10000-ish anchors. This is the break-it proof — if either passes now, the test is wrong.

- [ ] **Step 3: Rewrite `_anchor_points` as a composition of the two families**

Replace the whole function in `primitives.py`:

```python
def _merged_lines(network: Sequence[BaseGeometry]) -> tuple[list[BaseGeometry], float]:
    """The network as a flat line list plus its total length. `unary_union` explodes any Multi*
    input, so streets given as a MultiLineString (a block with a hole/courtyard) are handled."""
    merged = unary_union(network)
    lines = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return lines, sum(ln.length for ln in lines)


def _arclength_samples(lines: Sequence[BaseGeometry], total: float,
                       count: int) -> list[tuple[float, float]]:
    """`count` points spaced evenly by arc-length along the network. `_rnd`-snapped, de-duplicated,
    sorted for determinism. Yields ~`count` + one per line, not exactly `count`."""
    pts: set[tuple[float, float]] = set()
    if total > 0 and count > 0:
        step = total / count
        for ln in lines:
            d = 0.0
            while d <= ln.length:
                pts.add(_rnd(_xy(ln.interpolate(d).coords[0])))
                d += step
    return sorted(pts)


def _vertices_and_samples(lines: Sequence[BaseGeometry], total: float,
                          n: int) -> list[tuple[float, float]]:
    """Every network vertex plus `n` arc-length samples -- the uncapped anchor family. Vertices are
    what make committed-segment endpoints anchors, so continuations come for free."""
    pts: set[tuple[float, float]] = {_rnd(_xy(c)) for ln in lines for c in ln.coords}
    pts.update(_arclength_samples(lines, total, n))
    return sorted(pts)


def _anchor_points(network: Sequence[BaseGeometry], n: int,
                    max_anchors: int = 0) -> list[tuple[float, float]]:
    """Anchors for candidate generation: every network vertex plus `n` arc-length samples.

    `max_anchors > 0` is a CAP, not a mode switch. If the uncapped family already fits, it is
    returned untouched; only when it does not does this fall back to ~`max_anchors` arc-length
    samples, and even then it returns whichever set is smaller. So the cap can never inflate the
    anchor count -- which the previous implementation did, replacing the vertex family outright and
    yielding 129 anchors where uncapped gave 39 (measured 1.69x wall clock at block scale).

    When the cap DOES bind the result is the sampled family, exactly as measured in
    notes/2026-08-11-max-anchors-is-a-region-scale-win.md -- subsampling the vertex set instead
    would preserve continuations and might well be better, but it is unmeasured.
    """
    lines, total = _merged_lines(network)
    full = _vertices_and_samples(lines, total, n)
    if max_anchors <= 0 or len(full) <= max_anchors:
        return full
    sampled = _arclength_samples(lines, total, max_anchors)
    return sampled if len(sampled) < len(full) else full
```

- [ ] **Step 4: Run the new tests and the existing one**

```bash
pixi run pytest tests/methods/test_arterial.py -k "anchor" -v
```
Expected: PASS, including `test_anchor_points_max_anchors_caps_and_default_matches_uncapped` unchanged — its fixture uses a *binding* cap (uncapped 47, cap 8), so the new semantics returns the sampled set exactly as before.

- [ ] **Step 5: Run the full suite and the linters**

```bash
pixi run pytest -q && pixi run lint && pixi run typecheck
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/methods/arterial/primitives.py tests/methods/test_arterial.py
git commit -m "fix: max_anchors is a cap, not a mode switch

max_anchors > 0 REPLACED the per-vertex anchor family with arc-length samples and
returned early, so a cap above the vertex count inflated the set -- 129 anchors where
uncapped gives 39, measured 1.69x wall clock at block scale and 4.19x at cap=256. The
name promised a maximum and delivered a mode switch.

Now: build the uncapped family, and only if it exceeds the cap fall back to sampling,
returning whichever set is smaller. Guarantees len(result) <= len(uncapped) including
near the threshold, where the sampled family can come out larger.

The binding branch is unchanged, so every region measurement stands. Only the
non-binding case changes, and nothing tested it -- the existing max_anchors test uses
a binding cap (uncapped 47, cap 8) and passes as written. Both new guards fail on the
previous implementation."
```

---

### Task 3: `ChordRealizer` Protocol and its two implementations

Types only — no wiring. Keeps the diff reviewable and lets Task 4 be a pure substitution.

**Files:**
- Modify: `src/reblock/methods/arterial/realize.py`
- Test: `tests/methods/test_realize.py` (create)

**Interfaces:**
- Consumes: `_snap(chord, sg, lam) -> LineString | None` and `_SnapGraph` from Task 1.
- Produces:
  - `ChordRealizer` Protocol: `realize(chord: LineString, sg: _SnapGraph) -> LineString | None`, `snaps: bool`, `identity: RealizerIdentity`
  - `SnapToBoundary(lam: float = 2.0)`, `IdealChord()`
  - `RealizerIdentity: TypeAlias = SnapToBoundary | IdealChord`

- [ ] **Step 1: Write the failing test**

Create `tests/methods/test_realize.py`:

```python
from __future__ import annotations

from shapely.geometry import LineString

from reblock.methods.arterial.realize import ChordRealizer, IdealChord, SnapToBoundary


def test_ideal_chord_returns_the_chord_untouched() -> None:
    chord = LineString([(0.0, 0.0), (10.0, 10.0)])
    assert IdealChord().realize(chord, sg=None) is chord


def test_realizers_report_whether_they_snap() -> None:
    """`snaps` exists so no consumer has to ask which realizer it holds."""
    assert SnapToBoundary().snaps is True
    assert IdealChord().snaps is False


def test_ideal_chord_identity_carries_no_lam() -> None:
    """lam is meaningless without snapping. Two aspirational configs differing only in lam
    computed identical roads under different cache keys before this."""
    assert IdealChord().identity == IdealChord().identity
    assert SnapToBoundary(lam=2.0).identity != SnapToBoundary(lam=3.0).identity


def test_both_satisfy_the_protocol() -> None:
    realizers: list[ChordRealizer] = [SnapToBoundary(), IdealChord()]
    assert len(realizers) == 2
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pixi run pytest tests/methods/test_realize.py -v
```
Expected: FAIL with `ImportError: cannot import name 'ChordRealizer'`.

- [ ] **Step 3: Add the Protocol and implementations to `realize.py`**

```python
from typing import Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class ChordRealizer(Protocol):
    """How an ideal straight chord becomes the road that is actually scored and committed.

    Injected rather than selected by a `mode` string, so nothing downstream asks which realization
    it has. `snaps` exists for the same reason: the two places outside `realize` that used to test
    `mode == "buildable"` ask the realizer what it does instead of re-deriving it from a name.
    """

    @property
    def snaps(self) -> bool:
        """True when realization follows the parcel-boundary graph, so committed roads planarize
        into the existing network and the incremental scoring branch applies."""

    @property
    def identity(self) -> RealizerIdentity:
        """The proposal-affecting part of this realizer, for the derive-cache key."""

    def realize(self, chord: LineString, sg: _SnapGraph | None) -> LineString | None: ...


@dataclass(frozen=True)
class SnapToBoundary:
    """Buildable realization: the boundary-graph path hugging the ideal line.

    `lam` weights how hard the path hugs the chord (edge cost = length + lam * dist(midpoint,
    chord)). It lives here, on the only strategy that uses it, rather than on the reblocker where
    it was silently dead for aspirational runs -- and dead in the cache key too.
    """

    lam: float = 2.0

    @property
    def snaps(self) -> bool:
        return True

    @property
    def identity(self) -> RealizerIdentity:
        return self          # every field affects the proposal

    def realize(self, chord: LineString, sg: _SnapGraph | None) -> LineString | None:
        assert sg is not None, "SnapToBoundary needs a snap graph"
        return _snap(chord, sg, self.lam)


@dataclass(frozen=True)
class IdealChord:
    """Aspirational realization: the straight chord itself, unsnapped.

    A DIAGNOSTIC that isolates the effect of frontage-snapping, not a universal directness ceiling
    -- see the design doc's correction note.
    """

    @property
    def snaps(self) -> bool:
        return False

    @property
    def identity(self) -> RealizerIdentity:
        return self

    def realize(self, chord: LineString, sg: _SnapGraph | None) -> LineString | None:
        del sg
        return chord


RealizerIdentity: TypeAlias = SnapToBoundary | IdealChord
```

- [ ] **Step 4: Run the test**

```bash
pixi run pytest tests/methods/test_realize.py -v && pixi run lint && pixi run typecheck
```
Expected: PASS, green.

- [ ] **Step 5: Commit**

```bash
git add src/reblock/methods/arterial/realize.py tests/methods/test_realize.py
git commit -m "feat: ChordRealizer protocol -- SnapToBoundary(lam) and IdealChord

Types only, not yet wired. lam moves onto the strategy that uses it: it is read only
inside _snap, which runs only when the mode snaps, so on aspirational runs it was a
dead field -- including in ArterialIdentity, where two configs differing only in lam
got different cache keys for provably identical output.

`snaps` is a property rather than a mode string so the two sites outside realization
that tested mode == 'buildable' can ask the realizer what it does."
```

---

### Task 4: Wire the realizer through; delete `mode` and `lam`

**Files:**
- Modify: `src/reblock/methods/arterial/scoring.py` (`_StepState`, `eval_candidate`)
- Modify: `src/reblock/methods/arterial/engines.py` (`_greedy_arterials`, `_greedy_arterials_lazy`)
- Modify: `src/reblock/methods/arterial/reblocker.py` (`GreedyArterialReblocker`, `ArterialIdentity`)
- Test: `tests/methods/test_arterial.py`

**Interfaces:**
- Consumes: `ChordRealizer`, `SnapToBoundary`, `IdealChord`, `RealizerIdentity` from Task 3.
- Produces: `_StepState.realizer: ChordRealizer` (replacing `.mode` and `.lam`); `GreedyArterialReblocker.realizer: ChordRealizer = SnapToBoundary()`; `ArterialIdentity.realizer: RealizerIdentity`.

- [ ] **Step 1: Write the failing test**

Add to `tests/methods/test_arterial.py`:

```python
def test_lam_does_not_enter_identity_for_the_aspirational_realizer() -> None:
    """IdealChord never snaps, so lam cannot affect its roads. Two such configs must share a
    cache key. Before this they did not, and recomputed identical output under distinct keys."""
    a = GreedyArterialReblocker(objective="directness", realizer=IdealChord())
    b = GreedyArterialReblocker(objective="directness", realizer=IdealChord())
    assert a.identity == b.identity
    # and the snapping realizer's lam MUST still discriminate
    c = GreedyArterialReblocker(objective="directness", realizer=SnapToBoundary(lam=2.0))
    d = GreedyArterialReblocker(objective="directness", realizer=SnapToBoundary(lam=9.0))
    assert c.identity != d.identity


def test_both_realizers_produce_roads() -> None:
    """Replaces test_both_modes_produce_roads. Integration-level check that each realizer is
    actually consulted end to end, on the default engine."""
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, 4.0)], crs=UTM)
    block = _two_arm_block(pts)
    for realizer in (SnapToBoundary(), IdealChord()):
        proposal = GreedyArterialReblocker(
            objective="directness", max_roads=1, realizer=realizer).propose(block)
        assert len(proposal.roads) >= 1
        assert proposal.params["realizer"] == type(realizer).__name__
```

**Note on what is deliberately NOT tested here.** An earlier draft asserted that swapping the
realizer changes the resulting WKT. On `_two_arm_block` the arms are 1-wide columns, so a
boundary-graph path and a straight chord can legitimately coincide — the assertion would be
fragile rather than wrong-detecting. Realizer sensitivity is proven deterministically by the unit
tests in Task 3 (`IdealChord().realize(chord, sg) is chord`, and `SnapToBoundary` delegating to
`_snap`), which is where it belongs.

Replace the existing `test_both_modes_produce_roads` with the version above rather than adding
alongside it. Build blocks inline via the file's existing `_two_arm_block` helper — this module has
no block fixture.

- [ ] **Step 2: Run it and confirm it fails**

```bash
pixi run pytest tests/methods/test_arterial.py -k "lam_does_not_enter or swapping_the_realizer" -v
```
Expected: FAIL — `GreedyArterialReblocker` has no `realizer` keyword yet.

- [ ] **Step 3: Replace `mode`/`lam` with `realizer` in `_StepState` and `eval_candidate`**

In `scoring.py`, in `_StepState`, delete the `mode: str` and `lam: float` fields and add:

```python
    realizer: ChordRealizer
```

In `eval_candidate`, replace the three `mode`/`lam` reads:

```python
    real = st.realizer.realize(chord, st.sg)
    if real is None or real.length == 0:
        return 0.0, None
    trial: GeoDataFrame | None = None
    if st.step is not None:
        e, direct = st.step.score_candidate(real)
        raw = (e if st.objective == "efficiency" else direct) - st.base_val
    elif st.realizer.snaps:
        trial = _explode(_union_with(st.base_merged, real), st.crs, 2.0 * st.half_width_m)
        raw = _score(st.objective, st.block, trial, st.adj, st.base_burden, st.ctx) - st.base_val
    else:
        trial = _planarize(st.committed + [real], st.crs, 2.0 * st.half_width_m)
        raw = _score(st.objective, st.block, trial, st.adj, st.base_burden, st.ctx) - st.base_val
```

- [ ] **Step 4: Replace the remaining `mode` read in both engines**

In `engines.py`, both `_greedy_arterials` and `_greedy_arterials_lazy` take `mode: str, … lam: float` — replace both parameters with `realizer: ChordRealizer`. The step-context line becomes:

```python
        step = ctx.step(base) if (ctx is not None and realizer.snaps) else None
```

and the `_StepState(...)` construction passes `realizer=realizer` instead of `mode=mode, lam=lam`.

- [ ] **Step 5: Replace the fields on the reblocker**

In `reblocker.py`, delete `mode: str = "buildable"` and `lam: float = 2.0` from `GreedyArterialReblocker` and add:

```python
    realizer: ChordRealizer = SnapToBoundary()
```

In `ArterialIdentity`, delete `mode: str` and `lam: float` and add `realizer: RealizerIdentity`. In the `identity` property, pass `realizer=self.realizer.identity`. In `propose`, pass `realizer=self.realizer` to the engine call, and drop `mode`/`lam` from the `Proposal.params` dict, replacing `"mode": self.mode` with `"realizer": type(self.realizer).__name__`.

- [ ] **Step 6: Run the tests**

```bash
pixi run pytest tests/methods -q && pixi run lint && pixi run typecheck
```
Expected: PASS. `test_arterial_proposal_wkt_unchanged` must still pass — the buildable default is `SnapToBoundary(lam=2.0)`, identical to the old `mode="buildable", lam=2.0`.

- [ ] **Step 7: Commit**

```bash
git add -A src/reblock/methods/arterial tests/methods
git commit -m "refactor: inject ChordRealizer; delete mode and lam

mode was a string switched on in three places and lam was a field that did nothing
unless mode snapped -- including in the cache key, so two aspirational configs
differing only in lam recomputed identical roads. Both are now one injected strategy,
and the two sites outside realization ask realizer.snaps instead of re-deriving
behaviour from a name.

Cache keys change here, so the derive cache starts missing. Harmless -- it is
content-addressed, so a key change is a miss, not a failure."
```

---

### Task 5: Inject `CandidatePolicySpec`; delete `_make_policy`

**Files:**
- Modify: `src/reblock/methods/arterial/policies.py`
- Modify: `src/reblock/methods/arterial/engines.py` (`_greedy_arterials_lazy`)
- Test: `tests/methods/test_policies.py` (create)

**Interfaces:**
- Consumes: `_FixedPolicy`, `_GrowPolicy`, `_FaithfulPolicy` from Task 1.
- Produces:
  - `CandidatePolicy` Protocol: `initial() -> list[LineString]`, `after_commit(committed, step) -> tuple[list[LineString], list[str]]`
  - `CandidatePolicySpec` Protocol: `build(block, streets, n_anchors, top_k, adj, max_anchors) -> CandidatePolicy`
  - `Grow()`, `Fixed()`, `Faithful()` — frozen, no fields, each its own identity
  - `PolicyIdentity: TypeAlias = Grow | Fixed | Faithful`

- [ ] **Step 1: Write the failing test**

Create `tests/methods/test_policies.py`:

```python
from __future__ import annotations

import pytest

from reblock.methods.arterial.policies import Faithful, Fixed, Grow


def test_specs_are_their_own_identity_and_distinct() -> None:
    assert Grow().identity == Grow().identity
    assert Grow().identity != Fixed().identity
    assert Fixed().identity != Faithful().identity


def test_there_is_no_string_factory_left() -> None:
    """_make_policy resolved a closed set of three by string, with a ValueError fallback that a
    typo reached at runtime instead of at type-check time."""
    from reblock.methods.arterial import policies
    assert not hasattr(policies, "_make_policy")
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pixi run pytest tests/methods/test_policies.py -v
```
Expected: FAIL with `ImportError: cannot import name 'Faithful'`.

- [ ] **Step 3: Add the Protocols and specs to `policies.py`**

```python
@runtime_checkable
class CandidatePolicy(Protocol):
    """Which candidates the lazy engine keeps alive as roads commit. Stateful, per block."""

    def initial(self) -> list[LineString]: ...
    def after_commit(self, committed: list[LineString],
                     step: int) -> tuple[list[LineString], list[str]]: ...


@runtime_checkable
class CandidatePolicySpec(Protocol):
    """The CONFIGURABLE half of a policy. The policies themselves close over block state (block,
    adjacency, seed candidates), which is per-proposal and cannot be built where config is read --
    so config injects a spec and the engine calls `build` once per block."""

    @property
    def identity(self) -> PolicyIdentity: ...

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy: ...


def _seed(block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
          adj: list[set[int]], max_anchors: int
          ) -> tuple[list[tuple[float, float]], list[LineString]]:
    anchors0 = _anchor_points(streets, n_anchors, max_anchors)
    targets0 = _deep_targets(block, None, top_k, adj)
    return anchors0, _candidate_chords(anchors0, targets0)


@dataclass(frozen=True)
class Fixed:
    """Score only the step-0 candidate set forever. Cheapest, and blind to continuations."""

    @property
    def identity(self) -> PolicyIdentity:
        return self

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy:
        _, initial = _seed(block, streets, n_anchors, top_k, adj, max_anchors)
        return _FixedPolicy(initial)


@dataclass(frozen=True)
class Grow:
    """Add continuations from each committed road's vertices as they appear. The shipped default."""

    @property
    def identity(self) -> PolicyIdentity:
        return self

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy:
        anchors0, initial = _seed(block, streets, n_anchors, top_k, adj, max_anchors)
        return _GrowPolicy(block, adj, list(anchors0), top_k,
                           {ls.wkt for ls in initial}, initial)


@dataclass(frozen=True)
class Faithful:
    """Regenerate the exact greedy's candidate set every step. With rescore_every=1 this makes the
    lazy engine byte-identical to the exact one -- the oracle the lazy path is checked against."""

    @property
    def identity(self) -> PolicyIdentity:
        return self

    def build(self, block: Block, streets: list[BaseGeometry], n_anchors: int, top_k: int,
              adj: list[set[int]], max_anchors: int) -> CandidatePolicy:
        _, initial = _seed(block, streets, n_anchors, top_k, adj, max_anchors)
        return _FaithfulPolicy(block, list(streets), n_anchors, adj, top_k,
                               {ls.wkt for ls in initial}, initial, max_anchors)


PolicyIdentity: TypeAlias = Fixed | Grow | Faithful
```

Then **delete `_make_policy` entirely.**

- [ ] **Step 4: Take the spec in the lazy engine**

In `engines.py`, change `_greedy_arterials_lazy`'s `candidate_policy: str = "grow"` parameter to `policy_spec: CandidatePolicySpec`, and replace the `_make_policy(...)` call with:

```python
    policy = policy_spec.build(block, streets, n_anchors, top_k, adj, max_anchors)
```

- [ ] **Step 5: Run the tests**

```bash
pixi run pytest tests/methods -q && pixi run lint && pixi run typecheck
```
Expected: PASS, including `test_lazy_faithful_rescore1_equals_exact` — the existing lazy≡exact oracle, now driven by `Faithful()`.

- [ ] **Step 6: Commit**

```bash
git add -A src/reblock/methods/arterial tests/methods
git commit -m "refactor: inject CandidatePolicySpec; delete the string factory

_make_policy resolved a closed set of three by string with a ValueError fallback --
a registry with extra steps, reached at runtime instead of at type-check time. The
three policies already shared an identical interface, so the conversion is nearly
free.

Split into spec and instance because the policies close over block state (block,
adjacency, seed candidates), which is per-proposal and cannot be built where config
is read. Config injects the spec; the engine calls build() once per block."
```

---

### Task 6: Inject `ArterialEngine`; delete `lazy`, `candidate_policy`, `rescore_every`

**Files:**
- Modify: `src/reblock/methods/arterial/engines.py`
- Modify: `src/reblock/methods/arterial/reblocker.py`
- Test: `tests/methods/test_engines.py`

**Interfaces:**
- Consumes: `_greedy_arterials`, `_greedy_arterials_lazy` (Task 1), `ChordRealizer` (Task 3), `CandidatePolicySpec` (Task 5).
- Produces:
  - `ArterialEngine` Protocol: `run(block, *, objective, cost, realizer, n_anchors, top_k, max_roads, half_width_m, workers, max_anchors) -> GeoDataFrame`, `identity: EngineIdentity`
  - `ExactEngine()`, `LazyEngine(policy: CandidatePolicySpec = Grow(), rescore_every: int = 0)`
  - `EngineIdentity: TypeAlias = ExactEngine | LazyEngine | ShortlistIdentity` (ShortlistIdentity added in Task 7)

- [ ] **Step 1: Write the failing test**

Add to `tests/methods/test_engines.py`:

```python
def test_engines_are_their_own_identity_and_discriminate() -> None:
    assert ExactEngine().identity != LazyEngine().identity
    assert LazyEngine(rescore_every=0).identity != LazyEngine(rescore_every=1).identity
    assert LazyEngine(policy=Grow()).identity != LazyEngine(policy=Faithful()).identity


def test_reblocker_has_no_engine_flags_left() -> None:
    """lazy + candidate_policy + rescore_every jointly picked an engine; a fourth would have made
    it worse. They are one injected instance now."""
    fields = {f.name for f in dataclasses.fields(GreedyArterialReblocker)}
    assert {"lazy", "candidate_policy", "rescore_every", "mode", "lam"} & fields == set()
    assert {"engine", "realizer"} <= fields
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pixi run pytest tests/methods/test_engines.py -k "own_identity or no_engine_flags" -v
```
Expected: FAIL with `ImportError: cannot import name 'ExactEngine'`.

- [ ] **Step 3: Add the Protocol and the two engines**

In `engines.py`:

```python
@runtime_checkable
class ArterialEngine(Protocol):
    """How the greedy searches: which candidates get scored exactly, each step.

    Injected rather than picked by three co-dependent flags. Every engine reuses the same scoring
    machinery (`eval_candidate`, `_STEP_STATE`, the fork pool) -- only candidate selection differs.
    """

    @property
    def identity(self) -> EngineIdentity: ...

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame: ...


@dataclass(frozen=True)
class ExactEngine:
    """Score every candidate, every step. The reference path every other engine is checked against."""

    @property
    def identity(self) -> EngineIdentity:
        return self

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame:
        return _greedy_arterials(
            block, objective=objective, cost=cost, realizer=realizer, n_anchors=n_anchors,
            top_k=top_k, max_roads=max_roads, half_width_m=half_width_m, workers=workers,
            max_anchors=max_anchors)


@dataclass(frozen=True)
class LazyEngine:
    """CELF lazy-greedy: drive selection with a max-heap of stale upper bounds.

    VALID ONLY FOR SUBMODULAR OBJECTIVES. That holds for directness; it does NOT hold for
    access-burden reduction, where it was measured diverging from the exact greedy on 6 of 6 blocks
    and SLOWER in 4 of 6 -- an approximation for no speed. Use ShortlistEngine for access.
    """

    policy: CandidatePolicySpec = Grow()
    rescore_every: int = 0          # 0 = pure lazy; N = full re-score every N commits

    @property
    def identity(self) -> EngineIdentity:
        return self

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame:
        return _greedy_arterials_lazy(
            block, objective=objective, cost=cost, realizer=realizer, n_anchors=n_anchors,
            top_k=top_k, max_roads=max_roads, half_width_m=half_width_m, workers=workers,
            policy_spec=self.policy, rescore_every=self.rescore_every, max_anchors=max_anchors)
```

- [ ] **Step 4: Replace the dispatch on the reblocker**

In `reblocker.py`, delete `lazy`, `candidate_policy`, `rescore_every` from `GreedyArterialReblocker` and add `engine: ArterialEngine = ExactEngine()`. Replace the whole `if self.lazy: … else: …` block in `propose` with:

```python
        roads = self.engine.run(
            block, objective=self.objective, cost=self.cost, realizer=self.realizer,
            n_anchors=self.n_anchors, top_k=self.top_k, max_roads=self.max_roads,
            half_width_m=self.road_width_m / 2.0, workers=self.workers,
            max_anchors=self.max_anchors)
```

In `ArterialIdentity`, delete `lazy`, `candidate_policy`, `rescore_every` and add `engine: EngineIdentity`; the property passes `engine=self.engine.identity`. Replace `"lazy": self.lazy` in `Proposal.params` with `"engine": type(self.engine).__name__`.

- [ ] **Step 5: Run the tests**

```bash
pixi run pytest tests/methods -q && pixi run lint && pixi run typecheck
```
Expected: PASS, including `test_lazy_faithful_rescore1_equals_exact` via `LazyEngine(policy=Faithful(), rescore_every=1)`.

- [ ] **Step 6: Commit**

```bash
git add -A src/reblock/methods/arterial tests/methods
git commit -m "refactor: inject ArterialEngine; delete lazy/candidate_policy/rescore_every

Three fields jointly picked an engine and were dispatched on in propose; a fourth for
the shortlist would have made it worse. One injected instance now, resolved where
config is read, and downstream code never asks which engine it has.

rescore_every and the policy move onto LazyEngine, where they always apply -- they
were meaningless unless lazy was true."
```

---

### Task 7: `ShortlistEngine`, and delete the duplicate harness

**Files:**
- Modify: `src/reblock/methods/arterial/engines.py`
- Create: `src/reblock/methods/arterial/shortlist.py`
- Delete: `scripts/perf/shortlist_greedy.py`, `scripts/perf/control_check.py`
- Modify: `scripts/perf/{selectors,null_model,stochastic_restarts,region_shortlist,shortlist_ab}.py`
- Test: `tests/methods/test_engines.py`

**Interfaces:**
- Consumes: `ArterialEngine` (Task 6), `_anchor_points`/`_candidate_chords`/`_deep_targets` (Task 1).
- Produces: `ShortlistEngine(k: int = 512, threads: int = 8)`, `ShortlistIdentity(k: int)`, and `first_order_score(chords, ctx, threads) -> np.ndarray` in `shortlist.py`.

- [ ] **Step 1: Write the failing test — the control oracle**

Add to `tests/methods/test_engines.py`:

```python
def test_shortlist_with_non_binding_k_is_the_exact_engine() -> None:
    """The shortlist re-states the exact step loop so an injected ranking can cut the candidate
    list mid-loop. With k above every step's candidate count it must reduce to the exact greedy
    EXACTLY -- the two have separate copies of a dozen per-step setup lines, and dropping one
    (committed_disp, base_val, the step context) changes scores silently rather than crashing.

    Uses the access objective with cost=displacement because that is the combination the shortlist
    exists for, and _two_arm_block supplies the building points displacement needs."""
    pts = gpd.GeoDataFrame(geometry=[Point(0.5, y) for y in range(1, 8)] + [Point(10.5, 4)],
                           crs=UTM)
    block = _two_arm_block(pts)
    kw: dict[str, object] = dict(objective="access", cost="displacement",
                                 realizer=SnapToBoundary(), max_roads=3,
                                 road_width_m=DEFAULT_ROAD_WIDTH_M, workers=2)
    want = GreedyArterialReblocker(engine=ExactEngine(), **kw).propose(block)          # type: ignore[arg-type]
    got = GreedyArterialReblocker(engine=ShortlistEngine(k=10_000_000), **kw).propose(block)  # type: ignore[arg-type]
    assert [g.wkt for g in got.roads.geometry] == [g.wkt for g in want.roads.geometry]


def test_shortlist_threads_do_not_enter_identity() -> None:
    """threads is a parallelism knob, same category as workers -- it cannot change the roads, so
    it must not split the cache key."""
    assert ShortlistEngine(k=512, threads=1).identity == ShortlistEngine(k=512, threads=8).identity
    assert ShortlistEngine(k=512).identity != ShortlistEngine(k=256).identity
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
pixi run pytest tests/methods/test_engines.py -k shortlist -v
```
Expected: FAIL with `ImportError: cannot import name 'ShortlistEngine'`.

- [ ] **Step 3: Move the ranking and the selector seam into `shortlist.py`**

Copy `first_order_score`, `RankContext`, `CHUNK`, `RANK_RADIUS` **and the `FirstOrder` selector** from `scripts/perf/selectors.py` into `src/reblock/methods/arterial/shortlist.py` verbatim, keeping their docstrings — they carry the measured justification (weights are `d² − 1` because the greedy optimizes `Σd²`; threads saturate at 8 and degrade at 16).

Also define the seam the research harnesses need, so there is **one** implementation of the step loop rather than a production copy and a research copy:

```python
@runtime_checkable
class CandidateSelector(Protocol):
    """Which of a step's candidates get scored exactly. Production always uses `FirstOrder`; the
    seam exists because the research harnesses in scripts/perf compare selectors against each
    other (uniform-random as a null model, stochastic draws for best-of-R restarts) and would
    otherwise need their own copy of the step loop -- a duplicate that would silently drift from
    production, which is exactly the hazard that kept arterial_incremental.py untracked."""

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]: ...
```

- [ ] **Step 4: Add `_greedy_shortlist` and `ShortlistEngine` to `engines.py`**

Port `greedy_shortlist` from `scripts/perf/shortlist_greedy.py` into `engines.py` as a module-level function `_greedy_shortlist`, replacing its `mode`/`lam` parameters with `realizer: ChordRealizer` (matching Task 4) and keeping its `selector: CandidateSelector` parameter. `ShortlistEngine.run` supplies `FirstOrder(self.k, self.threads)`:

```python
@dataclass(frozen=True)
class ShortlistEngine:
    """Tier 2: rank every candidate by a cheap first-order estimate, score only the top `k` exactly.

    Needed because CELF is invalid for the access objective (not submodular). Ranks by
    (sum of d^2-1 over parcels the chord fronts) / (buildings in its corridor) -- chosen by
    measurement: against exact displacement it correlates +0.92 where chord length manages +0.65,
    for the same single bulk `dwithin`.

    `k=512` is the value every region result was measured at. Saturation bounds it only from ABOVE:
    512/1024/2048/4096 produce a bit-identical network, so overshooting is free and the unmeasured
    direction is downward. `threads=8` is the measured optimum (354.9 s at 1, 104.3 s at 8, and
    134.0 s at 16 -- the STRtree query is memory-bandwidth bound). At block scale threads is a
    no-op by construction: a few thousand candidates is one chunk.
    """

    k: int = 512
    threads: int = 8

    @property
    def identity(self) -> EngineIdentity:
        return ShortlistIdentity(k=self.k)      # threads cannot change the roads

    def run(self, block: Block, *, objective: str, cost: str, realizer: ChordRealizer,
            n_anchors: int, top_k: int, max_roads: int, half_width_m: float,
            workers: int, max_anchors: int) -> GeoDataFrame:
        return _greedy_shortlist(
            block, objective=objective, cost=cost, realizer=realizer, n_anchors=n_anchors,
            top_k=top_k, max_roads=max_roads, half_width_m=half_width_m, workers=workers,
            max_anchors=max_anchors, selector=FirstOrder(self.k, threads=self.threads))


@dataclass(frozen=True)
class ShortlistIdentity:
    """ShortlistEngine's proposal-affecting part. `threads` is excluded deliberately."""

    k: int
```

Add `ShortlistIdentity` to the `EngineIdentity` union.

- [ ] **Step 5: Run the oracle, then break it to prove it guards something**

```bash
pixi run pytest tests/methods/test_engines.py -k shortlist -v
```
Expected: PASS. Then delete the `committed_disp=…` line from `_greedy_shortlist`'s `_StepState` construction, re-run, and confirm the oracle **FAILS**. Restore it.

- [ ] **Step 6: Delete the duplicate harness and repoint the rest**

```bash
git rm scripts/perf/shortlist_greedy.py scripts/perf/control_check.py
```

`control_check.py`'s oracle is now the test from Step 1 — that is where it belonged, since it is the only thing standing between a dropped per-step setup line and silently wrong scores.

`scripts/perf/selectors.py` **keeps** `ScoreAll`, `RandomSample` and `StochasticFirstOrder` — they are research arms with no production use — and deletes `FirstOrder`, `first_order_score`, `RankContext`, `CHUNK`, `RANK_RADIUS`, importing them from `reblock.methods.arterial.shortlist` instead. All four research selectors satisfy the production `CandidateSelector` Protocol.

In `scripts/perf/{null_model,stochastic_restarts,region_shortlist,shortlist_ab}.py`, replace

```python
from scripts.perf.shortlist_greedy import greedy_shortlist
```
with
```python
from reblock.methods.arterial.engines import _greedy_shortlist
```

and update each call site: `mode="buildable"` becomes `realizer=SnapToBoundary()`, and `greedy_shortlist(...)` becomes `_greedy_shortlist(...)`. The `selector=` argument is unchanged — that seam is why these harnesses do not need their own step loop.

- [ ] **Step 6b: Prove the harnesses still run**

```bash
pixi run python -m scripts.perf.shortlist_ab 2>&1 | tail -5
```
Expected: it runs and prints rows. If it cannot import, the seam is wrong — fix before committing rather than leaving a broken harness in the tree.

- [ ] **Step 7: Run everything**

```bash
pixi run pytest -q && pixi run lint && pixi run typecheck
```
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add -A src scripts tests
git commit -m "feat: ShortlistEngine -- tier 2 as a production engine

Ranks candidates by first-order access gain per building displaced and scores only the
top k exactly. Needed because CELF is invalid for the access objective, which is not
submodular. k=512 is where every region result was measured; saturation bounds it only
from above.

scripts/perf/shortlist_greedy.py is DELETED rather than kept -- once the engine ships
it is a duplicate of production logic, the same drift hazard that kept
arterial_incremental.py untracked. control_check.py's oracle becomes a test, which is
where it belonged: it is the only thing standing between a dropped per-step setup line
and silently wrong scores."
```

---

### Task 8: Migrate the 9 config sites

**Files:**
- Modify: `conf/method/greedy_arterial.yaml`, `greedy_arterial_displacement.yaml`, `greedy_arterial_repulsion.yaml`
- Modify: `conf/compare_config.yaml` (6 inline entries, lines 19–20 and 48–51)
- Test: `tests/methods/test_arterial.py` (`test_config_and_derivation_wiring`, `test_displacement_config_instantiates_with_right_params_and_identity`)

**Interfaces:**
- Consumes: every type from Tasks 3–7.
- Produces: configs that instantiate. No later task depends on their contents beyond Task 9's method lists.

- [ ] **Step 1: Update the two config tests to the new field names**

`test_config_and_derivation_wiring` and `test_displacement_config_instantiates_with_right_params_and_identity` assert on `mode`/`lazy`/`lam`. Rewrite their assertions against `realizer` and `engine`, e.g.:

```python
    assert isinstance(m.realizer, SnapToBoundary)
    assert isinstance(m.engine, LazyEngine)
    assert isinstance(m.engine.policy, Grow)
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
pixi run pytest tests/methods/test_arterial.py -k "config_and_derivation or displacement_config" -v
```
Expected: FAIL — the YAML still sets deleted fields, so Hydra raises on unexpected keyword.

- [ ] **Step 3: Migrate `conf/method/*.yaml`**

In each of the three files, delete the `mode`, `lam`, `lazy`, `candidate_policy` and `rescore_every` keys and add (keeping each file's existing `objective`/`cost`/`max_roads`):

```yaml
realizer:
  _target_: reblock.methods.arterial.SnapToBoundary
  lam: 2.0
engine:
  _target_: reblock.methods.arterial.LazyEngine
  policy:
    _target_: reblock.methods.arterial.Grow
max_anchors: 0           # 0 = no cap. A cap only ever REDUCES the anchor set now.
```

`greedy_arterial_displacement.yaml` is the aspirational one — use `_target_: reblock.methods.arterial.IdealChord` with **no `lam` key** and `engine: {_target_: reblock.methods.arterial.ExactEngine}`.

- [ ] **Step 4: Migrate the 6 entries in `conf/compare_config.yaml`**

```yaml
  greedy_arterial_buildable: {_target_: reblock.methods.arterial.GreedyArterialReblocker, objective: directness, max_roads: 15, realizer: {_target_: reblock.methods.arterial.SnapToBoundary}, engine: {_target_: reblock.methods.arterial.LazyEngine, policy: {_target_: reblock.methods.arterial.Grow}}}
  greedy_arterial_repulsion: {_target_: reblock.methods.arterial.GreedyArterialReblocker, objective: directness, cost: repulsion, max_roads: 15, realizer: {_target_: reblock.methods.arterial.SnapToBoundary}, engine: {_target_: reblock.methods.arterial.LazyEngine, policy: {_target_: reblock.methods.arterial.Grow}}}
  greedy_arterial_access_repulsion: {_target_: reblock.methods.arterial.GreedyArterialReblocker, objective: access, cost: repulsion, max_roads: 15, max_anchors: 128, realizer: {_target_: reblock.methods.arterial.SnapToBoundary}, engine: {_target_: reblock.methods.arterial.ShortlistEngine, k: 512}}
  greedy_arterial_access_displacement: {_target_: reblock.methods.arterial.GreedyArterialReblocker, objective: access, cost: displacement, max_roads: 15, max_anchors: 128, realizer: {_target_: reblock.methods.arterial.SnapToBoundary}, engine: {_target_: reblock.methods.arterial.ShortlistEngine, k: 512}}
  greedy_arterial_aspirational: {_target_: reblock.methods.arterial.GreedyArterialReblocker, objective: directness, realizer: {_target_: reblock.methods.arterial.IdealChord}, engine: {_target_: reblock.methods.arterial.ExactEngine}}
  greedy_arterial_displacement: {_target_: reblock.methods.arterial.GreedyArterialReblocker, objective: directness, cost: displacement, road_width_m: 7.0, realizer: {_target_: reblock.methods.arterial.IdealChord}, engine: {_target_: reblock.methods.arterial.ExactEngine}}
```

**Only the two `access` entries get `max_anchors: 128`.** The cap changes which candidates exist, so it changes directness outcomes too, and every measurement is `objective=access`.

Update the comment block above the access entries: the "NOT `lazy`" note now reads that they use `ShortlistEngine` because CELF's submodularity guarantee fails for access-burden reduction.

- [ ] **Step 5: Verify every config instantiates**

```bash
pixi run pytest tests/methods -q
pixi run python -c "
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path
with initialize_config_dir(version_base=None, config_dir=str(Path('conf').resolve())):
    cfg = compose(config_name='compare_config')
for name, m in cfg.all_methods.items():
    if 'arterial' in name:
        obj = instantiate(m)
        print(f'  {name:<38} engine={type(obj.engine).__name__:<16} '
              f'realizer={type(obj.realizer).__name__:<16} max_anchors={obj.max_anchors}')
"
```
Expected: all six instantiate; only the two access entries show `max_anchors=128`.

- [ ] **Step 6: Commit**

```bash
git add conf tests
git commit -m "config: migrate the 9 arterial sites to injected engine/realizer/policy

Rewritten, not shimmed. mode/lam become a realizer, lazy/candidate_policy/rescore_every
become an engine. The aspirational entries carry no lam key at all, which is the point.

max_anchors: 128 goes on the ACCESS methods only -- the cap changes which candidates
exist, so it changes directness outcomes too, and every measurement behind it is
objective=access."
```

---

### Task 9: Roll access into the multiblock variants and regenerate

The only task that changes published output.

**Files:**
- Modify: `conf/example/depth.yaml`, `depth_density.yaml`, `density_compactness.yaml` (the `methods:` list)
- Modify: `examples/**` (regenerated artifacts)

**Interfaces:**
- Consumes: the migrated configs from Task 8.
- Produces: nothing later depends on this.

- [ ] **Step 1: Add the access method to the three variant lists**

In each of the three files, change:

```yaml
methods: [clearance_looped, euclidean_grid, resistance_lp, cycle_native]
```
to:
```yaml
methods: [clearance_looped, euclidean_grid, resistance_lp, cycle_native, greedy_arterial_access_displacement]
```

- [ ] **Step 2: Confirm the working tree is clean before regenerating**

```bash
git status --porcelain examples/
```
Expected: empty. `scripts/regenerate_examples.sh` refuses to run otherwise — it restores from git on interruption and would discard uncommitted work.

- [ ] **Step 3: Dry-run the regeneration to confirm the variant wiring**

```bash
bash scripts/regenerate_examples.sh --dry-run
```
Expected: prints one `gen_example` command per variant × city plus `method_comparison` and the screen bake-off.

- [ ] **Step 4: Regenerate, detached**

The agent harness SIGTERMs long foreground runs (confirmed 2026-08-11, `comm='claude'` named itself as the sender). Budget 1–2 hours for the added access method across 6 multiblock examples.

```bash
setsid nohup bash scripts/regenerate_examples.sh > /tmp/regen.log 2>&1 < /dev/null &
disown
```
Poll `tail -5 /tmp/regen.log`. Do not use a foreground waiter.

- [ ] **Step 5: Check what actually moved**

```bash
git status --short examples/ | head -40
```
Expected: the 6 multiblock example directories gain access-method artifacts. Directness and aspirational outputs should be **unchanged** — if `examples/method-comparison` shows changes to non-access methods, something in Tasks 1–7 altered behaviour and must be found before committing.

- [ ] **Step 6: Run the full check**

```bash
pixi run check
```
Expected: lint, typecheck and the full suite green.

- [ ] **Step 7: Commit**

```bash
git add -A conf/example examples
git commit -m "examples: run the access objective in the three multiblock variants

The blocker is gone: the access method now finishes at region scale (7.6x median from
the anchor cap on top of tier 2 making it finish at all -- roughly 330x on the original
11.6h-unfinished problem).

The multiblock variants previously ran no arterial method at all, so this adds one
method to three lists. Directness and aspirational outputs are unchanged; only the
access artifacts are new."
```

---

## Self-Review

**Spec coverage:** §1 architecture → Tasks 1, 3, 5, 6, 7. §2 identity → Tasks 4, 6, 7 (identity fields land with each strategy). §3 cap → Task 2; `k`/`threads` defaults → Task 7. §4 config + rollout → Tasks 8, 9. §5 testing → the guards are distributed into the task that creates what they guard, which is what makes each one fail before its implementation exists.

**Guards that must fail first** (the break-it proof): Task 2 Step 2 (two guards fail on current code), Task 4 Step 2 (`lam`-in-identity), Task 7 Step 5 (delete `committed_disp` and confirm the oracle fails).

**Deliberately not covered**, per the spec's Scope: `objective`/`cost` remain strings; stochastic restarts are not built; directness keeps `max_anchors: 0`.

**Three issues this review caught and fixed inline:**

1. Task 4 referenced a `deep_block` fixture that does not exist — `tests/methods/test_arterial.py` builds blocks inline and the only shared fixture is `real_block` in `tests/conftest.py`. Now uses the file's own `_two_arm_block(building_points, h=9, gap_x1=10)` helper, which supplies the building points that `cost="displacement"` needs.
2. Task 4 also proposed asserting that swapping the realizer changes the resulting WKT. On `_two_arm_block` the arms are 1-wide columns, so a boundary-graph path and a straight chord can legitimately coincide — the assertion would be fragile rather than defect-detecting. Replaced with the deterministic unit tests in Task 3 plus an integration check that each realizer is consulted.
3. Task 7 called an undefined `_greedy_shortlist` and would have deleted `shortlist_greedy.py` out from under four research harnesses that inject non-production selectors. Fixed by lifting the `CandidateSelector` seam into production `shortlist.py`, so one step loop serves both — production always passing `FirstOrder`, the harnesses passing their research arms. Without this the "delete the duplicate" instruction would have forced the duplicate straight back.

**Type consistency check:** `_greedy_arterials` and `_greedy_arterials_lazy` lose `mode`/`lam` and gain `realizer` in Task 4; Tasks 6 and 7 call them with exactly that signature. `_greedy_arterials_lazy` gains `policy_spec` in Task 5 and `LazyEngine.run` passes `policy_spec=self.policy`. `EngineIdentity` is introduced in Task 6 with two members and extended to three in Task 7 — the union must be updated in Task 7 Step 4, not left at two.
