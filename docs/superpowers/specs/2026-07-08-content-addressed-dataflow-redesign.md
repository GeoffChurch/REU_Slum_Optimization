# reblock — Content-addressed dataflow redesign (north star)

**Status:** draft for review · **Date:** 2026-07-08 · **Branch:** `dataflow-redesign` (to be cut)

This supersedes the *later* slices of the flow-refactor: F1 (pure `run()`) and F2 (content-addressed
cache) are the seeds of this design and stay on `main`; the **F2 per-wrapper cache is replaced** by the
uniform `derive` primitive here, the **F3 plan is dropped** (its screen-stage falls out of the pipeline
for free), and F4's sweep lands on the clean core. The domain algorithms are untouched — this is an
architecture change (data types, caching, pipeline, config), not a geometry rewrite.

## Why — the four conflations

Every wart hit during the flow-refactor is a symptom of one of these:

1. **The Source ingests raw data *and* builds parcels.** Voronoi tessellation is a derivation, but it
   lives in `KblockSource.region()` — forcing a bolted-on build cache, `source_content_hash` threaded
   onto `Block`, and the Screen re-instantiating a whole `KblockSource` for its fine pass.
2. **Derivations are free functions but caching is bolted on per-function.** Four `cached_*` wrappers
   each hand-roll the same key tuple — that duplication *is* the `roads_key` wart, and it's why F2's
   review caught an incomplete `code_version` (key logic in four places → a callee missed).
3. **`run()` conflates selection, sampling, and iteration.** `max_blocks` means both "how many to
   reblock" and doubles as a sampler; the screen's *full* selection is lost the moment `max_blocks`
   truncates results — the whole reason F3 got stuck on "where does the flagged-map get the selection."
4. **Config is load-bearing dataflow.** `${block_ids}` interpolation + *mutating* `cfg.block_ids` in
   `main` uses the config object as a message bus. Data should flow as values.

## Target architecture — six layers

**1 — Immutable data with identity.** Every cacheable datum exposes a hashable `identity` (a stable
content-address). A `RawBlock = (block_id, boundary, building_points, streets, source_id)` where
`source_id` is a content hash of the source file(s). `RawBlock.identity = (source_id, block_id)`.
Derived data composes: `Proposal.identity = (block.identity, method.identity)`. The **Source does pure
ingest** — it yields `RawBlock`s; it does **not** tessellate.

**2 — Pure derivations, one graph.** Free, pure, deterministic functions, each `(inputs) -> output`:
`parcels(raw)`, `adjacency(parcels)`, `access_depth(parcels, streets, roads)`,
`geometric_access(parcels, streets, roads)`, `propose(method, parcels, streets)`,
`metrics(eval, derivations, proposal)`. Nothing lives "in" a Source/Block/Method object beyond being a
function. The existing algorithm bodies (Voronoi, BFS peel, geometric Dijkstra, topology builder, the
k-metrics) are **reused verbatim** inside these.

**3 — One memoization primitive.**
```python
def derive(fn: Derivation, *inputs: Identified) -> Any: ...
```
Computes `fn(*inputs)` with **L1 (in-process) + L2 (joblib disk)**, keyed on
`(fn.identity, tuple(i.identity for i in inputs))`. Keys are formed in **exactly one place**. What this
buys, for free, versus F2's four wrappers:
- **before/after split is automatic** — `roads=None` and a proposal are different input identities, so
  no manual `roads_key`.
- **`code_version` is `fn.identity`** — computed once, completely; no hand-maintained module list.
- **GEOS/PROJ** fold into `fn.identity` (one place).
- Unknown/empty identity → **bypass** (synthetic/test data never caches).

**4 — The pipeline as explicit dataflow** (values, not config):
```python
region    = source.raw_blocks()              # stream/list of RawBlock (pure ingest)
selection = screen.select(region)            # Selection: block_ids | ALL — a RETAINED value
picked    = sample(selection, region, n)     # sampling is its own stage, distinct from selection
results   = [reblock(block, method, evals) for block in picked]   # each step is a derive() call
```
`reblock` composes derivations (`parcels → propose → metrics`) through `derive`. There is **no
`max_blocks` overload** (selection ≠ sampling) and **no `cfg` mutation** (selection flows as a value).

**5 — Sweep + emit are outer combinators.** A `Run` produces a typed `RunOutput(selection, results)`. A
`Sweep` runs many `Run`s (varying method/params), sharing L1 in-process. Emitters consume the typed
output: `render(results)`, `scorecard(results)`, `flagged_map(selection, region)`. The flagged-map gets
the **full** selection because it is a value in the output, not a casualty of sampling.

**6 — Config only at the edge.** Hydra (or any CLI) parses arguments into **typed stage constructors**
(`Source`, `Screen`, `Method`, `Eval`, sampler, emitters) composed into a `PipelineSpec`. The core
pipeline is typed Python composition and never sees a `DictConfig`. Config-group ergonomics
(`data=`/`method=`/`screen=`) stay via typed factories; nothing mutates config.

## Key decisions (my calls + rationale — flag any in review)

- **D1 — Content-address on composed *identities*, not geometry WKB.** `RawBlock.identity =
  (source_id, block_id)` fully determines its content (sources are immutable content-hashed files;
  block_id is stable). Composing identities is O(1) per lookup; hashing block/road geometry per lookup
  is slow and GEOS-fragile. Cost: GEOS/PROJ must be folded into `fn.identity` deliberately (they are).
  *Rejected:* geometry-WKB hashing (fully automatic re: PROJ/partial edits, but per-lookup-expensive).
- **D2 — `fn.identity` = one centralized, complete version tag**, not a per-function dependency
  registry. It is `hash(source of all derivation modules) + geos_version + proj_version`, computed once
  at import. Coarse (any derivation edit invalidates everything) but **complete and safe** — the exact
  failure F2's review found. *Rejected:* a registry where each fn declares its callees (finer, but real
  machinery — an over-abstraction to resist).
- **D3 — Full typed pipeline; Hydra at the edge only.** The core is typed composition; no `cfg`
  threading/mutation. Bigger change than "keep config groups", but it's what removes conflation #4 at
  the root. *(Most worth your confirmation — it's the largest departure from today.)*
- **D4 — Preserve every domain algorithm.** Voronoi, BFS peel (`parcel_access_layers`), geometric
  Dijkstra, the topology road-builder, the k-metrics — reused verbatim inside the new pure derivations.
  Zero behavior change to the geometry; the pinned-value tests must still pass (re-expressed against the
  new types).
- **D5 — `RunOutput(selection, results)`** replaces the bare `list[Result]` return, so the selection is
  retained. `Result` keeps `(block, proposal, metrics-tuple)`.

## New contracts (concrete)

```python
class Identified(Protocol):
    @property
    def identity(self) -> Hashable: ...

@dataclass(frozen=True)
class RawBlock:            # pure ingest unit; NO parcels (that's a derivation)
    block_id: str
    boundary: Polygon
    building_points: tuple[Point, ...]   # or a small GeoSeries handle
    streets: GeoDataFrame
    source_id: str
    @property
    def identity(self): return (self.source_id, self.block_id)

class Source(Protocol):
    def raw_blocks(self) -> Iterable[RawBlock]: ...   # ingest only

class Screen(Protocol):
    def select(self, region: Iterable[RawBlock]) -> Selection: ...   # cheap features only

class Method(Protocol):
    identity: Hashable
    def propose(self, parcels, streets) -> Proposal: ...

class Eval(Protocol):
    identity: Hashable
    def score(self, block_derivations, proposal) -> Metrics: ...
```
`Selection` is `frozenset[str] | ALL` (a sentinel). `Proposal`/`Metrics`/`Result` keep today's fields
(`Result.metrics` stays a tuple). Parcels/adjacency/access become derivation *outputs*, not `Block`
fields.

## Migration strategy (detail → the plan)

A dedicated `dataflow-redesign` branch, built **bottom-up** so each step is independently testable, with
the old system removed in a final consolidation so the **merged result carries no dual system** (the
"no dual path" directive binds the shipped state, not transient branch state):

1. **Identity + `derive` primitive** (new `reblock.derive_graph`) with its own tests — the L1/L2 engine.
2. **Derivations** re-expressed as free pure functions over the new types (reusing algorithm bodies).
3. **`RawBlock` + Source ingest split** (`KblockSource.raw_blocks`, `ShapefileSource.raw_blocks`);
   `parcels` becomes a derivation.
4. **Pipeline** (`reblock`, `sample`, `RunOutput`) + Screen/Method/Eval on the new types.
5. **Entrypoint + emitters** (typed `PipelineSpec`; Hydra at the edge; `render` + `flagged_map` +
   `flagged_blocks.txt`; the one-command end-to-end).
6. **Delete the old** contracts/cache/run/screen-app and **migrate all remaining tests**; green at merge.

Each layer is a subagent-driven task ending green (new code + new tests); the old code is deleted in the
final layer, so no long-lived red branch and no shipped duality.

## Preserves / supersedes

- **Preserves:** all domain algorithms (D4); Hydra as the CLI edge; the F1 purity direction; the L1+L2
  two-tier caching *idea* (now unified in `derive`).
- **Supersedes / deletes:** F2's four `cached_*` wrappers + `Block.source_content_hash` (→ `derive` +
  identities); the F3 plan (screen-stage/flagged-map fall out of the pipeline); `run()`'s
  Source-instantiate-and-loop + `max_blocks` overload + `cfg.block_ids` mutation; the Source's build
  responsibility.

## Decisions to confirm in review

1. **D3** — full typed pipeline with Hydra only at the edge (vs keeping config groups as construction
   but stopping `cfg` mutation). Biggest departure; want your explicit call.
2. **D1** — identity-composition vs geometry-WKB content-addressing.
3. **Migration** — bottom-up-then-delete on one branch (chosen) vs a longer strangler with both systems
   parallel for several merges.
