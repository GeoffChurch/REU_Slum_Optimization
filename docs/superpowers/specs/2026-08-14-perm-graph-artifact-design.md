# Piece B: the permeability graph as an artifact type

**Date:** 2026-08-14
**Branch:** to be created off `main`
**Status:** design, approved section by section; not yet implemented
**Scope of this document:** piece B of the site redesign
(`specs/2026-08-13-site-redesign-design.md` §3 "The graph artifact type", §7 B). Python only —
no JS, no web toolchain. Pieces C–F each get their own spec.

## Why

The Permeability page has no figure. `docs/methodology/permeability.md` explains grounded egress
power `bᵀL⁻¹b`, the clearance-fraction footpath conductance and `max()` road upgrades in prose, and
ends with a placeholder comment where the picture should be. Every other methodology section carries
images; this one carries the project's most-explained and least-shown idea.

The site redesign's answer is not a web widget but a **Python artifact type**: one function derives
the graph's drawable form, `render.py` draws it to PNG. That ordering buys two things a
widget-first build would not:

* the Permeability page gets its figure **now**, with no JS and no bundle;
* piece C's `PermGraph` widget inherits a static fallback that already exists, so the page degrades
  to a picture with JS off, and the widget's output can be diffed against a Python twin rather than
  trusted.

**What ships:** four PNGs on one pinned block, a 2×2 figure block on the Permeability page, and a
derived structure that pieces C and F consume unchanged.

## §1 One structure, no matplotlib

**New module `src/reblock/perm_graph.py`.** Three homes it deliberately does not take:

* **not `derive_graph.py`** — that name is taken, and not by anything related: it is the
  content-addressed memoization primitive (`derive(fn, *inputs)`). A `derive_graph` function about
  drawing would be a trap for a reader.
* **not `render.py`** — piece C serialises this structure to JSON from `gen_site_pages.py`, and
  piece F adds a CI guard on the Pyodide import closure. Keeping the derive step free of matplotlib
  is what makes both work; inside `render.py` the JSON path drags matplotlib in behind it.
* **not `permeability.py`** — that module's docstring states the graph assembly and sparse solve are
  transcribed verbatim from the validated prototype, *do not re-derive*. A drawing concern does not
  live there. `perm_graph` imports from it.

`perm_graph` imports only numpy / geopandas / shapely / scipy — the same closure
`reblock.permeability` already has, so it stays inside piece F's declared Pyodide-available set.

### The dataclass

One frozen dataclass, all numpy arrays, no dicts and no string keys:

```python
@dataclass(frozen=True)
class GraphFigure:
    # nodes (parcel order, length n)
    cx: NDArray[np.float64]
    cy: NDArray[np.float64]
    potential: NDArray[np.float64]      # φ, the grounded egress solve
    ground_g: NDArray[np.float64]       # conductance to ground; 0 where not street-fronting
    # edges (Mesh order, length m; rows[k] < cols[k])
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    conductance: NDArray[np.float64]    # final g, after road upgrades
    footpath_g: NDArray[np.float64]     # the road-independent term
    upgraded: NDArray[np.bool_]         # conductance > footpath_g
    current: NDArray[np.float64]        # i = g(φ[rows] - φ[cols]), signed rows -> cols
    n: int
    p: float                            # the solver's own dissipated power
```

Two field choices worth stating:

* **`ground_g` per node, not a `ground` bool.** It is the quantity §5's energy identity needs, and it
  lets the ground halo be drawn weighted by the conductance actually present. The bool is recoverable
  as `ground_g > 0`, so nothing is lost.
* **`upgraded` is stored, not derived.** It equals `conductance > footpath_g`, which is exactly *the
  road raised this edge* — and needs no change to `edge_conductances`, which today returns only the
  final `g` with no covered-mask. Storing it means the mask is computed once, in Python; piece C's
  TypeScript consumes it rather than recomputing it, which is the whole point of one definition and
  two renderings.

  Note what `upgraded` claims and does not claim. A road-*covered* edge whose road term comes in
  below the footpath keeps the footpath under `max()` and reads here as not upgraded. That is the
  honest caption for the picture — the drawing shows which edges the road actually raised. Such an
  edge is possible in principle; the clamp it would once have tripped never fired in 19,023 mesh
  edges across 60 real blocks (`notes/2026-07-31-width-is-per-road.md`).

### The function

```python
def permeability_graph(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> GraphFigure
```

`adj` / `radii` are threaded exactly as `egress_power` and `permeability` already accept them, so a
region-scale caller does not rebuild `parcel_adjacency`.

`ground_g` is `params.g_street` where `mesh.ground`, else 0 — the same value `egress_power` folds
into the Laplacian diagonal. Current is `conductance * (potential[rows] - potential[cols])`.

### Where the solve comes from: one assembly, exposed

The figure needs the per-edge final conductances, and `egress_power` returns only `(P, v)`. Three ways
to get them, and the choice is forced by `permeability.py`'s own instruction that the graph assembly
and sparse solve are transcribed verbatim and must not be re-derived:

* **Re-assemble in `perm_graph`** — exactly the re-derivation that instruction forbids, and it would
  put a second Laplacian in the codebase to drift from the first. Rejected.
* **Call `egress_power`, then rebuild the mesh and conductances alongside it** — no duplicated solver,
  but two `footpath_mesh` builds and two `edge_conductances` passes per figure, the second purely to
  read back what the first already computed. Wasteful, and it invites the two to disagree.
* **Expose the assembly once, in `permeability.py`, and have both callers use it.** Taken.

So `permeability.py` gains one public function and `egress_power` becomes a thin wrapper over it:

```python
@dataclass(frozen=True)
class EgressSolution:
    p: float                            # b^T L^-1 b
    potential: NDArray[np.float64]      # v
    mesh: Mesh
    conductance: NDArray[np.float64]    # per-edge final g, after road upgrades

def solve_egress(block, roads, params, *, adj=None, radii=None) -> EgressSolution
```

`egress_power` keeps its signature and its callers, returning `(sol.p, sol.potential)`. There is
exactly **one** Laplacian assembly and one `spsolve` in the repo, and the figure reads the same
conductance array the metric was computed from rather than a second opinion about it. The ungrounded
and degenerate cases keep their present behaviour inside `solve_egress` (`p = inf`, zero potentials),
so `egress_power`'s contract is unchanged; `permeability_graph` is the one that refuses them.

`EgressSolution` and `GraphFigure` are not redundant: the first is the solver's output, carrying a
`Mesh`; the second is the *drawable* form — flat arrays, current, `upgraded`, `ground_g`, and nothing
that will not serialise to JSON in piece C.

**An ungrounded block raises `ValueError`.** `egress_power` returns `(inf, zeros)` when no parcel
fronts a street, because an ungrounded network has no well-defined dissipated power. A figure built
from those zeros would be a picture of no flow anywhere — silently wrong rather than absent, so the
failure is loud instead. This is a figure generator, not a batch metric; there is no aggregate for it
to keep marching through.

**No `n == 0` guard**, deliberately. `Block.__post_init__` already raises on empty parcels
(`contracts.py:56`), so that branch cannot be reached — it would be a silencer for a failure that
cannot happen, and a test for it would need an input the production code cannot construct.
`egress_power`'s own `n == 0` early return is dead for the same reason; it is pre-existing and out of
scope here, noted so it is not copied.

**Reusing the solve.** `permeability_graph` performs the same single solve `egress_power` does, so it
is not an added cost where a permeability render already happens — the caller can feed
`figure.potential` into the existing `perm` choropleth instead of calling `parcel_potentials`
separately. Piece B does not do that (see §6), but it is why the structure carries `potential` at
all rather than expecting a second call.

## §2 The renderer

**`render.py` gains `render_graph`, not a third `field=` value.** `_FIELD_CMAP` / `_FIELD_VMIN` are
keyed by a *choropleth colouring* fed to `parcels.plot(column="layer")`. A graph is a different
drawing — a `LineCollection` plus a node scatter — so a `"graph"` key in those dicts would be a value
that never reaches `parcels.plot`.

```python
def render_graph(
    figure: GraphFigure,
    block: Block,
    *,
    layer: Literal["conductance", "current"],
    vmax: float,
    width_norm: float,
    frame: BBox | None = None,
    roads: GeoDataFrame | None = None,
) -> Figure
```

`layer` selects what edge width encodes. Both quantities live on the same `GraphFigure`; the choice
is the caller's, resolved once where the figure set is defined (§3) rather than re-derived per draw.
`vmax` and `width_norm` are explicit for the same reason `render_before`/`render_after` take an
explicit `vmax`: a before/after pair must share its scales or the comparison is meaningless.

**What it draws, bottom to top:**

1. **Parcels as a pale wireframe** — outlines only, no fill. Filling by φ would state the same
   quantity twice in two shapes and drown the graph. This is the composition decision: the graph
   figure is *not* the perm choropleth with dots on top; the parcels recede so the graph is the
   subject.
2. **Roads**, when given, as the width-buffered union in `_ROAD_COLOR` at low alpha — so corridor and
   upgraded edges read as the same fact.
3. **Boundary and streets** in `_BOUNDARY_COLOR`, via a helper extracted from `_draw_heatmap` and
   shared with it. A reader has to see where ground *is*, and the MultiPolygon-skip rationale
   (`render.py:150`) is not being duplicated to get it.
4. **Edges** as one matplotlib `LineCollection` — the choice that keeps a 60k-edge region cheap;
   `geopandas`-per-row plotting is what would not scale. Grey base collection, then `upgraded` edges
   over the top in `_ROAD_COLOR`. Width is the chosen quantity divided by `width_norm` and clipped,
   scaled between a visible-hairline minimum and a fixed maximum, so one trunk edge cannot flatten
   the rest of the mesh into invisibility.
5. **Ground halos** — a ring under each node with `ground_g > 0`.
6. **Nodes** as geographic-radius disks (the existing `_point_disks` treatment, so this does not
   collapse at region scale later), filled by `potential` on `_PERM_CMAP` with `vmin=0` and the
   caller's `vmax`. Same colormap and same scale as the existing perm renders, so colour means the
   same thing on both images a reader sees on that page.

Palette constants, `frame_bbox` and `save_render` are reused as-is. No new colours are introduced;
piece C emits these same constants into the bundle, so a second palette here would surface as the
most visible possible drift.

## §3 The figure set and its generator

Four images, one block, both layers, before and after.

**The block is `ZAF.9.3.1_1_40972`** — the block `conf/example/method_comparison.yaml` pins. Small
enough that individual edges read, deep enough that φ has range, and already the block every method
page's before/after uses, so a reader recognises it.

|  | no roads | with roads |
|---|---|---|
| **width ∝ conductance** | the clearance-fraction mesh: packed fabric as faint threads, gaps thick; ground haloed | the same mesh with the road's upgraded edges in blue |
| **width ∝ current** | drainage in the status quo — flow crowding toward the few street-fronting parcels | current concentrated into the new corridor |

Filenames: `graph_conductance_before.png`, `graph_current_before.png`,
`graph_conductance_after.png`, `graph_current_after.png`.

**The 'after' road set is `clearance` at its Lens-B prefix** — the matched-permeability standard from
`conf/permeability.yaml`, via `prefix_to_permeability`. That is the same road set the site already
publishes for clearance on this block, so the figure agrees with the images beside it instead of
being a bespoke one-off.

**Both scales are shared across all four images.** Both figures are derived first; `vmax` is the
pooled maximum potential and `width_norm` the pooled 99th percentile of the relevant quantity
(computed per layer, shared across before/after within that layer). This is the discipline
`compare_budgets` already applies to `vmax` and `frame`; without it before and after are not
comparable and the pair teaches nothing.

### Its own entry point

`scripts/gen_perm_graph.py` → `examples/perm-graph/`, added to `scripts/regenerate_examples.sh`
beside `gen_screen_bakeoff` (also not a `gen_example` variant).

**Not folded into `examples/method-comparison/`**, for one reason: iterating on a figure's *design*
must not require re-running a ten-method comparison. This generator loads the one pinned block
through the same config and derivation path the example uses, and takes clearance's roads from the
content-addressed derivation cache — seconds, not the example's full run.

It writes four PNGs (~4 MB, block scale, committed like every other example artifact), a README
carrying the provenance (block id, method, prefix rule, regeneration command), and a
`perm_graph.json` (see §4).

`examples/perm-graph/` is **not** added to `examples/README.md`'s flagship table. That table lists
walkthroughs that reproduce a result from the CLI; this is a figure set for one site page. It gets a
one-line pointer from that file's prose so it is discoverable, and its own README for provenance.

## §4 Page wiring, with the numbers in the generator

`gen_perm_graph.py` writes `examples/perm-graph/perm_graph.json`: block id, method, `P*`,
permeability before and after, road length, parcel count, edge count.

`scripts/gen_site_pages.py` gains:

* a `PERMGRAPH = ROOT / "examples" / "perm-graph"` path constant beside `MC` / `MB` / `BAKEOFF`;
* a `PERMGRAPHFIGS` marker and its producer, which `_copy_asset`s the four images into
  `docs/assets/perm-graph/` and builds the 2×2 with `_figure()`, captions reading their numbers from
  `perm_graph.json`.

`docs/_partials/permeability.md`'s placeholder comment (the one `_render_partial` deliberately does
not strip, since only the *leading* comment goes) is replaced by `<!-- PERMGRAPHFIGS -->`.

Two rules this satisfies without extra machinery:

* **No number is typed into the prose.** The captions carry the permeability values and road length
  from the artifact, so every number shown graphically is also present as text — and the drift class
  piece A's truth pass closed stays closed.
* **The marker is guarded both ways already.** `tests/test_gen_site_pages.py:81,89` fail on a marker
  with no producer and on a producer used by no partial, so a half-wired figure block cannot ship.

## §5 Testing: a physics identity, not a golden image

The derived structure is checkable against the solver it came from, which is what makes "derive once,
render twice" safe when piece C adds a second renderer.

**Energy identity.** With `b` all-ones, `v = L⁻¹b`, so `vᵀLv = vᵀb = Σφ = P`. Expanded over the
drawn quantities:

    Σ_edges conductance·(φ[rows] − φ[cols])²  +  Σ_nodes ground_g·φ²  ==  p  ==  Σφ

Exact up to solver residual; assert at relative 1e-9.

**Per-node Kirchhoff.** `(Lv)ᵢ = bᵢ = 1` for every node, which in drawn quantities is: signed
incident current (`+current` scattered onto `rows`, `−current` onto `cols`) plus `ground_g·φ` equals
exactly 1. This is the assertion that the drawn currents are a valid flow with one unit injected per
parcel — and it catches indexing errors the aggregate identity can absorb.

**Fault injection is part of the task, not a nicety.** Each of these must be shown to fail before the
tests count as guards — three tests in this repo's recent history passed while guarding nothing:

| break | which assertion catches it |
|---|---|
| flip the current sign (`φ[cols] − φ[rows]`) | Kirchhoff: every node reads −1 |
| use `footpath_g` instead of `conductance` for current | both, wherever a road upgraded an edge |
| drop the `ground_g` terms | energy identity; Kirchhoff at grounded nodes |
| transpose `rows`/`cols` | Kirchhoff |

**Also tested:** `upgraded` is all-false for `roads=None` and has at least one true edge for a road
set known to cover the mesh; an ungrounded block raises `ValueError`. The renderer gets a
smoke test in `tests/test_render.py`'s existing style — assertions on the figure's artists, not on
pixels.

**Not a guard: comparing `figure.p` to `egress_power`.** After §1's refactor they read the same field
of the same `EgressSolution`, so that assertion cannot fail and would be exactly the kind of test that
defends a branch rather than the behaviour. What guards the refactor is the existing permeability
suite passing **bit-identically** — `solve_egress` is a pure extraction, so any moved digit is a bug,
not a re-baseline. `tests/test_permeability_width.py`'s pinned one-lane equality is the sharpest of
these.

`src/reblock/perm_graph.py` is under `src`, so `mypy --strict` covers it with no config change.

## §6 What B is not

* **No `compare_budgets` change.** The per-method render grid stays exactly as it is. Adding graph
  renders next to all 93 committed perm renders was costed at ~186 PNGs and +250–320 MB committed,
  against an `examples/` tree already at 395 MB, and rejected in favour of this focused set.
* **No GIF mode.** `animate.py` renders depth only and stays that way.
* **No widget, no JSON, no `.d.ts`, no `web/`.** Those are piece C, which consumes `GraphFigure`
  unchanged.
* **`parcel_potentials` is left alone.** It performs its own full solve and `permeability_graph`
  subsumes it, so whenever `compare_budgets` does grow graph renders that is a duplicate solve to
  remove. It has one caller and is not a legacy path; no action now, recorded so the consolidation is
  not rediscovered.

## Files touched

| file | change |
|---|---|
| `src/reblock/permeability.py` | extract `solve_egress` + `EgressSolution`; `egress_power` becomes its wrapper |
| `src/reblock/perm_graph.py` | **new** — `GraphFigure`, `permeability_graph` |
| `src/reblock/render.py` | **new** `render_graph`; boundary/streets helper extracted from `_draw_heatmap` |
| `scripts/gen_perm_graph.py` | **new** — the four-image generator + `perm_graph.json` |
| `scripts/regenerate_examples.sh` | one entry beside `gen_screen_bakeoff` |
| `scripts/gen_site_pages.py` | `PERMGRAPH` constant, `PERMGRAPHFIGS` marker + producer |
| `docs/_partials/permeability.md` | placeholder comment → `<!-- PERMGRAPHFIGS -->` |
| `examples/README.md` | one-line pointer to `perm-graph/` in prose (not the flagship table) |
| `examples/perm-graph/` | **new** — four PNGs, `perm_graph.json`, README |
| `tests/test_perm_graph.py` | **new** — §5's identities and fault injections |
| `tests/test_render.py` | `render_graph` smoke test |

## Open items

* **Region-scale legibility is designed for but unproven.** The renderer uses one `LineCollection`
  and geographic node radii specifically so a 11k-parcel region works, and the arithmetic is
  favourable (~45 px of node spacing at 300 dpi / 16 in). No region render ships in B, so this is a
  claim piece C or D tests, not one B establishes.
* **Whether the conductance 'after' image earns its slot on the page.** It differs from the 'before'
  only by the blue upgrade overlay, since the mesh is road-independent. It is cheap to emit either
  way; if the 2×2 reads as three images and a near-duplicate, the page shows three and the fourth
  stays an artifact.
