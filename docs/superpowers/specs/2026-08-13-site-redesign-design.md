# Site redesign: a methodology spine, an interactive explorer, and a truth pass

**Date:** 2026-08-13
**Branch:** `main`
**Status:** design, approved section by section; not yet implemented
**Scope of this document:** the umbrella architecture binding six shippable pieces (A–F below).
Piece A goes straight to an implementation plan from §7 here. Pieces B–F each get their own spec.

## Why

The published site is stale in *claim*, not merely in layout, and the repository holds strong
evidence that the site does not show.

**Wrong on the page today:**

| # | where | defect |
|---|---|---|
| 1 | `docs/methodology.md:30` | `density × compactness` (`n/P²`) is labelled **"the screening heuristic"**. Retired 2026-08-08. The shipped default is `depth_density_proxy` (`√(nA)/P · n/A`), which beats it on precision *and* recall at equal pool size — 27.5%/66.7% against 24.5%/58.9% (`examples/screen-bakeoff/README.md`). The site advertises a dominated screen as the method. |
| 2 | `docs/background.md:37` | The same wrong attribution, load-bearing in the prior-work narrative: the contrast drawn against Soman et al. (2020) credits a screen the project no longer uses. |
| 3 | `docs/methodology.md:31` | Displacement is defined as *"the expected number of **parcels** a road set displaces."* It is **buildings**. `Block.parcels` and `Block.building_points` are distinct fields (`src/reblock/contracts.py:46,50`); `budget.displacement` reads "expected homes displaced" (`src/reblock/budget.py:70`); `displacement_curve` spells it out — *"n_buildings = len(block.building_points) (buildings, not parcels)"*. The cost axis is documented against the wrong denominator. This is an error, not staleness. |
| 4 | `docs/_intro.md:64` | *"Seven road-generation methods."* The `METHODS` registry in `scripts/gen_site_pages.py` holds 11 entries; ten are published. |

**Missing from the page today:**

* The **screen bake-off** — validated against the City of Cape Town's own informal-structure
  survey (117,336 dwelling polygons, 189 clustered settlements, 682/16,451 blocks informal), with
  an AUC table and city-scale disagreement maps. It appears nowhere on the site.
* A **second city.** `examples/nairobi/` holds three full variants. Invisible.
* **Permeability's model.** One line on the site (`1 − dissipation fraction`) for a metric whose
  actual content is grounded egress power `bᵀL⁻¹b` over a footpath mesh with per-pair clearance
  conductance and `max()` road upgrades.
* **Displacement's model.** One (wrong) line, for a per-building disk model
  `cᵢ = max(0, 1 − dᵢ/rᵢ)` over a width-buffered unioned corridor.

Of `examples/`' seven flagships, the site surfaces two.

**Defect 4 is the instructive one.** `docs/_intro.md`'s own header already forbids it — *"Prose
lives here; every NUMBER lives in the generator. Never type a metric into this file."* It drifted
anyway, because "seven" does not look like a metric. The fix is structural, not editorial: that
count becomes a generated substitution alongside the existing `HEROLOGO` / `KEYRESULT` / `HERO` /
`KEYFIGURES` markers.

## Audience, and what it settles

**Primary reader: the academic evaluator.** The spine is therefore *claim → evidence → reproduce*.

The explorer is **a pedagogical aid with excellent visuals**, not a data-poking dashboard for
skeptics. It may carry dynamically-derived numbers, but never at the expense of intuitively
visualising each stage. Where the two compete, the visual wins.

## §1 Information architecture

```
Home
Background
Explore                    ← the chained walkthrough
Methodology
  ├ (index)  the pipeline end to end + glossary
  ├ Screening              ← widget: city choropleth + gate; ends with region growth
  ├ Permeability           ← widget: the graph            [NEW artifact type]
  ├ Displacement           ← widget: disks + corridor
  └ Methods
      ├ (index)            ← widget: frontier + targets
      └ 10 method pages
Results
  ├ Frontier benchmark     (was benchmark.md)
  ├ Screen bake-off        NEW
  └ Second city: Nairobi   NEW
Reproduce                  NEW
Team & References
```

**The organising invariant: the nav is the pipeline is the explorer.** The four methodology
sections are exactly the explorer's stages. Each section embeds its own stage's widget; the Explore
page chains the same four. A deep link (`explore/?block=…&stage=permeability`) and the Permeability
section's inline widget are *the same component with different props*. One spine, three renderings.

**Region growth folds into the end of Screening.** `conf/region_builder/` (`dense_cluster`,
`convex_hull`, `shape_standardizing`, `identity`) is a real pluggable stage and is the hinge in the
walkthrough — *pick the block you want* → grow it into a region. It is also where the pipeline flips
from whole-city-cheap to per-block-expensive. Folding it into Screening lets that section answer one
whole question — *where do we work?* — and narrates the hinge instead of skipping it. It does not
get its own page.

**Deliberately out:** a "what we tried that didn't work" page drawing on `docs/superpowers/notes/`.
Considered and declined; the notes stay internal.

## §2 The compute model

**Everything except one feature is a lookup.** Swapping screens, moving the gate, and setting a
displacement or permeability target require no browser computation:

* `street_first_ordered` gives every method's roads **one canonical drainage order**, and everything
  downstream is a prefix of it.
* `prefix_to_displacement` and `prefix_to_permeability` (`src/reblock/budget.py:796,833`) are both
  binary searches over prefix length, exploiting monotonicity.
* So a target slider in the browser is not approximating the Python. It is the **identical binary
  search over the identical monotone sequence**, and both the road geometry and the metrics fall out
  of one index.

**The one exception — draw-your-own-road — runs the real Python via Pyodide.** Decided in favour of
a hand-written JS mirror precisely to avoid a second implementation of permeability.

### Pyodide: verified feasible, with pins

`reblock.permeability`'s runtime import closure is **numpy, scipy, pandas, geopandas, pyproj,
shapely, networkx** — via `contracts`, `derive.access`, `derive.adjacency`, `mesh` (`budget` is
`TYPE_CHECKING` only). No `hydra`, no `matplotlib`, no `pyarrow`, no `joblib`. All seven are in
Pyodide 0.29.2 (Jan 2026); `networkx` is pure Python and installable via `micropip` regardless. The
module imports with **no surgery**.

`[project] dependencies = []` (everything heavy sits under `[tool.pixi.dependencies]`) means the
hatchling wheel installs through `micropip` without triggering dependency resolution. Keep it that
way.

**Pin the version in the CDN index URL:**

```js
loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/" })
```

jsDelivr serves immutable versioned paths, so a future Pyodide release cannot reach a pinned URL.
This matters: `geopandas` was *removed* in Pyodide 0.27, and `geopandas`/`pyproj`/`shapely` were all
*disabled* in 0.28 over build failures, returning only in 0.29.2 — two disappearances in two years,
of exactly the packages needed here.

**Self-hosting the wheels is not required** and is not proposed. Pinning solves the stability
problem on its own; self-hosting buys only the removal of a third-party runtime dependency and of
third-party requests from readers' browsers. Revisit only if SBU policy requires the latter.

**The real long-term risk is reblock's own import chain.** If someone later adds an import to it
(`pyarrow`, a numba kernel), the explorer dies silently while every test still passes — the quiet
failure mode the static-checkability rule targets. **Guard it:** a CI test asserting the
`reblock.permeability` import closure stays inside a declared allowlist of Pyodide-available
packages.

**Boot is lazy.** Pages load instantly from baked lookups; Pyodide boots only when a reader clicks
into authoring. The ~25–35 MB is paid by the reader who opted in, never by a visitor reading prose.

## §3 What gets baked

Three tiers, mirroring the pipeline's own cost structure — and the explorer teaches that structure
by having it:

| tier | scope | payload | size | loaded |
|---|---|---|---|---|
| **City** | 16,451 CT blocks | quantized polygons, 4 score columns, ground-truth informal flag, each metric's shipped floor | ~3 MB | on Screening / Explore ① |
| **Shortlist** | ~1,655 survivors | peeled `depth_density` score → the rerank | ~30 KB | with tier 1 |
| **Block** | 7 flagships | parcel polygons, graph (nodes, edges, conductances, ground), building points + radii, per-method ordered roads + full prefix table | ~0.5 MB each | per block, on demand |

**Recolouring never touches geometry.** Swapping screens is a re-fill pass over a buffer already in
memory, so the city recolour is instant. This implies **Canvas, not SVG**, at 16k polygons — decided
now, not discovered later.

### Two things must be newly emitted

* **The per-block city score table does not exist.** `examples/screen-bakeoff/screen_comparison.csv`
  is 5 rows — the summary. `scripts/gen_screen_bakeoff.py` necessarily computes per-block scores to
  produce its AUCs but writes only the aggregate. Emitting the per-block table is a small change to
  a script that already does the work.
* **The committed curves are too coarse.** `displacement_curve(..., n_points=20)` sweeps 20 samples;
  `examples/method-comparison/frontier_permeability.csv` is 143 rows across 9 method-block pairs.
  Bake **every prefix**, m = 0…R. With R ≲ 60 that is ≤60 rows per (block, method) — *smaller* than
  what is committed today, and exact rather than sampled. Costs R solves per method-block at bake
  time instead of 20.

**This is the long pole.** ~7 methods × 7 flagships of R solves each is wall-clock, not effort.
Start it during piece A so it is on disk when piece C needs it.

### Stage ② has a better story than "watch it improve"

The bake-off measured the expensive peeled stage as buying **2.4 points** of top-1% precision over
the free proxy (84.1% vs 81.7%). The honest beat is therefore: *watch the fine pass rerank the
shortlist, and notice it barely moves.* That is a real finding about where cost lives, and more
memorable than a stage that merely looks better.

### The graph artifact type

Built as a **Python renderer first**, not a web-only widget. One function derives the graph's
drawable form; `render.py` draws it to PNG (joining `render_before`/`render_after`);
`gen_site_pages.py` serialises the same structure to JSON for the widget. One definition of what the
graph *is*, two renderings.

What it draws, in rising order of value:

1. nodes at parcel centroids, filled by potential φ on the existing `YlOrRd` scale — dark = harder
   escape;
2. edges with width ∝ conductance, making the mesh's texture visible — packed fabric as thin faint
   threads, open gaps as thick ones;
3. road-upgraded edges in the road blue; ground-connected parcels haloed;
4. **edge current, `i = g(φᵢ − φⱼ)`** — the actual physics, one subtraction from `parcel_potentials`.

(4) is the prize. Draw current and the drainage tree *appears*; add a road and current visibly
concentrates into the new corridor. It is the best teaching image available in this project.

`DisplacementField` gets the equation-literal treatment: each building a disk of radius `rᵢ` (half
its nearest-neighbour distance), the corridor the width-buffered union, each building shaded by
`cᵢ = max(0, 1 − dᵢ/rᵢ)`, with `Σcᵢ` accumulating as roads are added. Since Pyodide is present
anyway, dragging two roads together to watch the union merge — and the cost *drop*, because overlap
is free by construction — is nearly free and makes a subtle modelling choice obvious.

### The bundle contract

`gen_site_pages.py` emits the JSON **and a generated `.d.ts`** describing it, so a renamed Python
field is a TypeScript error rather than a blank panel. The closed-set rule applied at the one
boundary where it would otherwise fail quietly.

## §4 Substrate: MkDocs stays

**Keep MkDocs Material for everything. The split-substrate option is rejected outright** — widgets
must embed in prose pages regardless, so a separate explorer app means two toolchains for one widget
library, with neither's benefit.

Reasoning against a full replacement (Observable Framework being the closest fit):

* the site is ~85% prose and generated evidence, ~15% interactive; choosing the substrate for the
  15% is backwards;
* a JS bundle step is needed either way — under MkDocs that is *one added step*, under Framework it
  is built in but everything else is rewritten;
* Framework's headline feature is build-time **data loaders**: Python computes, artifacts land on
  disk, the site reads them. `scripts/gen_site_pages.py` is already exactly that, working, in 864
  commented lines;
* the audience is academic evaluators. Reliability beats novelty.

"Keep MkDocs" has a clean version and a rotting version, and the difference matters more than the
choice:

| | rots | clean |
|---|---|---|
| widget code | inline `<script>` in markdown | `web/` TypeScript tree, one module per widget, esbuild → `docs/js/widgets.js` |
| pages | carry logic | carry a mount point only |
| data bundle | hand-kept in sync | generated JSON **+ generated `.d.ts`** |

**Do not enable Material's `navigation.instant`.** It swaps pages without reload and breaks naive
widget initialisation unless Material's `document$` observable is hooked. It is not enabled today;
leave it off.

Build additions: `web/` TypeScript → esbuild → `docs/js/widgets.js`, added to
`.github/workflows/deploy-site.yml` pinned exactly as `mkdocs-material==9.7.7` already is, plus a
`pixi run web` task for local iteration.

## §5 The widget library and the mount contract

| widget | teaches | embedded in | Explore stage |
|---|---|---|---|
| `ScreenMap` | metric choice, the gate, precision/recall against ground truth | Screening | ① ② |
| `RegionGrow` | block → region; the cheap/expensive hinge | Screening (end) | ③ |
| `PermGraph` | the egress graph, conductance, current | Permeability | ④ |
| `DisplacementField` | disks, corridor, `Σcᵢ` | Displacement | ④ |
| `Frontier` | methods, targets, the tradeoff | Methods | ⑤ |

Substrate: a projected canvas renderer with pan/zoom, a tiered data loader, and a state source.
Pyodide sits outside all of it, touched by one widget.

**The mount contract.** A page carries a placeholder and nothing else:

```html
<div data-widget="perm-graph"
     data-layers="current,ground"></div>
```

**Amended after piece D2 (2026-08-19):** the example above carried a `data-block="…"` attribute,
which shipped in C and was deleted end to end in D2 — every bundle already carries `block_id`, no
widget ever read the attribute, and a second source for one fact is drift waiting to happen. The
markup is corrected here rather than left as a record, because a spec's example markup is read as
the contract and `grep data-block` was returning it as if it were current.

`attr_list` and `md_in_html` are already enabled in `mkdocs.yml`; this needs no plugin.

**Embedded and chained are one component.** A widget never asks whether it is inline or in the
explorer. It reads a `StateSource` injected at mount: in prose, a frozen one built from the `data-*`
attributes; on Explore, a shared store synced to the URL. Resolve-upstream applied where it pays —
otherwise every widget grows an `if (embedded)` branch and the branch count climbs with the widget
count.

**URL-as-state.** The URL is the explorer's entire state:

```
explore/?city=capetown&metric=depth_density_proxy&gate=0.0128
        &block=ZAF.9.3.1_1_44882&method=clearance&target=disp:0.042
```

A reviewer can cite a *specific view*, and prose pages can link into one.

**Every widget degrades to a picture, for free.** Because the graph is a Python artifact type first,
`render.py` already produces the PNG. It sits inside the mount point as the fallback and the widget
replaces it on boot. No-JS, print, and a failed fetch all land on exactly today's site rather than a
blank box.

**One place *not* to inject a Protocol.** A common `Scorer` interface across `Frontier` and the
authoring widget is tempting and wrong — they do not answer the same question:

* `PrefixTable` — *"what does prefix m of method X score?"* — lookup, instant, used by `Frontier`;
* `Scorer` — *"what does this arbitrary road set score?"* — Pyodide, seconds, used by nothing else.

Forcing one interface produces a `TableScorer` that throws on inputs its type claims to accept. Two
names, no polymorphism, and **exactly one widget ever boots Pyodide**.

**The palette comes from Python.** `render.py` fixes `_CMAP`/`_PERM_CMAP` (both `YlOrRd`,
deliberately, so depth and potential read alike), `_ROAD_COLOR = #1E90FF`, `_BOUNDARY_COLOR`, and
the context greys. Those constants are emitted into the bundle rather than retyped in CSS —
otherwise the same block renders in two palettes on one page, the most visible possible drift.

**Accessibility** follows the choice the site already made by self-hosting Atkinson Hyperlegible:
keyboard-reachable controls, `prefers-reduced-motion` honoured, and every number shown graphically
also present as text.

## §6 Page dispositions

| page | disposition |
|---|---|
| `_intro.md` → `index.md` | revise; five cards become the new IA; the method count generated |
| `background.md` | keep; fix the attribution (defect 2); re-check the "peaks in Khayelitsha" claim, which was measured under the retired metric |
| `methodology.md` | **dissolves** into `methodology/index.md` + Screening + Permeability + Displacement + Methods |
| `methods/*.md` | 10 generated pages, content unchanged, renested under `methodology/` |
| `benchmark.md` | → `results/frontier.md`, joined by `results/bakeoff.md` and `results/nairobi.md` |
| `team.md` | unchanged |
| `metrics-north-star.md` | stays excluded / internal |

The Definitions table dissolves with `methodology.md`; each term moves to the section that actually
explains it, with a short glossary on `methodology/index.md` linking out so an evaluator still has
one place to look.

### Two hazards this refactor walks into

**The `exclude_docs` stale key.** `mkdocs.yml:33` excludes `methods/dream_come_true.md`, and its own
comment warns that if the key stops matching, the unpublished method **silently reappears as an
un-navigable orphan**. Renesting the methods pages is exactly the edit that breaks it. **Convert it
into a build-time assertion:** every `METHODS` slug must appear in either `nav` or `exclude_docs`,
turning a silent reappearance into a build failure. The hazard exists today independent of this
project; the refactor makes it likely.

**URL breakage.** MkDocs nav hierarchy is independent of file paths, so the methods pages *could* be
shown nested while their files and URLs stay put. **Decision: move them anyway**, taking
`/methodology/methods/peel/`. The site is young and lightly linked, `methodology.md`'s own URL breaks
regardless, and a redirect plugin would be one more pinned dependency plus precisely the
history-justified path the no-legacy rule refuses.

## §7 Sequencing

Six pieces. Each ships to the public site on its own; none blocks on a later one.

**A · Truth pass + IA** — *no JS, no new Python.*
The four defects of §Why; `methodology.md` split into five prose pages; `results/bakeoff.md` and
`results/nairobi.md` (both from artifacts already committed); `reproduce.md`; the renest; the
nav/`exclude_docs` build assertion; the generated method count.
→ *Ships:* the site stops being wrong. Highest value, lowest risk, no new tooling.
→ **Goes straight to an implementation plan from §6 + §Why.** No separate spec — it introduces no
new architecture.

**B · The graph artifact type** — *Python only.* Derive the drawable graph once; `render.py` gains a
graph mode; flagship blocks get graph PNGs.
→ *Ships:* Permeability gets its images, and every future widget's static fallback exists **before**
the widget does.

**C · Data bundle + widget substrate** — *the risky piece.* Per-block city score table, full prefix
tables, block-tier bundles, palette constants, generated `.d.ts`, the `web/` toolchain, mount
contract, `StateSource`, canvas renderer — proven end to end by exactly one widget, `PermGraph`
(smallest payload, biggest teaching payoff, and B just built its Python twin, so the two can be
diffed).
→ *Ships:* Permeability goes interactive.

**D · The remaining four widgets** — independent of each other, parallelizable.
`DisplacementField`, `Frontier`, `RegionGrow`, then `ScreenMap` **last within D**: 16k blocks is the
only real rendering-performance problem in the design.

**E · The Explore chain** — shared store, URL sync, stage rail. Thin once D lands.

**F · Draw-your-own-road** — pinned Pyodide, lazy boot, the import-closure CI guard, authoring UI.
Deliberately last: the only piece that is pure upside, and everything else is complete without it.

**Why `PermGraph` before `ScreenMap`,** since the opposite is tempting: A's headline fix is *the
screen changed and here is the evidence*, and that lands fine on committed PNGs — `city_map.png`,
`precision_recall.png`, `settlements.png` are already in `examples/screen-bakeoff/`. Permeability has
**nothing** today beyond a one-line definition. B+C take it from one line to the best figure on the
site.

## Open questions

* **The Khayelitsha claim** (`docs/background.md:39`) was measured under the retired
  `density_compactness` metric. Whether `depth_density_proxy` also peaks there must be checked
  before the sentence is rewritten rather than assumed; piece A should verify, not paraphrase.
* **Nairobi's screening tier.** Cape Town has ground truth; Nairobi has none
  (`examples/screen-bakeoff/README.md` says so explicitly). Whether the city tier ships for Nairobi
  at all — and if so, without the precision/recall readout — is a piece-D decision.
