# Method comparison: six reblockers, head-to-head on one deep block

Six reblockers graded head-to-head by a single flow metric — **permeability** — against a single cost
— **displacement** — on one deep informal block small enough that even `topology` runs (it's
**single-block-only**: a multi-block region gives it a disconnected source node and it crashes). The
six span the families this project ships: whole-graph `topology`, least-cost `clearance` and its
loop-closing refinement `clearance_looped`, the cycle-native `cycle_native` (its atomic move is a
**loop** out from the street and back, so it is bridgeless by construction rather than a tree with
connectors bolted on), a Manhattan `euclidean_grid`, and the real as-built `osm_footpaths`. The companion [`multiblock_depth`](../multiblock_depth/) flagship runs the scalable
methods on a whole settlement.

The block is **`ZAF.9.3.1_1_40972`** — the deepest block (by the depth proxy `√(n·A)/P`) in a
topology-tractable size window: **263 parcels, up to 7 deep**, auto-picked, no hand tuning.
📍 [See it on Google Maps](https://www.google.com/maps/@-33.97795,18.58064,18z) (every run logs this
link for its selection).

## The metric: permeability

Everything is graded on **one** number. Model collective egress as an electrical flow: every parcel
injects one unit of "escape current"; the existing street is ground. Parcels sit on an always-present
**footpath mesh** whose resistance rises where buildings crowd the path — an open gap is easy to walk,
a cramped alley between tight footprints is not (each footpath edge carries an *open-corridor* factor
`max(ε, 1 − 2r₀/d)`, `d` the parcel spacing and `r₀` a building half-width that scales with local
density). Roads are low-resistance shortcuts that *upgrade* the mesh where they run — so roads matter
most exactly where the fabric is densest and the footpaths are worst. The total dissipated power of
that flow measures how hard it is for *everyone* to get out at once — folding **distance**,
**contention** (many parcels sharing one road, penalised quadratically), **redundancy** (loops spread
the current), and now **local crowding** into a single scalar.

> **permeability = 1 − P(roads) / P(no roads)** ∈ [0, 1)

0 = the walkable status quo; higher = a more permeable network. It is **monotone** (roads only add
conductance, so permeability only rises — Rayleigh), **all-parcels** (no membership, no on/off), and a
**single sparse solve**. It rewards reach and loops and punishes bare drainage trees, which funnel all
egress through one root. Cost is **displacement** — the expected fraction of homes a road set grazes
(`Σcᵢ / n_buildings`, each building a disk of radius = ½ its nearest-neighbour distance, contributing
its probability of being clipped). Both axes are fractions in [0, 1); nothing else is reported.

## Reproduce

One command reblocks all six methods and emits the frontier, both lens tables, **and** every
before/after render (both colourings), no hand-placed assets — the whole example reproduces offline:
```bash
pixi run python -m scripts.gen_method_comparison
```
It reblocks the pinned block with each method (from ONE propose each), renders a before-heatmap plus
per-method after-heatmaps in both colourings, and builds the permeability-vs-displacement frontier.
`topology` is the slow pole (~7 min); every other method runs in seconds.
`osm_footpaths` loads a committed OSM snapshot (`desire_lines_40972.geojson`, 22 mapped ways — see
`scripts/fetch_desire_lines_snapshot.py`). The console output is captured in [`run.log`](run.log), the
source of truth for the tables below.

## Before: the block with no roads

The same block in **both colourings** — access-depth (blue = at a street, red = deep interior) and the
metric-aligned **permeability potential** (light = easy to escape, dark = hard). The dark core is the
reblocking target: a compact ring of deep parcels every method has to reach.

| access depth | permeability potential |
|---|---|
| ![before depth](before_depth.jpg) | ![before permeability](before_perm.jpg) |

## The frontier: permeability bought per home displaced

The whole trade-off, one line per method — **permeability** (y) against **displacement** (x). Pareto
dominance reads straight off it: up-and-to-the-left is better (more permeability, fewer homes moved).

![permeability vs displacement](frontier_ZAF.9.3.1_1_40972.png)

The shape tells the story, and it is a story about **loops**. The two networks that carry real
circulation — the as-built `osm_footpaths` and the cycle-native `cycle_native` — buy the most
permeability for the fewest homes, while **`clearance` is the laggard through the critical
mid-range**: a least-cost drainage *tree* has no redundancy by construction, so it plateaus while
every looped network keeps climbing. `euclidean_grid` starts slowest of all (a blind grid wastes
its first roads far from the deep core) and only catches up by paving to heavy displacement.

## Lens A — matched displacement: permeability at an equal home-cost

Truncate every method to its first road prefix that displaces **≥ 10% of homes**, then compare the
**permeability** each buys for that shared cost (every method reaches 10%, so all are compared honestly
at the budget):

| method | road | displacement | **permeability** |
|---|---|---|---|
| **cycle_native** | 242 m | 11.4% | **0.763** |
| osm_footpaths | 234 m | 12.0% | 0.720 |
| euclidean_grid | 270 m | 14.9% | 0.702 |
| clearance | 143 m | 12.3% | 0.678 |
| topology | 142 m | 10.6% | 0.670 |
| clearance_looped | 110 m | 10.1% | **0.644** |

At an equal ~10% home-cost, **the real footpaths are the most permeable network** (0.823) — the paths
people already walk form a loopy mesh no optimiser here beats at this budget — with the repulsion
arterial second (0.709) at the least road and fewest homes of the leaders. The bare `clearance` tree is
worst (0.588): same homes moved, a quarter less permeability, because a tree can't spread egress.

Access-depth colouring (deep interior draining as roads reach in):

| topology | clearance | greedy_arterial_repulsion |
|---|---|---|
| ![topology](after_topology_disp_depth.jpg) | ![clearance](after_clearance_disp_depth.jpg) | ![arterial](after_greedy_arterial_repulsion_disp_depth.jpg) |
| **clearance_looped** | **euclidean_grid** | **osm_footpaths** |
| ![clearance_looped](after_clearance_looped_disp_depth.jpg) | ![euclidean_grid](after_euclidean_grid_disp_depth.jpg) | ![osm_footpaths](after_osm_footpaths_disp_depth.jpg) |

Permeability-potential colouring (dark core lightening as escape gets easier):

| topology | clearance | greedy_arterial_repulsion |
|---|---|---|
| ![topology](after_topology_disp_perm.jpg) | ![clearance](after_clearance_disp_perm.jpg) | ![arterial](after_greedy_arterial_repulsion_disp_perm.jpg) |
| **clearance_looped** | **euclidean_grid** | **osm_footpaths** |
| ![clearance_looped](after_clearance_looped_disp_perm.jpg) | ![euclidean_grid](after_euclidean_grid_disp_perm.jpg) | ![osm_footpaths](after_osm_footpaths_disp_perm.jpg) |

## Lens B — matched permeability: homes displaced to reach a standard

Truncate every method to its first prefix reaching **permeability ≥ 0.60**, then compare the
**displacement** each spends to get there (lower is better). Every method clears the bar on this block:

| method | **displacement to P ≥ 0.60** | road |
|---|---|---|
| **osm_footpaths** | **4.3%** | 120 m |
| topology | 7.4% | 109 m |
| clearance_looped | 9.2% | 101 m |
| greedy_arterial_repulsion | 11.5% | 183 m |
| clearance | 12.2% | 174 m |
| euclidean_grid | **13.1%** | 270 m |

To reach the same permeability standard, **`osm_footpaths` moves the fewest homes (4.3%)** and
**`euclidean_grid` moves the most (13.1%)** — 3× as many for an identical outcome, because a blind grid
spends its first roads away from the deep core. The bare `clearance` tree is nearly as expensive
(12.2%): it has to pave its way to a permeability a loop gets for free.

Access-depth colouring:

| topology | clearance | greedy_arterial_repulsion |
|---|---|---|
| ![topology](after_topology_perm_depth.jpg) | ![clearance](after_clearance_perm_depth.jpg) | ![arterial](after_greedy_arterial_repulsion_perm_depth.jpg) |
| **clearance_looped** | **euclidean_grid** | **osm_footpaths** |
| ![clearance_looped](after_clearance_looped_perm_depth.jpg) | ![euclidean_grid](after_euclidean_grid_perm_depth.jpg) | ![osm_footpaths](after_osm_footpaths_perm_depth.jpg) |

Permeability-potential colouring:

| topology | clearance | greedy_arterial_repulsion |
|---|---|---|
| ![topology](after_topology_perm_perm.jpg) | ![clearance](after_clearance_perm_perm.jpg) | ![arterial](after_greedy_arterial_repulsion_perm_perm.jpg) |
| **clearance_looped** | **euclidean_grid** | **osm_footpaths** |
| ![clearance_looped](after_clearance_looped_perm_perm.jpg) | ![euclidean_grid](after_euclidean_grid_perm_perm.jpg) | ![osm_footpaths](after_osm_footpaths_perm_perm.jpg) |

## Each method under permeability

- **`osm_footpaths`** — the REAL informal network (mapped OSM footpaths, not an optimiser's output) —
  is the **most permeable per home displaced** on both lenses: 0.823 at a matched 10%, and it reaches
  the 0.60 standard at just **4.3% of homes**. The worn paths people already walk are a genuine,
  loop-rich, low-displacement mesh — a hard baseline to beat.
- **`greedy_arterial_repulsion`** is the **best optimiser here** and the low-road, low-home runner-up:
  second-highest permeability at matched displacement (0.709) for the least road (183 m). Because its
  cost is proximity to homes, it threads through-corridors along the gaps *between* clusters, closing
  loops while grazing few footprints — and it scales to the region (see
  [`multiblock_depth`](../multiblock_depth/)).
- **`euclidean_grid`** reaches high permeability eventually (0.699 at matched displacement) — a regular
  grid is loops by construction — but it is the **most expensive to a fixed standard** (13.1% of homes
  to hit 0.60) and the slowest starter, because a blind grid ignores where the deep parcels actually are.
- **`clearance_looped`** adds redundant connectors on top of a clearance base, lifting permeability well
  above plain clearance (0.661 vs 0.588 at matched displacement, and it reaches 0.60 at 9.2% vs
  clearance's 12.2%) — the loops are exactly what the metric rewards. On the whole settlement it is the
  region-scale workhorse.
- **`topology`** blankets the fabric with a whole-graph optimiser: it reaches the 0.60 standard cheaply
  (7.4%) but its curve then flattens into the pack — it builds a reach-everywhere tree, not a mesh, so
  extra road buys little extra permeability. Single-block-only.
- **`clearance`** is the balanced least-cost option and the **least permeable**: a drainage tree has no
  backup routes *by construction*, so it lags at matched displacement (0.588) and its mid-range plateau
  is the clearest signature on the frontier of a network that can't spread egress. Cheap road, expensive
  permeability.

**The takeaway:** permeability rewards **reach and loops** and punishes **bare drainage trees**, and
because footpaths are now worst exactly in the cramped core, it credits the roads that actually reach
that core. The real footpaths and the repulsion arterial deliver the most escape for the fewest homes
moved; the least-cost tree delivers the least. Pick the method by where you can afford to sit on the
frontier — and see [`multiblock_depth`](../multiblock_depth/) for the region-scale run of the scalable
methods.
