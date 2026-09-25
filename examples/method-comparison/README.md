# method-comparison

*Seven reblockers head-to-head on one deep informal block, small enough that even `topology` runs. The set spans the families this project ships: whole-graph `topology`, least-cost `clearance` and its loop-closing refinement, the cycle-native Loop Network, the objective optimised directly over paths, a Manhattan grid, and the real as-built OSM footpaths.*

**Metric:** `one deep block — every method, including the single-block-only prior art` — the block is pinned by id; the metric drives the colouring only.

## 3. The permeability frontier (benefit vs added road)

The frontier is the whole trade-off: **permeability** (benefit — the only benefit axis) on the y-axis against **displacement** (cost — the only cost axis) on the x-axis, one line per method. Pareto-dominance — which method buys more permeability for less displacement — reads straight off it (raw per-method samples are in `frontier_permeability.csv`, this dir):

![permeability vs displacement](frontier_ZAF.9.3.1_1_40972.png)

**Before any road is added**, the same region in both colorings: access-depth (blue = at a street, red = deep interior) vs permeability potential (dark = hard to escape, light = easy):

| access-depth | permeability potential |
|---|---|
| ![access-depth](before_depth.png) | ![permeability potential](before_perm.png) |

## 4. Each method on the ground

**Watch each method reblock** — roads added busiest-first, each preceded by whatever it needs to reach the street, so every frame is a network you could actually build. Every animation stops where its network first reaches the matched-permeability standard, so they end at the same benefit and you can read the disruption each one spent getting there; a method that never reaches it runs to its own full network. The deep interior drains as the network reaches in:

| Least-Cost Tree | Looped Tree | Loop Network | Loop Network (desire-routed) | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) | Topology |
|---|---|---|---|---|---|---|---|---|
| ![Least-Cost Tree](reblock_clearance.gif) | ![Looped Tree](reblock_clearance_looped.gif) | ![Loop Network](reblock_cycle_native.gif) | ![Loop Network (desire-routed)](reblock_cycle_native_betweenness_contrast.gif) | ![Grid](reblock_euclidean_grid.gif) | ![Frontage (street-priced)](reblock_greedy_arterial_access_displacement.gif) | ![OSM Footpaths](reblock_osm_footpaths.gif) | ![Direct Objective (LP)](reblock_resistance_lp.gif) | ![Topology](reblock_topology.gif) |

### Matched permeability (primary)

Every method truncated where permeability first reaches the standard target, so this compares **what each spends to get there** — in homes displaced and in metres of road. Pinning the benefit and comparing costs is the sounder direction: both costs appear in their own units, so no exchange rate between homes and metres is needed.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Looped Tree | 83 m | 4.8% | 60.7% |  |
| Least-Cost Tree | 89 m | 5.0% | 61.7% |  |
| Loop Network (desire-routed) | 146 m | 5.1% | 73.6% |  |
| Loop Network | 89 m | 3.5% | 61.8% |  |
| Grid | 130 m | 7.3% | 70.2% |  |
| Frontage (street-priced) | 178 m | 4.3% | 63.9% |  |
| OSM Footpaths | 174 m | 6.0% | 71.9% |  |
| Direct Objective (LP) | 120 m | 4.3% | 62.2% |  |
| Topology | 106 m | 6.3% | 61.6% |  |

> **On the access-objective arterial row.** This example scores ONE pinned block, and `objective=access` maximises a sum of squared *integer* depths — so different networks routinely tie exactly, and the greedy picks between them arbitrarily. Perturbing the gains by 1e-10 moves this method's burden reduction by up to 15% on a third of blocks. Its number here is one draw, not a stable estimate; the medians over many blocks reported in the notes are the trustworthy form. No other method in this table has that sensitivity.

Access-depth coloring:

| Looped Tree | Least-Cost Tree | Loop Network (desire-routed) | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) | Topology |
|---|---|---|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_depth.png) | ![Least-Cost Tree](after_clearance_perm_depth.png) | ![Loop Network (desire-routed)](after_cycle_native_betweenness_contrast_perm_depth.png) | ![Loop Network](after_cycle_native_perm_depth.png) | ![Grid](after_euclidean_grid_perm_depth.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_perm_depth.png) | ![OSM Footpaths](after_osm_footpaths_perm_depth.png) | ![Direct Objective (LP)](after_resistance_lp_perm_depth.png) | ![Topology](after_topology_perm_depth.png) |

Permeability-potential coloring:

| Looped Tree | Least-Cost Tree | Loop Network (desire-routed) | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) | Topology |
|---|---|---|---|---|---|---|---|---|
| ![Looped Tree](after_clearance_looped_perm_perm.png) | ![Least-Cost Tree](after_clearance_perm_perm.png) | ![Loop Network (desire-routed)](after_cycle_native_betweenness_contrast_perm_perm.png) | ![Loop Network](after_cycle_native_perm_perm.png) | ![Grid](after_euclidean_grid_perm_perm.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_perm_perm.png) | ![OSM Footpaths](after_osm_footpaths_perm_perm.png) | ![Direct Objective (LP)](after_resistance_lp_perm_perm.png) | ![Topology](after_topology_perm_perm.png) |

### Matched displacement (secondary)

Every method truncated to the same displacement %, so this compares the **permeability each buys for the same home-cost**. This lens budgets homes but **not road length**, and the two are not proportional — a metre through a gap displaces far less than a metre through the dense interior. So read `road_m` beside `permeability`: a method showing a higher number at several times the road length has not been shown to be better, only more expensive. Prefer the matched-permeability lens above, which prices both costs.

| Method | Road | Displacement | Permeability | Note |
|---|---|---|---|---|
| Least-Cost Tree | 191 m | 10.7% | 72.9% |  |
| Looped Tree | 177 m | 10.3% | 74.0% |  |
| Loop Network (desire-routed) | 380 m | 10.2% | 91.1% |  |
| Loop Network | 335 m | 10.5% | 88.0% |  |
| Grid | 218 m | 13.0% | 78.5% |  |
| Frontage (street-priced) | 493 m | 10.2% | 91.0% |  |
| OSM Footpaths | 287 m | 11.9% | 85.7% |  |
| Direct Objective (LP) | 438 m | 10.1% | 91.2% |  |
| Topology | 212 m | 10.1% | 72.4% |  |

> **On the access-objective arterial row.** This example scores ONE pinned block, and `objective=access` maximises a sum of squared *integer* depths — so different networks routinely tie exactly, and the greedy picks between them arbitrarily. Perturbing the gains by 1e-10 moves this method's burden reduction by up to 15% on a third of blocks. Its number here is one draw, not a stable estimate; the medians over many blocks reported in the notes are the trustworthy form. No other method in this table has that sensitivity.

Access-depth coloring:

| Least-Cost Tree | Looped Tree | Loop Network (desire-routed) | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) | Topology |
|---|---|---|---|---|---|---|---|---|
| ![Least-Cost Tree](after_clearance_disp_depth.png) | ![Looped Tree](after_clearance_looped_disp_depth.png) | ![Loop Network (desire-routed)](after_cycle_native_betweenness_contrast_disp_depth.png) | ![Loop Network](after_cycle_native_disp_depth.png) | ![Grid](after_euclidean_grid_disp_depth.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_disp_depth.png) | ![OSM Footpaths](after_osm_footpaths_disp_depth.png) | ![Direct Objective (LP)](after_resistance_lp_disp_depth.png) | ![Topology](after_topology_disp_depth.png) |

Permeability-potential coloring:

| Least-Cost Tree | Looped Tree | Loop Network (desire-routed) | Loop Network | Grid | Frontage (street-priced) | OSM Footpaths | Direct Objective (LP) | Topology |
|---|---|---|---|---|---|---|---|---|
| ![Least-Cost Tree](after_clearance_disp_perm.png) | ![Looped Tree](after_clearance_looped_disp_perm.png) | ![Loop Network (desire-routed)](after_cycle_native_betweenness_contrast_disp_perm.png) | ![Loop Network](after_cycle_native_disp_perm.png) | ![Grid](after_euclidean_grid_disp_perm.png) | ![Frontage (street-priced)](after_greedy_arterial_access_displacement_disp_perm.png) | ![OSM Footpaths](after_osm_footpaths_disp_perm.png) | ![Direct Objective (LP)](after_resistance_lp_disp_perm.png) | ![Topology](after_topology_disp_perm.png) |


## How this was generated

This example is machine-generated — one self-logging command emits the data, maps, curves, and this README:

```bash
pixi run python -m scripts.gen_example method_comparison
```
The full run log is in [`run.log`](run.log).

