# Base cost × loop policy across six regions: two of my recommendations retracted

**Date:** 2026-07-29
**Status:** measured, 6 regions (3 metrics × 2 cities), controlled.
`scratchpad/ot/tree_grid.py`.

## Why this was needed

The four tree methods in the example were **not comparable as configured** — each inherited
whatever its own config set, so they differed on three axes at once:

| method | base cost | depth_target | max_roads | loops |
|---|---|---|---|---|
| Plain Tree | pure length | 2 | 400 | none |
| Looped Plain Tree | pure length | 1 | 3000 | ≥40 m |
| Looped Tree | repulsion | 3 | 3000 | ≥40 m |
| Looped Tree (cheap) | repulsion | 3 | 3000 | ≥5 m |

Only the last pair was a clean comparison. This runs the full 2×3 grid — {repulsion, pure length}
× {none, ≥40 m, ≥5 m} — with `depth_target=3`, `max_roads=3000`, `budget_frac=0.30`,
`search_radius_m=60` fixed in every cell.

## 1. Repulsion earns its place (RETRACTS "a plain tree essentially ties the flagship")

Lens A: repulsion wins **15 of 18** cells, median advantage **+0.012** permeability.
Lens B: essentially a wash, repulsion using **+1.1%** more road at the median.

Pure length never dominates. My earlier claim was entirely the `depth_target` 2-vs-3 confound.

## 2. Loops earn their place — but only where the region is DEEP

Lens A, change in permeability from adding loops to the repulsion base:

| region | no loops | +loops ≥40 m | +loops ≥5 m |
|---|---|---|---|
| depth_density capetown | 0.698 | **+0.084** | **+0.087** |
| depth capetown | 0.729 | **+0.054** | **+0.051** |
| density_compactness nairobi | 0.788 | +0.021 | +0.021 |
| depth_density nairobi | 0.812 | +0.016 | +0.015 |
| density_compactness capetown | 0.742 | +0.000 | +0.004 |
| depth nairobi | 0.777 | −0.003 | −0.008 |

And on Lens B the two deep Cape Town regions drop **2187 m** and **644 m** of road for the same
permeability.

The pattern is coherent: in a deep region a tree forces long detours and a loop shortcut is worth a
lot; in a shallow compact one everything is already near a street, so loops add road for nothing.
**`density_compactness` capetown — the single region I had been testing on — is the case where
loops help least.** That is why I read them as not earning their place.

## 3. Cheap loops do NOT generalize (RETRACTS the flagship "improvement")

`min_loop_len_m` 40 → 5 was better on **2 of 6** regions on Lens A and **1 of 6** on Lens B. The
density_compactness Cape Town result that motivated it (+0.0037 permeability, −40 m road) is
region-specific, not an improvement to the method.

**The flagship keeps `min_loop_len_m: 40.0`.** `clearance_looped_cheap` is deleted rather than kept
as an unselected option.

## Consequences

Removed from the example and from `all_methods`: `demand_greedy_uniform`, `demand_looped`,
`clearance_looped_cheap`, and `demand_greedy` from the example (its desire-line prior was already
measured not to pay). The example now carries five live methods: `osm_footpaths` (the real
reference), `clearance_looped` (flagship), `euclidean_grid` (baseline),
`greedy_arterial_repulsion` (a different intervention), `flow_paths` (which lands on the real
network's position on both lenses).

`DemandGreedyReblocker` stays as a class with its tests and `conf/method` entry — it is the
mechanism the "pure length" arm of this grid is built from, and deleting it would remove the
ability to re-run the comparison that vindicated repulsion.

## The lesson worth keeping

Both retracted claims came from measuring on **one region** and generalizing. The grid cost about
40 minutes of wall clock and reversed both. Any claim about a method here should be checked across
the six example regions before it is acted on, because they differ in exactly the property —
depth — that decides whether loops pay.
