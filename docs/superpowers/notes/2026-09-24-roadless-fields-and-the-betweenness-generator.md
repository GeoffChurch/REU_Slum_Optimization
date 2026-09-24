# Routing toward betweenness ridges beats clearance on both lenses; on large blocks it is the frontier's middle point

**Date:** 2026-09-24
**Status:** measured, and shipped as four method presets: `betweenness_tree`,
`betweenness_tree_contrast`, `betweenness_looped` and `betweenness_looped_contrast`. The code is
`src/reblock/methods/betweenness/`, the field settings are `conf/betweenness.yaml`, and the plan is
`docs/superpowers/plans/2026-09-24-betweenness-desire-generator.md`.
Revises two July notes:
- [`clearance-desireline-alignment-findings`](2026-07-14-clearance-desireline-alignment-findings.md)
  rejected betweenness because it piles into pinch points. Repulsion removes that failure.
- The "ceiling" in [`desire-line-detection`](2026-07-15-desire-line-detection.md) is the chance
  rate.

The research code ran from a session scratchpad and was not kept. This note describes what it did.

## The question
The owner's idea (2026-09-23) was to colour every point of a block by the best road through it,
knowing no roads, so that lanes and desire lines emerge from the gaps between buildings. That raises
two questions:
- **As a picture:** does such a field find the real lanes?
- **As a generator:** does routing roads toward it make better networks?

Everything was measured on real Open Buildings footprints.

A picture is scored as a ranking of a block's free cells against "within 4 m of the ground truth":
- **AP** is the ranking's average precision.
- **Recall and precision** are July's length-matched metric: threshold the field so that its
  skeleton is as long as the ground truth, then score with a 4 m tolerance.
- **Core** restricts both the cells and the truth to those at least 10 m from the street.

A generator is scored exactly as the consensus study scores its arms, with
`scripts/consensus_matrix.score_recipient` (see [`consensus-re-measured`](2026-09-23-consensus-re-measured.md)):
- **Lens A** is permeability at D = 10% displacement.
- **Lens B** is the displacement needed to reach P* = 0.60.

## The fields

| field | what it computes | verdict |
|---|---|---|
| path field (the owner's) | the best road through each point: longest smooth road x width, with a curvature penalty only | best ranking on small blocks; saturates on large ones |
| repelled betweenness | per cell, how many best routes run through it: homes to the street (egress) and home to home (all pairs), repelled by buildings | best line placement |
| soft path field | log-sum-exp over routes instead of the max | **negative** |
| curvature-aware repelled betweenness | betweenness whose routes pay for turning | best picture overall; **shipped** |
| contrast against a no-buildings prior | the same counts, set against the counts on the empty block | sharper picture, worse ranking, better generator; **shipped** |

- **The path field** is a grayscale path opening. It uses 8 rotated cones and an elastica bend cost,
  with exact rotated clearance. Its 25 primitive steps are bucketed so that a straight road costs
  the same at any angle, to within 0.8%.
  - The owner dropped a jerk term, because a curvature penalty already punishes tight loops.
  - Open ground scores as road. The owner ruled that fine: it is "road potential".
- **Curvature-aware betweenness** takes shortest paths over (cell, 16 direction buckets).
  - Changing bucket costs `lam * dtheta^2 / 2` m.
  - The cost density is `1 + (r0 / clearance)^2`.
  - The measured setting is lam 50, r0 2 m.
- **The contrast** needs a prior, E: the same computation on the block with no buildings, where the
  density is exactly 1. The field is the signed root Poisson deviance of the observed counts O
  against E.

## Ground truths
- **OSM footpaths are edge-biased on large blocks.** Take the spine block (Bloekombos,
  ZAF.9.3.1_1_5810, 6,619 buildings): 52% of its mapped footpath length lies within 10 m of the
  street, against 11% of its free cells. The core is mostly unmapped, so OSM cannot score a large
  block's core.
- **July's ceiling was the chance rate.** Plain clearance scores AP 0.348 against OSM on 40972, where
  the base rate is 0.351. `desire-line-detection`'s "ceiling" of ~0.50 recall and 0.35 precision is
  essentially chance, because 35% of free cells lie within 4 m of a footpath.
- **Microsoft RoadDetections** (`microsoft/RoadDetections`, ODbL, from Bing imagery) was taken from
  the 2025.04.28 drop, `Southern_Africa.zip`. The Cape Town extract has 162,951 lines and is cached
  outside the repo.
  - It matches the lanes Google Maps draws in Bloekombos. On the spine it has 9.8 km of interior road
    and 8.2 km in the core, against OSM's 3.1 km and 1.5 km.
  - On small blocks it is mostly perimeter streets. The median block has 9% of its length in the
    core, and only 79 of 218 blocks have at least 50 m there.
- **Google Maps is not a ground truth.** Its geometry must not be extracted or traced: its terms
  forbid it, and OSM forbids it as a source. The owner offered screenshots, and they were declined
  for scoring.

## The fields as pictures
These are paired medians against clearance over the 220 consensus recipients (60-300 buildings),
with 95% bootstrap intervals.

| comparison | result |
|---|---|
| path field - clearance, AP, OSM whole | +0.032 [+0.019, +0.049] |
| path field - clearance, AP, Microsoft core | +0.051 [+0.015, +0.108] |
| repelled betweenness - clearance, length-matched precision (OSM whole, OSM core, Microsoft core) | +0.05 to +0.13 |
| curvature-aware - plain betweenness, AP, matched settings | OSM whole +0.004, OSM core +0.009, Microsoft core +0.020; better in 76-96% of blocks; precision unchanged |
| contrast - raw, AP | OSM whole -0.027, Microsoft core -0.014; contrast higher in only ~30% of blocks |

- **Repulsion removes July's failure.** Plain betweenness concentrated in the densest pinch points.
  With repulsion, the busiest 5% of cells have a median clearance of 2-5 m, against 1.4-2.3 m over
  all free cells.
- **The path field saturates on the spine.** A max over roads finds some long, smooth road through
  every walkable cell of a large connected fabric, so the field is near chance there. It takes
  counting (a sum over routes) to make shared lanes stand out.
- **All picture scores on small blocks are close to chance:** every field stays within ~0.06 AP of
  it against OSM.

On the spine, scored against Microsoft's core (chance AP 0.149):

| field | core AP | length-matched precision |
|---|---|---|
| clearance | 0.173 | 0.23 |
| plain betweenness | 0.227 | not recorded |
| curvature-aware betweenness (lam 50, r0 2) | 0.266 | 0.51 |

That is one block.

## The soft path field: negative
The soft field takes a log-sum-exp over routes, with a hard max inside each direction bucket. It
was meant to count routes rather than keep only the best one, and it fails its own synthetic checks:
- **Path entropy is extensive.** Zig-zag variants inside a corridor cost almost nothing, and their
  number grows like exp(length x width). So at T >= 3 m the field ranks corridor volume, not shared
  routes.
- **The count tests go the wrong way.** A crossing should outscore a lone corridor, but X - corridor
  is -2.49 at T = 3 and -29.4 at T = 10. The Y-trunk test fails the same way.
- **It has an angle bias** of 7-13% at T >= 3, against 1.4% for the hard field.

On the 220 blocks at lam 50, the lift in OSM-whole AP over clearance is:

| field | AP lift |
|---|---|
| hard path field | +0.030 |
| soft, T = 1 | +0.025 |
| soft, T = 3 | +0.010 |

That is a tie at best. Counting routes needs one route per origin-destination pair, which is
betweenness, not a soft max over lattice paths. That is what led to the curvature-aware betweenness.

## The contrast: a sharper picture that ranks worse
The deviance contrast takes the background to zero and leaves thin channels on Microsoft's lanes. It
also raises core recall by +0.04 to +0.08. But it ranks worse than the raw counts (see the table
above). Dividing out centrality removes real signal, because real lanes do run through the middle of
a block and toward its exits.

As a generator it is the better of the two on small blocks (below). The code's explanation is that
it spreads the network to the detours the buildings force. That mechanism is *untested*.

## As a generator
The builder is `DemandGreedyReblocker`, routed toward the field's ridges:
- **Ridges:** for q in 0.80, 0.85, 0.90, 0.95 and 0.98, threshold the field at its q-quantile over
  the free cells and skeletonize the result (Zhang-Suen).
- **Desire groups:** each skeleton becomes one group of weight 1/5.
- **Builder settings:** those of `conf/method/demand_greedy.yaml`.
- **Looped arms:** the builder wrapped in `LoopClosureRefiner`.

### 220 small recipients
These are paired against clearance, whose medians are 0.763 on Lens A and 0.066 on Lens B. The
fields were computed at 0.5 m, with every home as a source. The percentages are the share of blocks
where the arm is better.

| arm - clearance | Lens A permeability | Lens B displacement |
|---|---|---|
| toward raw | +0.022 [+0.014, +0.031], 65% | -0.006 [-0.010, -0.003], 66% |
| toward contrast | +0.032 [+0.023, +0.047], 71% | -0.007 [-0.010, -0.004], 65% |
| `NoDesire` (no field) | +0.000 | +0.001 |
| toward the real OSM footpaths (oracle) | +0.007, not significant | not recorded |

- **The gain is the field's.** `demand_greedy` with `NoDesire` ties clearance exactly.
- **The field guides to good roads, not to existing ones.** The oracle reproduces the footpaths
  (IoU at 10 m +0.140) but does not improve permeability.

### Held out
A strict replication is impossible, because the 220 are all the Cape Town blocks that pass the
study's selection rule. A relaxed rule selects sparser blocks, at 35-49 buildings/ha against 118:
- 60-300 buildings;
- at least 100 m of mapped interior footpath;
- an area cap.

| sample | arm | Lens A | Lens B |
|---|---|---|---|
| fresh Cape Town | raw | +0.025, n = 38, interval includes 0 | -0.005 [-0.007, -0.002], n = 140 |
| fresh Cape Town | contrast | +0.010, n = 38, interval includes 0 | -0.005 [-0.009, -0.002], n = 140 |
| Nairobi | raw | +0.054, n = 39 | -0.005 [-0.012, -0.002], n = 101 |
| Nairobi | contrast | +0.052 [+0.042, +0.074], 95%, n = 39 | -0.006 [-0.009, -0.002], n = 101 |

- **Lens B replicates in both cities.**
- **Lens A replicates in Nairobi.**
- **Fresh Cape Town is mostly unscoreable on Lens A**, because few of its sparse blocks reach 10%
  displacement.

### Bloekombos
The owner judged 40972 (263 parcels) too small to be robust. So the flagship is the spine block,
and after it the large-block study below.

| arm | Lens A | Lens B |
|---|---|---|
| cycle_native | 0.925 | 0.017 |
| looped contrast | 0.867 | 0.018 |
| greedy_arterial | 0.821* | 0.008 |
| raw | 0.819 | 0.033 |
| contrast | 0.815 | 0.048 |
| looped raw | 0.814 | 0.032 |
| clearance_looped | 0.788 | 0.028 |
| clearance | 0.745 | 0.047 |
| NoDesire | 0.721 | 0.053 |

\* greedy_arterial's whole network, which displaces only 3.5%.

The looped contrast beats clearance_looped on both lenses.

### 36 large blocks: the frontier
**The blocks** are those with at least 1,000 buildings at the top of the depth_density screen:
- Cape Town: 24 of the 33 eligible, 1,046-6,619 buildings;
- Nairobi: 12 of the 18 eligible, 1,182-4,353 buildings.

**The lineup** is the one tuned in `conf/example/explore.yaml`; `greedy_arterial` below is its
`greedy_arterial_access_displacement`. Everything ran at the footprint tier.

**The fields** were computed at 1 m from 400 sampled sources, which is the shipped setting.

The medians below are pooled over the 36 blocks:
- **Lens A** is taken over every block. Where an arm falls short of the 10% budget, it is scored on
  its whole network.
- **Lens B** is taken over the blocks that reach P*.

| arm | Lens A | reaches the Lens A budget | Lens B | full network road |
|---|---|---|---|---|
| cycle_native | 0.896 | 100% | 0.025 | 8.5 km |
| greedy_arterial | 0.880 | 44% (16 of 36) | 0.018 | 6.3 km |
| looped raw | 0.873 | 100% | 0.026 | 6.7 km |
| looped contrast | 0.871 | 100% | 0.026 | 6.7 km |
| raw | 0.856 | 94% | 0.028 | 5.2 km |
| contrast | 0.841 | 94% | 0.027 | 5.2 km |
| euclidean_grid | 0.822 | 17% | 0.045 | 2.7 km |
| clearance_looped | 0.818 | 89% | 0.031 | 3.3 km |
| clearance | 0.803 | 100% | 0.037 | 4.9 km |
| NoDesire | 0.792 | 100% | 0.038 | 4.8 km |

greedy_arterial's whole network displaces a median 8.4%.

The paired medians below come with 95% bootstrap intervals and the share of blocks where the first
arm is better:
- **Lens A** counts only the blocks where both arms reach the budget: n = 36 unless stated.
- **Lens B** counts the blocks where both reach P*: n = 36 throughout.

| comparison | Lens A | Lens B |
|---|---|---|
| raw - clearance | +0.056 [+0.037, +0.069], 88%, n = 34 | -0.010 [-0.012, -0.003], 78% |
| contrast - clearance | +0.042 [+0.031, +0.063], 88%, n = 34 | -0.006 [-0.011, -0.002], 81% |
| looped raw - clearance | +0.069 [+0.054, +0.080], 97% | -0.009 [-0.013, -0.003], 86% |
| looped contrast - clearance | +0.064 [+0.058, +0.079], 97% | -0.010 [-0.015, -0.003], 81% |
| looped raw - clearance_looped | +0.043 [+0.035, +0.060], 94%, n = 32 | -0.006 [-0.009, -0.003], 78% |
| looped contrast - clearance_looped | +0.048 [+0.033, +0.057], 97%, n = 32 | -0.004 [-0.009, -0.003], 78% |
| looped raw - cycle_native | -0.015 [-0.026, -0.009], 11% | -0.001 [-0.004, +0.003], 56% |
| looped contrast - cycle_native | -0.022 [-0.031, -0.008], 8% | -0.002 [-0.003, +0.003], 53% |
| looped arms - greedy_arterial | -0.032 to -0.038, n = 16 | +0.008; greedy_arterial better in 83-94% |
| cycle_native - greedy_arterial | -0.005 (a tie), n = 16 | +0.009 |

The Cape Town and Nairobi comparisons hold separately:
- **Against clearance,** every generator arm's intervals exclude 0 on both lenses in both cities.
- **Against clearance_looped,** the looped arms' intervals exclude 0 on both lenses in both cities.
- **The tree arms against clearance_looped** win only pooled (Lens A +0.025 and +0.017). In Cape Town
  alone their intervals include 0 on both lenses, so the tree presets are not claimed to beat
  clearance_looped.

**The Pareto frontier.** This counts the blocks where each arm is on the (Lens A, Lens B) frontier,
out of every arm run:

| arm | on the frontier |
|---|---|
| greedy_arterial | 30 of 36 |
| cycle_native | 22 of 36 |
| looped raw | 11 |
| looped contrast | 8 |
| raw | 6 |
| contrast | 1 |
| clearance, clearance_looped, euclidean_grid, NoDesire | 0 each |

- **Some generator arm is on the frontier in 17 of 36 blocks:** 8 of the 24 in Cape Town and 9 of
  the 12 in Nairobi.
- **It is usually the middle point,** between cycle_native (the highest Lens A) and greedy_arterial
  (the lowest Lens B).

## Not `flow_paths`
[`flow-paths-mimicry`](2026-07-29-flow-paths-mimicry.md) (2026-07-29) also routes all pairs, on the
substrate, with reinforcement. But it keeps the busiest edges *as the roads*. It was refuted as a
mimicry model: its footpath IoU ties everything at 0.27-0.30, and clearance beats it at matching the
streets.

The betweenness generator is a different thing:
- **It uses the field as a prior** for a builder that guarantees coverage.
- **It is scored on the published lenses,** not on mimicry.

The oracle row above makes the same point from the other side: reproducing real paths does not
make a network better.

## What shipped, and what did not
**Shipped: `reblock.methods.betweenness`.**
- **`BetweennessDesire`** is a `DesireLineSource`.
- **The contrast is a Strategy,** `FieldContrast`: either `RawShare` or `PriorDeviance(floor=1e-9)`.
- **Counts are memoized** through `derive_graph.derive`. The key carries the block and its tier,
  but not `workers`.
- **numba joins the runtime dependencies,** through the typed `reblock._jit.njit` wrapper.
- **Zhang-Suen thinning is ported from scikit-image** and pinned equal to it by test, so `src`
  imports no scikit-image.

**Shipped: `conf/betweenness.yaml`.** It holds the `raw` and `contrast` sources, both at the
large-block setting:
- res 1.0 m, r0 2.0 m, lam 50, 400 sources, seed 0;
- ridge quantiles 0.80 to 0.98.

`workers: ${cpu_count:}` forks one process per core for the all-pairs pass. A runner that already
forks per block must set `workers=1`.

**Shipped: four presets,** each a measured operating point. All four are also in
`compare_config`'s `all_methods`, and none is on a published example lineup.

| preset | builder | desire | frontier blocks (of 36) |
|---|---|---|---|
| `betweenness_tree` | `DemandGreedyReblocker` | raw | 6 |
| `betweenness_tree_contrast` | `DemandGreedyReblocker` | contrast | 1; the best Lens A on the 220 small blocks |
| `betweenness_looped` | `LoopClosureRefiner` around it | raw | 11 |
| `betweenness_looped_contrast` | `LoopClosureRefiner` around it | contrast | 8 |

The looped presets carry the region-scale loop settings that the frontier study measured, which is
explore's clearance_looped tuning: budget_frac 0.30, min_loop_len_m 40, search_radius_m 60, and so
on.

**Not shipped.** These stayed in the scratchpad and are recorded here only:
- the path field;
- the soft path field (dominated, above);
- plain betweenness.

The path field ranks best on small blocks, but it was never run as a generator. It is *untested*
there. That test is what would put it back in play, and saturation predicts that it would not help
on large blocks.

**The port is bit-identical to the research code.** This was checked against a fresh derive cache:
- **Counts:** all four count arrays (egress and all-pairs, observed and prior) matched exactly on 6
  blocks, 24 arrays in all. The blocks cover both cities, both resolutions (0.5 m with every home as
  a source, and 1 m with 400 sampled sources) and non-integer sampling scales.
- **Proposals:** 3 recipients x 2 contrasts matched exactly at every layer: the field, the
  desire-segment groups (in order) and the road geometries and widths. The largest recipient
  (5618, 759 buildings) was one of them.

**Timings.** Clearance runs in seconds. The field takes minutes on the 6,619-building block, even
with 40 workers.

| run | where | time |
|---|---|---|
| production, observed + prior | Bloekombos, 1 m, 400 sources, 40 workers | 199.5 s (observed 115.5 s, prior 83.9 s) |
| research, observed + prior | same | ~194 s |
| research, one source, single core | Bloekombos, 1 m | 7.6 s |
| research, per block, single core, median | 220 small blocks, 0.5 m, every home a source | 146 s |

In the production run, the all-pairs pass is 111.7 s of the observed time and 79.8 s of the prior's.

## What this settles, and what it does not
**Settled:**
- **The field is worth something as a prior.**
  - Every generator arm beats clearance on both lenses: on the 220 small Cape Town blocks, and on
    the 36 large ones in both cities.
  - The looped arms beat clearance_looped the same way.
  - `NoDesire` ties clearance, so the gain comes from the field.
- **A field that finds lanes is not the same as one that generates well.** The contrast ranks worse
  and generates better. The oracle reproduces footpaths and does not improve permeability.
- **Repelled betweenness does not pile into pinch points,** the failure July rejected betweenness
  for.
- **The soft path field is dead,** for a structural reason (extensive path entropy), not a tuning
  one.

**Not settled:**
- **It does not reach the top of the frontier.**
  - cycle_native has the best Lens A. It leads the looped arms by 0.015-0.022, and the generator
    is better in only 8-11% of blocks.
  - greedy_arterial has the best Lens B. It is better in 83-94% of blocks, although it reaches the
    Lens A budget on only 16 of 36.
  - The generator is not dominated either: it is the frontier's middle point in 17 of 36 blocks.
- **Road length and runtime were not frontier axes.** clearance_looped lays half the road (3.3 km
  against the looped generators' 6.7 km), and clearance runs in seconds against the field's minutes. Neither is dominated
  on those axes, so both stay.
- **The small-block gains were measured at 0.5 m with every home as a source.** The presets carry
  the large-block setting (1 m, 400 sampled sources). Whether that setting reproduces the
  small-block gains is *untested*.
- **Sparse Cape Town is unscoreable on held-out Lens A** (n = 38, intervals include 0).
- **Large-block cores have only an imagery-derived ground truth** (Microsoft), and the core-picture
  result on it is one block.
