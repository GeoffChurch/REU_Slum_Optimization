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
These are paired against clearance, whose medians are 0.763 on Lens A and 0.066 on Lens B. They
differ from `consensus-re-measured`'s 0.777 / 0.065 for the same recipients because that study
built them at the spacing-disc tier (its donor pool, `conf/donor_pool/capetown.yaml`, builds
`SpacingDiscs`), while this one rebuilt them at the footprint tier. Below 400 buildings every
home is a source, so the two field resolutions differ in nothing else. The percentages are the
share of blocks where the arm is better. Lens A n = 220, Lens B n = 217.

| arm - clearance | field | Lens A permeability | Lens B displacement |
|---|---|---|---|
| toward raw | 1 m (shipped) | +0.040 [+0.031, +0.052], 73% | -0.009 [-0.012, -0.006], 71% |
| toward contrast | 1 m (shipped) | +0.047 [+0.033, +0.060], 77% | -0.011 [-0.013, -0.007], 72% |
| toward raw | 0.5 m | +0.022 [+0.014, +0.031], 65% | -0.006 [-0.010, -0.003], 66% |
| toward contrast | 0.5 m | +0.032 [+0.023, +0.047], 71% | -0.007 [-0.010, -0.004], 65% |
| `NoDesire` (no field) | | +0.000 | +0.001 |
| toward the real OSM footpaths (oracle) | | +0.007, not significant | not recorded |

- **The gain is the field's.** `demand_greedy` with `NoDesire` ties clearance exactly.
- **The coarser field generates better.** Paired on the same blocks, raw at 1 m beats raw at
  0.5 m on both lenses (Lens A +0.013 [+0.003, +0.018], Lens B -0.002); for the contrast the two
  tie. Why is *untested*; fewer skeleton spurs at 1 m is one candidate.
- **At 1 m, contrast and raw tie** on these blocks.
- **The field guides to good roads, not to existing ones.** The oracle reproduces the footpaths
  (IoU at 10 m +0.140) but does not improve permeability.

### Held out
A strict replication is impossible, because the 220 are all the Cape Town blocks that pass the
study's selection rule. A relaxed rule selects sparser blocks, at 35-49 buildings/ha against 118:
- 60-300 buildings;
- at least 100 m of mapped interior footpath;
- an area cap.

Fields at the shipped 1 m; fresh Cape Town is 220 blocks, Nairobi 104. Lens A is taken where both
arms reach the 10% budget ("at budget"). The 0.5 m fields gave the same picture.

| sample | arm - clearance | Lens A at budget | Lens B |
|---|---|---|---|
| fresh Cape Town | raw | +0.023 [+0.001, +0.057], 65%, n = 40 | -0.006 [-0.009, -0.003], 64%, n = 144 |
| fresh Cape Town | contrast | +0.050 [+0.037, +0.078], 81%, n = 36 | -0.006 [-0.010, -0.002], 65%, n = 144 |
| Nairobi | raw | +0.073 [+0.045, +0.093], 86%, n = 37 | -0.006 [-0.011, -0.003], 70%, n = 99 |
| Nairobi | contrast | +0.056 [+0.033, +0.070], 87%, n = 39 | -0.007 [-0.011, -0.002], 68%, n = 99 |

- **Both lenses replicate in both cities.**
- **Fresh Cape Town's Lens A rests on few blocks:** only 24% of clearance's networks there reach
  10% displacement.
- **Contrast leads raw in fresh Cape Town** on both lenses (Lens B -0.003 [-0.004, -0.001], n = 140;
  Lens A at budget +0.035 [+0.014, +0.070], n = 34) and ties it in Nairobi.

### The settings after the field are a flat optimum
A one-at-a-time sweep from each shipped preset, on the 220 at 1 m, paired against the preset. 22
arms. None beat the shipped raw preset on both lenses with an interval excluding 0.

| knob | values tried | result against the preset |
|---|---|---|
| egress weight (egress share vs all-pairs share) | 0.25, 0.75, 1.0 (shipped 0.5) | ties; egress only is no better (Lens A -0.005, Lens B +0.002, intervals touch 0) |
| partial contrast, raw / (prior + offset)^alpha | 0.25, 0.5, 1.0 | ties; alpha = 1 borderline (Lens A +0.007 [+0.000, +0.012]) |
| ridge quantiles | sparse .90-.98, dense .60-.98 | ties from raw; sparse is worse from contrast (Lens A -0.011, Lens B +0.002) |
| gamma | 0.5, 2 | 0.5 worse (Lens A -0.007 to -0.009); 2 borderline (Lens A +0.002 [+0.000, +0.005]) |
| eps | 0.03, 0.3 | ties |
| buffer_m | 2, 5 | ties; 5 worse from contrast (Lens A -0.004) |

The two borderline settings (alpha = 1, gamma = 2) tie the shipped raw preset on both held-out
samples, every interval including 0. The all-pairs half of the field is not diluting it, which
was the seductive reason to expect egress-only to win.

### The field's own settings are flat too
One at a time from the shipped point (r0 2 m, lambda 50), new fields at 1 m, on the 220, paired
against the shipped raw preset (Lens A n = 220, Lens B n = 217):

| setting | Lens A | Lens B |
|---|---|---|
| lambda 10 | +0.000 [+0.000, +0.000] | +0.000, a tie |
| lambda 200 | +0.000 [-0.003, +0.000] | +0.000, a tie |
| r0 1 m, raw | -0.006 [-0.013, -0.003] | +0.002 [+0.001, +0.005] |
| r0 1 m, contrast | -0.012 [-0.018, -0.006] | +0.003 |
| r0 3.5 m (the 7 m road's half-width) | +0.000 [-0.001, +0.007] | a tie |

The bend penalty does not matter for generation anywhere from 10 to 200; the ranking study chose
50 for pictures. Weaker repulsion is worse on both lenses. Lens A rises with r0 up to 2 m and is
flat to 3.5 m; beyond 3.5 m is *untested*.

### Parallel duplication
The field is computed once, on the roadless block, so a builder could chase a ridge an earlier road
already serves. Measured on the 220: a road counts as duplicated when another road of the same
network lies within 15 m of it but is not reachable within 30 m along the network.
- **Whole networks:** the generators duplicate 0.12-0.14 of their length against clearance's 0.04
  (+0.065 and +0.086 [+0.048, +0.097], higher in 77-79% of blocks).
- **Where Lens A scores** (cut to 10% displacement): the excess is +7 m per block [+4, +11] of about
  190 m, no more than cycle_native's own excess (+12 m).
- **It costs nothing measured:** excess duplication does not track a smaller margin over clearance
  (every correlation within +-0.14).

So recomputing the field after the first roads has little to fix on small blocks. The 36 large
blocks are unchecked.

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

## Routing cycle_native along the field
**The mechanism.** `DesireWarpedSubstrate` (`reblock.methods.desire_substrate`) shortens the
substrate's edges along the ridges with demand_greedy's attraction, `length / (eps + demand)^gamma`
(buffer 3 m, eps 0.1). cycle_native routes the two legs of each candidate loop in that metric and
still chooses among the candidates by permeability gain per unit displacement. Its deepest-first
shortlist ranks parcels by the warped distance too.

### 36 large blocks
Paired against plain cycle_native (medians 0.896 / 0.025); Lens A at budget, n = 35-36:

| field, gamma | Lens A | Lens B | road at the Lens A cut | cycle_native's own time |
|---|---|---|---|---|
| raw, 0.5 | +0.012 [+0.008, +0.016], 83% | -0.006 [-0.008, -0.003], 78% | x1.20 | x1.16 |
| contrast, 0.5 | +0.012 [+0.007, +0.018], 92% | -0.005 [-0.008, -0.003], 83% | x1.22 | x1.18 |
| raw, 1 | +0.005 [+0.002, +0.012], 69% | -0.005 [-0.007, -0.004], 81% | x1.31 | x1.65 |
| contrast, 1 | +0.008 [+0.002, +0.013], 67% | -0.004 [-0.006, -0.002], 75% | x1.27 | x1.62 |
| raw, 0.25 | +0.008 [+0.005, +0.014] | -0.005 [-0.006, -0.003] | x1.22 | x1.51 |
| contrast, 0.25 | +0.013 [+0.009, +0.018] | -0.005 [-0.007, -0.003] | x1.17 | x1.48 |
| contrast, 2 | -0.017 [-0.030, -0.007] | -0.001, interval includes 0 | x1.31 | x1.80 |

- **The attraction wants to be gentle.** gamma 0.5 against 1: Lens A +0.004 (borderline), Lens B a
  tie, and less road and time. 0.25 ties 1 and is slower than 0.5; 2 is worse.
- **By city:** in Cape Town (n = 24) every gamma-0.5 arm wins on both lenses. In Nairobi (n = 12)
  the contrast at 0.5 wins on both (Lens A +0.017 [+0.005, +0.027], 12 of 12); raw's intervals touch 0.
- **Against greedy_arterial:** a tie on Lens A (n = 16); it still trails on Lens B by +0.005
  (greedy_arterial better in about 70% of blocks).
- **Against the looped generators:** raw at 0.5 against looped raw, Lens A +0.026 and Lens B -0.003;
  contrast at 0.5 against looped contrast, +0.041 and -0.005.
- **The frontier**, over the study lineup plus every desire-routed arm: greedy_arterial 23 of 36,
  contrast at 0.5 14, raw at 0.5 13, the looped generators 2 each, plain cycle_native 1. Some
  desire-routed arm is on it in 33 of 36 (Cape Town 21 of 24, Nairobi 12 of 12).

### Small blocks
Paired against plain cycle_native. Every desire arm reaches the Lens A budget on 99-100% of blocks.

| sample | field, gamma | Lens A | Lens B |
|---|---|---|---|
| the 220 | contrast, 0.5 | +0.010 [+0.004, +0.015] | a tie |
| the 220 | raw, 0.5 | +0.009 [+0.003, +0.015] | a tie |
| the 220 | raw, 1 | a tie | +0.005 [+0.002, +0.009], worse |
| fresh Cape Town (220) | contrast, 0.5 | +0.011 [+0.002, +0.015] | a tie |
| fresh Cape Town | raw, 0.5 | a tie | a tie |
| fresh Cape Town | raw, 1 | -0.008 [-0.019, -0.003], worse | +0.004 [+0.003, +0.007], worse |
| Nairobi (104) | contrast, 0.5 | +0.009 [+0.000, +0.016], borderline | a tie |
| Nairobi | raw, 0.5 | a tie | a tie |

- **The contrast at 0.5 is never worse than plain cycle_native** on either lens in any sample.
- **Raw at gamma 1 is a regression on small blocks,** which is one reason gamma 1 does not ship.
- **Plain cycle_native already beats both tree generators on both lenses here** (the 220: Lens A
  +0.044 and +0.043, Lens B -0.011 and -0.014), laying about 1.5 times the road at the Lens A cut
  (288 m against 190-193 m).

### The field inside greedy_arterial: no measurable lift
greedy_arterial does not route over a substrate: it snaps straight chords (between network anchors,
or to the deepest parcels) onto the parcel-boundary graph, each edge costing length + lam x its
distance from the chord. The test warped those edge lengths by the same attraction (gamma 0.5,
buffer 3, eps 0.1), leaving the chord term alone; the warped lengths are read only as the snap's
Dijkstra weight. 36 large blocks, paired against plain greedy_arterial:

| arm | Lens A (both at budget) | Lens B | blocks reaching the Lens A budget |
|---|---|---|---|
| contrast, lam 2 | +0.011, n = 11, interval includes 0 | -0.000, a tie | 5 fewer |
| raw, lam 2 | +0.009, n = 13, interval includes 0 | -0.001, a tie | 3 fewer |
| contrast, lam 1 | +0.007, n = 8, interval includes 0 | -0.002 [-0.004, -0.001], 81% | 8 fewer |

- **The one significant cell is confounded:** it also halves the chord pull, and plain greedy_arterial
  at lam 1 was not run. It holds in Cape Town only.
- **Every warped arm costs** 4-7% more road at the Lens A cut and 14-20% more time.
- **The Lens B lead does not change hands:** the arterial family still leads the desire-routed
  cycle_native by about 0.005.

The two other ways in, which change the candidate set instead of the snap, are null too (2026-09-26,
36 large blocks, paired against plain greedy_arterial; the harness reproduced plain greedy_arterial
exactly on its check blocks):

| candidate set added | Lens A | Lens B | blocks reaching the Lens A budget | road at the Lens A cut |
|---|---|---|---|---|
| anchors where the contrast field's top ridges meet the network | -0.001 [-0.006, +0.015] | -0.0002, a tie | 15 vs 16 | +3.7% |
| spurs to the top-ridge points in the deepest parcels | -0.005 [-0.021, +0.009] | -0.0003, a tie | 12 vs 16 | +1.4% |
| both | -0.004 [-0.021, +0.013] | -0.0003, a tie | 11 vs 16 | +3.2% |

So the field does not lift greedy_arterial through its snap, its anchors or its spur targets.

### r0 for the desire-routed cycle_native: the shipped 2 m stands
The contrast preset with the field's r0 at 3.5 and 5 m, on the 220: 3.5 m ties on both lenses; 5 m
ties Lens A and wins Lens B (-0.003 [-0.006, -0.001]), but ties on both lenses held out (fresh Cape
Town n = 220, Nairobi n = 104). Not run on the large blocks.

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

**Shipped: `reblock.methods.desire_substrate.DesireWarpedSubstrate`,** a substrate Strategy that
wraps any substrate and shortens its edges along a desire field. Not for `demand_greedy`, which
would apply the attraction twice.

**Shipped: six presets,** each a measured operating point. All six are also in `compare_config`'s
`all_methods`, and none is on a published example lineup.

| preset | method | desire | why it is on the frontier |
|---|---|---|---|
| `cycle_native_betweenness_contrast` | cycle_native on the warped substrate, gamma 0.5 | contrast | beats plain cycle_native on both lenses (large), Lens A with B tied (small) |
| `cycle_native_betweenness` | cycle_native on the warped substrate, gamma 0.5 | raw | the same on large blocks; no prior pass, so cheaper |
| `betweenness_tree` | `DemandGreedyReblocker` | raw | road: about 190 m at the Lens A cut against cycle_native's 288 m |
| `betweenness_tree_contrast` | `DemandGreedyReblocker` | contrast | road, as above; leads raw on fresh Cape Town |
| `betweenness_looped` | `LoopClosureRefiner` around the tree | raw | road: 6.7 km against about 9 km for the warped cycle_native (large) |
| `betweenness_looped_contrast` | `LoopClosureRefiner` around the tree | contrast | road, as above |

gamma 1 is not shipped: gamma 0.5 beats it on Lens A, ties it on Lens B, and costs less road and
time; raw at gamma 1 is also worse than plain cycle_native on small blocks.

The two cycle_native presets, built from `compare_config`, reproduce the research arms exactly:
every lens value equal on 8 of the 220 blocks (51 to 733 buildings), both fields.

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
  - On the 220 small Cape Town blocks, both tree arms (raw and contrast) beat clearance on both
    lenses, at the shipped 1 m and at 0.5 m, and replicate on held-out Cape Town and Nairobi
    blocks. The looped arms were not run there.
  - On the 36 large blocks, at the shipped 1 m and 400 sampled sources, all four generator arms
    beat clearance on both lenses, in both cities.
  - On the same 36, the looped arms beat clearance_looped on both lenses, in both cities.
  - On the 220, `NoDesire` ties clearance, so the gain comes from the field.
- **A field that finds lanes is not the same as one that generates well.** The contrast ranks
  worse, yet generates as well as raw on small blocks (better on fresh Cape Town); on the 36 large
  ones raw leads it. The oracle reproduces footpaths and does not improve permeability.
- **The builder and post-processing settings are at a flat optimum** (the sweep above), and so
  are the field's own (lambda from 10 to 200; r0 1 m is worse, 3.5 m ties).
- **The field lifts cycle_native, the Lens A leader,** routed along it at gamma 0.5: both lenses on
  36 large blocks, Lens A with Lens B tied on small ones. It moves the top of the frontier: some
  desire-routed arm is on it in 33 of 36 large blocks.
- **Repelled betweenness does not pile into pinch points,** the failure July rejected betweenness
  for.
- **The soft path field is dead,** for a structural reason (extensive path entropy), not a tuning
  one.

**Not settled:**
- **greedy_arterial still has the best Lens B,** by +0.005 over the desire-routed cycle_native.
  Warping its snapping graph by the field does not move it (above); the candidate-set entry points
  are *untested*.
- **Road length and runtime were not frontier axes.** The tree and looped generators stay on the
  frontier by road alone; clearance and clearance_looped by road and runtime.
- **Duplication on large blocks** is unmeasured; it bears only on the tree generators.
- **Large-block cores have only an imagery-derived ground truth** (Microsoft), and the core-picture
  result on it is one block.
