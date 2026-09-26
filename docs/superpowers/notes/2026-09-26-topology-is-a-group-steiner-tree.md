# Topology is a group Steiner tree, and an exact solver dominates it on its own objective

**Date:** 2026-09-25 to 2026-09-26 (living note: sections are added as studies finish)
**Status:** measured, scratch code only; nothing here is shipped.
Samples: the 220 consensus recipients (Cape Town, footprints) and the 36 large blocks of the
betweenness frontier study (24 Cape Town, 12 Nairobi, >= 1,000 buildings). Lenses as everywhere:
Lens A = permeability at 10% homes displaced, Lens B = displacement to P* = 0.60, both through
`compare.lens_prefixes`; paired medians with 95% bootstrap intervals.

## What topology optimizes
Read from the SFI working-paper version (Brelsford & Bettencourt 2015, SFI WP 2015-10-037; published
with Martin in EPB 2019, cited in `docs/background.md`):
- **Topological stage:** the least total length of new access (along parcel boundaries) that gives
  every parcel a corner on the road network ("universal access", peel depth 1). Solved by a greedy
  that repeatedly connects the deepest parcels; a "statistical" variant samples paths with
  probability ~ length^-alpha and keeps the shortest of many runs. The paper finds alpha 10-100 best.
- **Geometric stage (not in our `TopologyMethod`):** greedily add "bisecting" through-paths that most
  reduce the mean parcel-to-parcel travel distance.

Our `topology` preset runs one sample at alpha = 2.

## The same objective, solved exactly
On topology's own graph (`to_parcel_graph` + street marking), contract the street to a root and let
each interior parcel's corners be a group: the objective is a **group Steiner tree** (NP-hard, but
well studied). Solved with:
- a directed-cut LP (terminal per group, nested cuts, purging of slack cuts; HiGHS via scipy),
  which is nearly tight (0.01-0.2% below the optimum);
- the LP-guided shortest-path heuristic, then a MILP closed on integer cuts;
- for scale, round-robin dual ascent (bound within 3-5% of optimal in milliseconds) plus
  shortest-path, MST and vertex-insertion heuristics -- the "fast" solver, 2-4% above optimal on the
  weighted objective, median 3 s per small block and 159 s per large block.

k-hop access generalizes it: "every parcel within k parcels of one that fronts a road" is the same
problem with bigger groups, built on `ParcelAdjacency` so it means exactly what the peel scores
(an earlier version built the hops on the face graph and was wrong at k >= 1).

## Topology against the optimum (220 blocks)
- The solver proves the optimum on 214 of 219 feasible blocks (the rest: certified gaps of 1.4-36%,
  the large ones from an unconverged bound).
- Topology never matches it. New road over the optimum, median / p90 / max:

| topology | excess |
|---|---|
| shipped (alpha 2, one run) | +36% / +46% / +56% |
| best of 10 at alpha 10 | +14% / +18% / +27% |
| best of 10 at alpha 100 | +14% / +20% / +27% |
| best of all 30 runs | +13% / +16% / +27% |

- Time: topology's 30 runs take a median 890 s per block, the solver 21 s. With a 15-minute cap per
  run, topology times out on 36 blocks (all >= 219 buildings) and never finishes on 17; one uncapped
  alpha-10 run on the 759-parcel block took 11.8 h. On `ZAF.9.3.1_1_40972` (the method-comparison
  block) only the shipped run finished.

## The frontier at k = 0, 1, 2 (road, displacement)
With edge cost length + lambda x homes under the edge's 7 m corridor (lambda in metres per home),
sweeping lambda in {0, 0.1, 1, 3, 10, 30, 100, 300} traces least-road to least-displacement:
- **k = 0:** the solver family dominates topology on every block (one block needed lambda = 0.1) and
  dominates greedy_arterial at universal access.
- **k = 1:** topology survives on 3 of 201 blocks, all ties (two already at k = 1 with no road, one
  exactly equal to the optimum).
- **k = 2:** its apparent survivals are 54 trivial blocks (already at k = 2) and 7 ties within 1 m and
  0.5 points of displacement; none real.
- **Lens A's road axis is where topology survives:** at 10% displaced it is the cheap-road point
  (permeability 0.73 at 107 m) on 57 of 97 blocks; the least-road solver tree reaches only 0.67 there.

## A metric blind spot: k = 0 trees pave the yards
The displacement-weighted k = 0 trees (lambda 30-100) beat the whole lineup on both lenses on the 220
(vs desire-routed cycle_native: Lens A +0.042, Lens B -0.023; vs greedy_arterial +0.020 / -0.010),
and on Lens B on the 36 large blocks (vs desire-routed cycle_native -0.009, greedy_arterial -0.004;
Lens A wins Cape Town, loses Nairobi -0.026). Fast (heuristic) trees tie the optimal ones on both
lenses, so this is the trees' structure, not their optimality.

**Rendered, they are not road networks.** A 7 m branch reaches nearly every door, threading the gaps
between buildings: on the two largest Cape Town blocks the corridors cover 39-40% of the block and
45-47% of its open ground, with ~1 dead end per 5 parcels (5810: 32 km, 1,368 dead ends). They
displace few homes because the lenses charge **homes**, not land, metres or dead ends -- the same
class of artifact recorded for `resistance_lp` in `conf/example/explore.yaml`, connected rather than
fragmented. At k = 1 the trees read as a lane network (5810: 16 km, 2.4% of homes, 25% of the open
ground, 412 dead ends); at k = 2 as a sparse spine (10 km, 1.0%, 16%).

Seductive and wrong: "the displacement-weighted access tree is the best reblocker" -- it wins only
because the lenses cannot see land take.

## Loops by requirement: two disjoint exits (k = 1)
Survivable variant: every interior k = 1 group must reach the street by two edge-disjoint routes
through different street sectors (150 m of street each). Heuristic: Suurballe per group, farthest
first, built edges free; reverse-delete pruning checked by max-flow.
- **220 small blocks, lambda 30:** beats cycle_native and the desire-routed one on both lenses (Lens A
  +0.020 / +0.010 on the 78 blocks reaching the budget; Lens B -0.018 / -0.019), ties greedy_arterial
  and resistance_lp on Lens A and beats them on Lens B (-0.008). Whole network: 341 m (median) against
  680-800 m for the lineup, permeability 0.86 against >= 0.95.
- **36 large blocks:** loses Lens A to the desire-routed cycle_native (-0.036 [-0.066, -0.020]) and
  cycle_native (-0.026); Lens B ties the desire-routed one, beats plain cycle_native (-0.007); worse
  than greedy_arterial on both lenses in Nairobi.
- **Loophole:** a group is a parcel plus its neighbours, so two spurs arriving at different
  neighbours from opposite sides satisfy it -- the rendered networks still carry many dead ends.

### Closing the loophole, and a softer incentive (k = 1, lambda 30)
- **Same-corner:** both routes must end at one corner of the group (its 4 nearest tried), so every
  served parcel is on a real loop: 0 interior dead ends. Reverse-delete must re-check only the groups
  whose recorded routes use the removed edge; checking the whole built component ran for hours on 5810.
- **Redundancy prize p:** every group gets one route; the second is built only if it costs <= p more
  (metres + lambda x homes). p = 0 is a tree, large p approaches same-corner. Loops then appear where
  they are cheap -- near the block edge; from deep inside, a second route to another street sector
  costs more than 200 m, so the interior stays a tree (5810, 300 m central window: 72 dead ends at
  p = 200 against 79 for the k = 1 tree and 0 for same-corner).

36 large blocks, paired:

| variant | vs desire-routed cycle_native, Lens A / Lens B | vs greedy_arterial, Lens A / Lens B | road at the Lens A cut vs desire-routed |
|---|---|---|---|
| two exits, group | -0.036 [-0.065, -0.020] / -0.002 (tie) | -0.014 / +0.002 | 1.42x |
| same-corner | +0.009 [-0.001, +0.032] / +0.000 (tie) | +0.016 / +0.003 | 1.33x |
| prize 200 | +0.014 [+0.002, +0.028] / -0.002 (tie) | +0.019 / +0.001 (tie) | 1.43x |
| prize 50, 100 | worse; few blocks reach 10% displaced | | |

220 small blocks: same-corner and prize 200 tie the desire-routed cycle_native on Lens A and beat it
on Lens B (-0.010); they trail greedy_arterial and resistance_lp on Lens A by 0.006-0.010, tie Lens B.

Land on 5810: k = 1 tree 16.2 km, 2.4% of homes, 25% of the open ground; prize 200 23.2 km, 7.9%,
33%; same-corner 26.7 km, 10.5%, 37% (the k = 0 tree: 47%). The land caveat above applies with less
force but still applies: these are a third of the open ground at 7 m.

## The betweenness field as greedy_arterial entry points: null
Recorded in [the betweenness note](2026-09-24-roadless-fields-and-the-betweenness-generator.md).

## Repo bug found on the way
`reblock.methods.topology._streets_local_geometry` keeps only `LineString` rows, so a block whose
street is a `MultiLineString` (an inner ring) marks no road at all. One of the 220 blocks and several
large blocks have such streets. Not fixed.

## Open
- Let a second route close on existing lanes (not only a different street sector), so the prize can
  loop the interior; run the loop variants at k = 2; a travel-distance incentive needs a metric decision.
- Decide productionizing the solver (a least-road k-access preset family) and topology's fate: it is
  dominated or tied at k = 0, 1, 2 but keeps Lens A's road-axis point.
- Fix the `MultiLineString` street bug.
