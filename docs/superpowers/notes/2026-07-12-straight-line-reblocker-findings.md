# Straight-line reblocker — spike findings (positive; ready to productionize)

**Status:** validated spike (`scratchpad/straight_line_*.py`); recommend productionizing as a Method.
Companion to the [resistance-greedy findings](2026-07-12-resistance-greedy-spike-findings.md)
(negative). Origin: the user wanted "a couple of long roads, not a forest of stubs," and asked
whether the metrics can detect a straight-vs-wiggly tradeoff (they can — cost=length + directness).

## The method
A topology-style greedy with straight lines instead of graph-following shortest paths:
1. Pick the **deepest** parcel (max access-depth).
2. Draw a **straight line** from its rep to the **nearest point on the connected road+street network**.
3. Add it; repeat until every parcel is within the target depth (e.g. ≤ 2).

Two endpoint variants were tested: **A = geometric-nearest** (winner) and **B = topology's
graph-nearest connected parcel, straightened**. A beats B — straightening away topology's endpoint
choice, the geometric nearest is simply the more length-efficient target (B routes to a
sometimes-farther parcel). (B might win on blocks with real barriers; untested.)

## Speed (refinement 1: incremental access-depth)
A road only LOWERS depths, so instead of recomputing `parcel_access_layers` from scratch each step,
relax a multi-source BFS from the parcels the new road touches (STRtree `dwithin` query). Verified:
a full recompute on the output confirms the target depth is hit. **0.17 s on the 2017-parcel block —
233× over the naive recompute, and faster than dijkstra (0.7 s).** (Small blocks: byte-identical to
naive. Big blocks: ~94% identical; the rest is tie-breaking, immaterial to quality — grades
slightly better. Deterministic tie-breaking is a productionization to-do.)

## Grading vs the incumbents (AUC @ equal budget)

103-parcel block, cost=length:
| lens | straight-line | topology | dijkstra |
|---|---|---|---|
| access | **0.447** | 0.437 | 0.414 |
| resistance | 0.098 | 0.161 | **0.184** |
| directness | **0.065** | 0.027 | 0.006 |
| road length | **200 m** | 360 m | 591 m |

2017-parcel block, cost=length: access **0.780** vs dijkstra 0.711; directness **0.007** vs 0.002;
resistance 0.138 vs **0.227**. Wins access + directness at ~⅓ the road.

Displacement (refinement 2, cost=displacement, 103-parcel): straight-line displaces **31** buildings
(most) vs dijkstra 21, topology 23 — access@disp 0.368 vs dijkstra **0.735**. The straightness costs
buildings.

## Verdict
The straight-line is a genuine, distinct **aspirational** method — the minimal-length, most-direct
"few long roads" plan. It **wins access + directness per metre** at ⅓ the road, is **fast + simple**,
and **scales** where topology fails. Tradeoffs (both inherent to being aspirational, both quantified):
**no redundancy** → loses the resistance lens; **crosses parcels** → loses the displacement lens. Its
buildable counterpart (snap-to-frontage) converges to dijkstra/topology, so there's nothing separate
to build there — **dijkstra is the buildable version**.

Method landscape after this: **straight-line** = aspirational minimal few-long-roads (best
access/directness per metre); **dijkstra** = buildable coverage (best resistance/redundancy, low
displacement); **arterial** = strategic through-roads (directness objective, slow); **topology** =
middle, too slow to keep as-is.

**Recommendation: productionize as a `Method` (spec → plan → SDD)** — a straight-line / aspirational-
access reblocker that slots into compare/render alongside dijkstra/topology/arterial. Productionization
to-dos: deterministic tie-breaking; a config for the depth target; decide whether to offer an optional
loop-closing post-pass for redundancy.
