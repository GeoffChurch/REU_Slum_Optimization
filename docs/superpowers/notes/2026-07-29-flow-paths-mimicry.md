# Flow accumulation as a generative model: built, scored, refuted

**Date:** 2026-07-29
**Status:** measured. `src/reblock/methods/flow_paths.py`, scored by
`scripts/score_mimicry.py` (footpaths, 20 blocks) and `scripts/score_street_mimicry.py`
(streets, 3 regions).

Every reblocker here is a **drainage tree** — serve the worst-served parcel, repeat — and that
objective cannot produce a footpath network, because real footpaths do not try to serve everyone.
The generative story that fits is trail formation: least-effort routes, reinforced where trodden,
emerging where trips coincide. `FlowPathsReblocker` implements it — route every trip on the
substrate, accumulate edge traffic, reinforce and re-route, keep the busiest edges — with the
threshold on flow *volume*, so one field was supposed to give footpaths at a low cut and streets at
a high one.

## Footpath mimicry: no better than anything else

Length-matched to each block's real network (the reported metrics are IoU and Chamfer, whose
confound is how much line you draw):

| method | IoU@10m | recall (m) | road m | displacement |
|---|---|---|---|---|
| flow_paths (no reinforcement) | **0.303** | 13.7 | 232 | 0.168 |
| demand_greedy_uniform | 0.291 | 12.1 | 296 | 0.278 |
| flow_paths (gateway trips) | 0.288 | 13.8 | 182 | 0.147 |
| clearance / clearance_looped | 0.283 | 12.1 | 297 | 0.257 |
| flow_paths (full) | 0.273 | 12.4 | 202 | 0.172 |

Everything lands in 0.27–0.30, which is also where the earlier consensus work landed. **Flow
accumulation buys no agreement advantage.** Worse for the theory, the trail-formation feedback
*hurts*: 0.303 without reinforcement, 0.273 with. The mechanism that was supposed to be the whole
point is the part that costs.

## Street mimicry: clearance wins decisively

The first version of this test was mis-framed — scored per block, where interior streets barely
exist (IoU 0.000 at the median). Streets are a region-scale object. Redone properly: build regions,
**strip the inter-block streets out** (`region_block` hands them back as already-built, which would
be giving a method the answer), and ask each method to re-derive the interior network. Reference is
the real OSM street network clipped to the region.

| method | IoU@10m | IoU@20m | recall (m) | displacement |
|---|---|---|---|---|
| **clearance** | **0.184** | **0.399** | **42.7** | 0.149 |
| demand_greedy_uniform | 0.166 | 0.392 | 45.4 | 0.162 |
| flow_paths q=0.90 | 0.055 | 0.195 | 57.2 | 0.051 |
| flow_paths q=0.97 | 0.053 | 0.095 | 128.8 | 0.022 |
| flow_paths q=0.99 | 0.012 | 0.030 | 331.0 | 0.008 |

**Raising the flow cut makes it monotonically worse.** The hierarchy hypothesis — that streets are
the high-volume subset of the pedestrian field — is refuted on this evidence.

The likely reason is instructive: a region's streets follow the **cadastral structure**, the block
boundaries themselves. Clearance routes on a substrate built from parcel boundaries, so it traces
those lines almost by construction. Flow accumulation optimizes for *movement*, which cuts across
them. Streets are not where people walk most; they are where the subdivision put them.

## What the method does deliver

Sparsity and low displacement, exactly as designed: 202 m of road at 0.172 displacement against
clearance's 297 m at 0.257, and at q=0.90 on regions, 0.051 displacement against 0.149. It occupies
a genuinely different point on the cost curve. Whether that point is useful is a separate question
this note does not answer — it was built to mimic, and it does not mimic.

## Caveats

- Street test is n=3 regions. The effect is large and monotone in the quantile, which is more
  convincing than three points alone, but it is three regions.
- Footpath test is 20 blocks, one city.
- `max_sources=400` caps trips per block; a denser trip set might concentrate differently.
