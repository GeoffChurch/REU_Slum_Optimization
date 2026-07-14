# Screen cheap-gate: building density → depth proxy √(n·A)/P

**Date:** 2026-07-14 · **Status:** adopted (migrated `DenseCompactScreen`'s cheap gate) ·
**Question:** the cheap pre-filter that decides which blocks reach the expensive fine access-depth
pass gated on building *density* (`n/A`). Does that actually track deep nesting — and if not, what
cheap signal does?

**Verdict:** replace it. Building density is almost uncorrelated with real access depth
(Spearman ρ=0.15 on max depth). The closed-form **`depth_proxy = √(n·A)/P`** (n=building_count,
A=block_area_m2, P=block perimeter in metres) ranks true depth **~5× better** (ρ=0.76 max / 0.87
mean) from the same free columns, and it is the free incarnation of two more-expensive ideas we also
tested. Migrated the gate to it (default `depth_proxy_min=1.5`); no dual path kept.

## Why density fails and the proxy works

Deep nesting is **frontage-starvation**: parcels far from any street edge. The perimeter `P` *is* the
frontage; a block with many parcels per metre of frontage (and enough linear extent to stack rings)
is deep. Density `n/A` sees crowding, not frontage — a big block with a slum in it has its density
*diluted* by area, so density mis-ranks exactly the blocks we hunt.

The proxy is the geometry of ring-peeling: max ring depth ≈ inradius / parcel-width; inradius ≈ 2A/P
(hydraulic radius), parcel-width ≈ √(A/n), so depth ≈ 2·√(n·A)/P (the constant drops out of a
threshold). Three free columns, no parcel geometry.

## Measurements (600-block random Cape Town sample, ground truth = fine-pass access depth)

Spearman ρ vs true depth (cheap predictors, all from free columns unless noted):

| predictor | ρ(max) | ρ(mean) | notes |
|---|---|---|---|
| `n/A` (old gate) | 0.15 | 0.26 | crowding, not nesting |
| `n/P` | 0.49 | 0.62 | parcels per metre of frontage |
| **`√(n·A)/P`** | **0.76** | **0.87** | **adopted** |
| `n/P²` | 0.02 | 0.18 | density × compactness — useless |

Top-25 recall of the truly-deepest blocks: `n/A` 8%, `n/P` 28%, **`√(n·A)/P` 48–56%**.

## Alternatives tested and rejected (with data, not hand-waving)

- **Convex hull of the parcels** instead of the block polygon (to ignore empty land): converges here.
  Median hull-area / block-area = **1.03** — kblock polygons are already street-bounded and drawn
  tight around the fabric, so hull ≈ block. `ch_depth_est` ρ=0.75/0.74 ≈ the block version, actually
  slightly *worse* on mean depth because the hull smooths away boundary concavities that are real
  frontage. Would matter for a *coarse* administrative block source; not for kblock.
- **Intra-block cluster detection** (multi-slum blocks along a highway): same reason — fill ≈ 1 says
  blocks are morphologically coherent; a street-bounded block rarely spans two disjoint settlements.
  Not worth the DBSCAN cost on this data.
- **Per-parcel Euclidean distance-to-egress**, summarized per block: the max-distance version,
  normalized to parcel-widths (`max_dist · √(n/A)`), scores ρ=0.78/0.82 — statistically tied with
  the column proxy, because `max_dist ≈ inradius ≈ 2A/P`, so it *is* `2·√(n·A)/P` measured from real
  points. Needs a per-parcel geometry pass for no ranking gain. The scale-invariant ratios
  (`p99/p1`, coeff-of-variation) are the *weakest* (ρ=0.44, 0.27): depth genuinely grows with block
  size, and a dimensionless ratio discards exactly that signal.

Conclusion: `√(n·A)/P` is both the best cheap predictor and the cheapest (free-column arithmetic).
`eu_max_norm` is the only lever for the last few points of recall, at the cost of a per-parcel pass —
not worth it.

## Threshold

`depth_proxy` is a good *ranker* but an uncalibrated *threshold*, so the gate stays PERMISSIVE (the
fine `mean_depth_min` / `max_depth_min` do the real discrimination). Metro distribution (27,643 blocks
w/ ≥10 buildings): p50=1.53, p80=1.97, p90=2.27, p95=2.60. Recall-vs-load of `depth_proxy_min`:

| threshold | recall of confirmed-deep set | metro kept (fine-pass load) |
|---|---|---|
| 1.25 | 93.5% | 72% |
| **1.5** (default) | 74% | 52% |
| 1.75 | 52% | 33% |
| 2.0 | 33% | 19% |

Default **1.5**: retains ~74% of the blocks the old gate confirmed deep *and* admits ~9,500 deep
blocks density excluded. It is a one-line conf change with no golden impact (synthetic tests use
explicit per-fixture thresholds; the flagship sample block sits at proxy ~3.75, the full-metro
deepest seed at ~15), so easy to re-tune.

## Files

`screen/dense_compact.py` (`_depth_proxy`, `_cheap_survivors`, `DenseCompactScreen.depth_proxy_min`),
`derivations.py` (`ScreenSelectionInput.depth_proxy_min` + identity), `conf/screen/dense_compact.yaml`,
`emit.py` (`region_map` now colours `screen.png`/`region.png` by the proxy). Prototypes:
`scratchpad/{perim_density,ch_density,euclid_egress,proxy_threshold,proxy_recall}*.py`.
