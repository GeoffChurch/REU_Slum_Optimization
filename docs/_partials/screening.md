<!-- Handwritten partial for docs/methodology/screening.md. scripts/gen_site_pages.py prepends the
     do-not-edit note and fills the SCREENTABLE marker (named, not spelled out in full here, for
     the same reason docs/_partials/intro.md's own note gives -- spelling it out would fill this
     note too) from examples/screen-bakeoff/screen_comparison.csv. Edit HERE, never
     docs/methodology/screening.md (it is generated and gitignored). This file is committed but
     excluded from the built site (see exclude_docs in mkdocs.yml).

     The "why n/P^2 was retired" section below links to the published results/bakeoff.md page.
     It originally linked to the screen-bakeoff example on GitHub instead (mirroring how
     gen_site_pages.py's _mc_section links out to method-comparison), because results/bakeoff.md
     did not exist yet and a link to it would have failed `mkdocs build --strict`. Repointed now
     that page ships.

     No typed numbers anywhere in this file: the floor and the precision/AUC figures come from
     SCREENTABLE, not from prose. -->

# Screening

## What screening is for

Reblocking a block for real — tessellating its interior into a parcel graph, peeling out how deep
each parcel sits, routing roads through it — is not cheap. A metro has far too many blocks to run
that on every one of them. Screening is the cheap first pass that decides which blocks are worth
it: one score per block, cheap enough to compute for a whole metro in a single sweep.

## The shipped screen

Every screen considered here is a formula over three quantities free on every block: parcel count
**n**, area **A**, and perimeter **P**. The default is `depth_density_proxy`: `√(nA)/P · n/A`, the
depth proxy times density. A block scores high only if it is both **deep** — its interior sits
many parcels back from a street — and **crowded**. Every input comes straight from the block's
building count, area, and perimeter: no Voronoi tessellation, no peel, so the whole metro scores
in one pass.

Every candidate screen below is scored against the City of Cape Town's own informal-structure
survey — AUC across the whole ranking, precision at the top of it — plus each one's shipped
absolute floor where it has one; a screen never shipped as a gate carries none:

<!-- SCREENTABLE -->

## Why `n/P²` was retired

`density_compactness` (`n/P²` — density times compactness) was the previous default. At an equal
pool size, `depth_density_proxy` beats it on both precision and recall, so the change costs
nothing and settles no trade-off — see the full [screen bake-off](../results/bakeoff.md)
for the comparison. Its historical selling point — that it needs no Voronoi tessellation and no
peel — was never a real differentiator: the metric that replaced it needs no Voronoi tessellation
and no peel either.

## The gate is absolute, not a percentile

Both screens select on an **absolute** floor, not a percentile cut. A percentile redefines the
population every time the corpus changes: the same percentile can select a very different share
of a corpus, or the same share can mean a very different crowding level, depending on what pool
it is computed against. An absolute floor applies the same bar however the corpus grows or
shrinks — a percentile is only useful as the instrument for calibrating that bar, never as the
selection rule itself.

## From block to region

A screened block is not reblocked on its own. `region_builder` grows it into a right-sized,
multi-block **region** first:

- `dense_cluster` — add the densest adjacent block, greedily, up to a size budget.
- `convex_hull` — fill in every candidate block inside the seed group's convex hull.
- `shape_standardizing` — grow while scoring the shape of the union at every step, so the outline
  is a design choice rather than a side effect of growth order.
- `identity` — no growth: the screened block is the region.

The reason is architectural. A road proposed for a single block stops dead at that block's
boundary, whether or not the settlement's fabric continues past it; growing a region first lets
the roads that follow run continuous across block boundaries instead. This is also the hinge in
the pipeline: everything up to and including the screen is cheap enough to sweep a whole city at
once, and everything from `region_builder` on is per-block and expensive.
