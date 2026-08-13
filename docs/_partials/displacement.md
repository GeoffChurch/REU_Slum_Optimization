<!-- Handwritten partial for docs/methodology/displacement.md. scripts/gen_site_pages.py prepends
     the do-not-edit note and writes this straight through -- there are no markers to fill on this
     page (this page adds no markers and no producers). Edit HERE, never
     docs/methodology/displacement.md (it is generated and gitignored). This file is committed but
     excluded from the built site (see exclude_docs in mkdocs.yml).

     This page describes a MODEL, so its only quantities are symbolic parameters (rᵢ, dᵢ, cᵢ), never
     measured values -- no typed number belongs in this file. -->

# Displacement

> **Displacement** is the expected number of **buildings** a road set displaces — not parcels.
> Each building is a disk of radius `rᵢ`, half its nearest-neighbour distance. Its contribution is
> the probability the road corridor grazes it under a uniform size prior,
> `cᵢ = max(0, 1 − dᵢ/rᵢ)`, where `dᵢ` is the distance from the building to the corridor.
> Displacement is `Σcᵢ`; the reported fraction divides by the number of buildings.

This is the cost half of the project's one graded tradeoff — the benefit half is
[permeability](permeability.md).

## Width is per-road

Each road is buffered by its *own* `width_m` / 2 before anything is charged against it — there is no
width shared by every road in the set, and no global corridor width anywhere in the model. A narrow
lane therefore costs less corridor, and so displaces fewer buildings, than a wide street run along
the same line.

## Overlap is free, by construction

What a road set is charged against is the *union* of every individual road's own buffer. Two roads
that run side by side — opposing one-way lanes flanking the same median, say — fall inside one
shared corridor and are charged for it once; pulling them apart widens the union and costs more. No
separate rule against double-charging overlap is needed, and none exists — it falls straight out of
buffering each road on its own and only then taking the union, before any distance is measured.

## Parcels are not buildings

`Block.parcels` and `Block.building_points` are two distinct fields on the same block. A parcel is a
cell of the tessellated block interior — land, not a structure. A building is the structure actually
standing on some of that land. Displacement is counted over buildings, because a building is the
thing a road corridor threatens; a parcel with no building standing on it costs nothing to cross.

## Gap-hugging is free

Because `cᵢ` clips to exactly zero once a building's distance to the corridor reaches its own radius
`rᵢ`, a road that hugs the tightest gap it can find between two buildings pays nothing at all for how
close it runs. Displacement only ever prices a corridor that actually reaches into a building's own
disk — there is no credit for merely coming close.
