<!-- Handwritten partial for docs/methodology/displacement.md. scripts/gen_site_pages.py prepends
     the do-not-edit note and fills the markers below. Edit HERE, never
     docs/methodology/displacement.md (it is generated and gitignored). This file is committed but
     excluded from the built site (see exclude_docs in mkdocs.yml).

     Markers: DISPFIELD (the interactive field figure -- fallback PNG, mount point, and a caption
     whose every number is read out of examples/displacement-field/field.json).

     This page describes a MODEL, so the prose's only quantities are symbolic parameters (Aᵢ, Kᵢ,
     cᵢ). No measured number is typed HERE: the figure's numbers arrive through DISPFIELD, off the
     baked artifact. -->

# Displacement

> **Displacement** is how many **buildings** a road set takes — not parcels — counted by the share
> of each one it takes. Building `i` contributes `cᵢ = Kᵢ / Aᵢ`: the area `Kᵢ` of its footprint
> the road corridor covers, over its whole area `Aᵢ`. Displacement is `Σcᵢ`; the reported fraction
> divides by the number of buildings.
>
> Where the real footprint outlines are available, `Aᵢ` is the footprint. Where only a building's
> location is known, it is a disk of radius half its nearest-neighbour distance — an estimate of
> the footprint, measured by the same rule.

This is the cost half of the project's one graded tradeoff — the benefit half is
[permeability](permeability.md). The figure below prices the cost and nothing else: it says what a
road set displaces, never whether that road set is worth building. A road it reports as cheap is not
thereby a good road — what the road buys is permeability, and nothing here computes it.

<!-- DISPFIELD -->

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

`Block.parcels` and `Block.building_geometries` are two distinct fields on the same block. A parcel is a
cell of the tessellated block interior — land, not a structure — and that tessellation is the
**Voronoi diagram of the building points**, so a parcel is normally one building's own share of the
block: one cell, one building, as on the block drawn above. Normally, not always — clipping a cell
against a ragged block edge can split it into separate lobes, and each lobe becomes a parcel in its
own right. A lobe that ends up holding no building is charged nothing, because there is nothing
inside it to charge.

What separates the two is therefore not whether a parcel holds a building, but what the charge is
measured against. Displacement is charged **per building, against that building's own footprint**,
and never per parcel against parcel *area*. A road crossing one large parcel out at the sparse
edge of a block is charged for the part of the one building standing in it that the road covers,
not for how much land it takes; the same road through the packed core is charged, building by
building, the share `cᵢ` of each footprint it covers.

## Gap-hugging is free

Because `cᵢ` is exactly zero for any building the corridor does not cover, a road that hugs the
tightest gap it can find between two buildings pays nothing at all for how close it runs.
Displacement only ever prices the part of a building a corridor actually covers — there is no charge
for merely coming close, and no credit either.

## A share, not a verdict

A corridor clipping a building's corner costs that corner, not the whole home. That matters most
where methods are compared at *equal* displacement: a charge for any contact would over-count the
methods whose roads graze many buildings, and every comparison would then favour the others by
construction.
