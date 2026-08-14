<!-- Handwritten partial for docs/methodology/permeability.md. scripts/gen_site_pages.py prepends
     the do-not-edit note and fills one marker, PERMGRAPHFIGS, which reads
     examples/perm-graph/perm_graph.json and copies the four committed PNGs beside it -- written by
     scripts/gen_perm_graph.py. Edit HERE, never docs/methodology/permeability.md (it is generated
     and gitignored). This file is committed but excluded from the built site (see exclude_docs in
     mkdocs.yml).

     This page otherwise describes a MODEL, so its prose quantities are symbolic parameters
     (g_walk, P*, r_i), never measured values -- no number is typed directly into this file; every
     figure caption's numbers arrive via the marker. -->

# Permeability

Permeability is the benefit half of the project's one graded tradeoff — the cost half is
[displacement](displacement.md). It grades how easily every parcel in a region can reach a street
once new roads are added.

## The model

The model is an electrical one. Every parcel injects one unit of "escape current" into the network;
the existing street is grounded at potential 0. A parcel with only a long, narrow connection back to
the street needs a high potential to push its current through — the same way a narrow pipe needs
more pressure to pass the same flow — so it dissipates more power getting that one unit out. Summed
over every parcel, that gives one number for the whole region, the total dissipated power
`P = bᵀL⁻¹b`, where `L` is the grounded graph Laplacian built from the mesh's edge conductances and
`b` is the all-ones vector of per-parcel current injections.

That total, `P`, is reported as an improvement over doing nothing at all:
`permeability = 1 − P(roads)/P(no roads)`. Lower dissipated power means easier collective egress, so
permeability rises as `P` falls.

## Monotone by construction

Roads only ever *add* conductance. The mesh's nodes and edges are fixed by parcel geometry alone —
adding a road never creates a new edge to route through, it only raises the conductance of edges
that already exist — and a road-covered edge takes `max(footpath, road)` rather than replacing the
footpath term outright. So for any edge, in any region, an upgrade can never lower that edge's
conductance. That makes `P` monotone non-increasing as roads are added, and `permeability` therefore
monotone non-decreasing — with no clamp needed anywhere to force it.

That property is load-bearing, not incidental. It is what makes the permeability curves on
[Results](../results/frontier.md) valid to read as a monotone climb rather than something that might dip
and need explaining away, and it is what makes a target search well-defined: the least road that
reaches a permeability target `P*`, trusting that once one prefix clears `P*`, every longer prefix
stays clear too.

## The graph

Nodes are parcel centroids. Two layers of edges connect them:

- **Footpath mesh edges** join adjacent parcels — present whether or not any road exists — with
  conductance proportional to the *clearance fraction*, the share of the centroid-to-centroid line
  lying in neither building. That makes the estimate local rather than a single corridor width
  assumed for the whole block, so a tightly packed cluster and a loose one in the same block get
  their own, different, gap.
- **Ground edges** attach every parcel within tolerance of the existing street straight to ground,
  folded directly into that parcel's own row of the Laplacian diagonal. Ground is eliminated
  algebraically — it is never a graph node you could route through.

## The graph, drawn

<!-- PERMGRAPHFIGS -->
