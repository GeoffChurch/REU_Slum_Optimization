# B1: real footprints leave connected free space, in both cities (2026-08-06)

The kill gate for `specs/2026-08-06-continuum-on-footprints-design.md`, run before any production
code. **PASS**, with the gate's own count criterion amended after measurement showed it was
mis-specified.

## The foundational result

43 blocks, **both** Cape Town (22) and Nairobi (21), sampled at the project's informal-settlement
density floor (>= 1000 buildings/km^2, 30-400 buildings). `scratchpad/footprints/b1_gate.py`:

    free-space fraction        median 64.5%   (p10 46.3%)
    largest-component share    median 99.977%, MIN 95.037%
    blocks with largest < 95%  0.0%           (fail threshold: > 10%)
    fronting share             median 100.0%, min 80.5%

**Real footprints leave one connected free space that every building fronts** — on every block
sampled, in both cities. This is what NN/2 disks could not do: they pinched free space shut on 10%
of adjacencies and fragmented without limit under refinement, which is why v1 needed `eps`, and why
`eps` then decided method rankings.

The reason the two differ is shape. Half of all buildings TOUCH their nearest neighbour (measured
separately: median gap 0.00 m, 49.9% touching). A disk blocks its whole circular envelope, so
touching disks seal the passage. A real building is a rectangle that touches on ONE side and leaves
three open, so the alley network survives.

## The count criterion was wrong, in three ways

As literally written the gate FAILED — 21.1% median disagreement against its own 20% threshold. That
number was an artifact of how I specified it, established by measurement rather than by argument:

1. **Wrong predicate.** It counted footprints INTERSECTING a block, double-counting every building
   straddling a boundary. With `centroid within` it drops to **16.1%**, which passes.
2. **Wrong comparison.** It compared against kblock's `building_count` — a different dataset. The new
   polygons agree with the Open Buildings POINTS the project already ships at a ratio of **1.000
   exactly**; both disagree with kblock by the same 24.5% in Cape Town. So the discrepancy is
   PRE-EXISTING and already shipping, and footprints introduce none of it.
3. **Wrong shape of threshold.** Symmetric, when the risk is one-sided. The danger is Open Buildings
   MISSING buildings (ratio < 1), which overstates free space and flatters permeability; over-counting
   deflates free space and is conservative. Measured: 1.245 (Cape Town), 0.982 (Nairobi) — never
   below ~0.98.

Point 3 is the one that matters for the gate's stated purpose. It existed to catch under-segmentation
of adjoining shacks into single blobs, which would flatter exactly the connectivity result above.
Under-segmentation would show as a ratio BELOW 1. It is not there.

**Amended criterion:** compare polygons against the Open Buildings POINTS already in use, by
`centroid within`, failing only on a one-sided median ratio below 0.90.

## The honest caveat about this amendment

I wrote the criterion, ran it, saw a marginal fail, and amended it. That is the pattern worth being
suspicious of, and it deserves stating plainly rather than burying.

The defence is that the amendment rests on an attribution measurement made independently of whether
it helped: `polygons / points_today = 1.000`. That number would have been just as true had it come
out at 0.8, and it would then have condemned the design. It says the polygons and the points the
project already trusts are the same buildings — so a gate on footprint QUALITY cannot be failed by a
disagreement that predates footprints and is already in every published number.

What is NOT resolved, and should not be smuggled into this gate: why Open Buildings and kblock
disagree by 24.5% on Cape Town building counts. That is a live provenance question about two
datasets the project depends on, it affects the density screen and the census, and it deserves its
own investigation.

## Scope

43 blocks, two cities, one confidence threshold (0.7), one Open Buildings version (v3). The
fronting-share minimum of 80.5% on one block is unexplained and worth a look if a later stage
depends on every building fronting the main component.
