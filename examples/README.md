# Examples

Flagships that reproduce from the full Cape Town metro (`capetown_full`, auto-downloaded to
`~/.cache/reblock` on first use) via plain `reblock` CLI commands — no bespoke scripts.

| flagship | what it shows |
|---|---|
| [method-comparison](method-comparison/) | six reblockers — `topology`, `clearance`, Looped Tree, Loop Network, Grid, and `osm_footpaths` (the as-built OSM baseline) — graded on **permeability and displacement**. On one deep block, so `topology` (single-block-only) can run alongside the scalable methods. |
| [multiblock_depth](multiblock_depth/) | settlement-scale reblock driven end to end by the `depth` metric — the deepest street-access fabric regardless of crowding. Screen → grow a region → compare Looped Tree, Loop Network, Grid, Direct Objective (LP) and the as-built `osm_footpaths` baseline on the benefit-vs-added-road **frontier**. |
| [multiblock_depth_density](multiblock_depth_density/) | the same walkthrough driven by `depth × density` instead — one swap of the pluggable `BlockMetric` (`metric=depth_density`) re-aims the screen, region growth, and colouring at the genuinely *crowded* deep fabric, so it lands on a dense informal settlement rather than the deep-but-sparse blocks. |
| [screen-bakeoff](screen-bakeoff/) | which **screen** finds real informal settlements, graded against the City of Cape Town's own 117,336-dwelling structure survey. Every other example grades reblocking *methods*; this one grades the stage that decides which blocks get reblocked at all. The shipped `depth_density` proxy is **81.7%** precise in its top 1% against `n/P²`'s 56.1%, and at equal pool size beats it on precision *and* recall (27.5%/66.7% vs 24.5%/58.9%). City-wide and per-settlement maps show where they disagree. Cape Town only — no equivalent ground truth exists for Nairobi. |
| [multiblock_density_compactness](multiblock_density_compactness/) | the same walkthrough driven by `density × compactness = n/P²` (`metric=density_compactness`) — a **peel-free** metric (geometry only, no Voronoi), so it scores tight, built-up blocks by building count per perimeter². It lands on the *densest* region of the three (142 vs 117 vs 99 bldg/ha) yet the **shallowest** (mean depth 5.1 vs 8.7 vs 6.4), because it ignores access depth entirely. |

The bake-off is the comprehensive method comparison; the three `multiblock_*` variants are the **same
pipeline** differing only in the composable `BlockMetric` that drives it — screen, region growth, and
map colouring all follow whichever metric you select at the Hydra config edge. They make the metric's
character legible: `depth` finds deep-but-sparse fabric, `depth_density` the crowded-and-deep, and
`density_compactness` the densest-but-shallowest. Each variant's §3 grades Looped Tree, Loop
Network, Grid, Direct Objective (LP) and the real as-built `osm_footpaths` network (from a committed
per-region OSM snapshot) on the benefit-vs-road frontier — the as-built footpaths need far more road
to reach far less permeability than the optimised methods. Building points and dimmed surrounding blocks are
overlaid on every heatmap automatically.

Alongside the flagships, [`perm-graph/`](perm-graph/) holds the egress graph itself drawn four ways
— conductance and current, before and after roads — on the same pinned block `method-comparison`
grades. It is a figure set for the site's Permeability section rather than a graded example, which
is why it is not in the table above.

A second city, **[Nairobi](nairobi/)**, runs the same three variants on central Nairobi data. It's
shipped as-is: the screens and metric behaviour carry over, but Nairobi's block-size distribution and
OSM footpath coverage differ from Cape Town's, so its regions come out unevenly sized and one variant
has no osm baseline — see [`nairobi/README.md`](nairobi/README.md).
