# Examples

Flagships that reproduce from the full Cape Town metro (`capetown_full`, auto-downloaded to
`~/.cache/reblock` on first use) via plain `reblock` CLI commands — no bespoke scripts.

| flagship | what it shows |
|---|---|
| [method-comparison](method-comparison/) | four reblockers — `topology`, `clearance`, `greedy_arterial`, and `osm_footpaths` (the as-built OSM baseline) — graded on the **metric basis**: external connectivity, internal connectivity, and displacement. On one deep block, so `topology` (single-block-only) can run alongside the scalable methods. |
| [multiblock_depth](multiblock_depth/) | settlement-scale reblock driven end to end by the `depth` metric — the deepest street-access fabric regardless of crowding. Screen → grow a region → compare the **scalable** methods under two budgets (fixed access-depth + matched road budget) with per-method timing. |
| [multiblock_depth_density](multiblock_depth_density/) | the same walkthrough driven by `depth × density` instead — one swap of the pluggable `BlockMetric` (`metric=depth_density`) re-aims the screen, region growth, and colouring at the genuinely *crowded* deep fabric, so it lands on a dense informal settlement rather than the deep-but-sparse blocks. |
| [multiblock_density_compactness](multiblock_density_compactness/) | the same walkthrough driven by `density × compactness = n/P²` (`metric=density_compactness`) — a **peel-free** metric (geometry only, no Voronoi), so it scores tight, built-up blocks by building count per perimeter². It lands on the *densest* region of the three (142 vs 117 vs 99 bldg/ha) yet the **shallowest** (mean depth 5.1 vs 8.7 vs 6.4), because it ignores access depth entirely. |

The bake-off is the comprehensive method comparison; the three `multiblock_*` variants are the **same
pipeline** differing only in the composable `BlockMetric` that drives it — screen, region growth, and
map colouring all follow whichever metric you select at the Hydra config edge. They make the metric's
character legible: `depth` finds deep-but-sparse fabric, `depth_density` the crowded-and-deep, and
`density_compactness` the densest-but-shallowest. Building points and dimmed surrounding blocks are
overlaid on every heatmap automatically.
