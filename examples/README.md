# Examples

Flagships that reproduce from the full Cape Town metro (`capetown_full`, auto-downloaded to
`~/.cache/reblock` on first use) via plain `reblock` CLI commands — no bespoke scripts.

| flagship | what it shows |
|---|---|
| [method-comparison](method-comparison/) | four reblockers — `topology`, `clearance`, `greedy_arterial`, and `osm_footpaths` (the as-built OSM baseline) — graded on the **metric basis**: external connectivity, internal connectivity, and displacement. On one deep block, so `topology` (single-block-only) can run alongside the scalable methods. |
| [multiblock_depth](multiblock_depth/) | settlement-scale reblock driven end to end by the `depth` metric — the deepest street-access fabric regardless of crowding. Screen → grow a region → compare the **scalable** methods under two budgets (fixed access-depth + matched road budget) with per-method timing. |
| [multiblock_depth_density](multiblock_depth_density/) | the same walkthrough driven by `depth × density` instead — one swap of the pluggable `BlockMetric` (`metric=depth_density`) re-aims the screen, region growth, and colouring at the genuinely *crowded* deep fabric, so it lands on a dense informal settlement rather than the deep-but-sparse blocks. |

The bake-off is the comprehensive method comparison; the two `multiblock_*` variants are the **same
pipeline** differing only in the composable `BlockMetric` that drives it — screen, region growth, and
map colouring all follow whichever metric you select at the Hydra config edge. Building points and
dimmed surrounding blocks are overlaid on every heatmap automatically.
