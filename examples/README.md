# Examples

Two flagships, each reproducing from the full Cape Town metro (`capetown_full`, auto-downloaded to
`~/.cache/reblock` on first use) via plain `reblock` CLI commands — no bespoke scripts.

| flagship | what it shows |
|---|---|
| [method-comparison](method-comparison/) | four reblockers — `topology`, `clearance`, `greedy_arterial`, and `osm_footpaths` (the as-built OSM baseline) — graded on the **metric basis**: external connectivity, internal connectivity, and displacement. On one deep block, so `topology` (single-block-only) can run alongside the scalable methods. |
| [multiblock](multiblock/) | reblock a whole informal settlement (depth 24 → 3) with clearance, compare the **scalable** methods on it, and show why the substrate approach makes settlement-scale reblocking tractable. |

The small one is the comprehensive method bake-off; the large one shows what runs at scale on a real
settlement. Building points and dimmed surrounding blocks are overlaid on every heatmap automatically.
