# Examples

Two flagships, each reproducing from the full Cape Town metro (`capetown_full`, auto-downloaded to
`~/.cache/reblock` on first use) via plain `reblock` CLI commands — no bespoke scripts.

| flagship | what it shows |
|---|---|
| [method-comparison](method-comparison/) | **every** reblocker (dijkstra, peel, topology, mesh, greedy_arterial, clearance) graded on the four lenses, on one deep block small enough that all six run — including the two that don't scale |
| [multiblock](multiblock/) | reblock a whole informal settlement (depth 24 → 3) with clearance, and compare the **scalable** methods on it — the pipeline + method comparison at settlement scale, with clearance's depth/repulsion knobs and the substrate scaling payoff |

The small one is the comprehensive method bake-off; the large one shows what runs at scale on a real
settlement. Building points and dimmed surrounding blocks are overlaid on every heatmap automatically.
