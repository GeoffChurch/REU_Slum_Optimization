# Examples gallery

Each subdirectory is one recipe, with its exact command in that subdir's `README.md` and the
visualizations it produces. Everything here reproduces from the **committed sample data** (the DJI
sample and the 301-block Cape Town sample) — no downloads.

| example | what it shows |
|---|---|
| [single-block](single-block/) | reblock one block; before/after access-depth heatmaps |
| [detect-reblock](detect-reblock/) | screen a city → reblock the worst → flagged-map + before/after |
| [compare](compare/) | cost-benefit curves ranking methods per meter of road |
| [multi-block](multi-block/) | reblock adjacent blocks jointly (roads span the old boundary) + region-map |
| [convex-hull](convex-hull/) | `convex_hull` region builder: fill a disjoint group into one contiguous region |
| [displacement](displacement/) | straight roads: navigability vs buildings displaced |

Building points and dimmed surrounding blocks are overlaid on every heatmap automatically.
