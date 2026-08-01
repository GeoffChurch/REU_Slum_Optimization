# Nairobi examples

The same screen → grow region → **frontier** pipeline as the [Cape Town multiblock
examples](../README.md), run on **central Nairobi** — the country-wide Kenya kblock data clipped to
the Nairobi metro bbox + Open Buildings (`data=nairobi_full`, ~16,200 blocks, auto-downloaded to
`~/.cache/reblock`). Each variant is driven end to end by one composable `BlockMetric`; regenerate
with `pixi run python -m scripts.gen_multiblock_example <metric> nairobi`.

| variant | metric | region | osm baseline |
|---|---|---|---|
| [multiblock_depth](multiblock_depth/) | `depth` | **1 block** — a single giant deep block (~4.6 km²) | yes (76 mapped ways) |
| [multiblock_depth_density](multiblock_depth_density/) | `depth × density` | 7 blocks | yes (94 ways) |
| [multiblock_density_compactness](multiblock_density_compactness/) | `density × compactness = n/P²` | **89 blocks** — tiny dense blocks | — (OSM has ~no footpaths mapped there) |

**Nairobi is messier than Cape Town — shipped as-is.** Two things don't transfer from Cape Town:

- **Region sizes.** Nairobi's blocks are bimodal (giant + tiny), so the Cape-Town-tuned 3000-building
  region budget yields wildly different regions: the `depth` seed is one giant block that alone
  exceeds the budget (a 1-block "multiblock"), while `density_compactness`'s tiny blocks take 89 to
  fill it (a few are skipped for invalid cadastral geometry). No single budget fixes both.
- **OSM coverage.** The `depth`/`depth_density` regions have usable mapped footpaths, but the
  `density_compactness` region has essentially none, so its frontier grades only the
  synthesized methods (no as-built `osm_footpaths` line).

The screens and metric behaviour carry over cleanly; the region-growth tuning and OSM coverage are
what differ. Building points and dimmed surrounding blocks are overlaid on every heatmap.
