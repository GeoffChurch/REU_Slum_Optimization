# Nairobi examples

Run outputs only. Each subdirectory holds one variant's figures, lens CSVs, `run.log`, and a
generated `README.md` describing that run — written from its own `meta.json`, so its numbers cannot
drift from the data they describe.

**What these show, written up properly:**
[Second city: Nairobi](https://geoffchurch.github.io/REU_Slum_Optimization/results/nairobi/).
That page builds its variant table from these directories' own artifacts and explains what carries
over from Cape Town and what does not. It is the exposé; this file is the map.

**Regenerate a variant:**

```bash
pixi run python -m scripts.gen_example <variant> nairobi
```

where `<variant>` is the metric name — the subdirectory names below are `multiblock_<variant>`. Each
variant's own `meta.json` records the exact command that produced it.

- [`multiblock_depth_density/`](multiblock_depth_density/)

`multiblock_depth/` and `multiblock_density_compactness/` were deleted on 2026-09-20 — dropped
from the regeneration path, so the directories had frozen. Regenerate from
`conf/example/depth.yaml` or `conf/example/density_compactness.yaml` if you need them.

Source data is Kenya kblock clipped to the Nairobi metro bounding box plus Open Buildings
(`data=nairobi_full`), auto-downloaded to `~/.cache/reblock` on first use.
