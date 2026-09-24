# Examples

Run outputs only. Each flagship subdirectory holds that run's figures, lens CSVs, `run.log`, and a
generated `README.md` written from its own `meta.json` — so the numbers in it cannot drift from the
data they describe.

**What these show, written up properly**, with the figures rendered and the comparisons explained:

- [Frontier](https://geoffchurch.github.io/REU_Slum_Optimization/results/frontier/) — the
  reblockers graded on permeability against displacement.
- [Screen bake-off](https://geoffchurch.github.io/REU_Slum_Optimization/results/bakeoff/) — which
  screen actually finds informal settlements, graded against the City of Cape Town's own structure
  survey.
- [Second city: Nairobi](https://geoffchurch.github.io/REU_Slum_Optimization/results/nairobi/) —
  the same metric variants on Kenyan data.

Those pages read these same directories' artifacts, so they are the place the results are stated;
this file is the map to where the outputs live.

**Reproduce:** see
[Reproduce](https://geoffchurch.github.io/REU_Slum_Optimization/reproduce/) for the commands, or
each variant's own `meta.json`, which records the exact command that produced it.

## What is here

Flagships, reproducing from the full Cape Town metro (`capetown_full`, auto-downloaded to
`~/.cache/reblock`):

- [`method-comparison/`](method-comparison/) — the reblockers on one deep block, small enough that
  single-block-only `topology` runs alongside the scalable methods.
- [`multiblock_depth_density/`](multiblock_depth_density/) — the region grown from the
  `depth_density` screen's top block: a single block 24 rings deep at 115 buildings/ha.
  The `depth` and `density_compactness` variants were deleted on 2026-09-20: they were dropped
  from the regeneration path, so their directories had frozen, and the screen comparison they
  appeared to offer is made better and far cheaper by `screen-bakeoff/`. Regenerate either from
  `conf/example/depth.yaml` or `conf/example/density_compactness.yaml` if you want them back.
- [`screen-bakeoff/`](screen-bakeoff/) — grades the screen rather than the reblocker.
- [`nairobi/`](nairobi/) — the same metric variants on a second city.

Figure sets rather than graded examples, feeding the site's methodology pages:

- [`perm-graph/`](perm-graph/) — the egress graph itself, drawn conductance and current, before and
  after roads, on the block `method-comparison` grades.
- [`displacement-field/`](displacement-field/) — that block drawn as the displacement model: one
  disk per building, shaded by the share the road corridor takes.
- [`authoring/`](authoring/) — the block the site's draw-your-own-road widget rebuilds in the
  browser, at full float64.
- [`region-grow/`](region-grow/) and [`screen-map/`](screen-map/) — the payloads the site's
  interactive region-growth and city-screening figures fetch.
