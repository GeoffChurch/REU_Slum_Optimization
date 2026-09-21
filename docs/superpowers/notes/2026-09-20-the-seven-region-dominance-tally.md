# The seven-region dominance tally, kept because the regions are not

2026-09-20. `examples/multiblock_depth`, `examples/multiblock_density_compactness` and their two
Nairobi twins were deleted: the variants were dropped from the regeneration path in `c6ffab0`, so
the directories were frozen artifacts, 222 MB of PNGs presenting superseded numbers as current.
None of them had been regenerated since the `block_area_m2` latitude fix (`e33f33c`), and
`nairobi/multiblock_depth` predated the Open Buildings count switch (`d50856d`) as well.

**Deleting the artifacts should not delete the finding.** `scripts/pareto_dominance.py` read all
seven regions, and a stale region is still an internally valid test bed for method-versus-method
comparison — the blocks are real blocks and the roads are real roads, whatever screen selected
them. So the seven-region tally is recorded here verbatim before the four regions went, and
`REGIONS` now lists only the four that regenerate.

**What the deletion costs, stated rather than glossed:** `OSM Footpaths` appeared in four regions
and two were deleted, so its live evidence drops to two. `Topology` was already one-region
(`one-block`) and is unaffected. The remaining four regions are all generated under current code,
where the seven were a mix — so this trades breadth for consistency, and the breadth is recoverable
by regenerating the variants from `conf/example/depth.yaml` and
`conf/example/density_compactness.yaml`, which were kept for exactly that reason.

## The tally as it stood at seven regions

```
TERMINAL DISPLACEMENT per method per region (where each method's own budget stops it)
capetown/depth               Direct Objective (LP) 0.200  Loop Network 0.200  Looped Tree 0.199  Grid 0.122  Frontage (street-priced) 0.052  OSM Footpaths 0.039
capetown/depth_density       Looped Tree 0.266  Direct Objective (LP) 0.200  Loop Network 0.199  Grid 0.128  Frontage (street-priced) 0.070
capetown/density_compactness Loop Network 0.200  Direct Objective (LP) 0.200  Grid 0.128  Looped Tree 0.115  Frontage (street-priced) 0.069
nairobi/depth                Direct Objective (LP) 0.200  Loop Network 0.200  Looped Tree 0.136  Grid 0.131  Frontage (street-priced) 0.043  OSM Footpaths 0.019
nairobi/depth_density        Direct Objective (LP) 0.200  Loop Network 0.200  Looped Tree 0.140  Grid 0.130  Frontage (street-priced) 0.051  OSM Footpaths 0.015
nairobi/density_compactness  Direct Objective (LP) 0.200  Loop Network 0.159  Grid 0.137  Looped Tree 0.100  Frontage (street-priced) 0.015
one-block                    Looped Tree 0.828  Topology 0.680  Frontage (street-priced) 0.525  Grid 0.471  Least-Cost Tree 0.389  OSM Footpaths 0.344  Direct Objective (LP) 0.200  Loop Network 0.195

A COVERS B (over B's whole curve); worst margin is the permeability A has to spare at B's worst point
     Direct Objective (LP) covers Looped Tree              5/7 | capetown/depth:+0.128 capetown/depth_density:+0.064 capetown/density_compactness:-0.065 nairobi/depth:+0.146 nairobi/depth_density:+0.079 nairobi/density_compactness:+0.164 one-block:-0.156
     Direct Objective (LP) covers Grid                     5/7 | capetown/depth:+0.025 capetown/depth_density:-0.006 capetown/density_compactness:+0.235 nairobi/depth:+0.073 nairobi/depth_density:+0.118 nairobi/density_compactness:+0.128 one-block:-0.025
     Direct Objective (LP) covers Loop Network             4/7 | capetown/depth:-0.051 capetown/depth_density:-0.102 capetown/density_compactness:+0.001 nairobi/depth:+0.038 nairobi/depth_density:+0.064 nairobi/density_compactness:+0.068 one-block:-0.131
              Loop Network covers Looped Tree              3/7 | capetown/depth:+0.057 capetown/depth_density:+0.085 capetown/density_compactness:-0.108 nairobi/depth:+0.000 nairobi/depth_density:-0.100 nairobi/density_compactness:-0.131 one-block:-0.081
              Loop Network covers Grid                     3/7 | capetown/depth:-0.081 capetown/depth_density:+0.006 capetown/density_compactness:+0.193 nairobi/depth:-0.000 nairobi/depth_density:-0.076 nairobi/density_compactness:+0.055 one-block:-0.057
     Direct Objective (LP) covers OSM Footpaths            3/4 | capetown/depth:+0.497 nairobi/depth:+0.193 nairobi/depth_density:+0.268 one-block:-0.081
               Looped Tree covers OSM Footpaths            2/4 | capetown/depth:+0.315 nairobi/depth:-0.040 nairobi/depth_density:+0.066 one-block:-0.320
  Frontage (street-priced) covers Grid                     2/7 | capetown/depth:-0.098 capetown/depth_density:-0.092 capetown/density_compactness:+0.035 nairobi/depth:-0.034 nairobi/depth_density:-0.061 nairobi/density_compactness:-0.097 one-block:+0.017
  Frontage (street-priced) covers OSM Footpaths            2/4 | capetown/depth:+0.156 nairobi/depth:+0.170 nairobi/depth_density:-0.101 one-block:-0.042
               Looped Tree covers Grid                     1/7 | capetown/depth:-0.218 capetown/depth_density:-0.153 capetown/density_compactness:+0.119 nairobi/depth:-0.140 nairobi/depth_density:-0.025 nairobi/density_compactness:-0.058 one-block:-0.221
              Loop Network covers OSM Footpaths            1/2 | capetown/depth:+0.372 one-block:-0.092
                      Grid covers OSM Footpaths            1/4 | capetown/depth:+0.107 nairobi/depth:-0.070 nairobi/depth_density:-0.247 one-block:-0.167
  Frontage (street-priced) covers Looped Tree              1/7 | capetown/depth:-0.071 capetown/depth_density:-0.055 capetown/density_compactness:-0.084 nairobi/depth:+0.036 nairobi/depth_density:-0.274 nairobi/density_compactness:-0.039 one-block:-0.008
     Direct Objective (LP) covers Frontage (street-priced) 1/7 | capetown/depth:-0.016 capetown/depth_density:-0.054 capetown/density_compactness:-0.035 nairobi/depth:-0.033 nairobi/depth_density:+0.010 nairobi/density_compactness:-0.214 one-block:-0.148
  Frontage (street-priced) covers Topology                 1/1 | one-block:+0.010
```
