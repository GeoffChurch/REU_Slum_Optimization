<!-- Handwritten partial for docs/results/nairobi.md. scripts/gen_site_pages.py prepends the
     do-not-edit note and fills one marker, NAIROBITABLE, reading each examples/nairobi/*/
     variant's own meta.json (metric, region size) and lens_permeability.csv (whether it has an
     osm_footpaths baseline) -- see _nairobi_table(). Edit HERE, never docs/results/nairobi.md (it
     is generated and gitignored).

     No typed counts, percentages, or thresholds anywhere in this file -- digit or spelled-out
     word. That includes the variant count itself: it is exactly the row count NAIROBITABLE finds
     on disk, so it must never be asserted in prose (a spelled-out list-length count shipped once
     already on the sibling bake-off page and had to be reverted -- see d66f765). -->

# Second city: Nairobi

The same `data` → `screen` → `region_builder` → `method` → `eval` → `render` pipeline as the
[Cape Town examples](frontier.md), run on a second country: Kenya kblock data clipped to the
Nairobi metro bounding box, plus Open Buildings (`data=nairobi_full`), driven by the same
composable `BlockMetric` variants used throughout this site. Each row below is its own run through
that pipeline, reproducible end to end from the command recorded in its own `meta.json`.

<!-- NAIROBITABLE -->

## Shipped as-is, and why that is the honest framing

Nairobi's fabric is messier than Cape Town's, and these runs ship exactly as they came out rather
than being tuned to look tidy. What does not carry over from the Cape Town examples:

- **Region sizes.** Nairobi's blocks are bimodal — some enormous, some tiny — so a building budget
  tuned against Cape Town's block-size distribution produces very different regions depending on
  which end of that distribution a run's seed block happens to fall on. No single budget fits
  every case.
- **OSM footpath coverage.** The `depth` and `depth_density` regions above carry a usable as-built
  `osm_footpaths` baseline alongside the synthesized methods; the `density_compactness` region has
  essentially no mapped footpaths, so its frontier grades only the synthesized methods, with
  nothing as-built to compare against — the table's OSM baseline column shows this directly.

The screens and metric behaviour carry over cleanly from Cape Town; region-growth tuning and OSM
coverage are what differ, and the table above is where that difference actually shows up, rather
than a claim made about it in this paragraph.

## No ground truth here

The [screen bake-off](bakeoff.md) validates the screening stage against the City of Cape Town's
own informal-structure survey — and that survey has no Nairobi counterpart. The bake-off's own
caveats record the search behind that absence (the City's ArcGIS portal, openAFRICA, HDX, and OSM
Overpass) coming up empty. Its precision/recall evidence is Cape Town only and does not extend to
anything on this page: what appears above is what each method's frontier measures on its own
terms, not a validated claim about which Nairobi blocks actually need reblocking.
