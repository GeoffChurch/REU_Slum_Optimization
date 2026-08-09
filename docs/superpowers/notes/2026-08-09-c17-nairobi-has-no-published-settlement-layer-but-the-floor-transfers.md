# Nairobi: no published settlement layer found — but the new floor transfers better (2026-08-08)

## The negative result, so nobody repeats the search

There is **no readily available informal-settlement boundary polygon layer for Nairobi** comparable
to Cape Town's structure survey. Searched, all dry:

* general web search for Nairobi/Kibera/Mathare settlement shapefiles — community-mapping papers and
  Kenya *administrative* boundaries only;
* **HDX** (`package_search q=nairobi informal settlements`) — 2,627 datasets, all tabular indicators
  (`Housing, slums and informal settlements`, CSV/XLSX per country). No spatial settlement layer;
* **OSM Overpass** over the Nairobi bbox for `informal=yes`, `residential=informal`, and named
  `place=neighbourhood|suburb|quarter` matching the major settlements — **27 elements, 12 of them
  nodes**. The big settlements (Kibera, Mathare, Korogocho, Mukuru, Kawangware, Kangemi, Huruma,
  Viwandani) are POINTS. Only a handful of ways exist, mostly Kibera sub-zones (Soweto East Zone C/D)
  plus one unnamed `informal=yes` polygon.

Map Kibera's work is real but lives as feature data (schools, water points, bars) rather than a
settlement boundary layer. So Cape Town remains the only city here with usable ground truth.

## What IS answerable without ground truth: does the absolute floor transfer?

This is the entire argument for an absolute gate over a percentile. `metric.py` records the
complaint against percentiles: Cape Town's percentile-30 cut selects 7.6% of the ZAF+KEN corpus, so
the same rule "means two different things on the two corpora, off by a factor of four."

Measured, both floors, both cities:

    city         blocks   ddp >= 0.0128   share      n/P^2 >= 3.55e-4   share
    capetown     16,451           1,655   10.1%                 1,644   10.0%
    nairobi       3,500             169    4.8%                    79    2.3%

Both floors are calibrated to ~10% on Cape Town. Carried to Nairobi, **`depth_density_proxy` shrinks
by 2.1x and `density_compactness` by 4.3x** — so the new default transfers roughly twice as well as
the one it replaced, and n/P^2 still shows almost exactly the factor-four drift that motivated
abandoning percentiles in the first place.

## A weak but independent cross-city check

Using OSM's settlement POINTS as anchors (reliable for the major settlements even though their
boundaries are not mapped), distance from each Nairobi block centroid to the nearest known
settlement centre:

    selected by ddp floor      median 1.91 km   (n=169)
    selected by n/P^2 floor    median 2.32 km   (n=79)
    not selected               median 3.92 km   (n=3,331)

    within 2 km of a known settlement:
      20.4% of all blocks   53.8% of ddp-selected   41.8% of n/P^2-selected

`depth_density_proxy` enriches 2.6x over the base rate against n/P^2's 2.0x — the same ordering
C13/C14 found on Cape Town's real ground truth, now reproduced on a second city with entirely
independent (if weak) evidence.

## Caveats

* Six anchor points, hand-entered from published settlement locations. A 2 km radius around a point
  is not a boundary, and the big settlements are several km across, so the absolute percentages mean
  little — the COMPARISON between metrics is what carries.
* Nairobi's kblock corpus is 3,500 blocks against Cape Town's 16,451, so its shares are noisier.
* The floor was calibrated on Cape Town. That it transfers better than the alternative is evidence
  for the choice, not proof the value is right for Kenya; a Kenyan calibration needs Kenyan ground
  truth, which does not appear to exist yet.
