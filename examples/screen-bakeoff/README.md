# Screen bake-off — which screen actually finds informal settlements?

Every other example grades reblocking **methods**. This one grades the **screens** that decide which
blocks get reblocked at all — a stage that was never validated against ground truth until
2026-08-08, and where the answer turned out to matter.

Reproduce with `pixi run python -m scripts.gen_screen_bakeoff` (downloads ~18 MB of ground truth
once). Add `--counts kblock` to reproduce a pre-2026-09-17 figure; the default is `open_buildings`,
which is what `conf/building_count/` ships.

## Ground truth

The City of Cape Town's own informal-structure survey: **117,336 dwelling polygons** digitised from
February 2018 aerial photography at 1:200, published via University of Edinburgh DataShare
([doi:10.7488/ds/2758](https://doi.org/10.7488/ds/2758)). Median structure area **29.5 m²** —
shack-scale, and independently reproduced at 28.9 m² by Google Open Buildings on the same blocks.

The file carries no settlement-name field, so extents are clustered from the structures themselves
(`reblock.data.informal`): **189 settlements covering 15.4 km²**, retaining 97.9% of all structures.
A block counts as informal when at least 30% of its area falls inside one — **775 of 18,309 Cape
Town blocks, 4.23%**.

**This label set is recall-limited, and that is not a footnote here.** It is a 2018 photograph;
blocks screened on a 2023-vintage building layer include settlements that did not exist when it was
taken. A 35-block hand adjudication (`data/adjudication/`) found **9 blocks the survey calls formal
that are visibly dense informal fabric, and 0 in the other direction**. Every precision figure below
is therefore a *lower bound*, and it is biased against whichever screen is better at finding new
settlements. See `docs/superpowers/notes/2026-09-16-the-ground-truth-has-a-temporal-recall-limit.md`.

## Result

All four metrics below are **cheap** — computable from the free kblock columns (`building_count`,
area, perimeter), no Voronoi and no peel. That matters, because `density_compactness`'s historical
selling point was precisely that it needs no peel, and its competitors don't either.

| metric | AUC | prec@1% | prec@5% | prec@30% | recall@30% |
|---|---|---|---|---|---|
| `density` — n/A | **0.921** | 0.552 | 0.391 | 0.132 | 0.933 |
| `depth_density` proxy — √(nA)/P · n/A | 0.919 | **0.645** | 0.403 | 0.132 | 0.937 |
| `density_compactness` — n/P² | 0.848 | 0.590 | 0.327 | 0.115 | 0.814 |
| `depth` proxy — √(nA)/P | 0.704 | 0.426 | 0.242 | 0.085 | 0.601 |

`density` and the `depth_density` proxy are tied on AUC, but **`depth_density` is far more precise
at the top — 64.5% against 55.2% in the first 1%** — and a screen is exactly a top-slice operation.
That is why it is the shipped default. Note that at 1% `density_compactness` (59.0%) now edges
`density` even while losing badly on AUC; ranking the whole corpus and ranking its head are
different questions, which is the entire reason this table has both columns.

### At the k anyone actually uses

`prec@1%` is 183 blocks. Nobody reblocks 183 blocks — the screen exists to put a handful of
candidates in front of a human. On the hand labels, for the shipped `depth_density` screen:

| count source | prec@1 | prec@5 | prec@15 |
|---|---|---|---|
| Open Buildings (default) | 1/1 | **5/5** | **15/15** |
| kblock (Ecopia) | 1/1 | 4/4 (1 unadjudicated) | 13/13 (2 unadjudicated) |

Every block in the shipped screen's top 15 is a real informal settlement. The survey credits it with
11 of 15 — the four it withholds are the temporal misses above, not screen errors.

### At the shipped absolute floors

| screen | blocks | precision | recall |
|---|---|---|---|
| `depth_density_proxy ≥ 0.0128` (default) | 3,169 | **20.0%** | **81.9%** |
| `density_compactness ≥ 3.55e-4` (previous) | 3,049 | 17.5% | 68.8% |

Better on **both** axes at near-equal size — so the change costs nothing and adjudicates no
trade-off. That was the original argument for the switch and it survives the count change intact.

**Both floors now select about 1.9× the blocks they were calibrated for**, because the proxy is
`n^1.5/(P·√A)` — degree 1.5 in the building count — so a count source that runs higher inflates every
score against a fixed threshold. The constants were deliberately left alone: the floor sits
*downstream* of the expensive peel, so its width costs no compute, and a wider floor can only help
retention. It is a compute-free knob, not an operating point, and reading its precision as "the
screen is right about one block in five" is reading the wrong number — see the previous section.

![precision and recall](precision_recall.png)

## Where they disagree

959 blocks are gained by the new default and 839 dropped. The gained are **15.2%** informal by the
survey and the dropped **5.2%** — a 2.9× difference in the right direction. The two views show the
same disagreement at different scales.

![city map](city_map.png)

City-wide, the pattern is unmistakable: the green (gained) concentrates in the Cape Flats settlement
belt, while the red (dropped) scatters across formal suburbs that n/P² liked for being small and
compact.

![settlement zooms](settlements.png)

Zoomed onto the four settlements where the screens differ most, the mechanism is visible.
**Green sits inside and along the gold settlement outlines; red sits outside them.** n/P² rewards
compactness, which is a property of small tidy formal blocks as much as of shacks — measured, its
compactness term alone scores AUC 0.530, barely above chance, and multiplying density by it *costs*
0.921 → 0.848.

## Caveats

- **Cape Town only.** No equivalent published layer was found for Nairobi — searched across the
  City's ArcGIS portal, openAFRICA, HDX and OSM Overpass; see `reblock.data.informal`. The absolute
  floor does transfer better than the one it replaced (shrinking 2.0× to Nairobi against n/P²'s
  2.9×), but that is evidence for the choice, not a Kenyan calibration. Both ratios were
  re-measured under Open Buildings; they were 2.1× and 4.3× on Ecopia counts.
- The 30% cover threshold is a choice; the metric **ordering** was verified stable at every
  threshold from 10% to 90%.
- **The eligible pool is count-dependent.** `MIN_COUNT = 30` is a threshold on the count itself, so
  Open Buildings admits 18,309 blocks where Ecopia admitted 16,451. The 1,858 it adds are mostly
  blocks Ecopia could not see — which is why the count is resolved *before* the filter, not after.
- Ground-truth structures are from 2018, against blocks built from later OSM and Open Buildings.
  Quantified above rather than waved at: 9 confirmed misses in 35 hand adjudications.
- A single Google Open Buildings feature, 90th-percentile footprint area, scores **AUC 0.943** and
  beats every metric here — at the cost of a polygon download this screen does not need. Backlogged.
