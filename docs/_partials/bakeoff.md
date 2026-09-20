<!-- Handwritten partial for docs/results/bakeoff.md. scripts/gen_site_pages.py prepends the
     do-not-edit note and fills four markers, all sourced from examples/screen-bakeoff/:
     SCREENTABLE and BAKEOFFFLOORS read screen_comparison.csv, BAKEOFFFIGS copies the three
     committed PNGs, and BAKEOFFSCALE reads ground_truth.json -- written by
     `pixi run python -m scripts.gen_screen_bakeoff` alongside the CSV. Edit HERE, never
     docs/results/bakeoff.md (it is generated and gitignored).

     BAKEOFFSCALE returns a COMPLETE SENTENCE or "" (ruling F5), never a noun phrase to be spliced
     mid-sentence -- it is placed below as its own paragraph so "The ground truth" section reads
     correctly with it entirely absent (partial checkout, artifact not yet regenerated).

     No typed counts, percentages, or thresholds anywhere in this file -- every number on the page
     arrives via a marker, including the cover-fraction threshold, which is described only
     qualitatively below (see Caveats) rather than typed, since it is not surfaced by any marker.
     Calendar dates (2018, 2026-08-08) are not project measurements subject to drift and are typed
     directly, matching the rest of the site (e.g. docs/background.md's citation year). -->

# Screen bake-off

Every other result on this site grades a reblocking **method** — the road network proposed for a
block already chosen for reblocking. This page grades something upstream of all of that: the
**screen** that decides which blocks get reblocked in the first place. That stage ran unvalidated
against any real ground truth until 2026-08-08.

## The ground truth

Ground truth is the City of Cape Town's own informal-structure survey, digitised from February
2018 aerial photography at 1:200 and published via the University of Edinburgh DataShare
([doi:10.7488/ds/2758](https://doi.org/10.7488/ds/2758)). The survey file carries no
settlement-name field, so settlement extents are clustered from the structures themselves rather
than looked up by name, and a block is labelled informal once enough of its area falls inside one
of those extents — a threshold choice, not a measured fact (see Caveats below).

<!-- BAKEOFFSCALE -->

## The ranking

The candidate screens below, each cheap enough to score an entire metro in a single pass — no
Voronoi tessellation, no peel — are ranked against the survey above: AUC across the whole ranking,
precision at the very top of it, and each screen's shipped absolute floor where it has one.

<!-- SCREENTABLE -->

The AUC column is the area under the **ROC** curve — the share of informal-versus-formal block
pairs the screen ranks the right way round, ties counted as half. It scores a ranking end to end,
weighting agreement across the whole sweep from top to bottom equally. A deployed screen never
reads that whole sweep, though — only the slice above its gate ever decides which blocks get
reblocked, and everything below the cutoff is discarded unread. That is why the shipped default is
the screen it is even where a competitor's AUC looks stronger: what matters is which screen
separates cleanly at the top of the ranking, not across all of it.

Note that the figure further down plots **precision against recall**, which is a different curve
from the one that area is under. The ROC area answers "how good is the ordering overall"; the
precision–recall curve answers "what do you get for a given cutoff", which is the question a
deployed screen actually faces. They are reported together because they disagree about which
screen looks best, and that disagreement is the point rather than an inconsistency.

## Head to head, at the shipped floor

The shipped default and the screen it replaced, each cut at its own shipped absolute floor, on
pool sizes close enough to compare directly:

<!-- BAKEOFFFLOORS -->

## The honest headline

Even the better screen is right about roughly one block in four it selects — most of what it flags
is not, in fact, informal settlement. The screen it replaced did worse: about three
non-settlement blocks selected for every settlement block it correctly caught. Screening a whole
metro this cheaply is hard, and that is the headline this page leads with, not the one it buries.

Read that against the **best possible** column above, though, because it is not a failure of
ranking alone. Each floor admits a pool substantially larger than the number of blocks that are
informal at all, and precision cannot exceed the share of the pool that could be right — so even a
perfect ordering, one that put every informal block above every formal one, would still be wrong
about most of what it selected at these pool sizes. The gap between the achieved precision and that
ceiling is what the ranking is responsible for; the ceiling itself is a property of where the floor
is set. Lowering the pool size would raise both, at the cost of recall.

## In pictures

Views of the same comparison: the statistical picture above, where the shipped screen and its
predecessor disagree across the whole city, and a close look at the settlements where that
disagreement is sharpest. Green is gained by the shipped default; red is dropped.

<!-- BAKEOFFFIGS -->

## Caveats

- **Cape Town only.** No equivalent published informal-settlement layer was found for Nairobi —
  see `reblock.data.informal` for what was searched and ruled out.
- **The informal-area threshold is a choice.** How much of a block's area has to sit inside a
  settlement extent before the block counts as informal is a judgment call, not a measured fact —
  but the screens' relative ranking was checked and holds across a wide range of alternative
  choices, not only the one used to produce the numbers above.
- **Ground truth is older than the blocks it is checked against, and the drift runs one way.**
  The survey structures date to 2018; the blocks they are matched against are built from more
  recent OSM and Open Buildings data. A settlement that grew after the survey is invisible to it,
  while the reverse cannot happen — so a block can be wrongly counted as a false positive, but
  never wrongly counted as a true one. **Every precision figure on this page is therefore a lower
  bound.** This is not only a theoretical worry: among the very highest-scoring blocks of the two
  weaker screens are a pair that the survey places outside every settlement, yet whose buildings
  are smaller than those in blocks the survey does mark as informal. Whether they grew after 2018
  or are genuinely formal has not been checked against imagery.
- **A feature that wins on AUC and loses where it matters.** A single Google Open Buildings
  feature — 90th-percentile building-footprint area — scores the best AUC on this page, 0.937.
  It is a bad screen: **16.9% precision in the top 1%**, against the shipped screen's 64.5%.
  Footprint size carries no scale term, so it finds small blocks of small structures — outbuildings,
  garages, tight formal rows. This is the same trap as `density` and `depth_density` tying on AUC
  while one is far better at the head, and it is why this page reports both columns.

  Used instead as a *divisor*, `depth_density ÷ p90` does beat the shipped screen on both AUC
  (0.941) and precision@1% (0.656). It is still not shipped, and the reason is measured rather
  than cautious: with both top-15s fully hand-adjudicated the two **tie at ceiling, 15 of 15
  each**, and the divisor is worse at pre-filter retention. The AUC advantage is real against the
  survey and worth nothing against ground truth.

  It was once recorded here as unshipped for needing a polygon download. That was wrong twice
  over: `area_in_meters` is a column of the points file the pipeline already reads, so the
  feature is free — and free was never the obstacle.
