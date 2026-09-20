# The adjudication rule

The one question every judge answers about a block — human or agent, satellite or schematic.
It lives here rather than in a prompt so that every labelling run answers the *same* question,
and so a disagreement between runs is a disagreement about a block rather than about the task.

## The question

> **Does this block contain a dense informal settlement?**

Not "is the whole block informal", and not "are the buildings poor". Dense, informal, and
present *somewhere inside the red boundary*.

## The labels

| label | meaning |
|---|---|
| `all-dense-informal` | essentially the whole block is dense informal fabric |
| `some-dense-informal` | part of it is; the rest is formal, open, industrial, anything |
| `no-dense-informal` | none of it is |
| `unclear` | the rendering cannot support a call |

**`all-` and `some-` mean the same thing for every analysis.** The question is whether there is
fabric here that reblocking would serve, and a block that is half township and half shacks has
it. The split is recorded because it is free to record and may be interesting later — it is
**descriptive, not calibrated**: nobody checks whether "some" means 10% or 60%, and nothing
downstream depends on it. Do not start treating it as a measured extent.

## What counts

**Dense informal fabric** is small structures packed tightly with an irregular layout and no
plot subdivision — shacks abutting one another, footpaths rather than streets, roofs of
corrugated metal or mixed salvage.

**Rural and peri-urban settlement is `no-dense-informal`.** Dispersed structures on large plots,
smallholdings, farm buildings — even where the buildings are small, irregular or poor. Small
buildings alone do not make a slum. The defining features are **density** and **access
deprivation**: a rural homestead reaches a road, and threading more roads through it would serve
nobody. `ZAF.9.3.1_1_63818` is the worked example — 3,141 ha, 1,688 buildings, median footprint
99.7 m², labelled `no-dense-informal`.

**Containment is enough.** `ZAF.9.3.1_1_38988` is 17.5 ha of mixed fabric containing a
settlement; it is `some-dense-informal`, not `no-`. A screen that surfaces it is right to.

**Formal townships are `no-dense-informal`** however modest the houses: regular plots, a street
grid, uniform roofs. South African RDP housing is small — around 40 m² — and formal. Size is not
the test; layout is.

## When to say `unclear`

Say it. A satellite tile marked `⚠ TOO COARSE FOR SHACKS` cannot distinguish shacks from small
formal houses, and a guess there is worse than an abstention because it enters the analysis
looking like evidence. The scorer excludes `unclear` from both the numerator and the denominator,
so an honest abstention costs nothing and a guess costs accuracy.

## What the judge sees

Two renderings, and they fail differently.

- **`satellite/`** — aerial imagery, block outlined red. Carries roof material, construction
  uniformity, vehicles, road surfacing, vegetation: information no screen has. **Prefer it when
  the two disagree.** Raster, so it degrades on large blocks; the image states its resolution.
- **`schematic/`** — dark buildings, blue official streets, pale open ground. Vector, so sharp at
  any block size, and the only usable rendering for the largest blocks. But it shows a judge
  essentially what the screens themselves compute, so its verdicts are expected to correlate
  with the metrics being graded whether or not they are right. **It is the circular input, kept
  so the circularity can be measured rather than assumed.**

A kblock block **is** a street-bounded face, so every boundary drawn in blue is an official
street. Interior linear features are footpaths, not roads — the distinction most easily got
wrong, and the reason the schematic draws streets at all.

## Provenance: the `selection` column

`control_sample.csv` carries a `selection` column on every adjudicated row, recording **how the
block was chosen for adjudication** — not how it entered the sample. The sample itself is a
stratified random draw; this column is about what happened *after* that.

| value | meaning | usable for a population estimate? |
|---|---|---|
| `random` | drawn uniformly at random from the stratum's unadjudicated rows | **yes** |
| `queue` | chosen because it was interesting — agent/survey disagreement, high leverage | **no** |
| `dd_proxy_prefix` | the stratum's highest-`dd_proxy` unadjudicated row, taken repeatedly | **no** |

**Only `random` rows may enter a Horvitz–Thompson estimate of the survey-miss rate.** The other
two are selected on something correlated with the outcome, so their rate is not the stratum's
rate. `queue` selects *for* disagreement, which is why `10to30pct`'s six queue rows are 5/6
positive — that is the selection rule reporting itself, not a measurement. `dd_proxy_prefix`
selects for the screen's own suspicion, which biases the other way.

**A fully-adjudicated stratum is exempt.** If every sampled row in a stratum has a verdict, the
order they were done in cannot matter, and all of them count. The bias exists only while a
stratum is partially done — which is every intermediate state, and we stop at every one of them.
Hence `random`: it makes every stopping point valid rather than only the last.

**`adhoc_verdicts.csv` is never merged into `control_sample.csv`.** All of its rows came from a
`dd_proxy/p90 top-15` sheet and six of seven were picked for being survey-`formal`; three of them
are control-sample `top1pct` blocks, so merging looks free and would push that stratum toward
100%. They are real verdicts about real blocks and they stay where they are.

## Deferral: examined, but not labelled

A row with no `verdict` and a note beginning `DEFERRED` was **looked at and deliberately not
labelled**, because the label set did not fit what was there. It is not `unclear` — that is for
imagery too poor to support a call, and it is a statement about the *rendering*. Deferral is a
statement about the *labels*.

**A deferred note must carry the measurements**, so the block can be relabelled from the record
when the label set changes rather than re-adjudicated: counts, density, the footprint size
quantiles, built fraction, and the peel's ring histogram. The verdict is the perishable part; the
observations are not.

**Deferred rows are informative non-response, not missing data.** They were drawn at random and
deferred *because of what was seen*, so dropping them from the denominator biases the survey-miss
rate — the same warning 19138's note carries. Count them explicitly, with a bound.

**One marker, not two.** `19138` originally said `SKIPPED`; it was migrated to `DEFERRED`, which
is the same judgement. `unclear` already covers "the rendering cannot support a call", so a second
word for "the labels do not fit" bought nothing and cost a reader one more thing to know.

### The deferral rate says the label set fails where the estimate needs it

| stratum | selection | n | labelled | deferred |
|---|---|---|---|---|
| `10to30pct` | `queue` | 6 | 6 | **0%** |
| `10to30pct` | `random` | 5 | 2 | **60%** |
| `rest` | `dd_proxy_prefix` | 9 | 8 | 11% |

This is structural, not sampling noise, and the mechanism is visible in the table. `queue` picked
blocks where an agent and the survey **disagreed**, which are blocks with enough character for
both to have formed an opinion — so of course they label cleanly. `rest` is mostly unambiguous
formal suburb. **Random draws from `10to30pct` land in the middle of the distribution, which is
exactly where fabric is mixed and the four labels fail** — and exactly where the survey-miss
estimate lives.

The consequence is worth stating plainly: **at a 60% deferral rate the miss-rate estimate cannot
be completed with this label set.** Adjudicating all 25 sampled `10to30pct` rows would yield ~10
usable labels and ~15 informative non-responses, which cannot be dropped. Refining the labels is
not a tidy-up, it is the blocking step. Every deferred note carries its measurements precisely so
that the refinement can relabel them without re-adjudicating.

### Why the labels do not fit: they collapse two axes into one

The rule names the defining features as **density** and **access deprivation**, and then offers
four labels that vary only in extent. Three blocks separate the two features:

| block | dense? | access-deprived? | what it is |
|---|---|---|---|
| `ZAF.9.3.1_1_19138` | yes (56/ha) | no (depth 2, 0% past ring 2) | dense informal fabric, every home two houses from a street |
| `ZAF.9.3.1_1_20856` | no (21/ha) | spuriously (depth 6, but 467 m² parcels) | scattered shacks; the rings are tessellation, not footpaths |
| `ZAF.9.3.1_1_40760` | partly (59/ha) | no (depth 3) | formal row housing densified with backyard infill |

A label set indexed on both axes fits all three; the current one forces each into a box that
loses the distinction the screen is actually ranking on.

### And a third axis: tenure, which nothing we feed the screen can see

`ZAF.9.3.1_1_39399` added one more. **Street-view observation** (adjudicator, ground): surveyed
plot boundaries with painted masonry walls and steel gates, corrugated-iron dwellings behind each.
Formal **cadastre**, informal **dwellings** — the same morphology as `40760`, reached by a
different modality.

It is invisible to every input the screen has. A walled plot full of shacks photographs from above
as *shacks*; the wall does not appear. The peel tessellates building points and sees neither the
wall nor the plot line. Only the street carries it.

Density and depth say **whether there is a problem**. Tenure says **whether this tool fixes it** —
a road cannot be threaded through titled, walled plots the way it can through unplanned fabric,
however deprived the household behind the wall. That makes tenure orthogonal to the other two and
load-bearing for what a flagged block is *for*.

**Falsifiable, untested:** if this morphology is common in `10to30pct`, the survey's partial-cover
figures may be tracking it — a surveyor records the shacks they can see, on plots the cadastre
calls formal. Prediction: within that stratum `survey_cover` tracks shack fraction (a footprint
statistic) rather than peel depth. Checkable from data already on disk, with no new labelling.

Recorded here as the concrete motivation for a refinement, not as a proposal — no label set is
being changed yet.

### Street view is a third rendering, and we do not generate it

The two renderings above are what the *pipeline* produces. The adjudicator also has Google Street
View, and on `39399` it was decisive where both generated views were not. Note the asymmetry: an
agent judge sees only `satellite/` and `schematic/`, so a human verdict resting on street view is
**not** a verdict an agent could have reached. When comparing human and agent accuracy, a block
adjudicated from the ground is not a fair item — record in the note when street view carried the
call, as `39399` does.
