# The bake-off's ground truth predates the blocks it grades, and two top-3 blocks show it (2026-09-16)

Prompted by a plain question: of the top blocks each candidate screen selects, are the ones that
look like settlements correctly annotated? For the two shipped-quality screens, yes, unambiguously.
For the two weak ones the answer exposes a limit of the ground truth rather than of the screen.

## Top-3 per screen, against the survey label

`scratchpad/top3_vs_truth.py`. `cover` is the share of block area inside a clustered settlement
extent; the label is `cover >= 0.30`.

| screen | #1 | #2 | #3 |
|---|---|---|---|
| depth_density proxy `√(nA)/P · n/A` | INFORMAL 0.949 | INFORMAL 1.000 | INFORMAL 1.000 |
| depth proxy `√(nA)/P` | INFORMAL 0.949 | INFORMAL 1.000 | INFORMAL 1.000 |
| density `n/A` | **formal 0.000** | **formal 0.000** | INFORMAL 1.000 |
| density_compactness `n/P²` | INFORMAL 1.000 | **formal 0.000** | INFORMAL 1.000 |

Across all four lists only **two** distinct blocks are unlabelled, and both are shared by the two
weak screens: `ZAF.9.3.1_1_41782` (157 buildings in ~0.83 ha) and `ZAF.9.3.1_1_44681` (53 in
~0.29 ha). Neither is a near-miss — both sit at `cover = 0.000`, outside all 189 extents.

That much is the known failure mode of `n/A`: it over-selects SMALL blocks, where `√(nA)/P` rewards
extent as well. What follows is the part that is not about the screens.

## By footprint, the two unlabelled blocks are more shack-scale than confirmed settlements

`scratchpad/suspect_footprints.py`, Open Buildings polygons, centroid-within-block:

    block                  label      n_ob   median m²    p90 m²
    ZAF.9.3.1_1_41782      formal      200      18.1       33.0
    ZAF.9.3.1_1_44681      formal       67      19.0       37.0
    ZAF.9.3.1_1_38528      INFORMAL   2017      20.6       44.0
    ZAF.9.3.1_1_22633      INFORMAL   1521      32.5       62.3
    ZAF.9.3.1_1_44531      INFORMAL     61      27.4       47.2

City-wide reference (C18): informal median 28.9 m², formal 61.1 m².

The two unlabelled blocks have SMALLER footprints than every labelled-informal control. C18's
caveat — that OB morphology cannot adjudicate BETWEEN density-based screens, being correlated with
density — does not bind here: the question is not which screen ranks better but whether one block's
fabric is shack-scale, and footprint area is direct evidence for that. The survey's own median
(29.5 m²) is independently reproduced by OB (28.9 m²).

## It is not the extent clustering. The survey itself has nothing there

The obvious explanation was that DBSCAN dropped a small cluster: extents discard anything under
`MIN_STRUCTURES = 20`, and one block has only 53 buildings. **Wrong.** `scratchpad/why_unlabelled.py`
counts raw survey structures whose centroid falls inside each block:

    ZAF.9.3.1_1_41782   formal        0 survey structures
    ZAF.9.3.1_1_44681   formal        0
    ZAF.9.3.1_1_38528   INFORMAL   2971
    ZAF.9.3.1_1_44531   INFORMAL     31

Zero. The label is "formal" because the source has nothing there, not because the labelling step
lost it.

## The likely explanation, and what it costs

The survey is **February 2018** aerial photography. The kblock geometry and Open Buildings layers
are years later. A settlement that grew after 2018 is invisible to this ground truth **by
construction**, and would present exactly as these two do: dense, shack-scale, and unlabelled.

*Untested.* Distinguishing "grew after 2018" from "genuinely formal small dense housing" needs
imagery or local knowledge, neither of which this note has. Both blocks are in the Cape Flats:

* `ZAF.9.3.1_1_41782` — https://www.google.com/maps/@-34.00511,18.56871,17z
* `ZAF.9.3.1_1_44681` — https://www.google.com/maps/@-34.01487,18.57948,17z

**If it holds, it is a ceiling on measured precision that is not the screens' fault.** Some blocks
scored as false positives may be true settlements the ground truth predates, so every precision
number on the bake-off page is a *lower* bound. That compounds with the structural ceiling already
reported there: the shipped floor admits 1,655 blocks where only 682 are informal, capping precision
at 41.2% however good the ordering.

**It does not change the ranking**, and should not be read as rehabilitating `density`. The two
shipped-quality screens put three genuine settlements in their top 3; the weak ones did not, and
their misses are small blocks either way.

## Scope, and what would settle it

Two blocks. Footprint scale is strong evidence, not proof. Nothing here was checked against imagery.

The cheap decisive test is manual: adjudicate the top *k* blocks of each screen by eye — aerial
photography, or a neighbourhood name where one exists — and record a second label alongside the
survey's. At k = 5–10 per screen that is tens of blocks, an afternoon, and it would convert every
claim above from inference to measurement.
