<!-- Handwritten partial for docs/methodology/index.md. scripts/gen_site_pages.py prepends the
     do-not-edit note and writes this straight through -- there are no markers to fill on this
     page, only prose that links out to the pages that give each term its full treatment. Edit
     HERE, never docs/methodology/index.md (it is generated and gitignored). This file is
     committed but excluded from the built site (see exclude_docs in mkdocs.yml). -->

# Methodology

`reblock` is one composable pipeline. Every stage is a swappable [Hydra](https://hydra.cc)
component, and every run writes its output to disk, so a result reproduces from a single command:

**`data` → `screen` → `region_builder` → `method` → `eval` → `render`**

## The pipeline

1. **Data.** Load building footprints and existing streets for a city, and derive each block's
   parcel graph — who neighbours whom, and how many parcels deep each one sits from a street.
2. **Screen.** Score every block in the metro with one cheap heuristic and flag the most
   access-starved ones, fast enough to sweep an entire city in a single pass. See
   [Screening](screening.md).
3. **Region builder.** Grow each flagged block into a right-sized, multi-block **region** before
   any road is proposed, so the roads that follow can stay continuous across block boundaries
   instead of stopping at the edge of one separately-solved block. See [Screening](screening.md).
4. **Method.** Run a pluggable [reblocker](methods/index.md) that proposes new roads for the
   region — the roads themselves, and how many, are entirely the method's choice.
5. **Eval.** Grade the result the same way for every method: **permeability** — how easily every
   parcel can reach a street — bought against **displacement**, the homes it costs.
6. **Render.** Draw the proposed roads and the before/after maps used throughout
   [Results](../benchmark.md).

## Glossary

- **[Density](screening.md)** (`n/A`) — parcels per unit block area.
- **[Compactness](screening.md)** (`A/P²`) — how round a block is; high for a disc, low for a
  sliver.
- **[The depth proxy](screening.md)** (`√(nA)/P`) — a closed-form estimate of access depth, cheap
  enough to compute for every block in a metro.
- **[Permeability](permeability.md)** — how easily every parcel in a region can reach a street once
  roads are added.
- **[Displacement](displacement.md)** — the homes a road set costs.
