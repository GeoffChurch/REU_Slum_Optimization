# Methodology

The project is one composable pipeline: **`data → screen → reblock`**, with every stage a
swappable [Hydra](https://hydra.cc) component and every output written to disk so the whole run
reproduces from a single command.

## The pipeline

1. **Data.** Load building footprints and existing streets for a metropolitan area, and derive each
   block's parcel graph — who neighbours whom, and how deep each parcel sits from a street.
2. **Screen.** Score every block in the metro with a cheap heuristic and flag the most
   access-starved ones, then grow the top block into a right-sized, multi-block **region**. This is
   fast enough to sweep an entire city in one pass (see [Background](background.md)).
3. **Reblock.** Run a pluggable [method](methods/index.md) that proposes new roads for the region,
   adding road until an access target is met. Because the region spans many blocks, proposed roads
   stay continuous across block boundaries.

The result is then **graded** — the same way for every method — on how much access it unlocks
against how many homes it costs.

## Definitions

The screen and the grading rest on a small set of per-block geometric quantities. For a block of
**n** parcels with area **A** and perimeter **P**:

| Term | Definition |
|---|---|
| **density** | `n / A` — how many parcels per unit area |
| **compactness** | `A / P²` — how round the block is (high for a disc, low for a sliver) |
| **density × compactness** | `n / P²` — the **screening heuristic**: crowded *and* compact blocks score highest |
| **displacement** | the expected number of parcels a road set displaces — the **cost** axis |
| **permeability** | `1 − dissipation fraction` — how easily every parcel can reach a street once roads are added — the **benefit** axis |

Every method is judged on the **permeability it buys per unit of displacement**. That single
tradeoff — read straight off the [Results](benchmark.md) frontier — is how methods are compared.

## Where this is heading

The current metrics grade the *road geometry*. The next steps push toward engineering realism:

- **Utility networks** — model the water, sewer, drainage, and electricity corridors a road opens,
  not just the road itself.
- **Planning standards** — validate proposed layouts against municipal planning and design codes.
- **Engineering metrics** — score proposals on quantities residents feel directly, such as
  emergency-response time and utility accessibility.
