# Background

## The problem

Roughly **1.1 billion people** live in informal settlements — neighbourhoods that grew without a
planned street layout. When homes are not reached by a road, they are also not reached by the
services that ride along roads: ambulances and fire trucks cannot get in, and piped water, sewer
drainage, and power have no corridor to run through. *Reblocking* is the practice of adding the
**least-disruptive** set of new roads that reconnects every parcel to the existing street network,
displacing as few homes as possible.

## Why roads matter

| Function | What a road brings to a settlement |
|---|---|
| **Emergency access** | Faster ambulance, fire, and disaster response reaching every home |
| **Utility corridors** | Space to run water, sewer drainage, and power infrastructure |
| **Accessibility** | Everyday mobility for residents — access to schools, jobs, and services |

A reblocking proposal is only useful if it buys these gains cheaply. That is the tradeoff this
project measures: the access a road network unlocks, against the homes it displaces to do so.

## Prior work

This project builds directly on two lines of research from the Santa Fe Institute and the Mansueto
Institute.

**Optimal reblocking.** Brelsford, Martin & Bettencourt (2019) framed reblocking as a formal
optimisation over a block's parcel graph. Their method is *optimal* but its worst-case running
time is exponential in the number of parcels, so it cannot run on large blocks, and it is
inherently **single-block**: each block is solved in isolation. The methods here run efficiently
and scale to regions made of many large blocks, which lets added roads stay **continuous across
block boundaries** rather than stopping at the edge of each separately-solved block.

**Detecting where to reblock.** Soman et al. (2020) detect informal settlements worldwide by a
topological analysis of crowdsourced maps — but their per-block Voronoi tessellation is
prohibitively expensive to run at metropolitan scale. This project instead screens with a cheap
**density × compactness** heuristic (`n/P²`) that scores every block in an entire metro in a single
fast sweep, and which peaks in the **Khayelitsha** informal settlement of Cape Town — the region
the [Results](benchmark.md) benchmark grows and reblocks.

### References

- Brelsford, C., Martin, T., & Bettencourt, L. M. (2019). Optimal reblocking as a practical tool
  for neighborhood development. *Environment and Planning B: Urban Analytics and City Science*,
  46(2), 303–321.
- Soman, S., Beukes, A., Nederhood, C., Marchio, N., & Bettencourt, L. M. A. (2020). Worldwide
  Detection of Informal Settlements via Topological Analysis of Crowdsourced Digital Maps.
  *ISPRS International Journal of Geo-Information*, 9(11), 685.
