# Rebuilding access, one block at a time

**Stony Brook University REU · REU_Slum_Optimization**

We generate and evaluate road-network proposals for informal urban settlements — comparing
methods that trade off accessibility, cost, displacement, and buildability across real
slum-block data from Cape Town, Djibouti, and Nairobi.

## The problem

Over a billion people live in informal settlements without direct access to roads — which means
no direct access to water, sanitation, or emergency services either. *Reblocking* proposes the
least-disruptive new roads needed to connect every parcel to the existing street network.

## What the pipeline does

`reblock` screens a whole city for its most access-starved blocks, grows each into a right-sized
region, routes complementary roads with a pluggable method, and grades the result on external
connectivity (access burden removed), internal connectivity (backup-route redundancy), and
displacement — all as composable [Hydra](https://hydra.cc) stages. Every example, table, and
figure on this site is machine-generated from run artifacts committed in the repository: the
numbers can never drift from the data.

## The methods

Every method proposes a different road network for the same blocks; all are graded on the same
metric basis. Each page shows the method's roads on the ground and its numbers from the actual
runs.

| Method | Idea | Status |
|---|---|---|
| [Peel](methods/peel.md) | steepest descent down the access-depth peel — the access-optimal baseline | <span class="pill pill-done">baseline</span> |
| [Clearance](methods/clearance.md) | least-cost paths that repel from homes | <span class="pill pill-done">evaluated</span> |
| [Clearance (looped)](methods/clearance_looped.md) | clearance + loop-closing connectors for redundancy | <span class="pill pill-done">evaluated</span> |
| [Greedy Arterial (buildable)](methods/greedy_arterial_buildable.md) | best straight arterial per metre, snapped to buildable frontage | <span class="pill pill-done">evaluated</span> |
| [OSM Footpaths](methods/osm_footpaths.md) | the real as-built footpath network, as the baseline to beat | <span class="pill pill-done">evaluated</span> |
| [Euclidean Grid](methods/euclidean_grid.md) | a density-adaptive Manhattan-style grid | <span class="pill pill-done">evaluated</span> |
| [Dream Come True](methods/dream_come_true.md) | desire lines detected from imagery | <span class="pill pill-progress">in progress</span> |

## The benchmarks

The [Benchmarks](benchmark.md) page puts the methods side by side twice: six methods head-to-head
on one deep Cape Town block, and the scalable methods on a 12-block, 11,000+-parcel settlement
region graded along the full benefit-vs-added-road frontier. The
[Metrics — North Star](metrics-north-star.md) note records where the metric design is heading.

## Team

- **Mentor: Geoffrey Churchill**
- **Daisy Sanchez** — Farmingdale State College
- **Elvin Mendoza** — Suffolk County Community College

Built on research from the Santa Fe Institute's
[Open Reblock](https://github.com/mansueto-institute/prclz) line of work, with informal footpath
data digitized by the Humanitarian OpenStreetMap community.
