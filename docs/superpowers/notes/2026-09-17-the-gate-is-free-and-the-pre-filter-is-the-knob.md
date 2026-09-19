# The bake-off was grading a screen nobody runs, and the floor was never the cost

2026-09-17, following the Open Buildings count switch (#76) and its example regeneration.

## The defect

`scripts/gen_screen_bakeoff.py` read `building_count` straight from the kblock parquet and never
called `reblock.data.counts.resolved`. It was the one consumer that never joined the `BuildingCount`
Strategy. So after #76 switched the pipeline to Open Buildings counts, a full 9h38m regeneration
rewrote every example and republished the bake-off **byte-identical** — `prec@1% = 0.817`, the
Ecopia figure that #76's own description cites as the *before*. `main` shipped an OB-ranked screen
and published an Ecopia-ranked evaluation of it.

Nothing raised. The output had the right shape, the right columns and plausible numbers; the only
symptom was `git status` reporting no change to a file whose inputs had all moved. **That is the
signature to watch for: a regeneration that produces no diff is evidence, not reassurance.**

**And there were two of them.** `scripts/gen_screen_map.py` carries its own `load_blocks`, written
as "the same filter and CRS `gen_screen_bakeoff.py`'s own `load()` uses ... generalised to both
cities" -- so it inherited the same bypass. That one did NOT go unnoticed, because
`tests/test_screen_map_bundle.py::test_precision_and_recall_at_the_shipped_floor_match_the_bakeoff`
recomputes precision and recall from the bundle's raw `n`/`A`/`P` and asserts they match the
bake-off CSV's floors. Fixing the bake-off alone made the two artifacts disagree and the test went
red immediately.

That is the difference worth keeping: the bake-off's own bypass was invisible for a day and a
9h38m regeneration, while the screen map's was caught in seconds -- because one published number
was cross-checked against an independently computed one and the other was not. **A duplicated load
path is a defect that propagates; a cross-artifact assertion is what makes it fail loudly.**

The fix makes `load(counter)` take a required `BuildingCount` with no default, and resolves the
count **before** the `MIN_COUNT` eligibility filter — filtering on the shipped column and then
overwriting it would grade the new count over the old count's pool. That is not cosmetic: Open
Buildings admits 18,309 Cape Town blocks against Ecopia's 16,451, and the 1,858 it adds are mostly
blocks Ecopia could not see.

## The floor is not an operating point, because it is downstream of the work

`DenseCompactScreen._compute_selection` runs: proxy -> keep top `proxy_keep_pct`% -> **peel the
survivors** -> `metric.fine` -> `gate.keep`. The gate runs *after* the expensive stage. Widening or
narrowing an absolute floor changes what is reported as flagged; it does not change what was peeled,
so **it costs no compute either way**.

This reframes the count switch's most alarming side effect. Both published floors select ~1.9x the
blocks they were calibrated for, because the proxy `n^1.5/(P*sqrt(A))` is degree 1.5 in the count and
a fixed threshold on a rescaled score is a different threshold:

    DEPTH_DENSITY_PROXY_FLOOR >= 0.0128     1,655 -> 3,169 blocks    .275/.667 -> .200/.819
    DENSITY_COMPACTNESS_FLOOR >= 3.55e-4    1,644 -> 3,049 blocks    .245/.589 -> .175/.688

Left alone deliberately. A wider floor is free and can only help retention, and the argument the
floor was chosen to make — that `depth_density_proxy` beats `n/P^2` on *both* axes at equal pool
size — survives the rescaling intact (.200/.819 against .175/.688).

*Open question:* an absolute threshold on a count-scale-dependent score will drift again with the
next count source. `Gate(kind="percentile")` already exists and is rank-based, hence invariant.
Untested whether the percentile that reproduces today's pool transfers across cities as well as the
absolute floor does (measured: 2.0x Cape Town->Nairobi shrink for the absolute floor).

## The knob that does cost something, and how far it can be cut

`proxy_keep_pct` decides how many blocks get a Voronoi tessellation and a BFS peel. Shipped value is
**50** (`conf/config.yaml:48`) — half the eligible corpus, 14,361 Cape Town blocks. It is inert for
`needs_peel=False` metrics, which includes the shipped default `depth_density_proxy`; it bites only
on `depth` and `depth_density`, which the flagship examples opt into.

`pixi run python -m scripts.proxy_keep_retention`, both cities, both peel metrics, comparing the
final ranked top-k against the 50% baseline:

    city      variant                1%          2%          5%         10%         25%         50%
    capetown  depth              288:===     575:===    1437:===    2873:===    7180:===   14360:===
    capetown  depth_density      288:===     575:===    1437:===    2873:===    7181:===   14361:===
    nairobi   depth               70:==X     140:===     349:===     697:===    1742:===    3483:===
    nairobi   depth_density       70:===     140:===     349:===     697:===    1742:===    3483:===

**2% is the first safe setting; 1% breaks top-15 on Nairobi `depth`.** Top-1 and top-5 survive
everywhere, including at 1%. Recommended default **5%** — 2.5x headroom over the observed break,
and a 10x cut in blocks peeled.

Cape Town alone would have said 1% was fine. The break only appears in the second city, which is the
argument for sweeping both rather than one.

## What this does NOT buy

A cold peel measured **0.17 s/block** on median-size Cape Town blocks across a 16-way fork pool, so
the fine pass is minutes, not hours. The 9h38m regeneration was **methods**: `cycle_native` at 95.7
minutes and `greedy_arterial_access_displacement` at ~75 on a 12,622-parcel region. Cutting
`proxy_keep_pct` 10x is free correctness hygiene and will not noticeably shorten a regeneration.
If the regeneration is the thing worth speeding up, those two methods are where the time is.
