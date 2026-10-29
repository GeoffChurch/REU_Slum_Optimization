<!-- Handwritten partial for docs/demo.md. scripts/gen_site_pages.py prepends the do-not-edit note
     and fills thirteen markers (named, not spelled out in full here, for the reason
     docs/_partials/intro.md's own note gives -- spelling them out would fill this note too), every
     one of them from examples/demo/demo.json, whose README names its producer on the
     research/roadless branch:

     * DEMOBLOCKSWITCH -- the block switch (web/src/widgets/block-switch.ts): it owns the `block`
       URL key, shows the chosen block's passages below and hides the others, and every figure on
       the page follows it. Each per-block passage is the generator's, one per block, all but the
       first hidden, so a reader without JavaScript reads the first block's.
     * DEMOPLACE -- each block, its buildings, where it is, and a Google Maps link.
     * DEMOGEOMETRY -- the screen's formula drawn on the block's real outline, ticked every parcel
       side along its street (its frontage slots), with a reference disk, square and strip of the
       same area and a buildings slider (web/src/widgets/screen-geometry.ts). No shape is
       presumed: the depth factor is the homes behind each frontage slot.
     * DEMOSCREENFACTORS -- the score's factors for every block on the page, side by side, and
       each one's true depth from the two-stage screen's peel.
     * DEMOSCREENMAP -- the site's own ScreenMap widget (the Screening and Explore pages' screen),
       here with a close-up of the block beside the city map. Not a second screen widget: the
       prototype's map is this one's, extended.
     * DEMOSCREENFACTS -- each block's shipped-screen score and rank, and its rank by the two-stage
       depth_density screen, each saying what it is ranked by; checked against the city bundle
       the map draws from.
     * DEMOSURVEYNOTE -- the floor's pool against the City's survey, and why a block the survey
       does not flag can still be plainly informal (the survey's age; the bake-off's caveats).
     * DEMOFIRES -- the fires reported near each block, as a list. NO map draws them: some reports'
       streets cannot be placed (owner, 2026-10-28), so none is drawn. Keep it that way.
     * DEMOMETHOD -- the method's name and how it works, in the bundle's words (the method is one
       setting of the producer, so it is never named here).
     * DEMOTIMES -- each block's recorded run time beside the original topology algorithm's,
       measured where it has been run, extrapolated (led by the range) where not.
     * DEMOBUILDOUT -- the build-out widget: the budget slider, each frame's escape-time plate
       (the one view the owner kept, 2026-10-28), and three plots whose dots follow the slider.
     * DEMOTOPOLOGYNOTE -- each block's footnote saying how its topology time was had.

     Edit HERE, never docs/demo.md (it is generated and gitignored). This file is committed but
     excluded from the built site (see exclude_docs in mkdocs.yml).

     NO NUMERALS anywhere in the prose below, the Explore partial's rule for its reason: a digit
     typed here is a second, unowned copy of a number that no re-bake moves. That includes the
     fires. Their dates, counts and distances are not artifact numbers -- they are quoted from
     news reports, each checked against its article -- but they are still numbers a reader takes
     as facts, so they live where the others do: in the bundle's `fires` (FIRES in the producer),
     rendered by DEMOFIRES, never typed here.
     tests/test_gen_site_pages.py::test_the_demo_partial_types_no_numerals holds the rule. -->

# Demo

<!-- DEMOBLOCKSWITCH -->

<!-- DEMOPLACE -->

Cape Town blocks, each followed from the city-wide screens that pick it out to the roads one method
builds through it. The shipped screen and a slower, deeper one disagree about which of them comes
first; the switch above picks the block every figure below shows. Every control writes to the address
bar, so a view you find here is a link you can share.

## Why this block {#screening}

A metro has far too many blocks to reblock every one of them, so the pipeline opens with one cheap
number per block and a floor on it. The shipped screen is the depth proxy times density, built from
nothing but each block's building count *n*, area *A* and perimeter *P*: √(*nA*)/*P* · *n*/*A*. A
block scores high only if it is both deep, its interior many parcels back from a street, and
crowded.

Why that formula finds deep blocks is geometry, and it needs no particular shape. Give each building
an equal square of ground; its side is √(*A*/*n*). Count how many of those squares fit along the
street: that is the block's frontage. Then √(*nA*)/*P* is simply how many homes stand behind each
square of frontage, and *n*/*A* how crowded they are. A block scores high when many homes share each
stretch of street and they are packed close. Below, the block's street is ticked off one parcel at a
time; swap its outline for a disk, a square or a strip of the same area, or change how many buildings
it holds.

<!-- DEMOGEOMETRY -->

<!-- DEMOSCREENFACTORS -->

Scored this way, every block in the city gets one number. Drag the floor to grow or shrink the pool,
and watch the block in the close-up. The floor is absolute, not a percentile, so it applies the same
bar however the corpus grows or shrinks; screening only decides which blocks are worth the expensive
work below. The slower screen peels the shipped screen's top blocks for their true depth and ranks
them again. The [Screening](methodology/screening.md) page says how the screens were chosen.

<!-- DEMOSCREENMAP -->

<!-- DEMOSCREENFACTS -->

<!-- DEMOSURVEYNOTE -->

## Shack fires reported near the block {#fires}

The screen reads only building counts and block shapes. News reports from the surrounding area show
what a deep, crowded block means when a fire starts: engines that cannot reach it, and neighbours who
fight it with buckets. None of these fires is confirmed inside the block itself, and none is drawn on
a map, because not every report's street can be placed; each place is stated as the report gives it.

<!-- DEMOFIRES -->

## Roads through the block {#methods}

<!-- DEMOMETHOD -->

Drag the budget to grow the network: each step lets the method displace a larger share of the homes'
floor area, the dots on the plots follow, and the rings of equal escape time shrink as the roads
reach into the block. Below, the method's own run time on the block is set against that of the
original [topology](methodology/methods/topology.md) algorithm, measured where it has been run on the
block and extrapolated where it has not.

<!-- DEMOTIMES -->

<!-- DEMOBUILDOUT -->

<!-- DEMOTOPOLOGYNOTE -->
