<!-- Handwritten partial for docs/demo.md. scripts/gen_site_pages.py prepends the do-not-edit note
     and fills eight markers (named, not spelled out in full here, for the reason
     docs/_partials/intro.md's own note gives -- spelling them out would fill this note too), every
     one of them from examples/demo/demo.json, whose README names its producer on the
     research/roadless branch:

     * DEMOPLACE -- the block, its buildings, where it is, and a Google Maps link.
     * DEMOSCREENMAP -- the site's own ScreenMap widget (the Screening and Explore pages' screen),
       here with a close-up of the block beside the city map. Not a second screen widget: the
       prototype's map is this one's, extended.
     * DEMOSCREENFACTS -- the block's score, rank and pool, checked against the city bundle the
       map draws from.
     * DEMOFIRES -- the fires reported near the block, as a list. NO map draws them: some reports'
       streets cannot be placed (owner, 2026-10-28), so none is drawn. Keep it that way.
     * DEMOMETHOD -- the method's name and how it works, in the bundle's words (the method is one
       setting of the producer, so it is never named here).
     * DEMOTIMES -- its recorded run time beside the original topology algorithm's, extrapolated,
       led by the range.
     * DEMOBUILDOUT -- the build-out widget: the budget slider, three pictures of each frame, and
       three plots whose dots follow the slider.
     * DEMOTOPOLOGYNOTE -- the footnote saying how the topology time was extrapolated.

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

<!-- DEMOPLACE -->

One Cape Town block, followed from the city-wide screen that picked it out to the roads one method
builds through it. Every control writes to the address bar, so a view you find here is a link you can
share.

## Why this block {#screening}

A metro has far too many blocks to reblock every one of them, so the pipeline opens with one cheap
number per block and a floor on it. The shipped screen is the depth proxy times density, built from
each block's building count, area and perimeter: a block scores high only if it is both deep, its
interior many parcels back from a street, and crowded. Drag the floor to grow or shrink the pool, and
watch this block in the close-up. The floor is absolute, not a percentile, so it applies the same bar
however the corpus grows or shrinks; screening only decides which blocks are worth the expensive work
below. The [Screening](methodology/screening.md) page says how the screen was chosen.

<!-- DEMOSCREENMAP -->

<!-- DEMOSCREENFACTS -->

## Shack fires reported near the block {#fires}

The screen reads only building counts and block shapes. News reports from the surrounding area show
what a deep, crowded block means when a fire starts: engines that cannot reach it, and neighbours who
fight it with buckets. None of these fires is confirmed inside the block itself, and none is drawn on
a map, because not every report's street can be placed; each place is stated as the report gives it.

<!-- DEMOFIRES -->

## Roads through the block {#methods}

<!-- DEMOMETHOD -->

Drag the budget to grow the network: each step lets the method displace a larger share of the homes'
floor area, and the dots on the plots follow. Switch the picture between the roads, each home's
escape time, and the time the roads save it: blue is good on both of those, and red on the first is
a long way out. Below, the method's own run time on this block is set against that of the original
[topology](methodology/methods/topology.md) algorithm, which on a block this size can only be
extrapolated.

<!-- DEMOTIMES -->

<!-- DEMOBUILDOUT -->

<!-- DEMOTOPOLOGYNOTE -->
