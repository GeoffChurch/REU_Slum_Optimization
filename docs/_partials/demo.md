<!-- Handwritten partial for docs/demo.md. scripts/gen_site_pages.py prepends the do-not-edit note
     and fills its markers, every number from examples/demo/demo.json (whose README names its
     producer on the research/roadless branch) or from the run artifacts each producer names:

     * DEMOPLACE -- the block, its buildings, where it is, and a Google Maps link.
     * SCREENMAP -- the site's own ScreenMap widget, exactly as Screening and Explore show it.
     * DEMOSCREENFACTS -- the block's score, rank and the score's two factors, checked against the
       city bundle the map draws from.
     * DEMOSURVEYNOTE -- one sentence on what the survey's gold does and does not say here.
     * DEMOPEEL -- the slower screen's measure: the flagship run's depth figure, its core
       magnified, a legend; its numbers from that run's meta.json and the bundle.
     * DEMOFIRES -- the fires reported near the block, as a list. NO map draws them: some reports'
       streets cannot be placed (owner, 2026-10-28). Keep it that way.
     * DEMOMETHOD -- the method's name and how it works, in the bundle's words.
     * DEMOBUILDOUT -- the build-out widget: the budget slider, each frame's escape-time plate,
       and three plots whose dots follow the slider.
     * DEMOCOMPARE -- our method beside the earlier one: on the comparison block both run, drawn
       and scored; on this page's block, run times only, topology's estimated.

     Edit HERE, never docs/demo.md (it is generated and gitignored). This file is committed but
     excluded from the built site (see exclude_docs in mkdocs.yml).

     NO NUMERALS anywhere in the prose below: a digit typed here is a second, unowned copy of a
     number that no re-bake moves. That includes the fires: their dates, counts and distances
     are quoted from news reports, but they live in the bundle's `fires` and DEMOFIRES renders
     them. tests/test_gen_site_pages.py::test_the_demo_partial_types_no_numerals holds the rule.
     Say each thing once: the captions carry what the figures need, the prose only what joins
     them. -->

# Demo

<!-- DEMOPLACE -->

One Cape Town block, from the screen that picks it out to the roads one method builds through it,
and that method set beside the earlier one. Every control writes to the address bar.

## Why this block {#screening}

The shipped screen scores every block in the city from three numbers it gets for free: building
count, area and perimeter. It multiplies how many homes stand behind each parcel-wide stretch of
street by how many homes there are per hectare, so only blocks both deep and crowded score high.

<!-- SCREENMAP -->

<!-- DEMOSCREENFACTS -->

<!-- DEMOSURVEYNOTE -->

The screen only picks candidates. A slower screen peels them: it gives each building its own parcel
and counts the rings of parcels in from the street.

<!-- DEMOPEEL -->

## Shack fires reported near the block {#fires}

News reports from around the block show what a deep, crowded block means in a fire. None is placed
inside the block, and none is mapped, since not every report's street can be placed.

<!-- DEMOFIRES -->

## Roads through the block {#methods}

<!-- DEMOMETHOD -->

Drag the budget: each step lets the method displace more of the homes' floor area, and the dots on
the plots follow.

<!-- DEMOBUILDOUT -->

## Our method and the earlier one {#comparison}

The original [topology](methodology/methods/topology.md) algorithm, set beside ours.

<!-- DEMOCOMPARE -->
