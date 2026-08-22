<!-- Handwritten partial for docs/explore.md. scripts/gen_site_pages.py prepends the do-not-edit
     note and fills five markers, one per pipeline stage (named, not spelled out in full here, for
     the same reason docs/_partials/intro.md's own note gives -- spelling them out would fill this
     note too):

     * SCREENMAP -- the city-scale screen. TWO bundle attributes, not one, which is the mount point
       `_write_page`'s rewriter needs its own `.replace()` per attribute name for.
     * REGIONGROW -- the block-to-region figure, from examples/region-grow/hood.json.
     * PERMGRAPHWIDGET -- the egress graph's current/after panel, the one panel a reader can drag.
       The Permeability page's PERMGRAPHFIGS draws all four; both go through `_perm_graph_panel`,
       so the two pages cannot quote perm_graph.json two different ways.
     * DISPFIELD -- the displacement model drawn literally, from examples/displacement-field.
     * FRONTIER -- the permeability-against-displacement chart. Same producer the Methods index
       calls directly, so that page and this one cannot disagree about the curves or the targets.

     Edit HERE, never docs/explore.md (it is generated and gitignored). This file is committed but
     excluded from the built site (see exclude_docs in mkdocs.yml).

     NO NUMERALS anywhere in the prose below. Each figure carries a caption read off the artifact it
     draws, so a digit typed here would be a second, unowned copy of a number that no re-bake moves
     -- the drift class the site's truth pass closed (_intro.md said "seven methods" while ten
     shipped). The block every stage follows is likewise never named here: four of the five captions
     read it off their own artifact.

     The one count in the prose is the word "Five" in the opening line, and what it counts is in
     THIS file -- the stage rail immediately below it and the marker list above. The same five, in
     the same order, are pinned by
     tests/test_gen_site_pages.py::test_explore_carries_all_five_mount_points.

     The rail is a plain <nav><ol> of in-page links; its numbers come from CSS `counter()` (see the
     stage-rail block in docs/stylesheets/sbu.css), never from digits typed into the <li>s. -->

# Explore

<nav class="sbu-stage-rail" aria-label="Pipeline stages">
  <ol>
    <li><a href="#screening">Screening</a></li>
    <li><a href="#growth">Growth</a></li>
    <li><a href="#permeability">Permeability</a></li>
    <li><a href="#displacement">Displacement</a></li>
    <li><a href="#methods">Methods</a></li>
  </ol>
</nav>

Five stages, one block. Everything below follows the same Cape Town block, from the city-wide screen
that first flagged it to the method frontier that scores the roads through it. Every control writes
to the address bar, so a view you find here is a link you can paste into a review.

## Screening {#screening}

A metro has far too many blocks to reblock every one of them, so the pipeline opens with one cheap
number per block and a floor on it. Drag the floor and watch the selected pool grow and shrink;
switch the metric to see how much the choice of number matters. Switching to Nairobi lands the same
*absolute* floor on a corpus it was never calibrated against — which is the whole reason the floor is
a score and not a percentile. The ringed block is the one every stage below follows.

<!-- SCREENMAP -->

## Growth {#growth}

A block is rarely the right unit of work: the roads that would serve it run through its neighbours,
and a road that stops at a boundary serves nobody on the other side. Growth takes a seed block and
accretes neighbours greedily until a building budget is spent. Drag the budget to watch the region
grow, and click any block in the neighbourhood to reseed from it. The greedy runs in the browser
rather than replaying a recording — which is what makes both of those controls live — and it is
checked block by block against the accretion runs the bundle records, so the rule you are dragging
is production's.

<!-- REGIONGROW -->

## Permeability {#permeability}

Permeability is the number the methods are scored on: how easily every parcel drains to the street.
Drag the road prefix to add roads in the order the method builds them, and watch current concentrate
into each new road while the permeability beside the picture climbs. Grey edges are the footpath mesh
the metric solves over, blue ones the edges a road raised, and node colour is egress potential —
hover the picture to read the nearest node's own.

<!-- PERMGRAPHWIDGET -->

## Displacement {#displacement}

A road that reaches everyone by demolishing everyone is not a solution, so every road is charged for
what it costs. Displacement asks how far the road's corridor reaches into each building's own disk.
Drag either road's endpoints, widen the corridor, and switch the second road on: two overlapping
corridors are charged once, not twice, which is why the cost is a property of the road *set* rather
than a sum over roads.

<!-- DISPFIELD -->

## Methods {#methods}

Every method is a different answer to the same trade — permeability bought with displacement — and
each curve traces one method's build-out, sample by sample, from no road at all. Set a displacement
or a permeability target and the verdicts under the chart say which methods clear both, and at what
least road. Click a method's name to isolate its curve, and again to bring the others back.

<!-- FRONTIER -->
