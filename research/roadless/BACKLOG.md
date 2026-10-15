# Roadless clearing: experiment backlog

A living list: add ideas, strike them when tried (the result goes to NOTES.md, with numbers),
delete them when shown dominated. Status tags: **prioritized** (the owner wants it done),
**queued**, **running**, **idea**, **owner's call**, **blocked**.

Baselines (NOTES.md, "Large blocks" and "Lens A at D 0.05"): greedy `S0.01cat` /
`S0.005cat` and SIMP + eps-scored incumbent `fw0.q3.i10.t0.001.e0.0001.k1s1e-06`, both J_2,
uni conductance, Lens A at D 0.05 (`D_LENS`; owner 2026-10-03, 0.10 saturates under sightline),
and Lens B from nested runs.

## Status of the leads (owner 2026-10-06: "leave no stone unturned. Every lead should be followed")

Presets now (NOTES, 2026-10-05 -- 06; D 0.05, J_2, uni):
- **Default: `.p256w8x2`** -- SIMP base + the undecided-set polish with pair moves. The 13 +0.0097
  at 1.26x, the 46 held out +0.004 vs base (ahead 89%, never behind).
- **Hard blocks: `.p256w8x2.GS0.01cate1`** -- plus the greedy as a second polished track, polished
  only when its raw clearing is within 1% of SIMP's polished one. The 13 +0.0119 at 1.21x the
  default (the greedy unscored); identical to the default on all 46 held out.
- **Cheap: greedy + polish** (polish_greedy.py, P64w8) -- against SIMP base, the 13 -0.004 at
  0.81x, the 46 -0.015; against the default, the 46 -0.0185 [-0.022, -0.015] (never ahead) at
  0.53x its time on one card.

Closed (measured, NOTES): multi-rounding `.r<n>` (held out +0.001 at 2x; late `.r2l2` the same at
1.58x); coarse-to-fine `.c0.75` (keeps gates at 0.85x, but with the polish dominated); coarse
multi-start `.C<h>.m<k>` and fine `.m<k>` (neither faster nor better); the two-phase polish `.pP`
(pairs reach the same blocks); seeding `.g` (loses where a better start polishes worse: `.G`
instead); greedy screening width `S0.01catw4` (+0.0006, marginal); pairs after the greedy (+0.0005,
marginal); resolution (res-check: h 0.75 within 0.01 on 35 of 59, h 1.0 understates gated blocks by
up to 0.31); sequential stopping (seqscreen.py: by the value of more blocks, 15 -- 24 of 46).

Also closed: the oversized blocks (the 17 over 1.4 km^2 and 53556), base and default at the
metric's resolution through local coarsening (`.a5x8`; 4287 `.a5x32`, 63612 `.a2.5x8`; NOTES, "The
oversized blocks at the metric's resolution"); 19593 and 6498 on the H100; 14401, 7851 and 32841's
default at a5x8. Every one of the 82 large blocks has the default at h 0.5 near its buildings.

Also closed: wider polish rounds (w16 = w8 at 1.28x) and late sampling on top of the hard-block
preset (+0.0001 at 1.33x); the greedy's unused scoring skipped (the preset 1.30x -> 1.21x).

Prioritized (owner 2026-10-08: "I want to do all of the following"), each entry below:
- **Goal-oriented refinement** (Coverage and model): done, the o1 goal mesh, on the 59 and the
  oversized blocks.
- **Sightline and the greedy on the composite** (Coverage and model): done, on the 59.
- **The cheap preset on the composite** (Coverage and model): done, on the oversized blocks.
- **The add/remove search as one Strategy** (New methods): built, with floating search.
- **An outcome model for kernelcore** (Experiment design): dropped on review (owner
  2026-10-09); the stop is built in seqscreen.py instead, by the value of more blocks (NOTES,
  "Stopping by the value of more blocks").

Owner's agenda (2026-10-10: "we can take things up one by one and write down all the open
items up front"), in the order proposed:
1. **The metric's resolution floor**: done (NOTES, "The metric's resolution floor"). Mean gaps
   keep their sign at h 0.35 and 0.25 and under shifted lattices, their size within about 20%;
   no floor added to tau (alignment is per-block noise the bootstrap already carries). A
   block's level is not: alignment alone moves it by up to 0.16, so per-block levels need a
   margin of about +-0.08 on some large blocks.
2. **Visualization**: the presets' clearings and the corridors they induce (the current through the
   opened space), side by side, so results can be looked at, not only scored. First page: 5810
   (NOTES, "Visualization: block 5810"). The corridor pictures use uniform, the metric's own
   conductance (owner 2026-10-13; item 4, decision 3), so what is shown is what is scored. Why a
   level moves (each home's term at two lattice offsets on 20543, 38366, 17608): done (NOTES,
   "Uniform conductance with an effective width", Q3). The homes whose demand sits in pockets
   joined to the street only by phantom passages (4-adjacent cells whose open sub-samples do not
   touch) or sub-cell cracks carry it: under U, 2 -- 5% of the people behind phantom passages hold
   17 -- 42% of J_2's baseline, and the share moves with the alignment (20543: 0.60 to 0.33 for all
   such homes). The grid fix (NOTES, "The grid fix: steps that conduct by their open width") takes
   it away: under the face aperture (Fws) SIMP's default's lens moves by at most 0.009 (P), 0.022
   (P_{1/2}) and 0.027 (J_2) between the two alignments, against 0.047, 0.234 and 0.147 today, and
   what is left is spread over the block (the homes whose crack exposure moves with the alignment
   carry about their share of the people): the lattice's ordinary resolution (item 1).
3. **The sightline metric and building transparency** (discussion): what each measures, where SIMP
   is weaker under sightline, and transparency as a model of the world against a device of the
   search (the translucent greedy). With it, the passable width, now implicit in h and the
   sub-sampling (a gap conducts by its sampled open fraction, which the alignment sets). Its angle
   error (NOTES, "The sightline metric's angle error"): uneven ray directions and unboosted turning
   edges; two fixes measured on a corridor. The owner's call on the turning boost (a model change:
   walkers stop going ballistic in a lane); then its gradient, uniform rays on the GPU and the
   composite mesh, and a re-score under the sightline metric. With uniform chosen for the metric
   and the pictures (owner 2026-10-13; item 4, decision 3), the turning-boost call and these
   follow-ups have no use left: parked, sightline stays selectable (kept or pruned in the
   consolidated pass). The passable width: the grid fix (item 4, decision 3) removes the phantom
   passages; closing real gaps under a width stays a separate choice.
   Buildings as imperfect insulators (owner 2026-10-11; NOTES, "The insulator world"): no single
   eps both meets the owner's first criterion (a walled-in home worse, but not massively worse,
   than its open neighbours: 1e-3 -- 3e-3) and keeps the gates (1e-4 or less). Tested for that
   criterion: the shadowed homes on their open side (kept), one door link per walled-in home, a
   toll after the solve (the door's numbers without its edge) and private walls (the tail halved
   by letting a pocket out on every side); crossing.py, stopped. Then the owner dropped the
   criterion: enclosed homes may be much worse off, with one wall resistance for everyone. Tested
   (walls.py; NOTES, "Uniform walls"): one number at eps 1e-4 (A) against two (B: the population
   still enclosed, then today's Lens A over the rest). No eps is canonical: the enclosed homes'
   weight in J_2 goes as 1/eps^2, and at 1e-4 one home holds 80% of 20543's; A also leaks 0.02 --
   0.04 at the gates. B is the same family's leak-free end, its optimizer two stages (free the
   enclosed homes, D median 0.0022 a block, then today's search; a search at the gate-keeping eps
   4.2e-6 is out, the GPU solver stalls there). The owner chose B (2026-10-11: "B SGTM"), and it
   is ported (NOTES, "The port": lifted.Placement, Scorer.enclosed_left; insulator.py,
   crossing.py and walls.py deleted, the door's, the toll's and the private walls' code with
   them). Scored (item 4): in B's order the best road prefix at D 0.05 was ahead of SIMP's stored
   default on 9 of the 12 large blocks with enclosed homes, on (i) alone. B's optimizer (NOTES,
   "B's optimizer: free first"; clear.free_first, then today's search with the rest of the
   budget) frees every home on all 12 for D 0.0001 -- 0.0026, at a median 0.003 of Lens A, and
   both presets now lead the best road on all 12. Not free-first yet: the composite mesh
   (free_first raises there; the a5x8 clearings predate B), the plain greedy (clear.greedy_block,
   the track), and the stored rows on the other blocks with enclosed homes (21 small ones among
   the 40 checked; the oversized).
4. **Evaluation** (discussion): (a) the roadless cost and benefit thought through (population as
   footprint area, J_2, the budget D); (b) a Galois connection between clearings and road
   networks over a fixed road universe: L(network) = the buildings it displaces, R(clearing) =
   every candidate road that fits the opened space, so L(N) <= C iff N <= R(C); the composites
   are idempotent by construction (R L a closure, L R an interior); the forgotten information is
   the corridor width; the roadless optimum then bounds every road network under the roadless
   benefit at equal displacement (the price of roads); (c) what counts as an exit: every kblock
   outline (now, so blocks are independent and a region only pools the budget), or (owner
   2026-10-10) only a region's own boundary, roads not used at all, streets wide open ground.
   Then the region's shape is part of the model: a hole is an exit inside it, a concavity an exit
   deep in it, and its size sets how much of J is walking along streets, which no clearing
   changes. Candidates: faces of the major-road network (a boundary that is a real exit), or
   kblock unions with holes filled and concavities closed, every building inside. Rectangles and
   circles buy no solver advantage: a sine or Fourier transform diagonalizes only uniform
   conductance, and buildings are zero-conductance holes. The one transform with a use is FFT
   homogenization (Moulinec--Suquet), a fabric's effective conductivity tensor per neighbourhood
   for a coarse two-scale model: tested (NOTES, "FFT homogenization"), not a design model (it
   erases gates and compresses the gaps between clearings), but a screening layer (NOTES, "FFT
   homogenization on 5810@major"): its first-order map at 100 m windows ranks 50 m tiles like the
   fine model (Spearman 0.98 -- 0.99), and a clearing confined to the map's top tiles (5x the
   budget's footprint, a quarter of the region's) keeps 90 -- 95% of the optimizers' Lens A.
   First test (NOTES, "A region whose only exit is its edge"): on 5810@major the budget leaves
   5810 for the formal grid in its hook, whose streets no longer count as access. Open: streets
   as faster ground rather than exits (a conductance on OSM ways, or the sightline metric, which
   needs a raster: 27M cells here), railways and freeways as barriers, which boundaries are real
   exits, a real restricted run at k 2 -- 3 (the cheap preset needs only a tile mask, SIMP ~50
   lines), and a budget pooled across faces, where screening pays; the fine model itself may
   reach built-up Cape Town (~5 -- 11 H100-h by extrapolation).
   Re-scoring the road methods (NOTES, "Re-scoring the road methods: the large-block sentinel"):
   on 4 large blocks the shipped metric and roadless order the road methods alike (tau 0.60 --
   0.87 for the walking model alone, against the 220's median 0.43), roadless separates the
   leaders (resistance_lp ahead), and SIMP's default beats the best road network at D 0.05 by
   +0.065 -- +0.777, a lower bound on (b)'s price of roads. The full run on the 60 blocks that
   fit at h 0.5 (NOTES, "Re-scoring the road methods: the 60 large blocks under B"; six methods,
   the networks kept in road_rows/scratch, ignored): the lenses agree as on the sentinel (tau
   medians +0.47 -- +0.80), resistance_lp leads all three, and in B's order SIMP's default, free
   first on the 12 with enclosed homes (item 3), is ahead of the best road on all 60 (by a median
   +0.152 where (i) ties, on 55, at least +0.047; on (i) on the other 5). Open: the 23
   oversized blocks (cycle_desire's cost and memory go with area) and the arterial (its
   `max_roads: 60` keeps it short of Lens A).
   **The metric decisions** (owner 2026-10-12, walking through four: the escape-time functional,
   the population, the conductance, the exits). (1) **P as the headline** ("leaning toward P with
   J_2 as a side report"): one solve, monotone, convex in the clearing (a certified bound), the
   shipped permeability's own quantity on the open-space raster; the equity view beside it is J_2
   or, if it can replace J_2, a priority-weighted P (injections w_i (u0_i)^q at baseline escape
   times, one solve and still convex; q = (p - 1)/2 matches J_p to first order on independent
   paths, so q 1/2 is J_2's analogue and q 1, injection growing with depth^2, J_3's). Tested on
   the 220 small blocks (NOTES, "Priority-weighted P"; SIMP base with free first, area D 0.05, all
   four objectives): P_{1/2}'s clearing is not behind J_2's on J_2's lens (+0.0009 [-0.0013,
   +0.0036]), and at the p95 home it is +0.0011 [-0.0028, +0.0050003], 3e-7 past the
   pre-registered 0.005, so as written it does not replace J_2; the miss is inside the
   bootstrap's Monte Carlo noise (0.00493 with 200,000 draws). It is better than J_2 at the max
   home (-0.020) and worse at the median (+0.005), clears deeper, gives up 0.0028 of P (J_2
   0.0015), takes one solve a step to J_2's two (SIMP base 28.7 task-h to 40.3) and carries a
   certified bound (0.16 above its clearing, nearly all of it the grey relaxation's); P_1
   overshoots (J_2's lens -0.009, the p95 home +0.007) and P fails both (-0.006, +0.016).
   **Decided (owner 2026-10-13): P_{1/2} is the equity view beside P** ("we can make P_{1/2} the
   equity view beside P"). J_2 stays in the site's account of the two families, as the
   established measure P_{1/2} stands in for.
   (2) **Area population for buildings** ("area for buildings makes the most sense ... relocate
   individual homes rather than demolishing well-designed large community infrastructure like a
   school, especially since people do come in and out of it"): area sets both a building's
   injection and its displacement cost; the sources are already outside the buildings
   (Placement's ring of free cells within 1 m), which was the owner's other concern.
   (3) **The conductance: decided (owner 2026-10-13), uniform (U), for the metric and the corridor
   pictures alike** ("LGTM, incl using uniform for the corridor pictures"; on new corridors: "I
   don't actually mind if there are no new corridors. That may be to be expected"). The
   effective-width study (NOTES, "Uniform conductance with an effective width") built B_r (each
   sub-sample conducting by min(its openness, the open share within r); P stays convex, a gap
   below 2 r conducts like width^2) and D_s (min(1, d / s), scoring only) as lifted.Params.cells (a
   Strategy), beside sightline (S) and its turning boost (S_t, lifted.TurnBoosted). U is the
   natural model (open ground conducts alike and buildings not at all, P the walkers' total escape
   time, no knob) and the most robust: the smallest angle error of every variant at every width
   (1.66 at 1 m; B_r 1.72 -- 1.85, D_s 2.39, S 5.32) and the least drift of P with the lattice. The
   principled alternative changes no decision: B_r keeps U's method order (tau 1.00 on P), moves
   the flow only a little onto lanes U's flow already uses (no new corridors at r up to 1.5 m), and
   SIMP under it gains nothing over U's clearing on its own lens. The variants stay selectable
   (pruning in the consolidated pass). **The grid fix: done** (owner 2026-10-13, "LGTM"; NOTES,
   "The grid fix: steps that conduct by their open width"): a step conducting by the share of its
   width open from cell centre to cell centre, read on the sub-samples; the candidate, Fws
   (`uni+apwfs`), traces an axis-aligned step across its face and lets a split cell conduct
   through one part only. Its reach is the flood fill of open sub-samples, so the phantom passages
   are gone; it encloses 0.09% of the small blocks' people (0.02% of the large), freed first for a
   median D 0.0031, moves 1.3% of the demand, takes the levels' lattice drift from up to 0.23 to
   at most 0.027 and the corridor's angle error at 1 m from 1.66 to 1.21, and keeps the method
   order (tau median 1.00 on every lens on the 13; SIMP's default leads on 10 -- 12,
   resistance_lp the best road). **Adopted** (owner 2026-10-13: "We can adopt Fws and delete the
   losing methods"; NOTES, the section's last paragraph): Fws is the operator's one step rule
   (lifted.step_widths), the rule before it (SmallestCell) and A, Aw and Aws deleted, and every
   conductance's name ends in `+apwfs` (uni+apwfs), so no row of the old rule is read as the new
   one's; the stored clearings stand, their scores are the old rule's until the consolidated pass
   re-scores them. **The passable width: studied** (owner 2026-10-13, "sweep it, as a persistence
   filtration"; NOTES, "The gap complex: passability, enclosure and flow without the raster"). Fws
   conducts a real crack of 1 -- 2 sub-samples at its sampled width and the equity lenses follow
   it; a gap nobody walks through is a wall. The gap complex (the open ground's constrained
   Delaunay triangulation, each passage's aperture the gap's width there, read once from the
   outlines) reproduces the raster's enclosure and free first on all 302 blocks up to the raster's
   resolution (every difference a crack under 12.5 cm, a sub-sample leak, the ring's cell-centre
   edge or the mixed cell's cleared share), and swept over w it finds no natural width: enclosure
   grows at every step (0.07% of the people at 0.125 m, 0.22% at 0.5, 1.0% at 1 m, every one freed
   first, D over 0.05 on 2 blocks at most) and the exit widths show only a shallow trough from 1 to
   35 cm between traced walls and lanes. Of three forms (closing, the shy distance: every footprint
   grown by w / 2, a ramp), closing and the shy distance at 0.5 m order the methods alike (tau
   median 1.00), and the width moves the order more than the heading lift does (tau median 0.79 --
   0.86 against 0.125 m, where the lift's is 1.00). The raster's lattice loses 0.14 -- 0.29 m of a
   lane at each wall and the lift 0.3 -- 1.1 m, the turning length's boundary layer: today's
   superlinear width law is theirs; the shy distance at 0.5 m reproduces the lifted metric's levels
   on the complex without a lift (median -0.003 on the 13, 0.297 and 0.270 against 0.301 and 0.271
   on 5810@major), gives SIMP a better landscape than closing's step, and makes its clearings carry
   a little more of the flow in wide lanes (+0.007 of it in lanes of 2 m or more), not fewer
   corridors. **Decided (owner 2026-10-14, "that SGTM"): the shy distance at w 0.5 m on the gap
   complex, as the metric and as its runtime**, isotropic, with the raster, its lift and the
   turning cost retired once the complex is measured to dominate them. A vehicle's width is not the
   walking metric's shy distance (the owner asked, 2026-10-14): at 3 m, a lane narrower walls in
   27% of the small blocks' people, 16% of the large blocks' and 2% of the oversized blocks'
   (summary_exits.csv), so enclosure, which comes first, would read vehicle access and P would only
   break its ties. Vehicle access is its own reading of the same filtration (a home's exit width is
   its widest path's bottleneck to the street; turning is not modelled), reported over W, since the
   widest-path study found reblocking pays for vehicles only above about 4.5 m with the ranking
   rotating with W; it comes with the vulnerability score (owner 2026-10-14: "Later, with
   vulnerability"). A GPU path is added only where measured faster: the complex's systems are
   planar and factor directly on a core (SuperLU 1.5 s at 690k unknowns, against pyamg's 16 -- 69 s
   a solve), and today's cost is the mesh rebuilt for every clearing, which a GPU cannot help.
   **The runtime's first phase: done** (NOTES, "The complex as the runtime: one fixed mesh"): one
   fixed mesh a block (every outline and its w / 2 offset noded at 0.1 mm, the footprints'
   interiors and the shy distance's zone strips meshed with the open ground, every face boundary
   split until the mesh is Delaunay, so no weight is clipped; cached by identity), a clearing a
   re-weighting of its passages with no GEOS call; the scorer, free first on the complex grown by w
   / 2, and the optimizers' seam (seam.py's Protocols, folded in phase 3; the metric resolved
   where a run is configured: metrics.metric_of), the raster and the complex both behind it, the
   raster's numbers bit for bit as before; the metric `cx+shy0.5+s0.125r1b1+R0.5g4x2q4`, its rows
   under that name.
   Against the rebuild: the baselines within 2.3e-4 on 16 of 18 blocks and every clearing's lenses
   within 3.3e-4 on 12 of 13, reach equal home for home; the outliers are necks a few millimetres
   wide (46841's 2.1 mm), where the rebuild converges toward the fixed mesh as s refines. An exact
   score takes 1 -- 11 s, 16x faster than the rebuild. The presets run on it on one core: each
   metric's optimizers lead on their own metric in B's order (the complex's SIMP on 9 of 13 against
   the raster's clearing, its cheap preset on 12), the losses the search's (the complex's own
   polish holds the raster's clearing or climbs from it), at 4 -- 73x the raster's GPU time (median
   11 -- 13x) and 2 -- 13 GB. The factorization is 57 -- 81% of the time; cuDSS on one H100
   refactors the same systems 240 -- 290x faster than SuperLU once a pattern's plan (16 -- 40 s) is
   made, about 3x on an exact score and 5x on a SIMP iteration as the host's work stands: a GPU
   path. **Decided (owner 2026-10-14)**: the GPU path ("Yes, add a GPU path": cuDSS through
   nvmath-python in the `gpu` group, the factorization a Strategy chosen where a run is configured,
   SuperLU on a core and cuDSS on a GPU), and the raster kept for the cross-checks, then retired
   ("we can keep the raster for the cross-checks and then retire it"), its speed and each retired
   variant to the wiki's dead-end record with their numbers. **Phase 2a: done** (NOTES, "The
   complex as the runtime: its speed and a GPU path"): the factorization a Strategy (gap_factor:
   SuperLU on a core, cuDSS on a GPU through nvmath-python 1.0.0 and cuDSS 0.8 in the `gpu` group,
   loading in the project's own environment here and on the cluster), every state assembled on one
   fixed pattern (scores equal phase 1's within 6.4e-11), the host's own work cut 1.5 -- 2.5x, free
   first's baseline complex cached with the mesh, and the metric's name carrying a digest of the
   code's constants (`cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v34d40e8a`; the stored rows migrated, every
   one re-scoring within 7.1e-12). On a GPU the two presets end at phase 1's clearings 29x (SIMP)
   and 9.7x (the cheap preset) faster than on a core, a median 0.38 and 0.93 of the raster's GPU
   time; a SIMP iteration on 5810 takes 1.6 s against 68, the region's baseline score 3.7 s against
   33 s. The polish scored exactly (`E`) gives the same clearings at 0.81 and 0.56 of the time on a
   core and 1.05 and 1.04 on a GPU, so each device keeps its own (`E` on a core, the eps world's on
   a GPU); the eps world's interiors lumped (`+relaxlumped`) was dominated (Lens A lower, slower on
   both devices) and deleted. **Phase 2b: done** (NOTES, "The complex as the runtime: the ports and
   the cross-checks"): what the raster still did alone ported to the complex. Roads are cut into
   the fixed mesh as atoms (gap_clear.Carving; a prefix a state, the shy distance exact on the
   atoms; against the rebuild L_P within 6.2e-4 on four of six blocks, the other two carrying phase
   1's necks, the enclosed homes equal at all 68 prefixes). The road lineup on the 60 is re-scored
   on the complex (resistance_lp still leads the medians, L_P 0.194 at D 0.05, and SIMP's default
   is ahead of the best road on all 60, by +0.123 where B (i) ties). pweight, the corridor pictures
   (gap_maps: the flow from the passages' fluxes, Kirchhoff's law guarded) and the block atlas,
   clearings_check and the store (every run keyed by its metric's name), and free first for the
   road methods are on the complex. The translucent search (the tension at eps 0.05) is on the
   frontier at the cheap preset's cost (+0.010 [+0.005, +0.019] over it, 11 of 13; SIMP's default
   still ahead), so it stays selectable and whether the cheap preset adopts it is the owner's call.
   Cross-checks against the raster: the road order is the raster's at Kendall tau +0.83 (+1.00 on
   the 26 blocks where B (i) ties among the methods; elsewhere enclosure decides, the shy distance
   walling in 174 homes behind gaps under 0.5 m that the raster's cells pass: the metric's choice);
   P_{1/2}'s weights rank alike (Spearman +0.99); the corridors agree (the raster's main lanes open
   on the complex but for 0.1 -- 1.0% of their cells, its flow where the complex holds dry 0.7 --
   2.6%: the closed gaps, the zones, its own lattice); each metric ranks its own optimizer's
   clearings first, as in phase 1. **Next**: phase 3, the retirement (that section's list: the
   shared parts of clear and relax moved out, common's raster names, the (b) modules and the mixed
   modules' raster parts deleted, each retired variant to the wiki's dead-end record with its
   numbers); then the region test on the complex, the replay test of epistemetry's pruning
   (Experiment design, below), and the consolidated pass. A lead before that pass: the search's
   weakness on the complex (the tension ranks the buildings sharing a zone at Spearman +0.83
   against +0.97 alone, and the complex's own polish climbs from the raster's clearings; the lumped
   tension's +0.042 on 22422's cheap preset a datum), tried with moves that clear a zone's
   buildings together. **Decided for phase 3 (owner 2026-10-15)**: the finished studies' code (the
   inventory's (c)) is deleted with the raster ("Delete the finished studies' code"), their numbers
   staying in NOTES and their summaries committed; the cheap preset becomes the translucent search
   at eps 0.05 and its plain form is deleted ("Make translucent the cheap preset and delete the
   plain version"). **Phase 3: done** (NOTES, "The complex as the runtime: the raster retired"):
   the raster, its lift and turning cost and the finished studies' code deleted (59 modules;
   research/roadless's Python from 125 modules and 34,899 lines to 67 and 17,826), their rows,
   summaries and pictures kept as records that no code reads (the exporter reads this code's
   metric only); the shared parts moved first (the runtime's Spec from its name, a fixed grammar
   of the raster's names in check_disjoint, the block-id parser, seqscreen's loaders, the
   free-roads guard as free_roads_check.py); seam's Protocols folded into the complex's types,
   the catchment's sweep into Catchment (bit for bit), the relaxation's dead `rtol` gone; the cheap
   preset the translucent search at eps 0.05 (`@e0.05`; the plain form's rows stay under their
   names); an unknown block id raises at the boundary (common.ids_of, write_bank, build_blocks;
   phase 2b's misspelt 7851 refused). Left for the consolidated pass: relax.Plan's `t<rtol>`, which
   moves nothing on the complex (dropping it renames every stored SIMP run), and the finished
   studies' code that reads no raster code (fft_homog's solver and two-scale code, perm_floor,
   atlas/nbhd, the region page's style and vectors). **Next**: the region test on the complex
   (decision (4)), then the consolidated pass.
   (4) **Exits: the region model** (owner: "we'd still only leak at the
   outer boundary, so all current must exit ... some of the outer boundary might not leak at all
   and might have 0 conductance (like a railroad)"; "we can adjust boundary leakage, uniform
   leakage, and also building-boundary leakage"). Everything compiles into conductances: barriers
   (railways, water) at zero except at crossings; holes plain open space; inside, the
   conductance formula of (3), a street boost an option only (the owner hopes it is not needed);
   exits are leaks to ground with three adjustable families: the region's outer boundary where it
   is a real road (none where it is a barrier); uniform (a walker arrives anywhere at a constant
   rate, trips of a typical length lambda; in the small-leak limit P's clearing-dependent part is
   everyone-to-everywhere, each home's term its mean effective resistance to the region); and
   building-boundary (destinations at buildings' rings, by area population, closer to everyone
   reaching everyone). Internal leaks only on open space connected at baseline (B's enclosed homes
   stay enclosed). Why internal leaks: boundary-only escape is the volume-to-point problem, whose
   optima are trees (constructal), so at city scale tendrils reaching in rather than
   interconnection. Per-block exits are the model's fast limit (a one-block region whose whole
   edge leaks), kept for the benchmark suites. **The test: done** (NOTES, "Exits as leaks: the
   region model on the complex (5810@major)"): the exits are three Strategies on the metric
   (gap_exits: barriers, leaks, a street boost), named in a suffix with their own digest
   (`+xB8c3U500va03a092b`), the per-block metric's name and digest unchanged; the leaks a fixed
   conductance on the diagonal (uniform sigma |c| / lam^2, the screened Laplacian, checked to 2.6%;
   building-boundary at the same total), only on cells live at baseline; the railway and water cut
   out of the region but at its crossings (one path inside 5810@major). On 5810@major, both presets
   at D 0.05 under edge, bar and lam 250 -- 1,000 m of both families (16 runs, 10.0 GPU-h; the
   street boost's two configurations cut for the budget, as ruled): the barriers nearly double P_0;
   the uniform leak changes Lens A (0.14 -- 0.28 against bar's 0.31) more than the clearing
   (Jaccard with bar's 0.55 -- 0.91, near-interchangeable under each other's model); the
   building-boundary leak changes the clearing (a quarter to a third into 5810, Jaccard
   0.19 -- 0.60) and leaves 0.04 -- 0.16 to gain; the corridors a clearing induces are a near-tree
   from the edge under edge (4 loops at q 0.9), 10 -- 11 with the barriers, 10 -- 34 with leaks; a
   formal home is 2.5 -- 3.6x as deep as a deep informal one under the boundary and the uniform
   leak (per block 0.02x), 1.1 -- 1.9x under the building-boundary leak; its self-trips small for
   the typical home but 9% population-weighted at 250 m, under 1% from 1,000 m; SIMP's default
   leads the cheap preset under every configuration. **Adopted (owner 2026-10-15)**: the barriers
   ("We can adopt the barriers") and the uniform leak at lam 500 m (the more realistic,
   "especially if you e.g. cleared an entire island so all traffic was internal"). Streets: "more
   conductive, but not make them leak more" (owner), so no street leak; the boost is the streets'
   lever. **Open**: the formal grid takes 89% (SIMP) of U500's clearing, and formal homes stay 2.6x
   as deep; whether the boost moves that (its check at U500, about 1.05 GPU-h), and whether the
   uniform leak's destinations on vacant land are the cause (a baseline diagnostic, cores only).
   For the site: present both families, powers of the escape time (J_p) and priority-weighted P,
   under established names.
5. **Transparency for the road methods** (Looped Tree/Network, Greedy Arterial): discounted
   collision penalties for more demolition and better end-state networks. Judged after 4, at
   equal displacement (a shift of the frontier, not a move along it, which the cost weights
   already make); the promising form discounts the buildings the roadless optimum clears, the
   map R of 4. The sentinel (owner 2026-10-12, "Leave out arterial"; NOTES, "Transparency for
   the road methods: the sentinel"; reblock.transparency, transparency_run.py): a discount moves
   1.5 -- 2.5x as much of the roads' displacement onto SIMP's set, but (c) at area D 0.05 gains
   only +0.006 -- +0.013 for the main forms (4 -- 6% of the gap to SIMP's default, none on gated
   22422), +0.006 -- +0.015 above random sets made as cheap. Through the screen: resistance_lp
   priced at f 0.1 (its price and route, +0.015, and the cheap preset's set beside it) and
   clearance_looped priced and routed at f 0.1 (the cheap preset's set beside it), each with its
   random control. cycle_native stopped on the (a) guard by 0.00015 (a median loss of 0.01015 at
   count D 0.05) and went on to the 13 by the controller's ruling. The 13 tuning blocks (NOTES,
   "Transparency for the road methods: the 13 tuning blocks"; every network proposed on the
   cluster's CPU nodes, cluster.py's `cpu` kind): the discounts move 1.7 -- 2.7x as much
   displacement onto SIMP's set and gain +0.020 -- +0.035 of (c) at area D 0.05 on average (8 --
   13% of the gap to SIMP's default, much of it gates on 46841, 23597 and 1558), above random sets
   made as cheap. Under the rule fixed before the run, **clearance_looped priced and routed on
   SIMP's set at f 0.1 goes on** (+0.0249 [+0.0079, +0.0575], (a) gained), with base, repulsion 4
   and its random control; cycle_native stops on the (a) guard (0.0142); resistance_lp stops below
   its own knob, chunks 4 (+0.0320), which itself leaves population enclosed on 20543: held to the
   candidates' B (i) veto, chunks 16 would be the bar and C priced (+0.0204, 12 of 13 up) would go
   on (**owner's call**: knobs are now held to the B (i) test, and resistance_lp's C priced goes
   on, decided after the 13's results). The 46 held-out blocks (NOTES, "Transparency for the road
   methods: the held-out blocks"; four rules fixed before any held-out result, every block run):
   **neither claim is a shift of the frontier**. clearance_looped's C priced and routed gains
   +0.0341 [+0.0230, +0.0464] of (c) at area D 0.05, beats repulsion 4 (+0.0259) and its random
   sets (+0.0348) and gains (a), but leaves 0.05% of the population enclosed on one block (5706)
   where base leaves none: rule 1 fails. resistance_lp's C priced gains +0.0313 [+0.0171,
   +0.0545], beats chunks 4 and 16 and its random sets, but is behind base on B (i) on four blocks
   and loses a median 0.010033 of (a) at count D 0.05: rules 1 and 4 fail. The gains are broad
   (medians +0.027 and +0.024, 13 -- 16% of the gap to SIMP's default), larger than on the 13;
   what fails is B's first criterion at the D 0.05 cut, on which all but one of the other 13
   discounted and knob arms also wall a few homes in on some block. Free first (owner 2026-10-12,
   "Yes, let's try to free enclosed homes first"; NOTES, "Transparency for the road methods: free
   first"; reblock.free_first, free_roads.FreeEnclosed): every arm of both methods first holds roads
   that free the homes enclosed at baseline (on the ten held-out blocks that have any, 21 homes, 18
   roads, D 0.0009 -- 0.0058 by area a block), the held-out rules unchanged, decided after their
   results. No arm then leaves anyone enclosed at D 0.05, and **clearance_looped's C priced and
   routed is a shift of the frontier** (rules 1 -- 4 hold: +0.0417 [+0.0304, +0.0552] over base,
   +0.0314 over repulsion 4, +0.0348 over its random sets, (a) gained); resistance_lp's C priced
   holds rules 1 -- 3 but fails rule 4 by more than before (a median (a) loss of 0.010978 at count D
   0.05). Free first cost clearance_looped's base most (-0.031 of (c) a touched block);
   resistance_lp's base and claim gained (+0.020, +0.026).
   **TODO** (owner 2026-10-12, "leave a TODO"): make free first the default for the road methods
   in the research lineup (free_roads.FreeEnclosed in place of reblock.free_first.HoldNothing in
   road_rescore's arms), as B's order implies, and re-score item 4's lineup with it. It changes a
   network only on a block with homes enclosed at baseline (12 of the 60 large blocks).

Owner's decisions on the other open items (2026-10-13: "Yes to everything, except vulnerability
can be done after we've done the entire website/README/examples revamp"):
- **Re-scoring waits for the metric.** The free-first default for the road methods (item 5's
  TODO), the oversized blocks and the arterial (item 4), free first for the plain greedy and the
  composite and the small blocks' stored rows (item 3), and the measurement gaps (one-card timing
  of the whole lineup with setup, S0.005cat's real clearing with the polish, the sightline
  lineup) run once, in one consolidated pass under the final metric (item 4's decisions).
- **In that pass, prune the dominated variants**: the speed-quality review's 17 (2026-10-12),
  dominance re-checked under P first, each to the wiki's dead-end record; the add/remove
  search's next rows go with them.
- **Before that pass, the refined solve's false stall**: ZAF.9.1.4_1_7283 (22 homes, under the
  source's floor, so never run before the region test) raises "refinement stalled at a backward
  error of 1.0" though its solution is a direct spsolve's: the componentwise backward error of
  rows whose terms all vanish beside the street (NOTES, "Exits as leaks: the region model on the
  complex (5810@major)", its last list). Any block with such rows would raise in the pass. Fix
  the stopping test, with a guard on 7283 and a fault.
- **The stored clearings made durable**: each clearing's core (preset, block, cleared building
  ids, D, Lens A, enclosed share) committed; rows no longer overwritten in place. Done 2026-10-13
  (NOTES, "The stored clearings, committed"): `research/roadless/clearings/`, 151 runs on 303
  blocks, 3.5 MB in git, read through `clearings.py`, which refuses a block whose footprints no
  longer match their fingerprint; the 12 blocks' pre-B rows the free-first runs overwrote were
  recovered from run road-b's shipped inputs (`D0.05.p2@road-b`). Still to do: the optimizers'
  no-overwrite guard (in the consolidated pass). The scripts that wrote rows without naming their
  metric (study.py, rescore.py, tails.py, the atlas's and fft's exports) went with the raster
  (phase 3), and the exporter reads only this code's metric: the old rows cannot mix with the
  new. The re-export after the priority-weighted P run is done (2026-10-13: its four runs
  on all 220 blocks, the store 61,351 rows).
- **`.superpowers/sdd/`'s reports** (6.6 MB since July, git-ignored): checked for findings not
  recorded elsewhere, those filed, the finished plans' workspaces deleted. Done 2026-10-13: five
  July footpath-conductance findings went to the wiki
  (`pages/reblock/footpath-conductance-from-parcel-geometry.md`), the other fifteen were already
  in git; 6.6 MB down to the three active study workspaces. Three citations now dangle (main's
  OT note -> `transfer-idea-feasibility.md`; `tests/test_permeability.py` -> `task-1-report.md`;
  `conf/permeability.yaml` and `reblock.permeability`'s docstring -> scratch scripts long gone);
  each citing text carries its substance. Fix them with the site revamp: editing
  `reblock.permeability` misses the derive cache of every method that imports it.
- **The vulnerability score** (free first in reverse: the least rubble that would wall a connected
  home in, a min cut): after the site/README/examples revamp. With it (owner 2026-10-14), vehicle
  access read from the gap complex's filtration: the people within a stretcher's carry of where a
  vehicle of width W reaches, over W (item 4, decision 3).

Open:
- **The rate bracket** (seqscreen.RATES, 2e-6 -- 1.9e-5 Lens A per GPU-second). A screen whose
  call turns on the rate prints an "ask" (this much Lens A for this much more time?); the
  owner's answer moves an end of the bracket to that screen's break-even rate. None is live.

Not pursued, with what would change that:
- *A Galerkin coarse model* (the solver's AMG level as the design problem's model, which keeps a
  sub-cell gate's conductance where re-gridding loses it): coarse runs were not cheap enough to pay
  even where they kept gates (`.C0.75.m2` 1.36x, dominated) -- worth it only if a coarse model were
  several times cheaper per update than h 0.75 and kept every gate.
- *The greedy's sweep kernel* (results identical, ~20 -- 30% of the greedy): the greedy is now a
  0.3 -- 0.4x add-on of the hard-block preset; worth it if the greedy becomes the main method.

## How we run experiments

- **Tune on the 13, report on the 44.** The 13 tuning blocks (6 gated collapses, 1558, 6
  controls incl. 5810) were used to choose the SIMP plan; the other 44 of the 56 large blocks
  that fit are untouched. Choose settings on the 13; the claim is the 44.
- **Screen before the full run.** Run a variant first on a small subset (the 13, or a 4-block
  sentinel: 22422 gated, 30848 gated and large, 5810 biggest, 9712 ordinary) and stop it if
  any block collapses (> 0.1 below its baseline) or its median time exceeds 3x the baseline's.
  Only survivors get the 44.
- **One GPU process at a time** (large blocks), queued in a detached script.
- **Sequential stopping** (done 2026-10-09). seqscreen.py: the Bayesian bootstrap's posterior
  on a variant's mean paired difference, a stop when no number of further blocks is worth its
  GPU time (the exact value of more blocks through the posterior mean's Polya urn), a 15-block
  floor and the collapse veto. kernelcore's reweighting collapses on these screens (NOTES).

## Greedy

- **Exchange refinement**: now a special case of the add/remove search below.
- **Batch size** (done: S0.005cat on the 44, +0.003 median, ahead 82%, 1.6x time; NOTES). Sentinel screen, Lens A at D 0.10, S0.005 /
  S0.01 / S0.02: 30848 0.776 / 0.770 / 0.750, 5810 0.329 / 0.319 / 0.306, 9712 0.181 / 0.176 /
  0.170, 22422 0.815 / 0.814 / -; monotone, smaller is better at 1.3 -- 3x the time; 0.02
  dropped from the full run.
- **Screening width** (idea). Exact-score more top candidates per round (as `M4` does).
- **Speed** (measured 2026-10-04, gramprobe.py and a timing probe; NOTES "Greedy speed"). A step
  on 30848 is 8 -- 12 s: the ranking (two solves at rtol 1e-3) 3 -- 4.5 s, the gram 1 -- 7 s,
  scoring 2 -- 6 s.
  - Scoring's rtol 1e-5 -> 1e-3 moves perm by at most 3.5e-5 and cannot change a pick (the
    picker ignores the scoring J). It is 5 -- 10% of a step: take it with the next change.
  - The gram scales with candidates x unknowns, about 0.03 s per candidate on 12 -- 16M
    unknowns. The waves do not matter (Kahn's pass is 0.1 -- 0.25 s for 3 -- 7K waves), so
    batching wave launches is dead.
  - The levers left: fewer candidates (the picker's `reach`, which changes picks), or a sweep
    kernel that reads each node's neighbour indices once rather than once per candidate column
    (results identical; GPU kernel work).
  - Altogether perhaps 20 -- 30% off the greedy, not 2x.

## SIMP

- **Cheaper incumbent** (done: `.k1s1e-06` on the 44 equals `.k1`, 15% faster; NOTES). Candidates scored in the eps world
  at eps 1e-6 (the relaxation's cached structure), the winner exactly. Sentinel: the same
  scores (0.8144 / 0.7822 / 0.3476 / 0.1871 vs 0.814 / 0.782 / 0.348 / 0.187), 10 -- 25%
  faster. Every 2nd iterate (`.k2`) is dead: it collapses on 22422, whose gate a single odd
  iterate holds (NOTES, "SIMP: every 2nd candidate, coarse-to-fine, damped OC, MMA").
- **Warm nested path** (measured, `.w3`: 2.6 -- 2.9x faster, Lens A 0.002 -- 0.022 below the
  cold path on 3 blocks; a cheaper preset; NOTES).
- **Memory** (partly done, 63fa484; 2026-10-04): structures are cached only for a pattern seen
  twice, and the greedy releases its tension before the exact scoring (22422 fits again); 1558,
  20023, 30796 fit under `.k1s1e-06`. Fixed 2026-10-04: the K-cycle's closures formed a reference
  cycle that kept every system's AMG hierarchy (2.6 GB on 30796) until the cycle collector ran,
  about 11 GB of dead hierarchies at the peak (NOTES, "Translucent greedy on 30796"). With a
  release of cupy's cached blocks before each new system, 14401, 7851 and 32841 now fit 48 GB for
  the greedy and SIMP (62 of 82 large blocks). Done: 19593 and 6498 on the H100; 53556, 45267 and
  the other 16 through local coarsening (Status, above).
- **Damped OC or MMA** (done 2026-10-04, null; NOTES). Damped OC ties or trails slightly on the
  13 (median -0.0005 / -0.0001), and MMA collapses on 22422. Both stay selectable (`.u`).
- **Sturdier gate capture** (from that null). On a gated block SIMP's answer is one lucky
  iterate's rounding: `.k2` and MMA each lost 22422 by perturbing the iterate sequence.
  - Several roundings per iterate (done, `.r<n>`, NOTES "SIMP multi-rounding"): `.r2` on the 13
    is ahead on 5 (+0.008 to +0.038) but on the 46 held out only +0.001 mean (none above 0.01),
    never behind, 2x the time: selectable for hard blocks, not a default. It partly rescues
    `.k2`'s and MMA's lost gate (0.55, not 0.78).
  - Cheaper `.r` (done): the winners come from the last two stages' random roundings, so
    `.r2l2` samples only there (1.58x, same gains as `.r2`).
  - Undecided-set polish (done, `.p<tries>w<width>`; NOTES "Late sampling and the polish"), and
    what followed it, all done: the larger cap `.p256w8`, the pair moves `x2` (together the
    default `.p256w8x2`), and the same polish on the greedy's clearing (the hard-block preset's
    second track `.GS0.01cate1`, and polish_greedy.py's cheap preset).
- **Multiple starts** (closed, NOTES "Multi-start on coarse grids"). A Frank-Wolfe start (fw5)
  lands where the uniform start does; a second, random start (`.p64w8.m2`) never beat the uniform
  one (identical on 12 of 13, 2.12x); coarse multi-start `.C<h>.m<k>` neither faster nor better.
- **Projection** (measured, kept selectable: `.b8-32` wins some gated blocks, loses some
  controls; not needed against collapses once the incumbent is on).
- **SIMP + incumbent under the sightline metric** (done: the 13 at D 0.10 and D 0.05, and the 46
  held-out blocks at D 0.05, NOTES "Sightline on the held-out blocks at D 0.05". SIMP leads the
  plain greedy by a median of +0.021 [+0.015, +0.026], ahead on 96% of blocks, and the translucent
  greedy by +0.005 [+0.002, +0.012] (all 46, with 30796 after the memory fix), at 3.1x their
  time. The translucent greedy keeps its gate tail, 18910 -0.165.)

## New methods

- **Add/remove subset search** (built 2026-10-09; NOTES "The add/remove search as one
  Strategy"). search.py: a Round (rank, refine, build, evaluate, accept) under a schedule; the
  greedy's pickers, exchange (the polish), grow-then-prune and floating search are its
  configurations. Floating search (FL3xS0.01catr0.005m8c100) is never below grow-then-prune or
  the greedy on the five blocks and repairs the prune's collapse (22422 at D 0.05: 0.782 against
  0.011), at 1.1 -- 2.2x its time. At one budget (d_max 0.05, the first live screen: NOTES,
  "Floating search at the cheap preset's budget") it is dominated by SIMP's default (-0.011 on
  15 held-out blocks at about 10x the time) and by swings (below); its niche is the whole curve
  in one run. Next rows, two new parts each (spec, "What the two new rows need"): a
  substitute-aware restore (a restore ranking that keeps the Tension, a restore-direction
  diverse builder) and screened adds (an add-direction refiner, a builder that reads the
  refined order). Wiki `pages/methods/plus-l-take-away-r.md`.
- **Swing schedules** (owner 2026-10-09; tested 2026-10-10, NOTES "Swing schedules at the
  cheap preset's budget" and "Swings as SIMP's finisher"). Several grow-then-prune swings past
  the budget, each smaller, then the polish (search.py
  `SW<grow>-<grow>-...x<picker>r<share>m<k>.P<t>w<w>`; from SIMP's answer `SIMP<plan>+SW...`).
  From the greedy they are dominated by SIMP's default (-0.005 at 3x its time; a 3x first swing
  ties 1.5x at 3x the time). As SIMP's finisher (one 1.2x swing and the polish) they are on the
  frontier: +0.0010 mean over SIMP's default on the 46 held out (half of it 30796, +0.024),
  never behind, at SIMP's time again; on the 13 tuning blocks +0.0001. Not a preset: the gain
  is one block's (30796), and no gate on SIMP's answer finds it (NOTES). The typical gain is a
  re-pick of a few buildings worth 0.8% more budget. What would change that: blocks like 30796
  common among the large ones (it is the only held-out block over 3000 buildings). The adaptive
  form waits on a schedule worth adapting. Neighbours: oscillating search, VNS, adaptive LNS
  (wiki page above, "Swing schedules").
- **The restore screen's solve on gated blocks** (2026-10-10, NOTES "Swings as SIMP's
  finisher"). Its eps world (EPS_SCREEN 1e-6) did not converge on 30848 (AMG-CG 2000 iterations,
  1.9e-5 -- 5.0e-5 against 1e-5); every restore round can meet it. Fixes to weigh: warm starts
  from the current state's solution, a looser rtol for a screen that only ranks, a larger
  screen eps. The exact score can meet it too on a fine grid: 38616 at h 0.25, uncleared, 4.9e-8
  after 2000 (NOTES, "The metric's resolution floor").

## Experiment design

- **Pruning by epistemetry's decision layer** (owner 2026-10-14: "Yes"; replayed 2026-10-15,
  NOTES "Pruning by epistemetry: a replay against seqscreen"): **seqscreen stays** for the
  consolidated pass. epireplay.py (deleted in phase 3 with the raster's rows it read; its rows
  stay) ran epistemetry's ValueOfMoreUnits on the finite-population
  bootstrap (c1784af, an overlay, no dependency) against seqscreen.py on 13 pairwise and 5
  many-arm finished designs, 100 orders each. Under the amended rule (loss below at every rate, no
  more wrong calls, no more asks) its adopt/keep/ask over the bracket fails in both veto settings:
  loss above at all three rates, 312 asks against 280, though 36 wrong calls against 66; the
  many-arm screen runs every arm to the end; the original rule's fixed-rate three-way fails too.
  Would come back with: re-deciding after every block (a diagnostic closes most of the gap), a
  linear-loss target over the bracket, a per-arm drop for many-arm screens (feedback sent to its
  author).
- **Adaptive block selection** (first look done, NOTES "Which blocks tell methods apart": collapses dominate; screen = gated sentinel + random sample; picking 'discriminating' blocks beyond the sentinel misleads. Remaining value: sequential sample size; wiki `pages/methods/adaptive-benchmark-items.md`).
  Fit a factor model to the method x block matrix of paired differences (axes like "opens
  gated pockets", "spreads in dense bands"); for a new variant run first the blocks with the
  most expected information per GPU-second about the decision, stop when it is made. The
  headline number still comes from the untouched set; keep some random blocks in every
  screen. Cheap first step: an SVD of the matrix we already have (greedy, SIMP variants,
  translucent search; 220 + 59 blocks) to see which blocks load on which axis.
- *Generalize kernelcore's outcome model* (dropped 2026-10-09). The screens' stop is built in
  seqscreen.py; reweighting draws collapses past about 4 blocks and a prior grid decides the stop
  (NOTES, "Stopping by the value of more blocks"). Would come back with a consumer that must
  choose among blocks that are not exchangeable (the adaptive selection above), designed against
  it.

## Coverage and model

- **Blocks over ~1.4 km^2** (done 2026-10-08): local coarsening (`lifted.CompositeGrid`, plan
  suffix `.a<d0>x<smax>`) keeps h 0.5 within d0 of every building, street and the block edge and
  doubles the cell with each doubling of distance; within 0.004 of uniform h 0.5 on the 59 at a5
  (order kept but for ties), though 14401 reads 0.009 low at a5 (0.0014 at a20: the order still
  kept), and 2.4 -- 5.1M cells on the oversized blocks. Also an idea on top of it: a separate d0
  for streets and the block edge.
- **Goal-oriented refinement** (**prioritized**). Refine where the dual-weighted residual (the
  adjoint, which the tension already solves for, weighting each cell's residual) says Lens A is
  sensitive, instead of by distance: 14401 reads 0.009 low at a5 where the 59 read at most 0.004,
  so distance alone misses some of what the score depends on. Measure: Lens A error against
  uniform h 0.5 per cell spent, on the 59 and 14401, against a5 and a10. Built (common.GoalMesh);
  with nothing cleared in the indicator's world it lost to distance on a third of the 59 (NOTES);
  with the buildings open (o1) it is ahead of distance on 49 of 53 blocks at f1.2 (2.8x less
  error at equal cells, median; 0.11 of the uniform cells fewer at equal error). Its five losers
  are two sign crossings of the distance meshes' bias (20952, 44602: a5 near zero by luck), a tie
  and two mild losses (NOTES). On the 19 oversized blocks at a5's cells it leaves a sixth of a5's
  error (median 4.9x less, ahead on 18, the 19th a tie), against the finest goal mesh that fits.
- **Sightline and the greedy on the composite** (**prioritized**). Sightline and SoftSightline
  scan raster lines (`needs_raster`), as the metric or as the greedy's translucent search; both
  raise on a composite. The greedy under uni has no raster check but has never run on one: first
  establish that it does (against uniform h 0.5 on the 59), then give the sightline scans a
  composite form. Both done (NOTES, "Sightline on the composite"): the scans run on the fine
  raster, painted and averaged per cell; under the sightline metric on the 59 every mesh reads as
  under uni (a20 p95 |error| 0.0006), the goal mesh ahead of distance on 38 of 40 but by less
  (1.9x, 0.05 of the cells; 8152 0.18x, not looked at). Next: the sightline presets on the
  oversized blocks.
- **The cheap preset on the composite** (**prioritized**). polish_greedy.py builds a uniform h 0.5
  grid; it takes a mesh, and greedy + polish runs on the oversized blocks. Needs the greedy on the
  composite (above).
- **Coarse-to-fine** (`.c1`, tried 2026-10-04; NOTES). 0.75x the time, but it collapses on 30848
  (-0.255). Kept selectable. Worth retrying only with a coarse h that keeps 30848's gate (0.75?)
  or with only the first stage coarse.

## Infrastructure

- **Faster submits** (done 2026-10-08; NOTES "Faster submits"): a staged `dwithin` near test, a
  count without the cells' attributes, 8 blocks counted at a time; sizing 16 -- 18x faster, the
  meshes identical. A goal-oriented mesh, whose cells come from a solve, will need its own
  `cells` (its cell budget, say) rather than a build at submit.
- **Cluster launcher: bookgen and mycooc on GeoffChurch/cluster_submit** (done): mycooc deleted
  its old launcher (3ad93ac5), bookgen deleted ltcluster with no compatibility path (b901184b).
- **CPU runs on orion and quadro2** (built 2026-10-12, for BACKLOG 5's 13 tuning blocks; owner:
  "run the 13- and 60-block stages on orion/quadro2"): cluster.py's `cpu` kind, one task per
  (block, item) of `--items` with `{id}` and `{item}` in the command, every task CPU only (no GPU
  sizing, no GPU request), pinned to the node `--node` names (required: orion or quadro2,
  CPU_NODES, the nodes whose GPUs are unusable), the cores and memory of a task required too;
  `--gpus`, cluster_submit's throttle, at most 64 at once on orion and 40 on quadro2. A run of CPU
  tasks is no longer held to the project's 4 GPUs' worth of tasks (GeoffChurch/cluster_submit
  e299a2b; the pin bumped past its site files, `load_cluster("ai")`). A block's tasks unpack its
  tar at once into the checkout they share, so each goes through cluster_unpack.py (every file
  renamed into place); road_rescore.py writes what a task returns whole (a temp name, then a
  rename), as a run's pull takes `*.parquet` while tasks still run. Guards: cluster_check.py
  `cpu` and `gpu` (fault-injected), and a smoke run on orion. The partition allocates whole
  cores (CR_CORE_MEMORY), so a 1-cpu task holds two CPUs and at most 36 run at once on orion, 32
  on quadro2; the 13 tuning blocks' 481 networks and their prefixes took 28 task-h, each
  network identical to its twin computed here, where one was (NOTES); the 46 held-out blocks'
  782, 22.7 task-h.
- **GPU sizing recalibrated** (done 2026-10-12): cluster.py's GB_PER_MILLION 0.7 -> 0.95, from
  the scorer's measured peaks on the 13 (0.83 -- 0.87 GB per million unknowns on the large
  blocks; at 0.7 item 4's run had sent 1558, 14401 and 20023 to 32 GB cards they would not fit).
  cluster_check.py `gpu <run> <rows>` checks a recorded scoring run's measured peaks against the
  sizing (every card holds its block's peak with 5% to spare; fault-injected at 0.7 and 0.9);
  transp-held-score passed it. SIMP and the greedies peak at ~0.65, so they get room to spare.
