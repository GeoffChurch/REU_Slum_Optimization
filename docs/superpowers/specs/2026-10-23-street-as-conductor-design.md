# Trips inside the block: the street as a conductor

Branch `research/roadless`, a study on block ZAF.9.3.1_1_5810; code under `.superpowers/sdd/2026-10-22-flow-roads/`, not committed.

## Why

The owner asked for a single objective whose optimum has loops without an arbitrary second load: "if you could have a single point be GND, but nobody knows which point it will be, only the probability distribution (maybe equal probability for every point in the interior of every building, and some other probability for the traversible points on the boundary of the region)".
A first test (NOTES, "Trips with destinations inside the block, scored with the street grounded") kept the street grounded in every realization.
That makes the street a perfect conductor: an interior trip goes out to the street and back in at no cost, so a tree whose branches meet the street already has loops through it.
Interior destinations then rewarded the loop connectors less than P does (clearance_looped's lead on the tree 0.020 against 0.096).
The test cannot tell "loops do not pay for interior trips" from "the model hands every tree free loops"; the street has to conduct at a finite rate.

## The objective

A trip's destination is the street (the block's way out) with probability `p`, and otherwise a building drawn by its area, a uniform point of the built interior.
Homes inject their population, here exactly proportional to building area, so the origins' law equals the interior destinations' law, `pi = q / Q`.

- **Egress** (probability `p`): today's operator, the street faces grounded; its part is P (everyone at once: contention) or D, the area-weighted lone resistance to the street.
- **Interior** (probability `1 - p`): the street conducts; one point, the destination `g`, is ground.
  With `H` the conducting-street operator's Green's function under any gauge, everyone at once to `g` dissipates `(q - Q e_g)^T H (q - Q e_g)`, and its expectation over `g ~ pi` is exactly `(Q^2 / 2) K`, with `K = E_{b, c ~ pi}[R_bc] = 2 (sum_b pi_b H_bb - pi^T H pi)`, the mean effective resistance between two random points of the built fabric.

So for interior trips the owner's everyone-to-one-random-point model and independent trips are the same functional: the crowding at the destination and the shared start average out to the pair resistances.
The two readings differ only in the egress part:

    all to one    E = p P + (1 - p) (Q^2 / 2) K
    independent   J = p D + (1 - p) K

`K` is the Kirchhoff-type index of the network weighted by the built area: convex in the conductances (each `R_bc` is), monotone (Rayleigh), and the classical objective of Ghosh, Boyd and Saberi (2008) for minimizing total effective resistance.

## The street as a 1D conductor

Today the mesh stops at the street band's inner line (`band_m` 1 m inside the street line), and each street face conducts its cell to ground (`FullMesh.fa`, `Tf`; a cell whose centre lies beyond its face is itself ground).
Meshing the band would change every mesh and every cache key.
Instead the street becomes a graph of nodes on the street centrelines, and only the faces' far end moves:

- **Nodes:** `geo.streets`' lines, noded at their crossings and cut every `h` metres; consecutive nodes on a line are joined with conductance `sigma f w / h`.
  `w` is the street's walking width and `f` its factor: the boost's own (`StreetBoost`: its factor where an OSM way of its classes within BOOST_M is tagged paved, else 1).
- **Faces:** each street face joins its cell to the nearest street node, through `Tf` in series with the band's own resistance (`band_m / (sigma f |face|)`).
  A ground cell is merged into its street node.
  Both sides of a street inside the block meet on its centreline, so crossing it is free, as it should be at this scale.
- **Demand:** a home's points in the band, which today escape at once, inject into their nearest street node.
- **Ground:** none but the gauge.
  RUNTIME's internal leak (`Lo6214b562`, `gap_exits.InternalLeak`: every cell's conductance to ground, 90% of it at buildings) is left out: an interior trip's destination is its one ground, and a leak would carry current from near one home to near another through the ground, a free connector as the grounded street was.
  One street node is the gauge (every probe injects zero in total), and only the homes on its component take part, the set fixed at the baseline so every state scores the same homes (a sealed pocket's homes, reached only through the leak, have no interior trips).
  A first run kept the leak as the only ground, after joining each leaking cell to a street node (2.86 million long edges) had wrecked the nested dissection's fill; its loops' gain was 0.017 and is superseded.
- **Not modelled:** walking beyond the street into the neighbouring blocks; a region's leaks; vehicles.

Assembly is the existing cells' passages (`Reach.W`) and faces (`Reach.Gf`) relabelled onto cells plus street nodes, factored by the run's `gap_factor.SuperLU` on a pattern with the nodes' coordinates (its nested dissection, its refinement).
`K` needs `sum_b pi_b H_bb` by Hutch++ (192 solves, the same probes in every state, as in `trips.py`) and one solve for `H pi`.

## Parameters

Required and in the run's identity: `w` (6 m, a residential street's), `h` (2 m), and the street factor's source (the RUNTIME boost).
`p` is a scoring choice, swept 1, 0.75, 0.5, 0.25, 0 on the same solves.

## Checks, each with its fault

- **The street-is-ground limit:** with the street factor times 1e6, `K` must equal the grounded test's `2 D - 2 P / Q^2` within Hutch++'s error (fault: the faces mapped to the wrong node).
- **Hutch++ against exact** on the baseline, one solve per home, as for D.
- **h:** `K` at `h` 1 m and 2 m within 1% (fault: a node spacing far coarser than the faces').

## The test

The eight networks scored in `trips.py` (clearance, clearance_looped, cycle_native canonical and in build order, ΔP-scored cycle_native with the legs anywhere and 7 m apart, plain cycle_native 7 m apart, ΔP greedy top 12), at area D 0.05, 0.10 and 0.15, each against its own mesh's baseline.

- **Prediction (argued):** with the street conducting, `K` rewards loops: clearance_looped's lead on the tree under `K` is above its 0.020 with the street grounded, and the loop networks gain on the tree networks as `p` falls.
- **Falsified if** clearance_looped and clearance tie under `K` at matched D within twice the standard error: then interior access, too, does not ask for loops on this block.
- Reported with the street factor both as the boost gives it and at 1 (unpaved), since most of 5810's lanes are tagged unpaved and the answer may turn on it.

## Outcome (2026-10-23)

The checks passed on the leak-free operator (Hutch++ 4.327 ± 0.011 against 4.318 exact; `h` 0.5%; every home with demand on the gauge's component).
clearance_looped led the tree under `K` by 0.022 and 0.019 at area D 0.10 and 0.15 (14% and 9% of the tree's own gain), and by 0.014 and 0.011 with the boost at 3 instead of 10: loops pay a little for interior trips, and neither the conducting street, nor leaving out the leak, nor a lower contrast makes it more, so the prediction failed and the tie criterion was not met.
Setting the street's conductance along it to 0 split the block's open ground into 897 pieces, so it is no counterfactual.
Numbers in NOTES, "Trips inside the block with the street conducting: loops pay a little".
