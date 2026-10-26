# Roads from the vote: a plan

The owner's idea (2026-10-25): clear some buildings, hypothetically or for real, solve the walking flow, and lay roads where the current is high, the current being a vote for where people move.
Current is conserved, so its ridges run as tributaries into the street and the roads it draws are connected by construction; power, the first-order value of paving a strip, is kept for choosing and ordering.
The vote recomputed as roads go in is the adaptive transport network (Hu and Cai; Bohn and Magnasco; Tero et al.): conductance grown toward |current|^(2/(1+gamma)) under a material budget, trees for gamma < 1 and loops for gamma > 1.
Context in research/roadless/NOTES.md: "Roads from the flow, frozen", "ΔP greedy: a faster step, and a screen that sees through buildings", "A road-placing SIMP on 5810 and 20954", "The damage term for vehicles".

## The design space, on block ZAF.9.3.1_1_5810

| axis | values |
|---|---|
| vote source | the original block (no clearing); U's clearing at D 0.05 (metrics.UNIFORM, `.f0`, under P); U under P_{1/2} (PriorityDemand(0.5), convex like P; one SIMP run to make) |
| clearing | real (demolished: in the scored state and in D); a device (the roads alone are built and counted) |
| signal | current density (gap_maps flow `jt`); power density (the frozen pilot's), as the control |
| dynamics | frozen ridges in vote order; periodic recompute every D step; adaptive iteration at gamma 0.5, 1, 1.5 |
| cost | the greedy branch: displacement plus repulsion (lam 0 and 5e-6), no length price, km reported; the adaptive branch: its material budget |
| lenses | L_P, L_{P_1/2}, J_2 (under RUNTIME), clutter (clutter.py), vehicle damage (damage.py), km, each at equal D |

## Stages

1. **Current ridges, frozen and periodic** (research/roadless/vote_roads.py, committed).
   - The vote: the cleared state's flow under RUNTIME (Model state with the clearing's k), current density per triangle on open ground painted on the block's BlockRaster at 1 m (power density for the control), ridges by betweenness.ridges at the pilot's quantiles (0.80 .. 0.98).
   - Roads from ridges: each quantile's skeleton merged into lines, linked to the street (road_simp.linked's rule), 7 m wide; order by quantile (the highest first: trunks), then by mean signal along the line.
   - Periodic: build to the next D step (0.01), solve again with the roads carved (the clearing kept, as the vote's world; the device case also with it dropped), the next ridges off the built roads, repeat to D 0.15.
   - Scoring: one carved mesh per network with its nested prefixes at the area D targets (own_mesh.py's method); a real clearing in the scored state (gap_clear.Model gets a public state of a clearing with a road prefix; `_state` already takes both) and in D (the clearing's buildings and the corridors together under common.MEASURE).
2. **The adaptive iteration** (road_simp.py's raster and answers).
   - An Update Strategy beside OC: Adaptive(gamma), the design set to |j|^(2/(1+gamma)) through the filter, rescaled to the material budget (Plan's D and per_km as now), damped; the clearing applied in the eps world as the starting state.
   - gamma 0.5, 1, 1.5, each from the original block and from U's clearing; answers as now (paved; roads linked to the street, rev 6).
3. **The greedy stage, chosen by epistemetry** (dogfood, owner 2026-10-25).
   - Candidate pools: the stage 1 and 2 networks' roads, each as a `simp:`-like pool beside the lineup's (dp_greedy.py's Pool Strategy gains one per source).
   - Which pools to build: GeoffChurch/epistemetry's `models` (a latent model over the declared factors: vote source, clearing, signal, dynamics, gamma) fitted on the stage 1 and 2 lenses as the prior for each pool's greedy result, and `design.decide` (GittinsIndex or kg_star) picking the next build at its cost (about 2 h each), stopping by its rule.
4. **Across blocks.** The survivors against the base (the lazy ΔP greedy, and cycle_native) through epistemetry's screen (ValueOfMoreUnits, AdoptKeepAskByStake), one screen per lens with L_P first.

## Global constraints

- Result-changing parameters required, keyword-only, in every run's name; Strategies resolved at the CLI.
- Every new check has a fault that makes it fail.
- At most 8 local processes in all; single-threaded pools; memory capped by systemd-run scopes; cluster runs through research/roadless/cluster.py.
- NOTES.md and BACKLOG.md wrapped at 99 -- 100 columns; this file one sentence per line.
