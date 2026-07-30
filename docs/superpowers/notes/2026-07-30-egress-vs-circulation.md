# Egress vs circulation: what permeability grounds against, and the all-pairs alternative

Prompted by a question about an example image -- why parcels beside dense footpaths read deep red --
which turned out to be correct rendering sitting on top of a real modelling question.

## 1. What ground is today

`permeability.egress_power` grounds a parcel when its polygon is within `STREET_TOL` of
**`block.streets`**. Two consequences that are easy to get wrong:

- **The method's own added roads never become ground.** They upgrade edge conductance
  (footpath -> road) along their corridor and nothing more. A proposed road gives parcels a better
  route to an existing street; it does not itself become an exit.
- **"Ground = the block boundary" is true for a single block and FALSE for a region.**
  `region_block` unions every member's streets -- "perimeter + inter-block = the full existing road
  network". Measured on `multiblock_density_compactness`, 18 blocks / 4,615 parcels:

  | | parcels |
  |---|---|
  | grounded (within tol of `block.streets`) | 1,059 |
  | ...on the outer boundary | 389 |
  | ...grounded by **interior** inter-block street | **670** |

  Nearly two thirds of ground is interior. This is deliberate -- routing on the full existing
  network means already-served parcels stay served, so a method adds only complementary roads -- but
  it means region-scale intuitions built on "the boundary" are wrong.

## 2. The disagreement between the two reported quantities

| | a road component that never reaches a street |
|---|---|
| `parcel_access_layers` (access depth) | grants **no** access -- `street_connectivity` seeds only components touching a street |
| `permeability` | **still lowers dissipated power** -- the corridor upgrades local adjacency conductance |

Measured, `osm_footpaths` lens-B prefix on `density_compactness`: **14 road components, 70.8% of
length street-connected**, and **all 136** adjacent-but-still-deep parcels touch only floating
fragments. So that method "reaches P* = 0.60" partly on road granting zero access.

### The obvious fix is backwards

The first instinct -- make permeability strict, drop conductance on ungrounded components -- is
wrong. A footpath linking three interior parcels genuinely helps you move: you use it to reach a
parcel that IS near a street, and you still pay footpath resistance for the last leg. That is
exactly what a resistance model represents and what a binary frontage ring does not.

**Permeability is the better model of the two.** Access depth is the cruder abstraction, and it is
the one that would need changing if consistency were the goal. Consistency is probably not the goal:
they answer different questions and both are reported.

Two changes are worth making regardless, and neither touches a metric:

- report **connected fraction** beside each lens row, so a number resting on floating road is visible
- render floating segments **dashed or paler**, so the picture explains itself

## 3. The alternative worth considering: all-pairs

Today's `b = ones(n)` is an **egress** model: every parcel injects one unit of current, all of it
flowing to the street. The natural alternative is every parcel wanting to reach every *other* parcel
-- **total effective resistance**, the Kirchhoff index:

    R_tot = sum_{i<j} R_ij = n * trace(L^+)

on the ungrounded Laplacian's pseudoinverse.

| | egress (shipped) | all-pairs |
|---|---|---|
| model | all current -> street | every pair exchanges current |
| needs a ground | yes | **no** |
| rewards | getting OUT | getting AROUND |
| floating road linking A-B-C | helps slightly | helps properly, and correctly does NOT help them reach the street |
| convex in conductances | yes | yes -- Ghosh-Boyd-Saberi is stated for this |
| monotone under an added road | yes (Rayleigh) | yes (Rayleigh) |
| cost | one sparse solve | `trace(L^+)`: all eigenvalues, n solves, or an estimator |

### Why it is attractive

1. **It is literally the Ghosh-Boyd-Saberi objective**, so the convexity the route-(A) LP relies on
   (`notes/2026-07-29-lp-route-a.md`) applies natively instead of by analogy.
2. **No ground at all**, which dissolves section 1 entirely -- including the block-vs-region
   asymmetry and the "do added roads count as street" question.
3. **It sees internal circulation**, which nothing shipped does. The "Bermuda triangle" livability
   concern in the backlog is exactly a circulation failure that egress cannot detect: a pocket can
   have fine egress and terrible internal connectivity.

### Why it is not obviously right

- **Pure all-pairs has a clear failure mode**: a settlement where everyone reaches everyone but
  nobody reaches the arterial road scores perfectly. Egress is not a detail to be dropped.
- So the honest form is a **combination** of egress and circulation -- which reintroduces a weighting
  question of exactly the kind deliberately refused for `homes + lambda*metres` in the lens work.
  That refusal was on the grounds that a weight is a values question, not a measurement one; the same
  objection applies here and deserves the same answer rather than a quietly chosen constant.
- **Cost.** Exact `trace(L^+)` is tractable at n ~ 4,600 and not at n ~ 11,000, where it wants a
  Spielman-Srivastava / Hutchinson estimator -- which puts sampling error inside a reported metric.
  That is a materially different contract from today's exact single solve.
- It would **change every published permeability number**, and the entire method comparison rests on
  the current one.

### Suggested route

Brainstorm -> spec, not a patch, starting from `specs/2026-07-22-permeability-metric-design.md`
(which recorded why the single flow metric replaced external + internal connectivity + commute
ratio -- the circulation question this reopens is precisely what "internal connectivity" used to
try to capture, and it was retired for being gameable). Any proposal should say what it does about
the gaming failure modes that killed the previous internal-connectivity metrics, since an all-pairs
term is a circulation measure and inherits that history.

A cheap first probe before committing to anything: compute `R_tot` alongside `P` on the existing
benchmark blocks and see whether it RANKS methods differently. If the ordering is the same, the
extra machinery buys nothing and the question closes cheaply.
