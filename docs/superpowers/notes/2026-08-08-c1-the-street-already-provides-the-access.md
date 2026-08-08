# C1: the existing street already provides the access a coverage term would measure (2026-08-08)

Kill gate before any spec, same discipline as A5/B1/B2. `scratchpad/coverage/c1_gate.py`.

Tests the idea that came out of
[the road/walk ratio note](2026-08-08-the-road-walk-ratio-has-no-physical-support.md): if a linear
conductance cannot express "a 6 m street admits an ambulance and a 1.4 m alley admits none", add a
benefit term with a width threshold -- the share of dwellings within D metres, **through free
space**, of a road at least W wide.

**It fails, and the control is what decided it.**

## W is inert today, so this tests the load-bearing half

`DEFAULT_ROAD_WIDTH_M == min_road_width_m == 7.0`, and every method emits exactly that (the only
two conf overrides also say 7.0). So every road clears any threshold <= 7 m and the width filter
selects nothing. Width becomes live only once methods choose widths, which is the width solver,
measured not to pay and now deleted.

What the gate *can* test is the half that matters regardless: does LOCAL coverage rank methods
differently from DILUTED flow? If not, no amount of width variation rescues it.

## The control

Coverage from the existing street alone, against coverage with each method's Lens A prefix added:

    D (m)   street only   with roads     added
       25         0.827        0.970    +0.143
       50         1.000        1.000    +0.000
       75         1.000        1.000    +0.000
      100         1.000        1.000    +0.000

**The existing street already puts every dwelling within 50 m.** The roads buy nothing, so the term
has no headroom to discriminate in. Only at D = 25 m is there any -- and there the ranking is
unstable: minimum per-block Kendall tau **-0.756**, with the winner changing on **6 of 10 blocks**.

That is B2's disease transplanted from `sigma` into `D`. The parameter is informative only in the
narrow window where it is also unstable, and stable only where it carries no signal.

    G1 discrimination      tau +0.270, defined on only 4/10 blocks    weak PASS
    G2 D-sensitivity       min tau -0.756, 6 winner changes           FAIL
    G3 not-a-length-proxy  spearman(coverage, road length) +0.020     PASS

## Two corrections to my own gate

Both made after the first run, and both made the verdict WORSE rather than better -- which is the
test of whether a post-hoc change is honest:

* **No control at all in the first version.** It compared an ABSOLUTE coverage against
  permeability's ratio-to-baseline, which is not like for like. Adding the street-only baseline is
  what exposed the saturation.
* **G1 printed "PASS" on a NaN.** At D = 50 coverage is constant across methods, so every rank ties
  and Kendall tau is undefined. Undefined is not a pass. It now reports as undefined, with the
  count of blocks where it is defined.

## Limitation -- stated, not acted on

These are B2's blocks, <= 150 parcels, inherited for comparability. **A reach metric saturates on
small blocks by construction**, and the deep blocks are where methods are already known to separate
(`voronoi-adjacency-partition-method`, the moderate/deep flagship split). So the sample is close to
the worst possible one for this particular question, and I should have seen that before running
rather than after.

But re-running on a different sample AFTER a FAIL is post-hoc, and this project has already been
burned twice by criteria that moved once the result was visible. Whether to retest on deep blocks is
the owner's call, not a correction to make unilaterally.

## What it means either way

Two metric ideas have now failed for the same shape of reason. Permeability's ordering depends on
`sigma_road/sigma_walk`, which has no defensible value in its stable band. Coverage's ordering
depends on `D`, which is only informative where it is unstable. In both cases the metric is
well-behaved exactly where it says nothing, and says something exactly where it is arbitrary.

There is also an uncomfortable reading of the control that does not depend on the gate at all: at
Lens A's matched displacement, on these blocks, **the access problem is already solved before any
method builds anything.** If that generalizes beyond small blocks, the comparison may be running on
fabric that has no access deficit large enough to separate methods -- which would be a finding about
the block sample, not about the metric.
