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

## The sample is the TARGET fabric -- and that makes the result structural

An earlier version of this note called the sample "close to the worst possible one" and suggested
retesting on deep blocks. **That was wrong on the facts.** Measured:

    block                parcels   area_m2  perim_m      n/P^2     /km2   depth
    ZAF.9.3.1_1_19362         50     8,121    357.7   3.91e-4    6,157       3
    ZAF.9.3.1_1_20571         64     7,994    381.7   4.39e-4    8,006       4
    ZAF.9.3.1_1_38870         71     7,039    382.3   4.86e-4   10,090       3
    ZAF.9.3.1_1_44534         79     5,223    300.1   8.77e-4   15,120       4
    ZAF.9.3.1_1_5517          87     6,543    375.4   6.17e-4   13,300       3
    ZAF.9.3.1_1_41942         94    12,860    490.9   3.90e-4    7,307       4
    ZAF.9.3.1_1_20343        104     9,406    381.9   7.13e-4   11,060       5
    ZAF.9.3.1_1_21024        115    12,790    487.3   4.84e-4    8,992       4
    ZAF.9.3.1_1_21159        129    14,510    556.3   4.17e-4    8,891       4
    ZAF.9.3.1_1_5530         150     9,580    508.1   5.81e-4   15,660       4

Every block clears `DENSITY_COMPACTNESS_FLOOR` (3.55e-4); median density is 9,539/km², well above
the ~4,500-5,700/km² the floor was sized for; and median peel depth is **4**, exactly the depth
`metric.py` records the floor as calibrated to buy. These are not shallow blocks that happen to be
small. They are the repo's own `density_compactness` selection.

**So the saturation is geometry, not sampling.** A block of ~10,000 m² with a ~500 m perimeter is
roughly 100 m across, so no interior point is more than ~50 m from its own boundary street. D = 50
covers everything BY CONSTRUCTION.

That makes the finding much stronger than a failed gate:

> **The screen selects for compactness; compactness is precisely what makes euclidean reach
> trivial; so any reach-based benefit term is structurally excluded by the screen itself.**

A coverage term cannot discriminate on this corpus no matter how D and W are chosen, because the
target fabric has no reach deficit to measure. If emergency access is a real concern, it is a
concern for the blocks this screen REJECTS -- the large, sprawling, low-n/P² ones where an interior
dwelling really can be 200 m from the nearest street. Testing there would be measuring a different
population, not rescuing this gate.

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
