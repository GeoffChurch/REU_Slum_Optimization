# The road/walk ratio has no physical support at the value the rankings need (2026-08-08)

B2 measured where method rankings destabilise on the continuum: stable at 67x-150x (per-block
Kendall tau +1.000), degrading at 35x, scrambling at 5x-15x. It failed as written, and the open
question was whether the physically defensible band sits above or below ~35x.

**It sits below. Both ways of deriving it land under the band, and they disagree with each other by
20x.** So the published ordering of methods rests on a constant that no physical reading supports.

## What changed to make the question answerable

On the parcel graph, a footpath edge is `g_walk * clearance_fraction / dist` -- a DIMENSIONLESS
shape over distance. The channel's width never enters, so `g_walk` silently carries two things at
once: how good the surface is, and how much of it there is.

In the continuum, conductance is `sigma * w / L` with `w` supplied by the geometry. `sigma` becomes
a per-unit-width conductivity and nothing else. That splits the tuned constant into a part the mesh
now represents for itself (width) and a part `sigma` must still carry (surface and mode) -- and only
the second is a free parameter.

## The walking channel is 3 m wide, not 1 m

`scratchpad/ratio/gap_widths.py`, 73 blocks across both cities, 163k ridge samples. Free space is
rasterized at 0.25 m, Euclidean-distance-transformed, and read on the medial ridge; twice the ridge
value is the local channel width, which is also the line a walker follows.

    band (bldg/km2)   blocks   samples     p10     p25   median     p75
    1-2k                  13    29,088    1.00    3.00    7.65    15.12
    2-4k                  27    71,937    0.50    1.80    4.47     8.50
    4-8k                  28    53,468    0.50    1.41    3.00     6.50
    8k+                    5     8,482    0.50    1.50    2.50     5.00

Tight alleys are real -- p10 is 0.5 m everywhere, p25 is ~1.4 m in the dense bands -- but the MEDIAN
channel is 2.5-3.0 m even at 8k+ buildings/km². Against a road's 6 m usable, width alone is a 2x
advantage, not a 200x one.

**Do not measure this as distance to the nearest building.** Tried first, and it reads a median of
0.00 m because 50.3% of buildings touch their nearest neighbour. B1 already established free space
stays connected (98.2-100% in one component): a rectangle touching one neighbour on one side leaves
its other three sides open. Nearest-neighbour distance measures the tightest contact, not the
channel.

## Three readings of the same ratio

`scratchpad/ratio/derive_ratio.py` prints every step, because the last attempt at this arithmetic
was wrong by 2.52x (513x vs 204x) from applying a per-block normalization on one side only.

**(1) Implied by today's calibration: 102x - 272x.** A road is `20.0/L` per direction, walking is
`0.0981/L`; that is the published 204x. Converting to per-unit-width conductivity divides the
walking side by the measured channel width and the road side by its usable width. The remaining
fork is on the road side -- whether the graph's single undirected edge worth 20.0 buys one 3 m lane
or the whole 6 m corridor is a modelling choice, and it moves `sigma_road` by 2x.

**(2) Derived from throughput -- 0.18x to 0.31x.** If `sigma` is capacity per metre of width, which
is the reading consistent with the project's own "collective, contention-aware" framing and with
`P = sum tau_i` being a congestion cost: a walkway carries 0.8-1.4 persons/m/s (HCM pedestrian LOS),
while a traffic lane at 1800 veh/h and 1.5 occupants over 3 m carries 0.25 persons/m/s. **Per metre
of width, a lane moves FEWER people than a footway.** Its advantage is that it is wide, and that is
already geometry.

**(3) Derived from speed -- 3.2x to 6.4x.** If `sigma` scales with how fast you cover ground:
walking 1.3 m/s against 15-30 km/h on a settlement access road.

## What this means

Reading (1) is circular. It is the tuned number re-expressed, and `conf/permeability.yaml` records
in its own comments that `g_walk` and the lane calibration were adopted because they "massively
improve method discrimination". Calibrating for spread and then citing the result as the ratio is
the same operation twice -- the circularity the road-first-mesh spec's own red team flagged.

Readings (2) and (3) are the honest ones, and **both land below B2's stable band**: 0.2-0.3x and
3-6x against a floor of ~35x. At those values B2 measured rankings scrambling, with the winner
changing on 2-3 blocks in 10 and worst-case tau +0.200.

**So the ordering of methods is a product of the calibration, not of the geometry.** That is the
finding, and it is larger than the metric refactor it came out of.

## The deeper problem: (2) and (3) disagree by 20x

That gap is not noise, it is the model asking a question it cannot answer. A linear conductance
prices *how much flow fits*; it has no term for *what kind of trip becomes possible*. And in a
settlement the road's real value is largely a THRESHOLD -- a 6 m street admits an ambulance, a fire
appliance and a refuse truck, and a 1.4 m alley admits none of them, at any conductivity. Reading
(2) says roads barely help; reading (3) says they help ~5x; neither expresses the thing that
actually motivates reblocking.

A conductance model cannot represent a threshold. If access for emergency vehicles is the binding
benefit -- and `PermeabilityParams.min_road_width_m` already justifies its own value on exactly
those grounds -- then permeability is measuring the wrong thing about roads, and the fix is a
benefit term with a width threshold in it, not a better value of `sigma`.

## Caveats

* The 10.19 resistance-per-metre walking figure is inherited from
  `notes/2026-08-06-road-walk-ratio-decides-the-ranking.md`, measured on the parcel graph.
* Ridge samples are weighted by cell count, not by traffic, so wide interior courtyards count more
  than their use would justify. This biases the measured channel width UP, which makes the width
  share of the advantage look larger and the required `sigma` ratio smaller -- i.e. it biases
  toward the conclusion drawn here. The dense bands (2.5-3.0 m) are the conservative read.
* Capacity and speed figures are generic engineering values, not settlement-specific measurements.
* B2's stability band is 10 blocks x 5 methods on the continuum prototype.
