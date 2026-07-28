# β = −9.58 was a lucky donor draw; the design was underpowered

**Date:** 2026-07-28
**Status:** measured on 500 pairs. This supersedes both `2026-07-27-gw-pair-matrix-findings.md`'s
headline **and** `2026-07-28-slope-is-pool-dependent.md`'s explanation of it.

## How it surfaced

A recipient × donor crossover was meant to ask whether the pool dependence came from the
recipients or the donors. Its control cell — legacy recipients, legacy donors, the same
population and **the same 20 recipients** as the run that produced β = −9.59 (p = 0.014) — came
back at **β = −3.97 (p = 0.31)**.

Not a bug: 10 of the 100 pairs were shared between the two runs and scored **identically**
(max difference 0.00e+00 on both `perm_gap` and `real_gw_dist`). The only difference was which
donors got drawn, via a different candidate multiplier.

Same pool, same recipients, different donors, and β moves by 5.6 units.

## What 500 pairs say

Scoring 25 donors per recipient instead of 5, over the same 20 legacy recipients:

```
all 500 pairs at once:   beta = -2.895   se = 1.604   p = 0.0717

resampling the 5-donors-per-recipient design 4,000 times:
  beta   median -2.834   95% interval [-11.218, +5.355]   sd 4.227
  sign   negative in 75.1% of draws
  SIGNIFICANT (p<0.05) in 11.7% of draws
  the published -9.58 sits at the 5.1st percentile of this distribution
```

**The headline was a 5th-percentile draw.** The design produced a significant result in about one
donor draw in eight, and the run that got published was one of them.

## What is actually true

- The best estimate available is **β = −2.90, SE 1.60, p = 0.072** on 500 pairs — the same sign as
  the headline, roughly a third the magnitude, and **not significant**.
- There is a **weak negative tendency**: 75% of donor draws come out negative, and the 500-pair
  point estimate is negative at p = 0.07. That is worth something, and it is not nothing.
- It is **not an established effect**, and no single-draw p-value from this design should be read
  as if it were.

## The previous explanation was also wrong

`2026-07-28-slope-is-pool-dependent.md` concluded that β was a property of the pool, on the
strength of −9.59 (legacy) vs +5.37 (screen). Both of those sit comfortably inside the donor-draw
95% interval of **[−11.2, +5.4]** measured here. The pool difference is within sampling noise, so
attributing it to the pool was reading structure into variance.

Everything that note ruled out — donor source, range restriction, outline, depth — was ruled out
correctly, and those measurements stand. What was wrong was the premise that there was a stable
pool difference to explain.

The outline decomposition in that note is unaffected in its own terms (it asked which component of
GW distance carried a slope, on one fixed sample), but it inherits the same caveat: it was
decomposing a slope that is itself mostly noise at n=100.

## The design lesson

**Five donors per recipient cannot resolve an effect of this size.** SE on 100 pairs is 3.81;
on 500 it is 1.60. With a true β near −2.9, 100 pairs has a standard error larger than the effect,
so the run's outcome is dominated by which donors it happened to draw.

Every future matrix should score **~25 donors per recipient**, and any β should be reported with a
donor-resampling interval, not a single-draw p-value. `scratchpad/ot/donor_draw_variance.py` does
this and should graduate into the benchmark script.

Cost is not a reason not to: with donors coming off the local PBF, 500 pairs took about 8 minutes.
The 100-pair design was inherited from when each donor meant an Overpass round trip.

## What this costs the programme

The bet is not dead — the sign is consistently negative and the 500-pair estimate is p = 0.07 —
but "closer donors transplant measurably better" is **not demonstrated**, and the number that has
been carried since 2026-07-27 as if it were should not be used.

The larger corpus is now the thing that decides it. The census plus provisioning gave 65,364
qualified blocks with 9.81M building points; a matrix over that pool at 25 donors per recipient,
with a resampling interval, is the experiment that can actually answer the question.
