# Permeability is a total escape TIME — and it is a sum, so the tail is unmeasured (2026-08-05)

`permeability` has always been described electrically: `P = b^T L^-1 b` is the power dissipated when
every parcel injects one unit of current toward a grounded street. There is a second reading of the
same algebra that is more useful, because it makes the metric's blind spot obvious.

## The identity

`v = L^-1 b` with `b = 1` solves `L v = 1`. Written out, with `L = D - W + diag(g_street *
1_grounded)`:

    sum_j w_ij (v_i - v_j)  +  g_i v_i  =  1

That is exactly the expected-time-to-absorption equation for the continuous-time random walk that
jumps from `i` to `j` at rate `w_ij` and is killed at rate `g_i` on street-fronting parcels. So:

> **`v_i` is the expected time for a walker starting at parcel `i` to reach a street, and
> `P = 1^T v = sum_i tau_i` is the TOTAL EXPECTED ESCAPE TIME of the block.**

Permeability, `1 - P(roads)/P(no roads)`, is therefore the **fractional reduction in total expected
escape time** that the roads buy. Same number, physical units.

Two things fall straight out.

**Why `||v||_1` equals `P`.** This was verified numerically when the metric was built and filed as a
curiosity. It is not a coincidence: `L` is an irreducibly diagonally dominant M-matrix, so
`L^-1 >= 0` entrywise and hence `v >= 0`. For a nonnegative vector `||v||_1 = 1^T v`, which is `P`.
The L1 norm is not a *choice* here — it is the quadratic form itself, written differently. (The
Löwner argument for why no other `L^p` member is monotone still stands and is independent of this.)

**Permeability is already a spectral quantity.** `P = integral_0^inf 1^T e^{-tL} 1 dt =
sum_k <1, phi_k>^2 / lambda_k`. It is a `lambda^-1`-weighted sum over the whole spectrum, dominated
by the low modes. Any "hierarchical mixing" or "spectral dimension" framing of the fabric is talking
about the same operator the metric already inverts.

## Verified, not asserted

`scratchpad/spectral/escape_time_check.py`.

**The interpretation**, on a 6-node graph whose `L` is built by hand — so nothing depends on trusting
the repo's mesh construction — against a Monte Carlo of the actual jump process (20,000 trials/node):

    node    v_i (solve)    Monte Carlo    rel err
       1         8.9531         9.0502      1.08%
       2         9.4573         9.4389      0.19%
       3         9.1492         9.0943      0.60%
       4         5.0316         5.1312      1.98%
       5         2.4000         2.3888      0.47%

All within Monte Carlo error at 20k trials. And `P = 1^T L^-1 1 = sum(v) = ||v||_1 = 44.7137`.

**The identity on real blocks**: `max |P - sum(v)| / P` over 12 Cape Town blocks is **3.4e-16** —
floating-point exact.

## The consequence: the metric is a SUM, and sums have no tail

`P = sum_i tau_i` weights every parcel equally and reports one number. Two road networks with
identical permeability can have very different *distributions* of `tau`. A network that halves the
escape time of parcels already near a street scores the same as one that rescues the deep core.

What that hides is **equity of egress**: whether the roads served everyone a little or rescued the
worst-off. `P` cannot tell you, because it added the two cases up.

**This is NOT the "Bermuda triangle" concern, and conflating them would be a mistake.** That one is
defined precisely (`notes/2026-07-30-egress-vs-circulation.md`) as *a pocket with fine egress and
terrible internal connectivity* — a **circulation** failure that egress by construction cannot
detect. `tau` is an egress quantity, so no statistic of it, tail or otherwise, addresses that. The
circulation diagnostic is the all-pairs / Kirchhoff column, already probed and already recommended as
a cheap third column rather than a replacement.

So there are two distinct cheap diagnostics for two distinct gaps: **all-pairs for circulation**
(n solves or a trace estimator) and **the `tau` tail for egress equity** (free). They are
complementary, not competing.

On 12 real blocks with no roads, `max(tau) / mean(tau)`:

    n parcels    mean tau      p95       max    max/mean
           50       81.07   182.76    194.80        2.40
           64       35.89   103.24    125.32        3.49
           73       36.21   118.74    138.91        3.84
           98       64.36   187.40    217.19        3.37
          245      452.61  1042.22   1117.50        2.47
          372      356.02   785.72    841.46        2.36

The ratio ranges **2.36 to 3.84**, so the tail is not a fixed multiple of the mean and does carry
information the scalar does not.

**The cost of reporting it is zero.** `egress_power` already returns `v` — it is computed for the
heatmap. `max(tau)`, a high quantile, or a Gini over `tau` needs no extra solve, no new mesh, and no
new parameter.

## The spectrum and the escape times are different objects

`L` is symmetric PD, so `L^-1` has eigenvalues `1/lambda_k` with the SAME eigenvectors. Those are
times — `lambda` is a rate — but they are **modal relaxation times**, one per eigenvector, where
`tau_i` is a **node** quantity. Same count, no correspondence:

    tau_i      9.7224  8.9531  9.4573  9.1492  5.0316  2.4000
    1/lambda_k 8.5537  1.3627  0.6152  0.2974  0.2052  0.1728

`tau` is not the spectrum of anything; it is the image of `1` under `L^-1`. The one place a
`1/lambda` IS literally an escape time: survival decays asymptotically as `e^{-lambda_min t}`, so
`1/lambda_min` is the residual lifetime of the last stragglers.

**Spectrum -> escape times.** `tau = sum_k (<phi_k,1>/lambda_k) phi_k` needs eigenvectors too, but
the aggregate collapses (verified to 6 digits):

> `P/n = sum_k w_k * (1/lambda_k)`, with `w_k = <phi_k,1>^2 / n` and `sum_k w_k = 1`.

The mean escape time is exactly a **convex combination of the modal relaxation times**, weighted by
each mode's overlap with the uniform vector — hence `1/lambda_max <= P/n <= 1/lambda_min`. In the
example above the slowest mode carries `w_1 = 0.863`, so `P/n = 7.452` sits near
`1/lambda_min = 8.554`.

**Escape times -> spectrum** needs MOMENTS, not the mean. `E_i[T^m] = m! (L^-m 1)_i`, checked at
`m = 2` by Monte Carlo (168.44 predicted vs 170.66 over 40k walks). Aggregating,
`1^T L^-m 1 = sum_k c_k^2 / lambda_k^m`, a determinate Stieltjes moment problem for
`sum_k c_k^2 delta_{1/lambda_k}`. So the full moment sequence recovers `{lambda_k, c_k^2}` — the
spectral measure **as seen by the uniform vector** — never the eigenvectors, and never a mode with
`<phi_k,1> = 0`.

By-product: `sum_i tau_i^2 = 1^T L^-2 1` exactly (378.910070 both ways), so `||tau||_2^2` is half the
aggregate second moment, `sum_i E_i[T^2] = 2 ||tau||_2^2`. The L2 norm does have a meaning; it is
simply not the one L1 has.

## `max(tau)` is NOT monotone — but a whole family of alternatives is

Adding conductance to one edge of the 6-node graph (`scratchpad/spectral/spectrum_vs_escape.py`):

    added conductance   1/lambda_min   max_i tau_i         P
              0.0             8.554         9.722    44.714
              0.5             8.533         9.541    44.633
              2.0             8.523         9.559    44.586
              8.0             8.518         9.576    44.564

`max(tau)` **falls, then rises**. That settles the caveat above with a counterexample rather than a
hedge: a tail statistic of `tau` cannot be an objective.

`P` and `1/lambda_min` both fall monotonically, and provably must. The general statement is stronger
than one eigenvalue, and is the useful result here:

> Adding conductance raises `L` in the Loewner order, so by Weyl monotonicity **every** `lambda_k`
> weakly increases. Hence **every Schatten norm `tr(L^-p) = sum_k lambda_k^-p` is monotone, for
> every `p > 0`** — with `p = 1` the trace and `p -> inf` giving `1/lambda_min`.

So `p` is a dial from mean-like to worst-case and **every setting is monotone**. Contrast `||tau||_p`,
where only `p = 1` is. The difference is precise and worth remembering: Schatten norms are functions
of the OPERATOR, and eigenvalue monotonicity transfers to them; `||tau||_p` is a function of one
VECTOR `L^-1 b`, and it does not transfer to that vector's components.

Caveat before anyone reaches for `tr(L^-1)`: it weights all modes equally, where `P` weights by
`c_k^2`. Trace-style rankers have already come back redundant against permeability here — the
all-pairs Kirchhoff probe scored Kendall tau +0.800 against it
(`notes/2026-07-30-egress-vs-circulation.md`). Monotone does not mean informative.

## What this does NOT establish

- **The decisive experiment has not been run.** The spread above is ACROSS blocks with no roads. What
  matters for a diagnostic is whether two proposals *on the same block at matched permeability*
  differ in their tail. If the tail turns out to be pinned by the total, this is worthless. That test
  is cheap and is the first thing to do — see the backlog's escape-time-distribution entry.
- **A tail statistic of `tau` is not monotone** — now demonstrated above, not merely suspected. So
  `max(tau)` is a **diagnostic, not an objective**; do not put it in a greedy's gain function. If an
  objective is wanted, `tr(L^-p)` is the monotone family to reach for instead.
- The escape-time reading changes no published number. It is the same `P`.

## Where the interpretation came from

Asked whether Batty's fractal-cities programme had anything to offer, and specifically whether
"mixing fast locally, then branching to another region and mixing fast there" is something a
Laplacian captures. Working out what our Laplacian's inverse actually *measures* turned out to be
worth more than the fractal literature it came from — see the backlog section
"Complexity-science cross-pollination" for the rest of that survey, most of which is lower value than
this.
