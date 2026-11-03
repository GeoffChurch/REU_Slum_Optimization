"""The screen's concepts (owner 2026-11-03): regions of blocks that are homogeneous inside and
distinct from what surrounds them, surprisingly well silhouetted for their size, for the Demo
Lineup artifact's height map.

The blocks (the site's ScreenMap bundle) each carry a weight w, their buildings, and a value v,
their log10 score (depth_density_proxy sqrt(nA)/P n/A, or density_compactness n/P^2; a score of
0 at the city's 1st percentile), on screen_clusters.py's street adjacency (blocks are street
faces; neighbours share >= 5 m of boundary within 0.5 m).

Candidates (a Strategy: Candidates; only Concepts implemented, the hull tree's nodes of
screen_clusters.py would be scored the same way). A concept, after formal concept analysis, is a
pair (R, [lo, hi]) closed both ways: R a connected component of the blocks with v in [lo, hi],
[lo, hi] the tightest band holding R's scores. So R is homogeneous (within its band), distinct
(every block bordering it, its ring, lies outside the band) and maximal. Its span is hi - lo;
its gap the lowest ring score above the band less the highest below (+inf with none). R is a
component of some window [x, x + d] exactly when span <= d < gap, so a concept lives over the
widths [span, gap). Enumeration: for each width d of the ladder WIDTHS (8, geometric, 0.05 to 1
decade, ratio 1.53; the scores' sd is 0.38 -- 0.59 decades, so the ladder runs from a tenth of
it, where most windows' components are single blocks, to two, where one component would hold
most of the city), an exact sweep of x up the scores, a block entering at x = v - d and leaving
once x passes v (entries first at a tie), the components each event changes (the entering
block's; the leaving block's neighbours' pieces) BFS'd and emitted, distinct member sets kept
(128-bit Zobrist hash, collisions counted and refused). Coverage, checked against every band's
components on small graphs: exactly the concepts living at some ladder width, so every concept
whose life gap / span exceeds 1.53 is found, and one is missed only if its life falls between
two rungs. Kept: >= 2 blocks, at most MAX_SHARE of the city's buildings, a ring with buildings.
A sweep is 0.1 s on Nairobi, 0.6 s on Cape Town (2.2 s on an autocorrelated MSR surrogate,
whose components are larger).

Score: the building-weighted silhouette against the ring. For block i of R, a_i the weighted
mean |v_i - v_j| over R's other blocks, b_i over the ring's; s_i = (b_i - a_i) / max(a_i, b_i);
S(R) their weighted mean (-1 to 1), with a and b their weighted means. O(k log k) a region from
sorted scores' prefix sums (numba). Also the contrast, R's weighted mean log score less the
ring's: signed, the top lists below take both signs, a second list only regions above their ring.

Size: n_eff = (sum w)^2 / sum w^2, the effective number of blocks (the ring's recorded too).
Calibration, per null, from NPERM replicates, each regenerating every candidate (so selection
is in the null): p = (1 + #replicates whose best S among their concepts with n_eff within a
factor WINDOW of the region's is >= S) / (1 + NPERM), on log10 n_eff bins of log10(WINDOW) / 8
(the region's n_eff snapped to its bin); z = (S - mean) / sd of every null concept's S at that
n_eff, a Nadaraya-Watson Gaussian kernel in log10 n_eff of bandwidth log10(WINDOW). A region
larger than any null concept gets p's floor (counted in the output: concepts_beyond_null_sizes).
The q95 and q50 of the per-replicate best go to the output as the page's null curve.

Nulls (a Strategy each: Null), reported side by side:
  shuffle: the log scores permuted among the blocks (weights and geometry kept).
  msr: Moran spectral randomization (Wagner and Dray 2015, Methods Ecol Evol 6:1169): the n - 1
    Moran eigenvector maps of the binary street adjacency (eigenvectors of the doubly centred
    HAH less K J/n, K past the spectrum so the constant falls out alone; one dense LAPACK evd a
    city, 1 s Nairobi, ~90 s Cape Town on EIG_THREADS threads, 2.7 GB), the centred scores'
    coefficients each sign-flipped ('singleton': the spectrum, hence Moran's I, kept exactly),
    mapped back, and the observed scores re-dealt by the surrogate's ranks (the marginal is the
    observed one, as under the shuffle: the silhouette is not invariant to a monotone
    transform). The re-deal costs a little of Moran's I (Nairobi 0.543 -> 0.536 +- 0.015 and
    0.544 -> 0.523 +- 0.022); IAAFT-style rounds re-imposing the magnitudes did not recover it
    (1, 3, 10 rounds: 0.534, 0.523 on Nairobi) and were dropped.
Replicates seeded default_rng([SEED, null index, replicate]); batches of BATCH per task so the
basis is read once a batch (no effect on a result).

Shape (the top lists only): the screen applied to the region as one block: its blocks' union
closed by CLOSE_M (the shared streets), P its exterior's length, A its area (holes excluded), n
its buildings: both metrics and x their floors; hull fill, the blocks' area over their convex
hull's. The top list: hull fill >= HULL_FILL_MIN, ranked by p under msr, then z under msr, TOP
disjoint (greedy: a region sharing a block with a better one skipped).

Also per metric: the counts of concepts at p < 0.05 and 0.01 under each null (over every
concept, nested ones included, so one region counts many times), and `follow`, the best-ranked
concept holding the bundle's followed block (Cape Town's 5810), hull fill not required.

Run 2026-11-03 (NPERM 999 per null, 16 workers, load ~1): Nairobi 45 s, Cape Town 250 s (the
eigenvectors 86 s, each metric's 1,998 replicates 74 -- 84 s). Concepts: Nairobi 14,903
(depth_density_proxy) and 13,166 (density_compactness); Cape Town 77,519 and 61,515. Moran's I
observed / MSR surrogates / shuffles: Cape Town 0.684 / 0.674 +- 0.011 / 0.000 and 0.578 / 0.552
+- 0.012 / 0.000; Nairobi 0.543 / 0.538 +- 0.013 / -0.001 and 0.544 / 0.525 +- 0.023 / -0.001.
What the silhouette finds: crisp edges, and crisp edges are mostly where a settled area meets
empty ground: the regions at the top of every list sit below the floor (0.3 -- 0.5x) in rings
far emptier still (0.01 -- 0.05x). Blocks above the floor grade into their surroundings, so few
make a crisp concept; 5810's best (depth_density_proxy) is itself and one neighbour, S 0.27,
rank 69,420 of 77,519. The shuffle overstates: concepts at p < 0.05 under shuffle / MSR, Cape
Town 214 / 3 and 1,035 / 139, Nairobi 185 / 42 and 104 / 32; on the top lists the shuffle's z
exceeds MSR's by 1.0 -- 1.4 for regions of n_eff 30 -- 160 (by ~0.05 for a few blocks, where
autocorrelation barely matters). Under MSR only Nairobi's 176-block region (both metrics, p
0.001) and Cape Town's density_compactness regions of 95 -- 222 blocks (p 0.004 -- 0.010) stand
clearly out; Cape Town's depth_density_proxy has no concept at p < 0.01.

    S=research/roadless/atlas/explore_proto/screen_concepts.py  # from the repo root, PYTHONPATH=.
    uv run python $S ~/src/wt_demo <out dir>    # concepts_<city>.json per city, both metrics
    uv run python $S check [fault]              # closure, silhouette, MSR; faults: widen,
                                                # ring-unweighted, msr-shuffle
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"           # the pool's workers single-threaded (the machine is shared)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple, Protocol  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
import screen_clusters as sc  # noqa: E402  (the script's directory is on the path)
import shapely  # noqa: E402
from numba import njit, types  # noqa: E402
from numba.typed import Dict  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.ndimage import maximum_filter1d  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

WIDTHS = tuple(float(x) for x in np.geomspace(0.05, 1.0, 8))   # decades: the band-width ladder
MAX_SHARE = 0.5        # a concept's buildings at most this share of the city's
WINDOW = 1.25          # p: the null's best S among concepts with n_eff within this factor
BINS_PER_WINDOW = 8    # n_eff bins per log10(WINDOW)
HULL_FILL_MIN = 0.5    # the top list: blocks' area / convex hull's at least this
TOP = 8
CLOSE_M = 5.0          # m: a region's outline is the union of its blocks closed by this
NPERM = {"capetown": 999, "nairobi": 999}       # replicates per null
SEED = 0
WORKERS = 16
EIG_THREADS = 16       # the one eigendecomposition per city (no effect on its result)
HASH_SEED = 20261103   # per-block 64-bit keys for de-duplicating member sets (no effect on a
                       # result unless two sets collide in 128 bits; collisions are counted)
CITIES = ("nairobi", "capetown")

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


# ---------------------------------------------------------------- the silhouette

@njit(cache=True)
def silhouette(vin: F64, win: F64, vring: F64, wring: F64) -> tuple[float, float, float]:
    """The region's building-weighted a, b and S: per block i of the region a_i the weighted
    mean |v_i - v_j| over the region's other blocks j, b_i over the ring's blocks, s_i =
    (b_i - a_i) / max(a_i, b_i); each the weighted mean over i. O(k log k): both sets sorted,
    the sums of |x - v_j| from prefix sums of the weights and weighted values."""
    oi = np.argsort(vin)
    x, wx = vin[oi], win[oi]
    orr = np.argsort(vring)
    y, wy = vring[orr], wring[orr]
    k, m = len(x), len(y)
    Wx, Sx = wx.sum(), (wx * x).sum()
    Wy, Sy = wy.sum(), (wy * y).sum()
    cw = cs = dw = ds = 0.0
    q = 0
    A = B = Ss = 0.0
    for i in range(k):
        cw += wx[i]
        cs += wx[i] * x[i]
        while q < m and y[q] <= x[i]:
            dw += wy[q]
            ds += wy[q] * y[q]
            q += 1
        din = x[i] * cw - cs + (Sx - cs) - x[i] * (Wx - cw)
        dring = x[i] * dw - ds + (Sy - ds) - x[i] * (Wy - dw)
        den = Wx - wx[i]
        a = din / den if den > 0 else 0.0
        b = dring / Wy
        mx = max(a, b)
        s = (b - a) / mx if mx > 0 else 0.0
        A += wx[i] * a
        B += wx[i] * b
        Ss += wx[i] * s
    return A / Wx, B / Wx, Ss / Wx


# ---------------------------------------------------------------- the concepts

class Scored(NamedTuple):
    """A generator's candidates, one row each: a block of it (the seed), its tight band, its
    and its ring's sizes and sums, a, b, the silhouette, and the ladder width that found it."""
    seed: I64
    lo: F64
    hi: F64
    blocks: I64
    W: F64
    W2: F64
    S: F64
    ring_blocks: I64
    Wr: F64
    W2r: F64
    Sr: F64
    a: F64
    b: F64
    sil: F64
    width: I64

    @property
    def n_eff(self) -> F64:
        return self.W ** 2 / self.W2


NCOL = 15


@njit(cache=True)
def _component(start: int, ok: NDArray[np.bool_], ptr: I64, nbr: I64, stamp: I64, st: int,
               queue: I64) -> int:
    """BFS from start over the blocks with ok; marks them stamp = st into queue[:k]; k."""
    queue[0] = start
    stamp[start] = st
    h, t = 0, 1
    while h < t:
        x = queue[h]
        h += 1
        for e in range(ptr[x], ptr[x + 1]):
            y = nbr[e]
            if ok[y] and stamp[y] != st:
                stamp[y] = st
                queue[t] = y
                t += 1
    return t


@njit(cache=True)
def _emit(k: int, st: int, wi: int, v: F64, w: F64, ptr: I64, nbr: I64, stamp: I64,
          rstamp: I64, queue: I64, key1: I64, key2: I64, cap: float, seen: Dict,
          out: F64, nout: int, coll: NDArray[np.int64]) -> tuple[F64, int]:
    """Score the component queue[:k] (its blocks stamped st) if it is new, >= 2 blocks, under
    the cap, with a ring holding buildings; append its row to out."""
    if k < 2:
        return out, nout
    h1 = np.int64(0)
    h2 = np.int64(0)
    W = 0.0
    for t in range(k):
        h1 += key1[queue[t]]
        h2 ^= key2[queue[t]]
        W += w[queue[t]]
    if W > cap:
        return out, nout
    if h1 in seen:
        if seen[h1] != h2:
            coll[0] += 1
        return out, nout
    seen[h1] = h2
    ring = np.empty(k * 8 + 8, dtype=np.int64)
    nr = 0
    for t in range(k):
        x = queue[t]
        for e in range(ptr[x], ptr[x + 1]):
            y = nbr[e]
            if stamp[y] != st and rstamp[y] != st:
                rstamp[y] = st
                if nr == len(ring):
                    r2 = np.empty(2 * nr, dtype=np.int64)
                    r2[:nr] = ring
                    ring = r2
                ring[nr] = y
                nr += 1
    if nr == 0:
        return out, nout
    mem = queue[:k]
    rg = ring[:nr]
    vin, win, vr, wr = v[mem], w[mem], v[rg], w[rg]
    if wr.sum() <= 0:
        return out, nout
    a, b, s = silhouette(vin, win, vr, wr)
    if nout == len(out):
        o2 = np.empty((2 * nout, NCOL))
        o2[:nout] = out
        out = o2
    row = out[nout]
    row[0], row[1], row[2], row[3], row[4] = mem.min(), vin.min(), vin.max(), k, W
    row[5], row[6], row[7], row[8] = (win * win).sum(), (win * vin).sum(), nr, wr.sum()
    row[9], row[10], row[11], row[12], row[13], row[14] = ((wr * wr).sum(), (wr * vr).sum(),
                                                           a, b, s, wi)
    return out, nout + 1


@njit(cache=True)
def concepts_of(v: F64, w: F64, ptr: I64, nbr: I64, widths: F64, cap: float, key1: I64,
                key2: I64) -> tuple[F64, int]:
    """Every distinct concept found by the ladder: per width d a sweep of the band [lo, lo + d]
    up the scores, blocks entering at lo = v - d and leaving once lo passes v (entries first at
    a tie), so the window is always the blocks of a band; after each event the components it
    changed (the entering block's; the leaving block's neighbours' pieces) are emitted."""
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    stamp = np.zeros(n, dtype=np.int64)
    rstamp = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    seen = Dict.empty(key_type=types.int64, value_type=types.int64)
    out = np.empty((1 << 14, NCOL))
    nout = 0
    coll = np.zeros(1, dtype=np.int64)
    st = 0
    for wi in range(len(widths)):
        d = widths[wi]
        inw = np.zeros(n, dtype=np.bool_)
        e = lv = 0
        while lv < n:
            if e < n and v[order[e]] - d <= v[order[lv]]:
                j = order[e]
                e += 1
                inw[j] = True
                st += 1
                k = _component(j, inw, ptr, nbr, stamp, st, queue)
                out, nout = _emit(k, st, wi, v, w, ptr, nbr, stamp, rstamp, queue, key1, key2,
                                  cap, seen, out, nout, coll)
            else:
                i = order[lv]
                lv += 1
                inw[i] = False
                ev = st
                for x in range(ptr[i], ptr[i + 1]):
                    u = nbr[x]
                    if inw[u] and stamp[u] <= ev:
                        st += 1
                        k = _component(u, inw, ptr, nbr, stamp, st, queue)
                        out, nout = _emit(k, st, wi, v, w, ptr, nbr, stamp, rstamp, queue,
                                          key1, key2, cap, seen, out, nout, coll)
    return out[:nout], int(coll[0])


@njit(cache=True)
def band_component(seed: int, lo: float, hi: float, v: F64, ptr: I64, nbr: I64) -> I64:
    """The component of {v in [lo, hi]} holding seed: a concept's members from its row."""
    n = len(v)
    ok = (v >= lo) & (v <= hi)
    stamp = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    k = _component(seed, ok, ptr, nbr, stamp, 1, queue)
    return np.sort(queue[:k])


class Candidates(Protocol):
    """A generator of candidate regions, each scored (Scored): the concepts here; the hull
    tree's nodes of screen_clusters.py would be another (its members, their ring, silhouette)."""
    name: str

    def scored(self, v: F64) -> Scored: ...

    def members(self, s: Scored, row: int, v: F64) -> I64: ...


@dataclass(frozen=True)
class Concepts:
    b: sc.Blocks
    widths: tuple[float, ...]
    max_share: float
    name: str = "concepts"

    def scored(self, v: F64) -> Scored:
        rng = np.random.default_rng(HASH_SEED)
        n = len(v)
        key1 = rng.integers(-2 ** 62, 2 ** 62, n, dtype=np.int64)
        key2 = rng.integers(-2 ** 62, 2 ** 62, n, dtype=np.int64)
        out, coll = concepts_of(v, self.b.w, self.b.nbr_ptr, self.b.nbr,
                                np.asarray(self.widths), self.max_share * self.b.w.sum(),
                                key1, key2)
        if coll:
            raise RuntimeError(f"{coll} member-set hash collisions")
        i = lambda c: out[:, c].astype(np.int64)    # noqa: E731
        return Scored(seed=i(0), lo=out[:, 1], hi=out[:, 2], blocks=i(3), W=out[:, 4],
                      W2=out[:, 5], S=out[:, 6], ring_blocks=i(7), Wr=out[:, 8], W2r=out[:, 9],
                      Sr=out[:, 10], a=out[:, 11], b=out[:, 12], sil=out[:, 13], width=i(14))

    def members(self, s: Scored, row: int, v: F64) -> I64:
        return band_component(int(s.seed[row]), float(s.lo[row]), float(s.hi[row]), v,
                              self.b.nbr_ptr, self.b.nbr)


def ring_of(mem: I64, b: sc.Blocks) -> I64:
    return np.setdiff1d(np.unique(np.concatenate(
        [b.nbr[b.nbr_ptr[i]:b.nbr_ptr[i + 1]] for i in mem])), mem)


# ---------------------------------------------------------------- the nulls

def adjacency(b: sc.Blocks) -> sparse.csr_matrix:
    n = len(b.w)
    return sparse.csr_matrix((np.ones(len(b.nbr)), b.nbr, b.nbr_ptr), shape=(n, n))


def moran(x: F64, A: sparse.csr_matrix) -> float:
    """Moran's I on the binary street adjacency: n / S0 * x'Ax / x'x, x centred."""
    xc = x - x.mean()
    return float(len(x) / A.sum() * (xc @ (A @ xc)) / (xc @ xc))


class Null(Protocol):
    name: str

    def surrogates(self, v: F64, rngs: list[np.random.Generator]) -> F64: ...


@dataclass(frozen=True)
class Shuffle:
    """The blocks' log scores permuted among the blocks (weights and geometry kept)."""
    name: str = "shuffle"

    def surrogates(self, v: F64, rngs: list[np.random.Generator]) -> F64:
        return np.stack([r.permutation(v) for r in rngs])


@dataclass(frozen=True)
class MSR:
    """Moran spectral randomization (Wagner and Dray 2015), 'singleton': the centred scores'
    coefficients on every Moran eigenvector map of the binary street adjacency (the n - 1
    eigenvectors of the doubly centred matrix HAH orthogonal to the constant), each sign
    flipped at random, mapped back: the autocorrelation spectrum, hence Moran's I, kept
    exactly. Then the observed scores re-dealt by the surrogate's ranks, so the marginal is the
    observed one, as the shuffle's (the silhouette is not invariant to a monotone transform),
    at a small cost in Moran's I. One replicate per generator in rngs, a column each, so the
    basis is read once per batch."""
    U: F64                   # n x (n - 1)
    name: str = "msr"

    def surrogates(self, v: F64, rngs: list[np.random.Generator]) -> F64:
        c = self.U.T @ (v - v.mean())
        E = np.stack([r.integers(0, 2, len(c)) * 2.0 - 1.0 for r in rngs], axis=1)
        X = self.U @ (E * c[:, None])
        ranked = np.sort(v)
        return ranked[np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0,
                                 kind="stable")].T


def mem_basis(A: sparse.csr_matrix) -> tuple[F64, F64]:
    """The Moran eigenvector maps: eigenvectors of HAH less K J/n (K past the spectrum, so the
    constant is the lowest and alone, dropped). Dense LAPACK (evd)."""
    n = A.shape[0]
    M = A.toarray()
    M -= M.mean(axis=0, keepdims=True)
    M -= M.mean(axis=1, keepdims=True)
    K = 2.0 * float(np.abs(A).sum(axis=1).max()) + 1.0
    M -= K / n
    with threadpool_limits(EIG_THREADS):
        lam, U = scipy.linalg.eigh(M, driver="evd", overwrite_a=True, check_finite=False)
    del M
    return lam[1:], U[:, 1:]


# ---------------------------------------------------------------- size calibration

STEP = float(np.log10(WINDOW)) / BINS_PER_WINDOW     # log10 n_eff per bin


def bins_of(n_eff: F64) -> I64:
    return np.floor(np.log10(n_eff) / STEP).astype(np.int64)


class Binned(NamedTuple):
    """One replicate's concepts by n_eff bin: the best S, and the count, sum and sum of squares."""
    best: F64
    cnt: F64
    s1: F64
    s2: F64


def binned(n_eff: F64, sil: F64, G: int) -> Binned:
    g = bins_of(n_eff)
    best = np.full(G, -np.inf)
    np.maximum.at(best, g, sil)
    return Binned(best=best, cnt=np.bincount(g, minlength=G).astype(float),
                  s1=np.bincount(g, sil, minlength=G), s2=np.bincount(g, sil ** 2, minlength=G))


class Calibration(NamedTuple):
    """A null's curves over the n_eff bins: per replicate the best S among its concepts with
    n_eff within WINDOW of the bin's (sorted over the replicates, for p), and the mean and sd of
    every null concept's S, Gaussian-kernel smoothed in log10 n_eff (bandwidth log10 WINDOW)."""
    wbest: F64               # R x G, sorted along the replicates
    mean: F64
    sd: F64
    support: F64             # the kernel's effective count of null concepts at the bin

    def p(self, n_eff: F64, sil: F64) -> F64:
        g = bins_of(n_eff)
        R = self.wbest.shape[0]
        ge = np.array([R - np.searchsorted(self.wbest[:, gi], s, side="left")
                       for gi, s in zip(g, sil, strict=True)])
        return (1.0 + ge) / (1.0 + R)

    def z(self, n_eff: F64, sil: F64) -> F64:
        g = bins_of(n_eff)
        return (sil - self.mean[g]) / self.sd[g]

    def q(self, level: float) -> F64:
        return np.quantile(self.wbest, level, axis=0, method="inverted_cdf")

    def unsupported(self, n_eff: F64) -> int:
        """How many of these n_eff no replicate's concept comes within WINDOW of (their p the
        floor: a size chance never reached)."""
        return int((~np.isfinite(self.wbest[-1, bins_of(n_eff)])).sum())


def calibrate(reps: list[Binned]) -> Calibration:
    best = np.stack([r.best for r in reps])
    wbest = np.sort(maximum_filter1d(best, size=2 * BINS_PER_WINDOW + 1, axis=1,
                                     mode="constant", cval=-np.inf), axis=0)
    cnt = np.sum([r.cnt for r in reps], axis=0)
    s1 = np.sum([r.s1 for r in reps], axis=0)
    s2 = np.sum([r.s2 for r in reps], axis=0)
    G = len(cnt)
    c = np.arange(G)
    K = np.exp(-0.5 * ((c[:, None] - c[None, :]) / BINS_PER_WINDOW) ** 2)
    kc = K @ cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (K @ s1) / kc
        sd = np.sqrt(np.maximum((K @ s2) / kc - mean ** 2, 0.0))
    return Calibration(wbest=wbest, mean=mean, sd=sd, support=kc)


# ---------------------------------------------------------------- the null replicates

_JOB: dict = {}


def _batch(task: tuple[int, list[int]]) -> list[tuple[Binned, float]]:
    ni, reps = task
    null, gen, v, A, G = _JOB["nulls"][ni], _JOB["gen"], _JOB["v"], _JOB["A"], _JOB["G"]
    X = null.surrogates(v, [np.random.default_rng([SEED, ni, r]) for r in reps])
    out = []
    for x in X:
        s = gen.scored(x)
        out.append((binned(s.n_eff, s.sil, G), moran(x, A)))
    return out


BATCH = 16             # replicates per task (no effect on a result: each its own generator)


def nulls_run(gen: Candidates, v: F64, A: sparse.csr_matrix, nulls: list[Null], nperm: int,
              G: int) -> list[tuple[Calibration, F64, float]]:
    """Per null: its calibration, its surrogates' Moran's I, and its seconds."""
    _JOB.update(nulls=nulls, gen=gen, v=v, A=A, G=G)
    tasks = [(ni, list(range(i, min(i + BATCH, nperm))))
             for ni in range(len(nulls)) for i in range(0, nperm, BATCH)]
    t0 = time.time()
    with get_context("fork").Pool(WORKERS) as pool:
        res = pool.map(_batch, tasks, chunksize=1)
    dt = time.time() - t0
    out = []
    for ni in range(len(nulls)):
        rows = [x for (tn, _), r in zip(tasks, res, strict=True) if tn == ni for x in r]
        out.append((calibrate([b for b, _ in rows]), np.array([m for _, m in rows]), dt))
    return out


# ---------------------------------------------------------------- shape and selection

def hull_fill(mem: I64, b: sc.Blocks) -> float:
    h = sc.hull(np.vstack([b.hull_pts[i] for i in mem]))
    return float(b.area[mem].sum() / max(sc.area_of(h), 1e-9))


def region_screen(mem: I64, b: sc.Blocks, floors: dict[str, float]) -> dict:
    """The screen on the region as one block: its blocks' union closed by CLOSE_M (the shared
    streets), P its exterior's length (holes not counted), A its area (holes, its ring's
    enclosed blocks, excluded), n its buildings."""
    u = shapely.union_all([b.polys[i] for i in mem]).buffer(CLOSE_M).buffer(-CLOSE_M)
    parts = shapely.get_parts(u)
    P = float(sum(p.exterior.length for p in parts))
    A, n = float(u.area), float(b.w[mem].sum())
    out = {"n": n, "A_m2": A, "P_m": P, "parts": len(parts)}
    for m, f in sc.METRICS.items():
        s = float(f(n, A, P))
        out[m] = {"score": s, "x_floor": s / floors[m]}
    return out


def selected(order: I64, gen: Candidates, s: Scored, v: F64, b: sc.Blocks
             ) -> list[tuple[int, I64, float]]:
    """The TOP first rows of order whose hull fill is at least HULL_FILL_MIN, no two sharing
    a block: (row, members, hull fill)."""
    used = np.zeros(len(v), dtype=bool)
    out = []
    for r in order:
        mem = gen.members(s, int(r), v)
        if used[mem].any():
            continue
        f = hull_fill(mem, b)
        if f < HULL_FILL_MIN:
            continue
        used[mem] = True
        out.append((int(r), mem, f))
        if len(out) == TOP:
            break
    return out


# ---------------------------------------------------------------- the run

def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def entry(r: int, mem: I64, fill: float, s: Scored, v: F64, b: sc.Blocks,
          floors: dict[str, float], metric: str, zp: dict[str, tuple[F64, F64]]) -> dict:
    W, Wr = float(s.W[r]), float(s.Wr[r])
    m_in, m_ring = float(s.S[r]) / W, float(s.Sr[r]) / Wr
    out = dict(
        members=mem.tolist(), blocks=int(s.blocks[r]), buildings=W, n_eff=float(s.n_eff[r]),
        ring_blocks=int(s.ring_blocks[r]), ring_n_eff=Wr ** 2 / float(s.W2r[r]),
        band=[float(10 ** s.lo[r]), float(10 ** s.hi[r])], width=WIDTHS[int(s.width[r])],
        mean_score=10 ** m_in, x_floor=10 ** m_in / floors[metric],
        mean_ring_score=10 ** m_ring, contrast=m_in - m_ring,
        a=float(s.a[r]), b=float(s.b[r]), silhouette=float(s.sil[r]))
    for name, (z, p) in zp.items():
        out[f"z_{name}"] = _finite(z[r])
        out[f"p_{name}"] = float(p[r])
    out.update(region_screen=region_screen(mem, b, floors), hull_fill=fill,
               area_km2=float(b.area[mem].sum() / 1e6),
               cx=float((b.cen[mem, 0] * b.w[mem]).sum() / W),
               cy=float((b.cen[mem, 1] * b.w[mem]).sum() / W))
    return out


def scanned(city: dict, b: sc.Blocks, A: sparse.csr_matrix, gen: Candidates,
            nulls: list[Null], ranking: Null, metric: str, nperm: int) -> dict:
    """One metric: the concepts, both nulls' calibrations, the TOP lists (ranked by ranking's
    p, then its z)."""
    v = sc.log_scores(city, metric)
    floors = {f["metric"]: f["value"] for f in city["floors"]}
    t0 = time.time()
    s = gen.scored(v)
    t_obs = time.time() - t0
    G = int(np.ceil(np.log10(len(v)) / STEP)) + 2
    I_obs = moran(v, A)
    runs = nulls_run(gen, v, A, nulls, nperm, G)
    zp = {nl.name: (cal.z(s.n_eff, s.sil), cal.p(s.n_eff, s.sil))
          for nl, (cal, _, _) in zip(nulls, runs, strict=True)}
    z_r, p_r = zp[ranking.name]
    order = np.lexsort((-np.nan_to_num(z_r, nan=-np.inf), p_r))
    above = order[(s.S / s.W - s.Sr / s.Wr)[order] > 0]
    lists = {k: [entry(r, mem, f, s, v, b, floors, metric, zp)
                 for r, mem, f in selected(o, gen, s, v, b)]
             for k, o in (("concepts", order), ("concepts_above", above))}
    # the bundle's followed block (Cape Town's 5810; Nairobi's bundle has none): its best-ranked
    # concept, hull fill not required
    fol = city.get("follow")
    follow = None
    if fol is not None:
        fi = int(fol["index"])
        for r in order[(s.lo[order] <= v[fi]) & (s.hi[order] >= v[fi])]:
            mem = gen.members(s, int(r), v)
            if fi in mem:
                follow = dict(block_id=fol["block_id"], index=fi,
                              rank=int(np.flatnonzero(order == r)[0]) + 1,
                              **entry(int(r), mem, hull_fill(mem, b), s, v, b, floors, metric,
                                      zp))
                break
    up = (s.S / s.W - s.Sr / s.Wr) > 0
    significant = {nm: {f"{lab}p<{a}": int((sel & (p < a)).sum())
                        for a in (0.05, 0.01) for lab, sel in (("", True), ("above ring ", up))}
                   for nm, (_, p) in zp.items()}
    g = np.arange(G)
    keep = np.isfinite(runs[0][0].q(0.95)) | np.isfinite(runs[1][0].q(0.95))
    params = dict(
        candidates=gen.name, widths=list(WIDTHS), max_share=MAX_SHARE, min_blocks=2,
        n_concepts=int(len(s.seed)),
        n_concepts_by_width=np.bincount(s.width, minlength=len(WIDTHS)).tolist(),
        seconds_observed=t_obs,
        nulls=[dict(name=nl.name, nperm=nperm, seed=SEED,
                    rng="numpy default_rng([seed, null index, replicate])", null_index=ni,
                    moran_observed=I_obs, moran_surrogates=dict(
                        mean=float(Is.mean()), sd=float(Is.std()), min=float(Is.min()),
                        max=float(Is.max())), seconds_both_nulls=dt,
                    concepts_beyond_null_sizes=cal.unsupported(s.n_eff))
               for ni, (nl, (cal, Is, dt)) in enumerate(zip(nulls, runs, strict=True))],
        size="n_eff = (sum w)^2 / sum w^2, w buildings",
        quantile_method=(f"p: per replicate the best silhouette among its concepts with n_eff "
                         f"within x/{WINDOW} of the region's, on log10 n_eff bins of "
                         f"log10({WINDOW})/{BINS_PER_WINDOW} (the region's n_eff snapped to its "
                         f"bin); p = (1 + #replicates >= S) / (1 + nperm). z: (S - mean) / sd of "
                         f"every null concept's S, Nadaraya-Watson Gaussian kernel in log10 n_eff, "
                         f"bandwidth log10({WINDOW})"),
        hull_fill_min=HULL_FILL_MIN, top=TOP, close_m=CLOSE_M,
        ranking=(f"p_{ranking.name} ascending, then z_{ranking.name} descending; hull fill >= "
                 f"hull_fill_min; disjoint (greedy)"),
        null_curve=dict(
            n_eff=(10 ** ((g[keep] + 0.5) * STEP)).tolist(),
            **{f"q95_best_{nl.name}": [_finite(x) for x in cal.q(0.95)[keep]]
               for nl, (cal, _, _) in zip(nulls, runs, strict=True)},
            **{f"q50_best_{nl.name}": [_finite(x) for x in cal.q(0.5)[keep]]
               for nl, (cal, _, _) in zip(nulls, runs, strict=True)},
            **{f"mean_{nl.name}": [_finite(x) for x in cal.mean[keep]]
               for nl, (cal, _, _) in zip(nulls, runs, strict=True)},
            **{f"sd_{nl.name}": [_finite(x) for x in cal.sd[keep]]
               for nl, (cal, _, _) in zip(nulls, runs, strict=True)}))
    return dict(params=params, significant=significant, follow=follow, **lists)


# ---------------------------------------------------------------- the checks

def _csr(n: int, a: I64, b_: I64) -> tuple[I64, I64]:
    src, dst = np.concatenate([a, b_]), np.concatenate([b_, a])
    o = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, src + 1, 1)
    return np.cumsum(ptr), dst[o].astype(np.int64)


def _sil_brute(vin: F64, win: F64, vr: F64, wr: F64) -> tuple[float, float, float]:
    A = B = S = 0.0
    for i in range(len(vin)):
        o = np.arange(len(vin)) != i
        a = float((win[o] * np.abs(vin[i] - vin[o])).sum() / win[o].sum()) if o.any() else 0.0
        bb = float((wr * np.abs(vin[i] - vr)).sum() / wr.sum())
        mx = max(a, bb)
        A, B, S = A + win[i] * a, B + win[i] * bb, S + win[i] * ((bb - a) / mx if mx > 0 else 0.0)
    return A / win.sum(), B / win.sum(), S / win.sum()


def _components(ok: NDArray[np.bool_], ptr: I64, nbr: I64) -> list[I64]:
    seen = np.zeros(len(ok), dtype=bool)
    out = []
    for i in np.flatnonzero(ok):
        if seen[i]:
            continue
        st, comp = [i], []
        seen[i] = True
        while st:
            x = st.pop()
            comp.append(x)
            for y in nbr[ptr[x]:ptr[x + 1]]:
                if ok[y] and not seen[y]:
                    seen[y] = True
                    st.append(y)
        out.append(np.sort(np.asarray(comp)))
    return out


def _closed(s: Scored, gen: Candidates, v: F64, ptr: I64, nbr: I64, w: F64) -> int:
    """Rows not closed both ways: band not the tight band of its members, a ring block inside
    it, or the sizes not the members'."""
    bad = 0
    for r in range(len(s.seed)):
        mem = gen.members(s, r, v)
        ring = np.setdiff1d(np.unique(np.concatenate([nbr[ptr[i]:ptr[i + 1]] for i in mem])),
                            mem)
        if not (v[mem].min() == s.lo[r] and v[mem].max() == s.hi[r]
                and not ((v[ring] >= s.lo[r]) & (v[ring] <= s.hi[r])).any()
                and len(mem) == s.blocks[r] and len(ring) == s.ring_blocks[r]
                and np.isclose(w[mem].sum(), s.W[r])):
            bad += 1
    return bad


@dataclass(frozen=True)
class _Graph:
    """A small graph standing in for Blocks (what Concepts reads)."""
    w: F64
    nbr_ptr: I64
    nbr: I64


def check() -> bool:
    rng = np.random.default_rng(SEED)
    bad = 0
    # 1. closure and coverage on small random grid graphs, against every band's components
    clo = cov = st_bad = 0
    nconc = nexp = 0
    for _t in range(40):
        R, C = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        n = R * C
        ij = np.arange(n).reshape(R, C)
        e = np.vstack([np.column_stack([ij[:, :-1].ravel(), ij[:, 1:].ravel()]),
                       np.column_stack([ij[:-1].ravel(), ij[1:].ravel()])])
        e = e[rng.random(len(e)) > 0.15]
        ptr, nbr = _csr(n, e[:, 0], e[:, 1])
        v = rng.normal(size=n) * 0.4 + 0.3 * np.sin(np.arange(n) / 3.0)
        w = rng.integers(1, 40, n).astype(float)
        gen = Concepts(_Graph(w, ptr, nbr), WIDTHS, MAX_SHARE)  # type: ignore[arg-type]
        s = gen.scored(v)
        clo += _closed(s, gen, v, ptr, nbr, w)
        found = {tuple(gen.members(s, r, v).tolist()): r for r in range(len(s.seed))}
        sv = np.sort(v)
        allc: dict[tuple, None] = {}
        for i in range(n):
            for j in range(i, n):
                for comp in _components((v >= sv[i]) & (v <= sv[j]), ptr, nbr):
                    allc[tuple(comp.tolist())] = None
        for key in allc:
            mem = np.asarray(key)
            ring = np.setdiff1d(np.unique(np.concatenate(
                [nbr[ptr[i]:ptr[i + 1]] for i in mem])), mem)
            if len(mem) < 2 or not len(ring) or w[mem].sum() > MAX_SHARE * w.sum():
                if key in found:
                    cov += 1          # found though filtered out
                continue
            lo, hi = v[mem].min(), v[mem].max()
            below, above = v[ring][v[ring] < lo], v[ring][v[ring] > hi]
            gap = (above.min() if len(above) else np.inf) - (below.max() if len(below) else -np.inf)
            expect = any(hi - lo <= d < gap for d in WIDTHS)
            nexp += expect
            if expect and key not in found:
                cov += 1
            if key in found:
                r = found[key]
                a, bb, S = _sil_brute(v[mem], w[mem], v[ring], w[ring])
                if not np.allclose((s.a[r], s.b[r], s.sil[r]), (a, bb, S), rtol=1e-9, atol=1e-12):
                    st_bad += 1
        nconc += len(found)
        cov += sum(k not in allc for k in found)
    print(f"  concepts on 40 grid graphs: {nconc} found, {nexp} expected (span <= d < gap for a "
          f"ladder d); not closed both ways: {clo}; expected but missed, or found but not a "
          f"concept / filtered: {cov}; a, b, S unlike brute force: {st_bad}")
    bad += clo + cov + st_bad
    # 2. the silhouette against brute force on random regions (ties, unequal weights)
    sb = 0
    for _t in range(500):
        k, m = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        vin, vr = np.round(rng.normal(size=k), 1), np.round(rng.normal(size=m) * 1.5, 1)
        win, wr = rng.exponential(size=k) + 0.01, rng.exponential(size=m) + 0.01
        if not np.allclose(silhouette(vin, win, vr, wr), _sil_brute(vin, win, vr, wr),
                           rtol=1e-9, atol=1e-12):
            sb += 1
    print(f"  silhouette on 500 random regions: unlike brute force {sb}")
    bad += sb
    # 3. Nairobi: closure on every concept; a, b, S on a sample; MSR's Moran's I
    root = Path("~/src/wt_demo").expanduser()
    city = json.loads((root / "examples" / "screen-map" / "nairobi.json").read_text())
    b = sc.blocks_of(city)
    A = adjacency(b)
    _, U = mem_basis(A)
    msr = MSR(U)
    gen = Concepts(b, WIDTHS, MAX_SHARE)
    for metric in sc.METRICS:
        v = sc.log_scores(city, metric)
        s = gen.scored(v)
        clo = _closed(s, gen, v, b.nbr_ptr, b.nbr, b.w)
        sb = 0
        for r in rng.choice(len(s.seed), 200, replace=False):
            mem = gen.members(s, int(r), v)
            ring = ring_of(mem, b)
            if not np.allclose((s.a[r], s.b[r], s.sil[r]),
                               _sil_brute(v[mem], b.w[mem], v[ring], b.w[ring]), rtol=1e-9):
                sb += 1
        I0 = moran(v, A)
        Is = np.array([moran(x, A) for x in msr.surrogates(
            v, [np.random.default_rng([SEED, 9, r]) for r in range(32)])])
        mi = int((np.abs(Is - I0) > 0.1).sum() + (abs(Is.mean() - I0) > 0.05))
        print(f"  nairobi {metric}: {len(s.seed)} concepts, not closed both ways {clo}; a, b, S "
              f"unlike brute force on 200: {sb}; Moran's I observed {I0:.3f}, 32 MSR surrogates "
              f"{Is.mean():.3f} +- {Is.std():.3f} [{Is.min():.3f}, {Is.max():.3f}] (each within "
              f"0.1, the mean within 0.05: {'yes' if not mi else 'NO'})")
        bad += clo + sb + mi
    return bad == 0


FAULTS = ("widen", "ring-unweighted", "msr-shuffle")


def main(argv: list[str]) -> None:
    if argv and argv[0] == "check":
        fault = argv[1] if len(argv) > 1 else None
        global silhouette, MSR
        if fault == "widen":             # a concept's band reported wider than its scores
            _scored = Concepts.scored
            Concepts.scored = lambda self, v: (  # type: ignore[method-assign]
                lambda s: s._replace(lo=s.lo - 1e-9, hi=s.hi + 1e-9))(_scored(self, v))
        elif fault == "ring-unweighted":  # the ring's term unweighted by buildings
            _sil = silhouette
            silhouette = lambda vi, wi, vr, wr: _sil(vi, wi, vr, np.ones_like(wr))  # noqa: E731
        elif fault == "msr-shuffle":     # a plain shuffle passed as MSR
            MSR = lambda U: Shuffle()  # type: ignore[assignment] # noqa: E731
        elif fault is not None:
            raise SystemExit(f"faults: {', '.join(FAULTS)}")
        ok = check()
        print(f"check{'' if fault is None else f' under the fault {fault}'}: "
              f"{'pass' if ok else 'FAIL'}")
        if not ok and fault is None:
            raise SystemExit(1)
        if ok and fault is not None:
            raise SystemExit(f"the fault {fault} went unseen: the check guards nothing")
        return
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root, out = Path(argv[0]).expanduser(), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in CITIES:
        t0 = time.time()
        city = json.loads((root / "examples" / "screen-map" / f"{name}.json").read_text())
        b = sc.blocks_of(city)
        A = adjacency(b)
        t1 = time.time()
        _, U = mem_basis(A)
        print(f"{name}: {len(b.w)} blocks, {len(b.nbr) // 2} street pairs; Moran eigenvectors "
              f"{time.time() - t1:.0f}s", flush=True)
        gen = Concepts(b, WIDTHS, MAX_SHARE)
        msr = MSR(U)
        nulls: list[Null] = [Shuffle(), msr]
        res = {}
        for metric in sc.METRICS:
            t1 = time.time()
            r = scanned(city, b, A, gen, nulls, msr, metric, NPERM[name])
            res[metric] = r
            pr = r["params"]
            print(f"  {metric}: {pr['n_concepts']} concepts; Moran's I observed "
                  f"{pr['nulls'][0]['moran_observed']:.3f}, surrogates "
                  + ", ".join(f"{x['name']} {x['moran_surrogates']['mean']:.3f} +- "
                              f"{x['moran_surrogates']['sd']:.3f}" for x in pr["nulls"])
                  + f" ({time.time() - t1:.0f}s)", flush=True)
            for k in ("concepts", "concepts_above"):
                for c in r[k][:4]:
                    print(f"    {k}: S {c['silhouette']:.3f} p_msr {c['p_msr']:.3f} z_msr "
                          f"{c['z_msr']:.1f} p_shuffle {c['p_shuffle']:.3f} z_shuffle "
                          f"{c['z_shuffle']:.1f} blocks {c['blocks']} n {c['buildings']:.0f} "
                          f"n_eff {c['n_eff']:.1f} x{c['x_floor']:.2f} region "
                          f"x{c['region_screen'][metric]['x_floor']:.2f} fill "
                          f"{c['hull_fill']:.2f} ({c['cx'] / 1e3:.1f}, {c['cy'] / 1e3:.1f}) km",
                          flush=True)
        (out / f"concepts_{name}.json").write_text(json.dumps(res))
        del U, msr, nulls
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
