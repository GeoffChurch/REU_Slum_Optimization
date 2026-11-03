"""The screen's best regular regions (owner 2026-11-03): on each city's screen height map (the
site's ScreenMap bundle: every block's n, A, P and boundary), the squares, rectangles, circles and
ellipses that score highest, under two objectives, for the Demo Lineup artifact's overlay.

The field. Each block's score (the bundle's metric: depth_density_proxy sqrt(nA)/P n/A, or
density_compactness n/P^2) as log10, painted on a PIX m raster at the pixel centres inside the
block (ground no block covers, the streets and the sea, is uncovered); a block scoring 0 at the
city's 1st percentile.

A shape is a kernel (a disk, an ellipse at an angle, an axis-aligned rectangle; a square is a
rectangle whose sides are equal) placed at every pixel. Of what it covers, `cov` is the covered
share and `mean` the covered pixels' mean log score; a placement counts only if cov >= MIN_COV
(no circle earns its score from one block on the shore).

  excess: the sum over the covered pixels of (log score - log floor) times the pixel's km^2, at
    the metric's shipped floor: a shape grows while what it adds is above the floor on balance,
    so the objective picks its own size; with no shape constraint its optimum is the inside of
    the floor's isocline.
  raw: per kernel (one size and shape), the placement's mean against every valid placement of
    the same kernel: z = (mean - mu_k) / sigma_k. Rank within a size cannot compare sizes (every
    size's best is rank 1); z can, and a large window's sigma is small. Not corrected for the
    number of placements: a small kernel has more nearly independent placements, so its best z
    is high by chance. On Cape Town every family's raw winner is the ladder's smallest shape
    inside 5810, the top block, larger than it: raw z finds the top block.
  surprise: the placement's mean against the best mean the same kernel reaches by chance, the
    blocks' log scores shuffled among the blocks (their geometry kept; NPERM shuffles, SEED):
    (mean - mean of the shuffled bests) / their spread. Kulldorff's scan statistic in z form:
    a region scores by how far it beats what the best region of its size would be with no
    spatial clustering, so the count of placements is accounted for.

Per family and objective the TOP best placements, disjoint (their centres at least the sum of
their larger semi-axes apart).

    S=research/roadless/atlas/explore_proto/screen_shapes.py   # from the repo root, PYTHONPATH=.
    uv run python $S ~/src/wt_demo <out dir>     # shapes_<city>.json per city, both metrics
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy import fft
from scipy.ndimage import maximum_filter

PIX = 50.0            # m: the raster
MIN_COV = 0.6         # a placement's covered share at least
TOP = 3               # shapes per family and objective
PEAKS = 40            # per kernel, local maxima kept as candidates
NPERM = 32            # surprise: the shuffles of the blocks' scores
SEED = 0              # their generator's seed
WORKERS = 16          # processes for the shuffles (single-threaded FFTs each)
PEAK_PX = 41          # px: their neighbourhood at most (the filter is O(size) per pixel)
CITIES = ("capetown", "nairobi")
METRICS = {"depth_density_proxy": lambda n, A, P: np.sqrt(n * A) / P * n / A,
           "density_compactness": lambda n, A, P: n / P ** 2}
# the kernels' ladders (m): geometric, from a few blocks to a district
RADII = np.geomspace(150, 6000, 14)
SIDES = np.geomspace(250, 12000, 14)
ELL_A = np.geomspace(250, 6000, 10)          # the semi-major axis
ELL_RATIO = (2.0, 3.0, 4.0)
ELL_ANGLES = tuple(np.deg2rad(a) for a in range(0, 180, 30))


@dataclass(frozen=True)
class Kernel:
    family: str              # circle | ellipse | square | rectangle
    a: float                 # m: radius, semi-major axis, or half-width
    b: float                 # m: radius, semi-minor axis, or half-height
    theta: float             # rad: the major axis from +x (ellipses)

    def mask(self) -> NDArray[np.float64]:
        if self.family in ("square", "rectangle"):
            return np.ones((2 * round(self.b / PIX) + 1, 2 * round(self.a / PIX) + 1))
        R = int(np.ceil(self.a / PIX))
        y, x = np.mgrid[-R:R + 1, -R:R + 1] * PIX
        c, s = np.cos(self.theta), np.sin(self.theta)
        u, v = x * c + y * s, -x * s + y * c
        return ((u / self.a) ** 2 + (v / self.b) ** 2 <= 1.0).astype(float)


@dataclass(frozen=True)
class Shape:
    family: str
    cx: float                # m: the bundle's local coordinates (y up)
    cy: float
    a: float
    b: float
    theta: float
    value: float             # the objective: excess (decades x km^2) or z
    mean_score: float        # the covered pixels' geometric-mean score
    cov: float
    area_km2: float


def kernels() -> list[Kernel]:
    ks = [Kernel("circle", r, r, 0.0) for r in RADII]
    ks += [Kernel("ellipse", a, a / q, t) for a in ELL_A for q in ELL_RATIO for t in ELL_ANGLES]
    for w in SIDES:
        for h in SIDES:
            ks.append(Kernel("square" if w == h else "rectangle", w / 2, h / 2, 0.0))
    return ks


def raster(city: dict) -> tuple[NDArray[np.int64], tuple[float, float]]:
    """Per pixel the index of the block covering its centre (-1 none); the raster's lower-left."""
    polys = [shapely.Polygon(r[0], r[1:]) for r in city["rings"]]
    x0, y0, x1, y1 = shapely.total_bounds(polys)
    nx, ny = int(np.ceil((x1 - x0) / PIX)) + 1, int(np.ceil((y1 - y0) / PIX)) + 1
    lab = np.full((ny, nx), -1, dtype=np.int64)
    for i, p in enumerate(polys):
        bx0, by0, bx1, by1 = p.bounds
        i0, i1 = int((bx0 - x0) / PIX), int(np.ceil((bx1 - x0) / PIX)) + 1
        j0, j1 = int((by0 - y0) / PIX), int(np.ceil((by1 - y0) / PIX)) + 1
        X, Y = np.meshgrid(x0 + (np.arange(i0, i1) + 0.5) * PIX,
                           y0 + (np.arange(j0, j1) + 0.5) * PIX)
        inside = shapely.contains_xy(p, X, Y)
        lab[j0:j1, i0:i1][inside] = i
    return lab, (x0, y0)


class Conv:
    """Convolutions of the two fields (log score times cover, cover) with any kernel, the fields'
    transforms computed once at a padding that fits the largest kernel."""

    def __init__(self, v: NDArray[np.float64], c: NDArray[np.float64], pad: int) -> None:
        self.shape = v.shape
        self.full = (fft.next_fast_len(v.shape[0] + 2 * pad),
                     fft.next_fast_len(v.shape[1] + 2 * pad))
        self.V = fft.rfft2(v, self.full)
        self.C = fft.rfft2(c, self.full)

    def __call__(self, k: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        K = fft.rfft2(k, self.full)
        hy, hx = k.shape[0] // 2, k.shape[1] // 2
        sl = (slice(hy, hy + self.shape[0]), slice(hx, hx + self.shape[1]))
        return (fft.irfft2(self.V * K, self.full)[sl], fft.irfft2(self.C * K, self.full)[sl])


_NULL: dict = {}        # the shuffles' inputs, set in the parent and inherited by forked workers


def shuffled_best(seed: int) -> NDArray[np.float64]:
    """Per kernel the best valid placement's mean log score with the blocks' scores shuffled."""
    lab, logs, ks, pad = _NULL["lab"], _NULL["logs"], _NULL["ks"], _NULL["pad"]
    perm = np.random.default_rng(seed).permutation(logs)
    cov = (lab >= 0).astype(float)
    conv = Conv(np.where(lab >= 0, perm[np.maximum(lab, 0)], 0.0), cov, pad)
    out = np.full(len(ks), np.nan)
    for j, k in enumerate(ks):
        m = k.mask()
        sv, sc = conv(m)
        ok = sc / m.sum() >= MIN_COV
        if ok.any():
            out[j] = float((sv[ok] / sc[ok]).max())
    return out


def best(city: dict, metric: str, lab: NDArray[np.int64], origin: tuple[float, float]) -> dict:
    n, A, P = (np.asarray(city[k], dtype=float) for k in ("n", "area_m2", "perimeter_m"))
    s = METRICS[metric](n, A, P)
    logs = np.log10(np.where(s > 0, s, np.nan))
    logs = np.where(np.isfinite(logs), logs, np.nanpercentile(logs, 1))
    floor = next(f["value"] for f in city["floors"] if f["metric"] == metric)
    cov = (lab >= 0).astype(float)
    v = np.where(lab >= 0, logs[np.maximum(lab, 0)], 0.0)
    ks = kernels()
    pad = max(max(k.mask().shape) for k in ks)
    conv = Conv(v, cov, pad)
    _NULL.update(lab=lab, logs=logs, ks=ks, pad=pad)
    seeds = np.random.SeedSequence(SEED).generate_state(NPERM)
    with get_context("fork").Pool(WORKERS) as pool:
        null = np.array(pool.map(shuffled_best, [int(x) for x in seeds]))
    null_mu, null_sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    px_km2 = (PIX / 1000) ** 2
    cand: dict[str, dict[str, list[tuple[float, Shape]]]] = {"excess": {}, "raw": {},
                                                             "surprise": {}}
    x0, y0 = origin
    for j, k in enumerate(ks):
        m = k.mask()
        sv, sc = conv(m)
        npx = m.sum()
        cshare = sc / npx
        ok = cshare >= MIN_COV
        if not ok.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(ok, sv / np.maximum(sc, 1e-9), np.nan)
        excess = np.where(ok, (sv - np.log10(floor) * sc) * px_km2, -np.inf)
        mu, sd = np.nanmean(mean), np.nanstd(mean)
        z = np.where(ok, (mean - mu) / sd, -np.inf)
        cal = np.where(ok, (mean - null_mu[j]) / null_sd[j], -np.inf)
        size = min(PEAK_PX, max(3, int(min(m.shape) // 2) | 1))   # local maxima, this far apart
        for obj, field in (("excess", excess), ("raw", z), ("surprise", cal)):
            peak = (field == maximum_filter(field, size=size, mode="constant", cval=-np.inf)) & ok
            idx = np.flatnonzero(peak)
            idx = idx[np.argsort(field.ravel()[idx])[::-1][:PEAKS]]
            for i in idx:
                jy, jx = divmod(int(i), field.shape[1])
                sh = Shape(family=k.family, cx=x0 + (jx + 0.5) * PIX, cy=y0 + (jy + 0.5) * PIX,
                           a=k.a, b=k.b, theta=k.theta, value=float(field[jy, jx]),
                           mean_score=float(10 ** mean[jy, jx]), cov=float(cshare[jy, jx]),
                           area_km2=float(npx * px_km2))
                cand[obj].setdefault(k.family, []).append((sh.value, sh))
    out: dict = {"metric": metric, "floor": floor, "pix_m": PIX, "min_cov": MIN_COV,
                 "nperm": NPERM, "seed": SEED}
    for obj, fams in cand.items():
        out[obj] = {}
        for fam, lst in fams.items():
            lst.sort(key=lambda q: -q[0])
            kept: list[Shape] = []
            for _, sh in lst:
                if any(np.hypot(sh.cx - o.cx, sh.cy - o.cy) < max(sh.a, sh.b) + max(o.a, o.b)
                       for o in kept):
                    continue
                kept.append(sh)
                if len(kept) == TOP:
                    break
            out[obj][fam] = [asdict(s) for s in kept]
    return out


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root, out = Path(argv[0]).expanduser(), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in CITIES:
        t0 = time.time()
        city = json.loads((root / "examples" / "screen-map" / f"{name}.json").read_text())
        lab, origin = raster(city)
        print(f"{name}: raster {lab.shape}, {(lab >= 0).mean():.2f} covered "
              f"({time.time() - t0:.0f}s)", flush=True)
        res = {m: best(city, m, lab, origin) for m in METRICS}
        (out / f"shapes_{name}.json").write_text(json.dumps(res))
        for m, r in res.items():
            for obj in ("excess", "raw", "surprise"):
                top = {f: round(v[0]["value"], 2) for f, v in r[obj].items() if v}
                print(f"  {m} {obj}: {top}", flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
