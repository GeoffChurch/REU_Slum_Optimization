"""Guards of the face aperture, the operator's step rule (lifted.step_widths; NOTES, "The grid fix:
steps that conduct by their open width"), each with its fault injection (a patch in one process,
no file edited): a guard passes on the code as it is and must fail under its fault.

    S=research/roadless/aperture_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S identity <ids,> [fault]
    uv run python $S open <ids,> [fault]
    uv run python $S channel - [fault]
    uv run python $S cracks - [fault]
    uv run python $S connect <ids,> [fault]
    uv run python $S convex <ids,> [fault]
    uv run python $S gradient <ids,> [fault]
    uv run python $S free <ids,> [fault]

`identity`: width_check's REFERENCE (57 numbers, computed at bd1704dc under the face aperture,
uni+apwfs, when the step rule was a Strategy) reproduces bit for bit (width_check.identity: the
scorer's baseline and placement, a random clearing's operator, P, J_2 and P_{1/2}, the
relaxation's value and gradient, the greedy's tension, current_map.flow, the sightline metric's).
Fault `ulp`: every step width above 0 one ulp up.

`open`: an open field gives the uniform operator, every step its whole weight. On synthetic
rectangles wholly inside the block (every sub-sample open, OPEN_SHAPES), lifted.edges equals the
families assembled here, family for family, bit for bit: every edge of the pattern, an along
edge's weight m_k / |v_k|^2, a turning edge's h^2 / (ell^2 dtheta). On real blocks with every
footprint open (the no-buildings field), every step whose tracks' cells are wholly open (inside
the block, every sub-sample) has width exactly 1 (the steps at the block's ragged edge are
counted, not required). Fault `sum`: the step by the sum over its tracks, not their mean.

`channel`: a straight open channel (checks.channel's polygon, no walls: the block is the channel)
of width W and length L aligned with axis k conducts m_k W / L in layer k at every lattice angle:
the layer's along edges alone form chains along v_k, each a series of steps, and their parallel
sum (each chain's conductance between its first and last cell, over the chains that run the
channel's length) over m_k W / L is within CHANNEL_EXACT of 1 at the axis-aligned angles and
W_eff within 2 CHANNEL_MARGIN of W at the oblique ones, at the K = 8 lattice angles in [0, 180)
and W in CHANNEL_W. Fault `max-track`: the step by its best track (any open track conducts the
whole step).

`cracks`: synthetic pairs of cells: two cells whose open sub-samples do not touch conduct nothing
(axis-aligned, diagonal and knight steps, several placements); a crack one sub-sample wide on an
axis-aligned step, running from centre to centre, conducts 1/S of it. Fault `open-fraction`: each
track by the smallest open fraction of the cells it passes through, not of its own sub-samples
(how much of each cell is open, not whether the open parts meet: the phantom passage).

`connect`: on blocks, the cells the aperture reaches from the street (lifted.grounded) against the
4-connected flood fill of the open sub-samples it reads (width.fine_bottleneck's connectivity,
split cells' minor parts closed): its axis-aligned steps join exactly the cells whose read
sub-samples touch across a face, and its cells' read parts are each one, so the two are equal,
cell for cell. The plain flood fill (every open sub-sample) is reported beside it: what it reaches
more, it reaches through split cells' minor parts. Fault `diagonal-leak`: tracks traced through
sub-sample centres only (a diagonal step through two sub-samples touching at a corner counts as
open).

`convex`: P is convex in the clearing fractions (q 1): along CONVEX_LINES random segments, every
along edge's width has second differences at most CONVEX_TOL (concave) and P's at least
-CONVEX_TOL of P. Fault `square`: a track by the square of its smallest factor (equal at every 0/1
state; not concave).

`gradient`: the relaxation's gradient (relax.Relaxation.value_grad: SIMP q 1 and 3, P and J_2,
eps 1e-4, a random fractional clearing) against central differences (step FD_STEP, solves to
FD_RTOL from zero) on GRAD_BUILDINGS buildings, within GRAD_TOL of the largest. Fault
`no-spread`: the openness's vjp without the spread S^2 / n (a footprint sub-sample's openness per
unit of its cell's open fraction).

`free`: free first (clear.free_first, budget 1, nothing held) frees every home enclosed at
baseline it takes a crossing for, by the metric's own test (free_first raises on a crossing that
does not: the region graph claimed a path the raster lacks). On 24240 home 1639's only near cells
whose node reaches the street are split (a corner contact; 45267 has one too). Fault
`split-source`: every open cell taken to take demand in the region graph (lifted.demand_cells),
so such a home's source is the street's own component and its crossing clears nothing.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402
import width_check  # noqa: E402

SEED = 20261013
EPS = 1e-4                # SIMP base's eps world
H = 0.5
ELL_M = 3.0
K = 8


def _params() -> lifted.Params:
    return lifted.Params(ELL_M, K, solver=lifted.CpuAMG(), cells=lifted.OpenFraction())


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


# ------------------------------------------------------------------ identity

def identity(ids: list[str]) -> list[str]:
    return width_check.identity(ids)


_STEP_WIDTHS = lifted.step_widths


def _ulp_widths(grid, o, p):
    for frac in _STEP_WIDTHS(grid, o, p):
        yield np.where(frac > 0, np.nextafter(frac, np.inf), 0.0)


# ------------------------------------------------------------------ open field

OPEN_SHAPES = ((5, 7), (9, 6), (12, 12))     # rectangles of cells, every sub-sample open


def synthetic(sub_open: np.ndarray, ground: tuple[tuple[int, int], ...] = ()) -> lifted.Grid:
    """A uniform Grid every sub-sample of which is inside the block, `sub_open` (ny S, nx S) the
    open ones (the rest footprint); `ground` the ground cells."""
    S = 4
    nyS, nxS = sub_open.shape
    ny, nx = nyS // S, nxS // S
    isub = np.ones((ny, nx, S * S), dtype=bool)
    bsub = ~lifted._to_cells(sub_open, S)
    building = bsub.mean(axis=-1) > 0.5
    g = np.zeros((ny, nx), dtype=bool)
    for r, c in ground:
        g[r, c] = True
    X, Y = np.meshgrid(np.arange(nx) * H, np.arange(ny) * H)
    return lifted.Grid(x0=0.0, y0=0.0, h=H, inside=isub.any(axis=-1), building=building,
                       ground=g, dist_b=ndimage.distance_transform_edt(~building) * H,
                       xy=np.stack([X, Y], axis=-1), isub=isub, bsub=bsub, S=S)


def _uniform_families(g: lifted.Grid, p: lifted.Params):
    """The open grid's operator assembled here (not by lifted): every edge of the pattern at its
    whole weight m_k / |v_k|^2 (every cell open, so a cell's id is its flat index), then every
    cell's turning edges at h^2 / (ell^2 dtheta)."""
    _v, _th, _m, gap = lifted.axes(p.K)
    for k, (a, b, _line, wk) in enumerate(g.pattern(p.K, np).along):
        yield a, b, np.full(len(a), k), np.full(len(a), k), np.full(len(a), wk)
    n = g.inside.size
    c = np.arange(n)
    for k in range(p.K):
        yield c, c, np.full(n, k), np.full(n, (k + 1) % p.K), np.ones(n) * (
            g.h * g.h / (p.ell_m ** 2 * gap[k]))


def _same_families(f0: list, f1: list) -> bool:
    return len(f0) == len(f1) and all(
        all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        for a, b in zip(f0, f1, strict=True))


def _whole_slots(g: lifted.Grid, k: int, a: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """Per edge of axis k (its first cell `a`), whether every cell its tracks pass through
    (lifted.axis_tracks's slots, by their offset from a) is wholly open."""
    v, _th, _m, _gap = lifted.axes(K)
    ny, nx = g.inside.shape
    r, c = a // nx, a % nx
    ok = np.ones(len(a), dtype=bool)
    for ox, oy in lifted.axis_tracks(int(v[k, 0]), int(v[k, 1]), g.S).cells:
        rr, cc = r + oy, c + ox
        on = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        ok &= on
        ok[on] &= whole[rr[on] * nx + cc[on]]
    return ok


def open_field(ids: list[str]) -> list[str]:
    bad = []
    p = _params()
    for ny, nx in OPEN_SHAPES:
        g = synthetic(np.ones((ny * 4, nx * 4), dtype=bool))
        same = _same_families(list(_uniform_families(g, p)), list(lifted.edges(g, g.ff0, p)))
        print(f"open {ny} x {nx} cells: lifted.edges equal to the uniform operator family for "
              f"family, bit for bit: {same}", flush=True)
        if not same:
            bad.append(f"{ny}x{nx}")
    for bid in ids:
        [b] = common.build_blocks([bid])
        g = lifted.Grid.of(*lifted._geometry(b), H, offset=lifted.OFFSET)
        o = g.isub.mean(axis=-1).ravel()                 # every footprint open
        whole = (g.isub.all(axis=-1)).ravel()
        n_whole = diff = n_edge = 0
        for k, ((a, _b, _line, _wk), w) in enumerate(zip(g.pattern(K, np).along,
                                                         lifted.step_widths(g, o, p),
                                                         strict=True)):
            inner = _whole_slots(g, k, a, whole)
            n_whole += int(inner.sum())
            diff += int((w[inner] != 1.0).sum())
            n_edge += int((~inner & (w != 1.0)).sum())
        ok = diff == 0
        print(f"{bid}: every footprint open, {n_whole} steps whose tracks' cells are wholly open, "
              f"{diff} not of width exactly 1; {n_edge} steps at the ragged block edge below 1 -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _sum_means(flat, ax, xp):
    acc = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        acc = m if acc is None else acc + m
    return acc if ax.coarse is None else xp.where(ax.coarse, 1.0, acc)


# ------------------------------------------------------------------ channel

CHANNEL_W = (1.0, 2.0, 4.0)
CHANNEL_L = 40.0
CHANNEL_EXACT = 1e-9          # an axis-aligned channel: m_k W exactly (its walls on sub-sample
                              # boundaries or not, every track is either in the open or not)
CHANNEL_MARGIN = 0.2          # an oblique one: the width lost at each wall, at most (m): half a
                              # sub-sample of the wall's rasterization, a track's half-thickness
                              # (its staircase, up to 0.71 of a sub-sample across) and half the
                              # tracks' spacing, about 0.19 m at 45 deg


def _layer_conduction(g: lifted.Grid, k: int, p: lifted.Params, e: np.ndarray,
                      L: float) -> float:
    """Layer k's conductance per unit length along the channel's middle half (clear of the
    rasterized ends): its along edges form chains along v_k, parallel to the walls; each chain's
    edges with both cells in the middle half are a series, conducting span / sum(1 / w) per unit
    length (span: the distance its first to its last cell covers), the chains in parallel. A
    channel of width W conducts m_k W per unit length."""
    o = g.ff0.ravel()
    fam = list(lifted.along_edges(g, o, p))
    _k, a, b, w = fam[k]
    cell = np.flatnonzero(o > 0)
    a, b = cell[a], cell[b]
    s = g.xy.reshape(-1, 2) @ e                           # position along the channel
    mid = (s[a] >= L / 4) & (s[b] <= 3 * L / 4)
    a, b, w = a[mid], b[mid], w[mid]
    n = o.size
    nxt = -np.ones(n, dtype=np.int64)
    nxt[a] = b
    wt = np.zeros(n)
    wt[a] = w
    has_prev = np.zeros(n, dtype=bool)
    has_prev[b] = True
    step = g.h * float(np.hypot(*lifted.axes(p.K)[0][k]))
    total = 0.0
    for start in np.flatnonzero((nxt >= 0) & ~has_prev):
        R, c = 0.0, start
        while nxt[c] >= 0:
            R += 1.0 / wt[c]
            c = nxt[c]
        span = s[c] - s[start]
        if span > L / 2 - 2 * step:                      # runs the middle half's length
            total += span / R
    return total


def channel(_ids: list[str]) -> list[str]:
    """The layer-k conductance per unit length of the aligned channel over m_k W (W_eff / W):
    1 at the axis-aligned angles, W_eff in [W - 2 CHANNEL_MARGIN, W] at the oblique ones."""
    bad = []
    v, th, m, _gap = lifted.axes(K)
    p = _params()
    for k in range(K):
        e = np.array([np.cos(th[k]), np.sin(th[k])])
        nrm = np.array([-e[1], e[0]])
        aligned = int(v[k, 0]) == 0 or int(v[k, 1]) == 0
        for W in CHANNEL_W:
            corners = [-nrm * W / 2, e * CHANNEL_L - nrm * W / 2, e * CHANNEL_L + nrm * W / 2,
                       nrm * W / 2]
            poly = Polygon(corners)
            g = lifted.Grid.of(poly, [], [LineString([corners[1], corners[2]])], H, band_m=1.0)
            x = _layer_conduction(g, k, p, e, CHANNEL_L) / (m[k] * W)
            if aligned:
                ok = abs(x - 1.0) <= CHANNEL_EXACT
            else:
                ok = W - 2 * CHANNEL_MARGIN <= x * W <= W + CHANNEL_EXACT
            print(f"axis {k} ({np.degrees(th[k]):.2f} deg) W {W:g}: layer {k}'s conductance per "
                  f"metre over m_k W {x:.4f} (W_eff {x * W:.3f} m) -> {'ok' if ok else 'FAILS'}",
                  flush=True)
            if not ok:
                bad.append(f"axis {k} W {W:g}")
    return bad


def _max_track(flat, ax, xp):
    best = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        best = m if best is None else xp.maximum(best, m)
    return best if ax.coarse is None else xp.where(ax.coarse, 1.0, best)


# ------------------------------------------------------------------ cracks

S4 = 4


def _pair(dx: int, dy: int, a_open: list[tuple[int, int]], b_open: list[tuple[int, int]],
          line_open: str) -> lifted.Grid:
    """A 3 x 4 block of cells, a at (row 1, col 1 if dx >= 0 else 2), b at a + (dx, dy) (cells),
    the open sub-samples of a and b as (row, col) within the cell; the cells the step crosses
    wholly open (`line_open` "all") or wholly footprint ("none"); every other cell footprint."""
    ny, nx = 4, 5
    sub = np.zeros((ny * S4, nx * S4), dtype=bool)
    ra, ca = 1, 1 if dx >= 0 else 3
    rb, cb = ra + dy, ca + dx
    for (r, c), cells in (((ra, ca), a_open), ((rb, cb), b_open)):
        for i, j in cells:
            sub[r * S4 + i, c * S4 + j] = True
    if line_open == "all":
        for ix, iy in lifted._line_cells(dx, dy):
            sub[(ra + iy) * S4:(ra + iy + 1) * S4, (ca + ix) * S4:(ca + ix + 1) * S4] = True
    return synthetic(sub)


def _width_of(g: lifted.Grid, k: int, dx: int, dy: int) -> float:
    ny, nx = g.inside.shape
    ra, ca = 1, 1 if dx >= 0 else 3
    a = ra * nx + ca
    o = g.ff0.ravel()
    w = list(lifted.step_widths(g, o, _params()))[k]
    i = np.flatnonzero(g.pattern(K, np).along[k][0] == a)
    return float(w[i[0]]) if len(i) else 0.0


CRACK_CASES = (
    # (name, axis, a's open sub-samples, b's, line cells, the width)
    ("axis 0, a crack one sub-sample wide, centre to centre", 0,
     [(1, j) for j in range(S4)], [(1, j) for j in range(S4)], "none", 1 / S4),
    ("axis 0, open rows that do not meet (a's row 0, b's row 3)", 0,
     [(0, j) for j in range(S4)], [(3, j) for j in range(S4)], "none", 0.0),
    ("axis 0, a's open half far from b, b's near a", 0,
     [(i, j) for i in range(S4) for j in (0, 1)], [(i, j) for i in range(S4) for j in (0, 1)],
     "none", 0.0),
    ("axis 4, open columns that do not meet (a's col 0, b's col 3)", 4,
     [(i, 0) for i in range(S4)], [(i, 3) for i in range(S4)], "none", 0.0),
    ("axis 2 (1, 1), a and b open, the cells crossed footprint", 2,
     [(i, j) for i in range(S4) for j in range(S4)], [(i, j) for i in range(S4) for j in range(S4)],
     "none", 0.0),
    ("axis 2 (1, 1), a and b open only at the corners away from each other", 2,
     [(0, 0)], [(3, 3)], "all", 0.0),
    ("axis 1 (2, 1), a and b open, the cells crossed footprint", 1,
     [(i, j) for i in range(S4) for j in range(S4)], [(i, j) for i in range(S4) for j in range(S4)],
     "none", 0.0),
)


def cracks(_ids: list[str]) -> list[str]:
    bad = []
    v, _th, _m, _gap = lifted.axes(K)
    for name, k, a_open, b_open, line, want in CRACK_CASES:
        dx, dy = int(v[k, 0]), int(v[k, 1])
        got = _width_of(_pair(dx, dy, a_open, b_open, line), k, dx, dy)
        ok = abs(got - want) < 1e-15
        print(f"{name}: {got:.4f} (want {want:.4f}) -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(name)
    return bad


def _open_fraction_means(flat, ax, xp):
    """Each track by the smallest open fraction (its cell's mean factor) of the cells it passes
    through, in place of the smallest factor of its own sub-samples."""
    cell = flat.reshape(-1, S4 * S4).mean(axis=1)
    acc = None
    for tr in ax.tracks:
        m = None
        for slot, _sub in tr:
            v = cell[ax.base[slot] // (S4 * S4)]
            m = v if m is None else xp.minimum(m, v)
        acc = m if acc is None else acc + m
    w = acc / len(ax.tracks)
    return w if ax.coarse is None else xp.where(ax.coarse, 1.0, w)


# ------------------------------------------------------------------ connect

def flood(g: lifted.Grid, keep: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 4-connected flood fill of open sub-samples from the ground cells'
    (width.fine_bottleneck's connectivity), over the sub-samples `keep` (cells, S*S) only if
    given: (per cell, reached; the fine lattice's component labels; per label, whether it holds a
    ground cell's open sub-sample)."""
    S = g.S
    op = g.isub & ~g.bsub
    if keep is not None:
        op = op & keep.reshape(op.shape).astype(bool)
    opened = lifted._fine(op, S)
    gr = lifted._fine(np.repeat(g.ground[..., None], S * S, axis=-1), S)
    lab, n = ndimage.label(opened)
    ok = np.zeros(n + 1, dtype=bool)
    ok[lab[opened & gr]] = True
    ok[0] = False
    return lifted._cell_sum(ok[lab].astype(float), S) > 0, lab, ok


def split_cells(g: lifted.Grid) -> np.ndarray:
    """(ny, nx): how many 4-connected parts each cell's open sub-samples form (0 if none)."""
    S = g.S
    op = (g.isub & ~g.bsub).reshape(-1, S, S)
    n = np.zeros(len(op), dtype=np.int64)
    for i in np.flatnonzero(op.any(axis=(1, 2))):
        n[i] = ndimage.label(op[i])[1]
    return n.reshape(g.inside.shape)


def connect(ids: list[str]) -> list[str]:
    bad = []
    p = _params()
    for bid in ids:
        [b] = common.build_blocks([bid])
        g = lifted.Grid.of(*lifted._geometry(b), H, offset=lifted.OFFSET)
        reach = lifted.grounded(g, g.ff0, p)
        keep = lifted._anchor_keep(p.cells.sub_factor(g, g.ff0, p.solver), g, np)
        exact = int((reach != flood(g, keep)[0]).sum())
        plain = flood(g)[0]
        ok = exact == 0
        print(f"{bid}: reach {int(reach.sum())}; against the flood fill of the sub-samples it "
              f"reads {exact} cells differ -> {'ok' if ok else 'FAILS'} (the plain flood fill, "
              f"every open sub-sample, reaches {int((plain & ~reach).sum())} cells more, through "
              f"split cells' minor parts, of {int((split_cells(g) >= 2).sum())} split cells; "
              f"{int((reach & ~plain).sum())} fewer)", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _centre_tracks(dx: int, dy: int, S: int) -> lifted.Tracks:
    """axis_tracks traced through sub-sample centres only: each track the sub-samples whose centre
    its line passes (Bresenham-like, one per row or column crossed), so a diagonal step through
    two sub-samples touching at a corner is open."""
    tr = _TRACKS(dx, dy, S)
    out = []
    for t in tr.tracks:
        keep = list(t)
        # keep every other square of a staircase: the ones a line through centres would visit
        if abs(dx) > 0 and abs(dy) > 0:
            keep = keep[::2]
        out.append(tuple(keep))
    return lifted.Tracks(cells=tr.cells, tracks=tuple(out))


_TRACKS = lifted.axis_tracks


# ------------------------------------------------------------------ convex

CONVEX_LINES = 4
CONVEX_POINTS = 7
CONVEX_TOL = 1e-9


def _clearing(b) -> clear.Clearing:
    return clear.Clearing(b, lifted.UniformMesh(H, offset=lifted.OFFSET), _params(),
                          population=common.POPULATIONS["area"], demand=common.PopulationDemand())


def convex(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        c = _clearing(b)
        rel = relax.Relaxation(c, 1.0, q=1.0, rtol=width_check.FD_RTOL, eps=EPS)
        rng = np.random.default_rng(SEED + 2)
        worst_w, worst_P = -np.inf, np.inf
        for _ in range(CONVEX_LINES):
            a, z = rng.uniform(0.0, 1.0, c.n), rng.uniform(0.0, 1.0, c.n)
            ws, P = [], []
            for t in np.linspace(0.0, 1.0, CONVEX_POINTS):
                x = a + t * (z - a)
                op = rel.open_of(x)
                op_eps = (op + EPS * (c.full - op)).ravel()
                ws.append(np.concatenate(list(lifted.step_widths(c.sc.grid, op_eps, c.p))))
                P.append(width_check._value(rel, x))
            W2 = np.diff(np.array(ws), 2, axis=0)
            P2 = np.diff(np.array(P), 2)
            worst_w = max(worst_w, float(W2.max()))
            worst_P = min(worst_P, float((P2 / np.max(P)).min()))
        ok = worst_w <= CONVEX_TOL and worst_P >= -CONVEX_TOL
        print(f"{bid}: on {CONVEX_LINES} lines, the step widths' largest second difference "
              f"{worst_w:.1e} (concave: <= {CONVEX_TOL:g}), P's smallest {worst_P:.1e} of P "
              f"(convex: >= -{CONVEX_TOL:g}) -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _square_means(flat, ax, xp):
    """A track by the square of its smallest factor (equal at every 0/1 state; convex, not
    concave, between). A track by the product of its factors (width_check's `product`) is the
    same kind of fault but leaves the eps world's system too ill-conditioned to solve (a closed
    track's eps^9 -- eps^13: AMG-CG did not converge)."""
    acc = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        acc = m * m if acc is None else acc + m * m
    w = acc / len(ax.tracks)
    return w if ax.coarse is None else xp.where(ax.coarse, 1.0, w)


# ------------------------------------------------------------------ gradient

GRAD_BUILDINGS = 6
GRAD_Q = (1.0, 3.0)
FD_STEP = 1e-5            # the widths are piecewise linear (a min per track): small steps
FD_RTOL = 1e-9
GRAD_TOL = 1e-5


def gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        c = _clearing(b)
        rng = np.random.default_rng(SEED)
        x0 = rng.uniform(0.1, 0.4, c.n)
        pick = rng.choice(c.n, GRAD_BUILDINGS, replace=False)
        for power in (1.0, 2.0):
            for q in GRAD_Q:
                rel = relax.Relaxation(c, power, q=q, rtol=FD_RTOL, eps=EPS)
                rel._u = None
                _, g = rel.value_grad(x0)
                fd = []
                for j in pick:
                    e = np.zeros(c.n)
                    e[j] = FD_STEP
                    fd.append((width_check._value(rel, x0 + e)
                               - width_check._value(rel, x0 - e)) / (2 * FD_STEP))
                err = float(np.abs(np.array(fd) - g[pick]).max() / np.abs(g[pick]).max())
                ok = err <= GRAD_TOL
                print(f"{bid} J_{power:g} SIMP q {q:g}: gradient against central differences on "
                      f"{GRAD_BUILDINGS} buildings, {err:.1e} of the largest -> "
                      f"{'ok' if ok else 'FAILS'}", flush=True)
                if not ok:
                    bad.append(f"{bid} p{power:g} q{q:g}")
    return bad


# ------------------------------------------------------------------ free first

def free(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        c = _clearing(b)
        sc = c.sc
        try:
            held = clear.free_first(c, budget=1.0, held=np.zeros(c.n, dtype=bool))
        except RuntimeError as e:
            print(f"{bid}: {e} -> FAILS", flush=True)
            bad.append(bid)
            continue
        left = sc.enclosed & ~common.reached(sc.pl, sc.grid, c.open_(held), c.p)
        print(f"{bid}: {int(sc.enclosed.sum())} homes enclosed at baseline, free first clears "
              f"{int(held.sum())} buildings (D {float(c.cost[held].sum()):.4f}), "
              f"{int(left.sum())} left enclosed -> ok", flush=True)
    return bad


def _every_open_cell(grid, open0, F0):
    return open0


def _no_spread_vjp(grid, A, xp):
    st = lifted._sub_state(grid, xp)
    return (st.cov * A).sum(axis=1)


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "identity": identity, "open": open_field, "channel": channel, "cracks": cracks,
    "connect": connect, "convex": convex, "gradient": gradient, "free": free}
FAULTS: dict[str, dict[str, Callable]] = {
    "identity": {"ulp": lambda: _patched(lifted, "step_widths", _ulp_widths)},
    "open": {"sum": lambda: _patched(lifted, "_track_means", _sum_means)},
    "channel": {"max-track": lambda: _patched(lifted, "_track_means", _max_track)},
    "cracks": {"open-fraction": lambda: _patched(lifted, "_track_means", _open_fraction_means)},
    "connect": {"diagonal-leak": lambda: _patched(lifted, "axis_tracks", _centre_tracks)},
    "convex": {"square": lambda: _patched(lifted, "_track_means", _square_means)},
    "gradient": {"no-spread": lambda: _patched(lifted, "_openness_vjp", _no_spread_vjp)},
    "free": {"split-source": lambda: _patched(lifted, "demand_cells", _every_open_cell)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], [] if a[1] == "-" else a[1].split(","), a[2] if len(a) == 3 else None)
